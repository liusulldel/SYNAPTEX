"""
SYNAPTEX·触链典 — Multimodal Memory System

Enables memories to anchor non-text modalities:
images, audio clips, video frames, and embedding vectors.

Each ModalityAnchor is attached to a MemoryUnit, enriching
the Zettelkasten graph with cross-modal retrieval capabilities.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import hashlib

from synaptex.types import MemoryUnit, ModalityAnchor


SUPPORTED_MODALITIES = {"image", "audio", "video", "embedding", "document"}

SUPPORTED_EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg"},
    "audio": {".mp3", ".wav", ".ogg", ".flac", ".m4a"},
    "video": {".mp4", ".webm", ".avi", ".mov", ".mkv"},
    "document": {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md"},
}


@dataclass
class MultimodalIndex:
    """Index structure for cross-modal retrieval."""
    modality_index: Dict[str, List[str]] = field(default_factory=lambda: {
        "image": [], "audio": [], "video": [], "embedding": [], "document": [],
    })
    path_to_memory: Dict[str, str] = field(default_factory=dict)  # path → memory_id
    memory_to_anchors: Dict[str, List[ModalityAnchor]] = field(default_factory=dict)


class MultimodalMemory:
    """
    Multimodal memory manager for SYNAPTEX.
    
    Capabilities:
    1. Attach images/audio/video to text memories as "anchors"
    2. Cross-modal retrieval: find text memories via image similarity
    3. Modality-aware context loading: only page-in relevant modalities
    4. Storage-efficient: stores paths/references, not raw binary data
    
    Usage:
        mm = MultimodalMemory()
        
        # Attach an image to a meeting memory
        mem = MemoryUnit(content_l3="Board meeting with whiteboard diagrams")
        anchor = mm.attach(mem, "image", "/path/to/whiteboard.jpg",
                          description="Whiteboard diagram of Q3 roadmap")
        
        # Find all memories with images
        image_memories = mm.find_by_modality("image")
    """

    def __init__(self, base_storage_path: Optional[str] = None):
        self.index = MultimodalIndex()
        self.base_path = Path(base_storage_path) if base_storage_path else None

    def _detect_modality(self, path: str) -> str:
        """Auto-detect modality from file extension."""
        ext = Path(path).suffix.lower()
        for modality, extensions in SUPPORTED_EXTENSIONS.items():
            if ext in extensions:
                return modality
        return "document"

    def _compute_file_hash(self, path: str) -> str:
        """Compute hash of file path for deduplication."""
        return hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]

    def attach(
        self,
        memory: MemoryUnit,
        modality: Optional[str] = None,
        path: str = "",
        description: str = "",
        embedding: Optional[List[float]] = None,
    ) -> ModalityAnchor:
        """
        Attach a multimodal anchor to a memory unit.
        
        Args:
            memory: Target MemoryUnit
            modality: "image", "audio", "video", "embedding", "document"
                     Auto-detected from path if not specified
            path: File path or URL to the media
            description: Human-readable description of the media
            embedding: Optional pre-computed embedding vector
            
        Returns:
            The created ModalityAnchor
        """
        # Auto-detect modality
        if not modality:
            modality = self._detect_modality(path)

        if modality not in SUPPORTED_MODALITIES:
            raise ValueError(f"Unsupported modality: {modality}. "
                           f"Supported: {SUPPORTED_MODALITIES}")

        anchor = ModalityAnchor(
            modality=modality,
            path=path,
            description=description,
            embedding=embedding,
            timestamp=datetime.now(),
        )

        # Attach to memory
        memory.modality_anchors.append(anchor)

        # Update indices
        self.index.modality_index[modality].append(memory.id)
        self.index.path_to_memory[path] = memory.id

        if memory.id not in self.index.memory_to_anchors:
            self.index.memory_to_anchors[memory.id] = []
        self.index.memory_to_anchors[memory.id].append(anchor)

        # Add modality as a tag for graph linking
        tag = f"has:{modality}"
        if tag not in memory.tags:
            memory.tags.append(tag)

        return anchor

    def detach(self, memory: MemoryUnit, path: str):
        """Remove a modality anchor from a memory."""
        memory.modality_anchors = [
            a for a in memory.modality_anchors if a.path != path
        ]
        self.index.path_to_memory.pop(path, None)
        
        if memory.id in self.index.memory_to_anchors:
            self.index.memory_to_anchors[memory.id] = [
                a for a in self.index.memory_to_anchors[memory.id]
                if a.path != path
            ]

    def find_by_modality(self, modality: str) -> List[str]:
        """Find all memory IDs that have a specific modality attached."""
        return list(set(self.index.modality_index.get(modality, [])))

    def get_anchors(self, memory_id: str) -> List[ModalityAnchor]:
        """Get all modality anchors for a specific memory."""
        return self.index.memory_to_anchors.get(memory_id, [])

    def find_by_embedding_similarity(
        self,
        query_embedding: List[float],
        modality: str = "image",
        top_k: int = 5,
    ) -> List[tuple]:
        """
        Find memories by embedding cosine similarity.
        
        Args:
            query_embedding: Query vector
            modality: Filter by modality type
            top_k: Number of results
            
        Returns:
            List of (memory_id, similarity_score) tuples
        """
        scores = []
        
        for memory_id, anchors in self.index.memory_to_anchors.items():
            for anchor in anchors:
                if anchor.modality == modality and anchor.embedding:
                    sim = self._cosine_similarity(query_embedding, anchor.embedding)
                    scores.append((memory_id, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get_stats(self) -> Dict:
        """Return multimodal index statistics."""
        return {
            "total_anchors": sum(
                len(v) for v in self.index.modality_index.values()
            ),
            "by_modality": {
                k: len(set(v)) for k, v in self.index.modality_index.items() if v
            },
            "memories_with_media": len(self.index.memory_to_anchors),
        }
