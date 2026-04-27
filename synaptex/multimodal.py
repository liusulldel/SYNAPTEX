"""Attachment references for non-text SYNAPTEX memory anchors."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

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
    """Indexes memory IDs by modality and attachment path."""

    modality_index: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "image": [],
            "audio": [],
            "video": [],
            "embedding": [],
            "document": [],
        }
    )
    path_to_memory: Dict[str, str] = field(default_factory=dict)
    memory_to_anchors: Dict[str, List[ModalityAnchor]] = field(default_factory=dict)


class MultimodalMemory:
    """Manage lightweight attachment references for memories."""

    def __init__(self, base_storage_path: Optional[str] = None):
        self.index = MultimodalIndex()
        self.base_path = Path(base_storage_path) if base_storage_path else None

    def _detect_modality(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        for modality, extensions in SUPPORTED_EXTENSIONS.items():
            if ext in extensions:
                return modality
        return "document"

    def _compute_file_hash(self, path: str) -> str:
        return hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]

    def attach(
        self,
        memory: MemoryUnit,
        modality: Optional[str] = None,
        path: str = "",
        description: str = "",
        embedding: Optional[List[float]] = None,
    ) -> ModalityAnchor:
        """Attach a path or embedding reference to a memory."""

        resolved_modality = modality or self._detect_modality(path)
        if resolved_modality not in SUPPORTED_MODALITIES:
            raise ValueError(f"Unsupported modality: {resolved_modality}")

        anchor = ModalityAnchor(
            modality=resolved_modality,
            path=path,
            description=description,
            embedding=embedding,
            timestamp=datetime.now(),
        )
        memory.modality_anchors.append(anchor)

        self.index.modality_index[resolved_modality].append(memory.id)
        if path:
            self.index.path_to_memory[path] = memory.id
        self.index.memory_to_anchors.setdefault(memory.id, []).append(anchor)

        tag = f"has:{resolved_modality}"
        if tag not in memory.tags:
            memory.tags.append(tag)

        return anchor

    def detach(self, memory: MemoryUnit, path: str) -> bool:
        """Remove one attachment path from a memory."""

        before = len(memory.modality_anchors)
        memory.modality_anchors = [anchor for anchor in memory.modality_anchors if anchor.path != path]
        self.index.path_to_memory.pop(path, None)

        anchors = self.index.memory_to_anchors.get(memory.id, [])
        self.index.memory_to_anchors[memory.id] = [anchor for anchor in anchors if anchor.path != path]
        if not self.index.memory_to_anchors[memory.id]:
            self.index.memory_to_anchors.pop(memory.id, None)

        for ids in self.index.modality_index.values():
            while memory.id in ids and not memory.modality_anchors:
                ids.remove(memory.id)

        return len(memory.modality_anchors) != before

    def remove_memory(self, memory_id: str) -> bool:
        """Remove all attachment references for a memory ID."""

        removed = memory_id in self.index.memory_to_anchors
        anchors = self.index.memory_to_anchors.pop(memory_id, [])
        for anchor in anchors:
            if anchor.path:
                self.index.path_to_memory.pop(anchor.path, None)
            if memory_id in self.index.modality_index.get(anchor.modality, []):
                self.index.modality_index[anchor.modality] = [
                    mid for mid in self.index.modality_index[anchor.modality] if mid != memory_id
                ]
        return removed

    def find_by_modality(self, modality: str) -> List[str]:
        return sorted(set(self.index.modality_index.get(modality, [])))

    def get_anchors(self, memory_id: str) -> List[ModalityAnchor]:
        return list(self.index.memory_to_anchors.get(memory_id, []))

    def find_by_embedding_similarity(
        self,
        query_embedding: List[float],
        modality: str = "image",
        top_k: int = 5,
    ) -> List[tuple]:
        scores = []
        for memory_id, anchors in self.index.memory_to_anchors.items():
            for anchor in anchors:
                if anchor.modality == modality and anchor.embedding:
                    similarity = self._cosine_similarity(query_embedding, anchor.embedding)
                    scores.append((memory_id, similarity))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(left * right for left, right in zip(a, b))
        norm_a = sum(value**2 for value in a) ** 0.5
        norm_b = sum(value**2 for value in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get_stats(self) -> Dict:
        return {
            "total_anchors": sum(len(values) for values in self.index.modality_index.values()),
            "by_modality": {
                key: len(set(values)) for key, values in self.index.modality_index.items() if values
            },
            "memories_with_media": len(self.index.memory_to_anchors),
        }
