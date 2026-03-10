"""
SYNAPTEX·触链典 — A-MEM Zettelkasten Knowledge Graph

Inspired by A-MEM (Xu et al., 2025):
Autonomous memory organization using Zettelkasten principles.
Memories self-organize into an evolving knowledge graph with:
- Auto-generated tags and descriptions
- Dynamic link generation between related memories
- Memory evolution: existing memories update when new info arrives
- Graph traversal for association-rich retrieval
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import re

from synaptex.types import MemoryUnit


@dataclass
class ZettelNode:
    """
    A node in the Zettelkasten graph.
    Wraps a MemoryUnit with graph-specific metadata.
    """
    memory: MemoryUnit
    neighbors: Set[str] = field(default_factory=set)   # neighbor memory IDs
    backlinks: Set[str] = field(default_factory=set)    # who links to me
    cluster_id: Optional[str] = None                     # semantic cluster


class AMEMGraph:
    """
    Autonomous Memory Graph with Zettelkasten-style linking.
    
    Core operations:
    1. INSERT: Add memory → auto-tag → auto-link to similar memories
    2. EVOLVE: When new memory overlaps with existing, update both
    3. TRAVERSE: Walk the graph from a seed memory to find associations
    4. PRUNE: Remove forgotten memories and re-link orphaned neighbors
    
    The graph maintains three index structures:
    - tag_index: tag → set of memory IDs
    - temporal_index: sorted by timestamp for range queries
    - content_fingerprint: hash → memory ID for deduplication
    
    Usage:
        graph = AMEMGraph()
        mem = MemoryUnit(content_l3="Met Prof. Zhang for thesis discussion")
        graph.insert(mem)
        
        related = graph.find_related(mem.id, depth=2)
    """

    def __init__(self, similarity_threshold: float = 0.3):
        self.nodes: Dict[str, ZettelNode] = {}
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_index: List[str] = []  # ordered by timestamp
        self.content_hashes: Dict[str, str] = {}  # hash → memory_id
        self.similarity_threshold = similarity_threshold

    @property
    def size(self) -> int:
        return len(self.nodes)

    def insert(self, memory: MemoryUnit) -> List[str]:
        """
        Insert a memory into the graph.
        
        Pipeline:
        1. Check for duplicates via content hash
        2. Auto-generate tags if not present
        3. Find and create links to similar memories
        4. Trigger evolution of overlapping memories
        
        Returns:
            List of memory IDs that were linked
        """
        # Deduplication check
        content_hash = memory.content_hash()
        if content_hash in self.content_hashes:
            existing_id = self.content_hashes[content_hash]
            # Evolve existing memory instead
            self._evolve(existing_id, memory)
            return [existing_id]

        # Auto-tag generation
        if not memory.tags:
            memory.tags = self._extract_tags(memory.content_l3)

        # Create node
        node = ZettelNode(memory=memory)
        self.nodes[memory.id] = node
        self.content_hashes[content_hash] = memory.id

        # Update tag index
        for tag in memory.tags:
            self.tag_index[tag].add(memory.id)

        # Insert into temporal index (maintain sorted order)
        self.temporal_index.append(memory.id)
        self.temporal_index.sort(
            key=lambda mid: self.nodes[mid].memory.timestamp
        )

        # Auto-link to similar memories
        linked_ids = self._auto_link(memory.id)

        return linked_ids

    def _extract_tags(self, text: str) -> List[str]:
        """
        Extract keyword tags from text content.
        
        Uses simple heuristics:
        - Named entities (capitalized words)
        - Domain keywords
        - CJK key terms
        """
        tags = []

        # Extract capitalized words (likely names/entities)
        caps = re.findall(r'\b[A-Z][a-z]+\b', text)
        tags.extend(caps[:5])

        # Extract CJK key terms (2-4 char sequences)
        cjk = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        tags.extend(cjk[:5])

        # Common domain keywords
        domain_keywords = {
            "meeting", "research", "paper", "deadline", "project",
            "thesis", "experiment", "data", "code", "review",
            "学术", "会议", "论文", "项目", "实验",
        }
        words = set(text.lower().split())
        tags.extend(words & domain_keywords)

        return list(set(tags))[:10]  # Cap at 10 tags

    def _compute_similarity(self, id_a: str, id_b: str) -> float:
        """
        Compute tag-based Jaccard similarity between two memories.
        
        In production, this would use embedding-based cosine similarity.
        For now, tag overlap is a fast proxy.
        """
        tags_a = set(self.nodes[id_a].memory.tags)
        tags_b = set(self.nodes[id_b].memory.tags)

        if not tags_a or not tags_b:
            return 0.0

        intersection = len(tags_a & tags_b)
        union = len(tags_a | tags_b)
        return intersection / union if union > 0 else 0.0

    def _auto_link(self, memory_id: str) -> List[str]:
        """
        Find and link similar memories based on tag overlap.
        Creates bidirectional links.
        """
        linked = []
        node = self.nodes[memory_id]

        for other_id, other_node in self.nodes.items():
            if other_id == memory_id:
                continue

            sim = self._compute_similarity(memory_id, other_id)
            if sim >= self.similarity_threshold:
                # Bidirectional link
                node.neighbors.add(other_id)
                node.memory.links.add(other_id)
                other_node.neighbors.add(memory_id)
                other_node.backlinks.add(memory_id)
                other_node.memory.links.add(memory_id)
                linked.append(other_id)

        return linked

    def _evolve(self, existing_id: str, new_memory: MemoryUnit):
        """
        Evolve an existing memory when new overlapping information arrives.
        
        - Merge tags
        - Update L3 content (append new information)
        - Boost dopamine weight if the topic recurs
        """
        existing = self.nodes[existing_id].memory

        # Merge tags
        new_tags = self._extract_tags(new_memory.content_l3)
        existing.tags = list(set(existing.tags + new_tags))[:15]

        # Append to L3
        existing.content_l3 += f"\n[UPDATE {datetime.now().isoformat()}] {new_memory.content_l3}"

        # Boost dopamine (recurring topic = increased importance)
        existing.dopamine_weight = min(1.0, existing.dopamine_weight + 0.05)
        existing.access_count += 1

        # Re-index tags
        for tag in new_tags:
            self.tag_index[tag].add(existing_id)

    def find_related(self, memory_id: str, depth: int = 2) -> List[MemoryUnit]:
        """
        BFS traversal from a seed memory to find associated memories.
        
        Args:
            memory_id: Starting node
            depth: Maximum hops from the seed
            
        Returns:
            List of related MemoryUnits, ordered by proximity
        """
        if memory_id not in self.nodes:
            return []

        visited = set()
        queue = [(memory_id, 0)]
        results = []

        while queue:
            current_id, current_depth = queue.pop(0)

            if current_id in visited or current_depth > depth:
                continue
            visited.add(current_id)

            if current_id != memory_id:
                results.append(self.nodes[current_id].memory)

            if current_depth < depth:
                for neighbor_id in self.nodes[current_id].neighbors:
                    if neighbor_id not in visited:
                        queue.append((neighbor_id, current_depth + 1))

        return results

    def find_by_tags(self, tags: List[str]) -> List[MemoryUnit]:
        """Find all memories matching any of the given tags."""
        matching_ids: Set[str] = set()
        for tag in tags:
            matching_ids.update(self.tag_index.get(tag, set()))

        return [self.nodes[mid].memory for mid in matching_ids if mid in self.nodes]

    def find_by_time_range(
        self, start: datetime, end: datetime
    ) -> List[MemoryUnit]:
        """Find memories within a time range using the temporal index."""
        return [
            self.nodes[mid].memory
            for mid in self.temporal_index
            if mid in self.nodes
            and start <= self.nodes[mid].memory.timestamp <= end
        ]

    def remove(self, memory_id: str):
        """Remove a memory and clean up all its links."""
        if memory_id not in self.nodes:
            return

        node = self.nodes[memory_id]

        # Remove from neighbors' link lists
        for neighbor_id in node.neighbors:
            if neighbor_id in self.nodes:
                self.nodes[neighbor_id].neighbors.discard(memory_id)
                self.nodes[neighbor_id].backlinks.discard(memory_id)
                self.nodes[neighbor_id].memory.links.discard(memory_id)

        # Remove from indices
        for tag in node.memory.tags:
            self.tag_index[tag].discard(memory_id)

        self.temporal_index = [mid for mid in self.temporal_index if mid != memory_id]
        content_hash = node.memory.content_hash()
        self.content_hashes.pop(content_hash, None)

        del self.nodes[memory_id]

    def get_graph_stats(self) -> Dict:
        """Return summary statistics of the knowledge graph."""
        total_links = sum(len(n.neighbors) for n in self.nodes.values())
        return {
            "total_memories": self.size,
            "total_links": total_links // 2,  # bidirectional
            "total_tags": len(self.tag_index),
            "avg_links_per_node": total_links / max(1, self.size),
            "orphan_nodes": sum(
                1 for n in self.nodes.values() if not n.neighbors
            ),
        }
