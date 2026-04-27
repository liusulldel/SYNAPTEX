"""Graph associations for SYNAPTEX memories."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Dict, List, Optional, Set

from synaptex.types import MemoryUnit


@dataclass
class ZettelNode:
    """A memory plus graph-specific links."""

    memory: MemoryUnit
    neighbors: Set[str] = field(default_factory=set)
    backlinks: Set[str] = field(default_factory=set)
    cluster_id: Optional[str] = None


class AMEMGraph:
    """Small Zettelkasten-style graph for memory associations."""

    DOMAIN_KEYWORDS = {
        "agent",
        "benchmark",
        "code",
        "context",
        "deadline",
        "experiment",
        "memory",
        "paper",
        "project",
        "review",
        "retrieval",
        "safety",
        "test",
        "tool",
    }

    def __init__(self, similarity_threshold: float = 0.3):
        self.nodes: Dict[str, ZettelNode] = {}
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_index: List[str] = []
        self.content_hashes: Dict[str, str] = {}
        self.similarity_threshold = similarity_threshold

    @property
    def size(self) -> int:
        return len(self.nodes)

    def get(self, memory_id: str) -> Optional[MemoryUnit]:
        node = self.nodes.get(memory_id)
        return node.memory if node else None

    def get_by_content_hash(self, content_hash: str) -> Optional[MemoryUnit]:
        memory_id = self.content_hashes.get(content_hash)
        return self.get(memory_id) if memory_id else None

    def insert(self, memory: MemoryUnit) -> List[str]:
        """Insert a memory and return IDs linked to it."""

        content_hash = memory.content_hash()
        if content_hash in self.content_hashes:
            existing_id = self.content_hashes[content_hash]
            self.add_tags(existing_id, memory.tags)
            return [existing_id]

        if not memory.tags:
            memory.tags = self._extract_tags(memory.content_l3)

        self.nodes[memory.id] = ZettelNode(memory=memory)
        self.content_hashes[content_hash] = memory.id

        for tag in memory.tags:
            self.tag_index[tag].add(memory.id)

        self.temporal_index.append(memory.id)
        self.temporal_index.sort(key=lambda mid: self.nodes[mid].memory.timestamp)

        return self._auto_link(memory.id)

    def add_tags(self, memory_id: str, tags: List[str]) -> None:
        """Add tags to an existing memory and update graph indices."""

        node = self.nodes.get(memory_id)
        if not node:
            return

        for tag in tags:
            if tag and tag not in node.memory.tags:
                node.memory.tags.append(tag)
            if tag:
                self.tag_index[tag].add(memory_id)

    def _extract_tags(self, text: str) -> List[str]:
        tags: List[str] = []

        caps = re.findall(r"\b[A-Z][a-z]+\b", text)
        tags.extend(caps[:5])

        words = {word.strip(".,:;!?()[]{}").lower() for word in text.split()}
        tags.extend(sorted(words & self.DOMAIN_KEYWORDS))

        hashtags = re.findall(r"#([A-Za-z0-9_-]+)", text)
        tags.extend(hashtags)

        deduped = []
        for tag in tags:
            if tag and tag not in deduped:
                deduped.append(tag)
        return deduped[:10]

    def _compute_similarity(self, id_a: str, id_b: str) -> float:
        tags_a = set(self.nodes[id_a].memory.tags)
        tags_b = set(self.nodes[id_b].memory.tags)
        if not tags_a or not tags_b:
            return 0.0
        return len(tags_a & tags_b) / len(tags_a | tags_b)

    def _auto_link(self, memory_id: str) -> List[str]:
        linked: List[str] = []
        node = self.nodes[memory_id]

        for other_id, other_node in self.nodes.items():
            if other_id == memory_id:
                continue

            similarity = self._compute_similarity(memory_id, other_id)
            if similarity >= self.similarity_threshold:
                node.neighbors.add(other_id)
                node.memory.links.add(other_id)
                other_node.neighbors.add(memory_id)
                other_node.backlinks.add(memory_id)
                other_node.memory.links.add(memory_id)
                linked.append(other_id)

        return linked

    def find_related(self, memory_id: str, depth: int = 2) -> List[MemoryUnit]:
        """Traverse graph links from a seed memory."""

        if memory_id not in self.nodes:
            return []

        visited = set()
        queue = [(memory_id, 0)]
        results: List[MemoryUnit] = []

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
        matching_ids: Set[str] = set()
        for tag in tags:
            matching_ids.update(self.tag_index.get(tag, set()))
        return [self.nodes[mid].memory for mid in matching_ids if mid in self.nodes]

    def find_by_time_range(self, start: datetime, end: datetime) -> List[MemoryUnit]:
        return [
            self.nodes[mid].memory
            for mid in self.temporal_index
            if mid in self.nodes and start <= self.nodes[mid].memory.timestamp <= end
        ]

    def remove(self, memory_id: str) -> bool:
        """Remove a memory and clean up graph links."""

        if memory_id not in self.nodes:
            return False

        node = self.nodes[memory_id]
        for neighbor_id in list(node.neighbors):
            if neighbor_id in self.nodes:
                self.nodes[neighbor_id].neighbors.discard(memory_id)
                self.nodes[neighbor_id].backlinks.discard(memory_id)
                self.nodes[neighbor_id].memory.links.discard(memory_id)

        for tag in node.memory.tags:
            self.tag_index[tag].discard(memory_id)
            if not self.tag_index[tag]:
                self.tag_index.pop(tag, None)

        self.temporal_index = [mid for mid in self.temporal_index if mid != memory_id]
        self.content_hashes.pop(node.memory.content_hash(), None)
        del self.nodes[memory_id]
        return True

    def get_graph_stats(self) -> Dict:
        total_links = sum(len(node.neighbors) for node in self.nodes.values())
        return {
            "total_memories": self.size,
            "total_links": total_links // 2,
            "total_tags": len(self.tag_index),
            "avg_links_per_node": total_links / max(1, self.size),
            "orphan_nodes": sum(1 for node in self.nodes.values() if not node.neighbors),
        }
