"""Indexed retrieval for SYNAPTEX memories."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import time
from typing import Dict, List, Optional, Set

from synaptex.types import MemoryUnit


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""

    memories: List[MemoryUnit]
    search_time_ms: float
    index_used: str
    candidates_scanned: int
    total_stored: int


@dataclass
class DAGNode:
    """Node in the tag hierarchy."""

    tag: str
    memory_ids: Set[str] = field(default_factory=set)
    children: Dict[str, "DAGNode"] = field(default_factory=dict)
    parent: Optional[str] = None


class SwiftMemEngine:
    """Lightweight query, tag, and time-window retrieval engine."""

    ROOT_CATEGORIES = {
        "research": ["paper", "experiment", "review", "benchmark"],
        "engineering": ["code", "test", "ci", "release"],
        "agents": ["memory", "tool", "context", "retrieval"],
        "operations": ["deadline", "meeting", "decision", "risk"],
        "other": [],
    }

    def __init__(self):
        self.memories: Dict[str, MemoryUnit] = {}
        self._temporal_keys: List[datetime] = []
        self._temporal_ids: List[str] = []
        self._dag_roots: Dict[str, DAGNode] = {}
        self._tag_flat: Dict[str, Set[str]] = defaultdict(set)
        self._build_dag_skeleton()

    def _build_dag_skeleton(self) -> None:
        for root_tag, children in self.ROOT_CATEGORIES.items():
            root = DAGNode(tag=root_tag)
            for child_tag in children:
                root.children[child_tag] = DAGNode(tag=child_tag, parent=root_tag)
            self._dag_roots[root_tag] = root

    def index(self, memory: MemoryUnit) -> None:
        """Index a memory, replacing any existing copy with the same ID."""

        if memory.id in self.memories:
            self.remove(memory.id)

        self.memories[memory.id] = memory

        pos = bisect_left(self._temporal_keys, memory.timestamp)
        self._temporal_keys.insert(pos, memory.timestamp)
        self._temporal_ids.insert(pos, memory.id)

        for tag in memory.tags:
            self._tag_flat[tag].add(memory.id)
            self._insert_into_dag(memory.id, tag)

    def _insert_into_dag(self, memory_id: str, tag: str) -> None:
        for root in self._dag_roots.values():
            if tag == root.tag:
                root.memory_ids.add(memory_id)
                return
            if tag in root.children:
                root.children[tag].memory_ids.add(memory_id)
                return
        self._dag_roots["other"].memory_ids.add(memory_id)

    def query_temporal(self, start: datetime, end: datetime) -> List[MemoryUnit]:
        left = bisect_left(self._temporal_keys, start)
        right = bisect_right(self._temporal_keys, end)
        return [
            self.memories[self._temporal_ids[index]]
            for index in range(left, right)
            if self._temporal_ids[index] in self.memories
        ]

    def query_semantic(self, tags: List[str]) -> List[MemoryUnit]:
        matching_ids: Set[str] = set()

        for tag in tags:
            matching_ids.update(self._tag_flat.get(tag, set()))
            if tag in self._dag_roots:
                root = self._dag_roots[tag]
                matching_ids.update(root.memory_ids)
                for child in root.children.values():
                    matching_ids.update(child.memory_ids)

        return [self.memories[mid] for mid in matching_ids if mid in self.memories]

    def query(
        self,
        text: str = "",
        tags: Optional[List[str]] = None,
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        limit: int = 20,
    ) -> RetrievalResult:
        """Query by text, tags, time window, or a combination."""

        started = time.perf_counter()
        results: Set[str] = set()
        query_tags = list(tags or [])
        used_temporal = False
        used_semantic = False

        if text:
            words = {word.strip(".,:;!?()[]{}").lower() for word in text.split()}
            for tag in self._tag_flat:
                if tag.lower() in words and tag not in query_tags:
                    query_tags.append(tag)

        if time_start and time_end:
            temporal_results = self.query_temporal(time_start, time_end)
            results.update(memory.id for memory in temporal_results)
            used_temporal = True

        if query_tags:
            semantic_results = self.query_semantic(query_tags)
            semantic_ids = {memory.id for memory in semantic_results}
            results = (results & semantic_ids) if used_temporal else semantic_ids
            used_semantic = True

        if not results and text:
            text_lower = text.lower()
            for memory_id, memory in self.memories.items():
                if text_lower in memory.content_l3.lower():
                    results.add(memory_id)

        if used_temporal and used_semantic:
            index_used = "hybrid"
        elif used_temporal:
            index_used = "temporal"
        elif used_semantic:
            index_used = "semantic_dag"
        else:
            index_used = "fullscan_fallback" if text else "none"

        result_memories = [self.memories[mid] for mid in results if mid in self.memories]
        result_memories.sort(key=lambda memory: (memory.dopamine_weight, memory.timestamp), reverse=True)
        elapsed_ms = (time.perf_counter() - started) * 1000

        return RetrievalResult(
            memories=result_memories[:limit],
            search_time_ms=round(elapsed_ms, 3),
            index_used=index_used,
            candidates_scanned=len(results),
            total_stored=len(self.memories),
        )

    def co_consolidate(self) -> None:
        pairs = sorted(zip(self._temporal_keys, self._temporal_ids), key=lambda pair: pair[0])
        self._temporal_keys = [pair[0] for pair in pairs]
        self._temporal_ids = [pair[1] for pair in pairs]

        self._tag_flat.clear()
        for root in self._dag_roots.values():
            root.memory_ids.clear()
            for child in root.children.values():
                child.memory_ids.clear()
        for memory in self.memories.values():
            for tag in memory.tags:
                self._tag_flat[tag].add(memory.id)
                self._insert_into_dag(memory.id, tag)

    def remove(self, memory_id: str) -> bool:
        if memory_id not in self.memories:
            return False

        memory = self.memories[memory_id]
        self._temporal_keys = [
            key for key, mid in zip(self._temporal_keys, self._temporal_ids) if mid != memory_id
        ]
        self._temporal_ids = [mid for mid in self._temporal_ids if mid != memory_id]

        for tag in memory.tags:
            self._tag_flat[tag].discard(memory_id)
            if not self._tag_flat[tag]:
                self._tag_flat.pop(tag, None)

        for root in self._dag_roots.values():
            root.memory_ids.discard(memory_id)
            for child in root.children.values():
                child.memory_ids.discard(memory_id)

        del self.memories[memory_id]
        return True

    def get_stats(self) -> Dict:
        return {
            "total_memories": len(self.memories),
            "temporal_index_size": len(self._temporal_keys),
            "total_tags": len(self._tag_flat),
            "dag_roots": list(self._dag_roots.keys()),
        }
