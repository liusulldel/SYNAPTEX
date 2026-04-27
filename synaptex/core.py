"""Core orchestration engine for SYNAPTEX."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from synaptex.amem_graph import AMEMGraph
from synaptex.dopamine import DopamineEncoder
from synaptex.forgetting_gate import ForgettingGate
from synaptex.memgpt_pager import MemGPTPager
from synaptex.multi_agent import SharedMemoryPool
from synaptex.multimodal import MultimodalMemory
from synaptex.polyglot_compressor import PolyglotCompressor
from synaptex.reasoning_bank import ReasoningBank
from synaptex.swiftmem import SwiftMemEngine
from synaptex.types import AgentIdentity, EmotionType, ImportanceLabel, MemoryUnit


class SynaptexEngine:
    """Main agentic memory engine.

    The engine wires together importance weighting, tri-layer compression,
    graph associations, indexed retrieval, active-context paging, shared
    memory, multimodal anchors, and reasoning traces.
    """

    def __init__(
        self,
        max_context_tokens: int = 4096,
        similarity_threshold: float = 0.3,
        enable_polyglot: bool = True,
    ):
        self.dopamine = DopamineEncoder()
        self.forgetter = ForgettingGate()
        self.compressor = PolyglotCompressor(enable_polyglot_routing=enable_polyglot)
        self.graph = AMEMGraph(similarity_threshold=similarity_threshold)
        self.retriever = SwiftMemEngine()
        self.pager = MemGPTPager(max_context_tokens=max_context_tokens)
        self.shared_memory = SharedMemoryPool()
        self.multimodal = MultimodalMemory()
        self.reasoning = ReasoningBank()

        self._total_encoded = 0
        self._total_duplicates = 0
        self._total_recalled = 0
        self._total_tokens_saved = 0

    def register_agent(self, agent: AgentIdentity) -> None:
        """Register an agent for shared-memory operations."""

        self.shared_memory.register_agent(agent)

    def _ensure_agent_registered(self, agent_id: str) -> None:
        if agent_id not in self.shared_memory.agents:
            self.register_agent(AgentIdentity(agent_id=agent_id))

    def _write_shared_memory(self, agent_id: Optional[str], memory: MemoryUnit) -> bool:
        if not agent_id:
            return False
        self._ensure_agent_registered(agent_id)
        return self.shared_memory.write(agent_id, memory, scope="public")

    def encode(
        self,
        text: str,
        emotion: Optional[EmotionType] = None,
        importance_label: Optional[ImportanceLabel] = None,
        importance: Optional[float] = None,
        category: str = "",
        timestamp: Optional[datetime] = None,
        media_paths: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
    ) -> MemoryUnit:
        """Create, compress, index, and store a memory."""

        if importance_label is not None and emotion is None:
            emotion = importance_label

        resolved_timestamp = timestamp or datetime.now()
        memory = self.compressor.compress_to_memory(
            text,
            resolved_timestamp,
            category,
            emotion_weight=importance or 0.5,
        )
        if category and category not in memory.tags:
            memory.tags.append(category)

        existing = self.graph.get_by_content_hash(memory.content_hash())
        if existing:
            self._total_duplicates += 1
            if category and category not in existing.tags:
                self.graph.add_tags(existing.id, [category])
                self.retriever.index(existing)
            self._write_shared_memory(agent_id, existing)
            return existing

        self.dopamine.encode(memory, emotion, importance)
        result = self.compressor.compress(text, resolved_timestamp, category, memory.dopamine_weight)
        memory.content_l1 = result.l1_text
        memory.content_l2 = result.l2_text
        memory.compressed_lang = result.l1_lang

        self.graph.insert(memory)
        self.retriever.index(memory)
        self.pager.store(memory)

        if media_paths:
            for path in media_paths:
                self.multimodal.attach(memory, path=path)

        self._write_shared_memory(agent_id, memory)

        self._total_encoded += 1
        self._total_tokens_saved += result.original_tokens_est - result.l1_tokens_est
        return memory

    def recall(
        self,
        query: str = "",
        tags: Optional[List[str]] = None,
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        depth: int = 2,
        limit: int = 10,
        auto_page_in: bool = True,
    ) -> List[MemoryUnit]:
        """Retrieve memories by text, tags, time window, and graph links."""

        swift_result = self.retriever.query(
            text=query,
            tags=tags,
            time_start=time_start,
            time_end=time_end,
            limit=limit,
        )
        results = list(swift_result.memories)

        def passes_filters(memory: MemoryUnit) -> bool:
            if time_start and memory.timestamp < time_start:
                return False
            if time_end and memory.timestamp > time_end:
                return False
            if tags and not any(tag in memory.tags for tag in tags):
                return False
            return True

        expanded = {memory.id for memory in results}
        for memory in results[:3]:
            for related in self.graph.find_related(memory.id, depth=depth):
                if related.id not in expanded and passes_filters(related):
                    results.append(related)
                    expanded.add(related.id)

        results = [
            memory
            for memory in results
            if self.forgetter.compute_decay(memory) > self.forgetter.forget_threshold
        ]
        results.sort(
            key=lambda memory: memory.dopamine_weight * memory.decay_score,
            reverse=True,
        )
        results = results[:limit]

        if auto_page_in and results:
            self.pager.auto_page_in(query, results, top_k=min(5, len(results)))

        self._total_recalled += len(results)
        return results

    def get_context(self) -> str:
        """Return compact active context for prompt injection."""

        return self.pager.get_context_summary()

    def get_active_memories(self) -> List[MemoryUnit]:
        """Return memories currently loaded into active context."""

        return self.pager.get_active_context()

    def forget(self, memory_id: str) -> bool:
        """Remove a memory from graph, retrieval, pager, and attachment indices."""

        removed = False
        removed = self.graph.remove(memory_id) or removed
        removed = self.retriever.remove(memory_id) or removed
        removed = self.pager.remove(memory_id, reason="manual_forget") or removed
        removed = self.multimodal.remove_memory(memory_id) or removed
        return removed

    def night_mode(self) -> Dict:
        """Run consolidation/decay and prune forgotten memories."""

        all_memories = list(self.retriever.memories.values())
        consolidation_stats = self.forgetter.night_consolidation(all_memories)

        for memory in all_memories:
            if memory.status.value == "forgotten":
                self.forget(memory.id)

        self.retriever.co_consolidate()
        return consolidation_stats

    def get_stats(self) -> Dict:
        """Return system statistics."""

        return {
            "engine": {
                "total_encoded": self._total_encoded,
                "total_duplicates": self._total_duplicates,
                "total_recalled": self._total_recalled,
                "total_tokens_saved": self._total_tokens_saved,
            },
            "graph": self.graph.get_graph_stats(),
            "retriever": self.retriever.get_stats(),
            "pager": self.pager.get_stats(),
            "multimodal": self.multimodal.get_stats(),
            "reasoning": self.reasoning.get_stats(),
            "shared_memory": self.shared_memory.get_pool_stats(),
        }
