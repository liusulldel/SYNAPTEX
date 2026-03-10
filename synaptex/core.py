"""
SYNAPTEX·触链典 — Core Engine (中枢路由器)

The central orchestrator that wires all subsystems together:
- Dopamine Encoder → Forgetting Gate → Polyglot Compressor
- A-MEM Graph ↔ SwiftMem Index ↔ MemGPT Pager
- Multi-Agent Pool ↔ ReasoningBank ↔ Multimodal Anchors

This is the single entry point for interacting with SYNAPTEX.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from datetime import datetime

from synaptex.types import MemoryUnit, EmotionType
from synaptex.dopamine import DopamineEncoder
from synaptex.forgetting_gate import ForgettingGate
from synaptex.polyglot_compressor import PolyglotCompressor
from synaptex.amem_graph import AMEMGraph
from synaptex.swiftmem import SwiftMemEngine
from synaptex.memgpt_pager import MemGPTPager
from synaptex.multi_agent import SharedMemoryPool
from synaptex.multimodal import MultimodalMemory
from synaptex.reasoning_bank import ReasoningBank


class SynaptexEngine:
    """
    ⚡ SYNAPTEX·触链典 — Central Memory Engine
    
    Orchestrates:
    ┌──────────────────────────────────────────────────────────┐
    │                    SYNAPTEX Core Router                   │
    │                                                          │
    │  Input → Dopamine → Compressor → [L1/L2/L3]            │
    │                                    ↓                     │
    │                    A-MEM Graph ←→ SwiftMem Index         │
    │                         ↓                                │
    │                    MemGPT Pager → Context Window          │
    │                         ↓                                │
    │              Forgetting Gate (periodic pruning)           │
    │                                                          │
    │  Multi-Agent Pool ←→ ReasoningBank ←→ Multimodal         │
    └──────────────────────────────────────────────────────────┘
    
    Usage:
        engine = SynaptexEngine()
        
        # Encode a new memory
        mem = engine.encode(
            "Had a breakthrough meeting with Prof. Zhang about Bayesian methods",
            emotion=EmotionType.JOY,
            importance=0.9,
            category="学术",
        )
        
        # Recall memories
        results = engine.recall("Zhang meeting")
        
        # Get compressed context for LLM injection
        context = engine.get_context()
    """

    def __init__(
        self,
        max_context_tokens: int = 4096,
        similarity_threshold: float = 0.3,
        enable_polyglot: bool = True,
    ):
        # Initialize all subsystems
        self.dopamine = DopamineEncoder()
        self.forgetter = ForgettingGate()
        self.compressor = PolyglotCompressor(enable_polyglot_routing=enable_polyglot)
        self.graph = AMEMGraph(similarity_threshold=similarity_threshold)
        self.retriever = SwiftMemEngine()
        self.pager = MemGPTPager(max_context_tokens=max_context_tokens)
        self.shared_memory = SharedMemoryPool()
        self.multimodal = MultimodalMemory()
        self.reasoning = ReasoningBank()

        # Statistics
        self._total_encoded = 0
        self._total_recalled = 0
        self._total_tokens_saved = 0

    def encode(
        self,
        text: str,
        emotion: Optional[EmotionType] = None,
        importance: Optional[float] = None,
        category: str = "",
        timestamp: Optional[datetime] = None,
        media_paths: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
    ) -> MemoryUnit:
        """
        Full encoding pipeline: text → dopamine → compress → index → store.
        
        Args:
            text: Raw input text
            emotion: Emotion label for dopamine weighting
            importance: Manual importance override ∈ [0, 1]
            category: Category tag (学术/生活/事业/etc.)
            timestamp: Event timestamp (defaults to now)
            media_paths: Optional media files to attach
            agent_id: If in multi-agent mode, the source agent
            
        Returns:
            Fully processed MemoryUnit
        """
        timestamp = timestamp or datetime.now()

        # 1. Create raw memory (L3)
        memory = self.compressor.compress_to_memory(
            text, timestamp, category,
            emotion_weight=importance or 0.5,
        )

        # 2. Dopamine encoding
        self.dopamine.encode(memory, emotion, importance)

        # 3. Re-compress L1 with updated dopamine weight
        result = self.compressor.compress(text, timestamp, category, memory.dopamine_weight)
        memory.content_l1 = result.l1_text
        memory.content_l2 = result.l2_text

        # 4. Insert into A-MEM knowledge graph
        if category:
            memory.tags.append(category)
        self.graph.insert(memory)

        # 5. Index in SwiftMem
        self.retriever.index(memory)

        # 6. Store in MemGPT pager (archival)
        self.pager.store(memory)

        # 7. Attach media if provided
        if media_paths:
            for path in media_paths:
                self.multimodal.attach(memory, path=path)

        # 8. Multi-agent write
        if agent_id:
            self.shared_memory.write(agent_id, memory, scope="public")

        # Stats
        self._total_encoded += 1
        self._total_tokens_saved += (result.original_tokens_est - result.l1_tokens_est)

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
        """
        Recall memories using hybrid retrieval.
        
        Combines:
        1. SwiftMem index search (fast, sub-linear)
        2. A-MEM graph traversal (association-rich)
        3. MemGPT paging (loads results into active context)
        
        Args:
            query: Free-text query
            tags: Tag filters
            time_start/end: Time range filters
            depth: Graph traversal depth
            limit: Max results
            auto_page_in: Automatically page results into RAM
            
        Returns:
            List of recalled MemoryUnits
        """
        # 1. SwiftMem retrieval
        swift_result = self.retriever.query(
            text=query, tags=tags,
            time_start=time_start, time_end=time_end,
            limit=limit,
        )
        results = swift_result.memories

        # 2. Graph-augmented retrieval: expand via associations
        expanded = set(m.id for m in results)
        for mem in results[:3]:  # Expand top-3 results
            related = self.graph.find_related(mem.id, depth=depth)
            for rel in related:
                if rel.id not in expanded:
                    results.append(rel)
                    expanded.add(rel.id)

        # 3. Apply forgetting gate (filter out decayed memories)
        results = [
            m for m in results
            if self.forgetter.compute_decay(m) > self.forgetter.forget_threshold
        ]

        # Sort by relevance (dopamine weight × decay score)
        results.sort(
            key=lambda m: m.dopamine_weight * m.decay_score,
            reverse=True,
        )
        results = results[:limit]

        # 4. Page into active context
        if auto_page_in and results:
            self.pager.auto_page_in(query, results, top_k=min(5, len(results)))

        # Stats
        self._total_recalled += len(results)

        return results

    def get_context(self) -> str:
        """
        Get the current active context (what's in RAM).
        Returns L1-compressed text for injection into LLM context window.
        """
        return self.pager.get_context_summary()

    def get_active_memories(self) -> List[MemoryUnit]:
        """Get all memories currently paged into active context."""
        return self.pager.get_active_context()

    def forget(self, memory_id: str):
        """Manually forget a specific memory."""
        self.graph.remove(memory_id)
        self.retriever.remove(memory_id)
        self.pager.page_out(
            memory_id, reason="manual_forget"
        ) if memory_id in str(self.pager.main_context) else None

    def night_mode(self) -> Dict:
        """
        Run overnight consolidation cycle.
        
        1. Apply forgetting gate to all memories
        2. Prune forgotten memories from indices
        3. Co-consolidate SwiftMem indices
        4. Return statistics
        """
        all_memories = list(self.retriever.memories.values())
        
        # Forgetting gate pass
        consolidation_stats = self.forgetter.night_consolidation(all_memories)
        
        # Prune forgotten from graph and index
        for mem in all_memories:
            if mem.status.value == "forgotten":
                self.graph.remove(mem.id)
                self.retriever.remove(mem.id)
        
        # Co-consolidate SwiftMem
        self.retriever.co_consolidate()
        
        return consolidation_stats

    def get_stats(self) -> Dict:
        """Return comprehensive system statistics."""
        return {
            "engine": {
                "total_encoded": self._total_encoded,
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
