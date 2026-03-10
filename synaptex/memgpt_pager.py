"""
SYNAPTEX·触链典 — MemGPT Virtual Context Pager

Inspired by MemGPT (Packer et al., 2023-2026):
OS-inspired virtual memory management for LLM context windows.

Treats the context window as RAM and archival storage as Disk:
- Main Context (RAM): Active working memory within token budget
- Archival Storage (Disk): Unlimited cold storage for historical memories
- Page-in/Page-out: Agent-controlled memory swapping

Breaks the physical context window limit by intelligently
paging memories in and out based on relevance and priority.
"""

from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

from synaptex.types import MemoryUnit, ContextPage


@dataclass
class PagingEvent:
    """Records a page-in or page-out event."""
    event_type: str  # "page_in" or "page_out"
    page_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str = ""


class MemGPTPager:
    """
    Virtual Context Manager — OS-inspired memory paging.
    
    Architecture:
    ┌────────────────────────────────┐
    │  Main Context (RAM)            │  ← Active pages, within token budget
    │  Token Budget: max_context_tokens│
    ├────────────────────────────────┤
    │  Warm Cache                    │  ← Recently evicted, fast re-load
    ├────────────────────────────────┤
    │  Archival Storage (Disk)       │  ← All memories, unlimited capacity
    └────────────────────────────────┘
    
    Eviction Policy: LRU with priority boosting
    - Pinned pages are never evicted
    - High-dopamine pages get priority boost
    - Least-recently-used pages evicted first
    
    Usage:
        pager = MemGPTPager(max_context_tokens=4096)
        pager.store(memory)  # Goes to archival
        pager.page_in(memory.id)  # Loads into main context
        
        context = pager.get_active_context()  # What's currently in "RAM"
    """

    def __init__(
        self,
        max_context_tokens: int = 4096,
        warm_cache_size: int = 50,
        tokens_per_l1: int = 15,
        tokens_per_l2: int = 80,
    ):
        self.max_tokens = max_context_tokens
        self.warm_cache_size = warm_cache_size
        self.tokens_per_l1 = tokens_per_l1
        self.tokens_per_l2 = tokens_per_l2
        
        # Main Context (RAM) — OrderedDict for LRU tracking
        self.main_context: OrderedDict[str, ContextPage] = OrderedDict()
        self.current_token_usage: int = 0
        
        # Warm Cache — recently evicted pages
        self.warm_cache: OrderedDict[str, ContextPage] = OrderedDict()
        
        # Archival Storage (Disk) — all memories
        self.archival: Dict[str, MemoryUnit] = {}
        
        # Pages mapping
        self.pages: Dict[str, ContextPage] = {}
        
        # Event log
        self.paging_history: List[PagingEvent] = []

    def _estimate_page_tokens(self, page: ContextPage) -> int:
        """Estimate token cost of loading a page into context."""
        if page.token_count > 0:
            return page.token_count
        
        # Estimate based on number of memories and their layers
        total = 0
        for mid in page.memory_ids:
            if mid in self.archival:
                mem = self.archival[mid]
                # L1 is what we'd load into context
                if mem.content_l1:
                    total += len(mem.content_l1)
                else:
                    total += self.tokens_per_l1
        
        page.token_count = max(1, total)
        return page.token_count

    def store(self, memory: MemoryUnit) -> str:
        """
        Store a memory in archival storage. Creates a page wrapper.
        Returns the page_id.
        """
        self.archival[memory.id] = memory
        
        # Create a page for this memory
        page = ContextPage(
            memory_ids=[memory.id],
            summary=memory.content_l1 or memory.content_l3[:50],
            priority=memory.dopamine_weight,
            is_pinned=False,
        )
        self._estimate_page_tokens(page)
        self.pages[page.page_id] = page
        
        return page.page_id

    def page_in(self, memory_id: str, reason: str = "") -> bool:
        """
        Load a memory from archival/warm cache into main context (RAM).
        
        If context is full, evicts LRU non-pinned pages first.
        
        Returns:
            True if page-in succeeded, False if memory not found
        """
        if memory_id not in self.archival:
            return False
        
        # Find the page containing this memory
        target_page = None
        for page in self.pages.values():
            if memory_id in page.memory_ids:
                target_page = page
                break
        
        if not target_page:
            return False
        
        # Already in main context?
        if target_page.page_id in self.main_context:
            # Move to end (most recently used)
            self.main_context.move_to_end(target_page.page_id)
            return True
        
        # Check warm cache first
        if target_page.page_id in self.warm_cache:
            self.warm_cache.pop(target_page.page_id)
        
        # Make room if needed
        page_tokens = self._estimate_page_tokens(target_page)
        while (
            self.current_token_usage + page_tokens > self.max_tokens
            and self.main_context
        ):
            self._evict_lru()
        
        # Page in
        self.main_context[target_page.page_id] = target_page
        self.current_token_usage += page_tokens
        target_page.last_paged_in = datetime.now()
        
        # Update access stats
        memory = self.archival[memory_id]
        memory.access_count += 1
        memory.last_accessed = datetime.now()
        
        self.paging_history.append(PagingEvent(
            event_type="page_in",
            page_id=target_page.page_id,
            reason=reason,
        ))
        
        return True

    def page_out(self, page_id: str, reason: str = ""):
        """Evict a specific page from main context to warm cache."""
        if page_id not in self.main_context:
            return
        
        page = self.main_context.pop(page_id)
        self.current_token_usage -= self._estimate_page_tokens(page)
        
        # Move to warm cache
        self.warm_cache[page_id] = page
        if len(self.warm_cache) > self.warm_cache_size:
            self.warm_cache.popitem(last=False)  # Remove oldest
        
        self.paging_history.append(PagingEvent(
            event_type="page_out",
            page_id=page_id,
            reason=reason,
        ))

    def _evict_lru(self):
        """Evict the least-recently-used non-pinned page."""
        for page_id in list(self.main_context.keys()):
            page = self.main_context[page_id]
            if not page.is_pinned:
                self.page_out(page_id, reason="LRU eviction")
                return
        
        # If all pages are pinned, we can't evict
        raise RuntimeError(
            "Cannot evict: all pages are pinned. "
            "Increase max_context_tokens or unpin some pages."
        )

    def pin(self, memory_id: str):
        """Pin a memory's page so it's never evicted."""
        for page in self.pages.values():
            if memory_id in page.memory_ids:
                page.is_pinned = True
                return

    def unpin(self, memory_id: str):
        """Unpin a memory's page, allowing eviction."""
        for page in self.pages.values():
            if memory_id in page.memory_ids:
                page.is_pinned = False
                return

    def get_active_context(self) -> List[MemoryUnit]:
        """
        Get all memories currently loaded in main context (RAM).
        This is what would be injected into the LLM's context window.
        """
        active_memories = []
        for page in self.main_context.values():
            for mid in page.memory_ids:
                if mid in self.archival:
                    active_memories.append(self.archival[mid])
        return active_memories

    def get_context_summary(self) -> str:
        """
        Generate a compressed summary of current context for injection.
        Uses L1 (Classical Chinese) for maximum compression.
        """
        memories = self.get_active_context()
        summaries = []
        for mem in memories:
            if mem.content_l1:
                summaries.append(mem.content_l1)
            else:
                summaries.append(mem.content_l3[:30] + "…")
        
        return " | ".join(summaries)

    def auto_page_in(
        self,
        query: str,
        memories: List[MemoryUnit],
        top_k: int = 5,
    ):
        """
        Automatically page in the most relevant memories for a query.
        Uses dopamine weight as a proxy for relevance.
        """
        # Sort by dopamine weight × recency
        scored = sorted(
            memories,
            key=lambda m: m.dopamine_weight * m.decay_score,
            reverse=True,
        )
        
        for mem in scored[:top_k]:
            self.page_in(mem.id, reason=f"auto: query='{query[:30]}'")

    def get_stats(self) -> Dict:
        """Return paging statistics."""
        return {
            "context_tokens_used": self.current_token_usage,
            "context_tokens_max": self.max_tokens,
            "context_utilization": f"{self.current_token_usage / self.max_tokens * 100:.1f}%",
            "pages_in_ram": len(self.main_context),
            "pages_in_warm_cache": len(self.warm_cache),
            "total_archival": len(self.archival),
            "total_paging_events": len(self.paging_history),
            "pinned_pages": sum(1 for p in self.main_context.values() if p.is_pinned),
        }
