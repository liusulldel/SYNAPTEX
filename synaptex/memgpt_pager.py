"""Active-context paging for SYNAPTEX memories."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from synaptex.types import ContextPage, MemoryUnit


@dataclass
class PagingEvent:
    """Records a page-in or page-out event."""

    event_type: str
    page_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str = ""


class MemGPTPager:
    """Manage active memory pages under a token budget."""

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
        self.main_context: OrderedDict[str, ContextPage] = OrderedDict()
        self.current_token_usage = 0
        self.warm_cache: OrderedDict[str, ContextPage] = OrderedDict()
        self.archival: Dict[str, MemoryUnit] = {}
        self.pages: Dict[str, ContextPage] = {}
        self.paging_history: List[PagingEvent] = []

    def _estimate_text_tokens(self, text: str) -> int:
        words = text.split()
        if len(words) > 1:
            return max(1, len(words))
        return max(1, len(text) // 4)

    def _estimate_page_tokens(self, page: ContextPage) -> int:
        if page.token_count > 0:
            return page.token_count

        total = 0
        for memory_id in page.memory_ids:
            memory = self.archival.get(memory_id)
            if not memory:
                continue
            total += self._estimate_text_tokens(memory.content_l1 or memory.content_l3)

        page.token_count = max(1, total or self.tokens_per_l1)
        return page.token_count

    def find_page_id(self, memory_id: str) -> Optional[str]:
        for page_id, page in self.pages.items():
            if memory_id in page.memory_ids:
                return page_id
        return None

    def store(self, memory: MemoryUnit) -> str:
        """Store a memory in archival storage and return its page ID."""

        self.archival[memory.id] = memory
        existing_page_id = self.find_page_id(memory.id)
        if existing_page_id:
            page = self.pages[existing_page_id]
            page.summary = memory.content_l1 or memory.content_l3[:80]
            page.priority = memory.dopamine_weight
            page.token_count = 0
            self._estimate_page_tokens(page)
            return existing_page_id

        page = ContextPage(
            memory_ids=[memory.id],
            summary=memory.content_l1 or memory.content_l3[:80],
            priority=memory.dopamine_weight,
            is_pinned=False,
        )
        self._estimate_page_tokens(page)
        self.pages[page.page_id] = page
        return page.page_id

    def page_in(self, memory_id: str, reason: str = "") -> bool:
        """Load a memory into active context if it fits the token budget."""

        if memory_id not in self.archival:
            return False

        page_id = self.find_page_id(memory_id)
        if not page_id:
            return False

        if page_id in self.main_context:
            self.main_context.move_to_end(page_id)
            return True

        page = self.pages[page_id]
        self.warm_cache.pop(page_id, None)
        page_tokens = self._estimate_page_tokens(page)
        if page_tokens > self.max_tokens:
            return False

        while self.current_token_usage + page_tokens > self.max_tokens and self.main_context:
            self._evict_lru()

        if self.current_token_usage + page_tokens > self.max_tokens:
            return False

        self.main_context[page_id] = page
        self.current_token_usage += page_tokens
        page.last_paged_in = datetime.now()

        memory = self.archival[memory_id]
        memory.access_count += 1
        memory.last_accessed = datetime.now()

        self.paging_history.append(PagingEvent("page_in", page_id, reason=reason))
        return True

    def page_out(self, page_id: str, reason: str = "") -> bool:
        """Evict a page from active context into the warm cache."""

        if page_id not in self.main_context:
            return False

        page = self.main_context.pop(page_id)
        self.current_token_usage = max(0, self.current_token_usage - self._estimate_page_tokens(page))

        self.warm_cache[page_id] = page
        if len(self.warm_cache) > self.warm_cache_size:
            self.warm_cache.popitem(last=False)

        self.paging_history.append(PagingEvent("page_out", page_id, reason=reason))
        return True

    def _evict_lru(self) -> None:
        for page_id, page in list(self.main_context.items()):
            if not page.is_pinned:
                self.page_out(page_id, reason="lru_eviction")
                return
        raise RuntimeError("Cannot evict context pages because all active pages are pinned.")

    def remove(self, memory_id: str, reason: str = "") -> bool:
        """Remove a memory from archival, active context, warm cache, and pages."""

        removed = memory_id in self.archival
        page_ids = [page_id for page_id, page in self.pages.items() if memory_id in page.memory_ids]

        for page_id in page_ids:
            self.page_out(page_id, reason=reason)
            self.warm_cache.pop(page_id, None)
            self.pages.pop(page_id, None)
            removed = True

        self.archival.pop(memory_id, None)
        return removed

    def pin(self, memory_id: str) -> bool:
        page_id = self.find_page_id(memory_id)
        if not page_id:
            return False
        self.pages[page_id].is_pinned = True
        return True

    def unpin(self, memory_id: str) -> bool:
        page_id = self.find_page_id(memory_id)
        if not page_id:
            return False
        self.pages[page_id].is_pinned = False
        return True

    def get_active_context(self) -> List[MemoryUnit]:
        active_memories: List[MemoryUnit] = []
        for page in self.main_context.values():
            for memory_id in page.memory_ids:
                memory = self.archival.get(memory_id)
                if memory:
                    active_memories.append(memory)
        return active_memories

    def get_context_summary(self) -> str:
        summaries = []
        for memory in self.get_active_context():
            summaries.append(memory.content_l1 or memory.content_l3[:80])
        return " | ".join(summaries)

    def auto_page_in(self, query: str, memories: List[MemoryUnit], top_k: int = 5) -> None:
        scored = sorted(
            memories,
            key=lambda memory: memory.dopamine_weight * memory.decay_score,
            reverse=True,
        )
        for memory in scored[:top_k]:
            self.page_in(memory.id, reason=f"auto: query='{query[:30]}'")

    def get_stats(self) -> Dict:
        return {
            "context_tokens_used": self.current_token_usage,
            "context_tokens_max": self.max_tokens,
            "context_utilization": f"{self.current_token_usage / self.max_tokens * 100:.1f}%",
            "pages_in_ram": len(self.main_context),
            "pages_in_warm_cache": len(self.warm_cache),
            "total_archival": len(self.archival),
            "total_paging_events": len(self.paging_history),
            "pinned_pages": sum(1 for page in self.main_context.values() if page.is_pinned),
        }
