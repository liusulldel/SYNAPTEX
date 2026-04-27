"""Decay and consolidation heuristics for SYNAPTEX memories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import List, Optional

from synaptex.types import MemoryStatus, MemoryUnit


@dataclass
class DecayResult:
    """Result of applying decay to one memory."""

    memory_id: str
    previous_score: float
    new_score: float
    should_forget: bool
    should_consolidate: bool


class ForgettingGate:
    """Data-dependent memory decay with access and importance reinforcement."""

    def __init__(
        self,
        base_half_life_hours: float = 168.0,
        dopamine_resistance_factor: float = 3.0,
        access_reinforcement: float = 0.05,
        forget_threshold: float = 0.05,
        consolidation_threshold: float = 0.90,
    ):
        self.base_half_life = base_half_life_hours
        self.dopamine_factor = dopamine_resistance_factor
        self.access_reinforcement = access_reinforcement
        self.forget_threshold = forget_threshold
        self.consolidation_threshold = consolidation_threshold

    def _effective_half_life(self, memory: MemoryUnit) -> float:
        importance_extension = 1.0 + self.dopamine_factor * memory.dopamine_weight
        access_extension = 1.0 + self.access_reinforcement * memory.access_count
        return self.base_half_life * importance_extension * access_extension

    def compute_decay(
        self,
        memory: MemoryUnit,
        current_time: Optional[datetime] = None,
    ) -> float:
        """Compute a retention score in ``[0, 1]``."""

        now = current_time or datetime.now()
        age_hours = (now - memory.timestamp).total_seconds() / 3600
        if age_hours <= 0:
            return 1.0

        half_life = self._effective_half_life(memory)
        decay = math.exp(-age_hours * math.log(2) / half_life)
        return max(0.0, min(1.0, decay))

    def apply_gate(
        self,
        memory: MemoryUnit,
        context_relevance: float = 1.0,
        current_time: Optional[datetime] = None,
    ) -> DecayResult:
        """Apply decay and update memory lifecycle status."""

        previous = memory.decay_score
        relevance = max(0.0, min(1.0, context_relevance))
        gated_decay = self.compute_decay(memory, current_time) * (0.3 + 0.7 * relevance)
        memory.decay_score = gated_decay

        should_forget = gated_decay < self.forget_threshold
        should_consolidate = (
            gated_decay > self.consolidation_threshold
            and memory.access_count >= 3
            and memory.dopamine_weight > 0.6
        )

        if should_forget:
            memory.status = MemoryStatus.FORGOTTEN
        elif should_consolidate:
            memory.status = MemoryStatus.CONSOLIDATED
        elif gated_decay < 0.3:
            memory.status = MemoryStatus.DECAYING

        return DecayResult(
            memory_id=memory.id,
            previous_score=previous,
            new_score=gated_decay,
            should_forget=should_forget,
            should_consolidate=should_consolidate,
        )

    def batch_gate(
        self,
        memories: List[MemoryUnit],
        context_relevance: float = 1.0,
    ) -> List[DecayResult]:
        """Apply decay to all memories."""

        return [self.apply_gate(memory, context_relevance) for memory in memories]

    def prune_forgotten(self, memories: List[MemoryUnit]) -> List[MemoryUnit]:
        """Return memories that are not marked forgotten."""

        return [memory for memory in memories if memory.status != MemoryStatus.FORGOTTEN]

    def night_consolidation(self, memories: List[MemoryUnit]) -> dict:
        """Run a batch consolidation/decay pass."""

        stats = {"total": len(memories), "consolidated": 0, "pruned": 0, "decaying": 0}
        for memory in memories:
            result = self.apply_gate(memory, context_relevance=0.5)
            if result.should_consolidate:
                stats["consolidated"] += 1
            elif result.should_forget:
                stats["pruned"] += 1
            elif memory.status == MemoryStatus.DECAYING:
                stats["decaying"] += 1
        return stats
