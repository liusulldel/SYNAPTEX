"""
SYNAPTEX·触链典 — Forgetting Transformer Gate (FoX)

Inspired by "Forgetting Transformer: Softmax Attention with a Forget Gate"
(Lin, Nikishin, He, Courville — 2025).

Implements data-dependent forgetting for memory units: instead of uniform
temporal decay, each memory's relevance is re-evaluated contextually.
Memories that are no longer contextually relevant decay faster,
while emotionally charged or frequently accessed memories resist decay.

Core formula:
    R(t) = e^(-t/S) × (1 + ε × dopamine_weight) × forget_gate(context)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import math

from synaptex.types import MemoryUnit, MemoryStatus


@dataclass
class DecayResult:
    """Result of applying forgetting gate to a memory."""
    memory_id: str
    previous_score: float
    new_score: float
    should_forget: bool
    should_consolidate: bool


class ForgettingGate:
    """
    Data-dependent forgetting gate for SYNAPTEX memories.
    
    Models three decay forces:
    1. Ebbinghaus temporal decay: R(t) = e^(-t/S)
    2. Dopamine resistance: high-ε memories resist forgetting
    3. Access reinforcement: frequently retrieved memories strengthen
    
    And one contextual gate:
    4. Relevance gate: if memory is contextually irrelevant, accelerate decay
    
    Parameters:
        base_half_life_hours: Time for a neutral memory to decay to 50%
        dopamine_resistance_factor: How much ε extends half-life
        access_reinforcement: Decay reduction per access
        forget_threshold: Below this score, memory is marked forgotten
        consolidation_threshold: Above this, memory is consolidated (permanent)
    """

    def __init__(
        self,
        base_half_life_hours: float = 168.0,  # 1 week
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
        """
        Compute effective half-life adjusted by dopamine weight and access count.
        
        High-emotion memories live much longer:
            S_eff = S_base × (1 + factor × ε) × (1 + 0.05 × access_count)
        """
        dopamine_extension = 1.0 + self.dopamine_factor * memory.dopamine_weight
        access_extension = 1.0 + self.access_reinforcement * memory.access_count
        return self.base_half_life * dopamine_extension * access_extension

    def compute_decay(
        self,
        memory: MemoryUnit,
        current_time: Optional[datetime] = None,
    ) -> float:
        """
        Compute current decay score R(t) for a memory.
        
        Returns:
            float ∈ [0, 1] where 1 = perfectly retained, 0 = forgotten
        """
        now = current_time or datetime.now()
        age_hours = (now - memory.timestamp).total_seconds() / 3600
        
        if age_hours <= 0:
            return 1.0

        half_life = self._effective_half_life(memory)
        
        # Ebbinghaus decay: R(t) = e^(-t × ln(2) / S)
        decay = math.exp(-age_hours * math.log(2) / half_life)
        
        return max(0.0, min(1.0, decay))

    def apply_gate(
        self,
        memory: MemoryUnit,
        context_relevance: float = 1.0,
        current_time: Optional[datetime] = None,
    ) -> DecayResult:
        """
        Apply the full forgetting gate pipeline to a memory.
        
        Args:
            memory: Memory unit to evaluate
            context_relevance: ∈ [0, 1], how relevant this memory is 
                              to the current context. Low values accelerate decay.
            current_time: Override for testing
            
        Returns:
            DecayResult with updated scores and lifecycle transitions
        """
        previous = memory.decay_score
        
        # Base temporal decay
        base_decay = self.compute_decay(memory, current_time)
        
        # Context-dependent gate modulation
        # If context_relevance is low, decay is amplified
        gated_decay = base_decay * (0.3 + 0.7 * context_relevance)
        
        # Update memory
        memory.decay_score = gated_decay
        
        # Lifecycle transitions
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
        """Apply forgetting gate to all memories, returning decay results."""
        return [self.apply_gate(m, context_relevance) for m in memories]

    def prune_forgotten(self, memories: List[MemoryUnit]) -> List[MemoryUnit]:
        """Remove forgotten memories from a list. Returns pruned list."""
        return [m for m in memories if m.status != MemoryStatus.FORGOTTEN]

    def night_consolidation(self, memories: List[MemoryUnit]) -> dict:
        """
        Simulate overnight memory consolidation (batch processing).
        
        During 'sleep', the system:
        1. Applies decay to all memories
        2. Consolidates high-value memories
        3. Merges similar decaying memories
        4. Prunes forgotten memories
        
        Returns summary statistics.
        """
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
