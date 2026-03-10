"""
SYNAPTEX·触链典 — Dopamine-Weighted Memory Encoder

Inspired by MemoryBank (Google, ACL 2025):
Injects emotional significance scalar ε ∈ [0, 1] into each memory,
modeling the human dopaminergic system's role in memory consolidation.

High-impact events (breakthroughs, conflicts, emotional peaks) receive
elevated dopamine weights, resisting Ebbinghaus temporal decay.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from synaptex.types import MemoryUnit, EmotionType


# Emotion → base dopamine weight mapping (calibrated to cognitive science)
EMOTION_DOPAMINE_MAP: Dict[EmotionType, float] = {
    EmotionType.JOY:          0.80,
    EmotionType.ANGER:        0.75,
    EmotionType.SURPRISE:     0.85,
    EmotionType.FEAR:         0.70,
    EmotionType.SADNESS:      0.60,
    EmotionType.TRUST:        0.50,
    EmotionType.ANTICIPATION: 0.65,
    EmotionType.NEUTRAL:      0.30,
}

# Keywords that boost dopamine regardless of detected emotion
HIGH_IMPACT_KEYWORDS = {
    "breakthrough", "discovery", "eureka", "fired", "hired", "married",
    "divorced", "accident", "promotion", "rejection", "accepted",
    "published", "deadline", "emergency", "love", "death", "birth",
    "突破", "发现", "录取", "拒绝", "发表", "紧急", "升职", "离职",
    "争吵", "和解", "表白", "分手", "毕业", "获奖", "失败", "成功",
}


@dataclass
class DopamineSignal:
    """Computed dopamine signal for a memory encoding event."""
    base_weight: float       # from emotion type
    keyword_boost: float     # from high-impact keyword detection
    recency_boost: float     # recent events get slight boost
    user_override: float     # explicit user-supplied importance
    final_weight: float      # clamped composite ε ∈ [0, 1]


class DopamineEncoder:
    """
    Assigns dopamine weights to incoming memories.
    
    Pipeline:
    1. Detect emotion type (or accept user-supplied label)
    2. Scan for high-impact keywords → keyword boost
    3. Apply recency boost for very recent events
    4. Combine with user override if provided
    5. Clamp to [0, 1] → final ε
    
    Usage:
        encoder = DopamineEncoder()
        memory = MemoryUnit(content_l3="Got accepted to PhD program!")
        signal = encoder.encode(memory, emotion=EmotionType.JOY)
        # memory.dopamine_weight is now ~0.92
    """

    def __init__(
        self,
        keyword_boost_delta: float = 0.15,
        recency_window_hours: float = 24.0,
        recency_boost_delta: float = 0.05,
    ):
        self.keyword_boost_delta = keyword_boost_delta
        self.recency_window_hours = recency_window_hours
        self.recency_boost_delta = recency_boost_delta

    def encode(
        self,
        memory: MemoryUnit,
        emotion: Optional[EmotionType] = None,
        user_importance: Optional[float] = None,
    ) -> DopamineSignal:
        """
        Compute and assign dopamine weight to a MemoryUnit.
        
        Args:
            memory: The memory to encode
            emotion: Detected or user-supplied emotion type
            user_importance: Explicit override ∈ [0, 1], blended with computed weight
            
        Returns:
            DopamineSignal with breakdown of weight computation
        """
        # 1. Base weight from emotion
        emotion = emotion or memory.emotion
        memory.emotion = emotion
        base = EMOTION_DOPAMINE_MAP.get(emotion, 0.3)

        # 2. High-impact keyword scan
        text_lower = memory.content_l3.lower()
        keyword_hits = sum(1 for kw in HIGH_IMPACT_KEYWORDS if kw in text_lower)
        kw_boost = min(keyword_hits * self.keyword_boost_delta, 0.3)

        # 3. Recency boost
        age_hours = (datetime.now() - memory.timestamp).total_seconds() / 3600
        recency = self.recency_boost_delta if age_hours < self.recency_window_hours else 0.0

        # 4. User override blending
        user_val = user_importance if user_importance is not None else 0.0
        if user_importance is not None:
            # Blend: 60% computed + 40% user override
            composite = 0.6 * (base + kw_boost + recency) + 0.4 * user_val
        else:
            composite = base + kw_boost + recency

        # 5. Clamp to [0, 1]
        final = max(0.0, min(1.0, composite))
        memory.dopamine_weight = final

        return DopamineSignal(
            base_weight=base,
            keyword_boost=kw_boost,
            recency_boost=recency,
            user_override=user_val,
            final_weight=final,
        )

    def batch_encode(
        self,
        memories: List[MemoryUnit],
        emotions: Optional[List[EmotionType]] = None,
    ) -> List[DopamineSignal]:
        """Encode dopamine weights for a batch of memories."""
        emotions = emotions or [None] * len(memories)
        return [
            self.encode(mem, emo)
            for mem, emo in zip(memories, emotions)
        ]
