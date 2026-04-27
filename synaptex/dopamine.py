"""Importance weighting for SYNAPTEX memories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from synaptex.types import EmotionType, MemoryUnit


EMOTION_DOPAMINE_MAP: Dict[EmotionType, float] = {
    EmotionType.JOY: 0.80,
    EmotionType.ANGER: 0.75,
    EmotionType.SURPRISE: 0.85,
    EmotionType.FEAR: 0.70,
    EmotionType.SADNESS: 0.60,
    EmotionType.TRUST: 0.50,
    EmotionType.ANTICIPATION: 0.65,
    EmotionType.NEUTRAL: 0.30,
}
IMPORTANCE_LABEL_WEIGHT_MAP = EMOTION_DOPAMINE_MAP

HIGH_IMPACT_KEYWORDS = {
    "accepted",
    "accident",
    "birth",
    "breakthrough",
    "deadline",
    "death",
    "discovery",
    "emergency",
    "fired",
    "hired",
    "promotion",
    "published",
    "rejected",
    "resolved",
    "risk",
    "urgent",
}


@dataclass
class DopamineSignal:
    """Breakdown of an importance-weight computation."""

    base_weight: float
    keyword_boost: float
    recency_boost: float
    user_override: float
    final_weight: float


class DopamineEncoder:
    """Assign an importance scalar to each memory.

    The public field is still named ``dopamine_weight`` for compatibility, but
    callers should treat it as a generic importance score in ``[0, 1]``.
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
        """Compute and assign an importance score to ``memory``."""

        emotion = emotion or memory.emotion
        memory.emotion = emotion
        base = EMOTION_DOPAMINE_MAP.get(emotion, 0.3)

        text_lower = memory.content_l3.lower()
        keyword_hits = sum(1 for keyword in HIGH_IMPACT_KEYWORDS if keyword in text_lower)
        keyword_boost = min(keyword_hits * self.keyword_boost_delta, 0.3)

        age_hours = (datetime.now() - memory.timestamp).total_seconds() / 3600
        recency_boost = self.recency_boost_delta if age_hours < self.recency_window_hours else 0.0

        user_value = user_importance if user_importance is not None else 0.0
        if user_importance is not None:
            composite = 0.6 * (base + keyword_boost + recency_boost) + 0.4 * user_value
        else:
            composite = base + keyword_boost + recency_boost

        final = max(0.0, min(1.0, composite))
        memory.dopamine_weight = final

        return DopamineSignal(
            base_weight=base,
            keyword_boost=keyword_boost,
            recency_boost=recency_boost,
            user_override=user_value,
            final_weight=final,
        )

    def batch_encode(
        self,
        memories: List[MemoryUnit],
        emotions: Optional[List[EmotionType]] = None,
    ) -> List[DopamineSignal]:
        """Encode importance weights for a batch of memories."""

        resolved_emotions = emotions or [None] * len(memories)
        return [self.encode(memory, emotion) for memory, emotion in zip(memories, resolved_emotions)]


class ImportanceEncoder(DopamineEncoder):
    """Neutral public alias for the legacy importance encoder name."""


ImportanceSignal = DopamineSignal
