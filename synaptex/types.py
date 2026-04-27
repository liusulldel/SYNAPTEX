"""Shared type definitions for SYNAPTEX."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
from typing import Any, Dict, List, Optional, Set
import uuid


class MemoryLayer(Enum):
    """Tri-layer memory levels.

    The legacy enum names are kept as aliases for compatibility.
    """

    L1_SUMMARY = "L1"
    L2_TIMELINE = "L2"
    L3_RAW = "L3"

    L1_SHIJI = "L1"
    L2_CHRONICLE = "L2"


class EmotionType(Enum):
    """Optional labels used by the importance-weighting heuristic."""

    JOY = "joy"
    ANGER = "anger"
    SADNESS = "sadness"
    FEAR = "fear"
    SURPRISE = "surprise"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"


ImportanceLabel = EmotionType


class MemoryStatus(Enum):
    """Lifecycle status of a memory unit."""

    ACTIVE = "active"
    DECAYING = "decaying"
    CONSOLIDATED = "consolidated"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"


@dataclass
class ModalityAnchor:
    """Reference to a non-text attachment."""

    modality: str
    path: str
    description: str = ""
    embedding: Optional[List[float]] = None
    timestamp: Optional[datetime] = None


@dataclass
class MemoryUnit:
    """Inspectable memory object used by all SYNAPTEX subsystems."""

    content_l3: str
    content_l2: str = ""
    content_l1: str = ""
    dopamine_weight: float = 0.5
    emotion: EmotionType = EmotionType.NEUTRAL
    timestamp: datetime = field(default_factory=datetime.now)
    decay_score: float = 1.0
    tags: List[str] = field(default_factory=list)
    links: Set[str] = field(default_factory=set)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    source_lang: str = "en"
    compressed_lang: str = "compact_en"
    status: MemoryStatus = MemoryStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    modality_anchors: List[ModalityAnchor] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    def content_hash(self) -> str:
        """Return a stable content fingerprint for deduplication."""

        return hashlib.sha256(self.content_l3.encode("utf-8")).hexdigest()[:16]

    @property
    def importance_weight(self) -> float:
        """Compatibility-safe neutral alias for ``dopamine_weight``."""

        return self.dopamine_weight

    @importance_weight.setter
    def importance_weight(self, value: float) -> None:
        self.dopamine_weight = value

    def token_savings_ratio(self) -> float:
        """Estimate savings from the raw L3 text to the compact L1 summary."""

        if not self.content_l1 or not self.content_l3:
            return 0.0

        l3_tokens = max(1, len(self.content_l3.split()))
        l1_words = self.content_l1.split()
        l1_tokens = len(l1_words) if l1_words else max(1, len(self.content_l1) // 4)
        ratio = 1.0 - (l1_tokens / l3_tokens)
        return max(0.0, min(1.0, ratio))


@dataclass
class ReasoningTrace:
    """Reusable reasoning pattern distilled from prior interactions."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    strategy_name: str = ""
    context_pattern: str = ""
    reasoning_steps: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    source_memories: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


@dataclass
class AgentIdentity:
    """Identity and permissions for a shared-memory agent."""

    agent_id: str
    agent_name: str = ""
    role: str = ""
    permissions: Set[str] = field(default_factory=lambda: {"read", "write"})
    private_memories: Set[str] = field(default_factory=set)


@dataclass
class ContextPage:
    """A page that can be loaded into or evicted from active context."""

    page_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    memory_ids: List[str] = field(default_factory=list)
    summary: str = ""
    token_count: int = 0
    priority: float = 0.0
    is_pinned: bool = False
    last_paged_in: Optional[datetime] = None
