"""
SYNAPTEX·触链典 — Shared Type Definitions

Core data structures used across all modules.
Designed for interoperability between the memory layers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
import uuid
import hashlib


class MemoryLayer(Enum):
    """Tri-layer memory pyramid levels."""
    L1_SHIJI = "L1"       # 史记摘要层 — Classical Chinese, max compression
    L2_CHRONICLE = "L2"   # 编年史骨架 — Semi-classical + Markdown timeline
    L3_RAW = "L3"         # 潜意识细节池 — Verbatim cold storage


class EmotionType(Enum):
    """Categorical emotion labels for dopamine weighting."""
    JOY = "joy"
    ANGER = "anger"
    SADNESS = "sadness"
    FEAR = "fear"
    SURPRISE = "surprise"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"


class MemoryStatus(Enum):
    """Lifecycle status of a memory unit."""
    ACTIVE = "active"
    DECAYING = "decaying"
    CONSOLIDATED = "consolidated"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"


@dataclass
class ModalityAnchor:
    """Reference to a non-text memory attachment."""
    modality: str  # "image", "audio", "video", "embedding"
    path: str
    description: str = ""
    embedding: Optional[List[float]] = None
    timestamp: Optional[datetime] = None


@dataclass
class MemoryUnit:
    """
    Fundamental unit of memory in SYNAPTEX.
    
    Each unit exists across all three layers simultaneously:
    - L1: Compressed Classical Chinese abstract
    - L2: Structured timeline entry
    - L3: Raw verbatim content
    
    Attributes:
        id: Unique identifier (UUID-based)
        content_l1: 文言文 compressed form (~10-15 tokens)
        content_l2: Semi-structured timeline form
        content_l3: Raw verbatim content
        dopamine_weight: Emotional significance ε ∈ [0, 1]
        emotion: Categorical emotion label
        timestamp: When the memory was created
        decay_score: Current Ebbinghaus decay value R(t)
        tags: Auto-generated keyword tags (for A-MEM graph)
        links: Set of linked memory IDs (Zettelkasten connections)
        access_count: Number of times this memory was retrieved
        last_accessed: Timestamp of last retrieval
        source_lang: Original language of the content
        compressed_lang: Language chosen by polyglot router for L1
        status: Lifecycle status
        metadata: Arbitrary metadata dict
        modality_anchors: Paths/refs to multimodal attachments
    """
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
    compressed_lang: str = "classical_chinese"
    status: MemoryStatus = MemoryStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    modality_anchors: List[ModalityAnchor] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    def content_hash(self) -> str:
        """SHA-256 fingerprint of L3 content for deduplication."""
        return hashlib.sha256(self.content_l3.encode("utf-8")).hexdigest()[:16]

    def token_savings_ratio(self) -> float:
        """Estimate token savings from L3 → L1 compression."""
        if not self.content_l1 or not self.content_l3:
            return 0.0
        # Rough token estimation: 1 CJK char ≈ 1 token, 1 EN word ≈ 1.3 tokens
        l3_tokens = len(self.content_l3.split()) * 1.3
        l1_tokens = len(self.content_l1)  # CJK chars ≈ tokens
        if l3_tokens == 0:
            return 0.0
        return 1.0 - (l1_tokens / l3_tokens)


@dataclass
class ModalityAnchor:
    """Reference to a non-text memory attachment."""
    modality: str  # "image", "audio", "video", "embedding"
    path: str
    description: str = ""
    embedding: Optional[List[float]] = None
    timestamp: Optional[datetime] = None


@dataclass
class ReasoningTrace:
    """
    A distilled reasoning strategy from ReasoningBank.
    
    Captures successful or failed reasoning patterns that can
    be reused across contexts without re-discovering them.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    strategy_name: str = ""
    context_pattern: str = ""  # When to apply this strategy
    reasoning_steps: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    source_memories: List[str] = field(default_factory=list)  # memory IDs
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


@dataclass
class AgentIdentity:
    """Identity of an agent in multi-agent shared memory."""
    agent_id: str
    agent_name: str = ""
    role: str = ""
    permissions: Set[str] = field(default_factory=lambda: {"read", "write"})
    private_memories: Set[str] = field(default_factory=set)


@dataclass 
class ContextPage:
    """
    A page unit for MemGPT-style virtual context management.
    
    Represents a chunk of memory that can be paged in/out
    of the active context window.
    """
    page_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    memory_ids: List[str] = field(default_factory=list)
    summary: str = ""
    token_count: int = 0
    priority: float = 0.0  # Higher = more likely to stay in context
    is_pinned: bool = False  # Pinned pages are never evicted
    last_paged_in: Optional[datetime] = None
