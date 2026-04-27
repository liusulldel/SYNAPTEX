"""Public package interface for SYNAPTEX."""

__version__ = "0.1.0"
__author__ = "Liu S / Cosmos Open Source Collective"

from synaptex.core import SynaptexEngine
from synaptex.dopamine import ImportanceEncoder, ImportanceSignal
from synaptex.multi_agent import SharedMemoryPool
from synaptex.types import AgentIdentity, EmotionType, ImportanceLabel, MemoryStatus, MemoryUnit

__all__ = [
    "AgentIdentity",
    "EmotionType",
    "ImportanceEncoder",
    "ImportanceLabel",
    "ImportanceSignal",
    "MemoryStatus",
    "MemoryUnit",
    "SharedMemoryPool",
    "SynaptexEngine",
]
