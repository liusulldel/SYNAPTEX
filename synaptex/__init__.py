"""Public package interface for SYNAPTEX."""

__version__ = "0.1.0"
__author__ = "Liu S / Cosmos Open Source Collective"

from synaptex.core import SynaptexEngine
from synaptex.types import EmotionType, MemoryUnit

__all__ = ["EmotionType", "MemoryUnit", "SynaptexEngine"]
