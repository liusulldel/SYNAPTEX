"""Heuristic tri-layer compression for SYNAPTEX memories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Dict, Optional, Tuple

from synaptex.types import MemoryUnit


@dataclass
class CompressionResult:
    """Result of compressing a raw event into memory layers."""

    original_text: str
    l1_text: str
    l2_text: str
    l1_lang: str
    original_tokens_est: int
    l1_tokens_est: int
    savings_ratio: float
    candidates_evaluated: Dict[str, Tuple[str, int]]


class PolyglotCompressor:
    """Small, inspectable compressor for L1 summaries and L2 timeline entries.

    The class name is retained for API compatibility. The current implementation
    is deterministic and local; it does not call external models or claim
    benchmarked semantic compression.
    """

    KEYWORD_REPLACEMENTS = {
        r"\bmeeting\b": "mtg",
        r"\bdiscussion\b": "discuss",
        r"\bdecision\b": "decide",
        r"\bdecided\b": "decided",
        r"\bretrieval\b": "retrieval",
        r"\bpersistence\b": "persist",
        r"\bdeadline\b": "deadline",
        r"\bexperiment\b": "experiment",
        r"\bresearch\b": "research",
        r"\bimportant\b": "important",
        r"\bpriority\b": "priority",
        r"\bprioritize\b": "prioritize",
        r"\bcompleted\b": "done",
        r"\bfinished\b": "done",
        r"\bfailed\b": "failed",
        r"\bsuccessful\b": "success",
    }

    FILLER_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "for",
        "from",
        "is",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }

    def __init__(self, enable_polyglot_routing: bool = True):
        self.polyglot_routing = enable_polyglot_routing

    def _estimate_tokens(self, text: str, lang: str = "en") -> int:
        """Return a lightweight token estimate without tokenizer dependency."""

        words = text.split()
        if words:
            multiplier = 1.15 if lang == "compact_en" else 1.3
            return max(1, int(len(words) * multiplier))
        return max(1, len(text) // 4)

    def _compress_english(self, text: str, max_terms: int = 18) -> str:
        """Create a compact English L1 summary using deterministic rules."""

        lowered = text.lower().strip()
        for pattern, replacement in self.KEYWORD_REPLACEMENTS.items():
            lowered = re.sub(pattern, replacement, lowered, flags=re.IGNORECASE)

        tokens = re.findall(r"[a-zA-Z0-9:_-]+", lowered)
        compact = [token for token in tokens if token not in self.FILLER_WORDS]
        if not compact:
            compact = tokens

        summary = " ".join(compact[:max_terms]).strip()
        return summary or text[:80].strip()

    def _generate_l2_entry(
        self,
        text: str,
        timestamp: datetime,
        category: str = "",
        emotion_marker: str = "",
    ) -> str:
        """Generate a structured timeline entry."""

        lines = [f"### {timestamp.strftime('%Y-%m-%d')} | {category or 'general'}"]
        lines.append(f"- event: {text}")
        if emotion_marker:
            lines.append(f"- importance: {emotion_marker}")
        return "\n".join(lines)

    def compress(
        self,
        text: str,
        timestamp: Optional[datetime] = None,
        category: str = "",
        emotion_weight: float = 0.5,
    ) -> CompressionResult:
        """Compress raw text into L1/L2 layers and report rough savings."""

        timestamp = timestamp or datetime.now()
        original_tokens = self._estimate_tokens(text, "en")

        l1_text = self._compress_english(text)
        l1_tokens = self._estimate_tokens(l1_text, "compact_en")
        candidates = {"compact_en": (l1_text, l1_tokens)}

        best_lang = "compact_en"
        best_text = l1_text
        best_tokens = l1_tokens

        if self.polyglot_routing:
            headline = " ".join(text.split()[:12])
            headline_tokens = self._estimate_tokens(headline, "en")
            candidates["headline"] = (headline, headline_tokens)
            if 0 < headline_tokens < best_tokens:
                best_lang = "headline"
                best_text = headline
                best_tokens = headline_tokens

        importance = f"{emotion_weight:.2f}" if emotion_weight > 0.3 else ""
        l2_text = self._generate_l2_entry(text, timestamp, category, importance)
        savings = 1.0 - (best_tokens / original_tokens) if original_tokens else 0.0

        return CompressionResult(
            original_text=text,
            l1_text=best_text,
            l2_text=l2_text,
            l1_lang=best_lang,
            original_tokens_est=original_tokens,
            l1_tokens_est=best_tokens,
            savings_ratio=max(0.0, min(1.0, savings)),
            candidates_evaluated=candidates,
        )

    def compress_to_memory(
        self,
        text: str,
        timestamp: Optional[datetime] = None,
        category: str = "",
        emotion_weight: float = 0.5,
    ) -> MemoryUnit:
        """Create a full MemoryUnit from raw text."""

        resolved_timestamp = timestamp or datetime.now()
        result = self.compress(text, resolved_timestamp, category, emotion_weight)

        return MemoryUnit(
            content_l3=text,
            content_l2=result.l2_text,
            content_l1=result.l1_text,
            dopamine_weight=emotion_weight,
            timestamp=resolved_timestamp,
            source_lang="en",
            compressed_lang=result.l1_lang,
        )
