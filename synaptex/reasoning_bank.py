"""Reusable reasoning traces for SYNAPTEX agent workflows."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from synaptex.types import ReasoningTrace


@dataclass
class InteractionRecord:
    """Raw interaction record that can be distilled into a strategy."""

    query: str
    response: str
    reasoning_chain: List[str]
    outcome: str
    context_tags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0


class ReasoningBank:
    """Store and retrieve compact reasoning strategies and anti-patterns."""

    def __init__(self, min_occurrences_to_distill: int = 2):
        self.traces: Dict[str, ReasoningTrace] = {}
        self.raw_records: List[InteractionRecord] = []
        self.pattern_index: Dict[str, List[str]] = defaultdict(list)
        self.min_occurrences = min_occurrences_to_distill

    def record(self, interaction: InteractionRecord) -> None:
        self.raw_records.append(interaction)
        similar_count = sum(
            1
            for record in self.raw_records
            if set(record.context_tags) & set(interaction.context_tags)
        )
        if similar_count >= self.min_occurrences:
            self._auto_distill(interaction.context_tags)

    def _auto_distill(self, trigger_tags: List[str]) -> None:
        relevant = [
            record
            for record in self.raw_records
            if set(record.context_tags) & set(trigger_tags)
        ]
        if len(relevant) < self.min_occurrences:
            return

        successes = [record for record in relevant if record.outcome == "success"]
        failures = [record for record in relevant if record.outcome == "failure"]

        if successes:
            common_steps = self._extract_common_steps(successes)
            if common_steps:
                trace = ReasoningTrace(
                    strategy_name=f"strategy_{'_'.join(trigger_tags[:2])}",
                    context_pattern=f"When dealing with: {', '.join(trigger_tags)}",
                    reasoning_steps=common_steps,
                    success_count=len(successes),
                )
                self._store_trace(trace, trigger_tags)

        if failures:
            common_steps = self._extract_common_steps(failures)
            if common_steps:
                trace = ReasoningTrace(
                    strategy_name=f"anti_{'_'.join(trigger_tags[:2])}",
                    context_pattern=f"Avoid when dealing with: {', '.join(trigger_tags)}",
                    reasoning_steps=[f"AVOID: {step}" for step in common_steps],
                    failure_count=len(failures),
                )
                self._store_trace(trace, trigger_tags)

    def _store_trace(self, trace: ReasoningTrace, tags: List[str]) -> None:
        self.traces[trace.id] = trace
        for tag in tags:
            if trace.id not in self.pattern_index[tag]:
                self.pattern_index[tag].append(trace.id)

    def _extract_common_steps(self, records: List[InteractionRecord]) -> List[str]:
        if not records:
            return []

        step_freq: Dict[str, int] = defaultdict(int)
        for record in records:
            for step in record.reasoning_chain:
                normalized = step.strip().lower()
                if normalized:
                    step_freq[normalized] += 1

        threshold = max(1, len(records) // 2)
        return [step for step, count in step_freq.items() if count >= threshold][:10]

    def distill(
        self,
        name: str,
        context_pattern: str,
        steps: List[str],
        tags: List[str],
    ) -> ReasoningTrace:
        trace = ReasoningTrace(
            strategy_name=name,
            context_pattern=context_pattern,
            reasoning_steps=steps,
        )
        self._store_trace(trace, tags)
        return trace

    def match(
        self,
        query: str = "",
        tags: Optional[List[str]] = None,
        top_k: int = 3,
        min_success_rate: float = 0.0,
    ) -> List[ReasoningTrace]:
        candidates: Dict[str, float] = {}
        match_tags = list(tags or [])

        if query:
            words = {word.strip(".,:;!?()[]{}").lower() for word in query.split()}
            for tag in self.pattern_index:
                if tag.lower() in words and tag not in match_tags:
                    match_tags.append(tag)

        for tag in match_tags:
            for trace_id in self.pattern_index.get(tag, []):
                candidates[trace_id] = candidates.get(trace_id, 0.0) + 1.0

        results = []
        for trace_id, _score in sorted(candidates.items(), key=lambda item: item[1], reverse=True):
            trace = self.traces.get(trace_id)
            if trace and trace.success_rate >= min_success_rate:
                results.append(trace)

        return results[:top_k]

    def report_outcome(self, trace_id: str, success: bool) -> None:
        trace = self.traces.get(trace_id)
        if not trace:
            return
        if success:
            trace.success_count += 1
        else:
            trace.failure_count += 1

    def get_all_strategies(self) -> List[ReasoningTrace]:
        return sorted(self.traces.values(), key=lambda trace: trace.success_rate, reverse=True)

    def format_for_injection(self, traces: List[ReasoningTrace]) -> str:
        lines = ["[SYNAPTEX ReasoningBank: Applicable Strategies]"]
        for index, trace in enumerate(traces, 1):
            lines.append(f"\nStrategy {index}: {trace.strategy_name}")
            lines.append(f"  Context: {trace.context_pattern}")
            lines.append(f"  Success rate: {trace.success_rate:.0%}")
            lines.append("  Steps:")
            for step_index, step in enumerate(trace.reasoning_steps, 1):
                lines.append(f"    {step_index}. {step}")
        return "\n".join(lines)

    def get_stats(self) -> Dict:
        return {
            "total_strategies": len(self.traces),
            "total_raw_records": len(self.raw_records),
            "total_tags_indexed": len(self.pattern_index),
            "avg_success_rate": (
                sum(trace.success_rate for trace in self.traces.values()) / max(1, len(self.traces))
            ),
            "successful_records": sum(
                1 for record in self.raw_records if record.outcome == "success"
            ),
            "failed_records": sum(1 for record in self.raw_records if record.outcome == "failure"),
        }
