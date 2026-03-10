"""
SYNAPTEX·触链典 — ReasoningBank Strategy Distillation

Inspired by ReasoningBank (Google Research, 2025):
Distills reasoning traces from both successful and failed interactions
into reusable, high-level strategies — enabling learning without retraining.

The agent doesn't just remember what happened; it remembers *how it reasoned*,
and can apply those patterns to novel situations.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from synaptex.types import ReasoningTrace


@dataclass
class InteractionRecord:
    """Raw record of an agent interaction to be distilled."""
    query: str
    response: str
    reasoning_chain: List[str]  # step-by-step reasoning
    outcome: str  # "success", "failure", "partial"
    context_tags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0


class ReasoningBank:
    """
    Strategy distillation engine.
    
    Pipeline:
    1. RECORD: Log interaction traces (query, reasoning, outcome)
    2. DISTILL: Extract reusable reasoning patterns from traces
    3. MATCH: For new queries, find applicable strategies
    4. APPLY: Inject matched strategies into agent's context
    5. EVOLVE: Update strategy success/failure counts
    
    Key insight from the paper:
    Failed interactions are just as valuable as successful ones —
    they teach the agent what NOT to do.
    
    Usage:
        bank = ReasoningBank()
        
        # Record a successful interaction
        bank.record(InteractionRecord(
            query="Prove theorem X",
            response="...",
            reasoning_chain=["Step 1: ...", "Step 2: ..."],
            outcome="success",
            context_tags=["proof", "algebra"],
        ))
        
        # Later, find applicable strategies
        strategies = bank.match("Prove theorem Y", tags=["proof"])
    """

    def __init__(self, min_occurrences_to_distill: int = 2):
        self.traces: Dict[str, ReasoningTrace] = {}
        self.raw_records: List[InteractionRecord] = []
        self.pattern_index: Dict[str, List[str]] = defaultdict(list)  # tag → trace_ids
        self.min_occurrences = min_occurrences_to_distill

    def record(self, interaction: InteractionRecord):
        """
        Record a raw interaction for later distillation.
        Automatically triggers distillation if enough similar records exist.
        """
        self.raw_records.append(interaction)
        
        # Check if we should auto-distill
        similar_count = sum(
            1 for r in self.raw_records
            if set(r.context_tags) & set(interaction.context_tags)
        )
        
        if similar_count >= self.min_occurrences:
            self._auto_distill(interaction.context_tags)

    def _auto_distill(self, trigger_tags: List[str]):
        """
        Auto-distill reasoning patterns from accumulated records.
        Finds common reasoning steps across similar interactions.
        """
        # Find records with overlapping tags
        relevant = [
            r for r in self.raw_records
            if set(r.context_tags) & set(trigger_tags)
        ]
        
        if len(relevant) < self.min_occurrences:
            return
        
        # Separate successes and failures
        successes = [r for r in relevant if r.outcome == "success"]
        failures = [r for r in relevant if r.outcome == "failure"]
        
        # Extract common steps from successful interactions
        if successes:
            common_steps = self._extract_common_steps(successes)
            if common_steps:
                trace = ReasoningTrace(
                    strategy_name=f"strategy_{'_'.join(trigger_tags[:2])}",
                    context_pattern=f"When dealing with: {', '.join(trigger_tags)}",
                    reasoning_steps=common_steps,
                    success_count=len(successes),
                    failure_count=0,
                    source_memories=[],
                )
                self.traces[trace.id] = trace
                for tag in trigger_tags:
                    self.pattern_index[tag].append(trace.id)
        
        # Extract anti-patterns from failures
        if failures:
            anti_steps = self._extract_common_steps(failures)
            if anti_steps:
                trace = ReasoningTrace(
                    strategy_name=f"anti_{'_'.join(trigger_tags[:2])}",
                    context_pattern=f"AVOID when dealing with: {', '.join(trigger_tags)}",
                    reasoning_steps=[f"⚠️ AVOID: {s}" for s in anti_steps],
                    success_count=0,
                    failure_count=len(failures),
                    source_memories=[],
                )
                self.traces[trace.id] = trace
                for tag in trigger_tags:
                    self.pattern_index[tag].append(trace.id)

    def _extract_common_steps(
        self, records: List[InteractionRecord]
    ) -> List[str]:
        """
        Extract reasoning steps that appear across multiple records.
        Uses simple keyword overlap as a proxy for step similarity.
        """
        if not records:
            return []
        
        # Count step frequency
        step_freq: Dict[str, int] = defaultdict(int)
        for record in records:
            for step in record.reasoning_chain:
                # Normalize step for comparison
                normalized = step.strip().lower()
                step_freq[normalized] += 1
        
        # Keep steps that appear in at least half the records
        threshold = max(1, len(records) // 2)
        common = [
            step for step, count in step_freq.items()
            if count >= threshold
        ]
        
        return common[:10]  # Cap at 10 steps

    def distill(
        self,
        name: str,
        context_pattern: str,
        steps: List[str],
        tags: List[str],
    ) -> ReasoningTrace:
        """
        Manually distill a reasoning strategy.
        
        Args:
            name: Human-readable strategy name
            context_pattern: When to apply this strategy
            steps: Ordered reasoning steps
            tags: Context tags for matching
            
        Returns:
            Created ReasoningTrace
        """
        trace = ReasoningTrace(
            strategy_name=name,
            context_pattern=context_pattern,
            reasoning_steps=steps,
        )
        
        self.traces[trace.id] = trace
        for tag in tags:
            self.pattern_index[tag].append(trace.id)
        
        return trace

    def match(
        self,
        query: str = "",
        tags: Optional[List[str]] = None,
        top_k: int = 3,
        min_success_rate: float = 0.0,
    ) -> List[ReasoningTrace]:
        """
        Find applicable reasoning strategies for a query/context.
        
        Args:
            query: Free-text query
            tags: Context tags to match against
            top_k: Maximum strategies to return
            min_success_rate: Filter by minimum success rate
            
        Returns:
            List of matching ReasoningTraces, sorted by success rate
        """
        candidates: Dict[str, float] = {}  # trace_id → relevance score
        
        # Match by tags
        match_tags = tags or []
        if query:
            # Extract potential tags from query
            words = set(query.lower().split())
            for tag in self.pattern_index:
                if tag.lower() in words:
                    match_tags.append(tag)
        
        for tag in match_tags:
            for trace_id in self.pattern_index.get(tag, []):
                candidates[trace_id] = candidates.get(trace_id, 0) + 1.0
        
        # Filter and sort
        results = []
        for trace_id, score in sorted(
            candidates.items(), key=lambda x: x[1], reverse=True
        ):
            trace = self.traces.get(trace_id)
            if trace and trace.success_rate >= min_success_rate:
                results.append(trace)
        
        return results[:top_k]

    def report_outcome(self, trace_id: str, success: bool):
        """
        Report the outcome of applying a strategy.
        Updates success/failure counts for learning.
        """
        if trace_id in self.traces:
            if success:
                self.traces[trace_id].success_count += 1
            else:
                self.traces[trace_id].failure_count += 1

    def get_all_strategies(self) -> List[ReasoningTrace]:
        """Return all distilled strategies, sorted by success rate."""
        return sorted(
            self.traces.values(),
            key=lambda t: t.success_rate,
            reverse=True,
        )

    def format_for_injection(self, traces: List[ReasoningTrace]) -> str:
        """
        Format matched strategies for context window injection.
        Compact format optimized for token efficiency.
        """
        lines = ["[SYNAPTEX ReasoningBank — Applicable Strategies]"]
        
        for i, trace in enumerate(traces, 1):
            lines.append(f"\n策略{i}: {trace.strategy_name}")
            lines.append(f"  适用场景: {trace.context_pattern}")
            lines.append(f"  成功率: {trace.success_rate:.0%}")
            lines.append(f"  步骤:")
            for j, step in enumerate(trace.reasoning_steps, 1):
                lines.append(f"    {j}. {step}")
        
        return "\n".join(lines)

    def get_stats(self) -> Dict:
        """Return ReasoningBank statistics."""
        return {
            "total_strategies": len(self.traces),
            "total_raw_records": len(self.raw_records),
            "total_tags_indexed": len(self.pattern_index),
            "avg_success_rate": (
                sum(t.success_rate for t in self.traces.values())
                / max(1, len(self.traces))
            ),
            "successful_records": sum(
                1 for r in self.raw_records if r.outcome == "success"
            ),
            "failed_records": sum(
                1 for r in self.raw_records if r.outcome == "failure"
            ),
        }
