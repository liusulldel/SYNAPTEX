from datetime import datetime, timedelta

from synaptex.dopamine import DopamineEncoder
from synaptex.forgetting_gate import ForgettingGate
from synaptex.multi_agent import SharedMemoryPool
from synaptex.reasoning_bank import InteractionRecord, ReasoningBank
from synaptex.types import AgentIdentity, EmotionType, MemoryUnit


def test_memory_hash_and_token_savings_are_stable():
    memory = MemoryUnit(content_l3="The agent saved a useful note.", content_l1="agent saved note")

    assert memory.content_hash() == MemoryUnit(content_l3=memory.content_l3).content_hash()
    assert 0.0 <= memory.token_savings_ratio() <= 1.0


def test_dopamine_encoder_respects_importance_override():
    memory = MemoryUnit(content_l3="Important deadline accepted.")
    signal = DopamineEncoder().encode(memory, EmotionType.SURPRISE, user_importance=1.0)

    assert signal.final_weight > 0.7
    assert memory.dopamine_weight == signal.final_weight


def test_forgetting_gate_marks_old_low_value_memory():
    memory = MemoryUnit(
        content_l3="Stale note",
        dopamine_weight=0.0,
        timestamp=datetime.now() - timedelta(days=365),
    )
    result = ForgettingGate(forget_threshold=0.5).apply_gate(memory)

    assert result.should_forget is True


def test_shared_memory_public_and_private_access():
    pool = SharedMemoryPool()
    pool.register_agent(AgentIdentity(agent_id="planner"))
    pool.register_agent(AgentIdentity(agent_id="executor"))

    public = MemoryUnit(content_l3="Shared plan", tags=["plan"])
    private = MemoryUnit(content_l3="Private scratchpad", tags=["scratch"])

    assert pool.write("planner", public, scope="public") is True
    assert pool.write("planner", private, scope="private") is True

    assert public in pool.read("executor", tags=["plan"])
    assert private not in pool.read("executor", tags=["scratch"])
    assert private in pool.read("planner", tags=["scratch"])


def test_reasoning_bank_distill_and_match():
    bank = ReasoningBank(min_occurrences_to_distill=2)
    bank.record(
        InteractionRecord(
            query="debug retrieval",
            response="ok",
            reasoning_chain=["Check failing assertion", "Add regression test"],
            outcome="success",
            context_tags=["debug"],
        )
    )
    bank.record(
        InteractionRecord(
            query="debug pager",
            response="ok",
            reasoning_chain=["Check failing assertion", "Add regression test"],
            outcome="success",
            context_tags=["debug"],
        )
    )

    matches = bank.match(tags=["debug"])
    assert matches
    assert "add regression test" in matches[0].reasoning_steps
