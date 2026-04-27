from datetime import datetime, timedelta

from synaptex import AgentIdentity, EmotionType, SynaptexEngine


def test_engine_quickstart_roundtrip():
    engine = SynaptexEngine(max_context_tokens=1024)

    memory = engine.encode(
        "The planner decided to prioritize retrieval tests before persistence.",
        emotion=EmotionType.TRUST,
        importance=0.85,
        category="engineering",
    )
    results = engine.recall(tags=["engineering"], limit=3)

    assert memory in results
    assert "planner" in results[0].content_l3.lower()
    assert engine.get_context()
    assert engine.get_stats()["engine"]["total_encoded"] == 1


def test_forget_removes_memory_from_retrieval_and_context():
    engine = SynaptexEngine(max_context_tokens=1024)
    memory = engine.encode("Forgettable incident for regression coverage.", category="qa")
    engine.recall(tags=["qa"])

    assert memory in engine.get_active_memories()
    assert engine.forget(memory.id) is True

    assert memory.id not in engine.retriever.memories
    assert memory.id not in engine.pager.archival
    assert memory not in engine.get_active_memories()
    assert engine.recall(tags=["qa"], auto_page_in=False) == []


def test_duplicate_encode_returns_canonical_memory_once():
    engine = SynaptexEngine(max_context_tokens=1024)

    first = engine.encode("Duplicate memory should have one canonical record.", category="qa")
    second = engine.encode("Duplicate memory should have one canonical record.", category="qa")

    assert second.id == first.id
    assert len(engine.graph.nodes) == 1
    assert len(engine.retriever.memories) == 1
    assert len(engine.pager.archival) == 1


def test_time_window_recall():
    engine = SynaptexEngine(max_context_tokens=1024)
    older = datetime.now() - timedelta(days=2)
    recent = datetime.now()

    engine.encode("Old planning note.", category="planning", timestamp=older)
    new_memory = engine.encode("Recent planning note.", category="planning", timestamp=recent)

    results = engine.recall(
        tags=["planning"],
        time_start=recent - timedelta(minutes=1),
        time_end=recent + timedelta(minutes=1),
        auto_page_in=False,
    )

    assert results == [new_memory]


def test_agent_registration_enables_shared_memory_write():
    engine = SynaptexEngine()
    engine.register_agent(AgentIdentity(agent_id="critic", role="review"))
    memory = engine.encode("Critic saw a risky assumption.", category="review", agent_id="critic")

    shared = engine.shared_memory.read("critic", tags=["review"])
    assert shared == [memory]
