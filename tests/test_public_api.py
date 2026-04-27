from synaptex import (
    AgentIdentity,
    EmotionType,
    ImportanceEncoder,
    ImportanceLabel,
    MemoryStatus,
    MemoryUnit,
    SharedMemoryPool,
    SynaptexEngine,
)


def test_public_api_exports_core_objects():
    assert SynaptexEngine
    assert MemoryUnit(content_l3="hello").status is MemoryStatus.ACTIVE
    assert EmotionType.TRUST.value == "trust"
    assert ImportanceLabel.TRUST.value == "trust"
    assert ImportanceEncoder
    memory = MemoryUnit(content_l3="hello", dopamine_weight=0.2)
    memory.importance_weight = 0.7
    assert memory.dopamine_weight == 0.7
    assert memory.importance_weight == 0.7
    assert AgentIdentity(agent_id="planner").permissions == {"read", "write"}
    assert SharedMemoryPool
