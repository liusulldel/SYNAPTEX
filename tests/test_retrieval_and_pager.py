from datetime import datetime, timedelta

from synaptex.memgpt_pager import MemGPTPager
from synaptex.swiftmem import SwiftMemEngine
from synaptex.types import MemoryUnit


def test_swiftmem_does_not_mutate_caller_tags():
    engine = SwiftMemEngine()
    memory = MemoryUnit(content_l3="Retrieval test memory", tags=["retrieval"])
    engine.index(memory)

    tags = ["retrieval"]
    engine.query(text="retrieval", tags=tags)

    assert tags == ["retrieval"]


def test_swiftmem_time_and_tag_query():
    engine = SwiftMemEngine()
    old = MemoryUnit(
        content_l3="Old systems note",
        tags=["systems"],
        timestamp=datetime.now() - timedelta(days=2),
    )
    recent = MemoryUnit(content_l3="Recent systems note", tags=["systems"])
    engine.index(old)
    engine.index(recent)

    result = engine.query(
        tags=["systems"],
        time_start=datetime.now() - timedelta(hours=1),
        time_end=datetime.now() + timedelta(hours=1),
    )

    assert result.memories == [recent]
    assert result.index_used == "hybrid"


def test_pager_rejects_oversized_page():
    pager = MemGPTPager(max_context_tokens=5)
    memory = MemoryUnit(content_l3="x" * 100, content_l1="x" * 100)

    pager.store(memory)

    assert pager.page_in(memory.id) is False
    assert pager.current_token_usage == 0
    assert pager.get_active_context() == []


def test_pager_remove_clears_all_page_state():
    pager = MemGPTPager(max_context_tokens=1024)
    memory = MemoryUnit(content_l3="Memory to remove", content_l1="Memory to remove")
    pager.store(memory)
    assert pager.page_in(memory.id) is True

    assert pager.remove(memory.id, reason="test") is True

    assert memory.id not in pager.archival
    assert pager.get_active_context() == []
    assert all(memory.id not in page.memory_ids for page in pager.pages.values())
