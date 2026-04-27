"""Runnable SYNAPTEX quickstart for agentic memory workflows."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synaptex import AgentIdentity, ImportanceLabel, SynaptexEngine


def main() -> None:
    engine = SynaptexEngine(max_context_tokens=512)
    engine.register_agent(AgentIdentity(agent_id="planner", role="planning"))

    engine.encode(
        "Planner decided to prioritize retrieval tests before persistence work.",
        importance_label=ImportanceLabel.TRUST,
        importance=0.85,
        category="engineering",
        agent_id="planner",
    )
    engine.encode(
        "Executor found that context paging must reject oversized memory pages.",
        importance_label=ImportanceLabel.SURPRISE,
        importance=0.75,
        category="engineering",
        agent_id="planner",
    )

    recalled = engine.recall(tags=["engineering"], limit=2)

    print("Recalled memories:")
    for memory in recalled:
        print(f"- {memory.content_l3}")

    print("\nActive context:")
    print(engine.get_context())

    print("\nStats:")
    print(engine.get_stats()["engine"])


if __name__ == "__main__":
    main()
