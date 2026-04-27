# SYNAPTEX

**Python toolkit for compact memory management in agent workflows.**

SYNAPTEX provides event capture, ranked retrieval, graph associations, active-context paging, and shared-memory coordination for Python systems.

Use it when a system needs to retain prior observations, retrieve relevant history, keep prompts within a token budget, or share selected state across components.

## Overview

The core loop is small:

1. Write an event with `encode(...)`.
2. Retrieve relevant memories with `recall(...)`.
3. Pull compact active context with `get_context()`.
4. Run `night_mode()` for consolidation and decay passes.

Each `MemoryUnit` stores three representations:

| Layer | Purpose |
| --- | --- |
| `L1` | Compact summary for prompt use |
| `L2` | Structured timeline entry for inspection |
| `L3` | Raw content for archival recall |

## Typical Uses

- Long-running agents that need durable task memory.
- Assistants that retrieve prior meetings, decisions, and constraints.
- Multi-component workflows with shared public memory and private notes.
- Context-window experiments that compare summaries against raw history.
- Systems that need inspectable memory state and access logs.

## Not A Vector Database

SYNAPTEX is not a hosted database, vector store, or production persistence layer. It is an embeddable memory orchestration toolkit that can sit beside vector databases, agent frameworks, or custom retrieval pipelines.

## Design Goals

- Small public API with inspectable Python objects.
- Retrieval by query, tags, and time windows.
- Token-aware active-context paging.
- Graph-linked associations between memories.
- Permissioned shared memory for multiple agents.
- Local-first behavior with no network calls or bundled secrets.

## Installation

From GitHub:

```bash
git clone https://github.com/liusulldel/SYNAPTEX.git
cd SYNAPTEX
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Quick Start

```python
from synaptex import ImportanceLabel, SynaptexEngine

engine = SynaptexEngine(max_context_tokens=2048)

memory = engine.encode(
    "The planner decided to prioritize retrieval tests before adding persistence.",
    importance_label=ImportanceLabel.TRUST,
    importance=0.85,
    category="engineering",
)

results = engine.recall(tags=["engineering"], limit=3)

print(memory.content_l1)
print(engine.get_context())
print([item.content_l3 for item in results])
```

Run the included example:

```bash
python examples/agentic_memory_demo.py
```

## Public API

| Object | Purpose |
| --- | --- |
| `SynaptexEngine` | Orchestration class for encode, recall, context, and consolidation |
| `MemoryUnit` | Inspectable memory record with L1/L2/L3 content and metadata |
| `ImportanceLabel` | Optional label used for importance weighting |
| `AgentIdentity` | Agent identity and permissions for shared memory |
| `SharedMemoryPool` | Permissioned public/private memory pool |

Common `SynaptexEngine` methods:

| Method | Purpose |
| --- | --- |
| `encode(...)` | Create, compress, index, and store a memory |
| `recall(...)` | Retrieve memories by query, tags, time window, and graph links |
| `get_context()` | Return compact active context for downstream prompts |
| `get_active_memories()` | Inspect currently paged memories |
| `forget(memory_id)` | Remove a memory from graph, retrieval, and active/archival context |
| `register_agent(...)` | Register an agent for shared-memory writes |
| `night_mode()` | Run decay and consolidation, then prune forgotten memories |
| `get_stats()` | Inspect engine, graph, retriever, pager, and shared-memory state |

## Architecture

```text
event text
  -> importance weighting
  -> tri-layer memory compression
  -> graph association + indexed retrieval
  -> archival storage + active-context paging
  -> compact context for downstream agents
```

## Modules

| Module | Purpose |
| --- | --- |
| `synaptex.core` | Main engine and orchestration |
| `synaptex.types` | Shared dataclasses and enums |
| `synaptex.dopamine` | Importance weighting from labels, keywords, and recency; legacy names remain as aliases |
| `synaptex.polyglot_compressor` | Heuristic tri-layer memory compression |
| `synaptex.amem_graph` | Graph associations and tag-based linking |
| `synaptex.swiftmem` | Query, tag, and time-window retrieval |
| `synaptex.memgpt_pager` | Active-context paging under a token budget |
| `synaptex.multi_agent` | Permissioned shared memory for agent teams |
| `synaptex.multimodal` | Attachment references for non-text memory anchors |
| `synaptex.reasoning_bank` | Reusable reasoning traces and anti-pattern notes |

## Current Limitations

- In-memory prototype: no durability guarantees yet.
- Compression is heuristic and not benchmarked as a semantic compressor.
- Retrieval is lightweight and inspectable, not embedding-based.
- APIs may change before a stable `1.0` release.
- Multi-agent permissions are local process controls, not a security sandbox.

## Release Hygiene

- Local-only package behavior: no network calls in the core library.
- No API keys, credentials, or external services are required.
- CI runs compile checks, tests, and package build.
- License metadata and repository license both use plain MIT.

## Purpose

SYNAPTEX explores how systems can retain more useful state, retrieve it reliably, and expose inspectable control surfaces while staying within a bounded context window.

## Contributing

Issues and pull requests are welcome, especially around retrieval quality, persistence adapters, benchmarks, documentation clarity, and integration examples.

## License

[MIT](LICENSE)
