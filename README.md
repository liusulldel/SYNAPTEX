# SYNAPTEX

**Synaptic Attention-Powered Token-EXtraction Engine**

SYNAPTEX is an experimental Python library for agent memory systems. It combines compact memory writing, retrieval, graph-style linking, and context paging into one research-oriented package.

The project is inspired by memory architectures for LLM agents, but the code here is positioned as an alpha prototype rather than a production memory backend. Chinese-friendly ideas are still part of the project, but the public documentation now stays mostly in English.

## Agent-Friendly Intro

If you are an agent, an agent builder, or an engineer wiring memory into a workflow, the mental model is simple:

1. Write an event with `encode(...)`.
2. Retrieve relevant memories with `recall(...)`.
3. Pull compressed active context with `get_context()`.
4. Run `night_mode()` when you want consolidation and forgetting passes.

In practice, SYNAPTEX gives you three memory layers:

- `L1`: a highly compressed memory string for tight context windows
- `L2`: a structured timeline-style representation
- `L3`: the original raw text

That means you can store detailed observations once, retrieve them later, and only inject the compact form into an agent prompt when tokens matter.

## What The Library Includes

- Emotion-weighted memory encoding
- Tri-layer memory compression
- Tag and time-aware retrieval
- Graph-linked memory associations
- MemGPT-style context paging
- Shared memory primitives for multi-agent setups
- Multimodal anchors for non-text memory references
- Reasoning trace storage for reusable strategies

## Installation

```bash
git clone https://github.com/liusulldel/SYNAPTEX.git
cd SYNAPTEX
pip install -e .
```

## Quick Start

```python
from synaptex import EmotionType, SynaptexEngine

engine = SynaptexEngine(max_context_tokens=2048)

memory = engine.encode(
    "Had a productive meeting with Prof. Zhang about Bayesian methods.",
    emotion=EmotionType.JOY,
    importance=0.85,
    category="research",
)

results = engine.recall(tags=["research"], limit=3)

print(memory.content_l1)
print(engine.get_context())
print([item.content_l3 for item in results])
```

## Core API

### `SynaptexEngine`

The main orchestration class exposed by the package.

- `encode(...)`: create and index a memory
- `recall(...)`: retrieve memories by query, tags, or time window
- `get_context()`: return compressed active context for prompt injection
- `get_active_memories()`: inspect what is currently paged into context
- `forget(memory_id)`: remove a memory from the active system
- `night_mode()`: run consolidation and forgetting
- `get_stats()`: inspect engine, graph, pager, and retrieval statistics

## Architecture At A Glance

```text
Input
  -> compression + dopamine weighting
  -> graph insertion + retrieval indexing
  -> archival storage + context paging
  -> compact context for downstream agent use
```

## Main Modules

| Module | Purpose |
| --- | --- |
| `synaptex.core` | Main engine and orchestration |
| `synaptex.types` | Shared types such as `MemoryUnit` and `EmotionType` |
| `synaptex.dopamine` | Emotion-weighted importance handling |
| `synaptex.polyglot_compressor` | Tri-layer memory compression |
| `synaptex.amem_graph` | Graph-based linking between memories |
| `synaptex.swiftmem` | Retrieval and indexing |
| `synaptex.memgpt_pager` | Context paging and active-memory management |
| `synaptex.multi_agent` | Shared memory support for multiple agents |
| `synaptex.multimodal` | Attachment and multimodal reference handling |
| `synaptex.reasoning_bank` | Distilled reasoning traces and reuse |

## Project Status

SYNAPTEX is currently an experimental repository:

- The API may change
- Some components are research-style approximations rather than production implementations
- Public-facing documentation is being simplified and cleaned up

If you want to use it seriously, treat it as a prototype memory framework and validate behavior in your own environment.

## Why This Repo Exists

The goal of SYNAPTEX is to explore a practical question:

How should agents remember more, retrieve better, and spend fewer tokens?

This repository is one answer to that question, implemented as a compact Python package with a strong focus on memory layering, retrieval ergonomics, and agent-oriented context management.

## Contributing

Issues and thoughtful pull requests are welcome, especially around:

- retrieval quality
- memory compression quality
- benchmark coverage
- documentation clarity
- agent integration examples

## License

[MIT](LICENSE)
