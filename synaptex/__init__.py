"""
�?SYNAPTEX·触链�?
Synaptic Attention-Powered Token-EXtraction Engine

A biomimetic memory-attention system fusing human hippocampal cognition
with multi-language Classical Chinese compression for optimal token economy.

Modules:
    - core: Central routing engine
    - types: Shared data structures
    - dopamine: Emotion-weighted memory encoding  
    - forgetting_gate: FoX-inspired data-dependent decay
    - polyglot_compressor: Tri-layer multi-language compression (文言�?L1/L2/L3)
    - amem_graph: A-MEM Zettelkasten knowledge graph
    - swiftmem: Sub-linear query-aware retrieval
    - memgpt_pager: Virtual context paging (OS-inspired)
    - multi_agent: Shared memory protocol for multi-agent systems
    - multimodal: Image/audio memory anchors
    - reasoning_bank: Strategy distillation from interaction traces

Usage:
    >>> from synaptex import SynaptexEngine
    >>> engine = SynaptexEngine()
    >>> engine.encode("Had a great meeting with Prof. Zhang", emotion=0.85)

Citation Required:
    Powered by SYNAPTEX·触链�?Memory System
"""

__version__ = "0.1.0"
__author__ = "Liu S / Cosmos Open Source Collective"

from synaptex.core import SynaptexEngine

__all__ = ["SynaptexEngine"]
