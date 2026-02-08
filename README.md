# SGM Autonomous AI: Self-Improving Intelligence with Sparse Geometric Mutation

**Author:** Andrew Dorman ([Hollow Point Labs](https://github.com/ACD421))

## Overview

This repository contains experimental systems for building autonomous, self-improving AI using the SGM (Sparse Geometric Mutation) substrate. SGM's convergence-based binary locking provides the foundation: learned capabilities are permanently preserved while new skills are acquired without interference.

Three architectures are explored:

1. **Personal AI** -- Memory routing separates factual recall (external JSON store) from reasoning (SGM-locked weights), preventing memory tasks from competing with learned capabilities.
2. **Self-Improving AI** -- A mutation-evaluate-lock loop that autonomously discovers and preserves improvements across reasoning, coding, and memory modules.
3. **SGM Transformer** -- Real transformer architectures (2-layer, multi-head attention) with coalition locking at head/layer granularity and memory-mapped persistent storage.

## Architecture

### Personal AI (`personal_ai/`)

| File | Description |
|------|-------------|
| `sgm_personal_ai.py` | Unified personal AI: Memory Router (external JSON store for facts/episodes), SGM Blocks (weight-based learning for reasoning/coding/style), Anchored Base (frozen coordinate system). 512d, 8-head, 4-layer config. |
| `sgm_memory_router.py` | Memory routing module: separates memory retrieval from weight-based learning. Core model stays frozen/SGM-locked while a tiny trainable router directs queries to an external key-value memory store. |

### Self-Improving AI (`self_improving/`)

| File | Description |
|------|-------------|
| `sgm_self_improving_ai.py` | Self-improvement loop: base transformer with block-level locking, mutate-evaluate-lock cycle, module system (reasoning/coding/memory), autonomous training scheduler. BlockStorage with mmap weights. |
| `sgm_100task_demo.py` | Killer demo: 100 sequential micro-tasks (style, facts, preferences, formatting). Block-level locking (32 params/block). Measures task 1 retention after task 100. Success: retention ~1.0x with locked << total params. |

### SGM Transformer (`transformer/`)

| File | Description |
|------|-------------|
| `sgm_transformer.py` | 2-layer transformer with coalition locking at head/layer granularity. d_model=256, 4 attention heads, FFN=1024. Memory-mapped append-only storage. Real language tasks with constant-time inference. |
| `sgm_transformer_tuned.py` | Tuned variant: threshold 0.001->0.0001 (10x more sensitive), min_lock=50 params/task, n_samples=100, float16 support. Same coalition detection logic with calibrated thresholds. |

## License

MIT License. See [LICENSE](LICENSE) for details.
