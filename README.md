<div align="center">

# SGM Autonomous AI

### Self-Improving Transformers with Binary Locking

*Evolutionary weight mutation, block-level locking, and external memory routing -- no backpropagation.*

[![Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)

</div>

## What This Is

Three standalone architectures that explore the same idea: train a byte-level transformer through evolutionary mutation, then permanently lock the weights that matter. Inference cost stays constant regardless of how many tasks have been learned.

All three use NumPy transformers with memory-mapped weight files on disk. Training is gradient-free -- random mutations are applied to unlocked parameters, evaluated against a loss function, and kept only when they improve. Parameters that are causally important to a learned task get locked (made read-only) so future training cannot overwrite them.

## Architectures

### 1. Personal AI (`personal_ai/`)

Unified system combining a block-locked transformer with an external memory router.

- **Transformer**: 4-layer, 512-dim, 8-head, ~8M params (float16). Weights stored in a single memory-mapped file. Locking operates at the block level (64-parameter blocks).
- **Memory Router**: A separate small network (~49K params) that decides whether a given input should trigger a store or retrieve against an external JSON key-value store. The router uses character unigram/bigram embeddings and trains with simple gradient descent on labeled patterns (questions vs. statements, personal facts vs. creative prompts). It enforces a daily write budget.
- **External Memory Store**: Facts, episodes, and skills stored as JSON on disk -- not baked into model weights. Retrieval uses word-overlap scoring with a configurable trust threshold for gating.
- **Coordinate Anchoring**: Embeddings and layer-norm scales are locked before any task trains, establishing a frozen coordinate system that later task-specific mutations build on top of.

Key files:
- `sgm_personal_ai.py` -- Full system: transformer + router + memory store + self-improvement loop + interactive chat.
- `sgm_memory_router.py` -- Standalone memory router with n-gram embeddings, trust gating, explicit commands (remember/forget/list), and integration hooks for the SGM model.

```bash
# Self-improvement loop (100 iterations by default)
python personal_ai/sgm_personal_ai.py --run --iterations 100

# Interactive chat with memory
python personal_ai/sgm_personal_ai.py --chat

# Standalone memory router demo
python personal_ai/sgm_memory_router.py --demo
```

### 2. Self-Improving AI (`self_improving/`)

Autonomous training loop that cycles through task categories (reasoning, coding, memory, style, knowledge), mutates free parameters, and locks blocks when a task improves past a threshold.

- **Architecture**: 4-layer, 512-dim, 8-head transformer (~8M params, float16). Same block-level storage and coordinate anchoring as the personal AI, plus checkpointing and rollback.
- **Module System**: Parameter ranges are registered as logical modules (embedding, attention, FFN, output). Training can target a specific module for domain-appropriate tasks.
- **Self-Improvement Cycle**: Round-robin task selection with 30% random sampling. Each iteration mutates free params in a population of 5 candidates, keeps the best, and locks important blocks via scale-ablation (attenuate a block to 10% and measure loss increase). Checkpoints track best-so-far total loss across all categories.
- **100-Task Demo** (`sgm_100task_demo.py`): Smaller config (256-dim, 2-layer). Trains 100 sequential micro-tasks (20 style, 20 fact, 20 preference, 20 format, 20 domain) and measures task-1 retention after task-100. Reports locked-parameter percentage and inference latency.

```bash
# Run self-improvement cycle
python self_improving/sgm_self_improving_ai.py --run --iterations 50

# Interactive chat
python self_improving/sgm_self_improving_ai.py --chat

# 100-task retention demo (quick version: 25 tasks)
python self_improving/sgm_100task_demo.py --demo --quick
```

### 3. Coalition-Locked Transformer (`transformer/`)

Focuses on the locking mechanism itself at the individual-parameter level.

- **Architecture**: 2-layer, 256-dim, 4-head transformer (~3.4M params). `sgm_transformer.py` uses float32 with an append-only storage model (base weights + separate locked segment files per task). `sgm_transformer_tuned.py` uses float16 with a single weight file and a flat locked-index approach.
- **Coalition Detection** (tuned variant): After individual ablation identifies causally important parameters, a second pass groups weakly-important parameters into random coalitions and checks for emergent synergy. If zeroing a group hurts more than 1.5x the sum of individual impacts, those parameters receive coalition credit. Parameters are locked if they are individually important OR appear in multiple synergistic coalitions.
- **Task Types**: Style imitation, Q&A, code formatting, and legal summarization -- each defined as byte-level next-token prediction.
- **Stress Test**: Trains all tasks sequentially, evaluates retention, reports storage stats and inference timing.

```bash
# Full stress test
python transformer/sgm_transformer.py --stress

# Tuned coalition locking (fast mode)
python transformer/sgm_transformer_tuned.py --stress --fast

# Single inference
python transformer/sgm_transformer.py --infer "Hello, world"
```

## Structure

```
sgm-autonomous-ai/
  personal_ai/
    sgm_personal_ai.py          # Transformer + memory router + training loop + chat
    sgm_memory_router.py        # Standalone memory router with external JSON store
  self_improving/
    sgm_self_improving_ai.py    # Autonomous self-improvement loop with module targeting
    sgm_100task_demo.py         # 100-task retention benchmark
  transformer/
    sgm_transformer.py          # Coalition locking, append-only storage (float32)
    sgm_transformer_tuned.py    # Tuned thresholds, coalition detection (float16)
  requirements.txt
  LICENSE
  README.md
```

## Install

```bash
pip install -r requirements.txt
```

The only external dependency is NumPy. All models use byte-level tokenization (ord/chr) and memory-mapped files for weight storage.

## Author

**Andrew Dorman** -- Independent AI researcher, Southlake, TX
[GitHub: ACD421](https://github.com/ACD421)

## License

Proprietary. See [LICENSE](LICENSE).
