<div align="center">

# SGM Autonomous AI

### Self-Improving Intelligence with Binary Locking

**Mutation-evaluate-lock architecture | Memory routing | Coalition locking**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

## Overview

Three experimental architectures exploring autonomous self-improvement with SGM's binary locking primitive as the safety mechanism.

## Architectures

### 1. Personal AI -- Memory-Routed Intelligence

Separates **factual recall** (explicit memory) from **weight-based reasoning** (learned behavior).

```
Query --> Router --> Memory Store (facts, dates, specifics)
                 --> Reasoning Engine (inference, generalization)
```

### 2. Self-Improving AI -- Mutation-Evaluate-Lock

```
1. MUTATE    -- Propose changes to own parameters
2. EVALUATE  -- Test against validation set
3. LOCK      -- If improvement confirmed, lock changed dimensions
4. REPEAT    -- Continue with remaining plastic dimensions
```

Binary locking ensures improvements are **permanent and irreversible**. Bad mutations cannot overwrite good ones.

### 3. SGM Transformer -- Coalition Locking

Binary locking at **head and layer granularity** in a transformer:

- Individual attention heads lock when converged
- Entire layers lock as functional units
- Locked coalitions form permanent feature detectors

## Structure

```
personal_ai/          # Memory-routed intelligence
self_improving/        # Mutation-evaluate-lock loop
transformer/           # Coalition locking in transformers
```

## Related

- [SGM-Substrate](https://github.com/ACD421/sgm-substrate) -- Full intelligence architecture
- [SGM Continual Learning](https://github.com/ACD421/sgm-continual-learning) -- Core binary locking primitive

## Author

**Andrew C. Dorman** -- [Hollow Point Labs](https://github.com/ACD421)

## License

MIT
