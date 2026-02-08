#!/usr/bin/env python3
"""
SGM SELF-IMPROVING AI
=====================
A self-improving AI system using the validated SGM memory substrate.

Components:
1. Base transformer with block-level locking
2. Self-improvement loop (mutate → evaluate → lock)
3. Module system (reasoning, coding, memory)
4. Autonomous training scheduler

Usage:
  python sgm_self_improving.py --run       # Start self-improvement loop
  python sgm_self_improving.py --status    # Check current state
  python sgm_self_improving.py --chat      # Interactive mode
"""

import numpy as np
import json
import time
import os
import sys
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import argparse
import hashlib
import shutil

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 2048
    vocab_size: int = 256
    max_seq_len: int = 512
    
    # Substrate
    block_size: int = 64
    dtype: np.dtype = np.float16
    
    # Training
    mutation_rate: float = 0.01
    population_size: int = 5
    steps_per_task: int = 30
    
    # Self-improvement
    improvement_threshold: float = 0.95  # Must be at least 5% better
    max_iterations: int = 1000
    checkpoint_interval: int = 10
    
    @property
    def d_head(self): return self.d_model // self.n_heads
    
    @property
    def total_params(self):
        embed = self.vocab_size * self.d_model
        per_layer = self.d_model * self.d_model * 4 + self.d_model * self.d_ff * 2 + self.d_model * 4
        output = self.d_model * self.vocab_size
        return embed + per_layer * self.n_layers + output

# =============================================================================
# BLOCK STORAGE (from validated substrate)
# =============================================================================

class BlockStorage:
    def __init__(self, path: str, cfg: Config):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.dtype = cfg.dtype
        
        self.weights_file = self.path / "weights.mmap"
        self.index_file = self.path / "index.json"
        self.log_file = self.path / "mutations.log"
        
        self.index = self._load_index()
    
    def _load_index(self):
        if self.index_file.exists():
            return json.load(open(self.index_file))
        return {
            "total_params": 0,
            "block_size": self.cfg.block_size,
            "locked_blocks": [],
            "tasks": [],
            "modules": {},
            "iteration": 0,
            "improvements": []
        }
    
    def _save_index(self):
        json.dump(self.index, open(self.index_file, 'w'), indent=2)
    
    def log_mutation(self, msg: str):
        with open(self.log_file, 'a') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    
    def init(self, n_params: int):
        self.index["total_params"] = n_params
        w = np.random.randn(n_params).astype(self.dtype) * 0.02
        fp = np.memmap(self.weights_file, dtype=self.dtype, mode='w+', shape=(n_params,))
        fp[:] = w
        fp.flush()
        del fp
        self._save_index()
        mb = n_params * 2 / 1024 / 1024
        print(f"[INIT] {n_params:,} params ({n_params // self.cfg.block_size} blocks, {mb:.1f}MB)")
    
    def get_weights(self, mode='r+'):
        return np.memmap(self.weights_file, dtype=self.dtype, mode=mode,
                        shape=(self.index["total_params"],))
    
    def get_free_blocks(self) -> np.ndarray:
        n_blocks = self.index["total_params"] // self.cfg.block_size
        locked = set(self.index["locked_blocks"])
        free = sorted(set(range(n_blocks)) - locked)
        return np.array(free, dtype=np.int64)
    
    def lock_blocks(self, name: str, block_indices: np.ndarray):
        new_locks = [int(b) for b in block_indices if b not in self.index["locked_blocks"]]
        self.index["locked_blocks"].extend(new_locks)
        self.index["tasks"].append({
            "name": name,
            "blocks": new_locks,
            "n_params": len(new_locks) * self.cfg.block_size,
            "time": time.time()
        })
        self._save_index()
        self.log_mutation(f"LOCK: {name} -> {len(new_locks)} blocks")
        return len(new_locks)
    
    def lock_param_range(self, name: str, start: int, end: int):
        bs = self.cfg.block_size
        blocks = list(range(start // bs, (end + bs - 1) // bs))
        return self.lock_blocks(name, np.array(blocks, dtype=np.int64))
    
    def register_module(self, name: str, param_range: Tuple[int, int]):
        self.index["modules"][name] = {"start": param_range[0], "end": param_range[1]}
        self._save_index()
    
    def checkpoint(self, name: str):
        checkpoint_dir = self.path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Copy current weights
        shutil.copy(self.weights_file, checkpoint_dir / f"{name}_weights.mmap")
        shutil.copy(self.index_file, checkpoint_dir / f"{name}_index.json")
        self.log_mutation(f"CHECKPOINT: {name}")
    
    def rollback(self, name: str):
        checkpoint_dir = self.path / "checkpoints"
        weights_ckpt = checkpoint_dir / f"{name}_weights.mmap"
        index_ckpt = checkpoint_dir / f"{name}_index.json"
        
        if weights_ckpt.exists() and index_ckpt.exists():
            shutil.copy(weights_ckpt, self.weights_file)
            shutil.copy(index_ckpt, self.index_file)
            self.index = self._load_index()
            self.log_mutation(f"ROLLBACK: {name}")
            return True
        return False
    
    def stats(self):
        total = self.index["total_params"]
        bs = self.cfg.block_size
        n_blocks = total // bs
        locked_blocks = len(self.index["locked_blocks"])
        return {
            "total_params": total,
            "total_blocks": n_blocks,
            "locked_blocks": locked_blocks,
            "locked_params": locked_blocks * bs,
            "free_blocks": n_blocks - locked_blocks,
            "pct_locked": locked_blocks / n_blocks * 100 if n_blocks else 0,
            "n_tasks": len(self.index["tasks"]),
            "iteration": self.index["iteration"]
        }

# =============================================================================
# TRANSFORMER (scaled up from demo)
# =============================================================================

class Transformer:
    def __init__(self, cfg: Config, storage: BlockStorage):
        self.cfg = cfg
        self.storage = storage
        self._build_router()
        
        if storage.index["total_params"] == 0:
            storage.init(self.total_params)
            self._register_modules()
    
    def _build_router(self):
        cfg = self.cfg
        self.map = {}
        off = 0
        
        self.map["embed"] = (off, off + cfg.vocab_size * cfg.d_model)
        off += cfg.vocab_size * cfg.d_model
        
        for L in range(cfg.n_layers):
            for name in ["q", "k", "v", "o"]:
                size = cfg.d_model * cfg.d_model
                self.map[f"L{L}_{name}"] = (off, off + size)
                off += size
            self.map[f"L{L}_ff1"] = (off, off + cfg.d_model * cfg.d_ff)
            off += cfg.d_model * cfg.d_ff
            self.map[f"L{L}_ff2"] = (off, off + cfg.d_ff * cfg.d_model)
            off += cfg.d_ff * cfg.d_model
            self.map[f"L{L}_ln1"] = (off, off + cfg.d_model * 2)
            off += cfg.d_model * 2
            self.map[f"L{L}_ln2"] = (off, off + cfg.d_model * 2)
            off += cfg.d_model * 2
        
        self.map["output"] = (off, off + cfg.d_model * cfg.vocab_size)
        off += cfg.d_model * cfg.vocab_size
        self.total_params = off
    
    def _register_modules(self):
        """Register logical modules for targeted training"""
        cfg = self.cfg
        
        # Embedding module
        self.storage.register_module("embedding", self.map["embed"])
        
        # Attention module (all layers)
        attn_start = self.map["L0_q"][0]
        attn_end = self.map[f"L{cfg.n_layers-1}_o"][1]
        self.storage.register_module("attention", (attn_start, attn_end))
        
        # FFN module (all layers)
        ffn_ranges = []
        for L in range(cfg.n_layers):
            ffn_ranges.append(self.map[f"L{L}_ff1"])
            ffn_ranges.append(self.map[f"L{L}_ff2"])
        self.storage.register_module("ffn", (ffn_ranges[0][0], ffn_ranges[-1][1]))
        
        # Output module
        self.storage.register_module("output", self.map["output"])
    
    def anchor_coordinate_system(self):
        """Lock representation anchors before training"""
        print("\n[ANCHOR] Locking coordinate systems...")
        
        # Lock embeddings
        s, e = self.map["embed"]
        n = self.storage.lock_param_range("ANCHOR_embed", s, e)
        print(f"  Embeddings: {n} blocks")
        
        # Lock LN scale (first half of each LN)
        for L in range(self.cfg.n_layers):
            for ln in [f"L{L}_ln1", f"L{L}_ln2"]:
                s, e = self.map[ln]
                scale_end = s + self.cfg.d_model
                n = self.storage.lock_param_range(f"ANCHOR_{ln}_scale", s, scale_end)
        
        stats = self.storage.stats()
        print(f"  Total anchored: {stats['locked_params']:,} params ({stats['pct_locked']:.1f}%)")
    
    def _get(self, w, name):
        s, e = self.map[name]
        return w[s:e]
    
    def forward(self, ids):
        cfg = self.cfg
        w = self.storage.get_weights('r')
        
        embed = self._get(w, "embed").reshape(cfg.vocab_size, cfg.d_model)
        x = embed[np.clip(ids, 0, cfg.vocab_size - 1)].astype(np.float32)
        
        for L in range(cfg.n_layers):
            ln1 = self._get(w, f"L{L}_ln1").reshape(2, cfg.d_model).astype(np.float32)
            xn = self._ln(x, ln1[0], ln1[1])
            
            Wq = self._get(w, f"L{L}_q").reshape(cfg.d_model, cfg.d_model).astype(np.float32)
            Wk = self._get(w, f"L{L}_k").reshape(cfg.d_model, cfg.d_model).astype(np.float32)
            Wv = self._get(w, f"L{L}_v").reshape(cfg.d_model, cfg.d_model).astype(np.float32)
            Wo = self._get(w, f"L{L}_o").reshape(cfg.d_model, cfg.d_model).astype(np.float32)
            
            q, k, v = xn @ Wq, xn @ Wk, xn @ Wv
            attn = self._mha(q, k, v) @ Wo
            x = x + attn
            
            ln2 = self._get(w, f"L{L}_ln2").reshape(2, cfg.d_model).astype(np.float32)
            xn = self._ln(x, ln2[0], ln2[1])
            
            ff1 = self._get(w, f"L{L}_ff1").reshape(cfg.d_model, cfg.d_ff).astype(np.float32)
            ff2 = self._get(w, f"L{L}_ff2").reshape(cfg.d_ff, cfg.d_model).astype(np.float32)
            x = x + self._gelu(xn @ ff1) @ ff2
        
        out = self._get(w, "output").reshape(cfg.d_model, cfg.vocab_size).astype(np.float32)
        del w
        return x @ out
    
    def _ln(self, x, g, b, eps=1e-5):
        m = x.mean(-1, keepdims=True)
        v = x.var(-1, keepdims=True)
        return g * (x - m) / np.sqrt(v + eps) + b
    
    def _mha(self, q, k, v):
        cfg = self.cfg
        seq = q.shape[0]
        q = q.reshape(seq, cfg.n_heads, cfg.d_head)
        k = k.reshape(seq, cfg.n_heads, cfg.d_head)
        v = v.reshape(seq, cfg.n_heads, cfg.d_head)
        
        scores = np.einsum('ihd,jhd->hij', q, k) / np.sqrt(cfg.d_head)
        mask = np.triu(np.ones((seq, seq)), 1) * -1e9
        scores = scores + mask
        attn = np.exp(scores - scores.max(-1, keepdims=True))
        attn = attn / (attn.sum(-1, keepdims=True) + 1e-9)
        out = np.einsum('hij,jhd->ihd', attn, v)
        return out.reshape(seq, cfg.d_model)
    
    def _gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def loss(self, ids, targets):
        logits = self.forward(ids)
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs = probs / (probs.sum(-1, keepdims=True) + 1e-9)
        return -np.mean(np.log(probs[np.arange(len(targets)), targets] + 1e-9))
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.8) -> str:
        ids = [min(ord(c), 255) for c in prompt]
        
        for _ in range(max_tokens):
            input_ids = np.array(ids[-self.cfg.max_seq_len:], dtype=np.int32)
            logits = self.forward(input_ids)
            
            last_logits = logits[-1] / temperature
            probs = np.exp(last_logits - last_logits.max())
            probs = probs / probs.sum()
            
            next_token = np.random.choice(len(probs), p=probs)
            ids.append(next_token)
            
            if next_token == ord('\n') and len(ids) > len(prompt) + 10:
                break
        
        return ''.join(chr(min(max(0, i), 127)) for i in ids)

# =============================================================================
# SELF-IMPROVEMENT ENGINE
# =============================================================================

class SelfImprover:
    def __init__(self, model: Transformer, storage: BlockStorage, cfg: Config):
        self.model = model
        self.storage = storage
        self.cfg = cfg
        
        self.task_library = self._build_task_library()
        self.metrics_history = []
    
    def _build_task_library(self) -> Dict[str, List[Tuple]]:
        """Build library of training tasks organized by module"""
        
        def encode(s): 
            return np.array([min(ord(c), 255) for c in s], dtype=np.int32)
        
        def make_data(texts): 
            return [(encode(t)[:-1], encode(t)[1:]) for t in texts if len(t) > 2]
        
        tasks = {
            "reasoning": [],
            "coding": [],
            "memory": [],
            "style": [],
            "knowledge": []
        }
        
        # Reasoning tasks
        tasks["reasoning"].extend([
            ("logic_if_then", make_data([
                "If A then B. A is true. Therefore B is true.",
                "If X then Y. Y is false. Therefore X is false.",
                "All cats are mammals. Fluffy is a cat. Therefore Fluffy is a mammal.",
            ])),
            ("logic_compare", make_data([
                "5 is greater than 3. 3 is greater than 1. Therefore 5 is greater than 1.",
                "A is before B. B is before C. Therefore A is before C.",
            ])),
            ("logic_negate", make_data([
                "It is not the case that all birds can fly.",
                "Not all statements are true.",
            ])),
        ])
        
        # Coding tasks
        tasks["coding"].extend([
            ("python_function", make_data([
                "def add(a, b):\n    return a + b\n",
                "def multiply(x, y):\n    return x * y\n",
                "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n",
            ])),
            ("python_loop", make_data([
                "for i in range(10):\n    print(i)\n",
                "while x > 0:\n    x = x - 1\n",
                "for item in items:\n    process(item)\n",
            ])),
            ("python_class", make_data([
                "class Counter:\n    def __init__(self):\n        self.count = 0\n    def increment(self):\n        self.count += 1\n",
            ])),
        ])
        
        # Memory/recall tasks
        tasks["memory"].extend([
            ("fact_recall", make_data([
                "Q: What is 2+2?\nA: 4",
                "Q: What color is the sky?\nA: Blue",
                "Q: How many days in a week?\nA: Seven",
            ])),
            ("context_recall", make_data([
                "The capital of France is Paris. Q: What is the capital of France? A: Paris",
                "My name is Alice. Q: What is my name? A: Alice",
            ])),
        ])
        
        # Style tasks
        tasks["style"].extend([
            ("formal", make_data([
                "I am writing to inform you of the following matter.",
                "Please be advised that your request has been processed.",
                "We acknowledge receipt of your correspondence.",
            ])),
            ("concise", make_data([
                "Yes.", "No.", "Done.", "Confirmed.", "Understood.",
            ])),
            ("technical", make_data([
                "The function returns a boolean value.",
                "Initialize the array with default values.",
                "The algorithm has O(n) time complexity.",
            ])),
        ])
        
        # Knowledge tasks
        tasks["knowledge"].extend([
            ("math", make_data([
                "The derivative of x^2 is 2x.",
                "The integral of 1/x is ln(x).",
                "Pi is approximately 3.14159.",
            ])),
            ("science", make_data([
                "Water boils at 100 degrees Celsius.",
                "The speed of light is approximately 300,000 km/s.",
                "DNA is a double helix structure.",
            ])),
        ])
        
        return tasks
    
    def train_task(self, name: str, data: List[Tuple], steps: int = None, 
                   target_module: str = None) -> Dict:
        """Train on a task, optionally targeting a specific module"""
        
        steps = steps or self.cfg.steps_per_task
        w = self.storage.get_weights('r+')
        free_blocks = self.storage.get_free_blocks()
        
        if len(free_blocks) == 0:
            del w
            return {"loss": float('inf'), "locked": 0, "improved": False}
        
        # Get free params
        bs = self.cfg.block_size
        
        if target_module and target_module in self.storage.index["modules"]:
            # Only train within target module
            mod = self.storage.index["modules"][target_module]
            mod_blocks = set(range(mod["start"] // bs, mod["end"] // bs))
            free_blocks = np.array([b for b in free_blocks if b in mod_blocks])
        
        if len(free_blocks) == 0:
            del w
            return {"loss": float('inf'), "locked": 0, "improved": False}
        
        free_params = []
        for b in free_blocks:
            free_params.extend(range(b * bs, (b + 1) * bs))
        free_params = np.array(free_params, dtype=np.int64)
        
        def loss():
            return np.mean([self.model.loss(d[0], d[1]) for d in data])
        
        initial_loss = loss()
        best_loss = initial_loss
        
        for step in range(steps):
            for _ in range(self.cfg.population_size):
                n = min(len(free_params) // 10 + 1, len(free_params))
                idx = np.random.choice(free_params, n, replace=False)
                old = w[idx].copy()
                w[idx] += np.random.randn(n).astype(w.dtype) * self.cfg.mutation_rate
                
                new_loss = loss()
                if new_loss < best_loss:
                    best_loss = new_loss
                else:
                    w[idx] = old
            
            if step % 5 == 0:
                w.flush()
        
        w.flush()
        
        # Determine if we improved enough
        improved = best_loss < initial_loss * self.cfg.improvement_threshold
        
        # Lock important blocks if improved
        locked = 0
        if improved:
            important = self._find_important_blocks(free_blocks, data, w)
            if len(important) > 0:
                locked = self.storage.lock_blocks(name, important)
        
        del w
        
        return {
            "loss": best_loss,
            "initial_loss": initial_loss,
            "improvement": (initial_loss - best_loss) / initial_loss * 100,
            "locked": locked,
            "improved": improved
        }
    
    def _find_important_blocks(self, free_blocks: np.ndarray, data: List[Tuple], 
                                w: np.memmap, n_test: int = 20) -> np.ndarray:
        """Find important blocks via scale ablation"""
        bs = self.cfg.block_size
        
        def loss():
            return np.mean([self.model.loss(d[0], d[1]) for d in data])
        
        base = loss()
        importance = np.zeros(len(free_blocks), dtype=np.float32)
        
        test_idx = np.random.choice(len(free_blocks), min(n_test, len(free_blocks)), replace=False)
        
        for i in test_idx:
            block = free_blocks[i]
            start, end = block * bs, (block + 1) * bs
            
            old = w[start:end].copy()
            w[start:end] *= 0.1
            
            importance[i] = loss() - base
            w[start:end] = old
        
        positive = np.where(importance > 0)[0]
        if len(positive) > 0:
            return free_blocks[positive]
        return np.array([], dtype=np.int64)
    
    def evaluate_all_tasks(self) -> Dict[str, float]:
        """Evaluate current performance on all task categories"""
        results = {}
        
        for category, tasks in self.task_library.items():
            total_loss = 0
            count = 0
            for name, data in tasks:
                if data:
                    task_loss = np.mean([self.model.loss(d[0], d[1]) for d in data])
                    total_loss += task_loss
                    count += 1
            results[category] = total_loss / count if count > 0 else float('inf')
        
        return results
    
    def run_improvement_cycle(self, iterations: int = None):
        """Main self-improvement loop"""
        
        iterations = iterations or self.cfg.max_iterations
        
        print("\n" + "="*70)
        print("SGM SELF-IMPROVEMENT CYCLE")
        print("="*70)
        
        # Initial evaluation
        initial_metrics = self.evaluate_all_tasks()
        print("\n[INITIAL STATE]")
        for cat, loss in initial_metrics.items():
            print(f"  {cat}: {loss:.4f}")
        
        self.storage.checkpoint("initial")
        
        # Flatten task list with priorities
        all_tasks = []
        for category, tasks in self.task_library.items():
            for name, data in tasks:
                all_tasks.append((category, name, data))
        
        best_checkpoint = "initial"
        best_total_loss = sum(initial_metrics.values())
        
        for iteration in range(iterations):
            self.storage.index["iteration"] = iteration
            self.storage._save_index()
            
            # Select task (round-robin with some randomness)
            task_idx = iteration % len(all_tasks)
            if np.random.random() < 0.3:  # 30% random selection
                task_idx = np.random.randint(len(all_tasks))
            
            category, name, data = all_tasks[task_idx]
            
            if not data:
                continue
            
            # Train on task
            result = self.train_task(f"{category}_{name}_iter{iteration}", data, 
                                    target_module=category if category in ["coding", "reasoning"] else None)
            
            # Log progress
            if iteration % 10 == 0 or result["improved"]:
                stats = self.storage.stats()
                print(f"\n[Iter {iteration}] {category}/{name}")
                print(f"  Loss: {result['initial_loss']:.4f} -> {result['loss']:.4f} ({result['improvement']:.1f}%)")
                print(f"  Locked: {stats['locked_blocks']} blocks ({stats['pct_locked']:.1f}%)")
                
                if result["improved"]:
                    print(f"  ✓ IMPROVED - locked {result['locked']} new blocks")
            
            # Checkpoint periodically
            if iteration % self.cfg.checkpoint_interval == 0 and iteration > 0:
                current_metrics = self.evaluate_all_tasks()
                current_total = sum(current_metrics.values())
                
                if current_total < best_total_loss:
                    best_total_loss = current_total
                    best_checkpoint = f"iter_{iteration}"
                    self.storage.checkpoint(best_checkpoint)
                    print(f"\n  [CHECKPOINT] New best at iteration {iteration}")
                
                self.metrics_history.append({
                    "iteration": iteration,
                    "metrics": current_metrics,
                    "total_loss": current_total,
                    "locked_pct": stats['pct_locked']
                })
            
            # Check for capacity exhaustion
            stats = self.storage.stats()
            if stats['pct_locked'] > 95:
                print(f"\n[STOP] Capacity exhausted at {stats['pct_locked']:.1f}%")
                break
        
        # Final evaluation
        print("\n" + "="*70)
        print("FINAL STATE")
        print("="*70)
        
        final_metrics = self.evaluate_all_tasks()
        stats = self.storage.stats()
        
        print(f"\n{'Category':<15} | {'Initial':>10} | {'Final':>10} | {'Change':>10}")
        print("-"*50)
        for cat in initial_metrics:
            init = initial_metrics[cat]
            final = final_metrics[cat]
            change = (init - final) / init * 100
            arrow = "↓" if change > 0 else "↑"
            print(f"{cat:<15} | {init:>10.4f} | {final:>10.4f} | {arrow}{abs(change):>8.1f}%")
        
        print(f"\nStorage: {stats['locked_params']:,} params locked ({stats['pct_locked']:.1f}%)")
        print(f"Iterations: {iteration + 1}")
        
        return {
            "initial": initial_metrics,
            "final": final_metrics,
            "stats": stats,
            "history": self.metrics_history
        }

# =============================================================================
# INTERACTIVE CHAT
# =============================================================================

def interactive_chat(model: Transformer):
    """Simple interactive chat interface"""
    print("\n" + "="*70)
    print("SGM AI - Interactive Mode")
    print("Type 'quit' to exit, 'status' for stats")
    print("="*70 + "\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            if prompt.lower() == 'quit':
                break
            if prompt.lower() == 'status':
                stats = model.storage.stats()
                print(f"\n[STATUS] Params: {stats['total_params']:,}, Locked: {stats['pct_locked']:.1f}%\n")
                continue
            
            response = model.generate(prompt + "\n", max_tokens=100)
            # Extract response after prompt
            if prompt in response:
                response = response[len(prompt):].strip()
            print(f"AI: {response}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SGM Self-Improving AI')
    parser.add_argument('--run', action='store_true', help='Run self-improvement cycle')
    parser.add_argument('--chat', action='store_true', help='Interactive chat mode')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--reset', action='store_true', help='Reset and start fresh')
    parser.add_argument('--iterations', type=int, default=100, help='Number of improvement iterations')
    parser.add_argument('--path', default='./sgm_ai', help='Storage path')
    args = parser.parse_args()
    
    # Reset if requested
    if args.reset and Path(args.path).exists():
        shutil.rmtree(args.path)
        print("[RESET] Cleared previous state")
    
    # Initialize
    cfg = Config()
    storage = BlockStorage(args.path, cfg)
    model = Transformer(cfg, storage)
    
    # Anchor on first run
    if storage.index["iteration"] == 0 and len(storage.index["tasks"]) == 0:
        model.anchor_coordinate_system()
    
    if args.status:
        stats = storage.stats()
        print("\n=== SGM AI STATUS ===")
        print(f"Total params: {stats['total_params']:,}")
        print(f"Locked: {stats['locked_params']:,} ({stats['pct_locked']:.1f}%)")
        print(f"Free: {stats['free_blocks']} blocks")
        print(f"Tasks learned: {stats['n_tasks']}")
        print(f"Iterations: {stats['iteration']}")
        
    elif args.run:
        improver = SelfImprover(model, storage, cfg)
        results = improver.run_improvement_cycle(iterations=args.iterations)
        
        # Save results
        with open(Path(args.path) / "results.json", 'w') as f:
            json.dump({
                "initial": results["initial"],
                "final": results["final"],
                "stats": results["stats"]
            }, f, indent=2)
        
    elif args.chat:
        interactive_chat(model)
        
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python sgm_self_improving.py --run --iterations 50")
        print("  python sgm_self_improving.py --chat")
        print("  python sgm_self_improving.py --status")

if __name__ == "__main__":
    main()