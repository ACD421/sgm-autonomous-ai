#!/usr/bin/env python3
"""
SGM TRANSFORMER - COALITION LOCKING (TUNED)
============================================
Your original method with calibrated thresholds.

Changes from original:
- threshold: 0.001 -> 0.0001 (10x more sensitive)
- min_lock: 50 params per task (prevents under-locking)
- n_samples: 50 -> 100 (better coverage)
- float16 support

Your coalition detection logic is unchanged.
"""

import numpy as np
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import argparse

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TransformerConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 1024
    vocab_size: int = 256
    max_seq_len: int = 128
    dtype: np.dtype = np.float16
    
    @property
    def d_head(self): return self.d_model // self.n_heads
    
    @property
    def total_params(self):
        embed = self.vocab_size * self.d_model
        qkvo = self.d_model * self.d_model * 4
        ffn = self.d_model * self.d_ff * 2
        ln = self.d_model * 4
        per_layer = qkvo + ffn + ln
        output = self.d_model * self.vocab_size
        return embed + per_layer * self.n_layers + output

# =============================================================================
# STORAGE (mmap, append-only)
# =============================================================================

class StorageManager:
    def __init__(self, base_path: str, dtype=np.float16):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype
        
        self.locked_path = self.base_path / "locked_segments"
        self.locked_path.mkdir(exist_ok=True)
        
        self.index_path = self.base_path / "lock_index.json"
        self.weights_path = self.base_path / "weights.mmap"
        
        self.index = self._load_index()
    
    def _load_index(self):
        if self.index_path.exists():
            return json.load(open(self.index_path))
        return {"tasks": [], "locked_ranges": [], "total_locked": 0, "total_params": 0}
    
    def _save_index(self):
        json.dump(self.index, open(self.index_path, 'w'), indent=2)
    
    def init_weights(self, total_params: int):
        self.index["total_params"] = total_params
        w = np.random.randn(total_params).astype(self.dtype) * 0.02
        fp = np.memmap(self.weights_path, dtype=self.dtype, mode='w+', shape=(total_params,))
        fp[:] = w
        fp.flush()
        del fp
        self._save_index()
        mb = total_params * (2 if self.dtype == np.float16 else 4) / 1024 / 1024
        print(f"Initialized {total_params:,} params ({self.dtype.__name__}, {mb:.1f}MB)")
    
    def get_weights(self, mode='r+'):
        return np.memmap(self.weights_path, dtype=self.dtype, mode=mode, 
                        shape=(self.index["total_params"],))
    
    def get_locked_mask(self):
        mask = np.zeros(self.index["total_params"], dtype=bool)
        for idx in self.index["locked_ranges"]:
            mask[idx] = True
        return mask
    
    def lock_task(self, task_name: str, indices: np.ndarray):
        task_id = len(self.index["tasks"])
        self.index["tasks"].append({
            "id": task_id, "name": task_name, 
            "n_params": len(indices), "timestamp": time.time()
        })
        self.index["total_locked"] += len(indices)
        self.index["locked_ranges"].extend(indices.tolist())
        self._save_index()
        return task_id
    
    def get_stats(self):
        total = self.index["total_params"]
        locked = len(set(self.index["locked_ranges"]))
        return {
            "total": total, "locked": locked, "free": total - locked,
            "pct": locked / total * 100 if total else 0,
            "n_tasks": len(self.index["tasks"])
        }

# =============================================================================
# PARAMETER ROUTER
# =============================================================================

class ParameterRouter:
    def __init__(self, cfg: TransformerConfig):
        self.cfg = cfg
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
        self.total = off + cfg.d_model * cfg.vocab_size

# =============================================================================
# TRANSFORMER
# =============================================================================

class SGMTransformer:
    def __init__(self, cfg: TransformerConfig, storage: StorageManager):
        self.cfg = cfg
        self.storage = storage
        self.router = ParameterRouter(cfg)
        
        if storage.index["total_params"] == 0:
            storage.init_weights(self.router.total)
    
    def _get_param(self, weights, name):
        s, e = self.router.map[name]
        return weights[s:e]
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        w = self.storage.get_weights('r')
        
        # Embedding
        embed = self._get_param(w, "embed").reshape(cfg.vocab_size, cfg.d_model)
        x = embed[input_ids].astype(np.float32)
        
        for L in range(cfg.n_layers):
            # Layer norm 1
            ln1 = self._get_param(w, f"L{L}_ln1").reshape(2, cfg.d_model).astype(np.float32)
            x_norm = self._layer_norm(x, ln1[0], ln1[1])
            
            # Self attention
            Wq = self._get_param(w, f"L{L}_q").reshape(cfg.d_model, cfg.d_model).astype(np.float32)
            Wk = self._get_param(w, f"L{L}_k").reshape(cfg.d_model, cfg.d_model).astype(np.float32)
            Wv = self._get_param(w, f"L{L}_v").reshape(cfg.d_model, cfg.d_model).astype(np.float32)
            Wo = self._get_param(w, f"L{L}_o").reshape(cfg.d_model, cfg.d_model).astype(np.float32)
            
            q, k, v = x_norm @ Wq, x_norm @ Wk, x_norm @ Wv
            attn_out = self._multihead_attention(q, k, v) @ Wo
            x = x + attn_out
            
            # Layer norm 2 + FFN
            ln2 = self._get_param(w, f"L{L}_ln2").reshape(2, cfg.d_model).astype(np.float32)
            x_norm = self._layer_norm(x, ln2[0], ln2[1])
            
            ff1 = self._get_param(w, f"L{L}_ff1").reshape(cfg.d_model, cfg.d_ff).astype(np.float32)
            ff2 = self._get_param(w, f"L{L}_ff2").reshape(cfg.d_ff, cfg.d_model).astype(np.float32)
            x = x + self._gelu(x_norm @ ff1) @ ff2
        
        # Output
        out_proj = self._get_param(w, "output").reshape(cfg.d_model, cfg.vocab_size).astype(np.float32)
        logits = x @ out_proj
        
        del w
        return logits
    
    def _layer_norm(self, x, gamma, beta, eps=1e-5):
        mean = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta
    
    def _multihead_attention(self, q, k, v):
        cfg = self.cfg
        seq_len = q.shape[0]
        
        q = q.reshape(seq_len, cfg.n_heads, cfg.d_head)
        k = k.reshape(seq_len, cfg.n_heads, cfg.d_head)
        v = v.reshape(seq_len, cfg.n_heads, cfg.d_head)
        
        scores = np.einsum('ihd,jhd->hij', q, k) / np.sqrt(cfg.d_head)
        mask = np.triu(np.ones((seq_len, seq_len)), 1) * -1e9
        scores = scores + mask
        
        attn = np.exp(scores - scores.max(-1, keepdims=True))
        attn = attn / (attn.sum(-1, keepdims=True) + 1e-9)
        
        out = np.einsum('hij,jhd->ihd', attn, v)
        return out.reshape(seq_len, cfg.d_model)
    
    def _gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def loss(self, input_ids, target_ids):
        logits = self.forward(input_ids)
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs = probs / (probs.sum(-1, keepdims=True) + 1e-9)
        return -np.mean(np.log(probs[np.arange(len(target_ids)), target_ids] + 1e-9))

# =============================================================================
# SGM TRAINER WITH COALITION DETECTION
# =============================================================================

class SGMTrainer:
    """Your original method with tuned thresholds"""
    
    def __init__(self, model: SGMTransformer, storage: StorageManager):
        self.model = model
        self.storage = storage
    
    def train_task(
        self,
        task_name: str,
        data: List[Tuple[np.ndarray, np.ndarray]],
        n_steps: int = 50,
        population_size: int = 5,
        mutation_rate: float = 0.02
    ) -> Dict:
        
        weights = self.storage.get_weights('r+')
        locked_mask = self.storage.get_locked_mask()
        free = np.where(~locked_mask)[0]
        
        if len(free) == 0:
            print(f"  No free parameters!")
            del weights
            return {"loss": float('inf'), "locked": 0}
        
        print(f"  Training on {len(free):,} free params")
        
        def compute_loss():
            return np.mean([self.model.loss(inp, tgt) for inp, tgt in data])
        
        best_loss = compute_loss()
        
        for step in range(n_steps):
            for _ in range(population_size):
                n_mutate = min(len(free) // 10 + 1, len(free))
                idx = np.random.choice(free, n_mutate, replace=False)
                
                old_vals = weights[idx].copy()
                weights[idx] += np.random.randn(n_mutate).astype(weights.dtype) * mutation_rate
                weights.flush()
                
                new_loss = compute_loss()
                if new_loss < best_loss:
                    best_loss = new_loss
                else:
                    weights[idx] = old_vals
                    weights.flush()
            
            if step % 10 == 0:
                print(f"    Step {step}: loss = {best_loss:.4f}")
        
        # === YOUR COALITION DETECTION (TUNED) ===
        important_dims = self._find_causal_coalition(free, data, weights)
        
        if len(important_dims) > 0:
            lock_indices = free[important_dims]
            self.storage.lock_task(task_name, lock_indices)
            print(f"  Locked {len(lock_indices)} params (coalition detection)")
        
        del weights
        return {"loss": best_loss, "locked": len(important_dims)}
    
    def _find_causal_coalition(
        self,
        free: np.ndarray,
        data: List[Tuple[np.ndarray, np.ndarray]],
        weights: np.memmap,
        # === TUNED THRESHOLDS ===
        threshold: float = 0.0001,   # Was 0.001 - now 10x more sensitive
        n_samples: int = 100,        # Was 50 - better coverage
        min_lock: int = 50,          # NEW - minimum params to lock
        coalition_size: int = 5,     # Group ablation size
        coalition_samples: int = 20  # Number of coalition tests
    ) -> np.ndarray:
        """
        Your original causal + coalition detection with calibrated thresholds.
        
        1. Individual ablation: zero each dim, measure loss delta
        2. Coalition detection: zero groups of weak dims, find emergent importance
        3. Lock dims that are individually OR collectively important
        """
        
        def compute_loss():
            return np.mean([self.model.loss(inp, tgt) for inp, tgt in data])
        
        base_loss = compute_loss()
        causal_scores = np.zeros(len(free), dtype=np.float32)
        coalition_credits = np.zeros(len(free), dtype=np.float32)
        
        # === INDIVIDUAL ABLATION ===
        test_indices = np.random.choice(len(free), min(n_samples, len(free)), replace=False)
        
        for idx in test_indices:
            dim = free[idx]
            old_val = weights[dim]
            weights[dim] = 0
            weights.flush()
            
            delta = compute_loss() - base_loss
            causal_scores[idx] = delta
            
            weights[dim] = old_val
            weights.flush()
        
        # === COALITION DETECTION ===
        # Find dims with weak individual signal
        weak_mask = (causal_scores > 0) & (causal_scores < threshold)
        weak_candidates = np.where(weak_mask)[0]
        
        if len(weak_candidates) >= coalition_size:
            for _ in range(coalition_samples):
                group = np.random.choice(weak_candidates, 
                                        min(coalition_size, len(weak_candidates)), 
                                        replace=False)
                
                # Ablate group together
                old_vals = weights[free[group]].copy()
                weights[free[group]] = 0
                weights.flush()
                
                group_delta = compute_loss() - base_loss
                
                # If group ablation hurts more than sum of individuals -> emergent importance
                individual_sum = np.sum(causal_scores[group])
                if group_delta > individual_sum * 1.5:  # 50% synergy threshold
                    coalition_credits[group] += 1
                
                weights[free[group]] = old_vals
                weights.flush()
        
        # === DETERMINE WHAT TO LOCK ===
        # Lock if: individual importance OR coalition membership
        important_individual = causal_scores > threshold
        important_coalition = coalition_credits >= 2
        important_mask = important_individual | important_coalition
        
        important_indices = np.where(important_mask)[0]
        
        # Fallback: if too few found, take top-N by causal score
        if len(important_indices) < min_lock:
            positive = np.where(causal_scores > 0)[0]
            if len(positive) > 0:
                sorted_by_score = positive[np.argsort(causal_scores[positive])]
                important_indices = sorted_by_score[-min_lock:]
        
        return important_indices

# =============================================================================
# TASKS
# =============================================================================

class LanguageTask:
    def __init__(self, name: str, texts: List[str]):
        self.name = name
        self.texts = texts
    
    def get_data(self):
        data = []
        for text in self.texts:
            ids = np.array([ord(c) for c in text], dtype=np.int32)
            if len(ids) > 2:
                data.append((ids[:-1], ids[1:]))
        return data


def get_example_tasks():
    return [
        LanguageTask("style_formal", [
            "I am writing to inform you of the recent developments.",
            "Please be advised that the meeting has been rescheduled.",
            "We acknowledge receipt of your correspondence.",
            "It is our understanding that you require assistance.",
            "Kindly note that the deadline has been extended.",
        ]),
        LanguageTask("style_casual", [
            "Hey! Just wanted to check in with you.",
            "So I was thinking we could grab lunch tomorrow?",
            "That sounds awesome, let's do it!",
            "No worries, we can figure it out later.",
            "Cool, catch you later then!",
        ]),
        LanguageTask("qa_technical", [
            "Q: What is a neural network?\nA: A computational model inspired by biological neurons.",
            "Q: What is backpropagation?\nA: An algorithm for computing gradients in neural networks.",
            "Q: What is overfitting?\nA: When a model memorizes training data instead of learning.",
        ]),
        LanguageTask("qa_personal", [
            "Q: What is your favorite color?\nA: My favorite color is blue.",
            "Q: Where do you live?\nA: I live in San Francisco.",
            "Q: What do you do?\nA: I work as a software engineer.",
        ]),
        LanguageTask("code_python", [
            "def hello():\n    print('Hello, world!')\n",
            "class MyClass:\n    def __init__(self):\n        self.value = 0\n",
            "for i in range(10):\n    print(i)\n",
        ]),
        LanguageTask("summarize_legal", [
            "Document: The party of the first part agrees to indemnify.\nSummary: Party A protects Party B.",
            "Document: This agreement shall be governed by California law.\nSummary: California law applies.",
        ]),
    ]

# =============================================================================
# STRESS TEST
# =============================================================================

def run_stress_test(base_path: str = "./sgm_weights", fast: bool = False):
    print("="*70)
    print("SGM TRANSFORMER - COALITION LOCKING STRESS TEST")
    print("="*70)
    
    # Clean start
    import shutil
    if Path(base_path).exists():
        shutil.rmtree(base_path)
    
    cfg = TransformerConfig()
    storage = StorageManager(base_path, dtype=cfg.dtype)
    model = SGMTransformer(cfg, storage)
    trainer = SGMTrainer(model, storage)
    
    print(f"\nConfig: {cfg.d_model}d, {cfg.n_heads}h, {cfg.n_layers}L, {cfg.dtype.__name__}")
    print(f"Total params: {cfg.total_params:,}")
    
    tasks = get_example_tasks()
    n_steps = 30 if fast else 50
    pop_size = 3 if fast else 5
    
    print(f"\n{'='*70}")
    print("PHASE 1: TRAINING")
    print("="*70)
    
    task_losses = {}
    
    for i, task in enumerate(tasks):
        print(f"\n[Task {i+1}/{len(tasks)}] {task.name}")
        data = task.get_data()
        result = trainer.train_task(task.name, data, n_steps=n_steps, population_size=pop_size)
        task_losses[task.name] = result["loss"]
        
        stats = storage.get_stats()
        print(f"  Final: {result['loss']:.4f}, Locked: {stats['locked']:,} ({stats['pct']:.1f}%)")
    
    print(f"\n{'='*70}")
    print("PHASE 2: RETENTION")
    print("="*70)
    
    print(f"\n{'Task':<20} | {'After':>10} | {'Now':>10} | {'Retention':>10}")
    print("-"*55)
    
    for task in tasks:
        data = task.get_data()
        current_loss = np.mean([model.loss(d[0], d[1]) for d in data])
        after_loss = task_losses[task.name]
        
        if after_loss > 0 and after_loss != float('inf'):
            retention = current_loss / after_loss
        else:
            retention = 0
        
        print(f"{task.name:<20} | {after_loss:>10.4f} | {current_loss:>10.4f} | {retention:>9.2f}x")
    
    print(f"\n{'='*70}")
    print("PHASE 3: STORAGE")
    print("="*70)
    
    stats = storage.get_stats()
    print(f"\n  Total params: {stats['total']:,}")
    print(f"  Locked: {stats['locked']:,} ({stats['pct']:.1f}%)")
    print(f"  Free: {stats['free']:,}")
    print(f"  Tasks: {stats['n_tasks']}")
    
    print(f"\n{'='*70}")
    print("PHASE 4: INFERENCE")
    print("="*70)
    
    prompt = np.array([ord(c) for c in "Hello, "], dtype=np.int32)
    times = []
    for _ in range(10):
        start = time.time()
        _ = model.forward(prompt)
        times.append(time.time() - start)
    
    print(f"\n  Forward pass: {np.mean(times)*1000:.2f}ms")
    print(f"  Scales with tasks: NO")
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print("="*70)

# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stress', action='store_true')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--path', default='./sgm_weights')
    args = parser.parse_args()
    
    if args.stress:
        run_stress_test(args.path, args.fast)
    else:
        print("Usage: python sgm_transformer_v2.py --stress [--fast]")