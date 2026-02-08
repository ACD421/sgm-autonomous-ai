#!/usr/bin/env python3
"""
SGM PERSONAL AI - UNIFIED
=========================
Self-improving AI with integrated memory routing.

Architecture:
  [Memory Router] → External JSON store (facts, episodes)
  [SGM Blocks]    → Weight-based learning (reasoning, coding, style)
  [Anchored Base] → Frozen coordinate system

Memory tasks use external store, NOT model weights.
All other tasks use SGM block locking.

Usage:
  python sgm_ai.py --run --iterations 100   # Self-improvement
  python sgm_ai.py --chat                   # Interactive with memory
  python sgm_ai.py --status                 # Show state
  python sgm_ai.py --reset                  # Fresh start
"""

import numpy as np
import json
import time
import shutil
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    # Model
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 2048
    vocab_size: int = 256
    max_seq_len: int = 512
    
    # SGM substrate
    block_size: int = 64
    dtype: np.dtype = np.float16
    
    # Training
    mutation_rate: float = 0.01
    population_size: int = 5
    steps_per_task: int = 30
    improvement_threshold: float = 0.95
    
    # Memory router
    router_dim: int = 256
    router_hidden: int = 64
    trust_threshold: float = 0.3
    daily_write_budget: int = 100
    
    @property
    def d_head(self): return self.d_model // self.n_heads


# =============================================================================
# EXTERNAL MEMORY STORE
# =============================================================================

class MemoryStore:
    """External key-value memory - NOT in model weights"""
    
    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.store_file = self.path / "memory.json"
        self.memories = self._load()
    
    def _load(self) -> Dict:
        if self.store_file.exists():
            return json.load(open(self.store_file))
        return {"facts": {}, "episodes": [], "skills": {}, 
                "meta": {"created": time.time(), "n_stores": 0, "n_retrievals": 0}}
    
    def save(self):
        json.dump(self.memories, open(self.store_file, 'w'), indent=2)
    
    def _hash(self, text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()[:16]
    
    def store_fact(self, key: str, value: str, confidence: float = 1.0):
        self.memories["facts"][self._hash(key)] = {
            "key": key, "value": value, "confidence": confidence,
            "timestamp": time.time(), "access_count": 0
        }
        self.memories["meta"]["n_stores"] += 1
        self.save()
    
    def store_episode(self, content: str, context: str = ""):
        self.memories["episodes"].append({
            "content": content, "context": context, 
            "timestamp": time.time(), "importance": 1.0
        })
        if len(self.memories["episodes"]) > 1000:
            self.memories["episodes"] = self.memories["episodes"][-1000:]
        self.memories["meta"]["n_stores"] += 1
        self.save()
    
    def retrieve_fact(self, query: str, threshold: float = 0.3) -> Optional[Dict]:
        h = self._hash(query)
        if h in self.memories["facts"]:
            self.memories["meta"]["n_retrievals"] += 1
            return self.memories["facts"][h]
        
        # Fuzzy match
        query_words = set(query.lower().split())
        best, best_score = None, 0
        for fact in self.memories["facts"].values():
            fact_words = set(fact["key"].lower().split())
            if fact_words:
                score = len(query_words & fact_words) / len(query_words | fact_words)
                if score > best_score and score >= threshold:
                    best, best_score = fact, score
        if best:
            self.memories["meta"]["n_retrievals"] += 1
        return best
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        results = []
        query_words = set(query.lower().split())
        
        for fact in self.memories["facts"].values():
            words = set(fact["key"].lower().split()) | set(fact["value"].lower().split())
            if query_words & words:
                score = len(query_words & words) / len(query_words)
                results.append({"type": "fact", "score": score, "data": fact})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def forget(self, query: str) -> bool:
        h = self._hash(query)
        if h in self.memories["facts"]:
            del self.memories["facts"][h]
            self.save()
            return True
        return False
    
    def list_facts(self, limit: int = 20) -> List[Dict]:
        return list(self.memories["facts"].values())[:limit]
    
    def stats(self) -> Dict:
        return {
            "n_facts": len(self.memories["facts"]),
            "n_episodes": len(self.memories["episodes"]),
            "n_stores": self.memories["meta"]["n_stores"],
            "n_retrievals": self.memories["meta"]["n_retrievals"]
        }


# =============================================================================
# MEMORY ROUTER
# =============================================================================

class MemoryRouter:
    """Tiny module that decides store/retrieve/trust"""
    
    def __init__(self, cfg: Config, path: Path):
        self.cfg = cfg
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        
        self.weights_file = self.path / "router.npz"
        self.memory = MemoryStore(self.path / "store")
        
        self._daily_writes = 0
        self._last_write_day = None
        
        if self.weights_file.exists():
            self._load()
        else:
            self._init()
    
    def _init(self):
        d, h = self.cfg.router_dim, self.cfg.router_hidden
        self.W_store = np.random.randn(d, h).astype(np.float32) * 0.1
        self.b_store = np.zeros(h, dtype=np.float32)
        self.W_store_out = np.random.randn(h, 1).astype(np.float32) * 0.1
        
        self.W_retrieve = np.random.randn(d, h).astype(np.float32) * 0.1
        self.b_retrieve = np.zeros(h, dtype=np.float32)
        self.W_retrieve_out = np.random.randn(h, 1).astype(np.float32) * 0.1
        
        self.W_trust = np.random.randn(d, h).astype(np.float32) * 0.1
        self.b_trust = np.zeros(h, dtype=np.float32)
        self.W_trust_out = np.random.randn(h, 1).astype(np.float32) * 0.1
        
        self.n_params = sum(w.size for w in [
            self.W_store, self.b_store, self.W_store_out,
            self.W_retrieve, self.b_retrieve, self.W_retrieve_out,
            self.W_trust, self.b_trust, self.W_trust_out
        ])
        self._save()
        self._train_initial()
    
    def _save(self):
        np.savez(self.weights_file,
            W_store=self.W_store, b_store=self.b_store, W_store_out=self.W_store_out,
            W_retrieve=self.W_retrieve, b_retrieve=self.b_retrieve, W_retrieve_out=self.W_retrieve_out,
            W_trust=self.W_trust, b_trust=self.b_trust, W_trust_out=self.W_trust_out)
    
    def _load(self):
        d = np.load(self.weights_file)
        self.W_store, self.b_store, self.W_store_out = d["W_store"], d["b_store"], d["W_store_out"]
        self.W_retrieve, self.b_retrieve, self.W_retrieve_out = d["W_retrieve"], d["b_retrieve"], d["W_retrieve_out"]
        self.W_trust, self.b_trust, self.W_trust_out = d["W_trust"], d["b_trust"], d["W_trust_out"]
        self.n_params = sum(w.size for w in [
            self.W_store, self.b_store, self.W_store_out,
            self.W_retrieve, self.b_retrieve, self.W_retrieve_out,
            self.W_trust, self.b_trust, self.W_trust_out
        ])
    
    def _embed(self, text: str) -> np.ndarray:
        d = self.cfg.router_dim
        unigram = np.zeros(128, dtype=np.float32)
        for c in text.lower():
            unigram[ord(c) % 128] += 1
        if unigram.sum() > 0: unigram /= unigram.sum()
        
        bigram = np.zeros(64, dtype=np.float32)
        tl = text.lower()
        for i in range(len(tl) - 1):
            bigram[hash(tl[i:i+2]) % 64] += 1
        if bigram.sum() > 0: bigram /= bigram.sum()
        
        feat = np.zeros(64, dtype=np.float32)
        if any(text.lower().startswith(q) for q in ["what", "who", "where", "when", "how", "why"]):
            feat[0] = 1.0
        if "?" in text: feat[1] = 1.0
        if "remember" in text.lower(): feat[2] = 1.0
        if "forget" in text.lower(): feat[3] = 1.0
        if "my " in text.lower() or "i am" in text.lower(): feat[4] = 1.0
        if " is " in text.lower(): feat[5] = 1.0
        
        return np.concatenate([unigram, bigram, feat])[:d]
    
    def _sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    def _relu(self, x): return np.maximum(0, x)
    
    def should_store(self, text: str) -> Tuple[bool, float]:
        x = self._embed(text)
        h = self._relu(x @ self.W_store + self.b_store)
        s = self._sigmoid(h @ self.W_store_out)[0]
        return s > 0.5, float(s)
    
    def should_retrieve(self, text: str) -> Tuple[bool, float]:
        x = self._embed(text)
        h = self._relu(x @ self.W_retrieve + self.b_retrieve)
        s = self._sigmoid(h @ self.W_retrieve_out)[0]
        return s > 0.5, float(s)
    
    def get_trust(self, text: str) -> float:
        x = self._embed(text)
        h = self._relu(x @ self.W_trust + self.b_trust)
        return float(self._sigmoid(h @ self.W_trust_out)[0])
    
    def _train_initial(self):
        """Train router on basic patterns"""
        data = [
            ("My name is Alice", "store", True), ("I live in Texas", "store", True),
            ("Remember my birthday", "store", True), ("Hello", "store", False),
            ("What is my name?", "retrieve", True), ("Where do I live?", "retrieve", True),
            ("Write a poem", "retrieve", False), ("Help me code", "retrieve", False),
            ("What is my name?", "trust", True), ("Who am I?", "trust", True),
            ("Write a story", "trust", False), ("Be creative", "trust", False),
        ]
        for _ in range(100):
            np.random.shuffle(data)
            for text, task, target in data:
                x = self._embed(text)
                t = float(target)
                lr = 0.05
                
                if task == "store":
                    h = self._relu(x @ self.W_store + self.b_store)
                    p = self._sigmoid(h @ self.W_store_out)[0]
                    g = (p - t) * p * (1 - p)
                    self.W_store_out -= lr * np.outer(h, [g])
                    gh = g * self.W_store_out.flatten() * (h > 0)
                    self.W_store -= lr * np.outer(x, gh)
                    self.b_store -= lr * gh
                elif task == "retrieve":
                    h = self._relu(x @ self.W_retrieve + self.b_retrieve)
                    p = self._sigmoid(h @ self.W_retrieve_out)[0]
                    g = (p - t) * p * (1 - p)
                    self.W_retrieve_out -= lr * np.outer(h, [g])
                    gh = g * self.W_retrieve_out.flatten() * (h > 0)
                    self.W_retrieve -= lr * np.outer(x, gh)
                    self.b_retrieve -= lr * gh
                elif task == "trust":
                    h = self._relu(x @ self.W_trust + self.b_trust)
                    p = self._sigmoid(h @ self.W_trust_out)[0]
                    g = (p - t) * p * (1 - p)
                    self.W_trust_out -= lr * np.outer(h, [g])
                    gh = g * self.W_trust_out.flatten() * (h > 0)
                    self.W_trust -= lr * np.outer(x, gh)
                    self.b_trust -= lr * gh
        self._save()
    
    def check_write_budget(self) -> bool:
        today = time.strftime("%Y-%m-%d")
        if self._last_write_day != today:
            self._last_write_day = today
            self._daily_writes = 0
        return self._daily_writes < self.cfg.daily_write_budget
    
    def process_input(self, text: str) -> Dict:
        should_ret, score = self.should_retrieve(text)
        trust = self.get_trust(text) if should_ret else 0.0
        memories = self.memory.search(text) if should_ret else []
        return {"retrieve": should_ret, "score": score, "trust": trust, "memories": memories}
    
    def process_output(self, inp: str, out: str):
        if not self.check_write_budget():
            return
        should_st, score = self.should_store(inp + " " + out)
        if should_st and score > 0.6:
            if "?" in inp or any(inp.lower().startswith(q) for q in ["what", "who", "where", "when"]):
                self.memory.store_fact(inp, out)
            else:
                self.memory.store_episode(out, context=inp)
            self._daily_writes += 1


# =============================================================================
# BLOCK STORAGE
# =============================================================================

class BlockStorage:
    def __init__(self, path: Path, cfg: Config):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        
        self.weights_file = self.path / "weights.mmap"
        self.index_file = self.path / "index.json"
        self.index = self._load_index()
    
    def _load_index(self):
        if self.index_file.exists():
            return json.load(open(self.index_file))
        return {"total_params": 0, "block_size": self.cfg.block_size, 
                "locked_blocks": [], "tasks": [], "iteration": 0}
    
    def _save_index(self):
        json.dump(self.index, open(self.index_file, 'w'), indent=2)
    
    def init(self, n_params: int):
        self.index["total_params"] = n_params
        w = np.random.randn(n_params).astype(self.cfg.dtype) * 0.02
        fp = np.memmap(self.weights_file, dtype=self.cfg.dtype, mode='w+', shape=(n_params,))
        fp[:] = w
        fp.flush()
        del fp
        self._save_index()
        print(f"[INIT] {n_params:,} params, {n_params // self.cfg.block_size} blocks")
    
    def get_weights(self, mode='r+'):
        return np.memmap(self.weights_file, dtype=self.cfg.dtype, mode=mode,
                        shape=(self.index["total_params"],))
    
    def get_free_blocks(self):
        n = self.index["total_params"] // self.cfg.block_size
        return np.array(sorted(set(range(n)) - set(self.index["locked_blocks"])), dtype=np.int64)
    
    def lock_blocks(self, name: str, blocks):
        new = [int(b) for b in blocks if b not in self.index["locked_blocks"]]
        self.index["locked_blocks"].extend(new)
        self.index["tasks"].append({"name": name, "blocks": new, "time": time.time()})
        self._save_index()
        return len(new)
    
    def lock_range(self, name: str, start: int, end: int):
        bs = self.cfg.block_size
        blocks = list(range(start // bs, (end + bs - 1) // bs))
        return self.lock_blocks(name, blocks)
    
    def stats(self):
        total = self.index["total_params"]
        bs = self.cfg.block_size
        n_blocks = total // bs
        locked = len(self.index["locked_blocks"])
        return {
            "total_params": total, "total_blocks": n_blocks,
            "locked_blocks": locked, "locked_params": locked * bs,
            "free_blocks": n_blocks - locked,
            "pct_locked": locked / n_blocks * 100 if n_blocks else 0,
            "n_tasks": len(self.index["tasks"]), "iteration": self.index["iteration"]
        }


# =============================================================================
# TRANSFORMER
# =============================================================================

class Transformer:
    def __init__(self, cfg: Config, storage: BlockStorage):
        self.cfg = cfg
        self.storage = storage
        self._build_map()
        if storage.index["total_params"] == 0:
            storage.init(self.total_params)
    
    def _build_map(self):
        cfg = self.cfg
        self.map = {}
        off = 0
        
        self.map["embed"] = (off, off + cfg.vocab_size * cfg.d_model)
        off += cfg.vocab_size * cfg.d_model
        
        for L in range(cfg.n_layers):
            for n in ["q", "k", "v", "o"]:
                self.map[f"L{L}_{n}"] = (off, off + cfg.d_model * cfg.d_model)
                off += cfg.d_model * cfg.d_model
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
    
    def anchor(self):
        print("[ANCHOR] Locking coordinate systems...")
        s, e = self.map["embed"]
        self.storage.lock_range("ANCHOR_embed", s, e)
        for L in range(self.cfg.n_layers):
            for ln in [f"L{L}_ln1", f"L{L}_ln2"]:
                s, e = self.map[ln]
                self.storage.lock_range(f"ANCHOR_{ln}_scale", s, s + self.cfg.d_model)
        print(f"  Anchored: {self.storage.stats()['pct_locked']:.1f}%")
    
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
            x = x + self._mha(q, k, v) @ Wo
            
            ln2 = self._get(w, f"L{L}_ln2").reshape(2, cfg.d_model).astype(np.float32)
            xn = self._ln(x, ln2[0], ln2[1])
            
            ff1 = self._get(w, f"L{L}_ff1").reshape(cfg.d_model, cfg.d_ff).astype(np.float32)
            ff2 = self._get(w, f"L{L}_ff2").reshape(cfg.d_ff, cfg.d_model).astype(np.float32)
            x = x + self._gelu(xn @ ff1) @ ff2
        
        out = self._get(w, "output").reshape(cfg.d_model, cfg.vocab_size).astype(np.float32)
        del w
        return x @ out
    
    def _ln(self, x, g, b, eps=1e-5):
        m, v = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
        return g * (x - m) / np.sqrt(v + eps) + b
    
    def _mha(self, q, k, v):
        cfg = self.cfg
        seq = q.shape[0]
        q = q.reshape(seq, cfg.n_heads, cfg.d_head)
        k = k.reshape(seq, cfg.n_heads, cfg.d_head)
        v = v.reshape(seq, cfg.n_heads, cfg.d_head)
        
        scores = np.einsum('ihd,jhd->hij', q, k) / np.sqrt(cfg.d_head)
        scores = scores + np.triu(np.ones((seq, seq)), 1) * -1e9
        attn = np.exp(scores - scores.max(-1, keepdims=True))
        attn = attn / (attn.sum(-1, keepdims=True) + 1e-9)
        return np.einsum('hij,jhd->ihd', attn, v).reshape(seq, cfg.d_model)
    
    def _gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def loss(self, ids, targets):
        logits = self.forward(ids)
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs = probs / (probs.sum(-1, keepdims=True) + 1e-9)
        return -np.mean(np.log(probs[np.arange(len(targets)), targets] + 1e-9))
    
    def generate(self, prompt: str, max_tokens: int = 100, temp: float = 0.8) -> str:
        ids = [min(ord(c), 255) for c in prompt]
        for _ in range(max_tokens):
            inp = np.array(ids[-self.cfg.max_seq_len:], dtype=np.int32)
            logits = self.forward(inp)[-1] / temp
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            ids.append(np.random.choice(len(probs), p=probs))
            if ids[-1] == ord('\n') and len(ids) > len(prompt) + 10:
                break
        return ''.join(chr(min(max(0, i), 127)) for i in ids)


# =============================================================================
# UNIFIED AI (model + memory)
# =============================================================================

class SGMAI:
    """Unified AI with SGM substrate + memory routing"""
    
    def __init__(self, path: str = "./sgm_ai"):
        self.path = Path(path)
        self.cfg = Config()
        
        self.storage = BlockStorage(self.path, self.cfg)
        self.model = Transformer(self.cfg, self.storage)
        self.router = MemoryRouter(self.cfg, self.path / "memory")
        
        # Anchor on first run
        if self.storage.index["iteration"] == 0 and len(self.storage.index["tasks"]) == 0:
            self.model.anchor()
    
    def _handle_command(self, prompt: str) -> Optional[str]:
        """Handle explicit memory commands"""
        p = prompt.lower().strip()
        
        if p.startswith("remember that ") or p.startswith("remember: "):
            fact = prompt[prompt.find(" ", 9) + 1:].strip()
            if " is " in fact:
                parts = fact.split(" is ", 1)
                self.router.memory.store_fact(parts[0], parts[1])
                return f"✓ Remembered: {parts[0]} is {parts[1]}"
            self.router.memory.store_episode(fact)
            return f"✓ Remembered: {fact}"
        
        if p.startswith("forget "):
            q = prompt[7:].strip()
            if self.router.memory.forget(q):
                return f"✓ Forgot: {q}"
            return f"✗ No memory for: {q}"
        
        if p in ["list memories", "what do you know", "show memories"]:
            facts = self.router.memory.list_facts()
            if not facts:
                return "I don't have any memories yet."
            lines = ["Here's what I remember:"]
            for f in facts[:15]:
                lines.append(f"  • {f['key']} → {f['value']}")
            return "\n".join(lines)
        
        return None
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        # Check commands
        cmd = self._handle_command(prompt)
        if cmd:
            return cmd
        
        # Check memory
        mem = self.router.process_input(prompt)
        
        aug_prompt = prompt
        if mem["retrieve"] and mem["memories"] and mem["trust"] >= self.cfg.trust_threshold:
            ctx = "\n".join(f"[Memory]: {m['data']['value']}" for m in mem["memories"][:2])
            aug_prompt = f"{ctx}\n\n{prompt}"
        
        # Generate
        output = self.model.generate(aug_prompt, max_tokens=max_tokens)
        if aug_prompt in output:
            output = output[len(aug_prompt):].strip()
        
        # Maybe store
        self.router.process_output(prompt, output)
        
        return output
    
    def train_task(self, name: str, data: List[Tuple], steps: int = 30) -> Dict:
        """Train on a task using SGM blocks"""
        w = self.storage.get_weights('r+')
        free = self.storage.get_free_blocks()
        
        if len(free) == 0:
            del w
            return {"loss": float('inf'), "locked": 0, "improved": False}
        
        bs = self.cfg.block_size
        free_params = np.array([i for b in free for i in range(b * bs, (b + 1) * bs)], dtype=np.int64)
        
        def loss():
            return np.mean([self.model.loss(d[0], d[1]) for d in data])
        
        init_loss = loss()
        best = init_loss
        
        for step in range(steps):
            for _ in range(self.cfg.population_size):
                n = min(len(free_params) // 10 + 1, len(free_params))
                idx = np.random.choice(free_params, n, replace=False)
                old = w[idx].copy()
                w[idx] += np.random.randn(n).astype(w.dtype) * self.cfg.mutation_rate
                new = loss()
                if new < best:
                    best = new
                else:
                    w[idx] = old
            if step % 5 == 0:
                w.flush()
        
        w.flush()
        improved = best < init_loss * self.cfg.improvement_threshold
        
        locked = 0
        if improved:
            # Find important blocks
            importance = np.zeros(len(free), dtype=np.float32)
            test_idx = np.random.choice(len(free), min(20, len(free)), replace=False)
            base = loss()
            for i in test_idx:
                block = free[i]
                s, e = block * bs, (block + 1) * bs
                old = w[s:e].copy()
                w[s:e] *= 0.1
                importance[i] = loss() - base
                w[s:e] = old
            
            pos = np.where(importance > 0)[0]
            if len(pos) > 0:
                locked = self.storage.lock_blocks(name, free[pos])
        
        del w
        return {"loss": best, "init": init_loss, "improve": (init_loss - best) / init_loss * 100, 
                "locked": locked, "improved": improved}
    
    def stats(self) -> Dict:
        s = self.storage.stats()
        m = self.router.memory.stats()
        return {**s, "memory": m, "router_params": self.router.n_params}


# =============================================================================
# TRAINING LOOP
# =============================================================================

def build_tasks():
    def enc(s): return np.array([min(ord(c), 255) for c in s], dtype=np.int32)
    def make(texts): return [(enc(t)[:-1], enc(t)[1:]) for t in texts if len(t) > 2]
    
    return {
        "reasoning": [
            ("logic", make(["If A then B. A is true. Therefore B.", "All X are Y. Z is X. Therefore Z is Y."])),
            ("compare", make(["5 > 3. 3 > 1. Therefore 5 > 1.", "A before B. B before C. A before C."])),
        ],
        "coding": [
            ("function", make(["def add(a,b):\n    return a+b\n", "def mul(x,y):\n    return x*y\n"])),
            ("loop", make(["for i in range(10):\n    print(i)\n", "while x>0:\n    x-=1\n"])),
        ],
        "style": [
            ("formal", make(["I am writing to inform you.", "Please be advised.", "We acknowledge."])),
            ("concise", make(["Yes.", "No.", "Done.", "OK.", "Confirmed."])),
        ],
        "knowledge": [
            ("math", make(["d/dx x^2 = 2x", "Pi = 3.14159", "e = 2.71828"])),
            ("science", make(["Water boils at 100C.", "Light speed = 3e8 m/s."])),
        ],
    }


def run_training(ai: SGMAI, iterations: int):
    print("\n" + "="*70)
    print("SGM SELF-IMPROVEMENT")
    print("="*70)
    
    tasks = build_tasks()
    all_tasks = [(cat, name, data) for cat, items in tasks.items() for name, data in items]
    
    # Initial eval
    def eval_cat(cat):
        total, n = 0, 0
        for name, data in tasks.get(cat, []):
            if data:
                total += np.mean([ai.model.loss(d[0], d[1]) for d in data])
                n += 1
        return total / n if n else float('inf')
    
    initial = {cat: eval_cat(cat) for cat in tasks}
    print("\n[INITIAL]")
    for cat, loss in initial.items():
        print(f"  {cat}: {loss:.4f}")
    
    for it in range(iterations):
        ai.storage.index["iteration"] = it
        ai.storage._save_index()
        
        # Select task
        idx = it % len(all_tasks)
        if np.random.random() < 0.3:
            idx = np.random.randint(len(all_tasks))
        
        cat, name, data = all_tasks[idx]
        if not data:
            continue
        
        result = ai.train_task(f"{cat}_{name}_i{it}", data)
        
        if it % 25 == 0 or result["improved"]:
            stats = ai.storage.stats()
            print(f"\n[{it}] {cat}/{name}: {result['init']:.3f}→{result['loss']:.3f} ({result['improve']:.1f}%)")
            print(f"  Locked: {stats['pct_locked']:.1f}%", end="")
            if result["improved"]:
                print(f" ✓ +{result['locked']} blocks")
            else:
                print()
        
        if stats['pct_locked'] > 90:
            print("\n[STOP] Capacity saturated")
            break
    
    # Final eval
    print("\n" + "="*70)
    print("FINAL")
    print("="*70)
    
    final = {cat: eval_cat(cat) for cat in tasks}
    print(f"\n{'Category':<12} | {'Initial':>8} | {'Final':>8} | {'Change':>8}")
    print("-"*45)
    for cat in tasks:
        i, f = initial[cat], final[cat]
        ch = (i - f) / i * 100
        print(f"{cat:<12} | {i:>8.3f} | {f:>8.3f} | {ch:>+7.1f}%")
    
    stats = ai.stats()
    print(f"\nParams: {stats['total_params']:,} | Locked: {stats['pct_locked']:.1f}%")
    print(f"Memory: {stats['memory']['n_facts']} facts, {stats['memory']['n_episodes']} episodes")


def interactive(ai: SGMAI):
    print("\n" + "="*70)
    print("SGM AI - Chat")
    print("Commands: 'quit', 'status', 'remember that...', 'forget...', 'list memories'")
    print("="*70 + "\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            if prompt.lower() == 'quit':
                break
            if prompt.lower() == 'status':
                s = ai.stats()
                print(f"\n[STATUS] Params: {s['total_params']:,}, Locked: {s['pct_locked']:.1f}%")
                print(f"  Memory: {s['memory']['n_facts']} facts, {s['memory']['n_episodes']} episodes\n")
                continue
            
            response = ai.generate(prompt, max_tokens=100)
            print(f"AI: {response}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SGM Personal AI')
    parser.add_argument('--run', action='store_true', help='Run self-improvement')
    parser.add_argument('--chat', action='store_true', help='Interactive chat')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--reset', action='store_true', help='Reset everything')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--path', default='./sgm_ai')
    args = parser.parse_args()
    
    if args.reset and Path(args.path).exists():
        shutil.rmtree(args.path)
        print("[RESET] Cleared")
    
    ai = SGMAI(args.path)
    
    if args.status:
        s = ai.stats()
        print("\n=== SGM AI STATUS ===")
        print(f"Total params: {s['total_params']:,}")
        print(f"Locked: {s['locked_params']:,} ({s['pct_locked']:.1f}%)")
        print(f"Tasks: {s['n_tasks']}")
        print(f"Iterations: {s['iteration']}")
        print(f"Memory: {s['memory']['n_facts']} facts, {s['memory']['n_episodes']} episodes")
        print(f"Router: {s['router_params']:,} params")
        
    elif args.run:
        run_training(ai, args.iterations)
        
    elif args.chat:
        interactive(ai)
        
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python sgm_ai.py --run --iterations 100")
        print("  python sgm_ai.py --chat")
        print("  python sgm_ai.py --status")


if __name__ == "__main__":
    main()