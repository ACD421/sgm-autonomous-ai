#!/usr/bin/env python3
"""
SGM PERSONAL AI MEMORY - KILLER DEMO
=====================================
Block-level locking + 100 sequential tasks.

Changes:
- Lock BLOCKS (32 params) instead of scalars
- 100 micro-tasks (style, facts, preferences, formatting)
- Measure task #1 after task #100
- Track locked params + inference latency

Success criteria:
- Retention ~= 1.0x
- Locked params << total params  
- Inference time unchanged

Usage:
  python sgm_personal_ai.py --demo          # Full 100-task demo
  python sgm_personal_ai.py --demo --quick  # 25-task quick test
"""

import numpy as np
import json
import time
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import argparse

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 1024
    vocab_size: int = 256
    block_size: int = 32  # Lock in blocks of 32 params
    dtype: np.dtype = np.float16
    
    @property
    def d_head(self): return self.d_model // self.n_heads
    
    @property
    def total_params(self):
        embed = self.vocab_size * self.d_model
        per_layer = self.d_model * self.d_model * 4 + self.d_model * self.d_ff * 2 + self.d_model * 4
        output = self.d_model * self.vocab_size
        return embed + per_layer * self.n_layers + output
    
    @property
    def n_blocks(self):
        return self.total_params // self.block_size

# =============================================================================
# STORAGE (block-level)
# =============================================================================

class BlockStorage:
    """Block-level locking storage"""
    
    def __init__(self, path: str, cfg: Config):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.dtype = cfg.dtype
        
        self.weights_file = self.path / "weights.mmap"
        self.index_file = self.path / "index.json"
        
        self.index = self._load_index()
    
    def _load_index(self):
        if self.index_file.exists():
            return json.load(open(self.index_file))
        return {
            "total_params": 0,
            "block_size": self.cfg.block_size,
            "locked_blocks": [],  # List of block indices
            "tasks": []
        }
    
    def _save_index(self):
        json.dump(self.index, open(self.index_file, 'w'))
    
    def init(self, n_params: int):
        self.index["total_params"] = n_params
        w = np.random.randn(n_params).astype(self.dtype) * 0.02
        fp = np.memmap(self.weights_file, dtype=self.dtype, mode='w+', shape=(n_params,))
        fp[:] = w
        fp.flush()
        del fp
        self._save_index()
        mb = n_params * 2 / 1024 / 1024
        print(f"Init: {n_params:,} params, {n_params // self.cfg.block_size} blocks, {mb:.1f}MB")
    
    def get_weights(self, mode='r+'):
        return np.memmap(self.weights_file, dtype=self.dtype, mode=mode,
                        shape=(self.index["total_params"],))
    
    def get_free_blocks(self) -> np.ndarray:
        """Return indices of unlocked blocks"""
        n_blocks = self.index["total_params"] // self.cfg.block_size
        all_blocks = set(range(n_blocks))
        locked = set(self.index["locked_blocks"])
        free = sorted(all_blocks - locked)
        return np.array(free, dtype=np.int64)
    
    def get_free_params(self) -> np.ndarray:
        """Return indices of all params in unlocked blocks"""
        free_blocks = self.get_free_blocks()
        bs = self.cfg.block_size
        params = []
        for b in free_blocks:
            params.extend(range(b * bs, (b + 1) * bs))
        return np.array(params, dtype=np.int64)
    
    def lock_blocks(self, task_name: str, block_indices: np.ndarray):
        """Lock specific blocks for a task"""
        new_locks = [int(b) for b in block_indices if b not in self.index["locked_blocks"]]
        self.index["locked_blocks"].extend(new_locks)
        self.index["tasks"].append({
            "name": task_name,
            "blocks": new_locks,
            "n_params": len(new_locks) * self.cfg.block_size,
            "time": time.time()
        })
        self._save_index()
        return len(new_locks)
    
    def lock_param_range(self, name: str, start: int, end: int):
        """Lock all blocks covering a parameter range"""
        bs = self.cfg.block_size
        blocks = list(range(start // bs, (end + bs - 1) // bs))
        return self.lock_blocks(name, np.array(blocks, dtype=np.int64))
    
    def stats(self):
        total = self.index["total_params"]
        bs = self.cfg.block_size
        n_blocks = total // bs
        locked_blocks = len(self.index["locked_blocks"])
        locked_params = locked_blocks * bs
        return {
            "total_params": total,
            "total_blocks": n_blocks,
            "locked_blocks": locked_blocks,
            "locked_params": locked_params,
            "free_blocks": n_blocks - locked_blocks,
            "free_params": total - locked_params,
            "pct_locked": locked_params / total * 100 if total else 0,
            "n_tasks": len(self.index["tasks"])
        }

# =============================================================================
# TRANSFORMER
# =============================================================================

class Transformer:
    def __init__(self, cfg: Config, storage: BlockStorage):
        self.cfg = cfg
        self.storage = storage
        self._build_router()
        
        if storage.index["total_params"] == 0:
            storage.init(self.total_params)
    
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
    
    def anchor_early_layers(self, seed: int = 42):
        """Lock representation anchors BEFORE any task trains.
        
        Light anchoring - only coordinate systems, not semantics:
        - Embeddings (token coordinate system)
        - Layer 0 LN scale only (not bias) - preserves centering flexibility
        - NO QKV anchoring - allows attention to specialize
        """
        print("\n  Anchoring coordinate systems...")
        
        # Lock embeddings (defines token space)
        s, e = self.map["embed"]
        n = self.storage.lock_param_range("ANCHOR_embed", s, e)
        print(f"    Locked embeddings: {n} blocks")
        
        # Lock only LN scale (gamma), not bias (beta)
        # LN params are packed as [gamma, beta] = [d_model, d_model]
        for ln in ["L0_ln1", "L0_ln2"]:
            s, e = self.map[ln]
            # Only lock first half (gamma/scale)
            scale_end = s + self.cfg.d_model
            n = self.storage.lock_param_range(f"ANCHOR_{ln}_scale", s, scale_end)
            print(f"    Locked {ln} scale: {n} blocks")
        
        # NO QKV anchoring - let attention specialize per task
        
        stats = self.storage.stats()
        print(f"    Total anchored: {stats['locked_params']:,} params ({stats['pct_locked']:.1f}%)")
    
    def _get(self, w, name):
        s, e = self.map[name]
        return w[s:e]
    
    def forward(self, ids):
        cfg = self.cfg
        w = self.storage.get_weights('r')
        
        embed = self._get(w, "embed").reshape(cfg.vocab_size, cfg.d_model)
        x = embed[ids].astype(np.float32)
        
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

# =============================================================================
# BLOCK-LEVEL TRAINER
# =============================================================================

class BlockTrainer:
    def __init__(self, model: Transformer, storage: BlockStorage, cfg: Config):
        self.model = model
        self.storage = storage
        self.cfg = cfg
    
    def train_task(self, name: str, data: List[Tuple], steps: int = 30, pop: int = 3, lr: float = 0.02):
        w = self.storage.get_weights('r+')
        free_blocks = self.storage.get_free_blocks()
        
        if len(free_blocks) == 0:
            del w
            return {"loss": float('inf'), "locked": 0}
        
        # Get free params from free blocks
        bs = self.cfg.block_size
        free_params = []
        for b in free_blocks:
            free_params.extend(range(b * bs, (b + 1) * bs))
        free_params = np.array(free_params, dtype=np.int64)
        
        def loss():
            return np.mean([self.model.loss(d[0], d[1]) for d in data])
        
        best = loss()
        
        for step in range(steps):
            for _ in range(pop):
                # Mutate random subset of free params
                n = min(len(free_params) // 10 + 1, len(free_params))
                idx = np.random.choice(free_params, n, replace=False)
                old = w[idx].copy()
                w[idx] += np.random.randn(n).astype(w.dtype) * lr
                # NO flush here - batched
                
                new = loss()
                if new < best:
                    best = new
                else:
                    w[idx] = old
            
            # Flush once per step, not per mutation
            if step % 5 == 0:
                w.flush()
        
        w.flush()  # Final flush
        
        # Find important BLOCKS via ablation
        important_blocks = self._find_important_blocks(free_blocks, data, w)
        
        if len(important_blocks) > 0:
            n_locked = self.storage.lock_blocks(name, important_blocks)
        else:
            n_locked = 0
        
        del w
        return {"loss": best, "locked": n_locked}
    
    def _find_important_blocks(self, free_blocks: np.ndarray, data: List[Tuple], w: np.memmap,
                                n_test: int = 30, min_lock: int = 1) -> np.ndarray:
        """Ablate blocks with noise (not zeroing), find important ones"""
        bs = self.cfg.block_size
        
        def loss():
            return np.mean([self.model.loss(d[0], d[1]) for d in data])
        
        base = loss()
        importance = np.zeros(len(free_blocks), dtype=np.float32)
        
        # Sample blocks to test
        test_idx = np.random.choice(len(free_blocks), min(n_test, len(free_blocks)), replace=False)
        
        for i in test_idx:
            block = free_blocks[i]
            start, end = block * bs, (block + 1) * bs
            
            old = w[start:end].copy()
            # Use scale attenuation instead of zeroing (less destructive)
            w[start:end] *= 0.1
            
            importance[i] = loss() - base
            
            w[start:end] = old
        
        w.flush()  # Single flush after all ablations
        
        # Return blocks with positive importance (ablation hurts)
        positive = np.where(importance > 0)[0]
        if len(positive) < min_lock:
            # Fallback: top-N
            top = np.argsort(importance)[-min_lock:]
            top = top[importance[top] > 0]
            if len(top) > 0:
                return free_blocks[top]
            return np.array([], dtype=np.int64)
        
        return free_blocks[positive]

# =============================================================================
# 100 MICRO-TASKS
# =============================================================================

def generate_100_tasks():
    """Generate 100 diverse micro-tasks"""
    
    def encode(s): 
        # Clamp to ASCII range
        return np.array([min(ord(c), 255) for c in s], dtype=np.int32)
    
    def make_data(texts): 
        return [(encode(t)[:-1], encode(t)[1:]) for t in texts if len(t) > 2]
    
    tasks = []
    
    # === STYLE TASKS (20) ===
    styles = [
        ("formal", ["I am writing to inform you.", "Please be advised.", "We acknowledge receipt."]),
        ("casual", ["Hey! What's up?", "That's cool!", "No worries!"]),
        ("academic", ["The hypothesis suggests.", "Evidence indicates.", "Further research is needed."]),
        ("poetic", ["The moon rises softly.", "Whispers in the wind.", "Dreams of distant stars."]),
        ("technical", ["Initialize the system.", "Configure parameters.", "Execute the function."]),
        ("friendly", ["Hope you're doing well!", "Great to hear from you!", "Looking forward to it!"]),
        ("assertive", ["This must be done.", "I expect results.", "No excuses."]),
        ("humble", ["I might be wrong.", "Perhaps consider.", "In my humble opinion."]),
        ("excited", ["Amazing news!", "I can't wait!", "This is incredible!"]),
        ("calm", ["Let's take our time.", "No rush needed.", "Everything is fine."]),
        ("professional", ["Per our discussion.", "As agreed upon.", "Moving forward."]),
        ("playful", ["Guess what!", "You won't believe it!", "Here's a fun fact!"]),
        ("serious", ["This is important.", "Pay attention.", "Critical matter."]),
        ("encouraging", ["You can do it!", "Keep going!", "Almost there!"]),
        ("apologetic", ["I'm sorry for.", "Please forgive me.", "My apologies."]),
        ("grateful", ["Thank you so much!", "I appreciate it.", "Grateful for your help."]),
        ("curious", ["I wonder why.", "How does that work?", "What if we tried?"]),
        ("confident", ["I know this works.", "Trust me on this.", "Absolutely certain."]),
        ("cautious", ["We should be careful.", "Let's consider risks.", "Proceed with caution."]),
        ("direct", ["Do this now.", "Here's the answer.", "Simple as that."]),
    ]
    
    for name, texts in styles:
        tasks.append((f"style_{name}", make_data(texts)))
    
    # === FACT TASKS (20) ===
    facts = [
        ("color", ["Q: Favorite color?\nA: Blue."]),
        ("food", ["Q: Favorite food?\nA: Pizza."]),
        ("city", ["Q: Where from?\nA: San Francisco."]),
        ("job", ["Q: Your job?\nA: Engineer."]),
        ("pet", ["Q: Have pets?\nA: A cat named Max."]),
        ("hobby", ["Q: Hobbies?\nA: Reading and hiking."]),
        ("music", ["Q: Favorite music?\nA: Jazz."]),
        ("movie", ["Q: Best movie?\nA: Inception."]),
        ("book", ["Q: Favorite book?\nA: Dune."]),
        ("sport", ["Q: Play sports?\nA: Tennis."]),
        ("language", ["Q: Languages?\nA: English and Spanish."]),
        ("birthday", ["Q: Birthday?\nA: March 15th."]),
        ("coffee", ["Q: Coffee or tea?\nA: Coffee, black."]),
        ("season", ["Q: Favorite season?\nA: Autumn."]),
        ("day", ["Q: Best day?\nA: Saturday."]),
        ("time", ["Q: Morning or night?\nA: Night owl."]),
        ("travel", ["Q: Dream destination?\nA: Japan."]),
        ("skill", ["Q: Best skill?\nA: Problem solving."]),
        ("fear", ["Q: Any fears?\nA: Public speaking."]),
        ("goal", ["Q: Life goal?\nA: Make a difference."]),
    ]
    
    for name, texts in facts:
        tasks.append((f"fact_{name}", make_data(texts)))
    
    # === PREFERENCE TASKS (20) ===
    prefs = [
        ("indent", ["Use 4 spaces for indentation.", "Always indent with 4 spaces."]),
        ("quotes", ["Use single quotes for strings.", "Prefer single over double."]),
        ("naming", ["Use snake_case for variables.", "Variables: my_variable_name."]),
        ("comments", ["Comment above the code.", "Comment goes here then code"]),
        ("brackets", ["Opening bracket same line.", "if x then y end"]),
        ("semicolons", ["Always use semicolons.", "const x = 1;"]),
        ("imports", ["Group imports by type.", "stdlib then third-party then local"]),
        ("functions", ["Keep functions small.", "Max 20 lines per function."]),
        ("classes", ["One class per file.", "class MyClass in my_class.py"]),
        ("tests", ["Test every function.", "def test_my_func():"]),
        ("docs", ["Docstrings for all public.", "Brief description here."]),
        ("types", ["Use type hints.", "def func(x: int) -> str:"]),
        ("errors", ["Handle all exceptions.", "try x except Error"]),
        ("logging", ["Log important events.", "logger.info Started"]),
        ("config", ["Use environment variables.", "os.environ KEY"]),
        ("constants", ["UPPERCASE for constants.", "MAX_SIZE = 100"]),
        ("returns", ["Early returns preferred.", "if not x: return"]),
        ("loops", ["Prefer comprehensions.", "x for x in items"]),
        ("dicts", ["Use .get() for safety.", "d.get key default"]),
        ("strings", ["Use f-strings.", "f Hello name"]),
    ]
    
    for name, texts in prefs:
        tasks.append((f"pref_{name}", make_data(texts)))
    
    # === FORMAT TASKS (20) ===
    formats = [
        ("date", ["Date format: YYYY-MM-DD", "Today: 2024-01-15"]),
        ("time", ["Time format: HH:MM", "Meeting at 14:30"]),
        ("currency", ["Currency: $X,XXX.XX", "Total: $1,234.56"]),
        ("phone", ["Phone: (XXX) XXX-XXXX", "Call: (555) 123-4567"]),
        ("email", ["Email: name@domain.com", "Contact: user@example.com"]),
        ("address", ["Address: Street, City, ST ZIP", "123 Main St, NYC, NY 10001"]),
        ("list_bullet", ["Lists use dashes", "- First item\n- Second item"]),
        ("list_number", ["Numbered lists: 1. 2. 3.", "1. First\n2. Second"]),
        ("heading", ["Headings: ## Title", "## Section Name"]),
        ("code_block", ["Code in triple backticks", "```python\ncode\n```"]),
        ("bold", ["Bold with **text**", "This is **important**"]),
        ("italic", ["Italic with *text*", "This is *emphasized*"]),
        ("link", ["Links: [text](url)", "[Click here](https://x.com)"]),
        ("quote", ["Quotes with >", "> This is a quote"]),
        ("table", ["Tables with |", "| A | B |\n|---|---|"]),
        ("hr", ["Horizontal rule: ---", "---"]),
        ("footnote", ["Footnotes: [^1]", "Text[^1]\n[^1]: Note"]),
        ("task", ["Tasks: [ ] and [x]", "[ ] Todo\n[x] Done"]),
        ("emoji", ["Emoji: :name:", ":smile: :thumbsup:"]),
        ("abbrev", ["Abbreviations expanded", "API means Application Programming Interface"]),
    ]
    
    for name, texts in formats:
        tasks.append((f"fmt_{name}", make_data(texts)))
    
    # === DOMAIN TASKS (20) ===
    domains = [
        ("legal", ["Party A shall indemnify.", "Governed by California law."]),
        ("medical", ["Patient presents with.", "Diagnosis: condition."]),
        ("finance", ["Q1 revenue increased.", "ROI of 15%."]),
        ("science", ["The experiment shows.", "Hypothesis confirmed."]),
        ("cooking", ["Preheat oven to 350F.", "Mix ingredients well."]),
        ("fitness", ["3 sets of 10 reps.", "Rest 60 seconds."]),
        ("travel", ["Flight departs at 8am.", "Hotel checkout by 11."]),
        ("gaming", ["Press X to continue.", "Level 5 unlocked."]),
        ("music_theory", ["C major chord: C-E-G.", "4/4 time signature."]),
        ("photography", ["ISO 100, f/2.8.", "Golden hour lighting."]),
        ("gardening", ["Plant in spring.", "Water twice weekly."]),
        ("diy", ["Tools needed: hammer.", "Step 1: measure twice."]),
        ("parenting", ["Bedtime at 8pm.", "Limit screen time."]),
        ("pets", ["Feed twice daily.", "Annual vet checkup."]),
        ("auto", ["Oil change every 5k.", "Check tire pressure."]),
        ("realestate", ["3BR/2BA, 1500sqft.", "Listed at $500k."]),
        ("hr", ["PTO accrues monthly.", "Review cycle Q4."]),
        ("marketing", ["CTR of 2.5%.", "A/B test results."]),
        ("security", ["Use 2FA.", "Rotate passwords."]),
        ("devops", ["CI/CD pipeline.", "Deploy to staging."]),
    ]
    
    for name, texts in domains:
        tasks.append((f"domain_{name}", make_data(texts)))
    
    return tasks

# =============================================================================
# KILLER DEMO
# =============================================================================

def run_killer_demo(path: str = "./sgm_personal", n_tasks: int = 100, quick: bool = False):
    print("="*70)
    print("SGM PERSONAL AI MEMORY - KILLER DEMO")
    print("="*70)
    
    # Clean start
    if Path(path).exists():
        shutil.rmtree(path)
    
    cfg = Config()
    storage = BlockStorage(path, cfg)
    model = Transformer(cfg, storage)
    model.anchor_early_layers()  # Lock representation anchors BEFORE training
    trainer = BlockTrainer(model, storage, cfg)
    
    print(f"\nConfig: {cfg.d_model}d, {cfg.n_heads}h, {cfg.n_layers}L")
    print(f"Block size: {cfg.block_size} params")
    print(f"Total: {cfg.total_params:,} params ({cfg.n_blocks} blocks)")
    
    # Get tasks
    all_tasks = generate_100_tasks()[:n_tasks]
    
    steps = 15 if quick else 25
    pop = 2 if quick else 3
    
    print(f"\n{'='*70}")
    print(f"TRAINING {n_tasks} TASKS")
    print("="*70)
    
    task_losses = {}
    checkpoints = [1, 10, 25, 50, 75, 100]
    
    # Store task #1 data for tracking
    task1_name, task1_data = all_tasks[0]
    
    start_time = time.time()
    
    for i, (name, data) in enumerate(all_tasks):
        result = trainer.train_task(name, data, steps=steps, pop=pop)
        task_losses[name] = result["loss"]
        
        # Progress update at checkpoints
        if (i + 1) in checkpoints or (i + 1) == n_tasks:
            stats = storage.stats()
            task1_loss = np.mean([model.loss(d[0], d[1]) for d in task1_data])
            task1_retention = task1_loss / task_losses[task1_name] if task_losses[task1_name] > 0 else 0
            
            elapsed = time.time() - start_time
            print(f"\n  After {i+1} tasks:")
            print(f"    Locked: {stats['locked_blocks']} blocks ({stats['pct_locked']:.1f}%)")
            print(f"    Task #1 retention: {task1_retention:.2f}x")
            print(f"    Time: {elapsed:.1f}s")
    
    # === FINAL EVALUATION ===
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print("="*70)
    
    # Measure all tasks
    print(f"\n{'Task':<25} | {'After':>8} | {'Now':>8} | {'Retention':>10}")
    print("-"*58)
    
    retentions = []
    for name, data in all_tasks[:10]:  # Show first 10
        after = task_losses[name]
        now = np.mean([model.loss(d[0], d[1]) for d in data])
        ret = now / after if after > 0 else 0
        retentions.append(ret)
        print(f"{name:<25} | {after:>8.3f} | {now:>8.3f} | {ret:>9.2f}x")
    
    print("...")
    
    # Last 5 tasks
    for name, data in all_tasks[-5:]:
        after = task_losses[name]
        now = np.mean([model.loss(d[0], d[1]) for d in data])
        ret = now / after if after > 0 else 0
        retentions.append(ret)
        print(f"{name:<25} | {after:>8.3f} | {now:>8.3f} | {ret:>9.2f}x")
    
    # Compute all retentions
    all_retentions = []
    for name, data in all_tasks:
        after = task_losses[name]
        now = np.mean([model.loss(d[0], d[1]) for d in data])
        ret = now / after if after > 0 else 0
        all_retentions.append(ret)
    
    # === SUMMARY ===
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    
    stats = storage.stats()
    
    # Task #1 final
    task1_final = np.mean([model.loss(d[0], d[1]) for d in task1_data])
    task1_ret = task1_final / task_losses[task1_name]
    
    # Inference timing
    prompt = np.array([ord(c) for c in "Hello "], dtype=np.int32)
    times = []
    for _ in range(20):
        t0 = time.time()
        _ = model.forward(prompt)
        times.append(time.time() - t0)
    
    avg_inference = np.mean(times) * 1000
    
    print(f"""
  TASK #1 AFTER {n_tasks-1} MORE TASKS:
    Initial loss: {task_losses[task1_name]:.4f}
    Final loss:   {task1_final:.4f}
    Retention:    {task1_ret:.2f}x {'[OK]' if task1_ret < 1.5 else '[X]'}

  ALL TASKS:
    Mean retention: {np.mean(all_retentions):.2f}x
    Median retention: {np.median(all_retentions):.2f}x
    Worst retention: {np.max(all_retentions):.2f}x

  STORAGE:
    Total params: {stats['total_params']:,}
    Locked params: {stats['locked_params']:,} ({stats['pct_locked']:.1f}%)
    Free params: {stats['free_params']:,}
    Params per task: {stats['locked_params'] // n_tasks if n_tasks else 0}

  INFERENCE:
    Latency: {avg_inference:.2f}ms
    Scales with tasks: NO

  VERDICT:
""")
    
    # Success criteria
    success = True
    
    if task1_ret > 1.5:
        print(f"    [X] Task #1 retention too high ({task1_ret:.2f}x > 1.5x)")
        success = False
    else:
        print(f"    [OK] Task #1 retention OK ({task1_ret:.2f}x)")

    if stats['pct_locked'] > 50:
        print(f"    [X] Too many params locked ({stats['pct_locked']:.1f}% > 50%)")
        success = False
    else:
        print(f"    [OK] Locked params OK ({stats['pct_locked']:.1f}%)")

    if avg_inference > 100:
        print(f"    [X] Inference too slow ({avg_inference:.0f}ms > 100ms)")
        success = False
    else:
        print(f"    [OK] Inference OK ({avg_inference:.1f}ms)")
    
    print(f"\n  {'='*50}")
    if success:
        print("  PERSONAL AI MEMORY SUBSTRATE: VALIDATED")
    else:
        print("  NEEDS TUNING")
    print(f"  {'='*50}")

# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run 100-task killer demo')
    parser.add_argument('--quick', action='store_true', help='Quick 25-task version')
    parser.add_argument('--path', default='./sgm_personal')
    args = parser.parse_args()
    
    if args.demo:
        n_tasks = 25 if args.quick else 100
        run_killer_demo(args.path, n_tasks=n_tasks, quick=args.quick)
    else:
        print("Usage:")
        print("  python sgm_personal_ai.py --demo         # Full 100-task demo")
        print("  python sgm_personal_ai.py --demo --quick # Quick 25-task test")