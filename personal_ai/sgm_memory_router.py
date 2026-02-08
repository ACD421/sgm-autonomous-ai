#!/usr/bin/env python3
"""
SGM MEMORY ROUTER
=================
Fixes memory regression by separating memory from core model weights.

Architecture:
  [ Core Model (frozen/SGM) ]
            ↓
  [ Memory Router (tiny, trainable) ]
            ↓
  [ External Memory Store (key-value) ]

Memory is RETRIEVED, not re-learned into weights.
This prevents memory tasks from competing with reasoning/style.

Usage:
  python sgm_memory_router.py --demo      # Run memory demo
  python sgm_memory_router.py --integrate # Add to existing SGM AI
"""

import numpy as np
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import argparse

# =============================================================================
# EXTERNAL MEMORY STORE
# =============================================================================

class MemoryStore:
    """External key-value memory - NOT in model weights"""
    
    def __init__(self, path: str = "./sgm_memory"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.store_file = self.path / "memory.json"
        self.index_file = self.path / "memory_index.json"
        
        self.memories = self._load()
        self.embeddings = {}  # In-memory embedding cache
    
    def _load(self) -> Dict:
        if self.store_file.exists():
            return json.load(open(self.store_file))
        return {
            "facts": {},      # key -> value
            "episodes": [],   # temporal sequence
            "skills": {},     # skill_name -> data
            "meta": {
                "created": time.time(),
                "n_stores": 0,
                "n_retrievals": 0
            }
        }
    
    def save(self):
        json.dump(self.memories, open(self.store_file, 'w'), indent=2)
    
    def _hash_key(self, text: str) -> str:
        """Create consistent key from text"""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()[:16]
    
    def store_fact(self, key: str, value: str, confidence: float = 1.0):
        """Store a fact (overwrites if exists)"""
        key_hash = self._hash_key(key)
        self.memories["facts"][key_hash] = {
            "key": key,
            "value": value,
            "confidence": confidence,
            "timestamp": time.time(),
            "access_count": 0
        }
        self.memories["meta"]["n_stores"] += 1
        self.save()
        return key_hash
    
    def store_episode(self, content: str, context: str = ""):
        """Store an episodic memory (temporal)"""
        self.memories["episodes"].append({
            "content": content,
            "context": context,
            "timestamp": time.time(),
            "importance": 1.0
        })
        # Keep last 1000 episodes
        if len(self.memories["episodes"]) > 1000:
            self.memories["episodes"] = self.memories["episodes"][-1000:]
        self.memories["meta"]["n_stores"] += 1
        self.save()
    
    def store_skill(self, name: str, data: Dict):
        """Store a learned skill/pattern"""
        self.memories["skills"][name] = {
            "data": data,
            "timestamp": time.time(),
            "use_count": 0
        }
        self.memories["meta"]["n_stores"] += 1
        self.save()
    
    def retrieve_fact(self, query: str, threshold: float = 0.5) -> Optional[Dict]:
        """Retrieve fact by exact or fuzzy match"""
        query_hash = self._hash_key(query)
        
        # Exact match
        if query_hash in self.memories["facts"]:
            fact = self.memories["facts"][query_hash]
            fact["access_count"] += 1
            self.memories["meta"]["n_retrievals"] += 1
            return fact
        
        # Fuzzy match - simple word overlap
        query_words = set(query.lower().split())
        best_match = None
        best_score = 0
        
        for key_hash, fact in self.memories["facts"].items():
            fact_words = set(fact["key"].lower().split())
            if not fact_words:
                continue
            overlap = len(query_words & fact_words) / len(query_words | fact_words)
            if overlap > best_score and overlap >= threshold:
                best_score = overlap
                best_match = fact
        
        if best_match:
            best_match["access_count"] += 1
            self.memories["meta"]["n_retrievals"] += 1
            return best_match
        
        return None
    
    def retrieve_recent_episodes(self, n: int = 5) -> List[Dict]:
        """Get most recent episodes"""
        self.memories["meta"]["n_retrievals"] += 1
        return self.memories["episodes"][-n:]
    
    def retrieve_skill(self, name: str) -> Optional[Dict]:
        """Retrieve a stored skill"""
        if name in self.memories["skills"]:
            skill = self.memories["skills"][name]
            skill["use_count"] += 1
            self.memories["meta"]["n_retrievals"] += 1
            return skill
        return None
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search all memories for relevant matches"""
        results = []
        query_words = set(query.lower().split())
        
        # Search facts
        for key_hash, fact in self.memories["facts"].items():
            fact_words = set(fact["key"].lower().split()) | set(fact["value"].lower().split())
            if query_words & fact_words:
                overlap = len(query_words & fact_words) / len(query_words)
                results.append({"type": "fact", "score": overlap, "data": fact})
        
        # Search episodes
        for ep in self.memories["episodes"]:
            ep_words = set(ep["content"].lower().split())
            if query_words & ep_words:
                overlap = len(query_words & ep_words) / len(query_words)
                results.append({"type": "episode", "score": overlap, "data": ep})
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def stats(self) -> Dict:
        return {
            "n_facts": len(self.memories["facts"]),
            "n_episodes": len(self.memories["episodes"]),
            "n_skills": len(self.memories["skills"]),
            "n_stores": self.memories["meta"]["n_stores"],
            "n_retrievals": self.memories["meta"]["n_retrievals"]
        }


# =============================================================================
# MEMORY ROUTER (tiny trainable module)
# =============================================================================

class MemoryRouter:
    """
    Tiny module that decides:
    - When to STORE to external memory
    - When to RETRIEVE from external memory
    - How much to trust retrieved memory vs model output
    
    This is the ONLY part that trains on memory tasks.
    Core model weights stay untouched for memory.
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 64, path: str = "./sgm_memory"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        
        self.weights_file = self.path / "router_weights.npy"
        self.config_file = self.path / "router_config.json"
        
        # Initialize or load weights
        if self.weights_file.exists():
            self._load_weights()
        else:
            self._init_weights()
        
        # External memory
        self.memory = MemoryStore(path)
    
    def _init_weights(self):
        """Initialize router weights (very small)"""
        # Store decision: input -> should_store (0-1)
        self.W_store = np.random.randn(self.input_dim, self.hidden_dim).astype(np.float32) * 0.1
        self.b_store = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W_store_out = np.random.randn(self.hidden_dim, 1).astype(np.float32) * 0.1
        
        # Retrieve decision: input -> should_retrieve (0-1)
        self.W_retrieve = np.random.randn(self.input_dim, self.hidden_dim).astype(np.float32) * 0.1
        self.b_retrieve = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W_retrieve_out = np.random.randn(self.hidden_dim, 1).astype(np.float32) * 0.1
        
        # Trust weight: how much to trust memory vs model
        self.W_trust = np.random.randn(self.input_dim, self.hidden_dim).astype(np.float32) * 0.1
        self.b_trust = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W_trust_out = np.random.randn(self.hidden_dim, 1).astype(np.float32) * 0.1
        
        # Count params BEFORE save
        self.n_params = (
            self.W_store.size + self.b_store.size + self.W_store_out.size +
            self.W_retrieve.size + self.b_retrieve.size + self.W_retrieve_out.size +
            self.W_trust.size + self.b_trust.size + self.W_trust_out.size
        )
        
        self._save_weights()
        print(f"[ROUTER] Initialized with {self.n_params:,} params")
    
    def _save_weights(self):
        np.savez(
            self.weights_file,
            W_store=self.W_store, b_store=self.b_store, W_store_out=self.W_store_out,
            W_retrieve=self.W_retrieve, b_retrieve=self.b_retrieve, W_retrieve_out=self.W_retrieve_out,
            W_trust=self.W_trust, b_trust=self.b_trust, W_trust_out=self.W_trust_out
        )
        config = {"input_dim": self.input_dim, "hidden_dim": self.hidden_dim, "n_params": self.n_params}
        json.dump(config, open(self.config_file, 'w'))
    
    def _load_weights(self):
        data = np.load(self.weights_file)
        self.W_store = data["W_store"]
        self.b_store = data["b_store"]
        self.W_store_out = data["W_store_out"]
        self.W_retrieve = data["W_retrieve"]
        self.b_retrieve = data["b_retrieve"]
        self.W_retrieve_out = data["W_retrieve_out"]
        self.W_trust = data["W_trust"]
        self.b_trust = data["b_trust"]
        self.W_trust_out = data["W_trust_out"]
        
        config = json.load(open(self.config_file))
        self.input_dim = config["input_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.n_params = config["n_params"]
        print(f"[ROUTER] Loaded {self.n_params:,} params")
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _embed_text(self, text: str) -> np.ndarray:
        """
        Embedding with n-gram features for better semantic clustering.
        Upgrade path: replace with frozen model embeddings later.
        """
        # Character unigrams (position-invariant)
        unigram = np.zeros(128, dtype=np.float32)
        for c in text.lower():
            idx = ord(c) % 128
            unigram[idx] += 1
        if unigram.sum() > 0:
            unigram /= unigram.sum()
        
        # Character bigrams (captures patterns)
        bigram = np.zeros(64, dtype=np.float32)
        text_lower = text.lower()
        for i in range(len(text_lower) - 1):
            h = hash(text_lower[i:i+2]) % 64
            bigram[h] += 1
        if bigram.sum() > 0:
            bigram /= bigram.sum()
        
        # Word-level features
        words = text.lower().split()
        word_feat = np.zeros(64, dtype=np.float32)
        
        # Question indicators
        if any(text.lower().startswith(q) for q in ["what", "who", "where", "when", "how", "why"]):
            word_feat[0] = 1.0
        if "?" in text:
            word_feat[1] = 1.0
        
        # Memory command indicators
        if "remember" in text.lower():
            word_feat[2] = 1.0
        if "forget" in text.lower():
            word_feat[3] = 1.0
        if "my " in text.lower() or "i am" in text.lower() or "i'm" in text.lower():
            word_feat[4] = 1.0  # Personal info
        
        # Fact indicators
        if " is " in text.lower() or " are " in text.lower():
            word_feat[5] = 1.0
        
        # Length features
        word_feat[10] = min(len(words) / 20.0, 1.0)
        word_feat[11] = min(len(text) / 200.0, 1.0)
        
        # Combine
        embedding = np.concatenate([unigram, bigram, word_feat])
        return embedding[:self.input_dim]
    
    def set_model_embedder(self, embed_fn):
        """
        Upgrade: use frozen model embeddings for better semantics.
        embed_fn: text -> np.ndarray of shape (input_dim,)
        """
        self._model_embed_fn = embed_fn
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding, using model if available"""
        if hasattr(self, '_model_embed_fn') and self._model_embed_fn is not None:
            return self._model_embed_fn(text)
        return self._embed_text(text)
    
    def should_store(self, text: str) -> Tuple[bool, float]:
        """Decide if this text should be stored in memory"""
        x = self._get_embedding(text)
        h = self._relu(x @ self.W_store + self.b_store)
        score = self._sigmoid(h @ self.W_store_out)[0]
        return score > 0.5, float(score)
    
    def should_retrieve(self, text: str) -> Tuple[bool, float]:
        """Decide if we should retrieve from memory for this query"""
        x = self._get_embedding(text)
        h = self._relu(x @ self.W_retrieve + self.b_retrieve)
        score = self._sigmoid(h @ self.W_retrieve_out)[0]
        return score > 0.5, float(score)
    
    def get_trust_weight(self, text: str) -> float:
        """How much to trust memory (0-1) vs model output"""
        x = self._get_embedding(text)
        h = self._relu(x @ self.W_trust + self.b_trust)
        score = self._sigmoid(h @ self.W_trust_out)[0]
        return float(score)
    
    def process_input(self, text: str) -> Dict:
        """
        Main routing function:
        1. Check if should retrieve
        2. If yes, search memory
        3. Return memory + trust weight
        """
        should_ret, ret_score = self.should_retrieve(text)
        
        result = {
            "should_retrieve": should_ret,
            "retrieve_score": ret_score,
            "memories": [],
            "trust_weight": 0.0
        }
        
        if should_ret:
            # Search memory
            memories = self.memory.search(text, top_k=3)
            result["memories"] = memories
            result["trust_weight"] = self.get_trust_weight(text)
        
        return result
    
    def process_output(self, input_text: str, output_text: str) -> Dict:
        """
        After model generates output:
        1. Check if should store
        2. If yes, store key-value pair
        """
        should_st, st_score = self.should_store(input_text + " " + output_text)
        
        result = {
            "should_store": should_st,
            "store_score": st_score,
            "stored": False
        }
        
        if should_st:
            # Detect if this is a fact (Q: ... A: ... pattern)
            if "?" in input_text or input_text.lower().startswith(("what", "who", "where", "when", "how", "why")):
                self.memory.store_fact(input_text, output_text)
                result["stored"] = True
                result["store_type"] = "fact"
            else:
                self.memory.store_episode(output_text, context=input_text)
                result["stored"] = True
                result["store_type"] = "episode"
        
        return result
    
    def train_step(self, examples: List[Tuple[str, str, bool]], lr: float = 0.01):
        """
        Train router on labeled examples.
        Each example: (text, type, should_activate)
        type: "store" | "retrieve" | "trust"
        
        This trains ONLY the router, not the main model.
        """
        for text, task_type, target in examples:
            x = self._get_embedding(text)
            target = float(target)
            
            if task_type == "store":
                # Forward
                h = self._relu(x @ self.W_store + self.b_store)
                pred = self._sigmoid(h @ self.W_store_out)[0]
                
                # Backward (simplified gradient)
                error = pred - target
                grad_out = error * pred * (1 - pred)
                
                # Update
                self.W_store_out -= lr * np.outer(h, [grad_out])
                grad_h = grad_out * self.W_store_out.flatten() * (h > 0)
                self.W_store -= lr * np.outer(x, grad_h)
                self.b_store -= lr * grad_h
                
            elif task_type == "retrieve":
                h = self._relu(x @ self.W_retrieve + self.b_retrieve)
                pred = self._sigmoid(h @ self.W_retrieve_out)[0]
                error = pred - target
                grad_out = error * pred * (1 - pred)
                self.W_retrieve_out -= lr * np.outer(h, [grad_out])
                grad_h = grad_out * self.W_retrieve_out.flatten() * (h > 0)
                self.W_retrieve -= lr * np.outer(x, grad_h)
                self.b_retrieve -= lr * grad_h
                
            elif task_type == "trust":
                h = self._relu(x @ self.W_trust + self.b_trust)
                pred = self._sigmoid(h @ self.W_trust_out)[0]
                error = pred - target
                grad_out = error * pred * (1 - pred)
                self.W_trust_out -= lr * np.outer(h, [grad_out])
                grad_h = grad_out * self.W_trust_out.flatten() * (h > 0)
                self.W_trust -= lr * np.outer(x, grad_h)
                self.b_trust -= lr * grad_h
        
        self._save_weights()
    
    def stats(self) -> Dict:
        mem_stats = self.memory.stats()
        return {
            "router_params": self.n_params,
            **mem_stats
        }


# =============================================================================
# INTEGRATED MEMORY-AUGMENTED MODEL
# =============================================================================

class MemoryAugmentedModel:
    """
    Wrapper that adds memory routing to any model.
    
    Flow:
    1. Input comes in
    2. Check for explicit memory commands
    3. Router decides: retrieve from memory?
    4. If yes AND trust > threshold: prepend relevant memories
    5. Model generates output
    6. Router decides: store this?
    7. If yes AND within budget: store to external memory
    
    Core model weights NEVER update for memory tasks.
    """
    
    def __init__(self, model, router: MemoryRouter, 
                 trust_threshold: float = 0.3,
                 daily_write_budget: int = 100):
        self.model = model
        self.router = router
        self.trust_threshold = trust_threshold
        self.daily_write_budget = daily_write_budget
        self._daily_writes = 0
        self._last_write_day = None
    
    def _check_write_budget(self) -> bool:
        """Check if we can write today"""
        today = time.strftime("%Y-%m-%d")
        if self._last_write_day != today:
            self._last_write_day = today
            self._daily_writes = 0
        return self._daily_writes < self.daily_write_budget
    
    def _handle_explicit_command(self, prompt: str) -> Optional[str]:
        """Handle explicit memory commands - bypass router"""
        prompt_lower = prompt.lower().strip()
        
        # REMEMBER command
        if prompt_lower.startswith("remember that ") or prompt_lower.startswith("remember: "):
            fact = prompt[prompt.find(" ", 9) + 1:].strip()
            # Try to parse as key-value
            if " is " in fact:
                parts = fact.split(" is ", 1)
                self.router.memory.store_fact(parts[0].strip(), parts[1].strip())
                return f"✓ Remembered: {parts[0].strip()} is {parts[1].strip()}"
            else:
                self.router.memory.store_episode(fact, context="explicit remember")
                return f"✓ Remembered: {fact}"
        
        # FORGET command
        if prompt_lower.startswith("forget ") or prompt_lower.startswith("forget: "):
            query = prompt[7:].strip()
            # Find and remove matching fact
            key_hash = self.router.memory._hash_key(query)
            if key_hash in self.router.memory.memories["facts"]:
                del self.router.memory.memories["facts"][key_hash]
                self.router.memory.save()
                return f"✓ Forgot: {query}"
            return f"✗ No memory found for: {query}"
        
        # LIST command
        if prompt_lower in ["list memories", "what do you know about me", "show memories", "list what you know"]:
            facts = self.router.memory.memories["facts"]
            if not facts:
                return "I don't have any stored memories yet."
            lines = ["Here's what I remember:"]
            for key_hash, fact in list(facts.items())[:20]:
                lines.append(f"  • {fact['key']} → {fact['value']}")
            if len(facts) > 20:
                lines.append(f"  ... and {len(facts) - 20} more")
            return "\n".join(lines)
        
        return None  # Not a command
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        # Step 0: Check for explicit commands
        command_result = self._handle_explicit_command(prompt)
        if command_result:
            return command_result
        
        # Step 1: Check memory
        mem_result = self.router.process_input(prompt)
        
        augmented_prompt = prompt
        memory_used = False
        
        # Only prepend memory if trust > threshold
        if (mem_result["should_retrieve"] and 
            mem_result["memories"] and 
            mem_result["trust_weight"] >= self.trust_threshold):
            
            # Format with confidence
            memory_lines = []
            for m in mem_result["memories"][:2]:
                value = m['data'].get('value', m['data'].get('content', ''))
                conf = m.get('score', mem_result['trust_weight'])
                memory_lines.append(f"[Memory|conf={conf:.0%}]: {value}")
            
            memory_context = "\n".join(memory_lines)
            augmented_prompt = f"{memory_context}\n\n{prompt}"
            memory_used = True
        
        # Step 2: Generate with model
        output = self.model.generate(augmented_prompt, max_tokens=max_tokens)
        
        # Extract just the new output
        if augmented_prompt in output:
            output = output[len(augmented_prompt):].strip()
        
        # Step 3: Maybe store (with budget check)
        if self._check_write_budget():
            store_result = self.router.process_output(prompt, output)
            if store_result.get("stored"):
                self._daily_writes += 1
        
        return output
    
    def teach_fact(self, key: str, value: str):
        """Explicitly teach a fact (bypasses router decision)"""
        self.router.memory.store_fact(key, value, confidence=1.0)
    
    def recall(self, query: str) -> Optional[str]:
        """Try to recall from memory"""
        fact = self.router.memory.retrieve_fact(query)
        if fact:
            return fact["value"]
        return None
    
    def stats(self) -> Dict:
        stats = self.router.stats()
        stats["daily_writes"] = self._daily_writes
        stats["write_budget"] = self.daily_write_budget
        return stats


# =============================================================================
# DEMO
# =============================================================================

def run_demo():
    print("="*70)
    print("SGM MEMORY ROUTER DEMO")
    print("="*70)
    
    # Initialize router
    router = MemoryRouter(input_dim=256, hidden_dim=64, path="./sgm_memory_demo")
    
    print(f"\nRouter params: {router.n_params:,}")
    print(f"Memory stats: {router.memory.stats()}")
    
    # Train router on example patterns
    print("\n[TRAINING ROUTER]")
    
    training_data = [
        # Store patterns (facts, important info)
        ("My name is Alice", "store", True),
        ("I live in San Francisco", "store", True),
        ("The capital of France is Paris", "store", True),
        ("Remember that my birthday is March 15", "store", True),
        ("My favorite color is blue", "store", True),
        ("Hello how are you", "store", False),
        ("What's the weather like", "store", False),
        ("Write me a poem", "store", False),
        
        # Retrieve patterns (questions, recalls)
        ("What is my name?", "retrieve", True),
        ("Where do I live?", "retrieve", True),
        ("What is the capital of France?", "retrieve", True),
        ("When is my birthday?", "retrieve", True),
        ("What is my favorite color?", "retrieve", True),
        ("Who am I?", "retrieve", True),
        ("Write me a poem", "retrieve", False),
        ("Help me code a function", "retrieve", False),
        ("Tell me a joke", "retrieve", False),
        
        # Trust patterns (factual questions = high trust)
        ("What is my name?", "trust", True),
        ("What is 2+2?", "trust", True),
        ("Who is the president?", "trust", True),
        ("Where do I work?", "trust", True),
        ("Write a creative story", "trust", False),
        ("What do you think about art?", "trust", False),
        ("Generate some ideas", "trust", False),
    ]
    
    for epoch in range(100):
        np.random.shuffle(training_data)
        router.train_step(training_data, lr=0.05)
    
    print("  Router trained on patterns (100 epochs)")
    
    # Store some facts
    print("\n[STORING FACTS]")
    
    facts = [
        ("What is my name?", "Andrew"),
        ("Where do I live?", "Southlake, Texas"),
        ("What is my favorite color?", "Blue"),
        ("What is my job?", "Cybersecurity researcher"),
        ("What am I working on?", "SGM memory substrate for AI"),
    ]
    
    for key, value in facts:
        router.memory.store_fact(key, value)
        print(f"  Stored: {key} -> {value}")
    
    # Test retrieval with trust gating
    print("\n[TESTING RETRIEVAL + TRUST GATING]")
    
    queries = [
        "What is my name?",
        "Where do I live?",
        "What am I working on?",
        "Write me a poem",  # Should NOT retrieve (low trust)
        "Tell me a joke",   # Should NOT retrieve
    ]
    
    for query in queries:
        result = router.process_input(query)
        print(f"\n  Query: {query}")
        print(f"  Retrieve: {result['should_retrieve']} (score: {result['retrieve_score']:.2f})")
        print(f"  Trust: {result['trust_weight']:.2f}")
        
        # Simulate trust gating (threshold 0.3)
        if result['should_retrieve'] and result['trust_weight'] >= 0.3:
            if result['memories']:
                print(f"  → PREPEND: {result['memories'][0]['data'].get('value', 'N/A')}")
            else:
                print(f"  → No matching memory")
        else:
            print(f"  → Skip memory (trust too low or not retrieve)")
    
    # Test explicit commands
    print("\n[TESTING EXPLICIT COMMANDS]")
    
    class DummyModel:
        def generate(self, prompt, max_tokens=100):
            return prompt + "\n[Generated output]"
    
    augmented = MemoryAugmentedModel(DummyModel(), router, trust_threshold=0.3)
    
    commands = [
        "remember that my dog's name is Max",
        "remember: I prefer dark mode",
        "list memories",
        "What is my name?",
        "forget my dog's name",
        "list memories",
    ]
    
    for cmd in commands:
        print(f"\n  > {cmd}")
        result = augmented.generate(cmd, max_tokens=50)
        print(f"  < {result[:200]}")
    
    # Stats
    print("\n" + "="*70)
    print("FINAL STATS")
    print("="*70)
    stats = augmented.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n[SUCCESS] Memory router with:")
    print("  ✓ N-gram embeddings (better than char-bag)")
    print("  ✓ Trust-gated memory prepending")
    print("  ✓ Explicit commands (remember/forget/list)")
    print("  ✓ Daily write budget")
    print("  ✓ External storage (not in model weights)")


def integrate_with_sgm():
    """Show how to integrate with existing SGM self-improving AI"""
    
    print("="*70)
    print("INTEGRATING MEMORY ROUTER WITH SGM AI")
    print("="*70)
    
    # Check if SGM AI exists
    sgm_path = Path("./sgm_ai")
    if not sgm_path.exists():
        print("\n[!] SGM AI not found at ./sgm_ai")
        print("    Run: python sgm_self_improving.py --run --iterations 10")
        print("    Then run this again.")
        return
    
    print("\n[LOADING SGM AI]")
    
    # Import the SGM model
    import sys
    sys.path.insert(0, ".")
    
    try:
        from sgm_self_improving import Config, BlockStorage, Transformer
        
        cfg = Config()
        storage = BlockStorage("./sgm_ai", cfg)
        model = Transformer(cfg, storage)
        
        stats = storage.stats()
        print(f"  Loaded: {stats['total_params']:,} params, {stats['pct_locked']:.1f}% locked")
        
    except ImportError as e:
        print(f"  Could not import SGM AI: {e}")
        print("  Make sure sgm_self_improving.py is in current directory")
        return
    
    # Create memory router
    print("\n[CREATING MEMORY ROUTER]")
    router = MemoryRouter(input_dim=256, hidden_dim=64, path="./sgm_ai/memory")
    
    # Wrap model
    print("\n[WRAPPING MODEL]")
    augmented = MemoryAugmentedModel(model, router)
    
    # Test
    print("\n[TESTING]")
    
    # Teach a fact
    augmented.teach_fact("What is the user's name?", "Andrew")
    print("  Taught: user's name is Andrew")
    
    # Recall
    recalled = augmented.recall("What is the user's name?")
    print(f"  Recalled: {recalled}")
    
    # Generate (memory will be prepended if relevant)
    print("\n[GENERATING WITH MEMORY]")
    output = augmented.generate("Hello, what is my name?", max_tokens=50)
    print(f"  Output: {output[:100]}...")
    
    print("\n[SUCCESS] Memory router integrated")
    print("  - Memory tasks now use external store")
    print("  - Core model weights unchanged for memory")
    print("  - Router decides when to store/retrieve")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SGM Memory Router')
    parser.add_argument('--demo', action='store_true', help='Run memory router demo')
    parser.add_argument('--integrate', action='store_true', help='Integrate with SGM AI')
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.integrate:
        integrate_with_sgm()
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python sgm_memory_router.py --demo")
        print("  python sgm_memory_router.py --integrate")