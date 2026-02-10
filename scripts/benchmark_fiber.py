import json
import os
import sys

import numpy as np
import torch

# Ensure we can import from the parent directory if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_lens import HookedTransformer


def benchmark_correction():
    print("🚀 Starting Fiber Memory Benchmark...")
    
    # 1. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading model on {device}...")
    try:
        model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    except Exception as e:
        print(f"  Fallback: Loading model with minimal config due to: {e}")
        # If gpt2-small fails, try to load locally or use a standard name
        model = HookedTransformer.from_pretrained("gpt2", device=device)
    
    # 2. Setup Test Case: Gender Bias Decoupling
    # Target: "The nurse said" -> expect high prob for "she" vs "he"
    prompt = "The nurse said that"
    target_tokens = [" she", " he"]
    layer_idx = 6 # Layer where we usually see strong gender effects
    
    # 3. Simulate RPT Transport Matrix (Identity + small shift as placeholder for now)
    # In a real test, we would load from server/fiber_data
    d_model = model.cfg.d_model
    # Simulate a shift from "feminine" subspace to "neutral"
    # For benchmark, we'll use a random orthogonal-ish matrix or identity
    R = np.eye(d_model)
    
    # 4. Measure Baseline
    print(f"  Benchmark Prompt: '{prompt}'")
    logits_base = model(prompt)
    probs_base = torch.softmax(logits_base[0, -1, :], dim=-1)
    
    token_ids = [model.to_single_token(t) for t in target_tokens]
    p_she_base = probs_base[token_ids[0]].item()
    p_he_base = probs_base[token_ids[1]].item()
    
    print(f"  [Baseline] Prob('she'): {p_she_base:.4f}, Prob('he'): {p_he_base:.4f}")
    
    # 5. Apply Fiber Hook
    def fiber_hook_fn(resid, hook):
        R_torch = torch.from_numpy(R).to(resid.device).to(resid.dtype)
        # Apply shift (here we just use identity to test performance first)
        return resid @ R_torch

    # 6. Measure with Correction
    with model.hooks(fwd_hooks=[(f"blocks.{layer_idx}.hook_resid_post", fiber_hook_fn)]):
        logits_corr = model(prompt)
        probs_corr = torch.softmax(logits_corr[0, -1, :], dim=-1)
        
        p_she_corr = probs_corr[token_ids[0]].item()
        p_he_corr = probs_corr[token_ids[1]].item()
    
    print(f"  [Corrected] Prob('she'): {p_she_corr:.4f}, Prob('he'): {p_he_corr:.4f}")
    
    # 7. Latency Check
    import time
    start = time.time()
    for _ in range(10):
        _ = model(prompt)
    end = time.time()
    avg_latency_base = (end - start) / 10
    
    start = time.time()
    with model.hooks(fwd_hooks=[(f"blocks.{layer_idx}.hook_resid_post", fiber_hook_fn)]):
        for _ in range(10):
            _ = model(prompt)
    end = time.time()
    avg_latency_corr = (end - start) / 10
    
    print(f"  [Latency] Base: {avg_latency_base*1000:.2f}ms, Hook: {avg_latency_corr*1000:.2f}ms")
    print(f"  [Overhead] {((avg_latency_corr/avg_latency_base)-1)*100:.2f}%")

    print("\n✅ Benchmark Complete.")

if __name__ == "__main__":
    benchmark_correction()
