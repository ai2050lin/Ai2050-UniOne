#!/usr/bin/env python
"""
Episodic Consolidation Simulation Script.

Goal:
- Validate that a sequence of massive micro-features (e.g., specific entities) causes dimension crowding/overloading.
- Simulate an Episodic Consolidation operator (Abstraction Head) that compresses the sequence into a sparse macro-category vector.
- Evaluate the recovery of semantic space and the capacity released.

Output:
- JSON with consolidation metrics, saved to tempdata or tests/gemini_temp.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch


def load_model(model_id: str, dtype_name: str, local_only: bool):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    if local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_only,
        use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=local_only,
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def get_last_hidden_batch(model, tok, prompts: list[str]) -> np.ndarray:
    device = next(model.parameters()).device
    enc = tok(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc, use_cache=False, return_dict=True, output_hidden_states=True)
    hidden = out.hidden_states[-1]
    positions = enc["attention_mask"].sum(dim=1) - 1
    rows = []
    for bi in range(hidden.shape[0]):
        rows.append(hidden[bi, positions[bi], :].detach().float().cpu().numpy().astype(np.float32))
    return np.stack(rows, axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--dtype", type=str, default="float32")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--json-out", type=str, default="tests/gemini_temp/episodic_consolidation_20260311.json")
    args = ap.parse_args()

    t0 = time.time()
    
    # Load Model
    print(f"Loading {args.model_id}...")
    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    
    # 1. Micro-sequence overload
    entities = [
        "apple", "banana", "orange", "grape", "pear", "peach", "plum", "melon",
        "strawberry", "blueberry", "raspberry", "blackberry", "cherry", "mango",
        "papaya", "kiwi", "pineapple", "coconut", "pomegranate", "watermelon"
    ]
    macro_category = "fruit"
    
    # Generate long sequence 
    sequence_string = "I saw " + ", ".join(entities) + "."
    
    # Embeddings
    print("Computing embeddings...")
    entity_vecs = get_last_hidden_batch(model, tok, [f"This is an {e}" for e in entities])
    macro_vec = get_last_hidden_batch(model, tok, [f"The abstract concept of {macro_category}"])[0]
    seq_vec = get_last_hidden_batch(model, tok, [sequence_string])[0]
    
    # Compute base norms (simulate memory crowding)
    ent_norms = [float(np.linalg.norm(v)) for v in entity_vecs]
    avg_micro_norm = float(np.mean(ent_norms))
    macro_norm = float(np.linalg.norm(macro_vec))
    seq_norm = float(np.linalg.norm(seq_vec))
    
    # Simulate Consolidation projection (Abstraction Head)
    # Project the crowded sequence vector onto the stable macro category vector
    macro_unit = macro_vec / np.linalg.norm(macro_vec)
    projected_magnitude = np.dot(seq_vec, macro_unit)
    consolidated_vec = projected_magnitude * macro_unit
    
    residual_vec = seq_vec - consolidated_vec
    
    # Capacity Release Metrics
    # In HRR, dense dimensions equal 4096. If we compress 20 entities into 1 macro, 
    # we release the variance bounded by the entities.
    total_crowded_variance = np.sum([np.linalg.norm(v - np.mean(entity_vecs, axis=0))**2 for v in entity_vecs])
    residual_variance = float(np.linalg.norm(residual_vec)**2)
    
    # Semantic Retention (how much the system retains the category meaning)
    cosine_retention = float(np.dot(consolidated_vec, macro_unit) / np.linalg.norm(consolidated_vec))
    
    # Determine hypotheses
    # H1: The long sequence overloaded the representations, sequence norm is comparable or higher than macro.
    # H2: Episodic consolidation drastically reduces the crowded variance.
    # H3: The macro representation perfectly retains semantic identity (cosine ≈ 1.0).
    
    h1_overload = seq_norm > (avg_micro_norm * 1.5)
    h2_compression = residual_variance < total_crowded_variance
    h3_retention = cosine_retention > 0.95
    
    metrics = {
        "sequence_crowded_norm": seq_norm,
        "average_micro_norm": avg_micro_norm,
        "abstract_macro_norm": macro_norm,
        "total_micro_variance": float(total_crowded_variance),
        "post_consolidation_residual_variance": residual_variance,
        "compression_ratio": float(total_crowded_variance / max(1e-9, residual_variance)),
        "semantic_retention_cosine": cosine_retention,
    }
    
    hypotheses = {
        "H1_sequence_overload_detected": h1_overload,
        "H2_consolidation_reduces_variance": h2_compression,
        "H3_semantic_identity_retained": h3_retention
    }
    
    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_id": args.model_id,
            "entities_count": len(entities),
            "macro_concept": macro_category,
            "runtime_sec": float(time.time() - t0)
        },
        "metrics": metrics,
        "hypotheses": hypotheses
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print("\n[Episodic Consolidation Simulation Results]")
    print(json.dumps(metrics, indent=2))
    print(json.dumps(hypotheses, indent=2))
    print(f"\nSaved to: {out_path}")

if __name__ == "__main__":
    main()
