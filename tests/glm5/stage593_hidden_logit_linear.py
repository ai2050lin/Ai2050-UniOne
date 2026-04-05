#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage593-mini: hidden->logit线性验证 + unembedding有效秩 — 快速版
只测Qwen3和GLM4两个模型（代表性），每个模型3个句子
"""

from __future__ import annotations
import sys, json, time, gc, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    load_model_bundle, free_model, discover_layers, move_batch_to_model_device
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

TEST_SENTENCES = [
    "The cat sat on the mat.",
    "Mathematics is the language of science.",
    "The capital of France is Paris.",
]


def get_unembedding_matrix(model):
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        return model.lm_head.weight.data.float().cpu()
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens.weight.data.float().cpu()
    if hasattr(model, 'get_output_embeddings'):
        oe = model.get_output_embeddings()
        if oe is not None and hasattr(oe, 'weight'):
            return oe.weight.data.float().cpu()
    return None


def run_single_model(model_key):
    print(f"\n=== {model_key.upper()} ===")
    t0 = time.time()
    bundle = load_model_bundle(model_key)
    if bundle is None:
        return {"error": f"Cannot load {model_key}"}
    model, tokenizer = bundle

    unembed = get_unembedding_matrix(model)
    if unembed is None:
        free_model(model)
        return {"error": "No unembedding matrix"}
    print(f"  unembed: {unembed.shape}")

    # SVD
    U, S, Vt = torch.linalg.svd(unembed, full_matrices=False)
    S_np = S.numpy()
    rank_1e5 = int(np.sum(S_np > 1e-5))
    rank_1e3 = int(np.sum(S_np > 1e-3))
    rank_1e1 = int(np.sum(S_np > 1e-1))
    cond = S_np[0] / S_np[-1] if S_np[-1] > 0 else 0
    cum_e = np.cumsum(S_np**2) / np.sum(S_np**2)
    r80 = int(np.searchsorted(cum_e, 0.80))
    r95 = int(np.searchsorted(cum_e, 0.95))
    print(f"  rank(>1e-5)={rank_1e5}, rank(>1e-3)={rank_1e3}, rank(>0.1)={rank_1e1}")
    print(f"  cond={cond:.2e}, 80%e@{r80}, 95%e@{r95}")

    # Linearity test
    layer_cosines = {}
    for sent in TEST_SENTENCES:
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
        enc = move_batch_to_model_device(model, enc)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            actual_logits = out.logits[0, -1, :].float().cpu()
        for li, h in enumerate(out.hidden_states):
            hv = h[0, -1, :].float().cpu()
            computed = hv @ unembed.T
            c = F.cosine_similarity(actual_logits.unsqueeze(0), computed.unsqueeze(0)).item()
            if str(li) not in layer_cosines:
                layer_cosines[str(li)] = []
            layer_cosines[str(li)].append(c)

    avg_cos = {k: float(np.mean(v)) for k, v in layer_cosines.items()}
    worst_layer = min(avg_cos, key=avg_cos.get)
    worst_val = avg_cos[worst_layer]
    all_perfect = all(v > 0.999999 for v in avg_cos.values())
    print(f"  worst_layer=L{worst_layer} cos={worst_val:.8f}")
    print(f"  all_perfect_linear={all_perfect}")

    free_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    elapsed = time.time() - t0
    print(f"  time={elapsed:.1f}s")

    return {
        "unembed_shape": list(unembed.shape),
        "rank_1e5": rank_1e5,
        "rank_1e3": rank_1e3,
        "rank_0.1": rank_1e1,
        "condition_number": f"{cond:.2e}",
        "rank_80pct_energy": r80,
        "rank_95pct_energy": r95,
        "worst_layer": int(worst_layer),
        "worst_cosine": worst_val,
        "all_perfect_linear": all_perfect,
        "per_layer_cosine_sample": {k: round(v, 8) for k, v in list(avg_cos.items())[:5]},
        "elapsed_s": round(elapsed, 1),
    }


results = {}
for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
    results[mk] = run_single_model(mk)

print("\n=== CROSS-MODEL SUMMARY ===")
print(f"{'':<12} {'Qwen3':>10} {'DS7B':>10} {'GLM4':>10} {'Gemma4':>10}")
print("-" * 55)
for key in ["rank_1e5", "rank_1e3", "rank_0.1", "worst_cosine", "all_perfect_linear"]:
    vals = []
    for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        v = results.get(mk, {}).get(key, "N/A")
        vals.append(str(v) if not isinstance(v, float) else f"{v:.8f}")
    print(f"{key:<12} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

out_path = OUTPUT_DIR / f"stage593_hidden_logit_{TIMESTAMP}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"timestamp": TIMESTAMP, "models": results}, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved to {out_path}")
