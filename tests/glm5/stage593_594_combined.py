#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""合并运行 stage593 + stage594"""

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

def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def get_unembed(model):
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        return model.lm_head.weight.data.float().cpu()
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens.weight.data.float().cpu()
    if hasattr(model, 'get_output_embeddings'):
        oe = model.get_output_embeddings()
        if oe and hasattr(oe, 'weight'):
            return oe.weight.data.float().cpu()
    return None

def run_all(model_key):
    print(f"\n{'='*50}\n  {model_key.upper()}\n{'='*50}")
    t0 = time.time()
    bundle = load_model_bundle(model_key)
    if not bundle:
        return {"error": f"Cannot load {model_key}"}
    model, tokenizer = bundle
    n_layers = len(discover_layers(model))
    r = {"n_layers": n_layers}

    # ===== Stage593: unembedding + linearity =====
    print("  [593] unembedding + linearity...")
    unembed = get_unembed(model)
    if unembed is not None:
        r["unembed_shape"] = list(unembed.shape)
        U, S, Vt = torch.linalg.svd(unembed, full_matrices=False)
        Sn = S.numpy()
        r["rank_1e5"] = int(np.sum(Sn > 1e-5))
        r["rank_1e3"] = int(np.sum(Sn > 1e-3))
        r["rank_0.1"] = int(np.sum(Sn > 0.1))
        r["cond"] = f"{Sn[0]/Sn[-1]:.2e}" if Sn[-1] > 0 else "inf"
        cum = np.cumsum(Sn**2) / np.sum(Sn**2)
        r["r80"] = int(np.searchsorted(cum, 0.80))
        r["r95"] = int(np.searchsorted(cum, 0.95))

        # Linearity test (2 sentences)
        for sent in ["The cat sat on the mat.", "Math is the language of science."]:
            enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
            enc = move_batch_to_model_device(model, enc)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
                actual = out.logits[0, -1, :].float().cpu()
            key = "lin_" + sent[:20]
            worst_cos, worst_li = 1.0, 0
            for li, h in enumerate(out.hidden_states):
                hv = h[0, -1, :].float().cpu()
                comp = hv @ unembed.T
                c = F.cosine_similarity(actual.unsqueeze(0), comp.unsqueeze(0)).item()
                if c < worst_cos:
                    worst_cos, worst_li = c, li
            r[key] = {"worst_cos": round(worst_cos, 8), "worst_layer": worst_li}
            print(f"    {sent[:30]}: worst L{worst_li} cos={worst_cos:.8f}")

    # ===== Stage594: embedding vs network =====
    print("  [594] embedding vs network...")
    disamb_pairs = [
        ("The river bank was muddy.", "The bank approved the loan.", "bank"),
        ("She ate a red apple.", "Apple released the iPhone.", "apple"),
        ("The factory plant employs workers.", "She watered the plant.", "plant"),
        ("He hit the nail with a hammer.", "She painted her fingernail.", "nail"),
        ("The hot spring resort.", "Spring is beautiful.", "spring"),
    ]
    disamb = []
    for s1, s2, w in disamb_pairs:
        e1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64)
        e2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64)
        e1 = move_batch_to_model_device(model, e1)
        e2 = move_batch_to_model_device(model, e2)
        with torch.no_grad():
            o1 = model(**e1, output_hidden_states=True)
            o2 = model(**e2, output_hidden_states=True)
        L0a = o1.hidden_states[0][0,-1,:].float().cpu()
        L0b = o2.hidden_states[0][0,-1,:].float().cpu()
        Lfa = o1.hidden_states[-1][0,-1,:].float().cpu()
        Lfb = o2.hidden_states[-1][0,-1,:].float().cpu()
        d_L0 = 1 - cos(L0a, L0b)
        d_Lf = 1 - cos(Lfa, Lfb)
        diff_L0 = L0a - L0b
        diff_Lf = Lfa - Lfb
        d_cos = cos(diff_L0, diff_Lf)
        proj = abs(torch.dot(diff_L0, diff_Lf)) / (torch.dot(diff_Lf, diff_Lf) + 1e-8)
        disamb.append({"word": w, "d_L0": round(d_L0,6), "d_Lf": round(d_Lf,6),
                       "gain": round(d_Lf - d_L0, 6), "dir_cos": round(d_cos, 6),
                       "proj": round(proj.item(), 6)})
    
    r["disamb"] = disamb
    if disamb:
        r["s594"] = {
            "mean_d_L0": round(np.mean([d["d_L0"] for d in disamb]), 6),
            "mean_d_Lf": round(np.mean([d["d_Lf"] for d in disamb]), 6),
            "mean_gain": round(np.mean([d["gain"] for d in disamb]), 6),
            "mean_dir_cos": round(np.mean([d["dir_cos"] for d in disamb]), 6),
            "mean_proj": round(np.mean([d["proj"] for d in disamb]), 6),
        }
    
    # Family test
    families = {
        "animal": ["cat", "dog", "bird", "fish", "horse", "mouse"],
        "country": ["France", "Japan", "Brazil", "China", "India", "Egypt"],
        "fruit": ["apple", "banana", "cherry", "grape", "mango", "peach"],
    }
    fam_r = {}
    for fn, members in families.items():
        sents = [f"The {m} is common." for m in members]
        L0s, Lfs = [], []
        for s in sents:
            e = tokenizer(s, return_tensors="pt", truncation=True, max_length=64)
            e = move_batch_to_model_device(model, e)
            with torch.no_grad():
                o = model(**e, output_hidden_states=True)
            L0s.append(o.hidden_states[0][0,-1,:].float().cpu())
            Lfs.append(o.hidden_states[-1][0,-1,:].float().cpu())
        L0m = torch.stack(L0s); Lfm = torch.stack(Lfs)
        L0c = L0m - L0m.mean(0); Lfc = Lfm - Lfm.mean(0)
        cL0 = F.cosine_similarity(L0c.unsqueeze(1), L0c.unsqueeze(0), dim=-1)
        cLf = F.cosine_similarity(Lfc.unsqueeze(1), Lfc.unsqueeze(0), dim=-1)
        mask = ~torch.eye(len(members), dtype=bool)
        _, S0, Vt0 = torch.linalg.svd(L0c, full_matrices=False)
        _, Sf, Vtf = torch.linalg.svd(Lfc, full_matrices=False)
        fam_r[fn] = {
            "L0_cos": round(cL0[mask].mean().item(), 4),
            "Lf_cos": round(cLf[mask].mean().item(), 4),
            "sub_align": round(float(torch.abs(Vt0[0] @ Vtf[0])), 4),
        }
    r["family"] = fam_r
    
    # Syntax test
    syn_pairs = [
        ("The boy ate the apple.", "The apple was eaten by the boy.", "act/pas"),
        ("She runs quickly.", "Does she run quickly?", "dec/int"),
    ]
    syn_r = []
    for s1, s2, lb in syn_pairs:
        e1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64)
        e2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64)
        e1 = move_batch_to_model_device(model, e1)
        e2 = move_batch_to_model_device(model, e2)
        with torch.no_grad():
            o1 = model(**e1, output_hidden_states=True)
            o2 = model(**e2, output_hidden_states=True)
        d_L0 = 1 - cos(o1.hidden_states[0][0,-1,:].float().cpu(), o2.hidden_states[0][0,-1,:].float().cpu())
        d_Lf = 1 - cos(o1.hidden_states[-1][0,-1,:].float().cpu(), o2.hidden_states[-1][0,-1,:].float().cpu())
        syn_r.append({"label": lb, "d_L0": round(d_L0,6), "d_Lf": round(d_Lf,6), "gain": round(d_Lf-d_L0,6)})
    r["syntax"] = syn_r

    # Print summary
    s = r.get("s594", {})
    print(f"    disamb: L0={s.get('mean_d_L0',0):.4f} Lf={s.get('mean_d_Lf',0):.4f} gain={s.get('mean_gain',0):.4f}")
    print(f"    dir_cos={s.get('mean_dir_cos',0):.4f} proj={s.get('mean_proj',0):.4f}")
    for fn, fv in fam_r.items():
        print(f"    {fn}: L0_cos={fv['L0_cos']:.4f} Lf_cos={fv['Lf_cos']:.4f} align={fv['sub_align']:.4f}")
    for sr in syn_r:
        print(f"    syntax {sr['label']}: L0={sr['d_L0']:.4f} Lf={sr['d_Lf']:.4f}")

    free_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    r["time_s"] = round(time.time() - t0, 1)
    return r

# Run all models
print("="*50 + "\n  Stage593+594 Combined\n" + "="*50)
results = {}
for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
    results[mk] = run_all(mk)

# Print cross-model table
print(f"\n{'='*50}\n  CROSS-MODEL SUMMARY\n{'='*50}")
print(f"\n  {'Metric':<20} {'Qwen3':>10} {'DS7B':>10} {'GLM4':>10} {'Gemma4':>10}")
print(f"  {'-'*55}")

rows = [
    ("rank(>1e-5)", lambda r: r.get("rank_1e5", "N/A")),
    ("rank(>0.1)", lambda r: r.get("rank_0.1", "N/A")),
    ("r95_energy", lambda r: r.get("r95", "N/A")),
    ("disamb_L0", lambda r: r.get("s594", {}).get("mean_d_L0", "N/A")),
    ("disamb_Lf", lambda r: r.get("s594", {}).get("mean_d_Lf", "N/A")),
    ("net_gain", lambda r: r.get("s594", {}).get("mean_gain", "N/A")),
    ("dir_cos", lambda r: r.get("s594", {}).get("mean_dir_cos", "N/A")),
    ("L0_proj", lambda r: r.get("s594", {}).get("mean_proj", "N/A")),
    ("worst_lin_cos", lambda r: min(
        (v["worst_cos"] for k, v in r.items() if k.startswith("lin_")),
        default="N/A")),
]
for label, fn in rows:
    vals = []
    for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        try:
            v = fn(results.get(mk, {}))
            vals.append(f"{v:.6f}" if isinstance(v, float) else str(v))
        except:
            vals.append("N/A")
    print(f"  {label:<20} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

# Family cross-model
print(f"\n  Family alignment:")
for fn in ["animal", "country", "fruit"]:
    vals = []
    for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        v = results.get(mk, {}).get("family", {}).get(fn, {}).get("sub_align", "N/A")
        vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
    print(f"    {fn:<10} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

out_path = OUTPUT_DIR / f"stage593_594_combined_{TIMESTAMP}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"timestamp": TIMESTAMP, "models": results}, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved to {out_path}")
