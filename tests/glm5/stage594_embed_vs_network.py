#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage594: embedding预编码 vs 网络加工分离 — 四模型对比
目标：
  1. 对比embedding层(L0)与末层编码的差异
  2. 测量"预编码部分"和"网络加工部分"各占多少比例
  3. 分析不同类型信息（家族、消歧、语义、语法）在L0和末层的编码差异
  4. 验证哪些能力是预编码的，哪些是网络加工产生的
模型：Qwen3 / DeepSeek7B / GLM4 / Gemma4（依次运行）
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


def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def get_hs_pair(model, tokenizer, s1, s2):
    """获取两个句子的embedding层和末层hidden state"""
    enc1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64)
    enc2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64)
    enc1 = move_batch_to_model_device(model, enc1)
    enc2 = move_batch_to_model_device(model, enc2)
    with torch.no_grad():
        out1 = model(**enc1, output_hidden_states=True)
        out2 = model(**enc2, output_hidden_states=True)
    return {
        "L0_1": out1.hidden_states[0][0, -1, :].float().cpu(),
        "L0_2": out2.hidden_states[0][0, -1, :].float().cpu(),
        "Lf_1": out1.hidden_states[-1][0, -1, :].float().cpu(),
        "Lf_2": out2.hidden_states[-1][0, -1, :].float().cpu(),
    }


def analyze_pair(model, tokenizer, s1, s2, label):
    """分析一对句子在L0和末层的编码差异"""
    hs = get_hs_pair(model, tokenizer, s1, s2)
    
    # L0和末层的cosine（同句）
    cos_same_L0 = cos(hs["L0_1"], hs["L0_2"])
    cos_same_Lf = cos(hs["Lf_1"], hs["Lf_2"])
    
    # 跨层cosine（同句L0 vs 末层）
    cos_cross_1 = cos(hs["L0_1"], hs["Lf_1"])
    cos_cross_2 = cos(hs["L0_2"], hs["Lf_2"])
    
    # 消歧信息量：1 - cos(s1, s2)
    disamb_L0 = 1 - cos_same_L0
    disamb_Lf = 1 - cos_same_Lf
    
    # 网络加工增益 = Lf消歧 - L0消歧
    network_gain = disamb_Lf - disamb_L0
    
    # 编码方向变化：cos(L0差异方向, Lf差异方向)
    diff_L0 = hs["L0_1"] - hs["L0_2"]
    diff_Lf = hs["Lf_1"] - hs["Lf_2"]
    dir_cos = cos(diff_L0, diff_Lf)
    
    # L0贡献比例：如果Lf差异方向与L0差异方向一致
    # 使用投影：|diff_L0 . diff_Lf| / |diff_Lf|^2
    proj_ratio = abs(torch.dot(diff_L0, diff_Lf)) / (torch.dot(diff_Lf, diff_Lf) + 1e-8)
    
    return {
        "label": label,
        "cos_same_L0": round(cos_same_L0, 6),
        "cos_same_Lf": round(cos_same_Lf, 6),
        "cos_cross_1": round(cos_cross_1, 6),
        "disamb_L0": round(disamb_L0, 6),
        "disamb_Lf": round(disamb_Lf, 6),
        "network_gain": round(network_gain, 6),
        "direction_cos": round(dir_cos, 6),
        "L0_projection_ratio": round(proj_ratio.item(), 6),
    }


def run_model(model_key):
    """对单个模型运行分析"""
    print(f"\n{'='*60}")
    print(f"  {model_key.upper()} — embedding预编码 vs 网络加工分离")
    print(f"{'='*60}")
    
    t0 = time.time()
    bundle = load_model_bundle(model_key)
    if bundle is None:
        return {"error": f"Cannot load {model_key}"}
    model, tokenizer = bundle
    n_layers = len(discover_layers(model))
    
    # === 测试1：消歧词对 ===
    disamb_pairs = [
        ("The river bank was muddy.", "The bank approved the loan.", "bank"),
        ("She ate a red apple.", "Apple released the iPhone.", "apple"),
        ("The factory plant employs workers.", "She watered the plant.", "plant"),
        ("The hot spring resort.", "Spring is beautiful.", "spring"),
        ("He hit the nail with a hammer.", "She painted her fingernail.", "nail"),
    ]
    
    print("  [1] 消歧词对分析:")
    disamb_results = []
    for s1, s2, word in disamb_pairs:
        r = analyze_pair(model, tokenizer, s1, s2, word)
        disamb_results.append(r)
        print(f"    {word}: L0_disamb={r['disamb_L0']:.4f}, Lf_disamb={r['disamb_Lf']:.4f}, "
              f"gain={r['network_gain']:.4f}, dir_cos={r['direction_cos']:.4f}, "
              f"L0_proj={r['L0_projection_ratio']:.4f}")
    
    # === 测试2：同义句对 ===
    para_pairs = [
        ("The cat is on the mat.", "A feline rests upon the rug.", "cat=on"),
        ("The weather is nice today.", "It is beautiful outside.", "weather"),
        ("He is very happy.", "She feels joyful.", "happy"),
        ("The book is interesting.", "This publication is fascinating.", "book"),
    ]
    
    print("  [2] 同义句对分析:")
    para_results = []
    for s1, s2, label in para_pairs:
        r = analyze_pair(model, tokenizer, s1, s2, label)
        para_results.append(r)
        print(f"    {label}: L0_cos={r['cos_same_L0']:.4f}, Lf_cos={r['cos_same_Lf']:.4f}, "
              f"dir_cos={r['direction_cos']:.4f}")
    
    # === 测试3：家族成员 ===
    family_groups = {
        "animal": ["cat", "dog", "bird", "fish", "horse", "mouse"],
        "country": ["France", "Japan", "Brazil", "China", "India", "Egypt"],
        "fruit": ["apple", "banana", "cherry", "grape", "mango", "peach"],
    }
    
    print("  [3] 家族成员embedding vs 末层:")
    family_results = {}
    for fam_name, members in family_groups.items():
        sentences = [f"The {m} is common." for m in members]
        encs = [tokenizer(s, return_tensors="pt", truncation=True, max_length=64) for s in sentences]
        encs = [move_batch_to_model_device(model, e) for e in encs]
        
        L0_vecs, Lf_vecs = [], []
        with torch.no_grad():
            for e in encs:
                out = model(**e, output_hidden_states=True)
                L0_vecs.append(out.hidden_states[0][0, -1, :].float().cpu())
                Lf_vecs.append(out.hidden_states[-1][0, -1, :].float().cpu())
        
        # 计算家族内cosine矩阵在L0和末层
        L0_mat = torch.stack([v - torch.stack(L0_vecs).mean(0) for v in L0_vecs])
        Lf_mat = torch.stack([v - torch.stack(Lf_vecs).mean(0) for v in Lf_vecs])
        
        L0_cos_mat = F.cosine_similarity(L0_mat.unsqueeze(1), L0_mat.unsqueeze(0), dim=-1)
        Lf_cos_mat = F.cosine_similarity(Lf_mat.unsqueeze(1), Lf_mat.unsqueeze(0), dim=-1)
        
        # 平均家族内cosine（排除对角线）
        mask = ~torch.eye(len(members), dtype=bool)
        L0_fam_cos = L0_cos_mat[mask].mean().item()
        Lf_fam_cos = Lf_cos_mat[mask].mean().item()
        
        # 编码空间方向保持度
        U_l0, S_l0, Vt_l0 = torch.linalg.svd(L0_mat, full_matrices=False)
        U_lf, S_lf, Vt_lf = torch.linalg.svd(Lf_mat, full_matrices=False)
        # 子空间角度
        cos_subspace = float(torch.abs(Vt_l0[0] @ Vt_lf[0]))
        
        family_results[fam_name] = {
            "L0_fam_cosine": round(L0_fam_cos, 6),
            "Lf_fam_cosine": round(Lf_fam_cos, 6),
            "subspace_alignment": round(cos_subspace, 6),
        }
        print(f"    {fam_name}: L0_fam_cos={L0_fam_cos:.4f}, Lf_fam_cos={Lf_fam_cos:.4f}, "
              f"subspace_align={cos_subspace:.4f}")
    
    # === 测试4：句法结构 ===
    syntax_pairs = [
        ("The boy ate the apple.", "The apple was eaten by the boy.", "active/passive"),
        ("She runs quickly.", "Does she run quickly?", "declarative/interrogative"),
        ("The big dog barks.", "The dog that is big barks.", "adj/relative_clause"),
    ]
    
    print("  [4] 句法结构分析:")
    syntax_results = []
    for s1, s2, label in syntax_pairs:
        r = analyze_pair(model, tokenizer, s1, s2, label)
        syntax_results.append(r)
        print(f"    {label}: L0_cos={r['cos_same_L0']:.4f}, Lf_cos={r['cos_same_Lf']:.4f}, "
              f"disamb_L0={r['disamb_L0']:.4f}, disamb_Lf={r['disamb_Lf']:.4f}")
    
    # 汇总
    summary = {}
    
    # 消歧汇总
    if disamb_results:
        summary["mean_disamb_L0"] = round(np.mean([r["disamb_L0"] for r in disamb_results]), 6)
        summary["mean_disamb_Lf"] = round(np.mean([r["disamb_Lf"] for r in disamb_results]), 6)
        summary["mean_network_gain"] = round(np.mean([r["network_gain"] for r in disamb_results]), 6)
        summary["mean_L0_proj_ratio"] = round(np.mean([r["L0_projection_ratio"] for r in disamb_results]), 6)
        summary["mean_direction_cos"] = round(np.mean([r["direction_cos"] for r in disamb_results]), 6)
        # L0贡献的消歧比例
        total = summary["mean_disamb_L0"] + summary["mean_network_gain"]
        summary["L0_disamb_fraction"] = round(summary["mean_disamb_L0"] / max(total, 1e-8), 4)
    
    # 家族汇总
    if family_results:
        summary["mean_family_L0_cos"] = round(np.mean([v["L0_fam_cosine"] for v in family_results.values()]), 6)
        summary["mean_family_Lf_cos"] = round(np.mean([v["Lf_fam_cosine"] for v in family_results.values()]), 6)
        summary["mean_subspace_alignment"] = round(np.mean([v["subspace_alignment"] for v in family_results.values()]), 6)
    
    # 句法汇总
    if syntax_results:
        summary["mean_syntax_disamb_L0"] = round(np.mean([r["disamb_L0"] for r in syntax_results]), 6)
        summary["mean_syntax_disamb_Lf"] = round(np.mean([r["disamb_Lf"] for r in syntax_results]), 6)
    
    elapsed = time.time() - t0
    summary["elapsed_s"] = round(elapsed, 1)
    summary["n_layers"] = n_layers
    
    print(f"\n  汇总:")
    print(f"    消歧L0: {summary.get('mean_disamb_L0', 0):.4f}, Lf: {summary.get('mean_disamb_Lf', 0):.4f}")
    print(f"    网络增益: {summary.get('mean_network_gain', 0):.4f}")
    print(f"    L0消歧占比: {summary.get('L0_disamb_fraction', 0)*100:.1f}%")
    print(f"    L0投影比: {summary.get('mean_L0_proj_ratio', 0):.4f}")
    print(f"    家族L0 cos: {summary.get('mean_family_L0_cos', 0):.4f}")
    print(f"    子空间对齐: {summary.get('mean_subspace_alignment', 0):.4f}")
    print(f"    句法消歧L0: {summary.get('mean_syntax_disamb_L0', 0):.4f}")
    print(f"    句法消歧Lf: {summary.get('mean_syntax_disamb_Lf', 0):.4f}")
    
    free_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "disamb": disamb_results,
        "para": para_results,
        "family": family_results,
        "syntax": syntax_results,
        "summary": summary,
    }


def main():
    print("=" * 60)
    print("  Stage594: embedding预编码 vs 网络加工分离")
    print("=" * 60)
    
    results = {}
    for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        results[mk] = run_model(mk)
    
    # 跨模型对比
    print(f"\n{'='*60}")
    print("  CROSS-MODEL SUMMARY")
    print(f"{'='*60}")
    
    metrics = [
        ("disamb_L0", "mean_disamb_L0"),
        ("disamb_Lf", "mean_disamb_Lf"),
        ("network_gain", "mean_network_gain"),
        ("L0_disamb%", "L0_disamb_fraction"),
        ("L0_proj", "mean_L0_proj_ratio"),
        ("dir_cos", "mean_direction_cos"),
        ("family_L0_cos", "mean_family_L0_cos"),
        ("family_Lf_cos", "mean_family_Lf_cos"),
        ("subspace_align", "mean_subspace_alignment"),
        ("syntax_L0", "mean_syntax_disamb_L0"),
        ("syntax_Lf", "mean_syntax_disamb_Lf"),
    ]
    
    print(f"\n  {'Metric':<20} {'Qwen3':>10} {'DS7B':>10} {'GLM4':>10} {'Gemma4':>10}")
    print(f"  {'-'*55}")
    for label, key in metrics:
        vals = []
        for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
            v = results.get(mk, {}).get("summary", {}).get(key, "N/A")
            vals.append(f"{v:.6f}" if isinstance(v, float) else str(v))
        print(f"  {label:<20} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")
    
    # 保存
    output = {"timestamp": TIMESTAMP, "models": results}
    out_path = OUTPUT_DIR / f"stage594_embed_vs_network_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
