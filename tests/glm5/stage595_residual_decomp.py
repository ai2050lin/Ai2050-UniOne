#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage595: 残差分解 — 属性残差+关系残差+任务残差 — 四模型对比
目标：
  1. 对名词在知识/语法/属性/联想四任务中的编码做差
  2. 分析残差结构：哪些维度承载属性、关系、任务信息
  3. 测试残差的正交性
  4. 跨模型对比
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

def get_hs(model, tokenizer, sentence, layer=-1):
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return out.hidden_states[layer][0, -1, :].float().cpu()

# 名词+四种任务模板
NOUN_TASKS = {
    "cat": {
        "neutral": "I saw a cat.",
        "attribute": "The cat is small and furry.",
        "relation": "The cat chased the mouse.",
        "knowledge": "Cats are mammals that purr.",
        "association": "I love cats, they remind me of kittens.",
    },
    "France": {
        "neutral": "France is a country.",
        "attribute": "France is large and beautiful.",
        "relation": "France borders Germany and Spain.",
        "knowledge": "Paris is the capital of France.",
        "association": "France reminds me of wine and cheese.",
    },
    "apple": {
        "neutral": "I ate an apple.",
        "attribute": "The apple is red and sweet.",
        "relation": "The apple fell from the tree.",
        "knowledge": "Apples grow in orchards.",
        "association": "Apple makes the iPhone.",
    },
    "water": {
        "neutral": "I need water.",
        "attribute": "The water is clear and cold.",
        "relation": "The water flows into the river.",
        "knowledge": "Water is H2O.",
        "association": "Water reminds me of the ocean.",
    },
    "book": {
        "neutral": "I read a book.",
        "attribute": "The book is thick and heavy.",
        "relation": "The book is on the table.",
        "knowledge": "Books contain knowledge and stories.",
        "association": "Books remind me of libraries.",
    },
}


def decompose_residuals(model, tokenizer, noun, tasks):
    """对一个名词做残差分解"""
    hs = {}
    for task_name, sentence in tasks.items():
        hs[task_name] = get_hs(model, tokenizer, sentence)
    
    neutral = hs["neutral"]
    results = {}
    
    # 残差 = 任务编码 - 中性编码
    for task_name in ["attribute", "relation", "knowledge", "association"]:
        residual = hs[task_name] - neutral
        results[f"{task_name}_norm"] = round(residual.norm().item(), 6)
        results[f"{task_name}_dir"] = residual / (residual.norm() + 1e-8)
    
    # 残差之间的正交性（余弦相似度）
    task_names = ["attribute", "relation", "knowledge", "association"]
    ortho_matrix = {}
    for i, t1 in enumerate(task_names):
        for j, t2 in enumerate(task_names):
            if i < j:
                c = cos(results[f"{t1}_dir"], results[f"{t2}_dir"])
                ortho_matrix[f"{t1}_vs_{t2}"] = round(c, 6)
    results["orthogonality"] = ortho_matrix
    
    # 残差的SVD分析
    residual_matrix = torch.stack([
        results[f"{t}_dir"] for t in task_names
    ])
    U, S, Vt = torch.linalg.svd(residual_matrix, full_matrices=False)
    results["residual_rank"] = int((S / S[0] > 0.1).sum())
    results["residual_sv_ratio_2nd"] = round((S[1] / S[0]).item(), 4) if len(S) > 1 else 0
    results["residual_sv_ratio_3rd"] = round((S[2] / S[0]).item(), 4) if len(S) > 2 else 0
    results["residual_sv_ratio_4th"] = round((S[3] / S[0]).item(), 4) if len(S) > 3 else 0
    
    # 残差在低频/高频维度的分布
    dim = neutral.shape[0]
    band_size = dim // 8
    band_energy = {}
    for bi in range(8):
        start = bi * band_size
        end = start + band_size
        energy = sum(results[f"{t}_dir"][start:end].norm().item()**2 for t in task_names)
        band_energy[f"band{bi}"] = round(energy / 4, 6)  # 归一化到每任务
    results["band_energy"] = band_energy
    
    # 残差与消歧方向的对齐
    # 用两个不同语境的词来定义消歧方向（如果有歧义的话）
    # 这里用knowledge和association作为两个极端
    disamb_dir = hs["knowledge"] - hs["association"]
    disamb_dir = disamb_dir / (disamb_dir.norm() + 1e-8)
    for t in task_names:
        align = abs(torch.dot(results[f"{t}_dir"], disamb_dir)).item()
        results[f"{t}_disamb_align"] = round(align, 6)
    
    return results


def run_model(model_key):
    print(f"\n{'='*50}\n  {model_key.upper()}\n{'='*50}")
    t0 = time.time()
    bundle = load_model_bundle(model_key)
    if not bundle:
        return {"error": f"Cannot load {model_key}"}
    model, tokenizer = bundle
    
    word_results = {}
    for noun, tasks in NOUN_TASKS.items():
        print(f"  {noun}...", end="", flush=True)
        try:
            r = decompose_residuals(model, tokenizer, noun, tasks)
            word_results[noun] = r
            print(f" rank={r['residual_rank']}, sv2={r['residual_sv_ratio_2nd']:.3f}")
        except Exception as e:
            print(f" ERROR: {e}")
            word_results[noun] = {"error": str(e)}
    
    # 汇总
    summary = {}
    valid = {k: v for k, v in word_results.items() if "error" not in v}
    
    if valid:
        # 残差范数
        for t in ["attribute", "relation", "knowledge", "association"]:
            norms = [v[f"{t}_norm"] for v in valid.values()]
            summary[f"mean_{t}_norm"] = round(float(np.mean(norms)), 6)
        
        # 正交性
        pairs = ["attribute_vs_relation", "attribute_vs_knowledge", "attribute_vs_association",
                 "relation_vs_knowledge", "relation_vs_association", "knowledge_vs_association"]
        for p in pairs:
            vals = [v["orthogonality"][p] for v in valid.values() if p in v.get("orthogonality", {})]
            if vals:
                summary[f"mean_ortho_{p}"] = round(float(np.mean(vals)), 6)
                summary[f"std_ortho_{p}"] = round(float(np.std(vals)), 6)
        
        # 残差秩
        ranks = [v["residual_rank"] for v in valid.values()]
        summary["mean_residual_rank"] = round(float(np.mean(ranks)), 2)
        
        # SV ratios
        for sv_key in ["residual_sv_ratio_2nd", "residual_sv_ratio_3rd", "residual_sv_ratio_4th"]:
            vals = [v[sv_key] for v in valid.values()]
            summary[f"mean_{sv_key}"] = round(float(np.mean(vals)), 4)
        
        # 消歧对齐
        for t in ["attribute", "relation", "knowledge", "association"]:
            vals = [v[f"{t}_disamb_align"] for v in valid.values()]
            summary[f"mean_{t}_disamb_align"] = round(float(np.mean(vals)), 6)
        
        # Band energy
        for bi in range(8):
            vals = [v["band_energy"][f"band{bi}"] for v in valid.values()]
            summary[f"mean_band{bi}_energy"] = round(float(np.mean(vals)), 6)
    
    elapsed = time.time() - t0
    summary["elapsed_s"] = round(elapsed, 1)
    
    print(f"\n  Summary:")
    print(f"    mean rank: {summary.get('mean_residual_rank', 'N/A')}")
    for t in ["attribute", "relation", "knowledge", "association"]:
        print(f"    {t}_norm: {summary.get(f'mean_{t}_norm', 0):.4f}")
    print(f"    ortho attr_rel: {summary.get('mean_ortho_attribute_vs_relation', 0):.4f}")
    print(f"    ortho attr_know: {summary.get('mean_ortho_attribute_vs_knowledge', 0):.4f}")
    print(f"    time: {elapsed:.1f}s")
    
    free_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    return {"word_results": word_results, "summary": summary}


def main():
    print("="*50 + "\n  Stage595: Residual Decomposition\n" + "="*50)
    
    results = {}
    for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        results[mk] = run_model(mk)
    
    # Cross-model
    print(f"\n{'='*50}\n  CROSS-MODEL SUMMARY\n{'='*50}")
    print(f"\n  {'Metric':<25} {'Qwen3':>10} {'DS7B':>10} {'GLM4':>10} {'Gemma4':>10}")
    print(f"  {'-'*60}")
    
    metrics = [
        ("residual_rank", "mean_residual_rank"),
        ("attr_norm", "mean_attribute_norm"),
        ("rel_norm", "mean_relation_norm"),
        ("know_norm", "mean_knowledge_norm"),
        ("assoc_norm", "mean_association_norm"),
        ("ortho_attr_rel", "mean_ortho_attribute_vs_relation"),
        ("ortho_attr_know", "mean_ortho_attribute_vs_knowledge"),
        ("ortho_attr_assoc", "mean_ortho_attribute_vs_association"),
        ("ortho_rel_know", "mean_ortho_relation_vs_knowledge"),
        ("sv_ratio_2nd", "mean_residual_sv_ratio_2nd"),
        ("sv_ratio_4th", "mean_residual_sv_ratio_4th"),
        ("band0_energy", "mean_band0_energy"),
        ("band7_energy", "mean_band7_energy"),
    ]
    
    for label, key in metrics:
        vals = []
        for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
            v = results.get(mk, {}).get("summary", {}).get(key, "N/A")
            vals.append(f"{v:.6f}" if isinstance(v, float) else str(v))
        print(f"  {label:<25} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")
    
    output = {"timestamp": TIMESTAMP, "models": results}
    out_path = OUTPUT_DIR / f"stage595_residual_decomp_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
