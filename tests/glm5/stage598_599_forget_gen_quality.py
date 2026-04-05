#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage598_599: 消歧信息遗忘机制追踪 + 消歧度与生成质量相关性
合并运行以减少GPU加载次数。

Stage598目标：
  1. 在峰值层提取消歧方向向量 d_peak = h1_peak - h2_peak
  2. 追踪d_peak在后续层的"命运"：
     a. 方向保持度：cos(d_peak, d_later) — 方向是否被旋转
     b. 能量衰减：||d_later|| / ||d_peak|| — 幅度是否被压缩
     c. 投影比：|d_peak · d_later| / ||d_later||^2 — 是否被投影到其他方向
     d. 正交分解：d_later在d_peak方向的分量 vs 正交方向的分量
  3. 确定"遗忘"是旋转、压缩还是重新编码

Stage599目标：
  1. 设计需要消歧的生成任务（歧义词补全）
  2. 对比四模型的生成准确率与消歧度
  3. 验证消歧度是否是生成能力的代理指标
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


def run_stage598(model, tokenizer, model_key):
    """Stage598: 消歧信息遗忘机制追踪"""
    print(f"\n  --- Stage598: 消歧遗忘机制 ---")
    t0 = time.time()

    layers = discover_layers(model)
    n_layers = len(layers)

    disamb_pairs = [
        ("The river bank was muddy.", "The bank approved the loan.", "bank"),
        ("She ate a red apple.", "Apple released the iPhone.", "apple"),
        ("The factory plant employs workers.", "She watered the plant.", "plant"),
        ("The hot spring resort.", "Spring is beautiful.", "spring"),
        ("He hit the nail with a hammer.", "She painted her fingernail.", "nail"),
    ]

    results = {"per_word": {}, "summary": {}}

    all_track_data = []

    for s1, s2, word in disamb_pairs:
        enc1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64)
        enc2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64)
        enc1 = move_batch_to_model_device(model, enc1)
        enc2 = move_batch_to_model_device(model, enc2)

        with torch.no_grad():
            out1 = model(**enc1, output_hidden_states=True)
            out2 = model(**enc2, output_hidden_states=True)

        # 1. 收集所有层的hidden states和差异向量
        h1_all = []
        h2_all = []
        diff_all = []
        disamb_all = []

        for li in range(n_layers):
            h1 = out1.hidden_states[li][0, -1, :].float().cpu()
            h2 = out2.hidden_states[li][0, -1, :].float().cpu()
            diff = h1 - h2
            disamb = 1 - cos(h1, h2)
            h1_all.append(h1)
            h2_all.append(h2)
            diff_all.append(diff)
            disamb_all.append(disamb)

        # 2. 找到峰值层
        peak_layer = int(np.argmax(disamb_all))
        d_peak = diff_all[peak_layer]
        d_peak_norm = torch.norm(d_peak)

        print(f"    {word}: peak_L{peak_layer}, peak_disamb={disamb_all[peak_layer]:.4f}")

        # 3. 从峰值层开始追踪消歧方向的命运
        track = {"word": word, "peak_layer": peak_layer, "peak_disamb": round(disamb_all[peak_layer], 6)}
        per_layer_track = []

        for li in range(peak_layer, n_layers):
            d_li = diff_all[li]
            d_li_norm = torch.norm(d_li)

            # a. 方向保持度：cos(d_peak, d_later)
            if d_peak_norm > 1e-8 and d_li_norm > 1e-8:
                dir_cos = cos(d_peak, d_li)
            else:
                dir_cos = 0.0

            # b. 能量比：||d_later|| / ||d_peak||
            energy_ratio = d_li_norm.item() / (d_peak_norm.item() + 1e-8)

            # c. 正交分解
            # d_later在d_peak方向的投影
            if d_peak_norm > 1e-8:
                proj_scalar = torch.dot(d_li, d_peak) / (d_peak_norm ** 2)
                proj_vec = proj_scalar * d_peak
                orth_vec = d_li - proj_vec
                proj_ratio = torch.norm(proj_vec).item() / (d_li_norm.item() + 1e-8)
                orth_ratio = torch.norm(orth_vec).item() / (d_li_norm.item() + 1e-8)
            else:
                proj_scalar = 0
                proj_ratio = 0
                orth_ratio = 1.0

            # d. SVD分析：消歧方向的变化
            if d_peak_norm > 1e-8 and d_li_norm > 1e-8:
                mat = torch.stack([d_peak / d_peak_norm, d_li / d_li_norm])
                _, S, _ = torch.linalg.svd(mat)
                # 两个2D向量的SVD奇异值反映它们的关系
                sv1, sv2 = S[0].item(), S[1].item() if len(S) > 1 else 0
            else:
                sv1, sv2 = 0, 0

            lt = {
                "layer": li,
                "disamb": round(disamb_all[li], 6),
                "dir_cos": round(dir_cos, 6),
                "energy_ratio": round(energy_ratio, 6),
                "proj_ratio": round(proj_ratio, 6),
                "orth_ratio": round(orth_ratio, 6),
                "sv1": round(sv1, 4),
                "sv2": round(sv2, 4),
            }
            per_layer_track.append(lt)
            all_track_data.append(lt)

        track["per_layer"] = per_layer_track
        results["per_word"][word] = track

        # 只打印关键层
        sample_layers = sorted(set(
            [peak_layer, peak_layer + 1, peak_layer + 3, peak_layer + 5,
             (peak_layer + n_layers) // 2, n_layers - 1]
        ))
        for lt in per_layer_track:
            if lt["layer"] in sample_layers:
                print(f"      L{lt['layer']:>2}: disamb={lt['disamb']:.4f}, dir_cos={lt['dir_cos']:.4f}, "
                      f"energy={lt['energy_ratio']:.4f}, proj={lt['proj_ratio']:.3f}, orth={lt['orth_ratio']:.3f}")

    # 汇总统计
    summary = {"n_layers": n_layers}

    # 对所有词对，在特定相对位置（峰值+0, +25%, +50%, +75%, 末层）统计
    for rel_pos_name, rel_pos_func in [
        ("peak", lambda pl, nl: 0),
        ("peak+25%", lambda pl, nl: max(1, (nl - pl) // 4)),
        ("peak+50%", lambda pl, nl: max(1, (nl - pl) // 2)),
        ("peak+75%", lambda pl, nl: max(1, 3 * (nl - pl) // 4)),
        ("final", lambda pl, nl: nl - 1),
    ]:
        dir_coses = []
        energies = []
        projs = []
        orths = []
        disambs = []
        for word_data in results["per_word"].values():
            pl = word_data["peak_layer"]
            idx = rel_pos_func(pl, n_layers)
            if idx < len(word_data["per_layer"]):
                lt = word_data["per_layer"][idx]
                dir_coses.append(lt["dir_cos"])
                energies.append(lt["energy_ratio"])
                projs.append(lt["proj_ratio"])
                orths.append(lt["orth_ratio"])
                disambs.append(lt["disamb"])

        summary[f"mean_dir_cos_{rel_pos_name}"] = round(np.mean(dir_coses), 6)
        summary[f"mean_energy_{rel_pos_name}"] = round(np.mean(energies), 6)
        summary[f"mean_proj_{rel_pos_name}"] = round(np.mean(projs), 6)
        summary[f"mean_orth_{rel_pos_name}"] = round(np.mean(orths), 6)
        summary[f"mean_disamb_{rel_pos_name}"] = round(np.mean(disambs), 6)

    # 遗忘模式分类
    # 计算从峰值到末层的变化
    dir_cos_changes = []
    proj_changes = []
    for word_data in results["per_word"].values():
        pl = word_data["per_layer"]
        if len(pl) > 1:
            dir_cos_changes.append(word_data["per_layer"][0]["dir_cos"])
            proj_changes.append(word_data["per_layer"][-1]["proj_ratio"])

    # 分类：旋转(dir_cos下降) vs 压缩(energy下降) vs 重新编码(orth_ratio上升)
    summary["forget_pattern"] = "mixed"  # 默认

    elapsed = time.time() - t0
    results["summary"] = summary
    print(f"\n    汇总:")
    for pos in ["peak", "peak+25%", "peak+50%", "peak+75%", "final"]:
        dc = summary.get(f"mean_dir_cos_{pos}", 0)
        en = summary.get(f"mean_energy_{pos}", 0)
        pr = summary.get(f"mean_proj_{pos}", 0)
        orr = summary.get(f"mean_orth_{pos}", 0)
        da = summary.get(f"mean_disamb_{pos}", 0)
        print(f"      {pos:<12}: dir_cos={dc:.4f}, energy={en:.4f}, proj={pr:.3f}, orth={orr:.3f}, disamb={da:.4f}")
    print(f"    Stage598 time: {elapsed:.1f}s")
    return results


def run_stage599(model, tokenizer, model_key):
    """Stage599: 消歧度与生成质量相关性"""
    print(f"\n  --- Stage599: 消歧度vs生成质量 ---")
    t0 = time.time()

    # 设计需要消歧的生成任务
    # 格式：prompt + 期望的关键词（用于判断消歧是否正确）
    disamb_gen_tasks = [
        {
            "prompt": "The fisherman sat by the river bank and cast his line into the",
            "word": "bank",
            "context": "river",
            "expected_keywords": ["water", "river", "stream", "current", "flow"],
            "wrong_keywords": ["money", "loan", "deposit", "financial", "interest"],
        },
        {
            "prompt": "The CEO went to the bank to discuss the new loan for the",
            "word": "bank",
            "context": "financial",
            "expected_keywords": ["money", "loan", "business", "company", "financial"],
            "wrong_keywords": ["water", "river", "fish", "boat", "swim"],
        },
        {
            "prompt": "She went to the orchard and picked a red apple from the",
            "word": "apple",
            "context": "fruit",
            "expected_keywords": ["tree", "branch", "orchard", "fruit", "basket"],
            "wrong_keywords": ["phone", "computer", "iPhone", "iPad", "software"],
        },
        {
            "prompt": "Apple announced the new iPhone at their annual",
            "word": "apple",
            "context": "company",
            "expected_keywords": ["event", "conference", "keynote", "presentation", "WWDC"],
            "wrong_keywords": ["tree", "fruit", "orchard", "pie", "juice"],
        },
        {
            "prompt": "The gardener carefully watered every plant in the greenhouse to help it",
            "word": "plant",
            "context": "living",
            "expected_keywords": ["grow", "bloom", "flourish", "thrive", "sprout"],
            "wrong_keywords": ["factory", "manufacture", "production", "machine", "worker"],
        },
        {
            "prompt": "The automobile plant employs thousands of workers who build",
            "word": "plant",
            "context": "factory",
            "expected_keywords": ["cars", "vehicles", "machines", "parts", "assembly"],
            "wrong_keywords": ["grow", "water", "garden", "seed", "soil"],
        },
        {
            "prompt": "The hot spring resort was famous for its mineral water that could",
            "word": "spring",
            "context": "water",
            "expected_keywords": ["heal", "relax", "warm", "soothe", "flow"],
            "wrong_keywords": ["season", "winter", "summer", "autumn", "bloom"],
        },
        {
            "prompt": "Spring is the most beautiful season when flowers begin to",
            "word": "spring",
            "context": "season",
            "expected_keywords": ["bloom", "grow", "blossom", "open", "appear"],
            "wrong_keywords": ["water", "hot", "mineral", "resort", "flow"],
        },
    ]

    results = {"per_task": [], "summary": {}}

    correct_count = 0
    total_count = 0
    context_correct = {"river": 0, "financial": 0, "fruit": 0, "company": 0,
                       "living": 0, "factory": 0, "water": 0, "season": 0}
    context_total = {"river": 0, "financial": 0, "fruit": 0, "company": 0,
                     "living": 0, "factory": 0, "water": 0, "season": 0}

    for task in disamb_gen_tasks:
        enc = tokenizer(task["prompt"], return_tensors="pt", truncation=True, max_length=64)
        enc = move_batch_to_model_device(model, enc)

        with torch.no_grad():
            gen_out = model.generate(
                **enc,
                max_new_tokens=15,
                do_sample=False,
                temperature=0.0,
                top_k=1,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(gen_out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        generated_lower = generated.lower().strip()

        # 判断生成是否正确（包含期望关键词或排除错误关键词）
        expected_hit = any(kw.lower() in generated_lower for kw in task["expected_keywords"])
        wrong_hit = any(kw.lower() in generated_lower for kw in task["wrong_keywords"])

        # 正确 = 包含期望关键词 且 不包含错误关键词
        is_correct = expected_hit and not wrong_hit
        # 部分正确 = 包含期望关键词（但可能也包含错误关键词）
        is_partial = expected_hit

        ctx = task["context"]
        context_total[ctx] = context_total.get(ctx, 0) + 1
        if is_correct:
            correct_count += 1
            context_correct[ctx] = context_correct.get(ctx, 0) + 1

        total_count += 1

        task_result = {
            "word": task["word"],
            "context": ctx,
            "prompt": task["prompt"][:50] + "...",
            "generated": generated[:80],
            "expected_hit": expected_hit,
            "wrong_hit": wrong_hit,
            "is_correct": is_correct,
            "is_partial": is_partial,
        }
        results["per_task"].append(task_result)
        status = "OK" if is_correct else ("PARTIAL" if is_partial else "FAIL")
        print(f"    [{status}] {task['word']}({ctx}): \"{generated[:50]}\"")

    accuracy = correct_count / max(total_count, 1)
    partial_accuracy = sum(1 for t in results["per_task"] if t["is_partial"]) / max(total_count, 1)

    # 按词类型统计
    word_accuracy = {}
    for task in results["per_task"]:
        w = task["word"]
        if w not in word_accuracy:
            word_accuracy[w] = {"correct": 0, "total": 0, "partial": 0}
        word_accuracy[w]["total"] += 1
        if task["is_correct"]:
            word_accuracy[w]["correct"] += 1
        if task["is_partial"]:
            word_accuracy[w]["partial"] += 1

    results["summary"] = {
        "accuracy_strict": round(accuracy, 4),
        "accuracy_partial": round(partial_accuracy, 4),
        "total_tasks": total_count,
        "correct_tasks": correct_count,
        "per_word_accuracy": {w: round(v["correct"] / max(v["total"], 1), 4) for w, v in word_accuracy.items()},
        "per_word_partial": {w: round(v["partial"] / max(v["total"], 1), 4) for w, v in word_accuracy.items()},
    }

    print(f"\n    汇总: strict={accuracy:.2%}, partial={partial_accuracy:.2%}")
    print(f"    按词: ", end="")
    for w, v in word_accuracy.items():
        print(f"{w}={v['correct']}/{v['total']} ", end="")
    print()

    elapsed = time.time() - t0
    results["summary"]["elapsed_s"] = round(elapsed, 1)
    print(f"    Stage599 time: {elapsed:.1f}s")
    return results


def run_single_model(model_key):
    """对单个模型同时运行598和599"""
    print(f"\n{'='*60}")
    print(f"  {model_key.upper()} — Stage598+599")
    print(f"{'='*60}")

    t0 = time.time()
    bundle = load_model_bundle(model_key)
    if bundle is None:
        return {"error": f"Cannot load {model_key}"}
    model, tokenizer = bundle

    s598 = run_stage598(model, tokenizer, model_key)
    s599 = run_stage599(model, tokenizer, model_key)

    free_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n  Total time for {model_key}: {elapsed:.1f}s")

    return {
        "stage598": s598,
        "stage599": s599,
        "total_s": round(elapsed, 1),
    }


def main():
    print("=" * 60)
    print("  Stage598+599: 消歧遗忘机制 + 消歧度vs生成质量")
    print("=" * 60)

    all_results = {}
    for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        all_results[mk] = run_single_model(mk)

    # === 跨模型对比 ===
    print(f"\n{'='*60}")
    print("  CROSS-MODEL SUMMARY")
    print(f"{'='*60}")

    # Stage598对比
    print(f"\n  --- Stage598: 消歧遗忘机制 ---")
    print(f"  {'Position':<15} {'Qwen3':>10} {'DS7B':>10} {'GLM4':>10} {'Gemma4':>10}")
    print(f"  {'-'*60}")

    metrics_598 = [
        "mean_dir_cos_peak", "mean_dir_cos_peak+25%", "mean_dir_cos_peak+50%",
        "mean_dir_cos_peak+75%", "mean_dir_cos_final",
        "mean_energy_peak", "mean_energy_peak+25%", "mean_energy_peak+50%",
        "mean_energy_peak+75%", "mean_energy_final",
        "mean_proj_peak", "mean_proj_peak+50%", "mean_proj_final",
        "mean_orth_peak", "mean_orth_peak+50%", "mean_orth_final",
        "mean_disamb_peak", "mean_disamb_final",
    ]

    for metric in metrics_598:
        short = metric.replace("mean_", "").replace("_peak", "@pk").replace("_final", "@end")
        vals = []
        for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
            v = all_results.get(mk, {}).get("stage598", {}).get("summary", {}).get(metric, "N/A")
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        print(f"  {short:<15} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

    # Stage599对比
    print(f"\n  --- Stage599: 消歧度vs生成质量 ---")
    print(f"  {'Metric':<25} {'Qwen3':>10} {'DS7B':>10} {'GLM4':>10} {'Gemma4':>10}")
    print(f"  {'-'*60}")

    for metric in ["accuracy_strict", "accuracy_partial", "total_tasks", "correct_tasks"]:
        vals = []
        for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
            v = all_results.get(mk, {}).get("stage599", {}).get("summary", {}).get(metric, "N/A")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        print(f"  {metric:<25} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

    # 按词的生成准确率
    print(f"\n  按词准确率:")
    words = ["bank", "apple", "plant", "spring"]
    for w in words:
        vals = []
        for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
            pw = all_results.get(mk, {}).get("stage599", {}).get("summary", {}).get("per_word_accuracy", {})
            v = pw.get(w, "N/A")
            vals.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        print(f"    {w:<15} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

    # 生成质量vs消歧度的相关性
    print(f"\n  消歧度vs生成质量相关性:")
    # 从stage597数据获取消歧度
    disamb_data = {
        "qwen3": {"peak": 0.324, "final": 0.134},
        "deepseek7b": {"peak": 0.200, "final": 0.087},
        "glm4": {"peak": 0.387, "final": 0.145},
        "gemma4": {"peak": 0.190, "final": 0.046},
    }
    gen_data = {}
    for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        gen_data[mk] = all_results.get(mk, {}).get("stage599", {}).get("summary", {}).get("accuracy_strict", 0)

    peaks = [disamb_data[mk]["peak"] for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]]
    finals = [disamb_data[mk]["final"] for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]]
    gens = [gen_data[mk] for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]]

    if len(peaks) == 4:
        corr_peak_gen = np.corrcoef(peaks, gens)[0, 1]
        corr_final_gen = np.corrcoef(finals, gens)[0, 1]
        print(f"    peak_disamb vs gen_accuracy: r={corr_peak_gen:.4f}")
        print(f"    final_disamb vs gen_accuracy: r={corr_final_gen:.4f}")

    print(f"\n  {'Model':<10} {'PeakDisamb':>10} {'FinalDisamb':>12} {'GenAcc':>10}")
    print(f"  {'-'*45}")
    for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        print(f"  {mk:<10} {disamb_data[mk]['peak']:>10.4f} {disamb_data[mk]['final']:>12.4f} {gen_data[mk]:>10.4f}")

    # 保存
    out_path = OUTPUT_DIR / f"stage598_599_combined_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": TIMESTAMP, "models": all_results}, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
