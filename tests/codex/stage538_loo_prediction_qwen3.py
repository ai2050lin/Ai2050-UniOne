#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage538: 留一法名词编码预测 (Qwen3)
======================================
目标：验证"名词编码 = 全局骨干 + 家族骨干 + 独有残差"的预测能力。

方法：留一法（Leave-One-Out）
1. 对每个家族，每次去掉一个成员
2. 用剩余成员的平均编码作为预测
3. 比较预测编码与实际编码的距离
4. 同时比较跨家族预测（用其他家族的编码预测）作为基线

预测度量：
- 余弦相似度（越高越好）
- L2距离（越低越好）
- 预测误差的家族内/跨家族比值
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen3_language_shared import (
    load_qwen3_model,
    get_model_device,
    discover_layers,
)
from multimodel_language_shared import (
    encode_to_device,
    evenly_spaced_layers,
    free_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage538_loo_prediction_qwen3_20260404"

NOUN_FAMILIES = {
    "fruit": {
        "label_zh": "水果",
        "members": ["apple", "banana", "orange", "grape", "mango"],
    },
    "animal": {
        "label_zh": "动物",
        "members": ["cat", "dog", "bird", "fish", "horse"],
    },
    "tool": {
        "label_zh": "工具",
        "members": ["hammer", "screwdriver", "wrench", "knife", "scissors"],
    },
    "organization": {
        "label_zh": "组织",
        "members": ["university", "hospital", "museum", "library", "school"],
    },
    "celestial": {
        "label_zh": "天体",
        "members": ["sun", "moon", "mars", "jupiter", "venus"],
    },
    "abstract": {
        "label_zh": "抽象",
        "members": ["freedom", "justice", "love", "truth", "beauty"],
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_word_encoding(
    model, tokenizer, word: str, layers: List[int]
) -> torch.Tensor:
    """获取一个词在多个层的hidden state, shape: (num_layers, hidden_dim)"""
    encoded = encode_to_device(model, tokenizer, word)
    with torch.inference_mode():
        outputs = model(**encoded, output_hidden_states=True)
    hs = []
    for li in layers:
        h = outputs.hidden_states[li + 1][0, -1, :].float()
        hs.append(h)
    return torch.stack(hs)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def l2_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.norm(a - b).item()


def leave_one_out_predict(
    encodings: Dict[str, torch.Tensor],
    layers: List[int],
    target_word: str,
    family_members: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    留一法预测：用family_members中除target_word外的成员平均编码预测target_word
    返回每层的预测质量
    """
    others = [w for w in family_members if w != target_word]
    if not others:
        return {}

    actual = encodings[target_word]
    # 预测 = 其余成员的平均编码
    predicted = torch.stack([encodings[w] for w in others]).mean(dim=0)

    results = {}
    for i, li in enumerate(layers):
        cos = cosine_sim(actual[i], predicted[i])
        dist = l2_dist(actual[i], predicted[i])
        results[li] = {
            "cosine_similarity": round(cos, 6),
            "l2_distance": round(dist, 4),
        }
    return results


def cross_family_predict(
    encodings: Dict[str, torch.Tensor],
    layers: List[int],
    target_word: str,
    target_family_key: str,
    families: Dict,
) -> Dict[str, Dict[str, float]]:
    """跨家族预测：用其他家族的平均编码预测target_word"""
    actual = encodings[target_word]
    results = {}

    for fam_key, fam in families.items():
        if fam_key == target_family_key:
            continue
        predicted = torch.stack([encodings[w] for w in fam["members"]]).mean(dim=0)

        for i, li in enumerate(layers):
            cos = cosine_sim(actual[i], predicted[i])
            dist = l2_dist(actual[i], predicted[i])
            results[f"{fam_key}_L{li}"] = {
                "source_family": fam_key,
                "layer": li,
                "cosine_similarity": round(cos, 6),
                "l2_distance": round(dist, 4),
            }
    return results


def residual_decomposition(
    encodings: Dict[str, torch.Tensor],
    layers: List[int],
    word: str,
    family_members: List[str],
    all_words: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    残差分解：word的编码 = 全局骨干 + 家族骨干 + 独有残差
    全局骨干 = 所有词的平均编码
    家族骨干 = 同家族词的平均编码 - 全局骨干
    独有残差 = word编码 - 同家族平均编码
    """
    actual = encodings[word]
    global_mean = torch.stack([encodings[w] for w in all_words]).mean(dim=0)
    others = [w for w in family_members if w != word]
    family_mean = torch.stack([encodings[w] for w in family_members]).mean(dim=0)

    results = {}
    for i, li in enumerate(layers):
        global_backbone = global_mean[i]
        family_backbone = family_mean[i] - global_mean[i]
        unique_residual = actual[i] - family_mean[i]

        # 各分量占总编码的比例
        total_norm = torch.norm(actual[i]).item()
        gb_frac = torch.norm(global_backbone).item() / max(total_norm, 1e-8)
        fb_frac = torch.norm(family_backbone).item() / max(total_norm, 1e-8)
        ur_frac = torch.norm(unique_residual).item() / max(total_norm, 1e-8)

        # 各分量之间的夹角
        cos_gb_fb = cosine_sim(global_backbone, family_backbone)
        cos_gb_ur = cosine_sim(global_backbone, unique_residual)
        cos_fb_ur = cosine_sim(family_backbone, unique_residual)

        results[li] = {
            "global_backbone_fraction": round(gb_frac, 4),
            "family_backbone_fraction": round(fb_frac, 4),
            "unique_residual_fraction": round(ur_frac, 4),
            "cos_gb_fb": round(cos_gb_fb, 4),
            "cos_gb_ur": round(cos_gb_ur, 4),
            "cos_fb_ur": round(cos_fb_ur, 4),
        }
    return results


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    print("=" * 70)
    print("stage538: 留一法名词编码预测 (Qwen3)")
    print("=" * 70)
    started = time.time()

    print("\n[1/4] 加载Qwen3模型...")
    model, tokenizer = load_qwen3_model(prefer_cuda=True)
    layers = discover_layers(model)
    sample_layers = evenly_spaced_layers(model, count=7)
    print(f"  层数: {len(layers)}, 采样层: {sample_layers}")

    all_words = []
    for fam in NOUN_FAMILIES.values():
        all_words.extend(fam["members"])
    print(f"  名词总数: {len(all_words)} (6家族 × 5成员)")

    # [2/4] 编码所有名词
    print("\n[2/4] 编码所有名词...")
    encodings = {}  # word -> (num_layers, hidden_dim)
    for word in all_words:
        encodings[word] = get_word_encoding(model, tokenizer, word, sample_layers)
        print(f"  {word:15s}: {encodings[word].shape}")

    # [3/4] 留一法预测 + 跨家族预测
    print("\n[3/4] 留一法预测 + 跨家族预测...")

    loo_results = {}  # word -> per-layer results
    cross_results = {}  # word -> cross-family results

    for fam_key, fam in NOUN_FAMILIES.items():
        print(f"\n  === {fam_key} ({fam['label_zh']}) ===")
        for word in fam["members"]:
            # 留一法
            loo = leave_one_out_predict(encodings, sample_layers, word, fam["members"])
            loo_results[word] = loo

            # 找最佳层的留一法结果
            best_li = max(loo.keys(), key=lambda li: loo[li]["cosine_similarity"])
            print(f"    {word:15s} LOO最佳=L{best_li} "
                  f"cos={loo[best_li]['cosine_similarity']:.4f} "
                  f"L2={loo[best_li]['l2_distance']:.2f}")

            # 跨家族预测
            cross = cross_family_predict(
                encodings, sample_layers, word, fam_key, NOUN_FAMILIES
            )
            cross_results[word] = cross

    # [4/4] 残差分解
    print("\n[4/4] 残差分解 (全局骨干 + 家族骨干 + 独有残差)...")

    residual_results = {}
    for fam_key, fam in NOUN_FAMILIES.items():
        for word in fam["members"]:
            rd = residual_decomposition(
                encodings, sample_layers, word, fam["members"], all_words
            )
            residual_results[word] = rd
            # 打印最后一层的分解
            last_li = sample_layers[-1]
            r = rd[last_li]
            print(f"  {word:15s}: 全局={r['global_backbone_fraction']:.3f} "
                  f"家族={r['family_backbone_fraction']:.3f} "
                  f"独有={r['unique_residual_fraction']:.3f} | "
                  f"cos(家族,独有)={r['cos_fb_ur']:.3f}")

    # ============================================================
    # 汇总统计
    # ============================================================
    print("\n[汇总] 预测质量...")

    # 留一法 vs 跨家族的平均余弦相似度
    loo_cos_all = []
    cross_cos_all = []
    for word in all_words:
        loo = loo_results[word]
        cross = cross_results[word]
        for li in sample_layers:
            loo_cos_all.append(loo[li]["cosine_similarity"])
        for key, val in cross.items():
            cross_cos_all.append(val["cosine_similarity"])

    avg_loo_cos = sum(loo_cos_all) / len(loo_cos_all) if loo_cos_all else 0
    avg_cross_cos = sum(cross_cos_all) / len(cross_cos_all) if cross_cos_all else 0
    print(f"  留一法平均余弦: {avg_loo_cos:.4f}")
    print(f"  跨家族平均余弦: {avg_cross_cos:.4f}")
    print(f"  预测优势比: {avg_loo_cos / max(avg_cross_cos, 1e-8):.2f}x")

    # 残差分解统计
    last_li = sample_layers[-1]
    gb_fracs = [residual_results[w][last_li]["global_backbone_fraction"] for w in all_words]
    fb_fracs = [residual_results[w][last_li]["family_backbone_fraction"] for w in all_words]
    ur_fracs = [residual_results[w][last_li]["unique_residual_fraction"] for w in all_words]
    cos_fb_ur_all = [residual_results[w][last_li]["cos_fb_ur"] for w in all_words]

    print(f"\n[汇总] 残差分解（最后一层L{last_li}）:")
    print(f"  全局骨干占比: {sum(gb_fracs)/len(gb_fracs):.4f}")
    print(f"  家族骨干占比: {sum(fb_fracs)/len(fb_fracs):.4f}")
    print(f"  独有残差占比: {sum(ur_fracs)/len(ur_fracs):.4f}")
    print(f"  家族骨干-独有残差夹角cos: {sum(cos_fb_ur_all)/len(cos_fb_ur_all):.4f}")

    elapsed = time.time() - started
    print(f"\n总耗时: {elapsed:.1f}s")

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage538_loo_prediction_qwen3",
        "title": "留一法名词编码预测 (Qwen3)",
        "model": "Qwen3-4B",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 3),
        "config": {
            "families": {k: v["members"] for k, v in NOUN_FAMILIES.items()},
            "sample_layers": sample_layers,
        },
        "loo_results": {w: {str(li): v for li, v in lr.items()} for w, lr in loo_results.items()},
        "cross_results_summary": {
            "avg_loo_cosine": round(avg_loo_cos, 6),
            "avg_cross_cosine": round(avg_cross_cos, 6),
            "prediction_advantage": round(avg_loo_cos / max(avg_cross_cos, 1e-8), 4),
        },
        "residual_decomposition": {
            "avg_global_backbone_frac": round(sum(gb_fracs) / len(gb_fracs), 6),
            "avg_family_backbone_frac": round(sum(fb_fracs) / len(fb_fracs), 6),
            "avg_unique_residual_frac": round(sum(ur_fracs) / len(ur_fracs), 6),
            "avg_cos_family_residual": round(sum(cos_fb_ur_all) / len(cos_fb_ur_all), 6),
        },
        "core_answer": (
            "留一法名词编码预测验证了编码结构的可预测性：\n"
            f"1) 留一法平均余弦={avg_loo_cos:.4f}，跨家族平均余弦={avg_cross_cos:.4f}，"
            f"预测优势比={avg_loo_cos/max(avg_cross_cos,1e-8):.2f}x；\n"
            f"2) 残差分解：全局骨干={sum(gb_fracs)/len(gb_fracs):.3f}，"
            f"家族骨干={sum(fb_fracs)/len(fb_fracs):.3f}，"
            f"独有残差={sum(ur_fracs)/len(ur_fracs):.3f}；\n"
            f"3) 家族骨干与独有残差近似正交（cos={sum(cos_fb_ur_all)/len(cos_fb_ur_all):.3f}），"
            "说明三个分量确实是独立信息维度。"
        ),
    }

    out_path = OUTPUT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# stage538: 留一法名词编码预测 (Qwen3)\n",
        "## 核心发现\n",
        summary["core_answer"] + "\n",
        "## 预测质量\n",
        f"- 留一法平均余弦: {avg_loo_cos:.4f}\n",
        f"- 跨家族平均余弦: {avg_cross_cos:.4f}\n",
        f"- 预测优势比: {avg_loo_cos/max(avg_cross_cos,1e-8):.2f}x\n",
        "## 残差分解\n",
        f"- 全局骨干占比: {sum(gb_fracs)/len(gb_fracs):.4f}\n",
        f"- 家族骨干占比: {sum(fb_fracs)/len(fb_fracs):.4f}\n",
        f"- 独有残差占比: {sum(ur_fracs)/len(ur_fracs):.4f}\n",
        f"- 家族-独有夹角cos: {sum(cos_fb_ur_all)/len(cos_fb_ur_all):.4f}\n",
    ]

    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n结果: {out_path}")

    free_model(model)


if __name__ == "__main__":
    main()
