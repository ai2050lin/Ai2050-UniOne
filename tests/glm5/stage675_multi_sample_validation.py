#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage675: P29 多样本统计验证——方向纠缠度矩阵的可靠性

目标：每种能力使用10+个样本对，重新计算方向纠缠度矩阵，
     确认P26中的关键发现（DS7B能力融合、Gemma4多模态纠缠）是否稳定。

关键验证：
  1. DS7B的spatial-style cos=0.718是否稳定？（10样本均值 vs 1样本）
  2. Gemma4的方向纠缠度是否稳定高于纯语言模型？
  3. 不同prompt对之间cos的方差有多大？
"""

from __future__ import annotations

import sys
import io
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    discover_layers,
    free_model,
    load_model_bundle,
    MODEL_SPECS,
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


# ============================================================
# 多样本测试用例——每种能力10对
# ============================================================
@dataclass(frozen=True)
class SamplePair:
    capability: str
    prompt_a: str
    prompt_b: str

MULTI_SAMPLES = {
    "disamb": [
        SamplePair("disamb", "The river bank was muddy.", "The bank approved the loan."),
        SamplePair("disamb", "The bat flew across the dark cave.", "He swung the bat and hit a home run."),
        SamplePair("disamb", "The plant needs water and sunlight.", "The plant closed down and moved overseas."),
        SamplePair("disamb", "The mouse ran across the kitchen floor.", "She clicked the mouse to select the file."),
        SamplePair("disamb", "The match was very exciting to watch.", "He struck the match to light the candle."),
        SamplePair("disamb", "The bow was decorated with a ribbon.", "She took a bow after her performance."),
        SamplePair("disamb", "The ring was made of pure gold.", "Please ring the bell when you arrive."),
        SamplePair("disamb", "The watch was ticking on the wall.", "Please watch your step on the stairs."),
        SamplePair("disamb", "The saw was sharp enough to cut wood.", "I saw her walking in the park yesterday."),
        SamplePair("disamb", "The fair was held in the town square.", "The judge must be fair to both sides."),
    ],
    "syntax": [
        SamplePair("syntax", "She quickly ran home.", "Home she ran quickly."),
        SamplePair("syntax", "The boy ate the apple happily.", "Happily the boy ate the apple."),
        SamplePair("syntax", "She read the book with great interest.", "With great interest she read the book."),
        SamplePair("syntax", "The dog chased the cat across the yard.", "Across the yard the dog chased the cat."),
        SamplePair("syntax", "He carefully placed the vase on the table.", "On the table he carefully placed the vase."),
        SamplePair("syntax", "They sang the song together loudly.", "Together they sang the song loudly."),
        SamplePair("syntax", "She always arrives early for meetings.", "Early for meetings she always arrives."),
        SamplePair("syntax", "The children played in the garden happily.", "Happily the children played in the garden."),
        SamplePair("syntax", "He slowly opened the old wooden door.", "The old wooden door he slowly opened."),
        SamplePair("syntax", "The rain fell gently on the roof.", "Gently the rain fell on the roof."),
    ],
    "relation": [
        SamplePair("relation", "Paris is the capital of France.", "Berlin is the capital of Germany."),
        SamplePair("relation", "Tokyo is the capital of Japan.", "London is the capital of the United Kingdom."),
        SamplePair("relation", "Beijing is the capital of China.", "Moscow is the capital of Russia."),
        SamplePair("relation", "Rome is the capital of Italy.", "Madrid is the capital of Spain."),
        SamplePair("relation", "Ottawa is the capital of Canada.", "Canberra is the capital of Australia."),
        SamplePair("relation", "Athens is the capital of Greece.", "Lisbon is the capital of Portugal."),
        SamplePair("relation", "Seoul is the capital of South Korea.", "Bangkok is the capital of Thailand."),
        SamplePair("relation", "Cairo is the capital of Egypt.", "Nairobi is the capital of Kenya."),
        SamplePair("relation", "Brasilia is the capital of Brazil.", "Buenos Aires is the capital of Argentina."),
        SamplePair("relation", "New Delhi is the capital of India.", "Jakarta is the capital of Indonesia."),
    ],
    "style": [
        SamplePair("style", "The meeting was extremely productive.", "That get-together was quite fruitful."),
        SamplePair("style", "The scientist conducted rigorous experiments.", "The researcher did careful studies."),
        SamplePair("style", "The government implemented new policies.", "The authorities put in fresh rules."),
        SamplePair("style", "The executive made a strategic decision.", "The boss chose a smart move."),
        SamplePair("style", "The institution faces significant challenges.", "The organization has big problems."),
        SamplePair("style", "The demonstration was largely peaceful.", "The protest was mostly calm."),
        SamplePair("style", "The legislation requires immediate attention.", "The law needs quick action."),
        SamplePair("style", "The negotiations were exceptionally complex.", "The talks were really hard."),
        SamplePair("style", "The investigation revealed crucial evidence.", "The probe found key proof."),
        SamplePair("style", "The proposal received overwhelming support.", "The plan got huge backing."),
    ],
    "spatial": [
        SamplePair("spatial", "The cat is under the table.", "The bird is above the tree."),
        SamplePair("spatial", "The book is on the shelf.", "The keys are inside the drawer."),
        SamplePair("spatial", "The car is parked behind the building.", "The bicycle is next to the fence."),
        SamplePair("spatial", "The picture hangs above the fireplace.", "The lamp stands beside the sofa."),
        SamplePair("spatial", "The children played behind the house.", "The dog slept under the porch."),
        SamplePair("spatial", "The store is across the street.", "The park is around the corner."),
        SamplePair("spatial", "The bridge spans over the river.", "The tunnel goes through the mountain."),
        SamplePair("spatial", "The flower pot sits on the windowsill.", "The rug lies beneath the table."),
        SamplePair("spatial", "The clouds floated above the mountains.", "The fish swam beneath the surface."),
        SamplePair("spatial", "The path leads between the two trees.", "The gate opens onto the garden."),
    ],
    "temporal": [
        SamplePair("temporal", "Yesterday it rained heavily.", "Tomorrow it will snow."),
        SamplePair("temporal", "Last week she finished her project.", "Next month they will launch the product."),
        SamplePair("temporal", "Last year the company doubled revenue.", "Next year they plan to expand globally."),
        SamplePair("temporal", "Earlier today the temperature dropped.", "Later tonight the winds will pick up."),
        SamplePair("temporal", "In the morning she exercises.", "In the evening she reads books."),
        SamplePair("temporal", "During the summer we go swimming.", "During the winter we stay indoors."),
        SamplePair("temporal", "Before the meeting he reviewed the notes.", "After the meeting he wrote the summary."),
        SamplePair("temporal", "Centuries ago people believed the earth was flat.", "Decades from now technology will transform society."),
        SamplePair("temporal", "At dawn the birds begin to sing.", "At dusk the stars start to appear."),
        SamplePair("temporal", "He arrived shortly after noon.", "She departed well before midnight."),
    ],
}

CAPABILITIES = list(MULTI_SAMPLES.keys())


def get_last_hidden(model, tokenizer, text: str) -> torch.Tensor:
    """获取最后一层hidden state"""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        return outputs.hidden_states[-1][0, -1, :].float().cpu()


def compute_pair_direction(model, tokenizer, pair: SamplePair) -> torch.Tensor:
    """计算一对prompt的差异方向"""
    ha = get_last_hidden(model, tokenizer, pair.prompt_a)
    hb = get_last_hidden(model, tokenizer, pair.prompt_b)
    return ha - hb


def run_multi_sample_analysis(model, tokenizer, model_name: str) -> Dict:
    """多样本分析"""
    print(f"\n{'='*70}")
    print(f"  P29 多样本统计验证: {model_name}")
    print(f"  样本数: {sum(len(v) for v in MULTI_SAMPLES.values())} "
          f"({len(CAPABILITIES)}能力 x {len(MULTI_SAMPLES[CAPABILITIES[0]])}样本)")
    print(f"{'='*70}")

    # 1. 计算所有样本对的方向
    print("\n[1] 计算所有样本对的编码方向...")
    directions = {}  # cap -> [directions]
    for cap in CAPABILITIES:
        pairs = MULTI_SAMPLES[cap]
        dirs = []
        for pair in pairs:
            d = compute_pair_direction(model, tokenizer, pair)
            dirs.append(d)
        directions[cap] = dirs
        print(f"    {cap:>10}: {len(dirs)} 个方向, ||d|| = {statistics.mean([d.norm().item() for d in dirs]):.2f}")

    # 2. 计算能力间cos矩阵（多样本均值）
    print("\n[2] 计算能力间cos矩阵（多样本均值）...")
    cos_matrix = {}
    cos_stds = {}
    cos_matrix_raw = {}

    for cap_a in CAPABILITIES:
        for cap_b in CAPABILITIES:
            if cap_a == cap_b:
                cos_matrix[(cap_a, cap_b)] = 1.0
                cos_stds[(cap_a, cap_b)] = 0.0
                cos_matrix_raw[(cap_a, cap_b)] = [1.0]
                continue

            cross_cos = []
            for da in directions[cap_a]:
                for db in directions[cap_b]:
                    c = F.cosine_similarity(da.unsqueeze(0), db.unsqueeze(0)).item()
                    cross_cos.append(c)

            mean_cos = statistics.mean(cross_cos)
            std_cos = statistics.stdev(cross_cos) if len(cross_cos) > 1 else 0

            cos_matrix[(cap_a, cap_b)] = mean_cos
            cos_stds[(cap_a, cap_b)] = std_cos
            cos_matrix_raw[(cap_a, cap_b)] = cross_cos

    # 3. 打印cos矩阵
    print(f"\n  {'':>10}", end="")
    for cap in CAPABILITIES:
        print(f" {cap[:6]:>8}", end="")
    print()
    print(f"  {'-'*70}")

    for cap_a in CAPABILITIES:
        print(f"  {cap_a[:6]:>10}", end="")
        for cap_b in CAPABILITIES:
            v = cos_matrix[(cap_a, cap_b)]
            s = cos_stds[(cap_a, cap_b)]
            if cap_a == cap_b:
                print(f" {'--':>8}", end="")
            else:
                print(f" {v:>7.3f}", end="")
        print()

    # 4. 打印标准差矩阵
    print(f"\n  标准差矩阵:")
    print(f"  {'':>10}", end="")
    for cap in CAPABILITIES:
        print(f" {cap[:6]:>8}", end="")
    print()
    print(f"  {'-'*70}")

    for cap_a in CAPABILITIES:
        print(f"  {cap_a[:6]:>10}", end="")
        for cap_b in CAPABILITIES:
            s = cos_stds[(cap_a, cap_b)]
            if cap_a == cap_b:
                print(f" {'--':>8}", end="")
            else:
                print(f" {s:>7.3f}", end="")
        print()

    # 5. 关键指标
    print(f"\n[3] 关键指标:")
    all_cross_cos = []
    for (a, b), vals in cos_matrix_raw.items():
        if a != b:
            all_cross_cos.extend([abs(v) for v in vals])

    avg_abs_entanglement = statistics.mean(all_cross_cos)
    avg_entanglement = statistics.mean([v for (a, b), vals in cos_matrix_raw.items() if a != b for v in vals])
    std_entanglement = statistics.stdev(all_cross_cos) if len(all_cross_cos) > 1 else 0

    print(f"    平均|cos|纠缠度: {avg_abs_entanglement:.4f} (P26单样本: --)")
    print(f"    纠缠度标准差: {std_entanglement:.4f}")
    print(f"    样本总数: {len(all_cross_cos)}")

    # 6. DS7B关键验证：spatial-style高cos？
    spatial_style_cos = cos_matrix.get(("spatial", "style"), 0)
    spatial_style_std = cos_stds.get(("spatial", "style"), 0)
    spatial_style_raw = cos_matrix_raw.get(("spatial", "style"), [])
    print(f"\n[4] DS7B关键验证 (spatial↔style):")
    print(f"    多样本均值cos: {spatial_style_cos:.4f} ± {spatial_style_std:.4f}")
    print(f"    最小值: {min(abs(v) for v in spatial_style_raw):.4f}")
    print(f"    最大值: {max(abs(v) for v in spatial_style_raw):.4f}")
    print(f"    中位数: {statistics.median([abs(v) for v in spatial_style_raw]):.4f}")

    # 7. 找出最高/最低纠缠对
    print(f"\n[5] 能力纠缠排名 (按|均值cos|):")
    pairs_sorted = sorted(
        [(a, b, cos_matrix[(a,b)], cos_stds[(a,b)]) for a in CAPABILITIES for b in CAPABILITIES if a != b],
        key=lambda x: abs(x[2]), reverse=True
    )
    for rank, (a, b, mean, std) in enumerate(pairs_sorted[:10]):
        print(f"    #{rank+1:2d} {a:>8} ↔ {b:<8}: {mean:>7.4f} ± {std:.4f}")

    # 8. 能力内一致性
    print(f"\n[6] 能力内方向一致性（同一能力不同样本间的cos）:")
    for cap in CAPABILITIES:
        dirs = directions[cap]
        if len(dirs) < 2:
            print(f"    {cap:>10}: 样本不足")
            continue
        inner_cos = []
        for i in range(len(dirs)):
            for j in range(i+1, len(dirs)):
                c = F.cosine_similarity(dirs[i].unsqueeze(0), dirs[j].unsqueeze(0)).item()
                inner_cos.append(c)
        mean_inner = statistics.mean(inner_cos)
        std_inner = statistics.stdev(inner_cos) if len(inner_cos) > 1 else 0
        print(f"    {cap:>10}: 均值cos={mean_inner:>7.4f} ± {std_inner:.4f} ({len(inner_cos)}对)")

    return {
        "model_name": model_name,
        "avg_abs_entanglement": avg_abs_entanglement,
        "std_entanglement": std_entanglement,
        "spatial_style_cos": spatial_style_cos,
        "spatial_style_std": spatial_style_std,
        "cos_matrix": {f"{a},{b}": v for (a,b), v in cos_matrix.items()},
        "cos_stds": {f"{a},{b}": v for (a,b), v in cos_stds.items()},
        "top_pairs": [(a, b, m) for a, b, m, s in pairs_sorted[:10]],
        "inner_consistency": {},
    }


def main():
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"模型参数: {model_arg}")

    model, tokenizer = load_model_bundle(model_arg)
    if model is None:
        print(f"无法加载模型: {model_arg}")
        return

    model_name = MODEL_SPECS.get(model_arg, {}).get("label", model_arg)
    try:
        result = run_multi_sample_analysis(model, tokenizer, model_name)
    finally:
        free_model(model)

    # 保存结果
    output_file = OUTPUT_DIR / f"stage675_multi_sample_{model_arg}_{TIMESTAMP}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()
