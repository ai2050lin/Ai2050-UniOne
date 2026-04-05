#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from multimodel_language_shared import (
    ablate_layer_component,
    candidate_score_map,
    discover_layers,
    free_model,
    get_model_device,
    load_model_bundle,
    restore_layer_component,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage507_gemma4_hooking_smoke_20260404"
)

PROMPT = "只输出 A/B/C/D 中一个字母，不要解释。\n短语前缀是“新鲜的苹”，最可能补成：\nA. 果\nB. 菜\nC. 子\nD. 园"
CANDIDATES = ["A", "B", "C", "D"]
TARGET = "A"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    model, tokenizer = load_model_bundle("gemma4", prefer_cuda=True)
    try:
        layers = discover_layers(model)
        layer0 = layers[0]
        baseline_scores = candidate_score_map(model, tokenizer, PROMPT, CANDIDATES)
        baseline_target = float(baseline_scores[TARGET])

        attn_layer, attn_original = ablate_layer_component(model, 0, "attn")
        try:
            attn_scores = candidate_score_map(model, tokenizer, PROMPT, CANDIDATES)
        finally:
            restore_layer_component(attn_layer, "attn", attn_original)

        mlp_layer, mlp_original = ablate_layer_component(model, 0, "mlp")
        try:
            mlp_scores = candidate_score_map(model, tokenizer, PROMPT, CANDIDATES)
        finally:
            restore_layer_component(mlp_layer, "mlp", mlp_original)

        summary = {
            "stage": "stage507_gemma4_hooking_smoke",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(get_model_device(model)),
            "layer_count": len(layers),
            "layer0_type": type(layer0).__name__,
            "has_self_attn": hasattr(layer0, "self_attn"),
            "has_mlp": hasattr(layer0, "mlp"),
            "baseline_scores": {k: round(v, 6) for k, v in baseline_scores.items()},
            "attn_scores": {k: round(v, 6) for k, v in attn_scores.items()},
            "mlp_scores": {k: round(v, 6) for k, v in mlp_scores.items()},
            "attn_target_drop": round(baseline_target - float(attn_scores[TARGET]), 6),
            "mlp_target_drop": round(baseline_target - float(mlp_scores[TARGET]), 6),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUTPUT_DIR / "REPORT.md").write_text(
            "\n".join(
                [
                    "# stage507 Gemma4 挂钩冒烟测试",
                    "",
                    f"- 设备：`{summary['device']}`",
                    f"- 层数：`{summary['layer_count']}`",
                    f"- 第 0 层类型：`{summary['layer0_type']}`",
                    f"- 有 self_attn：`{summary['has_self_attn']}`",
                    f"- 有 mlp：`{summary['has_mlp']}`",
                    f"- attn 目标下降：`{summary['attn_target_drop']}`",
                    f"- mlp 目标下降：`{summary['mlp_target_drop']}`",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        free_model(model)


if __name__ == "__main__":
    main()
