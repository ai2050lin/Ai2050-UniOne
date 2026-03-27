#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage324_first_principles_cross_model_reinforced_review import run_analysis as run_stage324
from stage341_shared_carrier_3d_layer import run_analysis as run_stage341
from stage342_bias_deflection_3d_layer import run_analysis as run_stage342
from stage343_layerwise_amplification_3d_layer import run_analysis as run_stage343
from stage344_multispace_operator_3d_layer import run_analysis as run_stage344


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage345_five_layer_3d_client_manifest_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s324 = run_stage324(force=False)
    s341 = run_stage341(force=False)
    s342 = run_stage342(force=False)
    s343 = run_stage343(force=False)
    s344 = run_stage344(force=False)

    layers = [
        {"layer_name": s341["layer_name"], "layer_score": float(s341["layer_score"]), "source": "stage341"},
        {"layer_name": s342["layer_name"], "layer_score": float(s342["layer_score"]), "source": "stage342"},
        {"layer_name": s343["layer_name"], "layer_score": float(s343["layer_score"]), "source": "stage343"},
        {"layer_name": s344["layer_name"], "layer_score": float(s344["layer_score"]), "source": "stage344"},
        {"layer_name": "cross_model_compare_layer", "layer_score": float(s324["cross_model_score"]), "source": "stage324"},
    ]

    manifest = {
        "schema_version": "agi_3d_client_scene.v1",
        "manifest_name": "five_layer_test_system",
        "axes": {
            "x": "空间类型",
            "y": "层级深度",
            "z": "结构角色",
        },
        "layers": layers,
        "client_panels": [
            "模型切换面板",
            "对象与任务过滤面板",
            "3D 主场景",
            "原始数据详情面板",
            "层级时间滑条",
        ],
    }

    return {
        "schema_version": "agi_3d_client_scene.v1",
        "experiment_id": "stage345_five_layer_3d_client_manifest",
        "title": "五层测试体系 3D 客户端场景清单",
        "status_short": "five_layer_3d_client_manifest_ready",
        "manifest_score": sum(layer["layer_score"] for layer in layers) / max(1, len(layers)),
        "layers": layers,
        "top_gap_name": "五层场景已经能分别加载，但跨模型对照层仍然更薄，说明客户端第一版更适合做结构观察，不适合过早做统一理论展示。",
        "manifest": manifest,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "agi_3d_client_scene_v1.json").write_text(
        json.dumps(summary["manifest"], ensure_ascii=False, indent=2), encoding="utf-8-sig"
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="五层测试体系 3D 客户端场景清单")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
