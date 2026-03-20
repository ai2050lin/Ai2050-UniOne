from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

from stage56_fullsample_regression_runner import read_json

ROOT = Path(__file__).resolve().parents[2]


def mean(values: Iterable[float]) -> float:
    seq = list(values)
    return sum(seq) / len(seq) if seq else 0.0


def _corr_mean(feature_summary: Mapping[str, object]) -> float:
    correlations = dict(feature_summary.get("correlations", {}))
    if not correlations:
        return 0.0
    return mean(float(v) for v in correlations.values())


def build_summary(
    native_summary: Dict[str, object],
    closed_v2_summary: Dict[str, object],
    higher_summary: Dict[str, object],
) -> Dict[str, object]:
    native = dict(native_summary.get("native_proxy_summary", {}))
    g_native = dict(native.get("G_native_proxy", {}))
    l_base_native = dict(native.get("L_base_native_proxy", {}))
    l_select_native = dict(native.get("L_select_native_proxy", {}))

    g_drive = _corr_mean(g_native)
    l_base_load = abs(_corr_mean(l_base_native))
    l_select_instability = abs(_corr_mean(l_select_native))
    strict_confidence = float(dict(closed_v2_summary.get("support", {})).get("strict_closure_confidence", 0.0))

    atlas_learning_drive = max(g_drive - l_select_instability, 0.0)
    frontier_learning_drive = max(l_base_load + 0.5 * g_drive, 0.0)
    closure_learning_drive = max(strict_confidence + l_select_instability - 0.5 * l_base_load, 0.0)

    learning_equations = {
        "atlas_update": "Atlas_{t+1} = Atlas_t + eta_A * (G_drive - L_select_instability)",
        "frontier_update": "Frontier_{t+1} = Frontier_t + eta_F * (L_base_load + 0.5 * G_drive)",
        "closure_boundary_update": "Boundary_{t+1} = Boundary_t + eta_B * (Strict_confidence + L_select_instability - 0.5 * L_base_load)",
    }
    emergence_order = [
        {"stage": 1, "name": "图册与基础负载先成形", "score": atlas_learning_drive + l_base_load},
        {"stage": 2, "name": "一般闭包核后稳定", "score": g_drive + strict_confidence},
        {"stage": 3, "name": "严格选择层最晚收口", "score": l_select_instability},
    ]
    return {
        "record_type": "stage56_learning_dynamics_bridge_summary",
        "learning_state": {
            "G_drive": g_drive,
            "L_base_load": l_base_load,
            "L_select_instability": l_select_instability,
            "Strict_confidence": strict_confidence,
            "atlas_learning_drive": atlas_learning_drive,
            "frontier_learning_drive": frontier_learning_drive,
            "closure_learning_drive": closure_learning_drive,
        },
        "learning_equations": learning_equations,
        "emergence_order": emergence_order,
        "bridge_objects": dict(higher_summary.get("system_objects", {})),
        "main_judgment": (
            "当前闭式结构已经可以桥接到学习动力学层：图册先在一般驱动和选择不稳定之间形成，"
            "前沿由基础负载与一般驱动共同塑形，闭包边界则在严格信心和选择压力共同作用下后期收口。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 学习动力学桥接摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Learning State",
        json.dumps(summary.get("learning_state", {}), ensure_ascii=False, indent=2),
        "",
        "## Learning Equations",
        json.dumps(summary.get("learning_equations", {}), ensure_ascii=False, indent=2),
        "",
        "## Emergence Order",
        json.dumps(summary.get("emergence_order", []), ensure_ascii=False, indent=2),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bridge the current closed-form system toward learning dynamics")
    ap.add_argument(
        "--native-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_real_corpus_native_proxy_refinement_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--closed-v2-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_icspb_closed_equation_v2_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--higher-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_higher_order_math_system_v3_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_bridge_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        read_json(Path(args.native_json)),
        read_json(Path(args.closed_v2_json)),
        read_json(Path(args.higher_json)),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
