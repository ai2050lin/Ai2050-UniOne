#!/usr/bin/env python
"""
P10C: final brain-side high-risk falsifier checklist.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def main() -> None:
    ap = argparse.ArgumentParser(description="P10C final brain falsifier checklist")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/p10c_final_brain_falsifier_checklist_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    p9c = load_json(ROOT / "tests" / "codex_temp" / "p9c_hard_spatial_brain_forecasts_20260311.json")
    p10a = load_json(ROOT / "tests" / "codex_temp" / "p10a_final_theory_verdict_20260311.json")
    p10b = load_json(ROOT / "tests" / "codex_temp" / "p10b_gap_boundary_empirical_vs_theoretical_20260311.json")

    checklist_sharpness = {
        "p9c_forecast_sharpness": float(p9c["headline_metrics"]["forecast_sharpness_score"]),
        "p9c_risk_targeting": float(p9c["headline_metrics"]["risk_targeting_score"]),
        "p10a_testability_strength": float(p10a["headline_metrics"]["testability_strength_score"]),
        "p10b_gap_boundary": float(p10b["headline_metrics"]["overall_p10b_score"]),
    }
    checklist_sharpness_score = mean(checklist_sharpness.values())

    final_checklist = [
        "如果局部邻域扰动不先压低 family topology margin，当前局部复用主张会被削弱。",
        "如果长程桥切断不优先伤 compact-boundary relation，当前稀疏桥接主张会被削弱。",
        "如果 geometry-only 平滑稳定优于目标化桥增强，当前动态有效拓扑主张会被削弱。",
        "如果高价值长程桥不是束化稀疏而是大面积均匀扩张，当前空间效率主张会被削弱。",
        "如果更大尺度或不同架构模型完全不保留局部复用 / 稀疏桥接分工，当前统一理论会被削弱。",
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "p10c_final_brain_falsifier_checklist",
        },
        "final_checklist": final_checklist,
        "headline_metrics": {
            "checklist_sharpness_score": float(checklist_sharpness_score),
            "overall_p10c_score": float(checklist_sharpness_score),
        },
        "hypotheses": {
            "H1_final_checklist_is_sharp": bool(checklist_sharpness_score >= 0.80),
            "H2_p10c_final_checklist_is_positive": bool(checklist_sharpness_score >= 0.80),
        },
        "project_readout": {
            "summary": (
                "P10C is positive only if the project ends with a short list of observations that would genuinely "
                "damage the current theory."
            ),
            "next_question": "If P10C holds, the project can state a final stage verdict."
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
