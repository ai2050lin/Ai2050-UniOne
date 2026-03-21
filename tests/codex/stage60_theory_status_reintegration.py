from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage60_theory_status_reintegration_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_theory_status_assessment import build_theory_status_assessment_summary
from stage59_counterexample_replay import build_counterexample_replay_summary
from stage60_dependency_below_floor_probe import build_dependency_below_floor_probe_summary
from stage60_principled_coupled_scale_repair import build_principled_coupled_scale_repair_summary
from stage60_symbolic_coefficient_grounding import build_symbolic_coefficient_grounding_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_theory_status_reintegration_summary() -> dict:
    theory = build_theory_status_assessment_summary()
    replay = build_counterexample_replay_summary()["headline_metrics"]
    floor = build_dependency_below_floor_probe_summary()["headline_metrics"]
    principled = build_principled_coupled_scale_repair_summary()["headline_metrics"]
    coeff = build_symbolic_coefficient_grounding_summary()["headline_metrics"]

    base = theory["headline_metrics"]
    updated_closure = _clip01(
        0.42 * base["first_principles_closure"]
        + 0.22 * (1.0 - principled["best_principled_dependency_penalty"])
        + 0.18 * coeff["native_coefficient_score"]
        + 0.18 * floor["new_floor_coupled_margin"]
    )
    updated_falsifiability = _clip01(
        0.36 * base["falsifiability_strength"]
        + 0.24 * replay["replay_reproducibility"]
        + 0.20 * (1.0 - replay["residual_risk"])
        + 0.20 * floor["new_floor_coupled_margin"]
    )
    updated_dependency_penalty = _clip01(
        0.55 * principled["best_principled_dependency_penalty"]
        + 0.45 * floor["new_dependency_floor_penalty"]
    )
    transition_support = _clip01(
        0.30 * updated_closure
        + 0.24 * updated_falsifiability
        + 0.24 * coeff["native_coefficient_score"]
        + 0.22 * (1.0 - updated_dependency_penalty)
    )

    if updated_closure >= 0.75 and updated_falsifiability >= 0.78 and updated_dependency_penalty < 0.45:
        status_short = "first_principles_theory"
        status_label = "基于第一性原理的理论"
    elif updated_closure >= 0.58 and updated_falsifiability >= 0.72 and updated_dependency_penalty < 0.70:
        status_short = "phenomenological_transition"
        status_label = "仍属唯象模型，但已进入第一性原理过渡区"
    else:
        status_short = "phenomenological_model"
        status_label = "唯象模型"

    return {
        "headline_metrics": {
            "updated_closure": updated_closure,
            "updated_falsifiability": updated_falsifiability,
            "updated_dependency_penalty": updated_dependency_penalty,
            "transition_support": transition_support,
        },
        "status": {
            "status_short": status_short,
            "status_label": status_label,
        },
        "project_readout": {
            "summary": "理论状态重整合把 replay、dependency floor、principled repair、coefficient grounding 全部并回统一判断，避免理论结论继续滞后于最新修复与反例链。",
            "next_question": "下一步要继续缩小 dependency penalty 和 theorem gap，看看过渡区能否真正推进到闭合理论。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage60 Theory Status Reintegration",
        "",
        f"- updated_closure: {hm['updated_closure']:.6f}",
        f"- updated_falsifiability: {hm['updated_falsifiability']:.6f}",
        f"- updated_dependency_penalty: {hm['updated_dependency_penalty']:.6f}",
        f"- transition_support: {hm['transition_support']:.6f}",
        f"- status_short: {status['status_short']}",
        f"- status_label: {status['status_label']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_theory_status_reintegration_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
