from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage66_weight_principled_grounding_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_native_variable_candidate_mapping import build_native_variable_candidate_mapping_summary
from stage60_symbolic_coefficient_grounding import build_symbolic_coefficient_grounding_summary
from stage61_native_variable_regression import build_native_variable_regression_summary
from stage63_global_uniqueness_constraint import build_global_uniqueness_constraint_summary
from stage65_selector_master_equation_closure import build_selector_master_equation_closure_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_weight_principled_grounding_summary() -> dict:
    native = build_native_variable_candidate_mapping_summary()["headline_metrics"]
    coeff = build_symbolic_coefficient_grounding_summary()["headline_metrics"]
    regression = build_native_variable_regression_summary()["headline_metrics"]
    uniqueness = build_global_uniqueness_constraint_summary()["headline_metrics"]
    master = build_selector_master_equation_closure_summary()["headline_metrics"]

    structural_weight_grounding = _clip01(
        0.30 * native["native_mapping_completeness"]
        + 0.28 * coeff["native_coefficient_score"]
        + 0.24 * regression["mapping_fidelity"]
        + 0.18 * (1.0 - coeff["residual_grounding_gap"])
    )
    selector_weight_consistency = _clip01(
        0.34 * uniqueness["unique_selector_constraint"]
        + 0.28 * uniqueness["mathematical_uniqueness_score"]
        + 0.20 * master["equation_constraint_lock"]
        + 0.18 * master["master_equation_coherence"]
    )
    principled_weight_score = _clip01(
        0.42 * structural_weight_grounding
        + 0.34 * selector_weight_consistency
        + 0.24 * regression["fp_integrity"]
    )
    weight_subjectivity_penalty = _clip01(
        1.0
        - (
            0.46 * principled_weight_score
            + 0.28 * structural_weight_grounding
            + 0.26 * selector_weight_consistency
        )
    )

    return {
        "headline_metrics": {
            "structural_weight_grounding": structural_weight_grounding,
            "selector_weight_consistency": selector_weight_consistency,
            "principled_weight_score": principled_weight_score,
            "weight_subjectivity_penalty": weight_subjectivity_penalty,
        },
        "status": {
            "status_short": "weight_grounding_strengthened",
            "status_label": "主方程权重开始从经验设计转向编码结构落地，但仍未完全摆脱主观权重残留",
        },
        "project_readout": {
            "summary": "这一轮把主方程里的线性权重继续压回原生变量、符号系数和全局唯一性约束，目标是减少研究者手工配权造成的主观性。",
            "next_question": "下一步要继续检查这些权重落地后，最终身份指标能否更多地由原生变量重构，而不是由高层评分互相支撑。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage66 Weight Principled Grounding",
        "",
        f"- structural_weight_grounding: {hm['structural_weight_grounding']:.6f}",
        f"- selector_weight_consistency: {hm['selector_weight_consistency']:.6f}",
        f"- principled_weight_score: {hm['principled_weight_score']:.6f}",
        f"- weight_subjectivity_penalty: {hm['weight_subjectivity_penalty']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_weight_principled_grounding_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
