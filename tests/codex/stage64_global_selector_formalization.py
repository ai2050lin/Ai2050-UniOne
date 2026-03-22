from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage64_global_selector_formalization_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage60_symbolic_coefficient_grounding import build_symbolic_coefficient_grounding_summary
from stage62_first_principles_boundary_probe import build_first_principles_boundary_probe_summary
from stage62_uniqueness_hardening import build_uniqueness_hardening_summary
from stage63_global_uniqueness_constraint import build_global_uniqueness_constraint_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_global_selector_formalization_summary() -> dict:
    global_unique = build_global_uniqueness_constraint_summary()["headline_metrics"]
    coeff = build_symbolic_coefficient_grounding_summary()["headline_metrics"]
    uniq = build_uniqueness_hardening_summary()["headline_metrics"]
    boundary = build_first_principles_boundary_probe_summary()["headline_metrics"]

    selector_energy_coherence = _clip01(
        0.34 * global_unique["style_logic_grammar_alignment"]
        + 0.26 * global_unique["token_uniqueness_support"]
        + 0.22 * uniq["cross_task_lock_score"]
        + 0.18 * (1.0 - uniq["residual_uniqueness_gap"])
    )
    selector_formalization_score = _clip01(
        0.38 * global_unique["global_uniqueness_score"]
        + 0.24 * global_unique["mathematical_uniqueness_score"]
        + 0.18 * coeff["native_coefficient_score"]
        + 0.20 * boundary["boundary_closure"]
    )
    selector_closure = _clip01(
        0.40 * selector_formalization_score
        + 0.24 * selector_energy_coherence
        + 0.18 * uniq["hardened_uniqueness_score"]
        + 0.18 * (1.0 - coeff["residual_grounding_gap"])
    )
    residual_selector_gap = _clip01(1.0 - selector_closure)

    selector_system = {
        "principle": "w* = argmin_w [lambda_s E_style(w) + lambda_l E_logic(w) + lambda_y E_syntax(w) + lambda_c E_context(w) + lambda_m E_world(w) + lambda_g R_global(w)]",
        "constraint": "sum_i contribution_i(w*) -> unique feasible minimum under distributed participation",
        "coefficients": "lambda = Phi(a_density, r_return, q_context, f_reuse, g_route, h_pressure, p_plasticity)",
    }

    return {
        "headline_metrics": {
            "selector_energy_coherence": selector_energy_coherence,
            "selector_formalization_score": selector_formalization_score,
            "selector_closure": selector_closure,
            "residual_selector_gap": residual_selector_gap,
        },
        "status": {
            "status_short": "selector_formalized_not_closed",
            "status_label": "全局唯一选择器已经形式化，但还没有闭合成最终主方程",
        },
        "selector_system": selector_system,
        "project_readout": {
            "summary": "全局选择器形式化把语言中的全局唯一词选择压成一个显式的能量最小化主方程候选，尝试把解释层对象进一步推向主核层。",
            "next_question": "下一步要验证这个选择器主方程是否真的能削减 completion blocker，而不是只增加解释性。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage64 Global Selector Formalization",
        "",
        f"- selector_energy_coherence: {hm['selector_energy_coherence']:.6f}",
        f"- selector_formalization_score: {hm['selector_formalization_score']:.6f}",
        f"- selector_closure: {hm['selector_closure']:.6f}",
        f"- residual_selector_gap: {hm['residual_selector_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_global_selector_formalization_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
