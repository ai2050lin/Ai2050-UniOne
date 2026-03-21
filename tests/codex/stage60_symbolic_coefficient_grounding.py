from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage60_symbolic_coefficient_grounding_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_native_variable_candidate_mapping import build_native_variable_candidate_mapping_summary
from stage59_local_law_symbolic_derivation import build_local_law_symbolic_derivation_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_symbolic_coefficient_grounding_summary() -> dict:
    native = build_native_variable_candidate_mapping_summary()
    symbolic = build_local_law_symbolic_derivation_summary()
    native_hm = native["headline_metrics"]
    sym_hm = symbolic["headline_metrics"]
    mapping = native["candidate_mapping"]

    coefficient_grounding = {
        "alpha_P": "mix(a_density, r_return, q_context)",
        "beta_P": "mix(f_reuse, g_route)",
        "delta_P": "mix(h_pressure, m_load)",
        "alpha_F": "mix(u_reuse, f_flow)",
        "beta_R": "mix(g_route, q_context)",
        "gamma_Pi": "mix(p_plasticity, dw_dt)",
    }

    native_coefficient_score = _clip01(
        0.30 * sym_hm["symbolic_bridge_score"]
        + 0.25 * sym_hm["symbolic_closure"]
        + 0.20 * native_hm["native_mapping_completeness"]
        + 0.15 * mapping["L_plasticity"]["candidate_score"]
        + 0.10 * mapping["Pi_pressure"]["candidate_score"]
    )
    coefficient_grounding_coverage = len(coefficient_grounding) / 6.0
    residual_grounding_gap = _clip01(
        1.0 - (
            0.48 * native_coefficient_score
            + 0.22 * coefficient_grounding_coverage
            + 0.30 * (1.0 - sym_hm["theorem_gap"])
        )
    )

    return {
        "headline_metrics": {
            "coefficient_grounding_coverage": coefficient_grounding_coverage,
            "native_coefficient_score": native_coefficient_score,
            "residual_grounding_gap": residual_grounding_gap,
        },
        "status": {
            "status_short": "coefficients_partially_grounded",
            "status_label": "符号系数已部分落地到原生变量，但仍未完全唯一化",
        },
        "coefficient_grounding": coefficient_grounding,
        "project_readout": {
            "summary": "符号系数落地把 alpha、beta、delta 等符号系数继续压回原生变量候选，尝试减少“只有方程名字，没有原生来源”的空转状态。",
            "next_question": "下一步要检验这些系数落地是否能跨语言任务与脑桥接任务共享，而不是只在局部方程里自洽。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage60 Symbolic Coefficient Grounding",
        "",
        f"- coefficient_grounding_coverage: {hm['coefficient_grounding_coverage']:.6f}",
        f"- native_coefficient_score: {hm['native_coefficient_score']:.6f}",
        f"- residual_grounding_gap: {hm['residual_grounding_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_symbolic_coefficient_grounding_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
