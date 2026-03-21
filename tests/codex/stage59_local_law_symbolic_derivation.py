from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage59_local_law_symbolic_derivation_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_local_generative_law_emergence import build_local_generative_law_emergence_summary
from stage58_local_law_necessity_scan import build_local_law_necessity_scan_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_local_law_symbolic_derivation_summary() -> dict:
    local = build_local_generative_law_emergence_summary()
    necessity = build_local_law_necessity_scan_summary()
    hm = necessity["headline_metrics"]
    local_hm = local["headline_metrics"]

    symbolic_component_coverage = len(necessity["status"]["necessary_components"]) / 4.0
    assumption_penalty = _clip01(0.22 + 0.28 * hm["proof_gap"])
    symbolic_bridge_score = _clip01(
        0.42 * hm["necessity_strength"]
        + 0.28 * local_hm["derivability_score"]
        + 0.18 * symbolic_component_coverage
        + 0.12 * (1.0 - assumption_penalty)
    )
    symbolic_closure = _clip01(
        0.52 * symbolic_bridge_score
        + 0.24 * local_hm["local_law_emergence_score"]
        + 0.24 * (1.0 - hm["proof_gap"])
    )
    theorem_gap = _clip01(1.0 - symbolic_closure)

    symbolic_system = {
        "patch": "P_{t+1} = alpha_P * N(P_t) + beta_P * F_t + gamma_P * C_t - delta_P * Pi_t",
        "fiber": "F_{t+1} = alpha_F * X(P_t, R_t) + beta_F * G_t - delta_F * Pi_t",
        "route": "R_{t+1} = alpha_R * G(P_t, C_t) + beta_R * F_t - delta_R * cost_t",
        "pressure": "Pi_{t+1} = alpha_Pi * Pi_t + beta_Pi * load_t - gamma_Pi * plasticity_t",
    }
    derivation_steps = [
        "由 necessity scan 固定四个必要结构项：neighbor_patch、fiber_exchange、context_gate、pressure_regulation。",
        "把局部生成律中的数值更新式压成 patch、fiber、route、pressure 四条符号演化方程。",
        "用 necessity_strength 和 derivability_score 约束符号系数的可行域，而不是直接假定系数任意。",
        "当前仍缺少从原生变量到系数符号的严格唯一化，因此只能称为 symbolic bridge，不是闭合证明。",
    ]

    return {
        "headline_metrics": {
            "symbolic_component_coverage": symbolic_component_coverage,
            "assumption_penalty": assumption_penalty,
            "symbolic_bridge_score": symbolic_bridge_score,
            "symbolic_closure": symbolic_closure,
            "theorem_gap": theorem_gap,
        },
        "status": {
            "status_short": "symbolic_bridge_not_closed",
            "status_label": "符号桥已经成形，但还没有闭合成严格定理",
        },
        "symbolic_system": symbolic_system,
        "derivation_steps": derivation_steps,
        "project_readout": {
            "summary": "局部律符号化推进把必要性扫描和局部生成律合并成一组符号演化方程，目标是把“能跑通的数值律”推进到“可讨论证明结构的符号桥”。",
            "next_question": "下一步要把符号系数继续压回原生变量，而不是只在 patch、fiber、route、pressure 四层停住。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage59 Local Law Symbolic Derivation",
        "",
        f"- symbolic_component_coverage: {hm['symbolic_component_coverage']:.6f}",
        f"- assumption_penalty: {hm['assumption_penalty']:.6f}",
        f"- symbolic_bridge_score: {hm['symbolic_bridge_score']:.6f}",
        f"- symbolic_closure: {hm['symbolic_closure']:.6f}",
        f"- theorem_gap: {hm['theorem_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_local_law_symbolic_derivation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
