from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage66_first_principles_convergence_assessment_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage65_first_principles_identity_final_probe import build_first_principles_identity_final_probe_summary
from stage66_primitive_metric_decomposition import build_primitive_metric_decomposition_summary
from stage66_selector_uniqueness_proof_probe import build_selector_uniqueness_proof_probe_summary
from stage66_weight_principled_grounding import build_weight_principled_grounding_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_first_principles_convergence_assessment_summary() -> dict:
    final_probe = build_first_principles_identity_final_probe_summary()["headline_metrics"]
    weight = build_weight_principled_grounding_summary()["headline_metrics"]
    primitive = build_primitive_metric_decomposition_summary()["headline_metrics"]
    proof = build_selector_uniqueness_proof_probe_summary()["headline_metrics"]

    convergence_closure = _clip01(
        0.42 * final_probe["final_closure"]
        + 0.24 * primitive["native_metric_closure"]
        + 0.18 * proof["existence_support"]
        + 0.16 * (1.0 - weight["weight_subjectivity_penalty"])
    )
    convergence_falsifiability = _clip01(
        0.40 * final_probe["final_falsifiability"]
        + 0.24 * proof["stability_support"]
        + 0.20 * proof["uniqueness_support"]
        + 0.16 * (1.0 - primitive["primitive_reconstruction_error"])
    )
    convergence_dependency_penalty = _clip01(
        0.50 * final_probe["final_dependency_penalty"]
        + 0.24 * weight["weight_subjectivity_penalty"]
        + 0.16 * primitive["primitive_reconstruction_error"]
        + 0.10 * proof["proof_gap"]
    )
    convergence_identity_readiness = _clip01(
        0.30 * convergence_closure
        + 0.30 * convergence_falsifiability
        + 0.20 * (1.0 - convergence_dependency_penalty)
        + 0.20 * proof["proof_readiness"]
    )

    if convergence_closure >= 0.72 and convergence_falsifiability >= 0.76 and convergence_dependency_penalty < 0.34:
        status_short = "near_first_principles_theory"
        status_label = "理论身份已经逼近第一性原理理论边界，只差最后的严格证明和边界清零"
    else:
        status_short = "phenomenological_transition"
        status_label = "仍处于第一性原理过渡区后段，但比 stage65 更接近最终切换"

    return {
        "headline_metrics": {
            "convergence_closure": convergence_closure,
            "convergence_falsifiability": convergence_falsifiability,
            "convergence_dependency_penalty": convergence_dependency_penalty,
            "convergence_identity_readiness": convergence_identity_readiness,
        },
        "status": {
            "status_short": status_short,
            "status_label": status_label,
        },
        "project_readout": {
            "summary": "这一轮把权重原理化、原生变量重构和唯一性证明探针重新并回理论身份判断，测量项目是否开始从研究态评分系统转向更硬的第一性原理收束。",
            "next_question": "下一步要专门攻击剩余证明缺口和最后边界，避免理论长期停留在过渡区后段。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage66 First Principles Convergence Assessment",
        "",
        f"- convergence_closure: {hm['convergence_closure']:.6f}",
        f"- convergence_falsifiability: {hm['convergence_falsifiability']:.6f}",
        f"- convergence_dependency_penalty: {hm['convergence_dependency_penalty']:.6f}",
        f"- convergence_identity_readiness: {hm['convergence_identity_readiness']:.6f}",
        f"- status_short: {status['status_short']}",
        f"- status_label: {status['status_label']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_first_principles_convergence_assessment_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
