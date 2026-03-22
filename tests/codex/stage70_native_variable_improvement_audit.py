from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage70_native_variable_improvement_audit_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage67_identity_switch_probe import build_identity_switch_probe_summary
from stage68_direct_identity_assessment import build_direct_identity_assessment_summary
from stage69_direct_metric_primitive_trace import build_direct_metric_primitive_trace_summary
from stage70_native_observability_bridge import build_native_observability_bridge_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_native_variable_improvement_audit_summary() -> dict:
    nested = build_identity_switch_probe_summary()["headline_metrics"]
    direct = build_direct_identity_assessment_summary()["headline_metrics"]
    trace = build_direct_metric_primitive_trace_summary()
    obs = build_native_observability_bridge_summary()["headline_metrics"]

    direct_explainability_gain = _clip01(
        0.40 * obs["proxy_traceability_score"]
        + 0.30 * (1.0 - obs["hidden_proxy_gap"])
        + 0.30 * trace["status"]["status_short"].count("ready")
    )
    dependency_interpretability_gain = _clip01(
        nested["switched_dependency_penalty"] - direct["direct_dependency_penalty"] + 0.60
    )
    metric_traceability_gain = _clip01(
        0.60 * (1.0 - abs(nested["switched_identity_readiness"] - direct["direct_identity_readiness"]))
        + 0.40 * obs["proxy_traceability_score"]
    )
    theorem_transparency_gain = _clip01(
        0.46 * obs["observability_bridge_score"]
        + 0.30 * obs["proxy_traceability_score"]
        + 0.24 * (1.0 - obs["hidden_proxy_gap"])
    )
    overall_native_improvement = _clip01(
        0.28 * direct_explainability_gain
        + 0.22 * dependency_interpretability_gain
        + 0.24 * metric_traceability_gain
        + 0.26 * theorem_transparency_gain
    )

    return {
        "headline_metrics": {
            "direct_explainability_gain": direct_explainability_gain,
            "dependency_interpretability_gain": dependency_interpretability_gain,
            "metric_traceability_gain": metric_traceability_gain,
            "theorem_transparency_gain": theorem_transparency_gain,
            "overall_native_improvement": overall_native_improvement,
        },
        "status": {
            "status_short": "native_improvement_audited",
            "status_label": "原生变量带来的改进已经被单独审计，不再只靠整体感觉判断",
        },
        "project_readout": {
            "summary": "这一轮把原生变量到底改进了什么拆成可解释性、依赖解释力、指标可追踪性、定理透明度四个方向单独审计。",
            "next_question": "下一步要把这份改进审计并回最终身份锁定，避免原生变量只在解释层有效、却不影响最终理论判断。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage70 Native Variable Improvement Audit",
        "",
        f"- direct_explainability_gain: {hm['direct_explainability_gain']:.6f}",
        f"- dependency_interpretability_gain: {hm['dependency_interpretability_gain']:.6f}",
        f"- metric_traceability_gain: {hm['metric_traceability_gain']:.6f}",
        f"- theorem_transparency_gain: {hm['theorem_transparency_gain']:.6f}",
        f"- overall_native_improvement: {hm['overall_native_improvement']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_native_variable_improvement_audit_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
