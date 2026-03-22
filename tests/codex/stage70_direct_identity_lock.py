from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage70_direct_identity_lock_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage69_direct_identity_migration import build_direct_identity_migration_summary
from stage70_direct_stability_counterexample_probe import build_direct_stability_counterexample_probe_summary
from stage70_native_variable_improvement_audit import build_native_variable_improvement_audit_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_direct_identity_lock_summary() -> dict:
    migration = build_direct_identity_migration_summary()["headline_metrics"]
    counter = build_direct_stability_counterexample_probe_summary()["headline_metrics"]
    audit = build_native_variable_improvement_audit_summary()["headline_metrics"]

    locked_identity_readiness = _clip01(
        0.34 * migration["migrated_direct_identity_readiness"]
        + 0.24 * migration["migrated_direct_falsifiability"]
        + 0.20 * counter["adversarial_stability_support"]
        + 0.22 * audit["overall_native_improvement"]
    )
    identity_lock_confidence = _clip01(
        0.38 * locked_identity_readiness
        + 0.24 * (1.0 - counter["counterexample_pressure"])
        + 0.18 * audit["metric_traceability_gain"]
        + 0.20 * audit["theorem_transparency_gain"]
    )

    return {
        "headline_metrics": {
            "locked_identity_readiness": locked_identity_readiness,
            "identity_lock_confidence": identity_lock_confidence,
        },
        "status": {
            "status_short": "near_first_principles_theory" if locked_identity_readiness >= 0.79 else "phenomenological_transition",
            "status_label": "直算链身份已经被进一步锁定，后续应继续围绕稳定性定理和原生实测推进",
        },
        "project_readout": {
            "summary": "这一轮把主判断迁移、反例稳定性和原生变量改进审计重新合流，检查直算链的 near 状态是否已经更稳。",
            "next_question": "下一步要把原生变量从代理量继续推向更真实的网络可观测量，减少最后的隐藏代理缺口。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage70 Direct Identity Lock",
        "",
        f"- locked_identity_readiness: {hm['locked_identity_readiness']:.6f}",
        f"- identity_lock_confidence: {hm['identity_lock_confidence']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_direct_identity_lock_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
