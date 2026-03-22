from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage70_direct_stability_counterexample_probe_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage69_direct_stability_strengthening import build_direct_stability_strengthening_summary
from stage70_native_observability_bridge import build_native_observability_bridge_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_direct_stability_counterexample_probe_summary() -> dict:
    stability = build_direct_stability_strengthening_summary()["headline_metrics"]
    obs = build_native_observability_bridge_summary()["headline_metrics"]

    adversarial_stability_support = _clip01(
        stability["strengthened_direct_stability_support"]
        - 0.08 * (1.0 - obs["observability_bridge_score"])
        - 0.07 * obs["hidden_proxy_gap"]
    )
    counterexample_pressure = _clip01(
        0.52 * stability["residual_stability_gap"] + 0.48 * obs["hidden_proxy_gap"]
    )
    survives_counterexample = adversarial_stability_support >= 0.72

    return {
        "headline_metrics": {
            "adversarial_stability_support": adversarial_stability_support,
            "counterexample_pressure": counterexample_pressure,
            "survives_counterexample": survives_counterexample,
        },
        "status": {
            "status_short": "counterexample_survived" if survives_counterexample else "counterexample_failed",
            "status_label": "直算稳定性在更强反例压力下的表现已经被显式测压",
        },
        "project_readout": {
            "summary": "这一轮把更强的隐藏代理缺口和可观测性损失施加到直算稳定性上，检查当前稳定性是否只是温和场景下成立。",
            "next_question": "下一步要把反例探针结果并回身份锁定，判断 near 状态在更强压力下是否仍能保持。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage70 Direct Stability Counterexample Probe",
        "",
        f"- adversarial_stability_support: {hm['adversarial_stability_support']:.6f}",
        f"- counterexample_pressure: {hm['counterexample_pressure']:.6f}",
        f"- survives_counterexample: {hm['survives_counterexample']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_direct_stability_counterexample_probe_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
