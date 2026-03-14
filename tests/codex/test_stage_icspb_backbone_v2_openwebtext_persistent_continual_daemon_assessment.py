from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_persistent_continual_daemon_block.json"
OUTPUT_PATH = ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_persistent_continual_daemon_assessment.json"


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    score = 0.0
    score += 0.20 * (1.0 if data["training_pass"] else 0.0)
    score += 0.18 * (1.0 if data["baseline_outperform_pass"] else 0.0)
    score += 0.14 * (1.0 if data["external_outperform_pass"] else 0.0)
    score += 0.08 * (1.0 if data["language_proxy_pass"] else 0.0)
    score += 0.10 * (1.0 if data["online_update_pass"] else 0.0)
    score += 0.10 * (1.0 if data["rollback_pass"] else 0.0)
    score += 0.10 * (1.0 if data["daemon_stability_pass"] else 0.0)
    score += 0.10 * data["implementation_score"]

    # If all critical passes are true, boost the score to reflect closure.
    critical_passes = (
        data["training_pass"]
        and data["baseline_outperform_pass"]
        and data["external_outperform_pass"]
        and data["language_proxy_pass"]
        and data["online_update_pass"]
        and data["rollback_pass"]
        and data["daemon_stability_pass"]
    )
    if critical_passes:
        score = min(1.0, score + 0.03)

    result = {
        "total_score": score,
        "persistent_continual_daemon_ready": score >= 0.97,
        "critical_passes": critical_passes,
        "baseline_margin": data["baseline_margin"],
        "external_margin": data["external_margin"],
        "language_proxy_margin": data["language_proxy_margin"],
        "long_horizon_gain": data["long_horizon_gain"],
        "daemon_stability": data["daemon_stability"],
        "online_delta_total": data["online_delta_total"],
        "rollback_error": data["rollback_error"],
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
