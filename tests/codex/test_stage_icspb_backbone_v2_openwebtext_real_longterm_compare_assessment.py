from __future__ import annotations

import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_real_longterm_compare_block.json"
OUTPUT_PATH = ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_real_longterm_compare_assessment.json"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def main() -> None:
    for _ in range(80):
        if INPUT_PATH.exists():
            break
        time.sleep(0.25)
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    long_run_score = clamp01(data["long_horizon_gain"] / 0.90)
    baseline_score = clamp01(data["baseline_margin"] / 0.60)
    external_score = clamp01(data["external_margin"] / 0.50)
    language_score = clamp01(data["language_proxy_margin"] / 0.10)
    theorem_score = clamp01(data["proto_final"]["theorem_survival"])
    read_score = clamp01(data["proto_final"]["stable_read"])
    write_score = clamp01(max(data["proto_final"]["guarded_write"], 0.70 * data["proto_final"]["stress_balance"]))
    online_score = clamp01(0.75 + max(0.0, data["online_delta_total"]) / 0.05)
    rollback_score = clamp01(1.0 - data["rollback_error"] * 1_000_000.0)
    daemon_score = clamp01(data["daemon_stability"])
    margin_score = clamp01(data["proto_final"]["transport_margin"] / 10.0)
    data_score = clamp01(min(1.0, data["data_stats"]["sampled_chars"] / 240000.0))
    history_score = clamp01(min(1.0, len(data["proto_history_1"]) / 12.0))
    recovery_bonus = 0.08 if data["auto_recovery_triggered"] else 0.02
    closure_bonus = 0.14 if (
        data["training_pass"]
        and data["baseline_outperform_pass"]
        and data["external_outperform_pass"]
        and data["language_proxy_pass"]
        and data["online_update_pass"]
        and data["rollback_pass"]
        and data["proto_final"]["theorem_survival"] >= 0.995
        and data["proto_final"]["stable_read"] >= 0.995
        and data["long_horizon_gain"] > 0.85
        and data["daemon_stability"] >= 0.992
    ) else 0.0

    total_score = clamp01(
        0.16 * long_run_score
        + 0.12 * baseline_score
        + 0.12 * external_score
        + 0.08 * language_score
        + 0.10 * theorem_score
        + 0.08 * read_score
        + 0.06 * write_score
        + 0.08 * online_score
        + 0.06 * rollback_score
        + 0.05 * daemon_score
        + 0.04 * margin_score
        + 0.03 * data_score
        + 0.02 * history_score
        + recovery_bonus
        + closure_bonus
    )

    ready = (
        data["training_pass"]
        and data["baseline_outperform_pass"]
        and data["external_outperform_pass"]
        and data["language_proxy_pass"]
        and data["online_update_pass"]
        and data["rollback_pass"]
        and data["proto_final"]["theorem_survival"] >= 0.995
        and data["proto_final"]["stable_read"] >= 0.995
        and data["long_horizon_gain"] > 0.85
        and data["daemon_stability"] >= 0.992
        and total_score >= 0.96
    )

    result = {
        "long_run_score": long_run_score,
        "baseline_score": baseline_score,
        "external_score": external_score,
        "language_score": language_score,
        "theorem_score": theorem_score,
        "read_score": read_score,
        "write_score": write_score,
        "online_score": online_score,
        "rollback_score": rollback_score,
        "daemon_score": daemon_score,
        "margin_score": margin_score,
        "data_score": data_score,
        "history_score": history_score,
        "recovery_bonus": recovery_bonus,
        "closure_bonus": closure_bonus,
        "total_score": total_score,
        "real_longterm_compare_ready": ready,
        "prototype_name": data["prototype_name"],
        "baseline_name": data["baseline_name"],
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
