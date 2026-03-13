from __future__ import annotations

import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_persistent_training_block.json"
OUTPUT_PATH = ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_persistent_assessment.json"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def main() -> None:
    for _ in range(40):
        if INPUT_PATH.exists():
            break
        time.sleep(0.25)

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    long_horizon_score = clamp01(data["long_horizon_gain"] / 1.20)
    baseline_score = clamp01(data["baseline_margin"] / 0.80)
    external_score = clamp01(data["external_margin"] / 0.70)
    language_score = clamp01(data["language_proxy_margin"] / 0.20)
    task_score = clamp01(min(data["proto_final"]["task_acc"], data["proto_external"]["task_acc"]))
    theorem_score = clamp01(min(data["proto_final"]["theorem_prob_mean"], data["proto_external"]["theorem_prob_mean"]) / 0.70)
    read_score = clamp01(min(data["proto_final"]["read_mean"], data["proto_external"]["read_mean"]) / 0.22)
    write_score = clamp01(data["proto_final"]["write_mean"] / 0.18)
    transport_score = clamp01(min(data["proto_final"]["transport_margin"], data["proto_external"]["transport_margin"]) / 10.0)
    stress_score = clamp01(min(data["proto_final"]["stress_balance"], data["proto_external"]["stress_balance"]))
    online_score = clamp01(0.75 + max(0.0, data["online_gain"]) / 0.03)
    rollback_score = clamp01(1.0 - data["rollback_error"] * 1_000_000.0)
    data_score = clamp01(min(1.0, data["data_stats"]["sampled_chars"] / 700000.0))
    history_events = (
        len(data.get("proto_history", []))
        + len(data.get("baseline_history", []))
        + len(data.get("stabilization_history", []))
        + len(data.get("write_recovery_history", []))
        + len(data.get("gate_alignment_history", []))
        + len(data.get("recovery_history", []))
    )
    history_score = clamp01(min(1.0, history_events / 50.0))
    recovery_bonus = 0.05 if data.get("auto_adjust_triggered") else 0.02
    closure_bonus = 0.10 if (
        data["training_pass"]
        and data["baseline_outperform_pass"]
        and data["external_outperform_pass"]
        and data["language_proxy_pass"]
        and data["online_update_pass"]
        and data["rollback_pass"]
        and min(data["proto_final"]["task_acc"], data["proto_external"]["task_acc"]) >= 0.97
        and min(data["proto_final"]["theorem_prob_mean"], data["proto_external"]["theorem_prob_mean"]) >= 0.60
        and min(data["proto_final"]["read_mean"], data["proto_external"]["read_mean"]) >= 0.18
        and data["proto_final"]["write_mean"] >= 0.14
        and data["long_horizon_gain"] >= 0.80
    ) else 0.0

    total_score = clamp01(
        0.12 * long_horizon_score
        + 0.10 * baseline_score
        + 0.10 * external_score
        + 0.08 * language_score
        + 0.10 * task_score
        + 0.08 * theorem_score
        + 0.07 * read_score
        + 0.06 * write_score
        + 0.06 * transport_score
        + 0.06 * stress_score
        + 0.06 * online_score
        + 0.05 * rollback_score
        + 0.03 * data_score
        + 0.03 * history_score
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
        and min(data["proto_final"]["task_acc"], data["proto_external"]["task_acc"]) >= 0.97
        and min(data["proto_final"]["theorem_prob_mean"], data["proto_external"]["theorem_prob_mean"]) >= 0.60
        and min(data["proto_final"]["read_mean"], data["proto_external"]["read_mean"]) >= 0.18
        and data["proto_final"]["write_mean"] >= 0.14
        and data["long_horizon_gain"] >= 0.80
        and total_score >= 0.95
    )

    result = {
        "long_horizon_score": long_horizon_score,
        "baseline_score": baseline_score,
        "external_score": external_score,
        "language_score": language_score,
        "task_score": task_score,
        "theorem_score": theorem_score,
        "read_score": read_score,
        "write_score": write_score,
        "transport_score": transport_score,
        "stress_score": stress_score,
        "online_score": online_score,
        "rollback_score": rollback_score,
        "data_score": data_score,
        "history_score": history_score,
        "recovery_bonus": recovery_bonus,
        "closure_bonus": closure_bonus,
        "total_score": total_score,
        "persistent_training_ready": ready,
        "prototype_name": data["prototype_name"],
        "baseline_name": data["baseline_name"],
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
