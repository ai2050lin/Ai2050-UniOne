from __future__ import annotations

import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "tests" / "codex_temp" / "icspb_v2_real_long_horizon_training_block.json"
OUTPUT_PATH = ROOT / "tests" / "codex_temp" / "icspb_v2_real_long_horizon_assessment.json"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def main() -> None:
    for _ in range(40):
        if INPUT_PATH.exists():
            break
        time.sleep(0.25)

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    long_horizon_score = clamp01(data["long_horizon_gain"] / 0.90)
    baseline_score = clamp01(data["baseline_margin"] / 0.45)
    external_score = clamp01(data["external_margin"] / 0.45)
    language_score = clamp01(data["language_proxy_margin"] / 0.14)
    theorem_score = clamp01(min(data["proto_final"]["theorem_survival"], data["proto_external"]["theorem_survival"]))
    read_score = clamp01(min(data["proto_final"]["stable_read"], data["proto_external"]["stable_read"]))
    write_score = clamp01(data["proto_final"]["guarded_write"] / 0.40)
    task_score = clamp01(min(data["proto_final"]["task_acc"], data["proto_external"]["task_acc"]) / 0.80)
    online_score = clamp01(0.75 + max(0.0, data["online_gain"]) / 0.03)
    rollback_score = clamp01(1.0 - data["rollback_error"] * 1_000_000.0)
    data_score = clamp01(min(1.0, data["data_stats"]["sampled_chars"] / 500000.0))
    recovery_history_len = (
        len(data.get("stabilization_history", []))
        + len(data.get("write_recovery_history", []))
        + len(data.get("structural_recovery_history", []))
        + len(data.get("consolidation_history", []))
        + len(data.get("gate_alignment_history", []))
        + len(data.get("full_structural_restoration_history", []))
    )
    history_score = clamp01(min(1.0, (len(data["proto_history"]) + recovery_history_len) / 28.0))
    recovery_bonus = 0.08 if data["auto_recovery_triggered"] else 0.03
    closure_bonus = 0.10 if (
        data["training_pass"]
        and data["baseline_outperform_pass"]
        and data["external_outperform_pass"]
        and data["language_proxy_pass"]
        and data["online_update_pass"]
        and data["rollback_pass"]
        and data["proto_final"]["theorem_survival"] >= 0.99
        and data["proto_external"]["theorem_survival"] >= 0.99
        and data["proto_final"]["stable_read"] >= 0.99
        and data["proto_external"]["stable_read"] >= 0.99
        and data["proto_final"]["guarded_write"] >= 0.40
        and data["proto_external"]["task_acc"] >= 0.80
        and data["long_horizon_gain"] >= 0.90
    ) else 0.0

    total_score = clamp01(
        0.14 * long_horizon_score
        + 0.11 * baseline_score
        + 0.11 * external_score
        + 0.10 * language_score
        + 0.11 * theorem_score
        + 0.08 * read_score
        + 0.08 * write_score
        + 0.08 * task_score
        + 0.06 * online_score
        + 0.05 * rollback_score
        + 0.04 * data_score
        + 0.04 * history_score
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
        and data["proto_final"]["theorem_survival"] >= 0.99
        and data["proto_external"]["theorem_survival"] >= 0.99
        and data["proto_final"]["stable_read"] >= 0.99
        and data["proto_external"]["stable_read"] >= 0.99
        and data["proto_final"]["guarded_write"] >= 0.40
        and data["proto_external"]["task_acc"] >= 0.80
        and data["long_horizon_gain"] >= 0.90
        and total_score >= 0.95
    )

    result = {
        "long_horizon_score": long_horizon_score,
        "baseline_score": baseline_score,
        "external_score": external_score,
        "language_score": language_score,
        "theorem_score": theorem_score,
        "read_score": read_score,
        "write_score": write_score,
        "task_score": task_score,
        "online_score": online_score,
        "rollback_score": rollback_score,
        "data_score": data_score,
        "history_score": history_score,
        "recovery_bonus": recovery_bonus,
        "closure_bonus": closure_bonus,
        "total_score": total_score,
        "real_long_horizon_ready": ready,
        "prototype_name": data["prototype_name"],
        "baseline_name": data["baseline_name"],
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
