from __future__ import annotations

import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_longterm_training_block.json"
OUTPUT_PATH = ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_longterm_assessment.json"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def main() -> None:
    for _ in range(20):
        if INPUT_PATH.exists():
            break
        time.sleep(0.25)
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    train_score = clamp01((data["proto_initial"]["loss"] - data["proto_final"]["loss"]) / max(1e-6, data["proto_initial"]["loss"]))
    baseline_score = clamp01(data["baseline_margin"] / 0.12)
    theorem_score = clamp01(data["proto_final"]["theorem_survival"])
    read_score = clamp01(data["proto_final"]["stable_read"])
    guarded_write_score = clamp01(data["proto_final"]["guarded_write"])
    recovery_depth = clamp01(
        0.5 * min(1.0, len(data.get("stabilization_history", [])) / 2.0)
        + 0.5 * min(1.0, len(data.get("recovery_history", [])) / 2.0)
    )
    stress_write_score = clamp01(
        0.6 * data["proto_final"]["stress_balance"]
        + 0.4 * max(0.0, data["online_delta"]) / 0.01
    )
    write_score = max(guarded_write_score, 0.5 * recovery_depth + 0.5 * stress_write_score)
    online_score = clamp01(0.7 + max(0.0, data["online_delta"]) / 0.03)
    rollback_score = clamp01(1.0 - data["rollback_error"] * 1_000_000.0)
    margin_score = clamp01(data["proto_final"]["transport_margin"] / 10.0)
    data_score = clamp01(min(1.0, data["data_stats"]["sampled_chars"] / 180000.0))
    recovery_bonus = 0.08 if data["auto_recovery_triggered"] else 0.0
    closure_bonus = 0.10 if (
        data["training_pass"]
        and data["baseline_outperform_pass"]
        and data["online_update_pass"]
        and data["rollback_pass"]
        and data["proto_final"]["theorem_survival"] >= 0.98
        and data["proto_final"]["stable_read"] >= 0.98
    ) else 0.0

    total_score = clamp01(
        0.20 * train_score
        + 0.15 * baseline_score
        + 0.15 * theorem_score
        + 0.10 * read_score
        + 0.10 * write_score
        + 0.10 * online_score
        + 0.10 * rollback_score
        + 0.05 * margin_score
        + 0.05 * data_score
        + recovery_bonus
        + closure_bonus
    )

    ready = (
        data["training_pass"]
        and data["baseline_outperform_pass"]
        and data["online_update_pass"]
        and data["rollback_pass"]
        and data["proto_final"]["theorem_survival"] >= 0.98
        and data["proto_final"]["stable_read"] >= 0.98
        and total_score >= 0.92
    )

    result = {
        "train_score": train_score,
        "baseline_score": baseline_score,
        "theorem_score": theorem_score,
        "read_score": read_score,
        "write_score": write_score,
        "online_score": online_score,
        "rollback_score": rollback_score,
        "margin_score": margin_score,
        "data_score": data_score,
        "recovery_bonus": recovery_bonus,
        "closure_bonus": closure_bonus,
        "total_score": total_score,
        "real_longterm_training_ready": ready,
        "prototype_name": data["prototype_name"],
        "baseline_name": data["baseline_name"],
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
