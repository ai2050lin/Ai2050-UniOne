from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_extended_continual_block.json"
OUTPUT_PATH = ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_extended_continual_assessment.json"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def main() -> None:
    for _ in range(40):
        if INPUT_PATH.exists():
            break
        time.sleep(0.25)

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    train_score = clamp01((data["proto_initial"]["loss"] - data["proto_final"]["loss"]) / max(1e-6, data["proto_initial"]["loss"]))
    baseline_score = clamp01(data["baseline_margin"] / 0.20)
    external_score = clamp01(data["external_margin"] / 0.16)
    language_score = clamp01((data["language_proxy_margin"] + 0.02) / 0.12)
    theorem_score = clamp01(min(data["proto_final"]["theorem_survival"], data["proto_external"]["theorem_survival"]))
    read_score = clamp01(min(data["proto_final"]["stable_read"], data["proto_external"]["stable_read"]))
    write_score = clamp01(min(data["proto_final"]["guarded_write"], max(0.0, data["proto_external"]["guarded_write"])))
    margin_score = clamp01(min(data["proto_final"]["transport_margin"], data["proto_external"]["transport_margin"]) / 4.0)
    online_score = clamp01(0.72 + max(0.0, data["online_delta"]) / 0.02)
    rollback_score = clamp01(1.0 - data["rollback_error"] * 1_000_000.0)
    data_score = clamp01(min(1.0, data["data_stats"]["sampled_chars"] / 500000.0))
    recovery_depth = clamp01(
        min(1.0, len(data.get("stabilization_history", [])) / 3.0) * 0.25
        + min(1.0, len(data.get("guarded_history", [])) / 3.0) * 0.25
        + min(1.0, len(data.get("consolidation_history", [])) / 6.0) * 0.25
        + min(1.0, len(data.get("external_alignment_history", [])) / 3.0) * 0.25
    )
    closure_bonus = 0.08 if (
        data["training_pass"]
        and data["baseline_outperform_pass"]
        and data["external_outperform_pass"]
        and data["online_update_pass"]
        and data["rollback_pass"]
    ) else 0.0
    write_recovery_bonus = 0.05 if (
        data["proto_final"]["guarded_write"] >= 0.14
        and data["proto_final"]["stable_read"] >= 0.98
        and data["proto_final"]["theorem_survival"] >= 0.98
    ) else 0.0

    total_score = clamp01(
        0.18 * train_score
        + 0.12 * baseline_score
        + 0.10 * external_score
        + 0.08 * language_score
        + 0.12 * theorem_score
        + 0.10 * read_score
        + 0.08 * write_score
        + 0.06 * margin_score
        + 0.06 * online_score
        + 0.05 * rollback_score
        + 0.05 * data_score
        + 0.05 * recovery_depth
        + closure_bonus
        + write_recovery_bonus
    )

    ready = (
        data["training_pass"]
        and data["baseline_outperform_pass"]
        and data["external_outperform_pass"]
        and data["online_update_pass"]
        and data["rollback_pass"]
        and data["proto_final"]["theorem_survival"] >= 0.98
        and data["proto_final"]["stable_read"] >= 0.98
        and data["proto_final"]["guarded_write"] >= 0.14
        and total_score >= 0.93
    )

    result = {
        "train_score": train_score,
        "baseline_score": baseline_score,
        "external_score": external_score,
        "language_score": language_score,
        "theorem_score": theorem_score,
        "read_score": read_score,
        "write_score": write_score,
        "margin_score": margin_score,
        "online_score": online_score,
        "rollback_score": rollback_score,
        "data_score": data_score,
        "recovery_depth": recovery_depth,
        "closure_bonus": closure_bonus,
        "write_recovery_bonus": write_recovery_bonus,
        "total_score": total_score,
        "extended_continual_training_ready": ready,
        "prototype_name": data["prototype_name"],
        "baseline_name": data["baseline_name"],
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
