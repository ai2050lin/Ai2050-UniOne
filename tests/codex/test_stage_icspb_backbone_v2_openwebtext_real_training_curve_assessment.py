from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_real_training_curve_block.json"
OUTPUT_PATH = ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_real_training_curve_assessment.json"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def main() -> None:
    for _ in range(80):
        if INPUT_PATH.exists():
            break
        time.sleep(0.25)

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    # 长期真实文本训练不要求把 loss 压到极低，更重要的是达到稳定下降并显著优于基线。
    # 这里把“达到 25% 相对下降”视为训练有效的满分带。
    train_score = clamp01(
        (data["proto_initial"]["loss"] - data["proto_final"]["loss"])
        / max(1e-6, data["proto_initial"]["loss"] * 0.25)
    )
    baseline_score = clamp01(data["baseline_margin"] / 0.50)
    external_score = clamp01(data["external_margin"] / 0.40)
    language_score = clamp01((data["language_proxy_margin"] + 0.02) / 0.15)
    theorem_score = clamp01(min(data["proto_final"]["theorem_survival"], data["proto_external"]["theorem_survival"]))
    read_score = clamp01(min(data["proto_final"]["stable_read"], data["proto_external"]["stable_read"]))
    write_score = clamp01(min(data["proto_final"]["guarded_write"], max(0.0, data["proto_external"]["guarded_write"])))
    margin_score = clamp01(min(data["proto_final"]["transport_margin"], data["proto_external"]["transport_margin"]) / 6.0)
    online_score = clamp01(0.75 + max(0.0, data["online_delta"]) / 0.03)
    rollback_score = clamp01(1.0 - data["rollback_error"] * 1_000_000.0)
    data_score = clamp01(min(1.0, data["data_stats"]["sampled_chars"] / 900000.0))
    recovery_depth = clamp01(
        min(1.0, len(data["curve_history"].get("proto_warmup", [])) / 8.0) * 0.20
        + min(1.0, len(data["curve_history"].get("proto_guarded", [])) / 6.0) * 0.20
        + min(1.0, len(data["curve_history"].get("proto_external", [])) / 4.0) * 0.20
        + min(1.0, len(data["stabilization_history"]) / 6.0) * 0.20
        + min(1.0, len(data["curve_history"].get("proto_recovery", [])) / 5.0) * 0.20
    )
    closure_bonus = 0.10 if (
        data["training_pass"]
        and data["baseline_outperform_pass"]
        and data["external_outperform_pass"]
        and data["online_update_pass"]
        and data["rollback_pass"]
    ) else 0.0

    total_score = clamp01(
        0.17 * train_score
        + 0.12 * baseline_score
        + 0.10 * external_score
        + 0.08 * language_score
        + 0.12 * theorem_score
        + 0.11 * read_score
        + 0.08 * write_score
        + 0.07 * margin_score
        + 0.06 * online_score
        + 0.05 * rollback_score
        + 0.04 * data_score
        + 0.05 * recovery_depth
        + closure_bonus
    )

    ready = (
        data["training_pass"]
        and data["baseline_outperform_pass"]
        and data["external_outperform_pass"]
        and data["online_update_pass"]
        and data["rollback_pass"]
        and data["proto_final"]["theorem_survival"] >= 0.985
        and data["proto_final"]["stable_read"] >= 0.985
        and data["proto_final"]["guarded_write"] >= 0.14
        and total_score >= 0.95
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
        "total_score": total_score,
        "real_training_curve_ready": ready,
        "prototype_name": data["prototype_name"],
        "baseline_name": data["baseline_name"],
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
