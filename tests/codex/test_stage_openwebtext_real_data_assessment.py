from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "tests" / "codex_temp" / "openwebtext_real_data_block.json"
OUTPUT_PATH = ROOT / "tests" / "codex_temp" / "openwebtext_real_data_assessment.json"


def main() -> None:
    payload = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    loss_drop = float(payload["loss_drop"])
    train_score = min(1.0, max(0.0, loss_drop / 1.0))
    online_score = 1.0 if payload["online_update_pass"] else 0.0
    rollback_score = 1.0 if payload["rollback_pass"] else 0.0
    theorem_score = float(payload["theorem_survival"])
    read_score = float(payload["stable_read"])
    write_score = float(payload["guarded_write"])
    margin_score = min(1.0, max(0.0, float(payload["transport_margin"]) / 2.0))
    data_score = min(1.0, float(payload["data_stats"]["sampled_chars"]) / 50000.0)
    total_score = (
        0.20 * train_score
        + 0.15 * online_score
        + 0.10 * rollback_score
        + 0.15 * theorem_score
        + 0.10 * read_score
        + 0.10 * write_score
        + 0.10 * margin_score
        + 0.10 * data_score
    )
    result = {
        "train_score": train_score,
        "online_score": online_score,
        "rollback_score": rollback_score,
        "theorem_score": theorem_score,
        "read_score": read_score,
        "write_score": write_score,
        "margin_score": margin_score,
        "data_score": data_score,
        "total_score": total_score,
        "real_data_training_ready": total_score >= 0.78,
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
