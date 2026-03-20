import json
from pathlib import Path

try:
    from tests.codex.stage56_checkpoint_sequence_harvest import harvest_sequence
except ModuleNotFoundError:
    from stage56_checkpoint_sequence_harvest import harvest_sequence  # type: ignore


def test_checkpoint_sequence_harvest_detects_phases(tmp_path: Path) -> None:
    rows = [
        {"phase": "base_train", "valid_disc_mean": 0.1, "valid_general_norm": 1.0},
        {"phase": "base_train", "valid_disc_mean": 0.2, "valid_general_norm": 3.0},
        {"phase": "base_train", "valid_disc_mean": 0.15, "valid_general_norm": 2.0},
        {"phase": "online_inject", "disc_mean": 0.4},
        {"phase": "online_inject", "disc_mean": 0.8},
    ]
    path = tmp_path / "checkpoints.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    summary = harvest_sequence(path)
    assert summary["atlas_freeze_step"] == 2.0
    assert summary["frontier_shift_step"] == 2.0
    assert summary["boundary_hardening_step"] == 2.0
