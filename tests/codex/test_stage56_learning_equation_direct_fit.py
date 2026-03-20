import json
from pathlib import Path

try:
    from tests.codex.stage56_learning_equation_direct_fit import fit_learning_bridge
except ModuleNotFoundError:
    from stage56_learning_equation_direct_fit import fit_learning_bridge  # type: ignore


def test_learning_equation_direct_fit_runs(tmp_path: Path) -> None:
    checkpoint = {"atlas_freeze_step": 6.0, "frontier_shift_step": 6.0, "boundary_hardening_step": 8.0}
    gradient = {"delta": {"atlas_grad_delta": -0.1, "frontier_grad_delta": -2.0, "boundary_grad_delta": -0.5}}
    attractor = {"gap_shift": 0.04}
    cp = tmp_path / "checkpoint.json"
    gp = tmp_path / "gradient.json"
    ap = tmp_path / "attractor.json"
    cp.write_text(json.dumps(checkpoint, ensure_ascii=False), encoding="utf-8")
    gp.write_text(json.dumps(gradient, ensure_ascii=False), encoding="utf-8")
    ap.write_text(json.dumps(attractor, ensure_ascii=False), encoding="utf-8")
    summary = fit_learning_bridge(cp, gp, ap)
    assert summary["drives"]["frontier_learning_drive_v2"] > summary["drives"]["atlas_learning_drive_v2"]
