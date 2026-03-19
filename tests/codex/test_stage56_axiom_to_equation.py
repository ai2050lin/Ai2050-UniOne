from __future__ import annotations

from stage56_axiom_to_equation import build_constraints


def test_build_constraints_contains_six_axiom_constraints() -> None:
    out = build_constraints()
    assert out["record_type"] == "stage56_axiom_to_equation_summary"
    assert len(out["constraints"]) == 6
    assert "U_fit_plus" in out["proto_equation_system"]["master_equation"]
