from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_language_sufficiency_analysis.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_language_is_not_sufficient_for_all_goals() -> None:
    mod = _load_module()
    summary = mod.build_language_sufficiency_summary()
    hm = summary["headline_metrics"]
    assert hm["language_only_sufficiency"] > 0.7
    assert hm["missing_nonlanguage_mass"] > 0.0
    assert hm["language_solves_all_score"] < hm["language_only_sufficiency"]
