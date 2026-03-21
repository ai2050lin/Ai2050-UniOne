from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_contextual_trainable_prototype.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_contextual_trainable_prototype_positive() -> None:
    mod = _load_module()
    summary = mod.build_contextual_trainable_prototype_summary(steps=200)
    hm = summary["headline_metrics"]
    assert hm["train_fit"] > 0.0
    assert hm["heldout_generalization"] > 0.0

