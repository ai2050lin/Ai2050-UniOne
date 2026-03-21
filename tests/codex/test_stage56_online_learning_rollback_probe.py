from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_online_learning_rollback_probe.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_online_learning_rollback_probe_positive() -> None:
    mod = _load_module()
    summary = mod.build_online_learning_rollback_probe_summary(base_steps=200, online_steps=60)
    hm = summary["headline_metrics"]
    assert hm["online_fit_after"] >= hm["online_fit_before"]
    assert hm["base_retention"] > 0.0
    assert hm["route_split_retention"] >= 0.0


if __name__ == "__main__":
    test_online_learning_rollback_probe_positive()
