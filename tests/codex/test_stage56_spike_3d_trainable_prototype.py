from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_spike_3d_trainable_prototype.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_spike_3d_trainable_prototype_positive() -> None:
    mod = _load_module()
    summary = mod.build_spike_3d_trainable_prototype_summary(steps=120)
    hm = summary["headline_metrics"]
    assert 0.0 <= hm["topo_train_fit"] <= 1.0
    assert 0.0 <= hm["topo_heldout_generalization"] <= 1.0
    assert 0.0 <= hm["local_transport_score"] <= 1.0
    assert 0.0 <= hm["path_reuse_score"] <= 1.0
    assert hm["topology_trainable_margin"] > 0.0


if __name__ == "__main__":
    test_spike_3d_trainable_prototype_positive()
