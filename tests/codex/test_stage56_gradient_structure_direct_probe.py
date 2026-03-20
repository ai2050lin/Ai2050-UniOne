from __future__ import annotations

import torch

from stage56_gradient_structure_direct_probe import build_summary
from stage56_prototype_online_learning_experiment import Stage56ProtoNet


def test_build_summary_emits_gradient_deltas(tmp_path) -> None:
    model = Stage56ProtoNet(vocab_size=7)
    model_path = tmp_path / "m.pt"
    torch.save(model.state_dict(), model_path)
    summary = build_summary(model_path, order=7, base_cut=4)
    assert "boundary_grad_delta" in summary["delta"]
