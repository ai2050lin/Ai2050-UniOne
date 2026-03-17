from __future__ import annotations

from types import SimpleNamespace

import torch

from deepseek7b_three_pool_structure_scan import gate_spec_for_layer, slice_gate_output, zero_gate_indices


class FakeGateProjMLP:
    def __init__(self) -> None:
        self.gate_proj = torch.nn.Linear(4, 6, bias=False)


class FakeGateUpProjMLP:
    def __init__(self) -> None:
        self.gate_up_proj = torch.nn.Linear(4, 12, bias=False)


def test_gate_spec_for_gate_proj() -> None:
    layer = SimpleNamespace(mlp=FakeGateProjMLP())
    spec = gate_spec_for_layer(layer)
    assert spec.d_ff == 6
    assert spec.gate_start == 0
    assert spec.gate_end == 6
    assert spec.module is layer.mlp.gate_proj


def test_gate_spec_for_gate_up_proj() -> None:
    layer = SimpleNamespace(mlp=FakeGateUpProjMLP())
    spec = gate_spec_for_layer(layer)
    assert spec.d_ff == 6
    assert spec.gate_start == 0
    assert spec.gate_end == 6
    assert spec.module is layer.mlp.gate_up_proj


def test_slice_gate_output_uses_gate_half_for_glm() -> None:
    layer = SimpleNamespace(mlp=FakeGateUpProjMLP())
    spec = gate_spec_for_layer(layer)
    output = torch.arange(24, dtype=torch.float32).view(1, 2, 12)
    gate = slice_gate_output(output, spec)
    expected = output[..., :6]
    assert torch.equal(gate, expected)


def test_zero_gate_indices_only_changes_gate_half() -> None:
    layer = SimpleNamespace(mlp=FakeGateUpProjMLP())
    spec = gate_spec_for_layer(layer)
    output = torch.arange(12, dtype=torch.float32).view(1, 1, 12)
    ablated = zero_gate_indices(output, spec, torch.tensor([1, 4], dtype=torch.long))
    assert torch.equal(ablated[..., 6:], output[..., 6:])
    assert float(ablated[0, 0, 1]) == 0.0
    assert float(ablated[0, 0, 4]) == 0.0
    assert float(ablated[0, 0, 0]) == float(output[0, 0, 0])
    assert float(ablated[0, 0, 5]) == float(output[0, 0, 5])
