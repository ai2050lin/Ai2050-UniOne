import json
from pathlib import Path

import numpy as np

from deepseek7b_multidim_encoding_probe import (
    build_frontier_curve,
    build_specific_top_neurons,
    cumulative_mass_indices,
    load_pairs_from_json,
    resolve_gate_module_and_width,
    select_desc_indices,
    summarize_pair_frontier,
)


def test_select_desc_indices_topk_and_all_effective():
    vec = np.asarray([0.0, 3.0, 1.5, -2.0, 2.0], dtype=np.float32)

    top2 = select_desc_indices(vec, 2, positive_only=True).tolist()
    assert top2 == [1, 4]

    all_effective = select_desc_indices(vec, 0, positive_only=True).tolist()
    assert all_effective == [1, 4, 2]


def test_cumulative_mass_indices_tracks_support_size():
    vec = np.asarray([5.0, 3.0, 2.0, 0.0], dtype=np.float32)

    mass50 = cumulative_mass_indices(vec, 0.50).tolist()
    mass80 = cumulative_mass_indices(vec, 0.80).tolist()
    mass95 = cumulative_mass_indices(vec, 0.95).tolist()

    assert mass50 == [0]
    assert mass80 == [0, 1]
    assert mass95 == [0, 1, 2]


def test_build_frontier_curve_tracks_curve_points():
    vec = np.asarray([5.0, 3.0, 2.0, 0.0], dtype=np.float32)
    selected = select_desc_indices(vec, 0, positive_only=True)

    curve, idx_map = build_frontier_curve(
        mean_abs=vec,
        selected_idx=selected,
        d_ff=4,
        n_layers=1,
        mass_ratios=[0.5, 0.8, 0.95],
    )

    assert [row["neuron_count"] for row in curve] == [1, 2, 3]
    assert [round(row["compaction_ratio"], 4) for row in curve] == [0.3333, 0.6667, 1.0]
    assert idx_map["50"].tolist() == [0]
    assert idx_map["80"].tolist() == [0, 1]
    assert idx_map["95"].tolist() == [0, 1, 2]


def test_summarize_pair_frontier_emits_pair_curve():
    vec = np.asarray([5.0, 3.0, 2.0, 0.0], dtype=np.float32)
    out = summarize_pair_frontier(vec, top_k=0, d_ff=4, n_layers=1)
    assert out["selected_neuron_count"] == 3
    assert out["frontier_curve"][0]["mass_ratio"] == 0.01


def test_load_pairs_from_json_reads_external_corpus(tmp_path: Path):
    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps(
            {
                "style": [{"id": "s0", "a": "A", "b": "B"}],
                "logic": [{"a": "C", "b": "D"}],
                "syntax": [{"id": "x0", "a": "E", "b": "F"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = load_pairs_from_json(str(path))

    assert payload["style"][0]["id"] == "s0"
    assert payload["logic"][0]["id"] == "logic_pair_0000"
    assert payload["syntax"][0]["a"] == "E"


def test_load_pairs_from_json_supports_nested_pairs_key(tmp_path: Path):
    path = tmp_path / "pairs_nested.json"
    path.write_text(
        json.dumps(
            {
                "record_type": "stage56_natural_corpus_contrast_pairs",
                "pairs": {
                    "style": [{"a": "A", "b": "B"}],
                    "logic": [{"a": "C", "b": "D"}],
                    "syntax": [{"a": "E", "b": "F"}],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = load_pairs_from_json(str(path))

    assert payload["style"][0]["id"] == "style_pair_0000"
    assert payload["logic"][0]["b"] == "D"


def test_build_specific_top_neurons_all_effective_keeps_positive_specific_support():
    dims_out = {
        "style": {"mean_abs_vec": np.asarray([4.0, 1.0, 0.5], dtype=np.float32)},
        "logic": {"mean_abs_vec": np.asarray([1.0, 3.0, 0.5], dtype=np.float32)},
        "syntax": {"mean_abs_vec": np.asarray([1.0, 1.0, 2.0], dtype=np.float32)},
    }

    build_specific_top_neurons(dims_out, top_k=0, d_ff=3, dim_names=["style", "logic", "syntax"], preview_limit=8)

    assert dims_out["style"]["specific_selected_count"] == 1
    assert dims_out["style"]["specific_top_neurons"][0]["flat_index"] == 0
    assert dims_out["logic"]["specific_selected_count"] == 1
    assert dims_out["logic"]["specific_top_neurons"][0]["flat_index"] == 1
    assert dims_out["syntax"]["specific_selected_count"] == 1
    assert dims_out["syntax"]["specific_top_neurons"][0]["flat_index"] == 2


def test_resolve_gate_module_and_width_supports_gate_up_proj_layout():
    class FakeLinear:
        def __init__(self, out_features=None, in_features=None):
            self.out_features = out_features
            self.in_features = in_features

    class FakeMlp:
        def __init__(self):
            self.gate_up_proj = FakeLinear(out_features=24)
            self.down_proj = FakeLinear(in_features=12)

    class FakeLayer:
        def __init__(self):
            self.mlp = FakeMlp()

    gate_module, gate_width = resolve_gate_module_and_width(FakeLayer())
    assert gate_module.out_features == 24
    assert gate_width == 12
