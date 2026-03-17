from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from deepseek7b_stage4_minimal_circuit_search import (  # noqa: E402
    common_index_set,
    index_frequency,
    parse_subset_sizes,
    pick_subset_sizes,
    rank_neurons_by_baseline,
)


def test_parse_subset_sizes_dedupes_and_filters_nonpositive():
    assert parse_subset_sizes("48,32,32,0,-1,8") == [48, 32, 8]


def test_pick_subset_sizes_caps_to_total_minus_one():
    assert pick_subset_sizes(12, [48, 10, 6, 6, 3]) == [11, 10, 6, 3]


def test_rank_neurons_by_baseline_prefers_signature_members_then_later_layers():
    signature = {
        "signature_top_indices": [100, 300, 5],
        "signature_top_values": [0.9, 0.8, 0.7],
    }
    ranked = rank_neurons_by_baseline([5, 100, 250, 300], signature, d_ff=100)
    assert ranked == [100, 300, 5, 250]


def test_rank_neurons_by_baseline_pushes_global_common_indices_to_the_back():
    signature = {
        "signature_top_indices": [100, 300, 5],
        "signature_top_values": [0.9, 0.8, 0.7],
    }
    ranked = rank_neurons_by_baseline([5, 100, 250, 300], signature, d_ff=100, common_indices={100, 300})
    assert ranked == [5, 250, 100, 300]


def test_index_frequency_and_common_index_set_detect_repeated_indices():
    rows = [
        {"intervention": {"flat_indices": [1, 2, 3]}},
        {"intervention": {"flat_indices": [2, 3, 4]}},
        {"intervention": {"flat_indices": [2, 5]}},
    ]
    counts = index_frequency(rows)
    assert counts[2] == 3
    assert counts[3] == 2
    assert common_index_set(counts, total_rows=3, max_fraction=0.66) == {2, 3}
