from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_fruit_compatibility_kernel_extractor import (  # noqa: E402
    iter_proper_subsets,
    neuron_support_counts,
    remaining_kernel,
    unique_kernels,
)


def test_remaining_kernel_preserves_union_order() -> None:
    kernel = remaining_kernel([10, 20, 30, 40], [30, 10])
    assert kernel == (20, 40)


def test_unique_kernels_deduplicates_equivalent_rows() -> None:
    rows = [
        {"removed_neurons": [1, 4], "strict_positive_after_prune": True},
        {"removed_neurons": [1, 4], "strict_positive_after_prune": True},
        {"removed_neurons": [2, 3], "strict_positive_after_prune": True},
    ]
    kernels = unique_kernels(rows, [1, 2, 3, 4])
    assert kernels == [(2, 3), (1, 4)]


def test_iter_proper_subsets_covers_all_nonempty_strict_subsets() -> None:
    subsets = list(iter_proper_subsets([7, 8, 9]))
    assert (7,) in subsets
    assert (7, 8) in subsets
    assert (7, 8, 9) not in subsets


def test_neuron_support_counts_sorts_by_frequency() -> None:
    rows = neuron_support_counts([(1, 2), (2, 3), (2, 4)])
    assert rows[0] == {"neuron_index": 2, "support_count": 3}
