from __future__ import annotations

import sys
from pathlib import Path


CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_feature_layer_nativeization import build_feature_layer_nativeization_summary


def test_feature_layer_nativeization_is_positive() -> None:
    summary = build_feature_layer_nativeization_summary()
    hm = summary["headline_metrics"]
    assert hm["native_basis_v2"] > 0.0
    assert hm["native_separation_v2"] > 1.0
    assert hm["native_lock_v2"] > 0.0
    assert hm["feature_native_core_v2"] > hm["native_lock_v2"]
