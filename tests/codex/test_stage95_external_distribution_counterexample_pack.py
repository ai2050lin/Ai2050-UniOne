from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage95_external_distribution_counterexample_pack import build_external_distribution_counterexample_pack_summary


def test_stage95_external_distribution_counterexample_pack() -> None:
    summary = build_external_distribution_counterexample_pack_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["external_family_coverage"] >= 1.0
    assert hm["external_trigger_rate"] >= 0.60
    assert hm["path_alignment_rate"] >= 0.35
    assert hm["hardest_external_family_name"] in {
        "cross_corpus_domain_shift",
        "sensorimotor_channel_gap",
        "temporal_resolution_drop",
        "cross_lingual_projection_drift",
        "long_horizon_context_scatter",
        "embodied_reference_void",
        "rule_space_reindexing",
        "adversarial_external_mixture",
    }
    assert hm["hardest_external_path"] in {
        "language_plane->brain_plane",
        "language_plane->intelligence_plane",
        "language_plane->falsification_plane",
        "brain_plane->language_plane",
        "brain_plane->intelligence_plane",
        "brain_plane->falsification_plane",
        "intelligence_plane->language_plane",
        "intelligence_plane->brain_plane",
        "intelligence_plane->falsification_plane",
        "falsification_plane->language_plane",
        "falsification_plane->brain_plane",
        "falsification_plane->intelligence_plane",
    }
    assert hm["hardest_external_intensity"] >= 0.62
    assert hm["weakest_external_receiver"] in {
        "language_plane",
        "brain_plane",
        "intelligence_plane",
        "falsification_plane",
    }
    assert hm["weakest_external_receiver_floor"] <= 0.42
    assert hm["mean_strongest_path_intensity"] >= 0.60
    assert hm["external_distribution_counterexample_score"] >= 0.72
    assert len(summary["sample_records"]) == 8
    assert status["status_short"] in {
        "external_distribution_counterexample_pack_ready",
        "external_distribution_counterexample_pack_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage95_external_distribution_counterexample_pack_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
