from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage99_real_external_data_counterexample_pack import build_real_external_data_counterexample_pack_summary


def test_stage99_real_external_data_counterexample_pack() -> None:
    summary = build_real_external_data_counterexample_pack_summary()
    hm = summary["headline_metrics"]
    inventory = summary["dataset_inventory"]
    status = summary["status"]

    assert inventory["bilingual_entry_count"] >= 40
    assert inventory["english_entry_count"] >= 40
    assert hm["real_sample_coverage"] >= 1.0
    assert hm["real_trigger_rate"] >= 0.50
    assert hm["path_alignment_rate"] >= 0.50
    assert hm["receiver_alignment_rate"] >= 0.50
    assert hm["clause_alignment_rate"] >= 0.50
    assert hm["hardest_real_family_name"] in {
        "bilingual_concrete_transfer",
        "abstract_external_vocab",
        "tech_symbolic_vocab",
        "human_nature_real_mix",
        "celestial_vehicle_real_mix",
        "food_object_real_mix",
        "cross_lingual_alias_real_mix",
        "real_data_adversarial_mixture",
    }
    assert hm["hardest_real_path"] in {
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
    assert hm["hardest_real_intensity"] >= 0.68
    assert hm["weakest_real_receiver"] in {
        "language_plane",
        "brain_plane",
        "intelligence_plane",
        "falsification_plane",
    }
    assert hm["weakest_real_receiver_floor"] <= 0.25
    assert hm["mean_strongest_path_intensity"] >= 0.56
    assert hm["real_external_data_counterexample_score"] >= 0.76
    assert len(summary["sample_records"]) == 8
    assert status["status_short"] in {
        "real_external_data_counterexample_pack_ready",
        "real_external_data_counterexample_pack_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage99_real_external_data_counterexample_pack_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
