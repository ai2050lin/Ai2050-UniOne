from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_wordclass_causal_probe import build_abstract_probe, build_adjective_probe, build_adverb_probe, build_verb_probe  # noqa: E402


def test_adjective_probe_detects_modifier_fiber() -> None:
    probe = build_adjective_probe(
        {"concepts": {"apple": {"role_subsets": {"entity": {"drop_target": 1e-8}, "size": {"drop_target": 3e-5}, "weight": {"drop_target": 2e-5}}}}},
        {"causal_ablation": {"apple_attr": {"delta": -2e-4}}},
    )
    assert probe["judgement"] == "adjective_as_fiber"


def test_verb_probe_detects_transport() -> None:
    probe = build_verb_probe(
        [{"category": "action", "strict_positive_pair_ratio": 0.6, "mean_union_joint_adv": 0.05, "mean_union_synergy_joint": 0.01}],
        {"per_category": {"action": {"mean_protocol_role_pressure": 0.4, "encoding_class": "anchor_like"}}},
    )
    assert probe["judgement"] == "verb_transport"


def test_abstract_probe_detects_bundle() -> None:
    probe = build_abstract_probe(
        [
            {"category": "abstract", "top_instance_term": "meaning", "strict_positive_pair_ratio": 0.0, "mean_union_synergy_joint": -0.08},
            {"category": "abstract", "top_instance_term": "glory", "strict_positive_pair_ratio": 0.5, "mean_union_synergy_joint": 0.01},
        ],
        {"per_category": {"human": {"mean_protocol_role_pressure": 1.0}}},
    )
    assert probe["judgement"] == "abstract_bundle"


def test_adverb_probe_detects_control_modifier() -> None:
    probe = build_adverb_probe(
        {"quickly": {"margin": 0.001, "spread": 0.01, "top_category": "tech"}, "slowly": {"margin": 0.002, "spread": 0.02, "top_category": "abstract"}},
        {"dimensions": {"style": {"mean_pair_delta_l2": 1000.0}}},
    )
    assert probe["judgement"] == "adverb_control_modifier"
