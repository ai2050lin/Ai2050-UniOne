from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import test_continuous_input_grounding_proto as proto
import test_continuous_multimodal_grounding_proto as cmg


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_attribute_fiber_nativeization_20260320"

CONCEPT_ATTRIBUTES = {
    "apple": ["fruit", "edible", "round", "sweet", "concrete"],
    "banana": ["fruit", "edible", "sweet", "elongated", "concrete"],
    "pear": ["fruit", "edible", "sweet", "round", "concrete"],
    "cat": ["animal", "living", "mobile", "concrete", "domestic"],
    "dog": ["animal", "living", "mobile", "concrete", "domestic"],
    "horse": ["animal", "living", "mobile", "concrete", "large"],
    "truth": ["abstract", "cognitive", "symbolic", "stable"],
    "logic": ["abstract", "cognitive", "symbolic", "structured"],
    "memory": ["abstract", "cognitive", "symbolic", "persistent"],
}


def _concept_state(concept: str) -> np.ndarray:
    family = proto.concept_family(concept)
    return np.concatenate(
        [
            proto.family_basis()[family] + proto.concept_offset()[concept],
            cmg.lang_family_basis()[family] + cmg.lang_concept_offset()[concept],
        ],
        axis=0,
    ).astype(np.float32)


def _family_anchor(family: str) -> np.ndarray:
    return np.concatenate(
        [proto.family_basis()[family], cmg.lang_family_basis()[family]],
        axis=0,
    ).astype(np.float32)


def _family_concepts(family: str) -> list[str]:
    return proto.PHASE1[family] + proto.PHASE2[family]


def _normalize(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm == 0.0:
        return x.astype(np.float32)
    return (x / norm).astype(np.float32)


def build_attribute_fiber_nativeization_summary() -> dict:
    family_rows = {}
    apple_report = {}
    anchor_bundle_strengths = []
    local_bundle_strengths = []

    for family in proto.FAMILIES:
        concepts = _family_concepts(family)
        anchor = _family_anchor(family)
        states = {concept: _concept_state(concept) for concept in concepts}
        deltas = {concept: states[concept] - anchor for concept in concepts}

        all_attrs = sorted({attr for concept in concepts for attr in CONCEPT_ATTRIBUTES[concept]})
        anchor_attrs = []
        local_attrs = []
        local_fibers = {}

        for attr in all_attrs:
            pos_concepts = [c for c in concepts if attr in CONCEPT_ATTRIBUTES[c]]
            prevalence = len(pos_concepts)
            if prevalence == len(concepts):
                anchor_attrs.append(attr)
                continue
            if prevalence == 0:
                continue

            neg_concepts = [c for c in concepts if attr not in CONCEPT_ATTRIBUTES[c]]
            pos_mean = np.mean(np.stack([deltas[c] for c in pos_concepts], axis=0), axis=0)
            neg_mean = np.mean(np.stack([deltas[c] for c in neg_concepts], axis=0), axis=0)
            axis = _normalize(pos_mean - neg_mean)
            local_attrs.append(attr)
            local_fibers[attr] = {
                concept: float(np.dot(deltas[concept], axis))
                for concept in concepts
            }

        anchor_bundle_strength = float(len(anchor_attrs) / max(1, len(all_attrs)))
        local_bundle_strength = float(len(local_attrs) / max(1, len(all_attrs)))

        family_rows[family] = {
            "concepts": concepts,
            "anchor_attributes": anchor_attrs,
            "local_attributes": local_attrs,
            "anchor_bundle_strength": anchor_bundle_strength,
            "local_bundle_strength": local_bundle_strength,
            "local_fibers": local_fibers,
        }

        anchor_bundle_strengths.append(anchor_bundle_strength)
        local_bundle_strengths.append(local_bundle_strength)

        if family == "fruit":
            apple_report = {
                "anchor_attributes": anchor_attrs,
                "local_attributes": local_attrs,
                "local_fiber_coefficients": {
                    attr: local_fibers[attr]["apple"]
                    for attr in local_attrs
                },
            }

    return {
        "headline_metrics": {
            "mean_anchor_bundle_strength": float(np.mean(anchor_bundle_strengths)),
            "mean_local_bundle_strength": float(np.mean(local_bundle_strengths)),
            "apple_anchor_attribute_count": len(apple_report.get("anchor_attributes", [])),
            "apple_local_attribute_count": len(apple_report.get("local_attributes", [])),
            "apple_round_local_coeff": float(apple_report.get("local_fiber_coefficients", {}).get("round", 0.0)),
            "apple_elongated_local_coeff": float(
                apple_report.get("local_fiber_coefficients", {}).get("elongated", 0.0)
            ),
        },
        "family_attribute_systems": family_rows,
        "apple_nativeized_fibers": apple_report,
        "project_readout": {
            "summary": (
                "这一轮把属性纤维拆成两层：家族共享属性进入家族锚点，"
                "家族内部变化属性进入局部纤维。这样苹果的甜、可食用、具体性不再和圆形/细长混在一起。"
            ),
            "next_question": (
                "下一步要把家族锚点属性和局部纤维一起并回概念形成核，"
                "检验多家族口径下概念形成方程能否进一步收口。"
            ),
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 属性纤维原生化报告",
        "",
        f"- mean_anchor_bundle_strength: {hm['mean_anchor_bundle_strength']:.6f}",
        f"- mean_local_bundle_strength: {hm['mean_local_bundle_strength']:.6f}",
        f"- apple_anchor_attribute_count: {hm['apple_anchor_attribute_count']}",
        f"- apple_local_attribute_count: {hm['apple_local_attribute_count']}",
        f"- apple_round_local_coeff: {hm['apple_round_local_coeff']:.6f}",
        f"- apple_elongated_local_coeff: {hm['apple_elongated_local_coeff']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_attribute_fiber_nativeization_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
