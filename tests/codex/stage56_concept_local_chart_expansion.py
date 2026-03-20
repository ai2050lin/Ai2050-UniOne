from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import test_continuous_input_grounding_proto as proto
import test_continuous_multimodal_grounding_proto as cmg


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_concept_local_chart_expansion_20260320"


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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _family_concepts(family: str) -> list[str]:
    return proto.PHASE1[family] + proto.PHASE2[family]


def build_concept_local_chart_expansion_summary() -> dict:
    family_rows: dict[str, dict] = {}
    chart_supports = []
    separation_gaps = []
    anchor_strengths = []

    anchors = {family: _family_anchor(family) for family in proto.FAMILIES}
    all_other = {
        family: np.mean(np.stack([anchors[f] for f in proto.FAMILIES if f != family], axis=0), axis=0)
        for family in proto.FAMILIES
    }

    for family in proto.FAMILIES:
        concepts = _family_concepts(family)
        anchor = anchors[family]
        states = np.stack([_concept_state(concept) for concept in concepts], axis=0)
        deltas = states - anchor
        local_center = np.mean(deltas, axis=0)
        centered = deltas - local_center
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:2]
        recon = centered @ basis.T @ basis + local_center
        recon_errors = np.linalg.norm(deltas - recon, axis=1)

        own_dists = np.linalg.norm(states - anchor, axis=1)
        other_dists = np.linalg.norm(states - all_other[family], axis=1)
        separation_gap = float(np.mean(other_dists - own_dists))
        chart_support = float(np.mean(np.linalg.norm(deltas, axis=1)) - np.mean(recon_errors))
        anchor_strength = float(np.mean([_cosine(state, anchor) for state in states]))

        family_rows[family] = {
            "concepts": concepts,
            "anchor_strength_mean": anchor_strength,
            "chart_compactness": float(np.mean(singular_values[:2])),
            "chart_support": chart_support,
            "reconstruction_error_mean": float(np.mean(recon_errors)),
            "separation_gap": separation_gap,
            "singular_values": singular_values.astype(float).tolist(),
            "basis": basis.astype(float).tolist(),
        }

        chart_supports.append(chart_support)
        separation_gaps.append(separation_gap)
        anchor_strengths.append(anchor_strength)

    return {
        "headline_metrics": {
            "family_count": len(proto.FAMILIES),
            "mean_anchor_strength": float(np.mean(anchor_strengths)),
            "mean_chart_support": float(np.mean(chart_supports)),
            "mean_separation_gap": float(np.mean(separation_gaps)),
            "mean_chart_compactness": float(np.mean([row["chart_compactness"] for row in family_rows.values()])),
            "mean_reconstruction_error": float(np.mean([row["reconstruction_error_mean"] for row in family_rows.values()])),
        },
        "family_charts": family_rows,
        "project_readout": {
            "summary": (
                "这一轮把局部图册从水果扩到水果、动物、抽象三个家族。"
                "当前结果支持：概念形成不是单个苹果案例，而是多家族都能被写成家族骨架加局部图册。"
            ),
            "next_question": (
                "下一步要继续把属性纤维并回多家族局部图册，"
                "检查家族骨架、局部偏移和属性纤维是否能形成统一概念核。"
            ),
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 多家族局部图册扩展报告",
        "",
        f"- family_count: {hm['family_count']}",
        f"- mean_anchor_strength: {hm['mean_anchor_strength']:.6f}",
        f"- mean_chart_support: {hm['mean_chart_support']:.6f}",
        f"- mean_separation_gap: {hm['mean_separation_gap']:.6f}",
        f"- mean_chart_compactness: {hm['mean_chart_compactness']:.6f}",
        f"- mean_reconstruction_error: {hm['mean_reconstruction_error']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_concept_local_chart_expansion_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
