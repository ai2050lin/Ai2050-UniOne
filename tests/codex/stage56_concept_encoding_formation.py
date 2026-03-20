from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import test_continuous_input_grounding_proto as proto
import test_continuous_multimodal_grounding_proto as cmg


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_concept_encoding_formation_20260320"

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


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _concept_state(concept: str) -> np.ndarray:
    family = proto.concept_family(concept)
    return np.concatenate(
        [
            proto.family_basis()[family] + proto.concept_offset()[concept],
            cmg.lang_family_basis()[family] + cmg.lang_concept_offset()[concept],
        ],
        axis=0,
    ).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _attribute_axis(attr: str) -> np.ndarray:
    concepts = list(CONCEPT_ATTRIBUTES.keys())
    states = {concept: _concept_state(concept) for concept in concepts}
    family_centers = {}
    for family in proto.FAMILIES:
        family_concepts = [concept for concept in concepts if proto.concept_family(concept) == family]
        family_centers[family] = np.mean(np.stack([states[c] for c in family_concepts], axis=0), axis=0)
    centered = {concept: states[concept] - family_centers[proto.concept_family(concept)] for concept in concepts}

    pos = [centered[c] for c in concepts if attr in CONCEPT_ATTRIBUTES[c]]
    neg = [centered[c] for c in concepts if attr not in CONCEPT_ATTRIBUTES[c]]
    pos_mean = np.mean(np.stack(pos, axis=0), axis=0)
    neg_mean = np.mean(np.stack(neg, axis=0), axis=0)
    return (pos_mean - neg_mean).astype(np.float32)


def _projection_coeff(x: np.ndarray, axis: np.ndarray) -> float:
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        return 0.0
    return float(np.dot(x, axis / norm))


def build_concept_encoding_formation_summary() -> dict:
    refined = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_native_refinement_20260320" / "summary.json"
    )
    circuit = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_level_bridge_20260320" / "summary.json"
    )
    closed_v3 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v3_20260320" / "summary.json"
    )
    transfer = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_apple_banana_encoding_transfer_20260320" / "summary.json"
    )

    apple = _concept_state("apple")
    banana = _concept_state("banana")
    pear = _concept_state("pear")
    fruit_family = np.concatenate(
        [
            proto.family_basis()["fruit"],
            cmg.lang_family_basis()["fruit"],
        ],
        axis=0,
    ).astype(np.float32)

    apple_delta = apple - fruit_family
    banana_delta = banana - fruit_family
    pear_delta = pear - fruit_family

    fruit_local_matrix = np.stack([apple_delta, banana_delta, pear_delta], axis=0)
    local_center = np.mean(fruit_local_matrix, axis=0)
    centered = fruit_local_matrix - local_center
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    chart_basis = vh[:2]
    recon = centered @ chart_basis.T @ chart_basis + local_center
    recon_errors = np.linalg.norm(fruit_local_matrix - recon, axis=1)

    attribute_names = ["round", "elongated", "sweet", "edible", "concrete"]
    apple_attr = {}
    for attr in attribute_names:
        axis = _attribute_axis(attr)
        apple_attr[attr] = {
            "alignment": _cosine(apple_delta, axis),
            "coefficient": _projection_coeff(apple_delta, axis),
        }

    sorted_attrs = sorted(
        apple_attr.items(),
        key=lambda kv: abs(kv[1]["alignment"]),
        reverse=True,
    )

    rhm = refined["headline_metrics"]
    chm = circuit["headline_metrics"]
    vhm = closed_v3["headline_metrics"]
    thm = transfer["headline_metrics"]

    concept_seed = rhm["seed_refined"] * max(chm["excitatory_seed"], 1e-12)
    concept_binding = rhm["bind_refined"] * max(chm["synchrony_binding"], 1e-12)
    concept_embedding = rhm["embed_refined"] * max(vhm["structure_growth_v3"], 1e-12)
    concept_pressure = rhm["pressure_refined"] * max(vhm["cross_asset_pressure_v3"], 1e-12)
    concept_margin = concept_seed + concept_binding + concept_embedding - concept_pressure

    family_anchor_strength = _cosine(apple, fruit_family)
    apple_banana_transfer_support = thm["pred_vs_banana_cosine"]
    fruit_chart_compactness = float(np.mean(singular_values[:2]))

    return {
        "headline_metrics": {
            "family_anchor_strength": family_anchor_strength,
            "apple_local_offset_norm": float(np.linalg.norm(apple_delta)),
            "fruit_chart_compactness": fruit_chart_compactness,
            "concept_seed_drive": concept_seed,
            "concept_binding_drive": concept_binding,
            "concept_embedding_drive": concept_embedding,
            "concept_pressure": concept_pressure,
            "concept_encoding_margin": concept_margin,
            "apple_banana_transfer_support": apple_banana_transfer_support,
            "fruit_chart_reconstruction_error_mean": float(np.mean(recon_errors)),
        },
        "apple_attribute_fibers": {
            attr: {
                "alignment": values["alignment"],
                "coefficient": values["coefficient"],
            }
            for attr, values in apple_attr.items()
        },
        "apple_top_fibers": [
            {
                "attribute": attr,
                "alignment": values["alignment"],
                "coefficient": values["coefficient"],
            }
            for attr, values in sorted_attrs
        ],
        "fruit_chart": {
            "basis": chart_basis.astype(float).tolist(),
            "singular_values": singular_values.astype(float).tolist(),
            "reconstruction_errors": recon_errors.astype(float).tolist(),
        },
        "formation_equation": {
            "concept_kernel": "K_concept = S_seed + B_bind + E_embed - P_pressure",
            "apple_kernel": "K_apple = family_anchor + local_offset + attribute_fibers - structural_pressure",
            "chart_equation": "z_concept = z_family + delta_local + sum_i alpha_i * fiber_i",
            "growth_chain": "局部刺激 -> 编码种子 -> 回路绑定 -> 家族图册嵌入 -> 属性纤维定向 -> 稳态固化",
        },
        "project_readout": {
            "summary": (
                "这一轮把“苹果这样的概念是怎么被编码出来的”压成了一条更具体的形成链："
                "先有水果家族骨架，再有苹果的局部偏移，再由圆形、甜、可食用等属性纤维定向，"
                "最后在结构压力下形成稳定编码。"
            ),
            "next_question": (
                "下一步最关键的是把苹果、香蕉、梨三点局部图册继续做强，"
                "看概念形成链能不能跨更多家族和更多属性纤维稳定成立。"
            ),
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    top_fibers = summary["apple_top_fibers"][:3]
    lines = [
        "# 概念编码形成报告",
        "",
        f"- family_anchor_strength: {hm['family_anchor_strength']:.6f}",
        f"- apple_local_offset_norm: {hm['apple_local_offset_norm']:.6f}",
        f"- fruit_chart_compactness: {hm['fruit_chart_compactness']:.6f}",
        f"- concept_seed_drive: {hm['concept_seed_drive']:.6f}",
        f"- concept_binding_drive: {hm['concept_binding_drive']:.6f}",
        f"- concept_embedding_drive: {hm['concept_embedding_drive']:.6f}",
        f"- concept_pressure: {hm['concept_pressure']:.6f}",
        f"- concept_encoding_margin: {hm['concept_encoding_margin']:.6f}",
        f"- apple_banana_transfer_support: {hm['apple_banana_transfer_support']:.6f}",
        "",
        "## 苹果主属性纤维",
    ]
    for fiber in top_fibers:
        lines.append(
            f"- {fiber['attribute']}: alignment={fiber['alignment']:.6f}, coefficient={fiber['coefficient']:.6f}"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_concept_encoding_formation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
