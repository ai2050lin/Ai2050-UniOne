from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import test_continuous_input_grounding_proto as proto
import test_continuous_multimodal_grounding_proto as cmg


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_apple_banana_encoding_transfer_20260320"
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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _arr(x: np.ndarray) -> list[float]:
    return [float(v) for v in x.tolist()]


def _concept_state(concept: str) -> np.ndarray:
    family = proto.concept_family(concept)
    return np.concatenate(
        [
            proto.family_basis()[family] + proto.concept_offset()[concept],
            cmg.lang_family_basis()[family] + cmg.lang_concept_offset()[concept],
        ],
        axis=0,
    ).astype(np.float32)


def _attribute_axis(attr: str) -> np.ndarray:
    concepts = list(CONCEPT_ATTRIBUTES.keys())
    states = {concept: _concept_state(concept) for concept in concepts}
    family_centers = {}
    for family in proto.FAMILIES:
        family_concepts = [concept for concept in concepts if proto.concept_family(concept) == family]
        family_centers[family] = np.mean(np.stack([states[c] for c in family_concepts], axis=0), axis=0)
    centered = {concept: states[concept] - family_centers[proto.concept_family(concept)] for concept in concepts}

    pos = [centered[concept] for concept in concepts if attr in CONCEPT_ATTRIBUTES[concept]]
    neg = [centered[concept] for concept in concepts if attr not in CONCEPT_ATTRIBUTES[concept]]
    pos_mean = np.mean(np.stack(pos, axis=0), axis=0)
    neg_mean = np.mean(np.stack(neg, axis=0), axis=0)
    return (pos_mean - neg_mean).astype(np.float32)


def build_apple_banana_encoding_transfer_summary() -> dict:
    apple = _load_json(ROOT / "tests" / "codex_temp" / "theory_track_apple_concept_encoding_analysis_20260312.json")
    attrs = _load_json(ROOT / "tests" / "codex_temp" / "theory_track_attribute_axis_analysis_20260312.json")

    decomp = apple["apple_decomposition"]
    apple_full = np.array(
        decomp["apple_visual_state"] + decomp["apple_tactile_state"] + decomp["apple_language_state"],
        dtype=np.float32,
    )
    fruit_family_full = np.concatenate(
        [
            proto.family_basis()["fruit"],
            cmg.lang_family_basis()["fruit"],
        ],
        axis=0,
    ).astype(np.float32)

    banana_true = np.concatenate(
        [
            proto.family_basis()["fruit"] + proto.concept_offset()["banana"],
            cmg.lang_family_basis()["fruit"] + cmg.lang_concept_offset()["banana"],
        ],
        axis=0,
    ).astype(np.float32)
    cat_true = np.concatenate(
        [
            proto.family_basis()["animal"] + proto.concept_offset()["cat"],
            cmg.lang_family_basis()["animal"] + cmg.lang_concept_offset()["cat"],
        ],
        axis=0,
    ).astype(np.float32)

    centered_apple = apple_full - fruit_family_full

    round_axis = _attribute_axis("round")
    elongated_axis = _attribute_axis("elongated")

    round_unit = round_axis / max(float(np.linalg.norm(round_axis)), 1e-12)
    elongated_unit = elongated_axis / max(float(np.linalg.norm(elongated_axis)), 1e-12)

    apple_round_alignment = float(attrs["concept_attribute_alignment"]["apple"]["alignment"]["round"])
    banana_elongated_alignment = float(attrs["concept_attribute_alignment"]["banana"]["alignment"]["elongated"])

    apple_round_component = apple_round_alignment * round_unit
    banana_elongated_component = banana_elongated_alignment * elongated_unit
    shared_residual = centered_apple - apple_round_component

    banana_centered_pred = shared_residual + banana_elongated_component
    banana_pred = fruit_family_full + banana_centered_pred

    banana_language_pred = banana_pred[-8:]
    banana_language_true = banana_true[-8:]

    pred_vs_banana = _cosine(banana_pred, banana_true)
    pred_vs_cat = _cosine(banana_pred, cat_true)
    language_cosine = _cosine(banana_language_pred, banana_language_true)
    l2_error = float(np.linalg.norm(banana_pred - banana_true))

    banana_true_alignment = attrs["concept_attribute_alignment"]["banana"]["alignment"]
    predicted_alignment = {
        "elongated": float(_cosine(banana_centered_pred, elongated_axis)),
        "round": float(_cosine(banana_centered_pred, round_axis)),
    }

    answer = {
        "can_predict_full_embedding": bool(pred_vs_banana > pred_vs_cat and pred_vs_banana > 0.9),
        "can_predict_attribute_fiber": bool(
            predicted_alignment["elongated"] > predicted_alignment["round"]
            and predicted_alignment["elongated"] > 0.5
        ),
        "needs_extra_information": True,
    }

    return {
        "headline_metrics": {
            "pred_vs_banana_cosine": pred_vs_banana,
            "pred_vs_cat_cosine": pred_vs_cat,
            "banana_language_cosine": language_cosine,
            "banana_prediction_l2": l2_error,
            "predicted_elongated_alignment": predicted_alignment["elongated"],
            "predicted_round_alignment": predicted_alignment["round"],
            "true_elongated_alignment": float(banana_true_alignment["elongated"]),
            "true_round_alignment": float(banana_true_alignment.get("round", 0.0)),
        },
        "transfer_equation": {
            "family_term": "z_family(fruit)",
            "known_point": "z_apple = z_family + delta_apple",
            "attribute_swap": "delta_banana_pred = (delta_apple - proj_round(delta_apple)) + target_elongated * u_elongated",
            "banana_prediction": "z_banana_pred = z_family + delta_banana_pred",
        },
        "objects": {
            "banana_pred_state": _arr(banana_pred),
            "banana_true_state": _arr(banana_true),
            "banana_pred_language": _arr(banana_language_pred),
            "banana_true_language": _arr(banana_language_true),
            "shared_residual": _arr(shared_residual),
        },
        "answer": answer,
        "project_readout": {
            "summary": "这一版直接测试：已知苹果表示后，能否通过家族骨架加属性纤维替换，预测香蕉表示。当前结果更支持“可以预测到家族骨架和主属性纤维方向，但不能只靠苹果一个点精确恢复香蕉全部局部偏移”。",
            "next_question": "下一步要把苹果、香蕉、梨三点一起并场，检查多点局部图册是否能把香蕉残差继续压小。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 苹果到香蕉编码迁移报告",
        "",
        f"- pred_vs_banana_cosine: {hm['pred_vs_banana_cosine']:.6f}",
        f"- pred_vs_cat_cosine: {hm['pred_vs_cat_cosine']:.6f}",
        f"- banana_language_cosine: {hm['banana_language_cosine']:.6f}",
        f"- banana_prediction_l2: {hm['banana_prediction_l2']:.6f}",
        f"- predicted_elongated_alignment: {hm['predicted_elongated_alignment']:.6f}",
        f"- predicted_round_alignment: {hm['predicted_round_alignment']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_apple_banana_encoding_transfer_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
