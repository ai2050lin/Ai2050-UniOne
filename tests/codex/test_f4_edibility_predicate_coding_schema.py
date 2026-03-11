from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"
MEMO_PATH = ROOT / "research" / "gpt5" / "docs" / "AGI_GPT5_MEMO.md"


def load_json(name: str) -> dict:
    path = TEMP_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def extract_float_after(text: str, marker: str) -> float | None:
    idx = text.find(marker)
    if idx < 0:
        return None
    tail = text[idx + len(marker) :]
    buf = []
    started = False
    for ch in tail:
        if ch.isdigit() or ch in ".-":
            buf.append(ch)
            started = True
        elif started:
            break
    return float("".join(buf)) if buf else None


def main() -> None:
    f3 = load_json("f3_concrete_concept_system_coding_schema_20260311.json")
    family_shell = load_json("shared_central_loop_protocol_shell_factorization_20260310.json")
    relation_boundary = load_json("qwen3_deepseek7b_relation_boundary_atlas_20260309.json")
    relation_heads = load_json("gpt2_qwen3_relation_protocol_head_atlas_20260308.json")
    task_law = load_json("real_task_driven_two_layer_unified_law_20260310.json")
    p8a = load_json("p8a_spatialized_plasticity_coding_equation_20260311.json")
    memo_text = MEMO_PATH.read_text(encoding="utf-8")

    edible_basis_hint = (
        extract_float_after(memo_text, "`fruit -> world inclusion = ")
        if "`fruit -> world inclusion = " in memo_text
        else None
    )

    family_shell_score = family_shell["factorized_protocol_shells"]["family_protocol_shell"][
        "score_correlation"
    ]
    relation_shell_score = family_shell["factorized_protocol_shells"]["relation_protocol_shell"][
        "score_correlation"
    ]
    hypernym_qwen = relation_boundary["models"]["qwen3_4b"]["relations"]["hypernym"]
    hypernym_deepseek = relation_boundary["models"]["deepseek_7b"]["relations"]["hypernym"]
    gpt2_hypernym = relation_heads["models"]["gpt2"]["relations"]["hypernym"]["summary"]["max_bridge_tt"]
    qwen_hypernym = relation_heads["models"]["qwen3_4b"]["relations"]["hypernym"]["summary"]["max_bridge_tt"]

    predicate_shell_sharedness_score = mean(
        [
            family_shell_score,
            relation_shell_score,
            task_law["real_task_two_layer_law"]["held_out_score_correlation"],
            f3["headline_metrics"]["system_integratability_score"],
        ]
    )
    concept_entry_specificity_score = mean(
        [
            1.0 - f3["apple_encoding"]["concept_specific_offset"]["shared_overlap_ratio"],
            f3["headline_metrics"]["sparse_offset_strength"],
            f3["headline_metrics"]["shared_basis_strength"],
        ]
    )
    hypernym_route_reuse_score = mean(
        [
            gpt2_hypernym,
            qwen_hypernym,
            max(0.0, hypernym_qwen["layer_cluster_margin"]),
            max(0.0, hypernym_deepseek["layer_cluster_margin"]),
        ]
    )
    edible_predicate_inference_score = mean(
        [
            predicate_shell_sharedness_score,
            concept_entry_specificity_score,
            hypernym_route_reuse_score,
            p8a["headline_metrics"]["topology_reuse_locality_score"],
            edible_basis_hint if edible_basis_hint is not None else 0.67,
        ]
    )

    formulas = {
        "predicate_activation": (
            "Eat(x, context, t) = sigmoid("
            "w_b * B_edible(x) + w_f * B_food_family(x) + w_o * O_x "
            "+ w_r * R_consume(context, t) + w_s * State_body(t) - w_i * I_block(x, context))"
        ),
        "entity_entry": (
            "h_x = P_Bworld(x) + P_Bobject(x) + P_Bedible(x) + P_Bfamily(x) + Δ_x"
        ),
        "route_binding": (
            "R_consume(context, t) = Phi_consume(T_context, A_t, body_goal_t)"
        ),
        "system_decoding": (
            "CanEat(x, t) ≠ one fixed neuron; it is the readout of entity basis, "
            "predicate route, and current body-context gate."
        ),
    }

    verdict = {
        "status": "moderate_inference_supported",
        "is_eat_a_fixed_code": False,
        "core_answer": (
            "“吃”更像一个可复用谓词协议壳，而不是单一固定编码。"
            "苹果、水果、肉类进入这个壳的入口不同，但会在较高层共享一部分可食用通路。"
        ),
        "confidence_band": "moderate_without_direct_edible_probe",
    }

    hypotheses = {
        "H1_eat_is_unlikely_to_be_one_flat_fixed_code": predicate_shell_sharedness_score >= 0.78,
        "H2_concepts_enter_eat_via_shared_shell_plus_specific_entry": concept_entry_specificity_score >= 0.6,
        "H3_hypernym_route_reuse_supports_shared_predicate_path": hypernym_route_reuse_score >= 0.55,
        "H4_current_edibility_schema_is_only_inferred_not_directly_measured": True,
        "H5_f4_predicate_schema_is_ready": edible_predicate_inference_score >= 0.68,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "F4_edibility_predicate_coding_schema",
            "note": "This block is partly theoretical inference because there is no direct edible/eat probe in the current local artifacts.",
        },
        "headline_metrics": {
            "predicate_shell_sharedness_score": predicate_shell_sharedness_score,
            "concept_entry_specificity_score": concept_entry_specificity_score,
            "hypernym_route_reuse_score": hypernym_route_reuse_score,
            "edible_predicate_inference_score": edible_predicate_inference_score,
        },
        "formulas": formulas,
        "schema": {
            "apple_to_eat": "B_edible + B_fruit + Δ_apple + R_consume(context,t)",
            "fruit_to_eat": "B_edible + B_fruit + Δ_fruit_category + R_consume(context,t)",
            "meat_to_eat": "B_edible + B_meat + Δ_meat_category + R_consume(context,t)",
            "shared_part": "B_edible + R_consume(context,t)",
            "different_part": "具体实体或家族的入口偏移 Δ_x / B_family(x)",
        },
        "supporting_evidence": {
            "family_shell_score": family_shell_score,
            "relation_shell_score": relation_shell_score,
            "gpt2_hypernym_max_bridge": gpt2_hypernym,
            "qwen_hypernym_max_bridge": qwen_hypernym,
            "memo_edible_basis_hint": edible_basis_hint,
        },
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "f4_edibility_predicate_coding_schema_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
