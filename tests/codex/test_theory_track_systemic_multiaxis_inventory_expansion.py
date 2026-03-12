from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import test_continuous_input_grounding_proto as proto
import test_continuous_multimodal_grounding_proto as cmg


ROOT = Path(__file__).resolve().parents[2]


def base_state(family: str) -> np.ndarray:
    return np.concatenate([proto.family_basis()[family], cmg.lang_family_basis()[family]], axis=0).astype(np.float32)


def sparse_templates(specs: dict[str, list[int]], dim: int, weight: float) -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for name, dims in specs.items():
        vec = np.zeros(dim, dtype=np.float32)
        for idx in dims:
            vec[idx] = weight
        templates[name] = vec
    return templates


def build_family_axes() -> dict[str, list[np.ndarray]]:
    axes: dict[str, list[np.ndarray]] = {}
    for family in proto.FAMILIES:
        cat = base_state(family)
        top = np.argsort(np.abs(cat))[-20:][::-1].tolist()
        family_axes: list[np.ndarray] = []
        for idx in top[:14]:
            axis = np.zeros_like(cat)
            axis[idx] = 1.0
            if idx + 1 < axis.shape[0]:
                axis[idx + 1] = 0.16
            family_axes.append(axis.astype(np.float32))
        axes[family] = family_axes
    return axes


def context_templates(dim: int = 24) -> dict[str, np.ndarray]:
    specs = {
        "observe": [0, 8, 16],
        "compare": [1, 9, 17],
        "use": [2, 10, 18],
        "remember": [3, 11, 19],
        "explain": [4, 12, 20],
        "infer": [5, 13, 21],
        "predict": [6, 14, 22],
        "plan": [7, 15, 23],
        "justify": [0, 9, 18],
        "abstract": [3, 12, 21],
        "verify": [5, 14, 23],
        "commit": [6, 15, 22],
        "compare_tool": [2, 11, 20],
        "reflect": [1, 10, 19],
    }
    return sparse_templates(specs, dim, 0.075)


def relation_templates(dim: int = 24) -> dict[str, np.ndarray]:
    specs = {
        "same_family": [0, 1, 2],
        "contrast_family": [3, 4, 5],
        "part_of_chain": [6, 7, 8],
        "causal_predecessor": [9, 10, 11],
        "causal_successor": [12, 13, 14],
        "depends_on": [15, 16, 17],
        "supports": [18, 19, 20],
        "competes_with": [21, 22, 23],
        "explains": [0, 10, 20],
        "generalizes": [3, 13, 23],
        "refines": [2, 12, 22],
        "verifies": [5, 15, 19],
        "tool_calls": [4, 11, 18],
        "protocol_fields": [6, 14, 20],
    }
    return sparse_templates(specs, dim, 0.18)


def temporal_templates(dim: int = 24) -> dict[str, np.ndarray]:
    specs = {
        "t0_entry": [0, 1, 8],
        "t1_bind": [2, 3, 10],
        "t2_contextualize": [4, 5, 12],
        "t3_compare": [6, 7, 14],
        "t4_causal_link": [9, 11, 15],
        "t5_reason": [13, 17, 19],
        "t6_integrate": [18, 20, 21],
        "t7_readout": [16, 22, 23],
        "t8_verify": [1, 10, 19],
        "t9_commit": [4, 14, 22],
        "t10_reflect": [3, 12, 20],
        "t11_finalize": [7, 15, 23],
        "t12_tool_bridge": [5, 11, 21],
        "t13_protocol_commit": [6, 12, 22],
    }
    return sparse_templates(specs, dim, 0.094)


def protocol_templates(dim: int = 24) -> dict[str, np.ndarray]:
    specs = {
        "field_lookup": [0, 6, 12],
        "field_route": [1, 7, 13],
        "task_bind": [2, 8, 14],
        "tool_schema": [3, 9, 15],
        "tool_dispatch": [4, 10, 16],
        "result_bind": [5, 11, 17],
        "rollback": [6, 12, 18],
        "recover": [7, 13, 19],
        "align": [8, 14, 20],
        "commit": [9, 15, 21],
        "bridge": [10, 16, 22],
        "finalize": [11, 17, 23],
    }
    return sparse_templates(specs, dim, 0.11)


def build_chain_anchors(
    rng: np.random.Generator,
    families: list[str],
    chains_per_family: int,
    dim: int,
) -> dict[tuple[str, int], np.ndarray]:
    anchors: dict[tuple[str, int], np.ndarray] = {}
    recurrent_pool = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]
    for family in families:
        for chain_id in range(chains_per_family):
            vec = np.zeros(dim, dtype=np.float32)
            dims = rng.choice(recurrent_pool, size=6, replace=False)
            for idx, weight in zip(dims, rng.uniform(0.020, 0.038, size=6).astype(np.float32)):
                vec[int(idx)] = float(weight)
            anchors[(family, chain_id)] = vec
    return anchors


def build_successor_latents(
    rng: np.random.Generator,
    families: list[str],
    chains_per_family: int,
    temporal_names: list[str],
    dim: int,
) -> dict[tuple[str, int, str], np.ndarray]:
    latents: dict[tuple[str, int, str], np.ndarray] = {}
    recurrent_pool = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]
    primary = [0, 2, 4, 6, 9, 13, 18, 16, 10, 14, 5, 21, 11, 22]
    for family in families:
        for chain_id in range(chains_per_family):
            running = np.zeros(dim, dtype=np.float32)
            dims = rng.choice(recurrent_pool, size=5, replace=False)
            signature = np.zeros(dim, dtype=np.float32)
            for idx, weight in zip(dims, rng.uniform(0.026, 0.041, size=5).astype(np.float32)):
                signature[int(idx)] = float(weight)
            for stage_idx, stage_name in enumerate(temporal_names):
                drift = np.zeros(dim, dtype=np.float32)
                drift[primary[stage_idx % len(primary)]] += 0.024
                drift += 0.42 * signature
                if stage_idx > 0:
                    drift[primary[(stage_idx - 1) % len(primary)]] += 0.013
                if stage_idx + 1 < len(temporal_names):
                    drift[primary[(stage_idx + 1) % len(primary)]] += 0.009
                running = 0.935 * running + drift
                latents[(family, chain_id, stage_name)] = running.copy()
    return latents


def mean_pairwise(states: list[np.ndarray], ref: np.ndarray, limit: int) -> float:
    if not states:
        return 0.0
    sample = states[: min(limit, len(states))]
    return float(np.mean([np.linalg.norm(ref - other) for other in sample]))


def stage_index(name: str) -> int:
    return int(name[1:].split("_")[0]) if "_" in name else int(name[1:])


def synthesize_entry(
    rng: np.random.Generator,
    family: str,
    concept_idx: int,
    chain_id: int,
    context_name: str,
    relation_name: str,
    temporal_name: str,
    protocol_name: str,
    family_axes: dict[str, list[np.ndarray]],
    contexts: dict[str, np.ndarray],
    relations: dict[str, np.ndarray],
    temporals: dict[str, np.ndarray],
    protocols: dict[str, np.ndarray],
    chain_anchors: dict[tuple[str, int], np.ndarray],
    successor_latents: dict[tuple[str, int, str], np.ndarray],
) -> dict[str, object]:
    cat = base_state(family)
    axes = family_axes[family]
    chosen_axes = rng.choice(len(axes), size=5, replace=True)
    axis_weights = rng.uniform(0.01, 0.04, size=5).astype(np.float32)
    offset = np.zeros_like(cat)
    for axis_idx, weight in zip(chosen_axes, axis_weights):
        offset += weight * axes[int(axis_idx)]

    recurrent_dims = rng.choice([0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15], size=5, replace=False)
    for dim, weight in zip(recurrent_dims, rng.uniform(0.004, 0.010, size=5).astype(np.float32)):
        offset[int(dim)] += float(weight)

    noise = rng.normal(scale=0.0012, size=cat.shape[0]).astype(np.float32)
    state = (
        cat
        + offset
        + chain_anchors[(family, chain_id)]
        + contexts[context_name]
        + relations[relation_name]
        + temporals[temporal_name]
        + protocols[protocol_name]
        + successor_latents[(family, chain_id, temporal_name)]
        + noise
    ).astype(np.float32)
    return {
        "concept": f"{family}_mx_{concept_idx:04d}",
        "family": family,
        "context": context_name,
        "relation": relation_name,
        "temporal": temporal_name,
        "protocol": protocol_name,
        "chain_id": int(chain_id),
        "state": state,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory-track systemic multiaxis inventory expansion")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concepts-per-family", type=int, default=240)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_systemic_multiaxis_inventory_expansion_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    family_axes = build_family_axes()
    contexts = context_templates()
    relations = relation_templates()
    temporals = temporal_templates()
    protocols = protocol_templates()
    temporal_names = list(temporals.keys())
    dim = base_state(proto.FAMILIES[0]).shape[0]
    chains_per_family = (int(args.concepts_per_family) + len(temporal_names) - 1) // len(temporal_names)
    chain_anchors = build_chain_anchors(rng, list(proto.FAMILIES), chains_per_family, dim)
    successor_latents = build_successor_latents(rng, list(proto.FAMILIES), chains_per_family, temporal_names, dim)

    entries: list[dict[str, object]] = []
    by_family: dict[str, list[dict[str, object]]] = {family: [] for family in proto.FAMILIES}
    by_context: dict[str, list[np.ndarray]] = {name: [] for name in contexts}
    by_relation: dict[str, list[np.ndarray]] = {name: [] for name in relations}
    by_temporal: dict[str, list[np.ndarray]] = {name: [] for name in temporals}
    by_protocol: dict[str, list[np.ndarray]] = {name: [] for name in protocols}
    by_chain: dict[tuple[str, int], list[dict[str, object]]] = {}

    context_names = list(contexts.keys())
    relation_names = list(relations.keys())
    protocol_names = list(protocols.keys())
    for family in proto.FAMILIES:
        for concept_idx in range(int(args.concepts_per_family)):
            context_name = context_names[(concept_idx + len(family)) % len(context_names)]
            relation_name = relation_names[(concept_idx * 2 + len(family)) % len(relation_names)]
            temporal_name = temporal_names[concept_idx % len(temporal_names)]
            protocol_name = protocol_names[(concept_idx * 3 + len(family)) % len(protocol_names)]
            chain_id = concept_idx // len(temporal_names)
            entry = synthesize_entry(
                rng,
                family,
                concept_idx,
                chain_id,
                context_name,
                relation_name,
                temporal_name,
                protocol_name,
                family_axes,
                contexts,
                relations,
                temporals,
                protocols,
                chain_anchors,
                successor_latents,
            )
            entries.append(entry)
            by_family[family].append(entry)
            by_context[context_name].append(entry["state"])
            by_relation[relation_name].append(entry["state"])
            by_temporal[temporal_name].append(entry["state"])
            by_protocol[protocol_name].append(entry["state"])
            by_chain.setdefault((family, chain_id), []).append(entry)

    within_family = []
    cross_family = []
    same_context = []
    cross_context = []
    same_relation = []
    cross_relation = []
    same_temporal = []
    cross_temporal = []
    same_protocol = []
    cross_protocol = []
    same_chain_successor = []
    cross_chain_same_stage = []

    for idx, entry in enumerate(entries):
        state = entry["state"]
        family = entry["family"]
        family_states = [e["state"] for e in by_family[family] if e is not entry]
        foreign_states = [e["state"] for e in entries if e["family"] != family]
        within_family.append(mean_pairwise(family_states, state, 16))
        cross_family.append(mean_pairwise(foreign_states, state, 64))

        same_ctx = [e["state"] for j, e in enumerate(entries) if j != idx and e["context"] == entry["context"]]
        cross_ctx = [e["state"] for e in entries if e["context"] != entry["context"]]
        same_context.append(mean_pairwise(same_ctx, state, 16))
        cross_context.append(mean_pairwise(cross_ctx, state, 64))

        same_rel = [e["state"] for j, e in enumerate(entries) if j != idx and e["relation"] == entry["relation"]]
        cross_rel = [e["state"] for e in entries if e["relation"] != entry["relation"]]
        same_relation.append(mean_pairwise(same_rel, state, 16))
        cross_relation.append(mean_pairwise(cross_rel, state, 64))

        same_tmp = [e["state"] for j, e in enumerate(entries) if j != idx and e["temporal"] == entry["temporal"]]
        cross_tmp = [e["state"] for e in entries if e["temporal"] != entry["temporal"]]
        same_temporal.append(mean_pairwise(same_tmp, state, 16))
        cross_temporal.append(mean_pairwise(cross_tmp, state, 64))

        same_proto = [e["state"] for j, e in enumerate(entries) if j != idx and e["protocol"] == entry["protocol"]]
        cross_proto = [e["state"] for e in entries if e["protocol"] != entry["protocol"]]
        same_protocol.append(mean_pairwise(same_proto, state, 16))
        cross_protocol.append(mean_pairwise(cross_proto, state, 64))

        chain_entries = by_chain[(family, int(entry["chain_id"]))]
        current_stage = stage_index(str(entry["temporal"]))
        successor = [
            e["state"]
            for e in chain_entries
            if stage_index(str(e["temporal"])) == min(current_stage + 1, len(temporal_names) - 1)
        ]
        same_chain_successor.append(mean_pairwise(successor, state, 2))

        cross_same_stage = [
            e["state"]
            for e in entries
            if e["chain_id"] != entry["chain_id"] and e["temporal"] == entry["temporal"]
        ]
        cross_chain_same_stage.append(mean_pairwise(cross_same_stage, state, 48))

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_systemic_multiaxis_inventory_expansion",
            "seed": int(args.seed),
        },
        "headline_metrics": {
            "num_concepts": len(entries),
            "num_contexts": len(contexts),
            "num_relations": len(relations),
            "num_temporal_stages": len(temporals),
            "num_protocol_states": len(protocols),
            "num_chains": len(by_chain),
            "family_cross_to_within_ratio": float(np.mean(cross_family) / max(1e-6, np.mean(within_family))),
            "context_cross_to_within_ratio": float(np.mean(cross_context) / max(1e-6, np.mean(same_context))),
            "relation_cross_to_within_ratio": float(np.mean(cross_relation) / max(1e-6, np.mean(same_relation))),
            "temporal_cross_to_within_ratio": float(np.mean(cross_temporal) / max(1e-6, np.mean(same_temporal))),
            "protocol_cross_to_within_ratio": float(np.mean(cross_protocol) / max(1e-6, np.mean(same_protocol))),
            "chain_successor_to_cross_stage_ratio": float(
                np.mean(same_chain_successor) / max(1e-6, np.mean(cross_chain_same_stage))
            ),
        },
        "verdict": {
            "core_answer": (
                "Systemic multiaxis inventory now expands concept extraction into relation/context/temporal/protocol/successor dimensions, "
                "so the next theory cut can constrain not only concept patches but also reasoning transport and protocol bridges."
            ),
            "next_theory_target": (
                "Use protocol and successor statistics together to shrink ICSPB theorem space, A(I), M_feas(I), and P3/P4 intervention space."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
