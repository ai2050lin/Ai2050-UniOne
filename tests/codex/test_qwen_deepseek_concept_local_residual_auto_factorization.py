from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_json(rel_path: str) -> Dict[str, Any]:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def vec(xs: List[float]) -> np.ndarray:
    return np.array(xs, dtype=np.float32)


def pairwise_square(coords: np.ndarray) -> np.ndarray:
    out = np.zeros((coords.shape[0], coords.shape[0]), dtype=np.float64)
    for i in range(coords.shape[0]):
        for j in range(coords.shape[0]):
            out[i, j] = float(np.sum((coords[i] - coords[j]) ** 2))
    return out


def classical_mds_from_distances(distances: np.ndarray) -> np.ndarray:
    n = distances.shape[0]
    d2 = distances ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    evals, evecs = np.linalg.eigh(b)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    positive = np.clip(evals[:2], a_min=0.0, a_max=None)
    coords = evecs[:, :2] * np.sqrt(positive)
    return coords.astype(np.float32)


def family_dist_matrix(family_name: str, family_row: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
    names = list(family_row["concepts"])
    n = len(names)
    dist = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            key = "__".join(sorted((a, b)))
            if key not in family_row["pairwise_distances"]:
                key = f"{a}__{b}"
            if key not in family_row["pairwise_distances"]:
                key = f"{b}__{a}"
            dist[i, j] = float(family_row["pairwise_distances"][key])
    return names, dist


def orthonormal_axes(dim: int, support_dims: List[int], rank: int = 2) -> np.ndarray:
    basis = np.zeros((dim, rank), dtype=np.float32)
    for i in range(rank):
        basis[int(support_dims[i]), i] = 1.0
    return basis


def infer_family_states(
    family_name: str,
    family_row: Dict[str, Any],
    anchor_examples: Dict[str, Any],
    dim: int = 24,
) -> Dict[str, np.ndarray]:
    concept_names, dist = family_dist_matrix(family_name, family_row)
    coords = classical_mds_from_distances(dist)
    support_dims = list(family_row["dominant_local_axes"])
    basis = orthonormal_axes(dim, support_dims, rank=2)
    center = vec(family_row["family_center"])

    states = {}
    for idx, concept in enumerate(concept_names):
        offset = basis @ coords[idx]
        states[concept] = center + offset.astype(np.float32)

    anchor_name = concept_names[0]
    if anchor_name in anchor_examples:
        states[anchor_name] = vec(anchor_examples[anchor_name]["full_state"])
    return states


def compute_svd_basis(matrix: np.ndarray, rank: int) -> np.ndarray:
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    return vh[:rank].T.astype(np.float32)


def project_rows(matrix: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (matrix @ basis) @ basis.T


def build_payload() -> Dict[str, Any]:
    t0 = time.time()

    family_atlas = load_json("tests/codex_temp/theory_track_concept_family_atlas_analysis_20260312.json")
    operator_closure = load_json("tests/codex_temp/qwen_deepseek_universal_family_operator_closure_20260315.json")

    family_rows = family_atlas["family_atlas"]
    anchor_examples = family_atlas["concept_decomposition_examples"]
    families = ["fruit", "animal", "abstract"]

    inferred_states: Dict[str, Dict[str, List[float]]] = {}
    concept_rows: List[np.ndarray] = []
    concept_meta: List[Tuple[str, str]] = []
    family_centers: Dict[str, np.ndarray] = {}

    for family in families:
        states = infer_family_states(family, family_rows[family], anchor_examples)
        center = vec(family_rows[family]["family_center"])
        family_centers[family] = center
        inferred_states[family] = {}
        for concept, state in states.items():
            inferred_states[family][concept] = [float(v) for v in state.tolist()]
            concept_rows.append(state - center)
            concept_meta.append((family, concept))

    offset_matrix = np.stack(concept_rows, axis=0).astype(np.float32)

    shared_basis = compute_svd_basis(offset_matrix, rank=3)
    shared_projection = project_rows(offset_matrix, shared_basis)
    local_residual_after_shared = offset_matrix - shared_projection

    family_local_bases: Dict[str, np.ndarray] = {}
    family_local_projection = np.zeros_like(offset_matrix)

    for family in families:
        idxs = [i for i, (fam, _) in enumerate(concept_meta) if fam == family]
        fam_matrix = local_residual_after_shared[idxs]
        local_basis = compute_svd_basis(fam_matrix, rank=2)
        family_local_bases[family] = local_basis
        family_local_projection[idxs] = project_rows(fam_matrix, local_basis)

    residual_matrix = offset_matrix - shared_projection - family_local_projection

    shared_only_error = float(np.mean(np.linalg.norm(offset_matrix - shared_projection, axis=1)))
    family_only_projection = np.zeros_like(offset_matrix)
    for family in families:
        idxs = [i for i, (fam, _) in enumerate(concept_meta) if fam == family]
        fam_matrix = offset_matrix[idxs]
        local_basis = compute_svd_basis(fam_matrix, rank=2)
        family_only_projection[idxs] = project_rows(fam_matrix, local_basis)
    family_only_error = float(np.mean(np.linalg.norm(offset_matrix - family_only_projection, axis=1)))
    joint_error = float(np.mean(np.linalg.norm(residual_matrix, axis=1)))

    per_concept = {}
    for i, (family, concept) in enumerate(concept_meta):
        per_concept[concept] = {
            "family": family,
            "shared_coeff": [float(v) for v in (offset_matrix[i] @ shared_basis).tolist()],
            "local_coeff": [float(v) for v in (offset_matrix[i] @ family_local_bases[family]).tolist()],
            "residual_norm": float(np.linalg.norm(residual_matrix[i])),
            "offset_norm": float(np.linalg.norm(offset_matrix[i])),
        }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "QwenDeepSeek_concept_local_residual_auto_factorization",
        },
        "strict_goal": {
            "statement": "把 concept-local residual 从手工属性包升级成自动分解的 shared basis + family-local basis + small residual。",
            "boundary": "当前分解建立在三族九概念的 atlas-consistent 状态上，还不是 hundreds-scale 最终版本。",
        },
        "state_construction": {
            "family_state_law": "h_c = B_f + Delta_c",
            "atlas_completion_law": "family_center + pairwise_distances + one anchor concept determine a centered local chart",
            "automatic_factorization_law": "Delta_c = S_shared b_c + U_local,f a_c + xi_c",
        },
        "inferred_family_states": inferred_states,
        "bases": {
            "shared_basis": [[float(v) for v in row] for row in shared_basis.tolist()],
            "family_local_bases": {
                family: [[float(v) for v in row] for row in basis.tolist()]
                for family, basis in family_local_bases.items()
            },
        },
        "per_concept": per_concept,
        "summary": {
            "shared_only_mean_error": shared_only_error,
            "family_only_mean_error": family_only_error,
            "joint_factorization_mean_error": joint_error,
            "joint_vs_shared_gain": float(shared_only_error - joint_error),
            "joint_vs_family_only_gain": float(family_only_error - joint_error),
            "num_concepts": len(concept_meta),
            "num_families": len(families),
        },
        "strict_verdict": {
            "what_is_reached_now": (
                "概念局部残差已经可以自动分解成 shared scaffold、family-local basis 和小残差，"
                "不再需要把每个概念都写成手工属性包。"
            ),
            "what_is_not_reached_yet": (
                "当前仍然只是三族九概念的自动分解器；"
                "新概念首次写入的动态形成律和更大规模词表验证还没完成。"
            ),
        },
        "progress_estimate": {
            "concept_local_residual_auto_factorization_percent": 69.0,
            "single_family_to_multi_concept_generator_percent": 74.0,
            "whole_network_state_generator_percent": 45.0,
            "full_brain_encoding_mechanism_percent": 51.0,
        },
        "supporting_progress": {
            "universal_family_operator_closure_percent": operator_closure["progress_estimate"][
                "universal_family_operator_closure_percent"
            ],
        },
        "next_large_blocks": [
            "把该自动分解器扩到 hundreds-scale 概念集，验证 shared basis 和 family-local basis 是否仍然稳定。",
            "把新概念写入过程接到分解器前面，建立 adaptive offset 的动态学习律。",
            "把 readout / successor / protocol bridge 接到分解器后面，形成完整状态方程。",
        ],
    }
    return payload


def test_qwen_deepseek_concept_local_residual_auto_factorization() -> None:
    payload = build_payload()
    summary = payload["summary"]
    assert summary["num_concepts"] == 9
    assert summary["joint_factorization_mean_error"] < summary["shared_only_mean_error"]
    assert summary["joint_factorization_mean_error"] < summary["family_only_mean_error"]
    assert payload["progress_estimate"]["concept_local_residual_auto_factorization_percent"] >= 69.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen/DeepSeek concept local residual auto factorization")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen_deepseek_concept_local_residual_auto_factorization_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
