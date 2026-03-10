#!/usr/bin/env python
"""
Task block D, model-guided higher-dimensional grounding benchmark.

Use concept geometry extracted from:
- GPT-2
- Qwen3-4B
- DeepSeek-7B

to build a high-dimensional continuous grounding task and compare:
- direct raw-space prototypes
- raw shared-basis + offset
- geometry-aligned dual-store grounder

The key question is whether model-side concept geometry can help solve the
novel-concept vs retention tradeoff on a harder continuous-input benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


PHASE1 = {
    "fruit": ["apple", "banana"],
    "animal": ["cat", "dog"],
    "abstract": ["truth", "logic"],
}

PHASE2 = {
    "fruit": ["pear"],
    "animal": ["horse"],
    "abstract": ["memory"],
}

ALL_FAMILIES = ["fruit", "animal", "abstract"]
ALL_CONCEPTS = {family: PHASE1[family] + PHASE2[family] for family in ALL_FAMILIES}


def resolve_snapshot_dir(repo_dir_name: str) -> str:
    roots = [
        Path(r"D:\develop\model\hub"),
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for root in roots:
        snapshot_root = root / repo_dir_name / "snapshots"
        if not snapshot_root.exists():
            continue
        candidates = sorted([p for p in snapshot_root.iterdir() if p.is_dir()])
        if candidates:
            return str(candidates[-1])
    raise FileNotFoundError(f"Cannot resolve snapshot directory for {repo_dir_name}")


def model_specs() -> List[Tuple[str, str, str]]:
    return [
        ("gpt2", resolve_snapshot_dir("models--gpt2"), "float32"),
        ("qwen3_4b", resolve_snapshot_dir("models--Qwen--Qwen3-4B"), "bfloat16"),
        ("deepseek_7b", resolve_snapshot_dir("models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B"), "bfloat16"),
    ]


def load_model(model_path: str, dtype_name: str, prefer_cuda: bool):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    want_cuda = bool(prefer_cuda and torch.cuda.is_available())
    dtype = getattr(torch, dtype_name)
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "dtype": dtype,
        "device_map": "auto" if want_cuda else "cpu",
        "attn_implementation": "eager",
    }
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return model, tok


def prompts(word: str) -> List[str]:
    return [f"This is {word}", f"That is {word}"]


def run_model(model, tok, text: str):
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc, use_cache=False, output_hidden_states=True, return_dict=True)
    return out


def mean_stack(rows: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(rows, axis=0), axis=0).astype(np.float32)


def concept_vectors(model, tok) -> Dict[str, np.ndarray]:
    rows = {}
    for family in ALL_FAMILIES:
        for concept in ALL_CONCEPTS[family]:
            vecs = []
            for text in prompts(concept):
                out = run_model(model, tok, text)
                vec = out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy().astype(np.float32)
                vecs.append(vec)
            rows[concept] = mean_stack(vecs)
    return rows


def pca_embed(concept_vecs: Dict[str, np.ndarray], rank: int = 16) -> Dict[str, np.ndarray]:
    names = [concept for family in ALL_FAMILIES for concept in ALL_CONCEPTS[family]]
    mat = np.stack([concept_vecs[name] for name in names], axis=0).astype(np.float32)
    mu = np.mean(mat, axis=0).astype(np.float32)
    centered = mat - mu[None, :]
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:rank].T.astype(np.float32)
    coords = (centered @ basis).astype(np.float32)
    out = {}
    for idx, name in enumerate(names):
        out[name] = coords[idx]
    return out


def family_basis(coords: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    family_mu = {}
    concept_offset = {}
    for family in ALL_FAMILIES:
        xs = [coords[name] for name in ALL_CONCEPTS[family]]
        mu = np.mean(np.stack(xs, axis=0), axis=0).astype(np.float32)
        family_mu[family] = mu
        concept_offset[family] = {name: (coords[name] - mu).astype(np.float32) for name in ALL_CONCEPTS[family]}
    return family_mu, concept_offset


def make_transforms(seed: int, latent_dim: int = 16, obs_dim: int = 96):
    rng = np.random.default_rng(seed)
    visual = rng.normal(scale=0.65, size=(obs_dim // 2, latent_dim)).astype(np.float32)
    tactile = rng.normal(scale=0.60, size=(obs_dim // 2, latent_dim)).astype(np.float32)
    nuisance_v = rng.normal(scale=0.22, size=(obs_dim // 2, 4)).astype(np.float32)
    nuisance_t = rng.normal(scale=0.18, size=(obs_dim // 2, 4)).astype(np.float32)
    return visual, tactile, nuisance_v, nuisance_t


def sample_continuous_input(
    rng: np.random.Generator,
    coord: np.ndarray,
    family_id: int,
    visual: np.ndarray,
    tactile: np.ndarray,
    nuisance_v: np.ndarray,
    nuisance_t: np.ndarray,
    noise: float,
    dropout_p: float,
) -> np.ndarray:
    latent_noise = rng.normal(scale=0.10, size=coord.shape[0]).astype(np.float32)
    z = (coord + latent_noise).astype(np.float32)
    family_tag = np.zeros(4, dtype=np.float32)
    family_tag[family_id] = 1.0
    v = visual @ z + nuisance_v @ family_tag + rng.normal(scale=noise, size=visual.shape[0]).astype(np.float32)
    t = tactile @ z + nuisance_t @ family_tag + rng.normal(scale=noise * 0.75, size=tactile.shape[0]).astype(np.float32)
    x = np.concatenate([v, t], axis=0).astype(np.float32)
    mask = (rng.random(x.shape[0]) > dropout_p).astype(np.float32)
    return (x * mask).astype(np.float32)


class DirectPrototypeLearner:
    def __init__(self) -> None:
        self.family_proto: Dict[str, np.ndarray] = {}
        self.family_count: Dict[str, int] = {}
        self.concept_proto: Dict[str, np.ndarray] = {}
        self.concept_count: Dict[str, int] = {}

    @staticmethod
    def _ema(prev: np.ndarray | None, x: np.ndarray, count: int) -> np.ndarray:
        if prev is None:
            return x.astype(np.float32).copy()
        alpha = min(0.28, 1.0 / float(count + 1))
        return ((1.0 - alpha) * prev + alpha * x).astype(np.float32)

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        fc = self.family_count.get(family, 0)
        cc = self.concept_count.get(concept, 0)
        self.family_proto[family] = self._ema(self.family_proto.get(family), x, fc)
        self.concept_proto[concept] = self._ema(self.concept_proto.get(concept), x, cc)
        self.family_count[family] = fc + 1
        self.concept_count[concept] = cc + 1

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        family = min(self.family_proto, key=lambda name: float(np.sum(np.square(x - self.family_proto[name]))))
        candidates = [name for name in self.concept_proto if concept_family(name) == family]
        if not candidates:
            candidates = list(self.concept_proto.keys())
        concept = min(candidates, key=lambda name: float(np.sum(np.square(x - self.concept_proto[name]))))
        return family, concept


class SharedOffsetRawLearner:
    def __init__(self) -> None:
        self.family_basis: Dict[str, np.ndarray] = {}
        self.family_count: Dict[str, int] = {}
        self.concept_offset: Dict[str, np.ndarray] = {}
        self.concept_count: Dict[str, int] = {}
        self.concept_family: Dict[str, str] = {}

    @staticmethod
    def _ema(prev: np.ndarray | None, x: np.ndarray, count: int) -> np.ndarray:
        if prev is None:
            return x.astype(np.float32).copy()
        alpha = min(0.24, 1.0 / float(count + 1))
        return ((1.0 - alpha) * prev + alpha * x).astype(np.float32)

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        fc = self.family_count.get(family, 0)
        family_mean = self._ema(self.family_basis.get(family), x, fc)
        self.family_basis[family] = family_mean
        self.family_count[family] = fc + 1
        centered = (x - family_mean).astype(np.float32)
        cc = self.concept_count.get(concept, 0)
        self.concept_offset[concept] = self._ema(self.concept_offset.get(concept), centered, cc)
        self.concept_count[concept] = cc + 1
        self.concept_family[concept] = family

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        family = min(self.family_basis, key=lambda name: float(np.sum(np.square(x - self.family_basis[name]))))
        candidates = [name for name, fam in self.concept_family.items() if fam == family]
        concept = min(
            candidates,
            key=lambda name: float(np.sum(np.square((x - self.family_basis[family]) - self.concept_offset[name]))),
        )
        return family, concept


class GeometryAlignedDualStore:
    def __init__(self, latent_dim: int, reg: float = 1e-2) -> None:
        self.latent_dim = latent_dim
        self.reg = reg
        self.w = None
        self.b = None
        self.phase1_family_basis: Dict[str, np.ndarray] = {}
        self.phase1_concept_proto: Dict[str, np.ndarray] = {}
        self.phase2_concept_offset: Dict[str, np.ndarray] = {}
        self.phase2_count: Dict[str, int] = {}

    def fit_mapping(self, xs: Sequence[np.ndarray], ys: Sequence[np.ndarray]) -> None:
        x = np.stack(xs, axis=0).astype(np.float32)
        y = np.stack(ys, axis=0).astype(np.float32)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        gram = x_aug.T @ x_aug + self.reg * np.eye(x_aug.shape[1], dtype=np.float32)
        sol = np.linalg.solve(gram, x_aug.T @ y).astype(np.float32)
        self.w = sol[:-1]
        self.b = sol[-1]

    def encode(self, x: np.ndarray) -> np.ndarray:
        assert self.w is not None and self.b is not None
        return (x @ self.w + self.b).astype(np.float32)

    def store_phase1(self, family_basis_map: Dict[str, np.ndarray], concept_coords: Dict[str, np.ndarray]) -> None:
        self.phase1_family_basis = {k: v.copy() for k, v in family_basis_map.items()}
        self.phase1_concept_proto = {k: v.copy() for k, v in concept_coords.items() if k in sum(PHASE1.values(), [])}

    def train_phase2_concept(self, x: np.ndarray, concept: str) -> None:
        z = self.encode(x)
        family = concept_family(concept)
        centered = (z - self.phase1_family_basis[family]).astype(np.float32)
        count = self.phase2_count.get(concept, 0)
        prev = self.phase2_concept_offset.get(concept)
        if prev is None:
            nxt = (0.72 * centered).astype(np.float32)
        else:
            alpha = min(0.46, 1.0 / float(count + 1))
            nxt = ((1.0 - alpha) * prev + alpha * centered).astype(np.float32)
        self.phase2_concept_offset[concept] = nxt
        self.phase2_count[concept] = count + 1

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        z = self.encode(x)
        family = min(self.phase1_family_basis, key=lambda name: float(np.sum(np.square(z - self.phase1_family_basis[name]))))
        stable = [name for name in self.phase1_concept_proto if concept_family(name) == family]
        plastic = [name for name in self.phase2_concept_offset if concept_family(name) == family]

        stable_best = None
        stable_score = float("inf")
        for name in stable:
            score = float(np.sum(np.square(z - self.phase1_concept_proto[name])))
            if score < stable_score:
                stable_score = score
                stable_best = name

        plastic_best = None
        plastic_score = float("inf")
        for name in plastic:
            proto = self.phase1_family_basis[family] + self.phase2_concept_offset[name]
            score = float(np.sum(np.square(z - proto)))
            if score < plastic_score:
                plastic_score = score
                plastic_best = name

        if plastic_best is not None and plastic_score + 0.015 < stable_score:
            return family, plastic_best
        assert stable_best is not None
        return family, stable_best


def concept_family(concept: str) -> str:
    for family in ALL_FAMILIES:
        if concept in ALL_CONCEPTS[family]:
            return family
    raise KeyError(concept)


def evaluate_model_on_groups(model, coords: Dict[str, np.ndarray], groups: Dict[str, List[str]], transforms, repeats: int, rng, noise: float, dropout_p: float):
    family_ok = 0
    concept_ok = 0
    total = 0
    visual, tactile, nuisance_v, nuisance_t = transforms
    for _ in range(repeats):
        for family_idx, family in enumerate(ALL_FAMILIES):
            if family not in groups:
                continue
            for concept in groups[family]:
                x = sample_continuous_input(
                    rng, coords[concept], family_idx, visual, tactile, nuisance_v, nuisance_t, noise, dropout_p
                )
                pred_family, pred_concept = model.predict(x)
                family_ok += int(pred_family == family)
                concept_ok += int(pred_concept == concept)
                total += 1
    return {
        "family_accuracy": float(family_ok / max(1, total)),
        "concept_accuracy": float(concept_ok / max(1, total)),
    }


def run_grounding_for_model(model_name: str, coords: Dict[str, np.ndarray], seed: int, noise: float, dropout_p: float) -> Dict[str, object]:
    latent_dim = int(next(iter(coords.values())).shape[0])
    transforms = make_transforms(seed + abs(hash(model_name)) % 1000, latent_dim=latent_dim, obs_dim=96)
    visual, tactile, nuisance_v, nuisance_t = transforms
    family_mu, _concept_offset = family_basis(coords)
    rng = np.random.default_rng(seed)

    direct = DirectPrototypeLearner()
    raw_shared = SharedOffsetRawLearner()
    geom = GeometryAlignedDualStore(latent_dim=latent_dim)

    train_xs = []
    train_ys = []
    for _ in range(36):
        for family_idx, family in enumerate(ALL_FAMILIES):
            for concept in PHASE1[family]:
                x = sample_continuous_input(rng, coords[concept], family_idx, visual, tactile, nuisance_v, nuisance_t, noise, dropout_p)
                direct.train(x, family, concept)
                raw_shared.train(x, family, concept)
                train_xs.append(x)
                train_ys.append(coords[concept])
    geom.fit_mapping(train_xs, train_ys)
    geom.store_phase1(
        {family: family_mu[family] for family in ALL_FAMILIES},
        {concept: coords[concept] for family in ALL_FAMILIES for concept in PHASE1[family]},
    )

    phase1_eval = {
        "direct": evaluate_model_on_groups(direct, coords, PHASE1, transforms, 20, rng, noise, dropout_p),
        "raw_shared": evaluate_model_on_groups(raw_shared, coords, PHASE1, transforms, 20, rng, noise, dropout_p),
        "geometry_dual_store": evaluate_model_on_groups(geom, coords, PHASE1, transforms, 20, rng, noise, dropout_p),
    }

    for _ in range(2):
        for family_idx, family in enumerate(ALL_FAMILIES):
            for concept in PHASE2[family]:
                x = sample_continuous_input(rng, coords[concept], family_idx, visual, tactile, nuisance_v, nuisance_t, noise, dropout_p)
                direct.train(x, family, concept)
                raw_shared.train(x, family, concept)
                geom.train_phase2_concept(x, concept)

    novel_eval = {
        "direct": evaluate_model_on_groups(direct, coords, PHASE2, transforms, 24, rng, noise, dropout_p),
        "raw_shared": evaluate_model_on_groups(raw_shared, coords, PHASE2, transforms, 24, rng, noise, dropout_p),
        "geometry_dual_store": evaluate_model_on_groups(geom, coords, PHASE2, transforms, 24, rng, noise, dropout_p),
    }

    for _ in range(8):
        for family_idx, family in enumerate(ALL_FAMILIES):
            for concept in PHASE2[family]:
                x = sample_continuous_input(rng, coords[concept], family_idx, visual, tactile, nuisance_v, nuisance_t, noise, dropout_p)
                direct.train(x, family, concept)
                raw_shared.train(x, family, concept)
                geom.train_phase2_concept(x, concept)

    retention_eval = {
        "direct": evaluate_model_on_groups(direct, coords, PHASE1, transforms, 20, rng, noise, dropout_p),
        "raw_shared": evaluate_model_on_groups(raw_shared, coords, PHASE1, transforms, 20, rng, noise, dropout_p),
        "geometry_dual_store": evaluate_model_on_groups(geom, coords, PHASE1, transforms, 20, rng, noise, dropout_p),
    }

    overall_eval = {
        "direct": evaluate_model_on_groups(direct, coords, ALL_CONCEPTS, transforms, 20, rng, noise, dropout_p),
        "raw_shared": evaluate_model_on_groups(raw_shared, coords, ALL_CONCEPTS, transforms, 20, rng, noise, dropout_p),
        "geometry_dual_store": evaluate_model_on_groups(geom, coords, ALL_CONCEPTS, transforms, 20, rng, noise, dropout_p),
    }

    def grounding_score(row):
        return float(
            (
                phase1_eval[row]["family_accuracy"]
                + novel_eval[row]["family_accuracy"]
                + overall_eval[row]["family_accuracy"]
                + 2.0 * novel_eval[row]["concept_accuracy"]
                + 1.5 * overall_eval[row]["concept_accuracy"]
                + 1.0 * retention_eval[row]["concept_accuracy"]
            )
            / 7.5
        )

    systems = {}
    for name in ["direct", "raw_shared", "geometry_dual_store"]:
        systems[name] = {
            "phase1_family_accuracy": phase1_eval[name]["family_accuracy"],
            "phase1_concept_accuracy": phase1_eval[name]["concept_accuracy"],
            "novel_family_accuracy": novel_eval[name]["family_accuracy"],
            "novel_concept_accuracy": novel_eval[name]["concept_accuracy"],
            "retention_family_accuracy": retention_eval[name]["family_accuracy"],
            "retention_concept_accuracy": retention_eval[name]["concept_accuracy"],
            "overall_family_accuracy": overall_eval[name]["family_accuracy"],
            "overall_concept_accuracy": overall_eval[name]["concept_accuracy"],
            "grounding_score": grounding_score(name),
        }

    baseline = systems["direct"]
    return {
        "systems": systems,
        "gains_vs_direct": {
            name: {
                "grounding_score_gain": float(row["grounding_score"] - baseline["grounding_score"]),
                "novel_concept_gain": float(row["novel_concept_accuracy"] - baseline["novel_concept_accuracy"]),
                "retention_concept_gain": float(row["retention_concept_accuracy"] - baseline["retention_concept_accuracy"]),
                "overall_concept_gain": float(row["overall_concept_accuracy"] - baseline["overall_concept_accuracy"]),
            }
            for name, row in systems.items()
            if name != "direct"
        },
        "hypotheses": {
            "geometry_beats_direct_on_novel_and_retention": bool(
                systems["geometry_dual_store"]["novel_concept_accuracy"] > baseline["novel_concept_accuracy"]
                and systems["geometry_dual_store"]["retention_concept_accuracy"] > baseline["retention_concept_accuracy"]
            )
        },
    }


def analyze_model_geometry(model_name: str, model_path: str, dtype_name: str, prefer_cuda: bool, seed: int, noise: float, dropout_p: float) -> Dict[str, object]:
    t0 = time.time()
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    vecs = concept_vectors(model, tok)
    coords = pca_embed(vecs, rank=16)
    bridge = run_grounding_for_model(model_name, coords, seed, noise, dropout_p)
    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "meta": {
            "model_name": model_name,
            "model_path": model_path,
            "runtime_sec": float(time.time() - t0),
        },
        "bridge": bridge,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Higher-dimensional grounding bridge with GPT2/Qwen3/DeepSeek7B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", type=float, default=0.22)
    ap.add_argument("--dropout-p", type=float, default=0.14)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/gpt2_qwen3_deepseek7b_highdim_grounding_bridge_20260309.json",
    )
    args = ap.parse_args()

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
        },
        "models": {},
    }

    for model_name, model_path, dtype_name in model_specs():
        print(f"[run] {model_name} from {model_path}")
        row = analyze_model_geometry(model_name, model_path, dtype_name, not args.cpu_only, int(args.seed), float(args.noise), float(args.dropout_p))
        results["models"][model_name] = row
        geom = row["bridge"]["systems"]["geometry_dual_store"]
        direct = row["bridge"]["systems"]["direct"]
        print(
            f"[summary] {model_name} geometry_grounding={geom['grounding_score']:.4f} "
            f"direct_grounding={direct['grounding_score']:.4f} "
            f"novel_gain={geom['novel_concept_accuracy'] - direct['novel_concept_accuracy']:.4f} "
            f"retention_gain={geom['retention_concept_accuracy'] - direct['retention_concept_accuracy']:.4f}"
        )

    results["global_summary"] = {
        "models_geometry_beats_direct_on_novel_and_retention": {
            model_name: bool(row["bridge"]["hypotheses"]["geometry_beats_direct_on_novel_and_retention"])
            for model_name, row in results["models"].items()
        }
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
