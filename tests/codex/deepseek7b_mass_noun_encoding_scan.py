#!/usr/bin/env python
"""
Mass noun-to-neuron encoding scan for DeepSeek-7B style models.

Goal:
1) Collect neuron signatures for a large noun set.
2) Quantify reusable/shared encoding structure.
3) Report candidate mathematical regularities of concept encoding.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class NounItem:
    noun: str
    category: str


def default_noun_catalog() -> List[NounItem]:
    rows = [
        ("apple", "fruit"), ("banana", "fruit"), ("orange", "fruit"), ("grape", "fruit"), ("pear", "fruit"),
        ("peach", "fruit"), ("mango", "fruit"), ("lemon", "fruit"), ("strawberry", "fruit"), ("watermelon", "fruit"),
        ("pineapple", "fruit"), ("cherry", "fruit"), ("plum", "fruit"), ("kiwi", "fruit"), ("coconut", "fruit"),
        ("rabbit", "animal"), ("cat", "animal"), ("dog", "animal"), ("horse", "animal"), ("tiger", "animal"),
        ("lion", "animal"), ("bird", "animal"), ("fish", "animal"), ("elephant", "animal"), ("monkey", "animal"),
        ("wolf", "animal"), ("bear", "animal"), ("deer", "animal"), ("goat", "animal"), ("zebra", "animal"),
        ("sun", "celestial"), ("moon", "celestial"), ("star", "celestial"), ("planet", "celestial"), ("comet", "celestial"),
        ("galaxy", "celestial"), ("asteroid", "celestial"), ("meteor", "celestial"), ("satellite", "celestial"), ("nebula", "celestial"),
        ("cloud", "weather"), ("rain", "weather"), ("snow", "weather"), ("wind", "weather"), ("storm", "weather"),
        ("thunder", "weather"), ("lightning", "weather"), ("fog", "weather"), ("humidity", "weather"), ("temperature", "weather"),
        ("car", "vehicle"), ("bus", "vehicle"), ("train", "vehicle"), ("bicycle", "vehicle"), ("airplane", "vehicle"),
        ("ship", "vehicle"), ("truck", "vehicle"), ("motorcycle", "vehicle"), ("subway", "vehicle"), ("boat", "vehicle"),
        ("chair", "object"), ("table", "object"), ("bed", "object"), ("lamp", "object"), ("door", "object"),
        ("window", "object"), ("bottle", "object"), ("cup", "object"), ("spoon", "object"), ("knife", "object"),
        ("clock", "object"), ("mirror", "object"), ("phone", "object"), ("computer", "object"), ("keyboard", "object"),
        ("bread", "food"), ("rice", "food"), ("meat", "food"), ("soup", "food"), ("pizza", "food"),
        ("cake", "food"), ("coffee", "food"), ("tea", "food"), ("milk", "food"), ("cheese", "food"),
        ("noodle", "food"), ("egg", "food"), ("salad", "food"), ("butter", "food"), ("chocolate", "food"),
        ("tree", "nature"), ("flower", "nature"), ("grass", "nature"), ("forest", "nature"), ("river", "nature"),
        ("mountain", "nature"), ("ocean", "nature"), ("desert", "nature"), ("leaf", "nature"), ("seed", "nature"),
        ("child", "human"), ("teacher", "human"), ("doctor", "human"), ("student", "human"), ("parent", "human"),
        ("friend", "human"), ("king", "human"), ("queen", "human"), ("artist", "human"), ("worker", "human"),
        ("lawyer", "human"), ("pilot", "human"), ("engineer", "human"), ("farmer", "human"), ("nurse", "human"),
        ("algorithm", "tech"), ("data", "tech"), ("number", "tech"), ("equation", "tech"), ("database", "tech"),
        ("network", "tech"), ("software", "tech"), ("hardware", "tech"), ("robot", "tech"), ("chip", "tech"),
        ("love", "abstract"), ("hate", "abstract"), ("justice", "abstract"), ("peace", "abstract"), ("war", "abstract"),
        ("music", "abstract"), ("art", "abstract"), ("history", "abstract"), ("future", "abstract"), ("memory", "abstract"),
    ]
    return [NounItem(noun=n, category=c) for n, c in rows]


def load_nouns(path: str | None, max_nouns: int | None) -> List[NounItem]:
    if not path:
        rows = default_noun_catalog()
        return rows[:max_nouns] if max_nouns else rows

    out: List[NounItem] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"nouns file not found: {path}")

    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "," in s:
            noun, cat = [x.strip() for x in s.split(",", 1)]
            out.append(NounItem(noun=noun, category=cat or "uncategorized"))
        else:
            out.append(NounItem(noun=s, category="uncategorized"))

    if max_nouns:
        out = out[:max_nouns]
    return out


class GateCollector:
    def __init__(self, model):
        self.layers = list(model.model.layers)
        self.buffers: List[torch.Tensor | None] = [None for _ in self.layers]
        self.handles = []
        for li, layer in enumerate(self.layers):
            self.handles.append(layer.mlp.gate_proj.register_forward_hook(self._mk_hook(li)))

    def _mk_hook(self, layer_idx: int):
        def _hook(_module, _inputs, output):
            self.buffers[layer_idx] = output[0, -1, :].detach().float().cpu()
            return output

        return _hook

    def reset(self):
        for i in range(len(self.buffers)):
            self.buffers[i] = None

    def get_flat(self) -> np.ndarray:
        miss = [i for i, x in enumerate(self.buffers) if x is None]
        if miss:
            raise RuntimeError(f"Missing hook outputs for layers: {miss}")
        arr = np.concatenate([x.numpy() for x in self.buffers if x is not None], axis=0).astype(np.float32)
        return arr

    def close(self):
        for h in self.handles:
            h.remove()


def load_model(model_id: str, dtype_name: str, local_files_only: bool):
    dtype = getattr(torch, dtype_name)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def run_prompt(model, tok, text: str):
    device = next(model.parameters()).device
    inp = tok(text, return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.inference_mode():
        return model(**inp, use_cache=False, return_dict=True)


def noun_prompts(noun: str) -> List[str]:
    return [
        f"This is a {noun}",
        f"I saw a {noun}",
        f"People discuss {noun}",
        f"The {noun} is often",
        f"A {noun} can be",
    ]


def topk_indices(vec: np.ndarray, k: int) -> np.ndarray:
    k = min(k, vec.shape[0])
    if k <= 0:
        return np.array([], dtype=np.int64)
    idx = np.argpartition(vec, -k)[-k:]
    idx = idx[np.argsort(vec[idx])[::-1]]
    return idx


def index_to_layer_neuron(idx: int, d_ff: int) -> Tuple[int, int]:
    return idx // d_ff, idx % d_ff


def layer_distribution(indices: Sequence[int], d_ff: int) -> Dict[int, int]:
    ctr = Counter([index_to_layer_neuron(int(i), d_ff)[0] for i in indices])
    return dict(sorted(ctr.items()))


def normalized_entropy(values: np.ndarray) -> float:
    vals = values.astype(np.float64)
    s = vals.sum()
    if s <= 0:
        return 0.0
    p = vals / s
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    h = -(p * np.log(p)).sum()
    return float(h / np.log(len(values)))


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def pairwise_similarity(mat: np.ndarray, signatures: List[np.ndarray], categories: List[str]) -> Dict[str, float]:
    norms = np.linalg.norm(mat, axis=1) + 1e-8
    cos_m = (mat @ mat.T) / np.outer(norms, norms)

    n = mat.shape[0]
    within_cos = []
    between_cos = []
    within_j = []
    between_j = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_v = float(cos_m[i, j])
            jac_v = jaccard(signatures[i], signatures[j])
            if categories[i] == categories[j]:
                within_cos.append(cos_v)
                within_j.append(jac_v)
            else:
                between_cos.append(cos_v)
                between_j.append(jac_v)

    def _m(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    return {
        "within_category_cosine_mean": _m(within_cos),
        "between_category_cosine_mean": _m(between_cos),
        "within_category_jaccard_mean": _m(within_j),
        "between_category_jaccard_mean": _m(between_j),
        "cosine_gap": _m(within_cos) - _m(between_cos),
        "jaccard_gap": _m(within_j) - _m(between_j),
    }


def low_rank_stats(mat: np.ndarray) -> Dict[str, float]:
    x = mat.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    gram = x @ x.T  # noun x noun
    eig = np.linalg.eigvalsh(gram)
    eig = np.clip(eig, a_min=0.0, a_max=None)
    if eig.sum() <= 1e-12:
        return {
            "participation_ratio": 0.0,
            "top1_energy_ratio": 0.0,
            "top5_energy_ratio": 0.0,
        }
    pr = (eig.sum() ** 2) / (np.square(eig).sum() + 1e-12)
    eig_sorted = np.sort(eig)[::-1]
    top1 = float(eig_sorted[:1].sum() / eig_sorted.sum())
    top5 = float(eig_sorted[: min(5, len(eig_sorted))].sum() / eig_sorted.sum())
    return {
        "participation_ratio": float(pr),
        "top1_energy_ratio": top1,
        "top5_energy_ratio": top5,
    }


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _linear_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clip01((value - low) / (high - low))


def build_mechanism_scorecard(
    regularity: Dict[str, object],
    causal_ablation: Dict[str, object],
    n_nouns: int,
) -> Dict[str, object]:
    pair = (regularity.get("pairwise_similarity") or {}) if isinstance(regularity, dict) else {}
    low_rank = (regularity.get("low_rank_structure") or {}) if isinstance(regularity, dict) else {}

    cosine_gap = float(pair.get("cosine_gap", 0.0))
    jaccard_gap = float(pair.get("jaccard_gap", 0.0))
    reused_ratio = float(regularity.get("reused_neuron_ratio", 0.0)) if isinstance(regularity, dict) else 0.0
    layer_top3_ratio = float(regularity.get("top3_layer_usage_ratio", 0.0)) if isinstance(regularity, dict) else 0.0
    layer_entropy = float(regularity.get("layer_usage_entropy_norm", 0.0)) if isinstance(regularity, dict) else 0.0
    pr = float(low_rank.get("participation_ratio", 0.0))
    top5_energy = float(low_rank.get("top5_energy_ratio", 0.0))

    separation_score = 0.65 * _linear_score(cosine_gap, 0.005, 0.065) + 0.35 * _linear_score(jaccard_gap, 0.01, 0.2)
    sparsity_score = 1.0 - _linear_score(reused_ratio, 0.004, 0.05)
    focus_score = _linear_score(layer_top3_ratio, 0.18, 0.75)
    entropy_balance_score = 1.0 - min(abs(layer_entropy - 0.62) / 0.62, 1.0)
    reuse_structure_score = 0.5 * sparsity_score + 0.3 * focus_score + 0.2 * entropy_balance_score
    rank_compact_score = (
        (1.0 - _linear_score(pr, max(8.0, n_nouns * 0.4), max(16.0, n_nouns * 1.0))) * 0.45
        + _linear_score(top5_energy, 0.2, 0.85) * 0.55
    )

    causal_enabled = bool(causal_ablation.get("enabled")) if isinstance(causal_ablation, dict) else False
    causal_score = 0.0
    causal_evidence = {}
    if causal_enabled:
        agg = causal_ablation.get("aggregate") or {}
        reused_ab = causal_ablation.get("reused_ablation") or {}
        cm_prob = float(agg.get("mean_causal_margin_prob", 0.0))
        cm_logprob = float(agg.get("mean_causal_margin_logprob", 0.0))
        cm_rank = float(agg.get("mean_causal_margin_rank_worse", 0.0))
        cm_seq_logprob = float(agg.get("mean_causal_margin_seq_logprob", 0.0))
        cm_seq_avg_logprob = float(agg.get("mean_causal_margin_seq_avg_logprob", 0.0))
        cm_prob_z = float(agg.get("causal_margin_prob_z", 0.0))
        cm_seq_logprob_z = float(agg.get("causal_margin_seq_logprob_z", 0.0))
        pos_ratio = float(agg.get("positive_causal_margin_ratio", 0.0))
        reused_cm = float(reused_ab.get("mean_causal_margin_prob", 0.0)) if isinstance(reused_ab, dict) else 0.0
        causal_score = (
            0.22 * _linear_score(cm_prob, 0.003, 0.06)
            + 0.15 * _linear_score(cm_logprob, 0.003, 0.08)
            + 0.13 * _linear_score(cm_rank, 0.2, 6.0)
            + 0.14 * _linear_score(pos_ratio, 0.55, 0.95)
            + 0.12 * _linear_score(max(cm_prob_z, 0.0), 1.0, 3.5)
            + 0.14 * _linear_score(cm_seq_logprob, 0.003, 0.08)
            + 0.10 * _linear_score(cm_seq_avg_logprob, 0.003, 0.08)
        )
        causal_evidence = {
            "mean_causal_margin_prob": cm_prob,
            "mean_causal_margin_logprob": cm_logprob,
            "mean_causal_margin_rank_worse": cm_rank,
            "mean_causal_margin_seq_logprob": cm_seq_logprob,
            "mean_causal_margin_seq_avg_logprob": cm_seq_avg_logprob,
            "causal_margin_prob_z": cm_prob_z,
            "causal_margin_seq_logprob_z": cm_seq_logprob_z,
            "positive_causal_margin_ratio": pos_ratio,
            "reused_mean_causal_margin_prob": reused_cm,
        }

    if causal_enabled:
        overall = 0.28 * separation_score + 0.22 * reuse_structure_score + 0.2 * rank_compact_score + 0.3 * causal_score
    else:
        overall = 0.42 * separation_score + 0.33 * reuse_structure_score + 0.25 * rank_compact_score

    if overall >= 0.78:
        grade = "strong_mechanistic_evidence"
    elif overall >= 0.58:
        grade = "moderate_mechanistic_evidence"
    elif overall >= 0.42:
        grade = "weak_mechanistic_evidence"
    else:
        grade = "insufficient_evidence"

    guidance = []
    if separation_score < 0.5:
        guidance.append("Increase category coverage and rebalance samples to improve within-vs-between separation.")
    if reuse_structure_score < 0.5:
        guidance.append("Tune top_signature_k and reuse_threshold to isolate sparse reusable neurons.")
    if rank_compact_score < 0.5:
        guidance.append("Expand prompts per noun and repeat runs to confirm low-rank manifold structure.")
    if causal_enabled and causal_score < 0.55:
        guidance.append("Increase ablation samples and random trials to strengthen causal margin confidence.")
    if not causal_enabled:
        guidance.append("Enable --run-causal-ablation to upgrade from correlation evidence to causal evidence.")
    if not guidance:
        guidance.append("Proceed to minimal-circuit extraction and cross-lingual counterfactual validation.")

    return {
        "overall_score": float(overall),
        "grade": grade,
        "subscores": {
            "structure_separation": float(separation_score),
            "reuse_sparsity_structure": float(reuse_structure_score),
            "low_rank_compactness": float(rank_compact_score),
            "causal_evidence": float(causal_score),
        },
        "causal_enabled": causal_enabled,
        "signals": {
            "cosine_gap": cosine_gap,
            "jaccard_gap": jaccard_gap,
            "reused_neuron_ratio": reused_ratio,
            "top3_layer_usage_ratio": layer_top3_ratio,
            "layer_usage_entropy_norm": layer_entropy,
            "participation_ratio": pr,
            "top5_energy_ratio": top5_energy,
            **causal_evidence,
        },
        "guidance": guidance,
    }


def is_cjk_text(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def ablation_prompt_for_noun(noun: str) -> str:
    return "This is a"


def target_token_ids_for_noun(tok, noun: str) -> List[int]:
    if is_cjk_text(noun):
        ids = tok.encode(noun, add_special_tokens=False)
        return [int(x) for x in ids] if ids else []

    ids = tok.encode(" " + noun, add_special_tokens=False)
    if ids:
        return [int(x) for x in ids]
    ids = tok.encode(noun, add_special_tokens=False)
    return [int(x) for x in ids] if ids else []


def target_token_id_for_noun(tok, noun: str) -> int | None:
    ids = target_token_ids_for_noun(tok, noun)
    return int(ids[0]) if ids else None


def softmax_prob_at(logits: torch.Tensor, target_idx: int) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    return float(probs[target_idx].item())


def logprob_at(logits: torch.Tensor, target_idx: int) -> float:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return float(log_probs[target_idx].item())


def rank_of_token(logits: torch.Tensor, target_idx: int) -> int:
    target = logits[target_idx]
    return int((logits > target).sum().item() + 1)


def flat_indices_to_layer_map(indices: Sequence[int], d_ff: int) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = defaultdict(list)
    for idx in indices:
        li, ni = index_to_layer_neuron(int(idx), d_ff)
        out[li].append(ni)
    return {k: sorted(set(v)) for k, v in out.items()}


class LastTokenGateAblation:
    def __init__(self, model, layer_to_neurons: Dict[int, List[int]], ablate_all_positions: bool = False):
        self.handles = []
        self.layer_to_neurons = layer_to_neurons
        self.ablate_all_positions = ablate_all_positions
        for li, neurons in layer_to_neurons.items():
            if not neurons:
                continue
            layer = model.model.layers[li].mlp.gate_proj
            self.handles.append(layer.register_forward_hook(self._mk_hook(neurons, ablate_all_positions)))

    @staticmethod
    def _mk_hook(neurons: List[int], ablate_all_positions: bool):
        idx_cpu = torch.tensor(neurons, dtype=torch.long)

        def _hook(_module, _inputs, output):
            out = output.clone()
            idx = idx_cpu.to(out.device)
            if ablate_all_positions:
                out[:, :, idx] = 0.0
            else:
                out[:, -1, idx] = 0.0
            return out

        return _hook

    def close(self):
        for h in self.handles:
            h.remove()


def run_prompt_target_score(model, tok, prompt: str, target_token_id: int) -> Dict[str, float]:
    out = run_prompt(model, tok, prompt)
    logits = out.logits[0, -1, :].detach()
    return {
        "target_logit": float(logits[target_token_id].item()),
        "target_prob": softmax_prob_at(logits, target_token_id),
        "target_logprob": logprob_at(logits, target_token_id),
        "target_rank": float(rank_of_token(logits, target_token_id)),
    }


def run_prompt_sequence_score(model, tok, prefix: str, target_token_ids: Sequence[int]) -> Dict[str, float]:
    if not target_token_ids:
        return {"target_seq_logprob": 0.0, "target_seq_avg_logprob": 0.0}

    device = next(model.parameters()).device
    prefix_ids = tok.encode(prefix, add_special_tokens=False)
    if len(prefix_ids) == 0:
        return {"target_seq_logprob": 0.0, "target_seq_avg_logprob": 0.0}

    full_ids = prefix_ids + [int(x) for x in target_token_ids]
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = out.logits[0]  # [seq, vocab]

    start = len(prefix_ids)
    total = 0.0
    count = 0
    for i, token_id in enumerate(target_token_ids):
        pred_pos = start + i - 1
        if pred_pos < 0 or pred_pos >= logits.shape[0]:
            continue
        total += logprob_at(logits[pred_pos], int(token_id))
        count += 1
    avg = float(total / count) if count > 0 else 0.0
    return {"target_seq_logprob": float(total), "target_seq_avg_logprob": avg}


def sample_random_indices(total_neurons: int, sample_size: int, forbidden: set[int], rng: random.Random) -> np.ndarray:
    sample_size = min(max(sample_size, 0), total_neurons)
    if sample_size <= 0:
        return np.array([], dtype=np.int64)

    picked: List[int] = []
    while len(picked) < sample_size:
        need = sample_size - len(picked)
        draw = max(need * 3, 16)
        candidates = [rng.randrange(total_neurons) for _ in range(draw)]
        for c in candidates:
            if c in forbidden:
                continue
            picked.append(c)
            if len(picked) >= sample_size:
                break
        if len(picked) > sample_size:
            picked = picked[:sample_size]
    return np.asarray(picked[:sample_size], dtype=np.int64)


def ablation_eval_prompts(_noun: str) -> List[str]:
    # Prefix-only templates: model predicts the next noun token.
    return [
        "This is a",
        "I saw a",
        "People discuss",
        "An example is",
        "The object is",
    ]


def mean_confidence_interval(values: List[float], z: float = 1.96) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "sem": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sem = float(std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_low": float(mean - z * sem),
        "ci_high": float(mean + z * sem),
    }


def ablated_sequence_logprob(
    model,
    tok,
    prompt: str,
    target_token_ids: Sequence[int],
    subset_indices: Sequence[int],
    d_ff: int,
) -> float:
    if not subset_indices:
        return float(run_prompt_sequence_score(model, tok, prompt, target_token_ids)["target_seq_logprob"])
    subset_map = flat_indices_to_layer_map(subset_indices, d_ff)
    ablator = LastTokenGateAblation(model, subset_map, ablate_all_positions=True)
    try:
        out = run_prompt_sequence_score(model, tok, prompt, target_token_ids)
    finally:
        ablator.close()
    return float(out["target_seq_logprob"])


def extract_minimal_circuit_for_item(
    model,
    tok,
    prompt: str,
    target_token_ids: Sequence[int],
    candidate_indices: Sequence[int],
    d_ff: int,
    target_ratio: float,
    max_size: int,
) -> Dict[str, object]:
    unique: List[int] = []
    seen = set()
    for idx in candidate_indices:
        iv = int(idx)
        if iv in seen:
            continue
        seen.add(iv)
        unique.append(iv)

    base_seq = float(run_prompt_sequence_score(model, tok, prompt, target_token_ids)["target_seq_logprob"])

    if not unique:
        return {
            "prompt": prompt,
            "full_signature_drop_seq_logprob": 0.0,
            "target_drop_seq_logprob": 0.0,
            "subset_drop_seq_logprob": 0.0,
            "recovery_ratio": 0.0,
            "subset_flat_indices": [],
            "subset_size": 0,
            "subset_layer_distribution": {},
            "single_neuron_effects_top10": [],
        }

    def _drop_for(indices: Sequence[int]) -> float:
        ablated = ablated_sequence_logprob(model, tok, prompt, target_token_ids, indices, d_ff)
        return float(base_seq - ablated)

    full_drop = _drop_for(unique)
    target_drop = max(0.0, float(full_drop) * max(0.0, min(1.0, target_ratio)))
    if full_drop <= 1e-12:
        return {
            "prompt": prompt,
            "full_signature_drop_seq_logprob": float(full_drop),
            "target_drop_seq_logprob": float(target_drop),
            "subset_drop_seq_logprob": 0.0,
            "recovery_ratio": 0.0,
            "subset_flat_indices": [],
            "subset_size": 0,
            "subset_layer_distribution": {},
            "single_neuron_effects_top10": [],
        }

    single_effects = []
    for idx in unique:
        single_effects.append((idx, _drop_for([idx])))
    single_effects.sort(key=lambda x: x[1], reverse=True)

    selected: List[int] = []
    selected_drop = 0.0
    for idx, _eff in single_effects:
        if len(selected) >= max(1, int(max_size)):
            break
        trial = selected + [idx]
        trial_drop = _drop_for(trial)
        if trial_drop > selected_drop + 1e-8:
            selected = trial
            selected_drop = trial_drop
        if selected_drop >= target_drop:
            break

    if not selected and single_effects:
        selected = [int(single_effects[0][0])]
        selected_drop = _drop_for(selected)

    pruned = True
    while pruned and len(selected) > 1 and selected_drop >= target_drop:
        pruned = False
        for idx in list(selected):
            trial = [x for x in selected if x != idx]
            trial_drop = _drop_for(trial)
            if trial_drop >= target_drop:
                selected = trial
                selected_drop = trial_drop
                pruned = True
                break

    recovery_ratio = float(selected_drop / max(float(full_drop), 1e-9))
    top10 = []
    for idx, eff in single_effects[:10]:
        li, ni = index_to_layer_neuron(int(idx), d_ff)
        top10.append(
            {
                "flat_index": int(idx),
                "layer": int(li),
                "neuron": int(ni),
                "single_drop_seq_logprob": float(eff),
            }
        )
    return {
        "prompt": prompt,
        "full_signature_drop_seq_logprob": float(full_drop),
        "target_drop_seq_logprob": float(target_drop),
        "subset_drop_seq_logprob": float(selected_drop),
        "recovery_ratio": float(recovery_ratio),
        "subset_flat_indices": [int(x) for x in selected],
        "subset_size": int(len(selected)),
        "subset_layer_distribution": layer_distribution(selected, d_ff),
        "single_neuron_effects_top10": top10,
    }


def pick_counterfactual_targets(base_item: Dict[str, object], all_candidates: List[Dict[str, object]]) -> List[Tuple[str, Dict[str, object]]]:
    noun = str(base_item["noun"])
    cat = str(base_item["category"])
    same = sorted(
        [c for c in all_candidates if c["noun"] != noun and c["category"] == cat],
        key=lambda x: str(x["noun"]),
    )
    diff = sorted(
        [c for c in all_candidates if c["noun"] != noun and c["category"] != cat],
        key=lambda x: str(x["noun"]),
    )
    out: List[Tuple[str, Dict[str, object]]] = []
    if same:
        out.append(("same_category", same[0]))
    if diff:
        out.append(("cross_category", diff[0]))
    return out


def run_causal_ablation_suite(
    model,
    tok,
    noun_names: List[str],
    noun_categories: List[str],
    signatures: List[np.ndarray],
    category_prototypes: Dict[str, Dict[str, object]],
    top_reused_records: List[Dict[str, int]],
    d_ff: int,
    total_neurons: int,
    ablation_top_k: int,
    ablation_random_trials: int,
    ablation_max_nouns: int,
    ablation_per_category_max: int,
    ablation_reused_top_k: int,
    ablation_eval_prompt_count: int,
    ablation_sample_strategy: str,
    minimal_circuit_max_nouns: int,
    minimal_circuit_target_ratio: float,
    minimal_circuit_max_size: int,
    counterfactual_max_pairs: int,
    seed: int,
) -> Dict[str, object]:
    rng = random.Random(seed + 131)

    candidates = []
    for i, noun in enumerate(noun_names):
        tok_ids = target_token_ids_for_noun(tok, noun)
        if not tok_ids:
            continue
        candidates.append(
            {
                "noun": noun,
                "category": noun_categories[i],
                "target_token_id": int(tok_ids[0]),
                "target_token_ids": [int(x) for x in tok_ids],
                "prompts": ablation_eval_prompts(noun)[: max(1, int(ablation_eval_prompt_count))],
                "signature": signatures[i],
            }
        )

    if not candidates:
        return {
            "enabled": True,
            "n_eligible_nouns": 0,
            "message": "No eligible nouns for token-level ablation score.",
            "records": [],
            "aggregate": {},
            "reused_ablation": {},
            "category_ablation": {},
            "minimal_circuit": {},
            "counterfactual_validation": {},
        }

    if ablation_max_nouns > 0 and len(candidates) > ablation_max_nouns:
        if ablation_sample_strategy == "head":
            sampled = list(candidates[:ablation_max_nouns])
        else:
            sampled = rng.sample(candidates, ablation_max_nouns)
    else:
        sampled = list(candidates)

    records = []
    for item in sampled:
        sig = np.asarray(item["signature"], dtype=np.int64)[:ablation_top_k]
        if sig.size == 0:
            continue
        sig_map = flat_indices_to_layer_map(sig.tolist(), d_ff)
        prompt_metrics = []
        sig_prob_drops = []
        rnd_prob_drops = []
        sig_logit_drops = []
        rnd_logit_drops = []
        sig_logprob_drops = []
        rnd_logprob_drops = []
        sig_rank_worse = []
        rnd_rank_worse = []
        sig_seq_logprob_drops = []
        rnd_seq_logprob_drops = []
        sig_seq_avg_logprob_drops = []
        rnd_seq_avg_logprob_drops = []

        for prompt in item["prompts"]:
            base = run_prompt_target_score(model, tok, prompt, item["target_token_id"])
            ablator = LastTokenGateAblation(model, sig_map)
            try:
                sig_res = run_prompt_target_score(model, tok, prompt, item["target_token_id"])
            finally:
                ablator.close()

            base_seq = run_prompt_sequence_score(model, tok, prompt, item["target_token_ids"])
            seq_ablator = LastTokenGateAblation(model, sig_map, ablate_all_positions=True)
            try:
                sig_seq = run_prompt_sequence_score(model, tok, prompt, item["target_token_ids"])
            finally:
                seq_ablator.close()

            rnd_results = []
            rnd_seq_results = []
            forbidden = set(int(x) for x in sig.tolist())
            for _ in range(max(1, ablation_random_trials)):
                rnd_idx = sample_random_indices(total_neurons, int(sig.size), forbidden, rng)
                rnd_map = flat_indices_to_layer_map(rnd_idx.tolist(), d_ff)
                rnd_ab = LastTokenGateAblation(model, rnd_map)
                try:
                    rnd = run_prompt_target_score(model, tok, prompt, item["target_token_id"])
                finally:
                    rnd_ab.close()
                rnd_results.append(rnd)

                rnd_seq_ab = LastTokenGateAblation(model, rnd_map, ablate_all_positions=True)
                try:
                    rnd_seq = run_prompt_sequence_score(model, tok, prompt, item["target_token_ids"])
                finally:
                    rnd_seq_ab.close()
                rnd_seq_results.append(rnd_seq)

            rnd_prob_mean = float(np.mean([r["target_prob"] for r in rnd_results]))
            rnd_logit_mean = float(np.mean([r["target_logit"] for r in rnd_results]))
            rnd_logprob_mean = float(np.mean([r["target_logprob"] for r in rnd_results]))
            rnd_rank_mean = float(np.mean([r["target_rank"] for r in rnd_results]))
            rnd_seq_logprob_mean = float(np.mean([r["target_seq_logprob"] for r in rnd_seq_results]))
            rnd_seq_avg_logprob_mean = float(np.mean([r["target_seq_avg_logprob"] for r in rnd_seq_results]))

            sig_prob_drop = base["target_prob"] - sig_res["target_prob"]
            rnd_prob_drop = base["target_prob"] - rnd_prob_mean
            sig_logit_drop = base["target_logit"] - sig_res["target_logit"]
            rnd_logit_drop = base["target_logit"] - rnd_logit_mean
            sig_logprob_drop = base["target_logprob"] - sig_res["target_logprob"]
            rnd_logprob_drop = base["target_logprob"] - rnd_logprob_mean
            sig_rank_delta = sig_res["target_rank"] - base["target_rank"]
            rnd_rank_delta = rnd_rank_mean - base["target_rank"]
            sig_seq_logprob_drop = base_seq["target_seq_logprob"] - sig_seq["target_seq_logprob"]
            rnd_seq_logprob_drop = base_seq["target_seq_logprob"] - rnd_seq_logprob_mean
            sig_seq_avg_logprob_drop = base_seq["target_seq_avg_logprob"] - sig_seq["target_seq_avg_logprob"]
            rnd_seq_avg_logprob_drop = base_seq["target_seq_avg_logprob"] - rnd_seq_avg_logprob_mean

            sig_prob_drops.append(sig_prob_drop)
            rnd_prob_drops.append(rnd_prob_drop)
            sig_logit_drops.append(sig_logit_drop)
            rnd_logit_drops.append(rnd_logit_drop)
            sig_logprob_drops.append(sig_logprob_drop)
            rnd_logprob_drops.append(rnd_logprob_drop)
            sig_rank_worse.append(sig_rank_delta)
            rnd_rank_worse.append(rnd_rank_delta)
            sig_seq_logprob_drops.append(sig_seq_logprob_drop)
            rnd_seq_logprob_drops.append(rnd_seq_logprob_drop)
            sig_seq_avg_logprob_drops.append(sig_seq_avg_logprob_drop)
            rnd_seq_avg_logprob_drops.append(rnd_seq_avg_logprob_drop)

            prompt_metrics.append(
                {
                    "prompt": prompt,
                    "baseline_target_prob": base["target_prob"],
                    "signature_ablate_target_prob": sig_res["target_prob"],
                    "random_ablate_target_prob_mean": rnd_prob_mean,
                    "signature_prob_drop": float(sig_prob_drop),
                    "random_prob_drop_mean": float(rnd_prob_drop),
                    "causal_margin_prob": float(sig_prob_drop - rnd_prob_drop),
                    "baseline_target_logprob": base["target_logprob"],
                    "signature_ablate_target_logprob": sig_res["target_logprob"],
                    "random_ablate_target_logprob_mean": rnd_logprob_mean,
                    "signature_logprob_drop": float(sig_logprob_drop),
                    "random_logprob_drop_mean": float(rnd_logprob_drop),
                    "causal_margin_logprob": float(sig_logprob_drop - rnd_logprob_drop),
                    "baseline_target_rank": float(base["target_rank"]),
                    "signature_ablate_target_rank": float(sig_res["target_rank"]),
                    "random_ablate_target_rank_mean": float(rnd_rank_mean),
                    "signature_rank_worse": float(sig_rank_delta),
                    "random_rank_worse_mean": float(rnd_rank_delta),
                    "causal_margin_rank_worse": float(sig_rank_delta - rnd_rank_delta),
                    "baseline_target_seq_logprob": base_seq["target_seq_logprob"],
                    "signature_ablate_target_seq_logprob": sig_seq["target_seq_logprob"],
                    "random_ablate_target_seq_logprob_mean": rnd_seq_logprob_mean,
                    "signature_seq_logprob_drop": float(sig_seq_logprob_drop),
                    "random_seq_logprob_drop_mean": float(rnd_seq_logprob_drop),
                    "causal_margin_seq_logprob": float(sig_seq_logprob_drop - rnd_seq_logprob_drop),
                    "baseline_target_seq_avg_logprob": base_seq["target_seq_avg_logprob"],
                    "signature_ablate_target_seq_avg_logprob": sig_seq["target_seq_avg_logprob"],
                    "random_ablate_target_seq_avg_logprob_mean": rnd_seq_avg_logprob_mean,
                    "signature_seq_avg_logprob_drop": float(sig_seq_avg_logprob_drop),
                    "random_seq_avg_logprob_drop_mean": float(rnd_seq_avg_logprob_drop),
                    "causal_margin_seq_avg_logprob": float(sig_seq_avg_logprob_drop - rnd_seq_avg_logprob_drop),
                }
            )

        sig_prob_drop = float(np.mean(sig_prob_drops)) if sig_prob_drops else 0.0
        rnd_prob_drop = float(np.mean(rnd_prob_drops)) if rnd_prob_drops else 0.0
        sig_logit_drop = float(np.mean(sig_logit_drops)) if sig_logit_drops else 0.0
        rnd_logit_drop = float(np.mean(rnd_logit_drops)) if rnd_logit_drops else 0.0
        sig_logprob_drop = float(np.mean(sig_logprob_drops)) if sig_logprob_drops else 0.0
        rnd_logprob_drop = float(np.mean(rnd_logprob_drops)) if rnd_logprob_drops else 0.0
        sig_rank_worse_mean = float(np.mean(sig_rank_worse)) if sig_rank_worse else 0.0
        rnd_rank_worse_mean = float(np.mean(rnd_rank_worse)) if rnd_rank_worse else 0.0
        sig_seq_logprob_drop = float(np.mean(sig_seq_logprob_drops)) if sig_seq_logprob_drops else 0.0
        rnd_seq_logprob_drop = float(np.mean(rnd_seq_logprob_drops)) if rnd_seq_logprob_drops else 0.0
        sig_seq_avg_logprob_drop = float(np.mean(sig_seq_avg_logprob_drops)) if sig_seq_avg_logprob_drops else 0.0
        rnd_seq_avg_logprob_drop = float(np.mean(rnd_seq_avg_logprob_drops)) if rnd_seq_avg_logprob_drops else 0.0
        records.append(
            {
                "noun": item["noun"],
                "category": item["category"],
                "prompt_count": len(item["prompts"]),
                "prompt_metrics": prompt_metrics,
                "target_token_id": item["target_token_id"],
                "ablation_size": int(sig.size),
                "signature_prob_drop": float(sig_prob_drop),
                "random_prob_drop_mean": float(rnd_prob_drop),
                "causal_margin_prob": float(sig_prob_drop - rnd_prob_drop),
                "signature_logit_drop": float(sig_logit_drop),
                "random_logit_drop_mean": float(rnd_logit_drop),
                "causal_margin_logit": float(sig_logit_drop - rnd_logit_drop),
                "signature_logprob_drop": float(sig_logprob_drop),
                "random_logprob_drop_mean": float(rnd_logprob_drop),
                "causal_margin_logprob": float(sig_logprob_drop - rnd_logprob_drop),
                "signature_rank_worse": float(sig_rank_worse_mean),
                "random_rank_worse_mean": float(rnd_rank_worse_mean),
                "causal_margin_rank_worse": float(sig_rank_worse_mean - rnd_rank_worse_mean),
                "signature_seq_logprob_drop": float(sig_seq_logprob_drop),
                "random_seq_logprob_drop_mean": float(rnd_seq_logprob_drop),
                "causal_margin_seq_logprob": float(sig_seq_logprob_drop - rnd_seq_logprob_drop),
                "signature_seq_avg_logprob_drop": float(sig_seq_avg_logprob_drop),
                "random_seq_avg_logprob_drop_mean": float(rnd_seq_avg_logprob_drop),
                "causal_margin_seq_avg_logprob": float(sig_seq_avg_logprob_drop - rnd_seq_avg_logprob_drop),
            }
        )

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    margin_prob_values = [r["causal_margin_prob"] for r in records]
    margin_logit_values = [r["causal_margin_logit"] for r in records]
    margin_logprob_values = [r.get("causal_margin_logprob", 0.0) for r in records]
    margin_rank_values = [r.get("causal_margin_rank_worse", 0.0) for r in records]
    margin_seq_logprob_values = [r.get("causal_margin_seq_logprob", 0.0) for r in records]
    margin_seq_avg_logprob_values = [r.get("causal_margin_seq_avg_logprob", 0.0) for r in records]
    margin_prob_stats = mean_confidence_interval(margin_prob_values)
    margin_logit_stats = mean_confidence_interval(margin_logit_values)
    margin_logprob_stats = mean_confidence_interval(margin_logprob_values)
    margin_rank_stats = mean_confidence_interval(margin_rank_values)
    margin_seq_logprob_stats = mean_confidence_interval(margin_seq_logprob_values)
    margin_seq_avg_logprob_stats = mean_confidence_interval(margin_seq_avg_logprob_values)

    aggregate = {
        "n_tested_nouns": int(len(records)),
        "mean_signature_prob_drop": _mean([r["signature_prob_drop"] for r in records]),
        "mean_random_prob_drop": _mean([r["random_prob_drop_mean"] for r in records]),
        "mean_causal_margin_prob": _mean([r["causal_margin_prob"] for r in records]),
        "positive_causal_margin_ratio": float(
            np.mean([1.0 if r["causal_margin_prob"] > 0 else 0.0 for r in records]) if records else 0.0
        ),
        "mean_signature_logit_drop": _mean([r["signature_logit_drop"] for r in records]),
        "mean_random_logit_drop": _mean([r["random_logit_drop_mean"] for r in records]),
        "mean_causal_margin_logit": _mean([r["causal_margin_logit"] for r in records]),
        "mean_signature_logprob_drop": _mean([r.get("signature_logprob_drop", 0.0) for r in records]),
        "mean_random_logprob_drop": _mean([r.get("random_logprob_drop_mean", 0.0) for r in records]),
        "mean_causal_margin_logprob": _mean([r.get("causal_margin_logprob", 0.0) for r in records]),
        "mean_signature_rank_worse": _mean([r.get("signature_rank_worse", 0.0) for r in records]),
        "mean_random_rank_worse": _mean([r.get("random_rank_worse_mean", 0.0) for r in records]),
        "mean_causal_margin_rank_worse": _mean([r.get("causal_margin_rank_worse", 0.0) for r in records]),
        "mean_signature_seq_logprob_drop": _mean([r.get("signature_seq_logprob_drop", 0.0) for r in records]),
        "mean_random_seq_logprob_drop": _mean([r.get("random_seq_logprob_drop_mean", 0.0) for r in records]),
        "mean_causal_margin_seq_logprob": _mean([r.get("causal_margin_seq_logprob", 0.0) for r in records]),
        "mean_signature_seq_avg_logprob_drop": _mean([r.get("signature_seq_avg_logprob_drop", 0.0) for r in records]),
        "mean_random_seq_avg_logprob_drop": _mean([r.get("random_seq_avg_logprob_drop_mean", 0.0) for r in records]),
        "mean_causal_margin_seq_avg_logprob": _mean([r.get("causal_margin_seq_avg_logprob", 0.0) for r in records]),
        "causal_margin_prob_stats": margin_prob_stats,
        "causal_margin_logit_stats": margin_logit_stats,
        "causal_margin_logprob_stats": margin_logprob_stats,
        "causal_margin_rank_worse_stats": margin_rank_stats,
        "causal_margin_seq_logprob_stats": margin_seq_logprob_stats,
        "causal_margin_seq_avg_logprob_stats": margin_seq_avg_logprob_stats,
        "causal_margin_prob_z": float(
            margin_prob_stats["mean"] / max(margin_prob_stats["sem"], 1e-9) if margin_prob_stats["sem"] > 0 else 0.0
        ),
        "causal_margin_logit_z": float(
            margin_logit_stats["mean"] / max(margin_logit_stats["sem"], 1e-9) if margin_logit_stats["sem"] > 0 else 0.0
        ),
        "causal_margin_seq_logprob_z": float(
            margin_seq_logprob_stats["mean"] / max(margin_seq_logprob_stats["sem"], 1e-9)
            if margin_seq_logprob_stats["sem"] > 0
            else 0.0
        ),
        "causal_margin_seq_avg_logprob_z": float(
            margin_seq_avg_logprob_stats["mean"] / max(margin_seq_avg_logprob_stats["sem"], 1e-9)
            if margin_seq_avg_logprob_stats["sem"] > 0
            else 0.0
        ),
    }

    reused_indices = [int(x["flat_index"]) for x in top_reused_records[:ablation_reused_top_k]]
    reused_ablation = {}
    if reused_indices and records:
        reused_map = flat_indices_to_layer_map(reused_indices, d_ff)
        rnd_drops = []
        reused_drops = []
        reused_size = len(reused_indices)
        for rec in records:
            eval_prompt = ablation_eval_prompts(rec["noun"])[0]
            base = run_prompt_target_score(model, tok, eval_prompt, rec["target_token_id"])
            base_prob = base["target_prob"]
            reused_ab = LastTokenGateAblation(model, reused_map)
            try:
                reused_res = run_prompt_target_score(model, tok, eval_prompt, rec["target_token_id"])
            finally:
                reused_ab.close()
            reused_drop = base_prob - reused_res["target_prob"]
            reused_drops.append(reused_drop)

            rnd_local = []
            for _ in range(max(1, ablation_random_trials)):
                rnd_idx = sample_random_indices(total_neurons, reused_size, set(reused_indices), rng)
                rnd_map = flat_indices_to_layer_map(rnd_idx.tolist(), d_ff)
                rnd_ab = LastTokenGateAblation(model, rnd_map)
                try:
                    rnd = run_prompt_target_score(model, tok, eval_prompt, rec["target_token_id"])
                finally:
                    rnd_ab.close()
                rnd_local.append(base_prob - rnd["target_prob"])
            rnd_drops.append(float(np.mean(rnd_local)))

        reused_ablation = {
            "ablation_size": reused_size,
            "n_tested_nouns": len(records),
            "mean_reused_prob_drop": _mean(reused_drops),
            "mean_random_prob_drop": _mean(rnd_drops),
            "mean_causal_margin_prob": _mean(
                [reused_drops[i] - rnd_drops[i] for i in range(min(len(reused_drops), len(rnd_drops)))]
            ),
        }

    category_ablation = {}
    if records and category_prototypes:
        rec_by_cat = defaultdict(list)
        for rec in records:
            rec_by_cat[rec["category"]].append(rec)

        for cat, recs in rec_by_cat.items():
            proto = category_prototypes.get(cat, {})
            proto_idx = np.asarray(proto.get("prototype_top_indices", []), dtype=np.int64)[:ablation_top_k]
            if proto_idx.size == 0:
                continue
            use_recs = recs[:ablation_per_category_max] if ablation_per_category_max > 0 else recs
            proto_map = flat_indices_to_layer_map(proto_idx.tolist(), d_ff)
            proto_drops = []
            rnd_drops = []
            forbidden = set(int(x) for x in proto_idx.tolist())
            for rec in use_recs:
                eval_prompt = ablation_eval_prompts(rec["noun"])[0]
                base = run_prompt_target_score(model, tok, eval_prompt, rec["target_token_id"])
                base_prob = base["target_prob"]
                proto_ab = LastTokenGateAblation(model, proto_map)
                try:
                    proto_res = run_prompt_target_score(model, tok, eval_prompt, rec["target_token_id"])
                finally:
                    proto_ab.close()
                proto_drops.append(base_prob - proto_res["target_prob"])

                rnd_local = []
                for _ in range(max(1, ablation_random_trials)):
                    rnd_idx = sample_random_indices(total_neurons, int(proto_idx.size), forbidden, rng)
                    rnd_map = flat_indices_to_layer_map(rnd_idx.tolist(), d_ff)
                    rnd_ab = LastTokenGateAblation(model, rnd_map)
                    try:
                        rnd = run_prompt_target_score(model, tok, eval_prompt, rec["target_token_id"])
                    finally:
                        rnd_ab.close()
                    rnd_local.append(base_prob - rnd["target_prob"])
                rnd_drops.append(float(np.mean(rnd_local)))

            category_ablation[cat] = {
                "n_tested_nouns": len(use_recs),
                "ablation_size": int(proto_idx.size),
                "mean_category_prob_drop": _mean(proto_drops),
                "mean_random_prob_drop": _mean(rnd_drops),
                "mean_causal_margin_prob": _mean(
                    [proto_drops[i] - rnd_drops[i] for i in range(min(len(proto_drops), len(rnd_drops)))]
                ),
            }

    minimal_circuit_records = []
    sampled_for_circuit = sampled[: minimal_circuit_max_nouns] if minimal_circuit_max_nouns > 0 else sampled
    sampled_by_noun = {str(x["noun"]): x for x in sampled}
    for item in sampled_for_circuit:
        sig = np.asarray(item["signature"], dtype=np.int64)[:ablation_top_k]
        if sig.size == 0:
            continue
        prompt = item["prompts"][0] if item["prompts"] else ablation_eval_prompts(item["noun"])[0]
        rec = extract_minimal_circuit_for_item(
            model=model,
            tok=tok,
            prompt=prompt,
            target_token_ids=item["target_token_ids"],
            candidate_indices=sig.tolist(),
            d_ff=d_ff,
            target_ratio=minimal_circuit_target_ratio,
            max_size=minimal_circuit_max_size,
        )
        rec["noun"] = str(item["noun"])
        rec["category"] = str(item["category"])
        minimal_circuit_records.append(rec)

    minimal_circuit = {
        "enabled": bool(minimal_circuit_records),
        "n_tested_nouns": int(len(minimal_circuit_records)),
        "target_ratio": float(minimal_circuit_target_ratio),
        "records": minimal_circuit_records,
        "aggregate": {
            "mean_subset_size": _mean([float(r.get("subset_size", 0)) for r in minimal_circuit_records]),
            "mean_recovery_ratio": _mean([float(r.get("recovery_ratio", 0.0)) for r in minimal_circuit_records]),
            "mean_full_signature_drop_seq_logprob": _mean(
                [float(r.get("full_signature_drop_seq_logprob", 0.0)) for r in minimal_circuit_records]
            ),
            "mean_subset_drop_seq_logprob": _mean(
                [float(r.get("subset_drop_seq_logprob", 0.0)) for r in minimal_circuit_records]
            ),
            "high_recovery_ratio": float(
                np.mean([1.0 if float(r.get("recovery_ratio", 0.0)) >= minimal_circuit_target_ratio else 0.0 for r in minimal_circuit_records])
                if minimal_circuit_records
                else 0.0
            ),
        },
    }

    counterfactual_records = []
    if minimal_circuit_records:
        used = 0
        for rec in minimal_circuit_records:
            if counterfactual_max_pairs > 0 and used >= counterfactual_max_pairs:
                break
            noun = str(rec.get("noun", ""))
            base_item = sampled_by_noun.get(noun)
            if not base_item:
                continue
            subset = [int(x) for x in rec.get("subset_flat_indices", [])]
            if not subset:
                continue
            prompt = str(rec.get("prompt") or (base_item.get("prompts") or [ablation_eval_prompts(noun)[0]])[0])
            base_base = float(run_prompt_sequence_score(model, tok, prompt, base_item["target_token_ids"])["target_seq_logprob"])
            base_ab = ablated_sequence_logprob(model, tok, prompt, base_item["target_token_ids"], subset, d_ff)
            base_drop = float(base_base - base_ab)
            for relation, cf_item in pick_counterfactual_targets(base_item, candidates):
                cf_base = float(run_prompt_sequence_score(model, tok, prompt, cf_item["target_token_ids"])["target_seq_logprob"])
                cf_ab = ablated_sequence_logprob(model, tok, prompt, cf_item["target_token_ids"], subset, d_ff)
                cf_drop = float(cf_base - cf_ab)
                counterfactual_records.append(
                    {
                        "noun": noun,
                        "category": str(base_item["category"]),
                        "counterfactual_noun": str(cf_item["noun"]),
                        "counterfactual_category": str(cf_item["category"]),
                        "relation": relation,
                        "prompt": prompt,
                        "subset_size": int(len(subset)),
                        "base_drop_seq_logprob": float(base_drop),
                        "counterfactual_drop_seq_logprob": float(cf_drop),
                        "specificity_margin_seq_logprob": float(base_drop - cf_drop),
                    }
                )
                used += 1
                if counterfactual_max_pairs > 0 and used >= counterfactual_max_pairs:
                    break
            if counterfactual_max_pairs > 0 and used >= counterfactual_max_pairs:
                break

    same_margins = [float(r["specificity_margin_seq_logprob"]) for r in counterfactual_records if r["relation"] == "same_category"]
    cross_margins = [float(r["specificity_margin_seq_logprob"]) for r in counterfactual_records if r["relation"] == "cross_category"]
    all_margins = [float(r["specificity_margin_seq_logprob"]) for r in counterfactual_records]
    all_margin_stats = mean_confidence_interval(all_margins)
    counterfactual_validation = {
        "enabled": bool(counterfactual_records),
        "n_pairs": int(len(counterfactual_records)),
        "records": counterfactual_records,
        "aggregate": {
            "mean_specificity_margin_seq_logprob": _mean(all_margins),
            "mean_same_category_margin_seq_logprob": _mean(same_margins),
            "mean_cross_category_margin_seq_logprob": _mean(cross_margins),
            "positive_specificity_ratio": float(np.mean([1.0 if x > 0 else 0.0 for x in all_margins]) if all_margins else 0.0),
            "specificity_margin_stats": all_margin_stats,
            "specificity_margin_z": float(
                all_margin_stats["mean"] / max(all_margin_stats["sem"], 1e-9) if all_margin_stats["sem"] > 0 else 0.0
            ),
        },
    }

    return {
        "enabled": True,
        "n_eligible_nouns": len(candidates),
        "n_sampled_nouns": len(sampled),
        "records": records,
        "aggregate": aggregate,
        "reused_ablation": reused_ablation,
        "category_ablation": dict(sorted(category_ablation.items())),
        "minimal_circuit": minimal_circuit,
        "counterfactual_validation": counterfactual_validation,
        "config": {
            "ablation_top_k": int(ablation_top_k),
            "ablation_random_trials": int(ablation_random_trials),
            "ablation_max_nouns": int(ablation_max_nouns),
            "ablation_per_category_max": int(ablation_per_category_max),
            "ablation_reused_top_k": int(ablation_reused_top_k),
            "ablation_eval_prompt_count": int(ablation_eval_prompt_count),
            "ablation_sample_strategy": str(ablation_sample_strategy),
            "minimal_circuit_max_nouns": int(minimal_circuit_max_nouns),
            "minimal_circuit_target_ratio": float(minimal_circuit_target_ratio),
            "minimal_circuit_max_size": int(minimal_circuit_max_size),
            "counterfactual_max_pairs": int(counterfactual_max_pairs),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Mass noun neuron encoding scan")
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--nouns-file", default="", help="Optional CSV-like lines: noun,category")
    parser.add_argument("--max-nouns", type=int, default=0)
    parser.add_argument("--top-signature-k", type=int, default=120)
    parser.add_argument("--reuse-threshold", type=int, default=5)
    parser.add_argument("--run-causal-ablation", action="store_true", default=False)
    parser.add_argument("--ablation-top-k", type=int, default=24)
    parser.add_argument("--ablation-random-trials", type=int, default=3)
    parser.add_argument("--ablation-max-nouns", type=int, default=24)
    parser.add_argument("--ablation-per-category-max", type=int, default=6)
    parser.add_argument("--ablation-reused-top-k", type=int, default=24)
    parser.add_argument("--ablation-eval-prompt-count", type=int, default=3)
    parser.add_argument("--ablation-sample-strategy", choices=["random", "head"], default="random")
    parser.add_argument("--minimal-circuit-max-nouns", type=int, default=8)
    parser.add_argument("--minimal-circuit-target-ratio", type=float, default=0.8)
    parser.add_argument("--minimal-circuit-max-size", type=int, default=12)
    parser.add_argument("--counterfactual-max-pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_noun_scan_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    nouns = load_nouns(args.nouns_file or None, args.max_nouns if args.max_nouns > 0 else None)
    if len(nouns) < 2:
        raise ValueError("Need at least 2 nouns to run comparison.")

    t0 = time.time()
    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    collector = GateCollector(model)
    try:
        n_layers = len(model.model.layers)
        d_ff = model.model.layers[0].mlp.gate_proj.out_features
        total_neurons = n_layers * d_ff

        sums = {x.noun: np.zeros(total_neurons, dtype=np.float64) for x in nouns}
        counts = {x.noun: 0 for x in nouns}
        categories = {x.noun: x.category for x in nouns}

        for it in nouns:
            for p in noun_prompts(it.noun):
                collector.reset()
                _ = run_prompt(model, tok, p)
                flat = collector.get_flat()
                sums[it.noun] += flat
                counts[it.noun] += 1

        noun_names = [x.noun for x in nouns]
        noun_categories = [x.category for x in nouns]
        mat = np.zeros((len(nouns), total_neurons), dtype=np.float32)
        for i, nm in enumerate(noun_names):
            mat[i] = (sums[nm] / max(counts[nm], 1)).astype(np.float32)

        mean_vec = mat.mean(axis=0)
        std_vec = mat.std(axis=0) + 1e-8
        zmat = (mat - mean_vec) / std_vec

        signatures: List[np.ndarray] = []
        noun_records = []
        layer_usage = np.zeros(n_layers, dtype=np.int64)
        all_signature_indices: List[int] = []

        for i, nm in enumerate(noun_names):
            sig = topk_indices(zmat[i], args.top_signature_k)
            signatures.append(sig)
            all_signature_indices.extend(sig.tolist())
            ld = layer_distribution(sig, d_ff)
            for li, c in ld.items():
                layer_usage[li] += c
            centroid = float(np.mean([index_to_layer_neuron(int(x), d_ff)[0] for x in sig])) if len(sig) else -1.0
            noun_records.append(
                {
                    "noun": nm,
                    "category": categories[nm],
                    "signature_size": int(len(sig)),
                    "signature_top_indices": [int(x) for x in sig.tolist()],
                    "signature_layer_distribution": ld,
                    "signature_layer_centroid": centroid,
                    "mean_activation": float(mat[i].mean()),
                    "l2_norm": float(np.linalg.norm(mat[i])),
                }
            )

        reuse_counter = Counter(int(x) for x in all_signature_indices)
        reused = [idx for idx, c in reuse_counter.items() if c >= args.reuse_threshold]
        top_reused = sorted(reuse_counter.items(), key=lambda x: x[1], reverse=True)[:100]

        cat_to_rows = defaultdict(list)
        for i, cat in enumerate(noun_categories):
            cat_to_rows[cat].append(i)
        category_prototypes = {}
        for cat, rows in cat_to_rows.items():
            proto = mat[rows].mean(axis=0)
            proto_z = (proto - mean_vec) / std_vec
            top_idx = topk_indices(proto_z, args.top_signature_k)
            category_prototypes[cat] = {
                "n_nouns": len(rows),
                "prototype_top_indices": [int(x) for x in top_idx.tolist()],
                "prototype_layer_distribution": layer_distribution(top_idx, d_ff),
            }

        pair_stats = pairwise_similarity(mat, signatures, noun_categories)
        rank_stats = low_rank_stats(mat)

        regularity = {
            "layer_usage_distribution": {int(i): int(v) for i, v in enumerate(layer_usage.tolist()) if v > 0},
            "layer_usage_entropy_norm": normalized_entropy(layer_usage.astype(np.float64)),
            "top3_layer_usage_ratio": float(np.sort(layer_usage)[-3:].sum() / max(layer_usage.sum(), 1)),
            "reused_neuron_count": int(len(reused)),
            "reused_neuron_ratio": float(len(reused) / max(total_neurons, 1)),
            "reuse_threshold": int(args.reuse_threshold),
            "pairwise_similarity": pair_stats,
            "low_rank_structure": rank_stats,
        }

        top_reused_records = []
        for idx, c in top_reused:
            li, ni = index_to_layer_neuron(idx, d_ff)
            top_reused_records.append(
                {
                    "flat_index": int(idx),
                    "layer": int(li),
                    "neuron": int(ni),
                    "count": int(c),
                }
            )

        causal_ablation = {
            "enabled": False,
            "message": "Skipped. Use --run-causal-ablation to enable.",
        }
        if args.run_causal_ablation:
            causal_ablation = run_causal_ablation_suite(
                model=model,
                tok=tok,
                noun_names=noun_names,
                noun_categories=noun_categories,
                signatures=signatures,
                category_prototypes=category_prototypes,
                top_reused_records=top_reused_records,
                d_ff=d_ff,
                total_neurons=total_neurons,
                ablation_top_k=args.ablation_top_k,
                ablation_random_trials=args.ablation_random_trials,
                ablation_max_nouns=args.ablation_max_nouns,
                ablation_per_category_max=args.ablation_per_category_max,
                ablation_reused_top_k=args.ablation_reused_top_k,
                ablation_eval_prompt_count=args.ablation_eval_prompt_count,
                ablation_sample_strategy=args.ablation_sample_strategy,
                minimal_circuit_max_nouns=args.minimal_circuit_max_nouns,
                minimal_circuit_target_ratio=args.minimal_circuit_target_ratio,
                minimal_circuit_max_size=args.minimal_circuit_max_size,
                counterfactual_max_pairs=args.counterfactual_max_pairs,
                seed=args.seed,
            )

        mechanism_scorecard = build_mechanism_scorecard(regularity=regularity, causal_ablation=causal_ablation, n_nouns=len(nouns))

        result = {
            "model_id": args.model_id,
            "device": str(next(model.parameters()).device),
            "runtime_sec": float(time.time() - t0),
            "config": {
                "n_nouns": len(nouns),
                "n_layers": n_layers,
                "d_ff": d_ff,
                "total_neurons": total_neurons,
                "top_signature_k": args.top_signature_k,
                "reuse_threshold": args.reuse_threshold,
            },
            "noun_records": noun_records,
            "category_prototypes": dict(sorted(category_prototypes.items())),
            "top_reused_neurons": top_reused_records,
            "regularity": regularity,
            "causal_ablation": causal_ablation,
            "mechanism_scorecard": mechanism_scorecard,
        }

        json_path = out_dir / "mass_noun_encoding_scan.json"
        md_path = out_dir / "MASS_NOUN_ENCODING_SCAN_REPORT.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = [
            "# Mass Noun Encoding Scan Report",
            "",
            "## Core Findings",
            f"- Nouns scanned: {len(nouns)}",
            f"- Neuron space: {total_neurons} ({n_layers} layers x {d_ff} d_ff)",
            f"- Reused neurons (>= {args.reuse_threshold} nouns): {regularity['reused_neuron_count']} ({regularity['reused_neuron_ratio']:.6f})",
            f"- Layer usage entropy (normalized): {regularity['layer_usage_entropy_norm']:.4f}",
            f"- Top-3 layer usage ratio: {regularity['top3_layer_usage_ratio']:.4f}",
            "",
            "## Pairwise Structure",
            f"- Within-category cosine mean: {pair_stats['within_category_cosine_mean']:.4f}",
            f"- Between-category cosine mean: {pair_stats['between_category_cosine_mean']:.4f}",
            f"- Cosine gap: {pair_stats['cosine_gap']:.4f}",
            f"- Within-category Jaccard mean: {pair_stats['within_category_jaccard_mean']:.4f}",
            f"- Between-category Jaccard mean: {pair_stats['between_category_jaccard_mean']:.4f}",
            f"- Jaccard gap: {pair_stats['jaccard_gap']:.4f}",
            "",
            "## Low-Rank Encoding",
            f"- Participation ratio: {rank_stats['participation_ratio']:.4f}",
            f"- Top-1 energy ratio: {rank_stats['top1_energy_ratio']:.4f}",
            f"- Top-5 energy ratio: {rank_stats['top5_energy_ratio']:.4f}",
            "",
            "## Mechanism Scorecard",
            f"- Overall score: {mechanism_scorecard.get('overall_score', 0.0):.4f}",
            f"- Grade: {mechanism_scorecard.get('grade', 'unknown')}",
            f"- Structure separation: {mechanism_scorecard.get('subscores', {}).get('structure_separation', 0.0):.4f}",
            f"- Reuse sparsity structure: {mechanism_scorecard.get('subscores', {}).get('reuse_sparsity_structure', 0.0):.4f}",
            f"- Low-rank compactness: {mechanism_scorecard.get('subscores', {}).get('low_rank_compactness', 0.0):.4f}",
            f"- Causal evidence: {mechanism_scorecard.get('subscores', {}).get('causal_evidence', 0.0):.4f}",
            "",
            "### Guidance",
        ]
        for g in mechanism_scorecard.get("guidance", []):
            lines.append(f"- {g}")

        lines.extend(
            [
                "",
            "## Causal Ablation",
            ]
        )
        if causal_ablation.get("enabled"):
            agg = causal_ablation.get("aggregate", {})
            lines.extend(
                [
                    f"- Eligible nouns: {causal_ablation.get('n_eligible_nouns', 0)}",
                    f"- Sampled nouns: {causal_ablation.get('n_sampled_nouns', 0)}",
                    f"- Mean signature prob drop: {agg.get('mean_signature_prob_drop', 0.0):.6f}",
                    f"- Mean random prob drop: {agg.get('mean_random_prob_drop', 0.0):.6f}",
                    f"- Mean causal margin (prob): {agg.get('mean_causal_margin_prob', 0.0):.6f}",
                    f"- Mean causal margin (logprob): {agg.get('mean_causal_margin_logprob', 0.0):.6f}",
                    f"- Mean causal margin (rank-worse): {agg.get('mean_causal_margin_rank_worse', 0.0):.6f}",
                    f"- Mean causal margin (seq logprob): {agg.get('mean_causal_margin_seq_logprob', 0.0):.6f}",
                    f"- Mean causal margin (seq avg logprob): {agg.get('mean_causal_margin_seq_avg_logprob', 0.0):.6f}",
                    f"- Positive causal margin ratio: {agg.get('positive_causal_margin_ratio', 0.0):.4f}",
                    f"- Causal margin prob z-score: {agg.get('causal_margin_prob_z', 0.0):.4f}",
                    f"- Causal margin seq-logprob z-score: {agg.get('causal_margin_seq_logprob_z', 0.0):.4f}",
                ]
            )
            prob_ci = (agg.get("causal_margin_prob_stats") or {})
            lines.append(
                f"- Causal margin prob 95% CI: [{prob_ci.get('ci_low', 0.0):.6f}, {prob_ci.get('ci_high', 0.0):.6f}]"
            )
            seq_ci = (agg.get("causal_margin_seq_logprob_stats") or {})
            lines.append(
                f"- Causal margin seq-logprob 95% CI: [{seq_ci.get('ci_low', 0.0):.6f}, {seq_ci.get('ci_high', 0.0):.6f}]"
            )
            reused_stats = causal_ablation.get("reused_ablation") or {}
            if reused_stats:
                lines.extend(
                    [
                        "- Reused neuron ablation:",
                        f"  - mean reused prob drop: {reused_stats.get('mean_reused_prob_drop', 0.0):.6f}",
                        f"  - mean random prob drop: {reused_stats.get('mean_random_prob_drop', 0.0):.6f}",
                        f"  - mean causal margin: {reused_stats.get('mean_causal_margin_prob', 0.0):.6f}",
                    ]
                )
            min_circuit = causal_ablation.get("minimal_circuit") or {}
            if min_circuit.get("enabled"):
                min_agg = min_circuit.get("aggregate") or {}
                lines.extend(
                    [
                        "- Minimal circuit extraction:",
                        f"  - tested nouns: {min_circuit.get('n_tested_nouns', 0)}",
                        f"  - target ratio: {min_circuit.get('target_ratio', 0.0):.2f}",
                        f"  - mean subset size: {min_agg.get('mean_subset_size', 0.0):.4f}",
                        f"  - mean recovery ratio: {min_agg.get('mean_recovery_ratio', 0.0):.4f}",
                        f"  - high recovery ratio: {min_agg.get('high_recovery_ratio', 0.0):.4f}",
                    ]
                )
            cf_val = causal_ablation.get("counterfactual_validation") or {}
            if cf_val.get("enabled"):
                cf_agg = cf_val.get("aggregate") or {}
                cf_stats = cf_agg.get("specificity_margin_stats") or {}
                lines.extend(
                    [
                        "- Counterfactual validation:",
                        f"  - tested pairs: {cf_val.get('n_pairs', 0)}",
                        f"  - mean specificity margin (seq-logprob): {cf_agg.get('mean_specificity_margin_seq_logprob', 0.0):.6f}",
                        f"  - same-category margin: {cf_agg.get('mean_same_category_margin_seq_logprob', 0.0):.6f}",
                        f"  - cross-category margin: {cf_agg.get('mean_cross_category_margin_seq_logprob', 0.0):.6f}",
                        f"  - positive specificity ratio: {cf_agg.get('positive_specificity_ratio', 0.0):.4f}",
                        f"  - specificity z-score: {cf_agg.get('specificity_margin_z', 0.0):.4f}",
                        f"  - specificity 95% CI: [{cf_stats.get('ci_low', 0.0):.6f}, {cf_stats.get('ci_high', 0.0):.6f}]",
                    ]
                )
        else:
            lines.append(f"- {causal_ablation.get('message', 'Skipped')}")

        lines.extend(
            [
                "",
            "## Top Reused Neurons (Top-20)",
            ]
        )
        for rec in top_reused_records[:20]:
            lines.append(f"- L{rec['layer']}N{rec['neuron']}: used by {rec['count']} noun signatures")
        md_path.write_text("\n".join(lines), encoding="utf-8")

        print(f"[OK] Saved: {out_dir}")
        print(f"[OK] JSON: {json_path}")
        print(f"[OK] Report: {md_path}")
    finally:
        collector.close()


if __name__ == "__main__":
    main()

