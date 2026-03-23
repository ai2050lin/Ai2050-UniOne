#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage119: GPT-2 词嵌入全词有效编码扫描。

目标：
1. 扫描 GPT-2 词表中的干净英文词，建立可复用的词级编码基线。
2. 为每个词输出几何强度、主成分残差、语义家族对齐与有效编码分数。
3. 为后续 apple（苹果）类个案分析提供统一坐标系，而不是继续停留在单点观察。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from safetensors import safe_open
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(
    r"d:\develop\model\hub\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
)
SEED_CSV = PROJECT_ROOT / "tests" / "codex" / "deepseek7b_nouns_english_520_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage119_gpt2_embedding_full_vocab_scan_20260323"

WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]{1,29}$")
EPS = 1e-12

MICRO_GROUPS = {
    "micro_color": [
        "red",
        "green",
        "blue",
        "yellow",
        "black",
        "white",
        "purple",
        "brown",
        "pink",
        "orange",
        "gray",
        "gold",
        "silver",
    ],
    "micro_taste": [
        "sweet",
        "sour",
        "bitter",
        "salty",
        "spicy",
        "savory",
        "fresh",
        "juicy",
        "ripe",
        "dry",
    ],
    "micro_shape": [
        "round",
        "square",
        "sharp",
        "smooth",
        "rough",
        "flat",
        "curved",
        "thin",
        "thick",
    ],
    "micro_size": [
        "small",
        "large",
        "big",
        "tiny",
        "huge",
        "short",
        "tall",
        "long",
        "wide",
        "narrow",
        "heavy",
        "light",
    ],
    "micro_temperature": [
        "hot",
        "warm",
        "cold",
        "cool",
        "frozen",
        "burning",
    ],
    "micro_material": [
        "wooden",
        "metal",
        "metallic",
        "plastic",
        "glass",
        "paper",
        "stone",
        "liquid",
        "solid",
    ],
}

MACRO_GROUPS = {
    "macro_action": [
        "run",
        "walk",
        "jump",
        "eat",
        "drink",
        "think",
        "learn",
        "speak",
        "write",
        "read",
        "move",
        "change",
        "grow",
        "build",
        "destroy",
        "create",
        "reason",
        "argue",
        "prove",
        "judge",
        "choose",
        "remember",
        "forget",
        "explain",
        "compare",
        "classify",
        "imagine",
        "reflect",
        "travel",
        "connect",
        "divide",
        "unify",
        "expand",
        "compress",
        "predict",
    ],
    "macro_system": [
        "truth",
        "justice",
        "freedom",
        "beauty",
        "love",
        "wisdom",
        "logic",
        "language",
        "memory",
        "system",
        "structure",
        "meaning",
        "identity",
        "order",
        "chaos",
        "history",
        "future",
        "theory",
        "principle",
        "symmetry",
        "topology",
        "geometry",
        "mathematics",
        "science",
        "cause",
        "effect",
    ],
}

ANCHOR_WORDS = ["apple", "fruit", "justice", "red", "language", "run"]
TYPE_ANCHOR_WORDS = ["apple", "build", "beautiful", "quickly", "and"]

VERB_TYPE_SEEDS = [
    "run",
    "walk",
    "jump",
    "eat",
    "drink",
    "think",
    "learn",
    "speak",
    "write",
    "read",
    "move",
    "change",
    "grow",
    "build",
    "destroy",
    "create",
    "reason",
    "argue",
    "prove",
    "judge",
    "choose",
    "remember",
    "forget",
    "explain",
    "compare",
    "classify",
    "imagine",
    "reflect",
    "travel",
    "connect",
    "divide",
    "unify",
    "expand",
    "compress",
    "predict",
    "know",
    "see",
    "make",
    "take",
    "give",
    "bring",
    "carry",
    "become",
    "remain",
    "keep",
    "start",
    "finish",
    "open",
    "close",
    "turn",
    "show",
    "use",
    "need",
    "want",
    "say",
    "tell",
    "work",
    "play",
]

ADJECTIVE_TYPE_SEEDS = [
    "red",
    "green",
    "blue",
    "yellow",
    "black",
    "white",
    "purple",
    "brown",
    "pink",
    "orange",
    "gray",
    "sweet",
    "sour",
    "bitter",
    "salty",
    "spicy",
    "fresh",
    "juicy",
    "ripe",
    "dry",
    "round",
    "square",
    "sharp",
    "smooth",
    "rough",
    "flat",
    "curved",
    "thin",
    "thick",
    "small",
    "large",
    "big",
    "tiny",
    "huge",
    "short",
    "tall",
    "long",
    "wide",
    "narrow",
    "heavy",
    "light",
    "hot",
    "warm",
    "cold",
    "cool",
    "frozen",
    "wooden",
    "metallic",
    "plastic",
    "beautiful",
    "ugly",
    "happy",
    "sad",
    "good",
    "bad",
    "simple",
    "complex",
    "possible",
    "real",
    "natural",
    "social",
    "human",
    "local",
    "global",
    "formal",
    "abstract",
    "concrete",
    "stable",
    "dynamic",
    "bright",
    "dark",
    "clean",
    "dirty",
    "soft",
    "hard",
]

ADVERB_TYPE_SEEDS = [
    "quickly",
    "slowly",
    "softly",
    "roughly",
    "clearly",
    "deeply",
    "highly",
    "truly",
    "really",
    "mostly",
    "mainly",
    "simply",
    "exactly",
    "nearly",
    "fully",
    "partly",
    "directly",
    "strongly",
    "weakly",
    "finally",
    "probably",
    "perhaps",
    "always",
    "never",
    "often",
    "usually",
    "rarely",
    "sometimes",
    "together",
    "apart",
    "already",
    "still",
    "soon",
    "later",
    "therefore",
    "however",
    "thus",
    "almost",
]

FUNCTION_TYPE_SEEDS = [
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "who",
    "whom",
    "whose",
    "which",
    "what",
    "where",
    "when",
    "why",
    "how",
    "if",
    "because",
    "although",
    "though",
    "while",
    "and",
    "or",
    "but",
    "nor",
    "so",
    "for",
    "of",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "to",
    "into",
    "onto",
    "over",
    "under",
    "about",
    "after",
    "before",
    "between",
    "through",
    "during",
    "without",
    "within",
    "across",
    "among",
    "against",
    "around",
    "as",
    "than",
    "then",
    "not",
    "no",
    "yes",
    "can",
    "could",
    "should",
    "would",
    "may",
    "might",
    "must",
    "shall",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "be",
    "am",
    "is",
    "are",
    "was",
    "were",
    "been",
    "being",
    "will",
]


@dataclass(frozen=True)
class TokenVariant:
    token_id: int
    raw_token: str
    decoded: str
    normalized: str
    leading_space: bool


def discover_tokenizer() -> AutoTokenizer:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到本地 GPT-2 路径: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        use_fast=False,
    )
    return tokenizer


def load_embedding_weight() -> torch.Tensor:
    model_file = MODEL_PATH / "model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"未找到模型权重文件: {model_file}")
    with safe_open(str(model_file), framework="pt", device="cpu") as handle:
        if "wte.weight" not in handle.keys():
            raise KeyError("model.safetensors 中缺少 wte.weight")
        weight = handle.get_tensor("wte.weight").detach().float().cpu()
    return weight


def is_clean_lexical_word(word: str) -> bool:
    if not word or not WORD_RE.fullmatch(word):
        return False
    if word.count("'") > 1 or word.count("-") > 1:
        return False
    return True


def normalize_word(decoded: str) -> str:
    return decoded.strip().lower()


def display_token(raw_token: str) -> str:
    return raw_token.replace("Ġ", "<SP>").replace("Ċ", "<NL>").replace("ĉ", "<TAB>")


def collect_clean_variants(tokenizer: AutoTokenizer) -> Tuple[Dict[str, List[TokenVariant]], int]:
    variants: Dict[str, List[TokenVariant]] = defaultdict(list)
    skipped = 0
    for token_id in range(int(tokenizer.vocab_size)):
        raw_token = tokenizer.convert_ids_to_tokens(token_id)
        decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        normalized = normalize_word(decoded)
        if not is_clean_lexical_word(normalized):
            skipped += 1
            continue
        variants[normalized].append(
            TokenVariant(
                token_id=token_id,
                raw_token=raw_token,
                decoded=decoded,
                normalized=normalized,
                leading_space=bool(raw_token.startswith("Ġ") or decoded.startswith(" ")),
            )
        )
    return variants, skipped


def canonical_variant_key(variant: TokenVariant) -> Tuple[int, int, int]:
    stripped = variant.decoded.strip()
    is_lowercase = stripped == stripped.lower()
    if is_lowercase and variant.leading_space:
        tier = 0
    elif is_lowercase:
        tier = 1
    elif variant.leading_space:
        tier = 2
    else:
        tier = 3
    return (tier, len(stripped), variant.token_id)


def l2_normalize(mat: torch.Tensor) -> torch.Tensor:
    denom = torch.linalg.norm(mat, dim=-1, keepdim=True).clamp_min(1e-8)
    return mat / denom


def topk_energy_ratio(vec: torch.Tensor, k: int) -> float:
    sq = vec.float().pow(2)
    kk = int(min(max(1, k), sq.numel()))
    top_vals = torch.topk(sq, k=kk).values
    return float(top_vals.sum().item() / (sq.sum().item() + EPS))


def dims_to_energy(vec: torch.Tensor, threshold: float) -> int:
    sq = torch.sort(vec.float().pow(2), descending=True).values
    csum = torch.cumsum(sq, dim=0) / (sq.sum() + EPS)
    idx = int(torch.searchsorted(csum, torch.tensor(float(threshold)), right=False).item())
    return idx + 1


def percentile_ranks(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    out = [0.0 for _ in values]
    denom = max(1, len(values) - 1)
    for rank, idx in enumerate(order):
        out[idx] = rank / denom
    return out


def load_seed_groups() -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for name, words in MICRO_GROUPS.items():
        groups[name] = list(words)
    for name, words in MACRO_GROUPS.items():
        groups[name] = list(words)

    concrete_groups: Dict[str, List[str]] = defaultdict(list)
    abstract_words: List[str] = []
    with SEED_CSV.open("r", encoding="utf-8-sig") as fh:
        reader = csv.reader(row for row in fh if not row.startswith("#"))
        for row in reader:
            if len(row) < 2:
                continue
            word = row[0].strip().lower()
            category = row[1].strip().lower()
            if not word or not category:
                continue
            if category == "abstract":
                abstract_words.append(word)
            else:
                concrete_groups[f"meso_{category}"].append(word)

    groups["macro_abstract"] = abstract_words
    for name, words in concrete_groups.items():
        groups[name] = words
    return groups


def load_lexical_type_groups() -> Dict[str, List[str]]:
    noun_words: List[str] = []
    with SEED_CSV.open("r", encoding="utf-8-sig") as fh:
        reader = csv.reader(row for row in fh if not row.startswith("#"))
        for row in reader:
            if len(row) < 2:
                continue
            word = row[0].strip().lower()
            if word:
                noun_words.append(word)

    return {
        "noun": sorted(dict.fromkeys(noun_words + MACRO_GROUPS["macro_system"])),
        "verb": sorted(dict.fromkeys(VERB_TYPE_SEEDS)),
        "adjective": sorted(dict.fromkeys(ADJECTIVE_TYPE_SEEDS)),
        "adverb": sorted(dict.fromkeys(ADVERB_TYPE_SEEDS)),
        "function": sorted(dict.fromkeys(FUNCTION_TYPE_SEEDS)),
    }


def band_of_group(group_name: str) -> str:
    if group_name.startswith("micro_"):
        return "micro"
    if group_name.startswith("meso_"):
        return "meso"
    if group_name.startswith("macro_"):
        return "macro"
    return "unknown"


def lexical_type_band(name: str) -> str:
    return name


def build_canonical_inventory(
    variants: Dict[str, List[TokenVariant]],
    embed_weight: torch.Tensor,
) -> Tuple[List[Dict[str, object]], torch.Tensor, Dict[str, int]]:
    rows: List[Dict[str, object]] = []
    vectors: List[torch.Tensor] = []
    word_to_index: Dict[str, int] = {}

    for word in sorted(variants):
        variant_list = sorted(variants[word], key=canonical_variant_key)
        canonical = variant_list[0]
        row = {
            "word": word,
            "token_id": int(canonical.token_id),
            "display_token": display_token(canonical.raw_token),
            "raw_token": canonical.raw_token,
            "leading_space": bool(canonical.leading_space),
            "variant_count": int(len(variant_list)),
            "variant_token_ids": [int(item.token_id) for item in variant_list],
            "variant_display_tokens": [display_token(item.raw_token) for item in variant_list[:8]],
        }
        word_to_index[word] = len(rows)
        rows.append(row)
        vectors.append(embed_weight[canonical.token_id].detach().float())

    matrix = torch.stack(vectors, dim=0).float()
    return rows, matrix, word_to_index


def fit_named_models(
    raw_groups: Dict[str, List[str]],
    normalized_matrix: torch.Tensor,
    word_to_index: Dict[str, int],
    band_resolver,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]], float]:
    group_list: List[Dict[str, object]] = []
    group_models: Dict[str, Dict[str, object]] = {}
    total_seed_count = 0
    resolved_seed_count = 0

    for group_name in sorted(raw_groups):
        seed_words = sorted(dict.fromkeys(raw_groups[group_name]))
        total_seed_count += len(seed_words)
        resolved_pairs = [(word, word_to_index[word]) for word in seed_words if word in word_to_index]
        resolved_seed_count += len(resolved_pairs)
        resolved_words = [word for word, _idx in resolved_pairs]
        resolved_indices = [idx for _word, idx in resolved_pairs]
        if len(resolved_indices) < 3:
            continue

        seed_matrix = normalized_matrix[resolved_indices]
        centroid = l2_normalize(seed_matrix.mean(dim=0, keepdim=True))[0]
        centered = seed_matrix - centroid.unsqueeze(0)
        max_rank = max(0, min(4, centered.shape[0] - 1, centered.shape[1]))
        if max_rank > 0:
            _u, _s, basis = torch.pca_lowrank(centered, q=max_rank, center=False, niter=2)
        else:
            basis = torch.zeros((centered.shape[1], 0), dtype=torch.float32)

        group_models[group_name] = {
            "name": group_name,
            "band": band_resolver(group_name),
            "seed_words": resolved_words,
            "seed_matrix": seed_matrix,
            "centroid": centroid,
            "basis": basis.float(),
            "seed_count": int(len(resolved_words)),
        }
        group_list.append(
            {
                "name": group_name,
                "band": band_resolver(group_name),
                "resolved_seed_count": int(len(resolved_words)),
                "resolved_seed_words": resolved_words,
            }
        )

    coverage = resolved_seed_count / max(1, total_seed_count)
    return group_list, group_models, float(coverage)


def fit_group_models(
    normalized_matrix: torch.Tensor,
    word_to_index: Dict[str, int],
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]], float]:
    return fit_named_models(
        raw_groups=load_seed_groups(),
        normalized_matrix=normalized_matrix,
        word_to_index=word_to_index,
        band_resolver=band_of_group,
    )


def fit_lexical_type_models(
    normalized_matrix: torch.Tensor,
    word_to_index: Dict[str, int],
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]], float]:
    return fit_named_models(
        raw_groups=load_lexical_type_groups(),
        normalized_matrix=normalized_matrix,
        word_to_index=word_to_index,
        band_resolver=lexical_type_band,
    )


def top_seed_matches(
    vec_norm: torch.Tensor,
    group_model: Dict[str, object],
    top_k: int = 3,
) -> List[Dict[str, object]]:
    seed_matrix = group_model["seed_matrix"]
    seed_words = group_model["seed_words"]
    sims = seed_matrix @ vec_norm
    kk = int(min(max(1, top_k), sims.numel()))
    values, indices = torch.topk(sims, k=kk)
    out: List[Dict[str, object]] = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        out.append({"word": seed_words[int(idx)], "score": float(score)})
    return out


def basis_capture_ratio(
    vec_norm: torch.Tensor,
    group_model: Dict[str, object],
) -> float:
    centroid = group_model["centroid"]
    basis = group_model["basis"]
    delta = vec_norm - centroid
    denom = float(delta.pow(2).sum().item())
    if denom <= EPS or basis.numel() == 0:
        return 0.0
    coeff = basis.T @ delta
    proj = basis @ coeff
    return float(proj.pow(2).sum().item() / (denom + EPS))


def global_pc_basis(matrix: torch.Tensor, q: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = matrix.mean(dim=0, keepdim=True)
    centered = matrix - mean
    rank = max(1, min(q, centered.shape[0] - 1, centered.shape[1]))
    _u, _s, basis = torch.pca_lowrank(centered, q=rank, center=False, niter=2)
    return mean[0], basis.float()


def scan_word_rows(
    canonical_rows: List[Dict[str, object]],
    matrix: torch.Tensor,
    group_models: Dict[str, Dict[str, object]],
    lexical_type_models: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    normalized_matrix = l2_normalize(matrix)
    global_mean, global_basis = global_pc_basis(matrix, q=8)

    group_names = list(group_models.keys())
    centroid_matrix = torch.stack([group_models[name]["centroid"] for name in group_names], dim=0)
    lexical_type_names = list(lexical_type_models.keys())
    lexical_type_centroids = torch.stack(
        [lexical_type_models[name]["centroid"] for name in lexical_type_names],
        dim=0,
    )

    metric_columns: Dict[str, List[float]] = {
        "norm_l2": [],
        "centered_norm_l2": [],
        "top32_energy_ratio": [],
        "pc_residual_ratio": [],
        "group_score": [],
        "group_margin": [],
        "group_basis_capture": [],
        "lexical_type_score": [],
        "lexical_type_margin": [],
        "lexical_type_basis_capture": [],
    }

    scanned_rows: List[Dict[str, object]] = []
    for row, vec, vec_norm in zip(canonical_rows, matrix, normalized_matrix):
        centered = vec - global_mean
        coeff = global_basis.T @ centered
        proj = global_basis @ coeff
        pc_energy_ratio = float(proj.pow(2).sum().item() / (centered.pow(2).sum().item() + EPS))
        pc_residual_ratio = 1.0 - pc_energy_ratio

        group_scores = centroid_matrix @ vec_norm
        top_vals, top_idx = torch.topk(group_scores, k=min(3, group_scores.numel()))
        best_group = group_names[int(top_idx[0].item())]
        second_score = float(top_vals[1].item()) if top_vals.numel() > 1 else float(top_vals[0].item())
        group_margin = float(top_vals[0].item() - second_score)
        group_model = group_models[best_group]
        capture = basis_capture_ratio(vec_norm, group_model)

        top_groups = []
        for value, idx in zip(top_vals.tolist(), top_idx.tolist()):
            group_name = group_names[int(idx)]
            top_groups.append(
                {
                    "group": group_name,
                    "band": group_models[group_name]["band"],
                    "score": float(value),
                }
            )

        lexical_type_scores = lexical_type_centroids @ vec_norm
        type_vals, type_idx = torch.topk(lexical_type_scores, k=min(3, lexical_type_scores.numel()))
        lexical_type = lexical_type_names[int(type_idx[0].item())]
        second_type_score = float(type_vals[1].item()) if type_vals.numel() > 1 else float(type_vals[0].item())
        lexical_type_margin = float(type_vals[0].item() - second_type_score)
        lexical_type_model = lexical_type_models[lexical_type]
        lexical_type_capture = basis_capture_ratio(vec_norm, lexical_type_model)

        top_lexical_types = []
        for value, idx in zip(type_vals.tolist(), type_idx.tolist()):
            type_name = lexical_type_names[int(idx)]
            top_lexical_types.append(
                {
                    "lexical_type": type_name,
                    "score": float(value),
                }
            )

        scanned_row = {
            **row,
            "band": group_model["band"],
            "group": best_group,
            "group_score": float(top_vals[0].item()),
            "group_margin": group_margin,
            "group_basis_capture": capture,
            "norm_l2": float(vec.norm().item()),
            "centered_norm_l2": float(centered.norm().item()),
            "top8_energy_ratio": topk_energy_ratio(vec, 8),
            "top32_energy_ratio": topk_energy_ratio(vec, 32),
            "dims_to_50": int(dims_to_energy(vec, 0.50)),
            "dims_to_80": int(dims_to_energy(vec, 0.80)),
            "dims_to_95": int(dims_to_energy(vec, 0.95)),
            "pc8_energy_ratio": pc_energy_ratio,
            "pc_residual_ratio": pc_residual_ratio,
            "top_seed_matches": top_seed_matches(vec_norm, group_model, top_k=3),
            "top_groups": top_groups,
            "lexical_type": lexical_type,
            "lexical_type_score": float(type_vals[0].item()),
            "lexical_type_margin": lexical_type_margin,
            "lexical_type_basis_capture": lexical_type_capture,
            "top_lexical_type_matches": top_seed_matches(vec_norm, lexical_type_model, top_k=3),
            "top_lexical_types": top_lexical_types,
        }
        scanned_rows.append(scanned_row)

        for key in metric_columns:
            metric_columns[key].append(float(scanned_row[key]))

    norm_pct = percentile_ranks(metric_columns["norm_l2"])
    centered_pct = percentile_ranks(metric_columns["centered_norm_l2"])
    top32_pct = percentile_ranks(metric_columns["top32_energy_ratio"])
    residual_pct = percentile_ranks(metric_columns["pc_residual_ratio"])
    group_score_pct = percentile_ranks(metric_columns["group_score"])
    group_margin_pct = percentile_ranks(metric_columns["group_margin"])
    capture_pct = percentile_ranks(metric_columns["group_basis_capture"])
    type_score_pct = percentile_ranks(metric_columns["lexical_type_score"])
    type_margin_pct = percentile_ranks(metric_columns["lexical_type_margin"])
    type_capture_pct = percentile_ranks(metric_columns["lexical_type_basis_capture"])

    for idx, row in enumerate(scanned_rows):
        score = (
            0.12 * norm_pct[idx]
            + 0.12 * centered_pct[idx]
            + 0.10 * top32_pct[idx]
            + 0.10 * residual_pct[idx]
            + 0.16 * group_score_pct[idx]
            + 0.12 * group_margin_pct[idx]
            + 0.10 * capture_pct[idx]
            + 0.10 * type_score_pct[idx]
            + 0.10 * type_margin_pct[idx]
            + 0.08 * type_capture_pct[idx]
        )
        row["effective_encoding_score"] = float(score)

    scanned_rows.sort(key=lambda item: item["word"])
    return scanned_rows


def build_summary(
    tokenizer: AutoTokenizer,
    embed_weight: torch.Tensor,
    skipped_count: int,
    group_list: List[Dict[str, object]],
    seed_coverage: float,
    lexical_type_list: List[Dict[str, object]],
    lexical_type_coverage: float,
    rows: List[Dict[str, object]],
) -> Dict[str, object]:
    band_counts = Counter(row["band"] for row in rows)
    group_counts = Counter(row["group"] for row in rows)
    lexical_type_counts = Counter(row["lexical_type"] for row in rows)
    anchors = {}
    row_map = {row["word"]: row for row in rows}
    for word in ANCHOR_WORDS:
        found = row_map.get(word)
        if found is None:
            anchors[word] = None
            continue
        anchors[word] = {
            "token_id": found["token_id"],
            "display_token": found["display_token"],
            "band": found["band"],
            "group": found["group"],
            "group_score": found["group_score"],
            "group_margin": found["group_margin"],
            "group_basis_capture": found["group_basis_capture"],
            "lexical_type": found["lexical_type"],
            "lexical_type_score": found["lexical_type_score"],
            "lexical_type_margin": found["lexical_type_margin"],
            "effective_encoding_score": found["effective_encoding_score"],
            "top_seed_matches": found["top_seed_matches"],
            "top_lexical_type_matches": found["top_lexical_type_matches"],
        }

    type_anchors = {}
    for word in TYPE_ANCHOR_WORDS:
        found = row_map.get(word)
        if found is None:
            type_anchors[word] = None
            continue
        type_anchors[word] = {
            "lexical_type": found["lexical_type"],
            "lexical_type_score": found["lexical_type_score"],
            "lexical_type_margin": found["lexical_type_margin"],
            "top_lexical_type_matches": found["top_lexical_type_matches"],
            "group": found["group"],
            "effective_encoding_score": found["effective_encoding_score"],
        }

    top_rows = sorted(rows, key=lambda item: item["effective_encoding_score"], reverse=True)[:20]
    ambiguous_rows = sorted(rows, key=lambda item: (item["group_margin"], -item["norm_l2"]))[:20]
    mean_score = sum(float(row["effective_encoding_score"]) for row in rows) / max(1, len(rows))
    anchor_scores = [
        float(anchors[word]["effective_encoding_score"])
        for word in ANCHOR_WORDS
        if anchors[word] is not None
    ]
    baseline_score = sum(anchor_scores) / max(1, len(anchor_scores))

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage119_gpt2_embedding_full_vocab_scan",
        "title": "GPT-2 词嵌入全词有效编码扫描",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status_short": "gpt2_embedding_vocab_scan_ready",
        "model_name": "gpt2",
        "model_path": str(MODEL_PATH),
        "vocab_size": int(tokenizer.vocab_size),
        "embedding_dim": int(embed_weight.shape[1]),
        "clean_token_variant_count": int(sum(row["variant_count"] for row in rows)),
        "clean_unique_word_count": int(len(rows)),
        "skipped_non_lexical_token_count": int(skipped_count),
        "seed_group_count": int(len(group_list)),
        "seed_group_coverage": float(seed_coverage),
        "lexical_type_group_count": int(len(lexical_type_list)),
        "lexical_type_coverage": float(lexical_type_coverage),
        "band_counts": dict(sorted(band_counts.items())),
        "group_counts_top20": dict(group_counts.most_common(20)),
        "lexical_type_counts": dict(sorted(lexical_type_counts.items())),
        "mean_effective_encoding_score": float(mean_score),
        "effective_encoding_baseline_score": float(baseline_score),
        "anchor_words": anchors,
        "lexical_type_anchor_words": type_anchors,
        "top_effective_words": [
            {
                "word": row["word"],
                "band": row["band"],
                "group": row["group"],
                "lexical_type": row["lexical_type"],
                "score": row["effective_encoding_score"],
            }
            for row in top_rows
        ],
        "most_ambiguous_words": [
            {
                "word": row["word"],
                "group": row["group"],
                "lexical_type": row["lexical_type"],
                "group_margin": row["group_margin"],
                "effective_encoding_score": row["effective_encoding_score"],
            }
            for row in ambiguous_rows
        ],
        "seed_groups": group_list,
        "lexical_type_groups": lexical_type_list,
    }


def build_report(summary: Dict[str, object], rows: List[Dict[str, object]]) -> str:
    row_map = {row["word"]: row for row in rows}
    lines = [
        "# Stage119: GPT-2 词嵌入全词有效编码扫描",
        "",
        "## 核心结果",
        f"- 词表总规模: {summary['vocab_size']}",
        f"- 干净词变体数: {summary['clean_token_variant_count']}",
        f"- 干净唯一词数: {summary['clean_unique_word_count']}",
        f"- 种子家族覆盖率: {summary['seed_group_coverage']:.4f}",
        f"- 词类种子覆盖率: {summary['lexical_type_coverage']:.4f}",
        f"- 平均有效编码分数: {summary['mean_effective_encoding_score']:.4f}",
        f"- 锚点基线分数: {summary['effective_encoding_baseline_score']:.4f}",
        "",
        "## 三层解释",
        "- 全局几何强度: 用范数、能量集中度和主成分残差衡量词向量是否只是落在公共方向上。",
        "- 语义家族对齐: 用种子家族质心和子空间捕获率衡量词是否进入稳定语义簇。",
        "- 词类结构层: 额外输出 noun（名词）/ verb（动词）/ adjective（形容词）/ adverb（副词）/ function（功能词）坐标，方便后续做统一数学压缩。",
        "- 个案继承接口: apple（苹果）类分析后续可以直接读取该扫描表，不必重新做大范围清洗。",
        "",
        "## 锚点词",
    ]

    for word in ANCHOR_WORDS:
        row = row_map.get(word)
        if row is None:
            lines.append(f"- {word}: 未命中")
            continue
        top_seed_desc = ", ".join(
            f"{item['word']}:{item['score']:.3f}" for item in row["top_seed_matches"]
        )
        lines.append(
            "- "
            f"{word}: band={row['band']}, group={row['group']}, "
            f"score={row['effective_encoding_score']:.4f}, "
            f"margin={row['group_margin']:.4f}, seeds=[{top_seed_desc}]"
        )

    lines.extend(["", "## 词类锚点"])
    for word in TYPE_ANCHOR_WORDS:
        row = row_map.get(word)
        if row is None:
            lines.append(f"- {word}: 未命中")
            continue
        type_seed_desc = ", ".join(
            f"{item['word']}:{item['score']:.3f}" for item in row["top_lexical_type_matches"]
        )
        lines.append(
            "- "
            f"{word}: lexical_type={row['lexical_type']}, "
            f"type_score={row['lexical_type_score']:.4f}, "
            f"type_margin={row['lexical_type_margin']:.4f}, "
            f"matches=[{type_seed_desc}]"
        )

    lines.extend(
        [
            "",
            "## 下一步用途",
            "- 从 word_rows.csv 里挑出 apple（苹果）及其近邻簇，继续做 fruit（水果）家族偏移与子空间裂缝分析。",
            "- 从 most_ambiguous_words 里筛出边界词，优先看哪些词天然跨 micro（微观）/ meso（中观）/ macro（宏观）三层。",
            "- 从 lexical_type_counts 和词类锚点出发，继续做“词类投影是否对应统一动力系统中的不同观测叶层”这条数学主线。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    summary: Dict[str, object],
    rows: List[Dict[str, object]],
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    rows_jsonl_path = output_dir / "word_rows.jsonl"
    rows_csv_path = output_dir / "word_rows.csv"
    report_path = output_dir / "STAGE119_GPT2_EMBEDDING_FULL_VOCAB_SCAN_REPORT.md"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )

    with rows_jsonl_path.open("w", encoding="utf-8-sig") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_fields = [
        "word",
        "token_id",
        "display_token",
        "leading_space",
        "variant_count",
        "band",
        "group",
        "group_score",
        "group_margin",
        "group_basis_capture",
        "lexical_type",
        "lexical_type_score",
        "lexical_type_margin",
        "lexical_type_basis_capture",
        "effective_encoding_score",
        "norm_l2",
        "centered_norm_l2",
        "top8_energy_ratio",
        "top32_energy_ratio",
        "dims_to_50",
        "dims_to_80",
        "dims_to_95",
        "pc8_energy_ratio",
        "pc_residual_ratio",
        "top_seed_1",
        "top_seed_1_score",
        "top_seed_2",
        "top_seed_2_score",
        "top_seed_3",
        "top_seed_3_score",
        "top_type_seed_1",
        "top_type_seed_1_score",
        "top_type_seed_2",
        "top_type_seed_2_score",
        "top_type_seed_3",
        "top_type_seed_3_score",
    ]
    with rows_csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            seed_matches = row["top_seed_matches"]
            type_matches = row["top_lexical_type_matches"]
            payload = {
                "word": row["word"],
                "token_id": row["token_id"],
                "display_token": row["display_token"],
                "leading_space": row["leading_space"],
                "variant_count": row["variant_count"],
                "band": row["band"],
                "group": row["group"],
                "group_score": row["group_score"],
                "group_margin": row["group_margin"],
                "group_basis_capture": row["group_basis_capture"],
                "lexical_type": row["lexical_type"],
                "lexical_type_score": row["lexical_type_score"],
                "lexical_type_margin": row["lexical_type_margin"],
                "lexical_type_basis_capture": row["lexical_type_basis_capture"],
                "effective_encoding_score": row["effective_encoding_score"],
                "norm_l2": row["norm_l2"],
                "centered_norm_l2": row["centered_norm_l2"],
                "top8_energy_ratio": row["top8_energy_ratio"],
                "top32_energy_ratio": row["top32_energy_ratio"],
                "dims_to_50": row["dims_to_50"],
                "dims_to_80": row["dims_to_80"],
                "dims_to_95": row["dims_to_95"],
                "pc8_energy_ratio": row["pc8_energy_ratio"],
                "pc_residual_ratio": row["pc_residual_ratio"],
                "top_seed_1": seed_matches[0]["word"] if len(seed_matches) > 0 else "",
                "top_seed_1_score": seed_matches[0]["score"] if len(seed_matches) > 0 else "",
                "top_seed_2": seed_matches[1]["word"] if len(seed_matches) > 1 else "",
                "top_seed_2_score": seed_matches[1]["score"] if len(seed_matches) > 1 else "",
                "top_seed_3": seed_matches[2]["word"] if len(seed_matches) > 2 else "",
                "top_seed_3_score": seed_matches[2]["score"] if len(seed_matches) > 2 else "",
                "top_type_seed_1": type_matches[0]["word"] if len(type_matches) > 0 else "",
                "top_type_seed_1_score": type_matches[0]["score"] if len(type_matches) > 0 else "",
                "top_type_seed_2": type_matches[1]["word"] if len(type_matches) > 1 else "",
                "top_type_seed_2_score": type_matches[1]["score"] if len(type_matches) > 1 else "",
                "top_type_seed_3": type_matches[2]["word"] if len(type_matches) > 2 else "",
                "top_type_seed_3_score": type_matches[2]["score"] if len(type_matches) > 2 else "",
            }
            writer.writerow(payload)

    report_path.write_text(build_report(summary, rows), encoding="utf-8-sig")
    return {
        "summary": summary_path,
        "word_rows_jsonl": rows_jsonl_path,
        "word_rows_csv": rows_csv_path,
        "report": report_path,
    }


def run_analysis(output_dir: Path = OUTPUT_DIR) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    tokenizer = discover_tokenizer()
    embed_weight = load_embedding_weight()
    variants, skipped_count = collect_clean_variants(tokenizer)
    canonical_rows, matrix, word_to_index = build_canonical_inventory(variants, embed_weight)
    normalized_matrix = l2_normalize(matrix)
    group_list, group_models, seed_coverage = fit_group_models(normalized_matrix, word_to_index)
    lexical_type_list, lexical_type_models, lexical_type_coverage = fit_lexical_type_models(
        normalized_matrix,
        word_to_index,
    )
    rows = scan_word_rows(canonical_rows, matrix, group_models, lexical_type_models)
    summary = build_summary(
        tokenizer,
        embed_weight,
        skipped_count,
        group_list,
        seed_coverage,
        lexical_type_list,
        lexical_type_coverage,
        rows,
    )
    write_outputs(summary, rows, output_dir=output_dir)
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-2 词嵌入全词有效编码扫描")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="输出目录",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    summary, _rows = run_analysis(output_dir=out_dir)
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(out_dir),
                "clean_unique_word_count": summary["clean_unique_word_count"],
                "mean_effective_encoding_score": summary["mean_effective_encoding_score"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
