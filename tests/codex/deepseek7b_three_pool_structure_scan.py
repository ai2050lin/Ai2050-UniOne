#!/usr/bin/env python
"""
Three-pool structure scan for DeepSeek-7B style models.

This script is designed for a staged research workflow:
1) survey pool: broad inventory scan over enough concepts
2) deep pool: richer prompt/context coverage for focused concepts
3) closure pool: the hardest subset for neuron-level closure candidates

Data is exported in a stable JSONL-friendly format for downstream analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCHEMA_VERSION = "agi.deepseek.structure_scan.v1"
POOL_ORDER = ("survey", "deep", "closure")
POOL_TOPK = {
    "survey": 96,
    "deep": 160,
    "closure": 256,
}
KNOWN_MODEL_REPOS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B",
    "Qwen/Qwen3-4B": "models--Qwen--Qwen3-4B",
    "Qwen/Qwen2.5-7B": "models--Qwen--Qwen2.5-7B",
    "zai-org/GLM-4-9B-Chat-HF": "models--zai-org--GLM-4-9B-Chat-HF",
}


@dataclass(frozen=True)
class LexemeItem:
    term: str
    category: str
    language: str


@dataclass(frozen=True)
class PoolTask:
    item: LexemeItem
    pool: str


@dataclass(frozen=True)
class GateSpec:
    module: torch.nn.Module
    d_ff: int
    gate_start: int
    gate_end: int


def detect_language(term: str) -> str:
    for ch in term:
        if ord(ch) > 127:
            return "non_ascii"
    return "ascii"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def load_items(path: str, max_items: int) -> List[LexemeItem]:
    out: List[LexemeItem] = []
    for line in read_text(Path(path)).splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "," not in s:
            continue
        term, category = [x.strip() for x in s.split(",", 1)]
        if not term or not category:
            continue
        out.append(LexemeItem(term=term, category=category, language=detect_language(term)))
    if not out:
        raise ValueError(f"no usable items found in {path}")
    if max_items > 0 and len(out) > max_items:
        by_category: Dict[str, List[LexemeItem]] = defaultdict(list)
        for item in out:
            by_category[item.category].append(item)
        limited: List[LexemeItem] = []
        category_order = sorted(by_category)
        while len(limited) < max_items:
            progressed = False
            for category in category_order:
                if by_category[category]:
                    limited.append(by_category[category].pop(0))
                    progressed = True
                    if len(limited) >= max_items:
                        break
            if not progressed:
                break
        out = limited
    return out


def build_pool_tasks(
    items: Sequence[LexemeItem],
    survey_per_category: int,
    deep_per_category: int,
    closure_per_category: int,
    seed: int,
) -> List[PoolTask]:
    rng = random.Random(seed)
    by_category: Dict[str, List[LexemeItem]] = defaultdict(list)
    for item in items:
        by_category[item.category].append(item)

    tasks: List[PoolTask] = []
    for category in sorted(by_category):
        members = list(by_category[category])
        rng.shuffle(members)

        survey_count = min(len(members), max(0, survey_per_category))
        deep_count = min(survey_count, max(0, deep_per_category))
        closure_count = min(deep_count, max(0, closure_per_category))

        survey_members = members[:survey_count]
        deep_members = survey_members[:deep_count]
        closure_members = deep_members[:closure_count]

        for item in survey_members:
            tasks.append(PoolTask(item=item, pool="survey"))
        for item in deep_members:
            tasks.append(PoolTask(item=item, pool="deep"))
        for item in closure_members:
            tasks.append(PoolTask(item=item, pool="closure"))

    return tasks


def resolve_model_path(model_id: str) -> str:
    path = Path(model_id)
    if path.exists():
        return str(path)

    local_roots = [
        Path(r"D:\develop\model\hub"),
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    repo_dir_name = KNOWN_MODEL_REPOS.get(model_id)
    if repo_dir_name is not None:
        for root in local_roots:
            snap_root = root / repo_dir_name / "snapshots"
            if not snap_root.exists():
                continue
            candidates = sorted(p for p in snap_root.iterdir() if p.is_dir())
            if candidates:
                return str(candidates[-1])
    return model_id


def load_model(model_id: str, dtype_name: str, local_files_only: bool, device: str):
    model_ref = resolve_model_path(model_id)
    os.environ["HF_HUB_OFFLINE"] = "1" if local_files_only else os.environ.get("HF_HUB_OFFLINE", "0")
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if local_files_only else os.environ.get("TRANSFORMERS_OFFLINE", "0")

    tok = AutoTokenizer.from_pretrained(
        model_ref,
        local_files_only=local_files_only,
        trust_remote_code=True,
        use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if device == "auto":
        use_cuda = torch.cuda.is_available()
    elif device == "cuda":
        use_cuda = True
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available")
    else:
        use_cuda = False

    dtype = getattr(torch, dtype_name)
    kwargs = {
        "local_files_only": local_files_only,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "dtype": dtype,
        "device_map": "auto" if use_cuda else "cpu",
        "attn_implementation": "eager",
    }
    model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
    model.eval()
    return model, tok, model_ref


def gate_spec_for_layer(layer) -> GateSpec:
    mlp = layer.mlp
    if hasattr(mlp, "gate_proj"):
        module = mlp.gate_proj
        d_ff = int(module.out_features)
        return GateSpec(module=module, d_ff=d_ff, gate_start=0, gate_end=d_ff)
    if hasattr(mlp, "gate_up_proj"):
        module = mlp.gate_up_proj
        d_ff = int(module.out_features // 2)
        return GateSpec(module=module, d_ff=d_ff, gate_start=0, gate_end=d_ff)
    raise AttributeError(f"unsupported MLP gate structure: {type(mlp).__name__}")


def slice_gate_output(output, spec: GateSpec) -> torch.Tensor:
    tensor = output[0] if isinstance(output, tuple) else output
    return tensor[..., spec.gate_start : spec.gate_end]


def zero_gate_indices(output, spec: GateSpec, indices: torch.Tensor):
    if indices.numel() <= 0:
        return output
    if isinstance(output, tuple):
        tensor = output[0].clone()
        tensor[..., spec.gate_start + indices] = 0.0
        return (tensor,) + output[1:]
    tensor = output.clone()
    tensor[..., spec.gate_start + indices] = 0.0
    return tensor


class GateCollector:
    def __init__(self, model):
        self.layers = list(model.model.layers)
        self.buffers: List[torch.Tensor | None] = [None for _ in self.layers]
        self.specs = [gate_spec_for_layer(layer) for layer in self.layers]
        self.handles = []
        for li, spec in enumerate(self.specs):
            self.handles.append(spec.module.register_forward_hook(self._mk_hook(li)))

    def _mk_hook(self, layer_idx: int):
        def _hook(_module, _inputs, output):
            gate = slice_gate_output(output, self.specs[layer_idx])
            self.buffers[layer_idx] = gate[0, -1, :].detach().float().cpu()
            return output

        return _hook

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def d_ff(self) -> int:
        return int(self.specs[0].d_ff)

    @property
    def total_neurons(self) -> int:
        return self.n_layers * self.d_ff

    def reset(self) -> None:
        for i in range(len(self.buffers)):
            self.buffers[i] = None

    def get_flat(self) -> np.ndarray:
        miss = [i for i, x in enumerate(self.buffers) if x is None]
        if miss:
            raise RuntimeError(f"missing hook outputs for layers: {miss}")
        return np.concatenate([x.numpy() for x in self.buffers if x is not None], axis=0).astype(np.float32)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def run_prompt(model, tok, text: str):
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        return model(**enc, use_cache=False, return_dict=True)


def prompts_for_item(item: LexemeItem, pool: str) -> List[str]:
    term = item.term
    if pool == "survey":
        return [
            f"This is {term}.",
            f"People often discuss {term}.",
            f"The concept {term} belongs to",
        ]
    if pool == "deep":
        return [
            f"This is {term}.",
            f"I saw {term} yesterday.",
            f"The concept {term} is related to",
            f"When experts discuss {term}, they often mention",
            f"The role of {term} in a larger system is",
            f"Compared with similar concepts, {term} is",
            f"A definition of {term} is",
        ]
    if pool == "closure":
        return [
            f"This is {term}.",
            f"A precise definition of {term} is",
            f"The family of {term} can be described as",
            f"The key attribute of {term} is",
            f"The relation between {term} and nearby concepts is",
            f"In a reasoning chain, {term} usually leads to",
            f"The stage-conditioned continuation of {term} is",
            f"An incorrect family assignment for {term} would be",
            f"Under a protocol change, {term} should still preserve",
            f"The minimal explanation for {term} is",
        ]
    raise ValueError(f"unknown pool: {pool}")


def topk_with_values(vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    k = min(k, int(vec.shape[0]))
    if k <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    idx = np.argpartition(vec, -k)[-k:]
    idx = idx[np.argsort(vec[idx])[::-1]]
    vals = vec[idx].astype(np.float32)
    return idx.astype(np.int64), vals


def index_to_layer_neuron(idx: int, d_ff: int) -> Tuple[int, int]:
    return idx // d_ff, idx % d_ff


def layer_distribution(indices: Sequence[int], d_ff: int) -> Dict[str, int]:
    ctr = Counter(index_to_layer_neuron(int(i), d_ff)[0] for i in indices)
    return {str(k): int(v) for k, v in sorted(ctr.items())}


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def safe_mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return float(statistics.mean(xs)) if xs else 0.0


def finite_stats(vec: np.ndarray, d_ff: int) -> Dict[str, object]:
    finite_mask = np.isfinite(vec)
    nonfinite_mask = ~finite_mask
    nan_count = int(np.isnan(vec).sum())
    posinf_count = int(np.isposinf(vec).sum())
    neginf_count = int(np.isneginf(vec).sum())
    bad_layers = []
    if d_ff > 0 and nonfinite_mask.any():
        for layer_idx in range(int(vec.shape[0] // d_ff)):
            layer_bad = int(nonfinite_mask[layer_idx * d_ff : (layer_idx + 1) * d_ff].sum())
            if layer_bad:
                bad_layers.append({"layer": layer_idx, "nonfinite_count": layer_bad})
    return {
        "nonfinite_count": int(nonfinite_mask.sum()),
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "bad_layers": bad_layers,
    }


def prompt_stability(prompt_signatures: Sequence[Sequence[int]]) -> float:
    sims = []
    for i in range(len(prompt_signatures)):
        for j in range(i + 1, len(prompt_signatures)):
            sims.append(jaccard(prompt_signatures[i], prompt_signatures[j]))
    return safe_mean(sims)


def top_layer_ratio(layer_dist: Dict[str, int]) -> float:
    if not layer_dist:
        return 0.0
    vals = sorted((int(v) for v in layer_dist.values()), reverse=True)
    total = sum(vals)
    if total <= 0:
        return 0.0
    return float(sum(vals[:3]) / total)


def analyze_task(
    task: PoolTask,
    model,
    tok,
    collector: GateCollector,
    top_k: int,
    nonfinite_policy: str,
) -> Dict[str, object]:
    prompts = prompts_for_item(task.item, task.pool)
    sum_vec = np.zeros(collector.total_neurons, dtype=np.float64)
    prompt_records = []
    norms = []
    activation_means = []
    skipped_prompt_count = 0
    nonfinite_prompt_count = 0

    for prompt in prompts:
        collector.reset()
        _ = run_prompt(model, tok, prompt)
        flat = collector.get_flat()
        stats = finite_stats(flat, collector.d_ff)
        if int(stats["nonfinite_count"]) > 0:
            nonfinite_prompt_count += 1
            if nonfinite_policy == "raise":
                raise RuntimeError(
                    "non-finite activations detected: "
                    f"term={task.item.term!r}, category={task.item.category!r}, pool={task.pool!r}, "
                    f"prompt={prompt!r}, stats={json.dumps(stats, ensure_ascii=False)}"
                )
            if nonfinite_policy == "skip_prompt":
                skipped_prompt_count += 1
                continue
            raise ValueError(f"unknown nonfinite_policy: {nonfinite_policy}")
        sum_vec += flat
        norms.append(float(np.linalg.norm(flat)))
        activation_means.append(float(flat.mean()))
        idx, vals = topk_with_values(flat, top_k)
        prompt_records.append(
            {
                "prompt": prompt,
                "top_indices": [int(x) for x in idx.tolist()],
                "top_values": [float(x) for x in vals.tolist()],
                "layer_distribution": layer_distribution(idx, collector.d_ff),
                "l2_norm": float(np.linalg.norm(flat)),
                "mean_activation": float(flat.mean()),
                "finite_stats": stats,
            }
        )

    if not prompt_records:
        raise RuntimeError(
            f"no valid prompt records remained after filtering for term={task.item.term!r}, pool={task.pool!r}"
        )

    mean_vec = (sum_vec / len(prompt_records)).astype(np.float32)
    sig_idx, sig_vals = topk_with_values(mean_vec, top_k)
    aggregate_layer_dist = layer_distribution(sig_idx, collector.d_ff)

    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "pool_record",
        "item": {
            "term": task.item.term,
            "category": task.item.category,
            "language": task.item.language,
        },
        "pool": task.pool,
        "prompt_count": len(prompts),
        "valid_prompt_count": len(prompt_records),
        "skipped_prompt_count": skipped_prompt_count,
        "signature_top_k": int(top_k),
        "signature_top_indices": [int(x) for x in sig_idx.tolist()],
        "signature_top_values": [float(x) for x in sig_vals.tolist()],
        "signature_layer_distribution": aggregate_layer_dist,
        "aggregate": {
            "prompt_stability_jaccard_mean": prompt_stability([x["top_indices"] for x in prompt_records]),
            "mean_prompt_l2_norm": safe_mean(norms),
            "mean_prompt_activation": safe_mean(activation_means),
            "top3_layer_ratio": top_layer_ratio(aggregate_layer_dist),
            "nonfinite_prompt_count": nonfinite_prompt_count,
        },
        "model_dims": {
            "n_layers": collector.n_layers,
            "d_ff": collector.d_ff,
            "total_neurons": collector.total_neurons,
        },
        "prompt_records": prompt_records,
    }


def slim_record_for_storage(record: Dict[str, object], keep_prompt_records: bool) -> Dict[str, object]:
    payload = dict(record)
    if keep_prompt_records:
        return payload
    payload["prompt_records"] = []
    return payload


def build_family_prototypes(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for rec in records:
        grouped[(str(rec["pool"]), str(rec["item"]["category"]))].append(rec)

    prototypes: List[Dict[str, object]] = []
    for (pool, category), members in sorted(grouped.items()):
        member_count = len(members)
        if member_count <= 0:
            continue
        sig_len = max(len(m["signature_top_indices"]) for m in members)
        d_ff = int(members[0].get("model_dims", {}).get("d_ff", 18944))
        reuse_counter: Counter[int] = Counter()
        for member in members:
            reuse_counter.update(int(x) for x in member["signature_top_indices"])

        top_proto = reuse_counter.most_common(sig_len)
        proto_indices = [int(idx) for idx, _ in top_proto]
        shared_threshold = max(2, math.ceil(member_count * 0.5))
        shared_indices = [int(idx) for idx, c in reuse_counter.items() if c >= shared_threshold]

        prototypes.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "family_prototype",
                "pool": pool,
                "category": category,
                "member_count": member_count,
                "prototype_top_indices": proto_indices,
                "prototype_layer_distribution": layer_distribution(proto_indices, d_ff),
                "shared_neurons": sorted(shared_indices)[:512],
                "shared_neuron_count": len(shared_indices),
                "mean_prompt_stability": safe_mean(
                    float(member["aggregate"]["prompt_stability_jaccard_mean"]) for member in members
                ),
            }
        )
    return prototypes


def build_closure_candidates(
    records: Sequence[Dict[str, object]],
    family_prototypes: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    proto_map = {
        (str(proto["pool"]), str(proto["category"])): proto
        for proto in family_prototypes
    }

    by_pool: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for proto in family_prototypes:
        by_pool[str(proto["pool"])].append(proto)

    out: List[Dict[str, object]] = []
    for rec in records:
        pool = str(rec["pool"])
        if pool not in {"deep", "closure"}:
            continue
        category = str(rec["item"]["category"])
        sig = [int(x) for x in rec["signature_top_indices"]]
        same = proto_map.get((pool, category))
        if same is None:
            continue

        same_overlap = jaccard(sig, same["prototype_top_indices"])
        other_overlaps = [
            jaccard(sig, proto["prototype_top_indices"])
            for proto in by_pool[pool]
            if str(proto["category"]) != category
        ]
        best_other = max(other_overlaps) if other_overlaps else 0.0
        family_margin = same_overlap - best_other
        stability = float(rec["aggregate"]["prompt_stability_jaccard_mean"])
        layer_focus = float(rec["aggregate"]["top3_layer_ratio"])
        closure_score = max(
            0.0,
            min(
                1.0,
                0.45 * same_overlap
                + 0.30 * max(0.0, family_margin)
                + 0.15 * stability
                + 0.10 * layer_focus,
            ),
        )
        family_shared = set(int(x) for x in same["shared_neurons"])
        item_specific = [idx for idx in sig if idx not in family_shared]
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "closure_candidate",
                "pool": pool,
                "item": rec["item"],
                "family_support_jaccard": float(same_overlap),
                "best_other_family_jaccard": float(best_other),
                "wrong_family_margin": float(family_margin),
                "prompt_stability_jaccard_mean": float(stability),
                "top3_layer_ratio": float(layer_focus),
                "exact_closure_proxy": float(closure_score),
                "family_shared_neurons": sorted(int(x) for x in set(sig) & family_shared)[:256],
                "item_specific_neurons": [int(x) for x in item_specific[:256]],
                "signature_top_indices": sig,
            }
        )
    out.sort(key=lambda x: (x["pool"], -x["exact_closure_proxy"], x["item"]["category"], x["item"]["term"]))
    return out


def build_summary(
    records: Sequence[Dict[str, object]],
    family_prototypes: Sequence[Dict[str, object]],
    closure_candidates: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    pool_counts = Counter(str(x["pool"]) for x in records)
    category_counts = Counter(str(x["item"]["category"]) for x in records if x["pool"] == "survey")
    closure_top = closure_candidates[:20]
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "summary",
        "headline_metrics": {
            "record_count": len(records),
            "family_count": len(family_prototypes),
            "closure_candidate_count": len(closure_candidates),
            "survey_records": int(pool_counts.get("survey", 0)),
            "deep_records": int(pool_counts.get("deep", 0)),
            "closure_records": int(pool_counts.get("closure", 0)),
            "records_with_nonfinite_prompts": int(
                sum(1 for x in records if int(x["aggregate"].get("nonfinite_prompt_count", 0)) > 0)
            ),
            "nonfinite_prompt_count_total": int(
                sum(int(x["aggregate"].get("nonfinite_prompt_count", 0)) for x in records)
            ),
            "mean_prompt_stability_survey": safe_mean(
                float(x["aggregate"]["prompt_stability_jaccard_mean"]) for x in records if x["pool"] == "survey"
            ),
            "mean_prompt_stability_deep": safe_mean(
                float(x["aggregate"]["prompt_stability_jaccard_mean"]) for x in records if x["pool"] == "deep"
            ),
            "mean_prompt_stability_closure": safe_mean(
                float(x["aggregate"]["prompt_stability_jaccard_mean"]) for x in records if x["pool"] == "closure"
            ),
        },
        "category_coverage_survey": dict(sorted((k, int(v)) for k, v in category_counts.items())),
        "top_closure_candidates": [
            {
                "term": x["item"]["term"],
                "category": x["item"]["category"],
                "pool": x["pool"],
                "exact_closure_proxy": x["exact_closure_proxy"],
                "wrong_family_margin": x["wrong_family_margin"],
            }
            for x in closure_top
        ],
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_format_md(path: Path) -> None:
    lines = [
        "# Structure Scan Format",
        "",
        f"- schema_version: `{SCHEMA_VERSION}`",
        "- `manifest.json`: run metadata, file map, config, device, timing",
        "- `records.jsonl`: one line per item-pool analysis record",
        "- `families.jsonl`: one line per family prototype per pool",
        "- `closure_candidates.jsonl`: exact-closure candidate diagnostics",
        "- `summary.json`: compact headline metrics for quick loading",
        "- `REPORT.md`: human-readable stage summary",
        "",
        "## records.jsonl",
        "- `item.term`: analyzed token or concept string",
        "- `item.category`: family/category label from the input inventory",
        "- `pool`: survey/deep/closure",
        "- `valid_prompt_count` / `skipped_prompt_count`: prompt-level retention after nonfinite checks",
        "- `signature_top_indices`: sparse neuron signature",
        "- `prompt_records`: per-prompt sparse traces",
        "- survey pool can intentionally omit prompt-level traces to reduce export size",
        "",
        "## families.jsonl",
        "- `prototype_top_indices`: pooled family prototype signature",
        "- `shared_neurons`: neurons reused by a large fraction of family members",
        "",
        "## closure_candidates.jsonl",
        "- `family_support_jaccard`: overlap with same-family prototype",
        "- `wrong_family_margin`: same-family overlap minus best wrong-family overlap",
        "- `exact_closure_proxy`: compact closure score for prioritization",
        "",
        "## numeric stability",
        "- default dtype is `bfloat16` on GPU to reduce float16 nonfinite activations",
        "- `--nonfinite-policy raise` stops the run immediately when a prompt produces NaN/Inf",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_md(
    path: Path,
    manifest: Dict[str, object],
    summary: Dict[str, object],
    closure_candidates: Sequence[Dict[str, object]],
) -> None:
    headline = summary["headline_metrics"]
    lines = [
        "# DeepSeek Three-Pool Structure Scan Report",
        "",
        "## Run",
        f"- Model: {manifest['model_id']}",
        f"- Device: {manifest['device']}",
        f"- Runtime sec: {manifest['runtime_sec']:.4f}",
        f"- Input items: {manifest['counts']['input_items']}",
        f"- Tasks: {manifest['counts']['tasks']}",
        "",
        "## Headline Metrics",
        f"- Survey records: {headline['survey_records']}",
        f"- Deep records: {headline['deep_records']}",
        f"- Closure records: {headline['closure_records']}",
        f"- Family prototypes: {headline['family_count']}",
        f"- Closure candidates: {headline['closure_candidate_count']}",
        f"- Records with nonfinite prompts: {headline.get('records_with_nonfinite_prompts', 0)}",
        f"- Total nonfinite prompts: {headline.get('nonfinite_prompt_count_total', 0)}",
        f"- Mean survey prompt stability: {headline['mean_prompt_stability_survey']:.4f}",
        f"- Mean deep prompt stability: {headline['mean_prompt_stability_deep']:.4f}",
        f"- Mean closure prompt stability: {headline['mean_prompt_stability_closure']:.4f}",
        "",
        "## Survey Category Coverage",
    ]
    for category, count in summary["category_coverage_survey"].items():
        lines.append(f"- {category}: {count}")
    lines.extend(["", "## Top Closure Candidates"])
    for item in closure_candidates[:20]:
        lines.append(
            f"- {item['item']['term']} ({item['item']['category']}, {item['pool']}): "
            f"closure={item['exact_closure_proxy']:.4f}, wrong_family_margin={item['wrong_family_margin']:.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_bundle(
    out_dir: Path,
    manifest: Dict[str, object],
    records: Sequence[Dict[str, object]],
    family_prototypes: Sequence[Dict[str, object]],
    closure_candidates: Sequence[Dict[str, object]],
    summary: Dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "records.jsonl", records)
    write_jsonl(out_dir / "families.jsonl", family_prototypes)
    write_jsonl(out_dir / "closure_candidates.jsonl", closure_candidates)
    write_json(out_dir / "summary.json", summary)
    write_format_md(out_dir / "FORMAT.md")
    write_report_md(out_dir / "REPORT.md", manifest, summary, closure_candidates)


def main() -> None:
    ap = argparse.ArgumentParser(description="Three-pool DeepSeek structure scan")
    ap.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--items-file", default="tests/codex/deepseek7b_nouns_english_500plus.csv")
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--survey-per-category", type=int, default=24)
    ap.add_argument("--deep-per-category", type=int, default=8)
    ap.add_argument("--closure-per-category", type=int, default=3)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--store-survey-prompt-records", action="store_true", default=False)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--nonfinite-policy", choices=["raise", "skip_prompt"], default="raise")
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    items = load_items(args.items_file, args.max_items)
    tasks = build_pool_tasks(
        items=items,
        survey_per_category=args.survey_per_category,
        deep_per_category=args.deep_per_category,
        closure_per_category=args.closure_per_category,
        seed=args.seed,
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_three_pool_scan_{ts}")

    t0 = time.time()
    model, tok, model_ref = load_model(
        model_id=args.model_id,
        dtype_name=args.dtype,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    collector = GateCollector(model)

    try:
        records = []
        total_tasks = len(tasks)
        for i, task in enumerate(tasks, start=1):
            record = analyze_task(
                task,
                model=model,
                tok=tok,
                collector=collector,
                top_k=POOL_TOPK[task.pool],
                nonfinite_policy=args.nonfinite_policy,
            )
            record["run_index"] = i
            keep_prompt_records = task.pool != "survey" or bool(args.store_survey_prompt_records)
            records.append(slim_record_for_storage(record, keep_prompt_records=keep_prompt_records))
            if args.progress_every > 0 and (i % args.progress_every == 0 or i == total_tasks):
                print(
                    json.dumps(
                        {
                            "progress": {
                                "completed_tasks": i,
                                "total_tasks": total_tasks,
                                "pool": task.pool,
                                "term": task.item.term,
                            }
                        },
                        ensure_ascii=False,
                    )
                )

        family_prototypes = build_family_prototypes(records)
        closure_candidates = build_closure_candidates(records, family_prototypes)
        summary = build_summary(records, family_prototypes, closure_candidates)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "manifest",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runtime_sec": float(time.time() - t0),
            "model_id": args.model_id,
            "model_ref": model_ref,
            "device": str(next(model.parameters()).device),
            "config": {
                "items_file": args.items_file,
                "max_items": args.max_items,
                "survey_per_category": args.survey_per_category,
                "deep_per_category": args.deep_per_category,
                "closure_per_category": args.closure_per_category,
                "dtype": args.dtype,
                "device": args.device,
                "local_files_only": bool(args.local_files_only),
                "seed": int(args.seed),
                "nonfinite_policy": args.nonfinite_policy,
                "n_layers": collector.n_layers,
                "d_ff": collector.d_ff,
                "total_neurons": collector.total_neurons,
            },
            "files": {
                "manifest": "manifest.json",
                "records": "records.jsonl",
                "families": "families.jsonl",
                "closure_candidates": "closure_candidates.jsonl",
                "summary": "summary.json",
                "format": "FORMAT.md",
                "report": "REPORT.md",
            },
            "counts": {
                "input_items": len(items),
                "tasks": len(tasks),
                "records": len(records),
                "families": len(family_prototypes),
                "closure_candidates": len(closure_candidates),
            },
            "storage_policy": {
                "store_survey_prompt_records": bool(args.store_survey_prompt_records),
                "deep_prompt_records": True,
                "closure_prompt_records": True,
            },
        }
        export_bundle(
            out_dir=out_dir,
            manifest=manifest,
            records=records,
            family_prototypes=family_prototypes,
            closure_candidates=closure_candidates,
            summary=summary,
        )
    finally:
        collector.close()

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "manifest": str(out_dir / "manifest.json"),
                "summary": str(out_dir / "summary.json"),
                "record_count": len(records),
                "family_count": len(family_prototypes),
                "closure_candidate_count": len(closure_candidates),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
