#!/usr/bin/env python
"""
Search the minimal causal boundary of topology space T for concept->protocol-field calls.

Goal:
- move from "top-k meso-field exists" to:
  "how many heads are minimally needed to causally deform a concept's field call?"

We reuse the ranking from U(c, tau, l, h) and probe the boundary on:
- apple -> fruit
- cat -> animal
- truth -> abstract
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


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


def default_model_specs() -> List[Tuple[str, str]]:
    return [
        ("gpt2", resolve_snapshot_dir("models--gpt2")),
        ("qwen3_4b", resolve_snapshot_dir("models--Qwen--Qwen3-4B")),
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
    model.config.output_attentions = True
    return model, tok


def discover_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    raise RuntimeError("Cannot discover transformer layers")


def get_attention_module(layer: torch.nn.Module) -> torch.nn.Module:
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
        return layer.self_attn
    if hasattr(layer, "attn") and hasattr(layer.attn, "c_proj"):
        return layer.attn
    raise RuntimeError("Cannot find a probe-able attention module")


class HeadGroupAblator:
    def __init__(self, model):
        self.layers = discover_layers(model)
        self.handles = []
        self.active_map: Dict[int, set[int]] = {}
        self.head_dim = int(getattr(model.config, "hidden_size") // getattr(model.config, "num_attention_heads"))
        for li, layer in enumerate(self.layers):
            attn = get_attention_module(layer)
            target = attn.o_proj if hasattr(attn, "o_proj") else attn.c_proj
            self.handles.append(target.register_forward_pre_hook(self._pre_hook(li)))

    def _pre_hook(self, li: int):
        def fn(_module, inputs):
            heads = self.active_map.get(li)
            if not heads:
                return None
            x = inputs[0]
            y = x.clone()
            for head in heads:
                start = head * self.head_dim
                end = start + self.head_dim
                y[..., start:end] = 0
            return (y,)

        return fn

    def set_active_group(self, heads: Sequence[Tuple[int, int]]) -> None:
        active: Dict[int, set[int]] = defaultdict(set)
        for layer, head in heads:
            active[int(layer)].add(int(head))
        self.active_map = dict(active)

    def clear(self) -> None:
        self.active_map = {}

    def close(self) -> None:
        for h in self.handles:
            h.remove()


def protocol_prompts(field: str, word: str) -> List[str]:
    return [
        f"kind {field} item {word}",
        f"class {field} item {word}",
        f"group {field} item {word}",
    ]


def run_model(model, tok, text: str):
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc, use_cache=False, output_attentions=True, return_dict=True)
    return out, int(enc["input_ids"].shape[1])


def mean_stack(rows: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(rows, axis=0), axis=0).astype(np.float32)


def last_token_head_topo(out, target_len: int) -> Dict[Tuple[int, int], np.ndarray]:
    topo = {}
    for li, attn in enumerate(out.attentions):
        arr = attn[0].detach().float().cpu().numpy().astype(np.float32)
        last_row = arr[:, -1, :]
        pad = target_len - last_row.shape[1]
        if pad > 0:
            last_row = np.pad(last_row, ((0, 0), (0, pad)), mode="constant")
        for hi in range(last_row.shape[0]):
            topo[(li, hi)] = last_row[hi].astype(np.float32)
    return topo


def norm_delta(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = (float(np.linalg.norm(a)) + float(np.linalg.norm(b))) / 2.0 + eps
    return float(np.linalg.norm(a - b) / denom)


def load_mapping_json() -> Dict[str, object]:
    path = Path("tests/codex_temp/gpt2_qwen3_concept_protocol_field_mapping_20260308.json")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_k_values(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Need at least one k value")
    return values


def build_control_group(top_rows: Sequence[Dict[str, float]], use_k: int, n_heads: int) -> List[Tuple[int, int]]:
    top_group = [(int(row["layer"]), int(row["head"])) for row in top_rows[:use_k]]
    top_set = {(int(row["layer"]), int(row["head"])) for row in top_rows}
    need = Counter(layer for layer, _head in top_group)
    control = []
    for layer, count in need.items():
        picked = 0
        for head in range(n_heads):
            if (layer, head) not in top_set:
                control.append((layer, head))
                picked += 1
            if picked >= count:
                break
    return control


def compute_field_scores(
    model,
    tok,
    ablator: HeadGroupAblator | None,
    concept: str,
    fields: Sequence[str],
    group: Sequence[Tuple[int, int]] | None,
    fixed_fit_selectivity: Dict[str, Dict[Tuple[int, int], float]],
) -> Dict[str, float]:
    if ablator is not None:
        if group:
            ablator.set_active_group(group)
        else:
            ablator.clear()

    target_len = max(
        len(tok(text, add_special_tokens=False)["input_ids"])
        for field in fields
        for text in protocol_prompts(field, concept)
    )
    prompt_runs_by_field = {}
    for field in fields:
        rows = []
        for text in protocol_prompts(field, concept):
            out, _seq_len = run_model(model, tok, text)
            rows.append(last_token_head_topo(out, target_len))
        mean_map = {}
        for key in rows[0].keys():
            mean_map[key] = mean_stack([row[key] for row in rows])
        prompt_runs_by_field[field] = mean_map

    field_scores = {}
    for field in fields:
        total = 0.0
        negative_vecs_by_key = {}
        for key in prompt_runs_by_field[field].keys():
            negative_vecs_by_key[key] = mean_stack([prompt_runs_by_field[other][key] for other in fields if other != field])
        for key in prompt_runs_by_field[field].keys():
            protocol_delta = norm_delta(prompt_runs_by_field[field][key], negative_vecs_by_key[key])
            total += float(fixed_fit_selectivity[field][key] * protocol_delta)
        field_scores[field] = float(total)
    return field_scores


def analyze_model(
    model_name: str,
    model_path: str,
    dtype_name: str,
    prefer_cuda: bool,
    k_values: Sequence[int],
    collapse_threshold: float,
) -> Dict[str, object]:
    mapping = load_mapping_json()["models"][model_name]
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    n_heads = int(getattr(model.config, "num_attention_heads"))
    fields = list(mapping["meta"]["fields"])
    ablator = HeadGroupAblator(model)
    concepts = {}
    t0 = time.time()

    try:
        for concept, entry in mapping["concepts"].items():
            true_field = str(entry["true_field"])
            top_rows = entry["field_scores"][true_field]["top_heads"]
            fixed_fit_selectivity = {
                field: {
                    (int(row["layer"]), int(row["head"])): float(row["fit_selectivity"])
                    for row in entry["field_scores"][field]["top_heads"]
                }
                for field in fields
            }
            # expand to all heads with zeros when missing
            for field in fields:
                for li in range(int(getattr(model.config, "num_hidden_layers"))):
                    for hi in range(n_heads):
                        fixed_fit_selectivity[field].setdefault((li, hi), 0.0)

            baseline_scores = compute_field_scores(model, tok, ablator, concept, fields, None, fixed_fit_selectivity)
            baseline_margin = float(baseline_scores[true_field] - max(v for f, v in baseline_scores.items() if f != true_field))
            scans = {}
            for k in k_values:
                use_k = min(int(k), len(top_rows))
                top_group = [(int(row["layer"]), int(row["head"])) for row in top_rows[:use_k]]
                control_group = build_control_group(top_rows, use_k, n_heads)
                top_scores = compute_field_scores(model, tok, ablator, concept, fields, top_group, fixed_fit_selectivity)
                ctrl_scores = compute_field_scores(model, tok, ablator, concept, fields, control_group, fixed_fit_selectivity)
                top_margin = float(top_scores[true_field] - max(v for f, v in top_scores.items() if f != true_field))
                ctrl_margin = float(ctrl_scores[true_field] - max(v for f, v in ctrl_scores.items() if f != true_field))
                scans[str(k)] = {
                    "top_group": [{"layer": int(l), "head": int(h)} for l, h in top_group],
                    "control_group": [{"layer": int(l), "head": int(h)} for l, h in control_group],
                    "baseline_field_scores": baseline_scores,
                    "top_field_scores": top_scores,
                    "control_field_scores": ctrl_scores,
                    "summary": {
                        "baseline_margin": baseline_margin,
                        "top_margin": top_margin,
                        "control_margin": ctrl_margin,
                        "top_collapse_ratio": float(max(0.0, (baseline_margin - top_margin) / (baseline_margin + 1e-12))),
                        "control_collapse_ratio": float(max(0.0, (baseline_margin - ctrl_margin) / (baseline_margin + 1e-12))),
                        "causal_margin": float((ctrl_margin - top_margin) / (baseline_margin + 1e-12)),
                    },
                }

            passing_k = [
                int(k)
                for k, row in scans.items()
                if float(row["summary"]["top_collapse_ratio"]) >= collapse_threshold
                and float(row["summary"]["top_collapse_ratio"]) > float(row["summary"]["control_collapse_ratio"])
            ]
            concepts[concept] = {
                "true_field": true_field,
                "baseline_margin": baseline_margin,
                "k_scan": scans,
                "boundary_summary": {
                    "collapse_threshold": float(collapse_threshold),
                    "minimal_boundary_k": int(min(passing_k)) if passing_k else None,
                    "best_k_by_causal_margin": int(
                        max(scans.items(), key=lambda kv: float(kv[1]["summary"]["causal_margin"]))[0]
                    ),
                },
            }
    finally:
        ablator.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    global_summary = {
        "k_values": [int(k) for k in k_values],
        "collapse_threshold": float(collapse_threshold),
        "minimal_boundary_histogram": {},
        "runtime_sec": float(time.time() - t0),
    }
    hist = {}
    for row in concepts.values():
        value = row["boundary_summary"]["minimal_boundary_k"]
        key = "none" if value is None else str(value)
        hist[key] = hist.get(key, 0) + 1
    global_summary["minimal_boundary_histogram"] = hist

    return {
        "meta": {
            "model_name": model_name,
            "model_path": model_path,
            "device": str(next(model.parameters()).device),
            "hidden_size": int(getattr(model.config, "hidden_size")),
            "n_layers": int(getattr(model.config, "num_hidden_layers")),
            "n_heads": n_heads,
            "runtime_sec": float(time.time() - t0),
        },
        "concepts": concepts,
        "global_summary": global_summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Search the minimal causal boundary of topology field calls")
    ap.add_argument("--dtype-gpt2", type=str, default="float32")
    ap.add_argument("--dtype-qwen", type=str, default="float16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--k-values", type=str, default="1,2,4,8,16,32")
    ap.add_argument("--collapse-threshold", type=float, default=0.1)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/gpt2_qwen3_protocol_field_boundary_20260308.json",
    )
    args = ap.parse_args()

    k_values = parse_k_values(args.k_values)
    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "models": {}}
    for model_name, model_path in default_model_specs():
        dtype_name = args.dtype_gpt2 if model_name == "gpt2" else args.dtype_qwen
        print(f"[run] {model_name} from {model_path}")
        row = analyze_model(
            model_name,
            model_path,
            dtype_name,
            prefer_cuda=not args.cpu_only,
            k_values=k_values,
            collapse_threshold=float(args.collapse_threshold),
        )
        results["models"][model_name] = row
        print(f"[summary] {model_name} boundary_hist={row['global_summary']['minimal_boundary_histogram']}")

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
