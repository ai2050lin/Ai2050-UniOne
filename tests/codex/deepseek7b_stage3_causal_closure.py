from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from deepseek7b_three_pool_structure_scan import (  # noqa: E402
    GateCollector,
    LexemeItem,
    finite_stats,
    gate_spec_for_layer,
    jaccard,
    layer_distribution,
    load_model,
    prompts_for_item,
    run_prompt,
    topk_with_values,
    zero_gate_indices,
)


@dataclass(frozen=True)
class FocusTerm:
    term: str
    category: str
    role: str


def read_json(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str) -> List[Dict[str, object]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def first_token_id(tok, text: str) -> int | None:
    ids = tok(text, add_special_tokens=False).input_ids
    return int(ids[0]) if ids else None


def load_focus_terms(path: str) -> Dict[str, List[FocusTerm]]:
    payload = read_json(path)
    out: Dict[str, List[FocusTerm]] = {}
    for block in payload["categories"]:
        category = str(block["category"])
        out[category] = [
            FocusTerm(term=str(row["term"]), category=category, role=str(row["role"]))
            for row in block["roles"]
        ]
    return out


def build_category_priority(
    closure_candidates: Sequence[Dict[str, object]],
    top_n: int,
) -> List[str]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in closure_candidates:
        if str(row.get("pool")) == "closure":
            grouped[str(row["item"]["category"])].append(row)

    scored = []
    for category, rows in grouped.items():
        mean_margin = float(np.mean([float(r["wrong_family_margin"]) for r in rows])) if rows else 0.0
        positive_count = int(sum(1 for r in rows if float(r["wrong_family_margin"]) > 0.0))
        mean_proxy = float(np.mean([float(r["exact_closure_proxy"]) for r in rows])) if rows else 0.0
        scored.append((category, mean_margin, positive_count, mean_proxy))

    scored.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    return [x[0] for x in scored[: max(0, top_n)]]


def select_terms_for_family(rows: Sequence[FocusTerm], max_terms: int) -> List[FocusTerm]:
    role_order = {"anchor": 0, "challenger": 1, "support": 2, "fill": 3}
    ordered = sorted(rows, key=lambda x: (role_order.get(x.role, 99), x.term))
    return ordered[: max(0, max_terms)]


def register_ablation(model, flat_indices: Sequence[int], d_ff: int):
    by_layer: Dict[int, List[int]] = defaultdict(list)
    for idx in flat_indices:
        layer_idx = int(idx) // d_ff
        neuron_idx = int(idx) % d_ff
        by_layer[layer_idx].append(neuron_idx)

    handles = []
    device = next(model.parameters()).device
    for layer_idx, idxs in by_layer.items():
        idx_tensor = torch.tensor(sorted(set(idxs)), dtype=torch.long, device=device)
        spec = gate_spec_for_layer(model.model.layers[layer_idx])

        def _make_hook(local_idxs: torch.Tensor, local_spec):
            def _hook(_module, _inputs, output):
                return zero_gate_indices(output, local_spec, local_idxs)

            return _hook

        handles.append(spec.module.register_forward_hook(_make_hook(idx_tensor, spec)))
    return handles


def remove_handles(handles) -> None:
    for handle in handles:
        handle.remove()


def sample_random_like(indices: Sequence[int], total_neurons: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    k = min(len(indices), total_neurons)
    if k <= 0:
        return []
    return sorted(rng.sample(range(total_neurons), k))


def compute_signature(
    model,
    tok,
    collector: GateCollector,
    item: LexemeItem,
    top_k: int,
) -> Dict[str, object]:
    prompts = prompts_for_item(item, "closure")
    sum_vec = np.zeros(collector.total_neurons, dtype=np.float64)
    prompt_records = []
    norms = []
    for prompt in prompts:
        collector.reset()
        _ = run_prompt(model, tok, prompt)
        flat = collector.get_flat()
        stats = finite_stats(flat, collector.d_ff)
        if int(stats["nonfinite_count"]) > 0:
            raise RuntimeError(
                f"non-finite activations detected in stage3: term={item.term}, prompt={prompt}, stats={json.dumps(stats, ensure_ascii=False)}"
            )
        sum_vec += flat
        idx, vals = topk_with_values(flat, top_k)
        prompt_records.append(
            {
                "prompt": prompt,
                "top_indices": [int(x) for x in idx.tolist()],
                "top_values": [float(x) for x in vals.tolist()],
            }
        )
        norms.append(float(np.linalg.norm(flat)))

    mean_vec = (sum_vec / max(1, len(prompts))).astype(np.float32)
    sig_idx, sig_vals = topk_with_values(mean_vec, top_k)
    return {
        "signature_top_indices": [int(x) for x in sig_idx.tolist()],
        "signature_top_values": [float(x) for x in sig_vals.tolist()],
        "signature_layer_distribution": layer_distribution(sig_idx, collector.d_ff),
        "mean_prompt_l2_norm": float(np.mean(norms) if norms else 0.0),
        "prompt_records": prompt_records,
    }


def category_readout(model, tok, term: str, correct_category: str, all_categories: Sequence[str]) -> Dict[str, object]:
    prompt = f"The concept {term} belongs to"
    out = run_prompt(model, tok, prompt)
    logits = out.logits[0, -1, :].float().cpu()
    probs = torch.softmax(logits, dim=0)

    masses = []
    for category in all_categories:
        token_id = first_token_id(tok, " " + category)
        mass = float(probs[token_id].item()) if token_id is not None else 0.0
        masses.append({"category": category, "prob": mass, "token_id": token_id})
    masses.sort(key=lambda x: x["prob"], reverse=True)

    correct_mass = next((float(x["prob"]) for x in masses if x["category"] == correct_category), 0.0)
    best_other_mass = max((float(x["prob"]) for x in masses if x["category"] != correct_category), default=0.0)
    return {
        "prompt": prompt,
        "correct_category": correct_category,
        "correct_prob": correct_mass,
        "best_other_prob": best_other_mass,
        "category_margin": float(correct_mass - best_other_mass),
        "top_categories": masses[:5],
    }


def analyze_family_effect(
    item: FocusTerm,
    category_proto: Dict[str, object],
    all_protos_by_category: Dict[str, Dict[str, object]],
    baseline_sig: Dict[str, object],
    ablated_sig: Dict[str, object],
    baseline_readout: Dict[str, object],
    ablated_readout: Dict[str, object],
) -> Dict[str, object]:
    baseline_indices = baseline_sig["signature_top_indices"]
    ablated_indices = ablated_sig["signature_top_indices"]
    same_proto = [int(x) for x in category_proto["prototype_top_indices"]]
    other_overlaps = [
        jaccard(ablated_indices, proto["prototype_top_indices"])
        for category, proto in all_protos_by_category.items()
        if category != item.category
    ]
    same_overlap = jaccard(ablated_indices, same_proto)
    best_other = max(other_overlaps) if other_overlaps else 0.0
    margin = same_overlap - best_other

    base_same_overlap = jaccard(baseline_indices, same_proto)
    base_other_overlaps = [
        jaccard(baseline_indices, proto["prototype_top_indices"])
        for category, proto in all_protos_by_category.items()
        if category != item.category
    ]
    base_best_other = max(base_other_overlaps) if base_other_overlaps else 0.0
    base_margin = base_same_overlap - base_best_other

    return {
        "baseline_same_family_overlap": float(base_same_overlap),
        "baseline_best_other_overlap": float(base_best_other),
        "baseline_margin": float(base_margin),
        "ablated_same_family_overlap": float(same_overlap),
        "ablated_best_other_overlap": float(best_other),
        "ablated_margin": float(margin),
        "margin_drop": float(base_margin - margin),
        "baseline_to_ablated_signature_jaccard": float(jaccard(baseline_indices, ablated_indices)),
        "category_margin_baseline": float(baseline_readout["category_margin"]),
        "category_margin_ablated": float(ablated_readout["category_margin"]),
        "category_margin_drop": float(baseline_readout["category_margin"] - ablated_readout["category_margin"]),
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, summary: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# DeepSeek Stage3 Causal Closure Report",
        "",
        "## Headline",
        f"- Family count: {summary['family_count']}",
        f"- Term count: {summary['term_count']}",
        f"- Intervention records: {summary['intervention_record_count']}",
        f"- Mean shared margin drop: {summary['mean_shared_margin_drop']:.6f}",
        f"- Mean shared-random margin drop: {summary['mean_shared_random_margin_drop']:.6f}",
        f"- Mean specific margin drop: {summary['mean_specific_margin_drop']:.6f}",
        f"- Mean specific-random margin drop: {summary['mean_specific_random_margin_drop']:.6f}",
        f"- Mean shared category margin drop: {summary['mean_shared_category_margin_drop']:.6f}",
        f"- Mean shared-random category margin drop: {summary['mean_shared_random_category_margin_drop']:.6f}",
        f"- Mean specific category margin drop: {summary['mean_specific_category_margin_drop']:.6f}",
        f"- Mean specific-random category margin drop: {summary['mean_specific_random_category_margin_drop']:.6f}",
        "",
        "## Top Shared-Ablation Effects",
    ]
    shared_rows = [r for r in rows if r["intervention"]["kind"] == "family_shared"]
    shared_rows.sort(key=lambda x: x["effects"]["category_margin_drop"], reverse=True)
    for row in shared_rows[:15]:
        lines.append(
            f"- {row['item']['category']} / {row['item']['term']} [{row['item']['role']}]: "
            f"shared margin_drop={row['effects']['margin_drop']:.6f}, "
            f"category_margin_drop={row['effects']['category_margin_drop']:.6f}"
        )

    lines.extend(["", "## Top Specific-Ablation Effects"])
    spec_rows = [r for r in rows if r["intervention"]["kind"] == "item_specific"]
    spec_rows.sort(key=lambda x: x["effects"]["category_margin_drop"], reverse=True)
    for row in spec_rows[:15]:
        lines.append(
            f"- {row['item']['category']} / {row['item']['term']} [{row['item']['role']}]: "
            f"specific margin_drop={row['effects']['margin_drop']:.6f}, "
            f"category_margin_drop={row['effects']['category_margin_drop']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek stage3 causal closure validator")
    ap.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--focus-manifest", default="tempdata/deepseek7b_stage2_focus_20260316/focus_manifest.json")
    ap.add_argument("--stage2-families", default="tempdata/deepseek7b_three_pool_stage2_focus_bf16_20260316/families.jsonl")
    ap.add_argument("--stage2-closure", default="tempdata/deepseek7b_three_pool_stage2_focus_bf16_20260316/closure_candidates.jsonl")
    ap.add_argument("--family-count", type=int, default=4)
    ap.add_argument("--terms-per-family", type=int, default=4)
    ap.add_argument("--shared-k", type=int, default=48)
    ap.add_argument("--specific-k", type=int, default=24)
    ap.add_argument("--signature-top-k", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="tempdata/deepseek7b_stage3_causal_closure_20260316")
    args = ap.parse_args()

    focus_terms = load_focus_terms(args.focus_manifest)
    family_rows = read_jsonl(args.stage2_families)
    closure_rows = read_jsonl(args.stage2_closure)

    selected_categories = build_category_priority(closure_rows, top_n=args.family_count)
    selected_terms: Dict[str, List[FocusTerm]] = {
        category: select_terms_for_family(focus_terms.get(category, []), max_terms=args.terms_per_family)
        for category in selected_categories
    }

    family_proto_map = {
        str(row["category"]): row
        for row in family_rows
        if str(row.get("pool")) == "closure"
    }
    closure_map = {
        (str(row["item"]["category"]), str(row["item"]["term"])): row
        for row in closure_rows
        if str(row.get("pool")) == "closure"
    }
    selected_proto_map = {category: family_proto_map[category] for category in selected_categories if category in family_proto_map}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, tok, model_ref = load_model(
        model_id=args.model_id,
        dtype_name=args.dtype,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    collector = GateCollector(model)
    total_neurons = collector.total_neurons

    intervention_rows = []
    baseline_rows = []
    rng_seed = int(args.seed)

    try:
        for family_idx, category in enumerate(selected_categories):
            category_proto = selected_proto_map.get(category)
            if category_proto is None:
                continue
            shared_indices = [int(x) for x in category_proto.get("shared_neurons", [])[: max(0, args.shared_k)]]
            shared_random = sample_random_like(shared_indices, total_neurons=total_neurons, seed=rng_seed + family_idx * 17 + 1)

            for term_idx, focus_term in enumerate(selected_terms.get(category, [])):
                item = LexemeItem(term=focus_term.term, category=focus_term.category, language="ascii")
                baseline_sig = compute_signature(model, tok, collector, item, top_k=args.signature_top_k)
                baseline_readout = category_readout(model, tok, term=focus_term.term, correct_category=focus_term.category, all_categories=selected_categories)
                closure_row = closure_map.get((focus_term.category, focus_term.term), {})
                item_specific = [int(x) for x in closure_row.get("item_specific_neurons", [])[: max(0, args.specific_k)]]
                item_specific_random = sample_random_like(
                    item_specific,
                    total_neurons=total_neurons,
                    seed=rng_seed + family_idx * 101 + term_idx * 13 + 7,
                )

                baseline_rows.append(
                    {
                        "record_type": "stage3_baseline",
                        "item": {"term": focus_term.term, "category": focus_term.category, "role": focus_term.role},
                        "baseline_signature": baseline_sig,
                        "baseline_readout": baseline_readout,
                        "family_shared_k": len(shared_indices),
                        "item_specific_k": len(item_specific),
                    }
                )

                interventions = [
                    ("family_shared", shared_indices),
                    ("family_shared_random", shared_random),
                    ("item_specific", item_specific),
                    ("item_specific_random", item_specific_random),
                    ("combined", sorted(set(shared_indices) | set(item_specific))),
                ]
                for kind, flat_indices in interventions:
                    handles = register_ablation(model, flat_indices, collector.d_ff) if flat_indices else []
                    try:
                        ablated_sig = compute_signature(model, tok, collector, item, top_k=args.signature_top_k)
                        ablated_readout = category_readout(
                            model,
                            tok,
                            term=focus_term.term,
                            correct_category=focus_term.category,
                            all_categories=selected_categories,
                        )
                    finally:
                        remove_handles(handles)

                    effects = analyze_family_effect(
                        item=focus_term,
                        category_proto=category_proto,
                        all_protos_by_category=selected_proto_map,
                        baseline_sig=baseline_sig,
                        ablated_sig=ablated_sig,
                        baseline_readout=baseline_readout,
                        ablated_readout=ablated_readout,
                    )
                    intervention_rows.append(
                        {
                            "record_type": "stage3_intervention",
                            "item": {"term": focus_term.term, "category": focus_term.category, "role": focus_term.role},
                            "intervention": {
                                "kind": kind,
                                "neuron_count": len(flat_indices),
                                "layer_distribution": layer_distribution(flat_indices, collector.d_ff),
                                "flat_indices": [int(x) for x in flat_indices[:256]],
                            },
                            "effects": effects,
                            "ablated_signature": ablated_sig,
                            "ablated_readout": ablated_readout,
                        }
                    )
    finally:
        collector.close()

    shared_rows = [r for r in intervention_rows if r["intervention"]["kind"] == "family_shared"]
    shared_rand_rows = [r for r in intervention_rows if r["intervention"]["kind"] == "family_shared_random"]
    spec_rows = [r for r in intervention_rows if r["intervention"]["kind"] == "item_specific"]
    spec_rand_rows = [r for r in intervention_rows if r["intervention"]["kind"] == "item_specific_random"]
    summary = {
        "record_type": "stage3_summary",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": float(time.time() - t0),
        "model_id": args.model_id,
        "model_ref": model_ref,
        "family_count": len(selected_categories),
        "term_count": len(baseline_rows),
        "intervention_record_count": len(intervention_rows),
        "selected_categories": selected_categories,
        "mean_shared_margin_drop": float(np.mean([r["effects"]["margin_drop"] for r in shared_rows]) if shared_rows else 0.0),
        "mean_shared_random_margin_drop": float(
            np.mean([r["effects"]["margin_drop"] for r in shared_rand_rows]) if shared_rand_rows else 0.0
        ),
        "mean_specific_margin_drop": float(np.mean([r["effects"]["margin_drop"] for r in spec_rows]) if spec_rows else 0.0),
        "mean_specific_random_margin_drop": float(
            np.mean([r["effects"]["margin_drop"] for r in spec_rand_rows]) if spec_rand_rows else 0.0
        ),
        "mean_shared_category_margin_drop": float(
            np.mean([r["effects"]["category_margin_drop"] for r in shared_rows]) if shared_rows else 0.0
        ),
        "mean_shared_random_category_margin_drop": float(
            np.mean([r["effects"]["category_margin_drop"] for r in shared_rand_rows]) if shared_rand_rows else 0.0
        ),
        "mean_specific_category_margin_drop": float(
            np.mean([r["effects"]["category_margin_drop"] for r in spec_rows]) if spec_rows else 0.0
        ),
        "mean_specific_random_category_margin_drop": float(
            np.mean([r["effects"]["category_margin_drop"] for r in spec_rand_rows]) if spec_rand_rows else 0.0
        ),
    }

    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "baselines.jsonl", baseline_rows)
    write_jsonl(out_dir / "interventions.jsonl", intervention_rows)
    write_report(out_dir / "REPORT.md", summary, intervention_rows)

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "summary": str(out_dir / "summary.json"),
                "baseline_count": len(baseline_rows),
                "intervention_count": len(intervention_rows),
                "selected_categories": selected_categories,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
