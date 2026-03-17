#!/usr/bin/env python
"""
Expand the clean English noun inventory by mining the DeepSeek tokenizer vocab.

The pipeline stays local:
1) read the existing clean seed inventory
2) mine English-like single-token terms from the tokenizer
3) score them against category centroids built from seed token embeddings
4) export a balanced CSV plus JSONL metadata for later structural scans
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from deepseek7b_three_pool_structure_scan import load_model  # noqa: E402

TERM_RE = re.compile(r"^[a-z][a-z-]{2,23}$")
STOP_TERMS = {
    "the",
    "and",
    "are",
    "was",
    "were",
    "that",
    "this",
    "with",
    "from",
    "into",
    "onto",
    "than",
    "then",
    "when",
    "what",
    "where",
    "while",
    "which",
    "whose",
    "been",
    "being",
    "have",
    "has",
    "had",
    "will",
    "would",
    "shall",
    "could",
    "should",
    "might",
    "must",
    "about",
    "after",
    "before",
    "under",
    "over",
    "again",
    "still",
    "very",
    "more",
    "most",
    "many",
    "much",
    "some",
    "such",
    "each",
    "every",
    "other",
    "another",
    "these",
    "those",
    "their",
    "there",
    "here",
    "your",
    "our",
    "they",
    "them",
    "than",
    "does",
    "doing",
    "done",
    "make",
    "makes",
    "made",
    "take",
    "takes",
    "took",
    "give",
    "gives",
    "given",
    "need",
    "needs",
    "used",
    "using",
}


def read_seed_rows(path: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "," not in s:
            continue
        term, category = [x.strip() for x in s.split(",", 1)]
        if term and category:
            rows.append((term.lower(), category))
    if not rows:
        raise ValueError(f"no usable seed rows found in {path}")
    return rows


def normalize_term(text: str) -> Optional[str]:
    term = text.strip().lower()
    if not term or term in STOP_TERMS:
        return None
    if not TERM_RE.fullmatch(term):
        return None
    if "--" in term:
        return None
    return term


def resolve_single_token_id(tokenizer: AutoTokenizer, term: str) -> Optional[int]:
    for probe in (f" {term}", term):
        ids = tokenizer.encode(probe, add_special_tokens=False)
        if len(ids) != 1:
            continue
        decoded = tokenizer.decode(ids, clean_up_tokenization_spaces=False).strip().lower()
        if decoded == term:
            return int(ids[0])
    return None


def collect_candidate_terms(tokenizer: AutoTokenizer) -> Dict[str, int]:
    discovered: Dict[str, int] = {}
    vocab_len = len(tokenizer)
    for token_id in range(vocab_len):
        raw = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        term = normalize_term(raw)
        if term is None or term in discovered:
            continue
        resolved_id = resolve_single_token_id(tokenizer, term)
        if resolved_id is None:
            continue
        discovered[term] = resolved_id
    return discovered


def l2_normalize(mat: torch.Tensor) -> torch.Tensor:
    denom = torch.linalg.norm(mat, dim=-1, keepdim=True).clamp_min(1e-8)
    return mat / denom


def build_category_centroids(
    embed_weight: torch.Tensor,
    seed_rows: Sequence[Tuple[str, str]],
    tokenizer: AutoTokenizer,
) -> Dict[str, Dict[str, object]]:
    seed_ids_by_category: Dict[str, List[int]] = defaultdict(list)
    unresolved: List[Tuple[str, str]] = []
    for term, category in seed_rows:
        token_id = resolve_single_token_id(tokenizer, term)
        if token_id is None:
            unresolved.append((term, category))
            continue
        seed_ids_by_category[category].append(token_id)
    if unresolved:
        unresolved_preview = ", ".join(f"{term}/{category}" for term, category in unresolved[:8])
        print(f"[warn] unresolved seeds: {len(unresolved)} :: {unresolved_preview}")
    out: Dict[str, Dict[str, object]] = {}
    for category, ids in sorted(seed_ids_by_category.items()):
        uniq_ids = sorted(set(ids))
        vecs = embed_weight[uniq_ids]
        centroid = l2_normalize(vecs.float()).mean(dim=0, keepdim=True)
        out[category] = {
            "seed_token_ids": uniq_ids,
            "seed_count": len(uniq_ids),
            "centroid": l2_normalize(centroid)[0],
        }
    return out


def score_candidates(
    embed_weight: torch.Tensor,
    candidate_term_to_id: Dict[str, int],
    category_centroids: Dict[str, Dict[str, object]],
    seed_terms: Iterable[str],
    batch_size: int,
) -> List[Dict[str, object]]:
    categories = sorted(category_centroids)
    centroid_matrix = torch.stack([category_centroids[c]["centroid"] for c in categories], dim=0)
    seed_term_set = set(seed_terms)
    candidate_terms = [term for term in sorted(candidate_term_to_id) if term not in seed_term_set]
    candidate_ids = [candidate_term_to_id[term] for term in candidate_terms]
    rows: List[Dict[str, object]] = []
    for start in range(0, len(candidate_terms), batch_size):
        end = min(len(candidate_terms), start + batch_size)
        batch_terms = candidate_terms[start:end]
        batch_ids = candidate_ids[start:end]
        batch_vecs = l2_normalize(embed_weight[batch_ids].float())
        sims = batch_vecs @ centroid_matrix.T
        top_vals, top_idx = torch.topk(sims, k=min(3, sims.shape[1]), dim=1)
        for i, term in enumerate(batch_terms):
            score_map = {categories[j]: float(sims[i, j].item()) for j in range(len(categories))}
            top_category = categories[int(top_idx[i, 0].item())]
            second_category = categories[int(top_idx[i, 1].item())] if len(categories) > 1 else top_category
            top_score = float(top_vals[i, 0].item())
            second_score = float(top_vals[i, 1].item()) if top_vals.shape[1] > 1 else top_score
            third_score = float(top_vals[i, 2].item()) if top_vals.shape[1] > 2 else second_score
            rows.append(
                {
                    "record_type": "tokenizer_vocab_candidate",
                    "term": term,
                    "token_id": int(batch_ids[i]),
                    "top_category": top_category,
                    "top_score": top_score,
                    "second_category": second_category,
                    "second_score": second_score,
                    "margin": top_score - second_score,
                    "spread": top_score - third_score,
                    "category_scores": score_map,
                }
            )
    rows.sort(key=lambda x: (-float(x["top_score"]), -float(x["margin"]), str(x["term"])))
    return rows


def select_balanced_inventory(
    candidate_rows: Sequence[Dict[str, object]],
    per_category_target: int,
    min_score: float,
    min_margin: float,
) -> List[Dict[str, object]]:
    by_category: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in candidate_rows:
        if float(row["top_score"]) < min_score:
            continue
        if float(row["margin"]) < min_margin:
            continue
        by_category[str(row["top_category"])].append(dict(row))
    out: List[Dict[str, object]] = []
    for category in sorted(by_category):
        chosen = by_category[category][:per_category_target]
        for rank, row in enumerate(chosen, start=1):
            row["balanced_rank"] = rank
            out.append(row)
    out.sort(key=lambda x: (str(x["top_category"]), int(x["balanced_rank"])))
    return out


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = ["# noun,category"]
    for row in rows:
        lines.append(f"{row['term']},{row['top_category']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_seed_plus_expanded_csv(
    path: Path,
    seed_rows: Sequence[Tuple[str, str]],
    expanded_rows: Sequence[Dict[str, object]],
) -> None:
    seen = set()
    lines = ["# noun,category"]
    for term, category in seed_rows:
        key = (term, category)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{term},{category}")
    for row in expanded_rows:
        key = (str(row["term"]), str(row["top_category"]))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{key[0]},{key[1]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# DeepSeek Tokenizer Vocab Expander Report",
        "",
        f"- Runtime sec: {summary['runtime_sec']:.3f}",
        f"- Seed row count: {summary['seed_row_count']}",
        f"- Candidate term count: {summary['candidate_term_count']}",
        f"- Balanced row count: {summary['balanced_row_count']}",
        f"- Per-category target: {summary['per_category_target']}",
        f"- Min score: {summary['min_score']}",
        f"- Min margin: {summary['min_margin']}",
        "",
        "## Balanced Category Counts",
    ]
    for category, count in summary["balanced_category_counts"].items():
        lines.append(f"- {category}: {count}")
    lines.append("")
    lines.append("## Top Balanced Terms")
    for row in summary["top_balanced_terms"]:
        lines.append(
            f"- {row['top_category']} / {row['term']} / score={row['top_score']:.6f} / "
            f"margin={row['margin']:.6f} / token_id={row['token_id']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_format(path: Path) -> None:
    text = """# Format

- `expanded_vocab.csv`: balanced inventory for downstream structure scans; each row is `term,category`
- `combined_seed_plus_expanded.csv`: seed inventory plus balanced expansions, ready for larger structure scans
- `all_candidates.jsonl`: all mined tokenizer candidates with category score maps
- `balanced_candidates.jsonl`: the confidence-filtered balanced subset with ranks
- `summary.json`: run-level counts, thresholds, and top examples
- `REPORT.md`: short human-readable summary
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Expand clean noun inventory from DeepSeek tokenizer vocab")
    ap.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--seed-file", default="tests/codex/deepseek7b_nouns_english_520_clean.csv")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--per-category-target", type=int, default=150)
    ap.add_argument("--min-score", type=float, default=0.18)
    ap.add_argument("--min-margin", type=float, default=0.01)
    ap.add_argument("--output-dir", default="tempdata/deepseek7b_tokenizer_vocab_expander_1500_20260317")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = read_seed_rows(args.seed_file)
    model, tokenizer, model_ref = load_model(
        model_id=args.model_id,
        dtype_name=args.dtype,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    candidate_term_to_id = collect_candidate_terms(tokenizer)
    embed_weight = model.get_input_embeddings().weight.detach()
    if args.device == "cuda" and not embed_weight.is_cuda:
        embed_weight = embed_weight.to("cuda")

    category_centroids = build_category_centroids(embed_weight, seed_rows, tokenizer)
    candidate_rows = score_candidates(
        embed_weight=embed_weight,
        candidate_term_to_id=candidate_term_to_id,
        category_centroids=category_centroids,
        seed_terms=(term for term, _ in seed_rows),
        batch_size=args.batch_size,
    )
    balanced_rows = select_balanced_inventory(
        candidate_rows=candidate_rows,
        per_category_target=args.per_category_target,
        min_score=args.min_score,
        min_margin=args.min_margin,
    )

    balanced_category_counts = Counter(str(row["top_category"]) for row in balanced_rows)
    top_balanced_terms = sorted(
        balanced_rows,
        key=lambda x: (-float(x["top_score"]), -float(x["margin"]), str(x["term"])),
    )[:40]
    summary = {
        "record_type": "tokenizer_vocab_expander_summary",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": time.time() - t0,
        "model_id": args.model_id,
        "model_ref": model_ref,
        "seed_file": args.seed_file,
        "seed_row_count": len(seed_rows),
        "candidate_term_count": len(candidate_rows),
        "balanced_row_count": len(balanced_rows),
        "combined_row_count": len({(term, category) for term, category in seed_rows})
        + len({(str(row["term"]), str(row["top_category"])) for row in balanced_rows}),
        "per_category_target": args.per_category_target,
        "min_score": args.min_score,
        "min_margin": args.min_margin,
        "balanced_category_counts": dict(sorted((k, int(v)) for k, v in balanced_category_counts.items())),
        "seed_category_counts": dict(sorted((k, int(v["seed_count"])) for k, v in category_centroids.items())),
        "top_balanced_terms": [
            {
                "term": row["term"],
                "top_category": row["top_category"],
                "top_score": row["top_score"],
                "margin": row["margin"],
                "token_id": row["token_id"],
            }
            for row in top_balanced_terms
        ],
    }

    write_csv(out_dir / "expanded_vocab.csv", balanced_rows)
    write_seed_plus_expanded_csv(out_dir / "combined_seed_plus_expanded.csv", seed_rows, balanced_rows)
    write_jsonl(out_dir / "all_candidates.jsonl", candidate_rows)
    write_jsonl(out_dir / "balanced_candidates.jsonl", balanced_rows)
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "REPORT.md", summary)
    write_format(out_dir / "FORMAT.md")

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "summary": str(out_dir / "summary.json"),
                "candidate_term_count": len(candidate_rows),
                "balanced_row_count": len(balanced_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
