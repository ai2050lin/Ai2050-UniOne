from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


ROLE_THRESHOLDS = {
    "anchor": {"min_score": 0.22, "min_margin": 0.04},
    "challenger": {"min_score": 0.20, "min_margin": 0.03},
    "support": {"min_score": 0.20, "min_margin": 0.025},
    "fill": {"min_score": 0.0, "min_margin": 0.0},
}


def read_csv_items(path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            term = row[0].strip()
            if not term or term.startswith("#") or len(row) < 2:
                continue
            category = row[1].strip()
            if category:
                out[category].append(term)
    return dict(out)


def read_json(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def read_focus_manifest(path: str) -> Dict[str, List[Dict[str, object]]]:
    payload = read_json(path)
    return {str(row["category"]): list(row["roles"]) for row in payload["categories"]}


def seed_term_set(seed_file: str) -> Set[Tuple[str, str]]:
    rows = set()
    for category, terms in read_csv_items(seed_file).items():
        for term in terms:
            rows.add((term, category))
    return rows


def build_meta_map(path: str) -> Dict[str, Dict[str, object]]:
    return {str(row["term"]): row for row in read_jsonl(path)}


def build_stage_score_maps(
    records_file: str,
    closure_candidates_file: str,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    deep_score: Dict[Tuple[str, str], float] = {}
    for row in read_jsonl(records_file):
        if str(row.get("pool")) != "deep":
            continue
        key = (str(row["item"]["term"]), str(row["item"]["category"]))
        agg = row.get("aggregate", {})
        deep_score[key] = float(agg.get("prompt_stability_jaccard_mean", 0.0))
    closure_score: Dict[Tuple[str, str], float] = {}
    for row in read_jsonl(closure_candidates_file):
        if str(row.get("pool")) != "closure":
            continue
        key = (str(row["item"]["term"]), str(row["item"]["category"]))
        closure_score[key] = float(row.get("exact_closure_proxy", 0.0)) + max(
            0.0, float(row.get("wrong_family_margin", 0.0))
        )
    return deep_score, closure_score


def audit_focus_term(
    term: str,
    category: str,
    role: str,
    seed_terms: Set[Tuple[str, str]],
    meta_map: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    trusted_seed = (term, category) in seed_terms
    row = meta_map.get(term)
    flags: List[str] = []
    second_category = None
    score = None
    margin = None
    if trusted_seed:
        return {
            "term": term,
            "category": category,
            "role": role,
            "trusted_seed": True,
            "risky": False,
            "flags": flags,
            "top_score": None,
            "margin": None,
            "second_category": None,
        }
    if row is None:
        flags.append("missing_metadata")
    else:
        score = float(row["top_score"])
        margin = float(row["margin"])
        second_category = str(row["second_category"])
        if str(row["top_category"]) != category:
            flags.append("category_mismatch")
        thresholds = ROLE_THRESHOLDS.get(role, ROLE_THRESHOLDS["support"])
        if score < float(thresholds["min_score"]):
            flags.append("low_score")
        if margin < float(thresholds["min_margin"]):
            flags.append("low_margin")
    return {
        "term": term,
        "category": category,
        "role": role,
        "trusted_seed": False,
        "risky": bool(flags),
        "flags": flags,
        "top_score": score,
        "margin": margin,
        "second_category": second_category,
    }


def candidate_rank(
    term: str,
    category: str,
    seed_terms: Set[Tuple[str, str]],
    meta_map: Dict[str, Dict[str, object]],
    deep_score: Dict[Tuple[str, str], float],
    closure_score: Dict[Tuple[str, str], float],
) -> Tuple[float, float, float, float, str]:
    trusted = 1.0 if (term, category) in seed_terms else 0.0
    meta = meta_map.get(term, {})
    score = float(meta.get("top_score", 0.0))
    margin = float(meta.get("margin", 0.0))
    return (
        trusted,
        closure_score.get((term, category), 0.0),
        deep_score.get((term, category), 0.0),
        score + margin,
        term,
    )


def build_replacement_pool(
    category: str,
    source_by_category: Dict[str, List[str]],
    seed_terms: Set[Tuple[str, str]],
    meta_map: Dict[str, Dict[str, object]],
    deep_score: Dict[Tuple[str, str], float],
    closure_score: Dict[Tuple[str, str], float],
) -> List[str]:
    terms = sorted(
        set(source_by_category.get(category, [])),
        key=lambda term: candidate_rank(term, category, seed_terms, meta_map, deep_score, closure_score),
        reverse=True,
    )
    return terms


def find_replacement(
    category: str,
    role: str,
    used_terms: Set[str],
    replacement_pool: Sequence[str],
    seed_terms: Set[Tuple[str, str]],
    meta_map: Dict[str, Dict[str, object]],
) -> Optional[str]:
    for term in replacement_pool:
        if term in used_terms:
            continue
        audit = audit_focus_term(term, category, role, seed_terms, meta_map)
        if not bool(audit["risky"]):
            return term
    return None


def cleanup_plan(
    focus_plan: Dict[str, List[Dict[str, object]]],
    source_by_category: Dict[str, List[str]],
    seed_terms: Set[Tuple[str, str]],
    meta_map: Dict[str, Dict[str, object]],
    deep_score: Dict[Tuple[str, str], float],
    closure_score: Dict[Tuple[str, str], float],
) -> Tuple[Dict[str, List[Dict[str, object]]], List[Dict[str, object]], Dict[str, List[Dict[str, object]]]]:
    cleaned: Dict[str, List[Dict[str, object]]] = {}
    audit_rows: List[Dict[str, object]] = []
    hard_negative_board: Dict[str, List[Dict[str, object]]] = {}
    for category, rows in sorted(focus_plan.items()):
        replacement_pool = build_replacement_pool(
            category,
            source_by_category,
            seed_terms,
            meta_map,
            deep_score,
            closure_score,
        )
        used_terms = {str(row["term"]) for row in rows if (str(row["term"]), category) in seed_terms}
        cleaned_rows: List[Dict[str, object]] = []
        risky_rows: List[Dict[str, object]] = []
        for row in rows:
            term = str(row["term"])
            role = str(row["role"])
            audit = audit_focus_term(term, category, role, seed_terms, meta_map)
            replacement = None
            replaced = False
            final_term = term
            if bool(audit["risky"]):
                replacement = find_replacement(category, role, used_terms, replacement_pool, seed_terms, meta_map)
                if replacement is not None:
                    final_term = replacement
                    replaced = replacement != term
            used_terms.add(final_term)
            cleaned_rows.append({"term": final_term, "category": category, "role": role})
            audit_row = dict(audit)
            audit_row["replacement"] = replacement
            audit_row["replaced"] = replaced
            audit_rows.append(audit_row)
            if bool(audit["risky"]):
                risky_rows.append(
                    {
                        "term": term,
                        "role": role,
                        "flags": list(audit["flags"]),
                        "replacement": replacement,
                        "second_category": audit["second_category"],
                        "top_score": audit["top_score"],
                        "margin": audit["margin"],
                    }
                )
        cleaned[category] = cleaned_rows
        hard_negative_board[category] = risky_rows
    return cleaned, audit_rows, hard_negative_board


def write_focus_items_csv(path: Path, plan: Dict[str, List[Dict[str, object]]]) -> None:
    lines = ["# noun,category"]
    for category in sorted(plan):
        for row in plan[category]:
            lines.append(f"{row['term']},{row['category']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(
    path: Path,
    cleaned: Dict[str, List[Dict[str, object]]],
    audit_rows: Sequence[Dict[str, object]],
    hard_negative_board: Dict[str, List[Dict[str, object]]],
) -> None:
    replaced_rows = [row for row in audit_rows if bool(row["replaced"])]
    risky_rows = [row for row in audit_rows if bool(row["risky"])]
    lines = [
        "# DeepSeek Stage2 Focus Cleanup Report",
        "",
        f"- Category count: {len(cleaned)}",
        f"- Final item count: {sum(len(rows) for rows in cleaned.values())}",
        f"- Risky rows audited: {len(risky_rows)}",
        f"- Replaced rows: {len(replaced_rows)}",
        "",
        "## Replacements",
    ]
    for row in replaced_rows:
        lines.append(
            f"- {row['category']} / {row['role']} / {row['term']} -> {row['replacement']} / "
            f"flags={','.join(row['flags'])}"
        )
    lines.append("")
    lines.append("## Hard Negative Board")
    for category, rows in sorted(hard_negative_board.items()):
        lines.append(f"- {category}: {len(rows)} risky items")
        for row in rows[:8]:
            lines.append(
                f"  - {row['term']} [{row['role']}] / second={row['second_category']} / "
                f"score={row['top_score']} / margin={row['margin']} / flags={','.join(row['flags'])}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean stage2 focus items and build a hard negative board")
    ap.add_argument("--focus-manifest-file", default="tempdata/deepseek7b_stage2_focus_1504_20260317/focus_manifest.json")
    ap.add_argument("--source-items-file", default="tempdata/deepseek7b_tokenizer_vocab_expander_1500_20260317/combined_seed_plus_expanded.csv")
    ap.add_argument("--seed-file", default="tests/codex/deepseek7b_nouns_english_520_clean.csv")
    ap.add_argument("--candidate-metadata-file", default="tempdata/deepseek7b_tokenizer_vocab_expander_1500_20260317/all_candidates.jsonl")
    ap.add_argument("--records-file", default="tempdata/deepseek7b_three_pool_stage1_1504_bf16_20260317/records.jsonl")
    ap.add_argument("--closure-candidates-file", default="tempdata/deepseek7b_three_pool_stage1_1504_bf16_20260317/closure_candidates.jsonl")
    ap.add_argument("--output-dir", default="tempdata/deepseek7b_stage2_focus_cleanup_1504_20260317")
    args = ap.parse_args()

    focus_plan = read_focus_manifest(args.focus_manifest_file)
    source_by_category = read_csv_items(args.source_items_file)
    seed_terms = seed_term_set(args.seed_file)
    meta_map = build_meta_map(args.candidate_metadata_file)
    deep_score, closure_score = build_stage_score_maps(args.records_file, args.closure_candidates_file)

    cleaned, audit_rows, hard_negative_board = cleanup_plan(
        focus_plan=focus_plan,
        source_by_category=source_by_category,
        seed_terms=seed_terms,
        meta_map=meta_map,
        deep_score=deep_score,
        closure_score=closure_score,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_focus_items_csv(out_dir / "cleaned_focus_items.csv", cleaned)
    write_json(
        out_dir / "cleaned_focus_manifest.json",
        {
            "record_type": "stage2_focus_cleanup_manifest",
            "source": {
                "focus_manifest_file": args.focus_manifest_file,
                "source_items_file": args.source_items_file,
                "seed_file": args.seed_file,
                "candidate_metadata_file": args.candidate_metadata_file,
            },
            "categories": [
                {"category": category, "count": len(rows), "roles": rows}
                for category, rows in sorted(cleaned.items())
            ],
        },
    )
    write_json(out_dir / "audit.json", {"record_type": "stage2_focus_cleanup_audit", "rows": audit_rows})
    write_json(
        out_dir / "hard_negative_board.json",
        {"record_type": "stage2_focus_hard_negative_board", "categories": hard_negative_board},
    )
    write_report(out_dir / "REPORT.md", cleaned, audit_rows, hard_negative_board)

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "cleaned_focus_items_file": str(out_dir / "cleaned_focus_items.csv"),
                "audit_file": str(out_dir / "audit.json"),
                "hard_negative_board_file": str(out_dir / "hard_negative_board.json"),
                "replaced_count": sum(1 for row in audit_rows if bool(row["replaced"])),
                "risky_count": sum(1 for row in audit_rows if bool(row["risky"])),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
