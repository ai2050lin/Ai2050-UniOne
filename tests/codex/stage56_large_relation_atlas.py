from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for raw_row in reader:
            if not raw_row:
                continue
            first = str(raw_row[0]).strip()
            if not first:
                continue
            if first.startswith("#"):
                continue
            if len(raw_row) < 2:
                continue
            rows.append({"term": str(raw_row[0]).strip(), "category": str(raw_row[1]).strip()})
    return rows


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def average(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def parse_category_metrics(summary: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for row in summary.get("top_consensus_categories", []):
        category = str(row.get("category", ""))
        out[category] = {
            "strict_positive_pair_ratio": safe_float(row.get("strict_positive_pair_ratio")),
            "mean_union_joint_adv": safe_float(row.get("mean_union_joint_adv")),
            "mean_union_synergy_joint": safe_float(row.get("mean_union_synergy_joint")),
        }
    return out


def build_manual_relation_specs() -> List[Dict[str, object]]:
    return [
        {"family": "gender_role_swap", "kind": "quadruplet", "word_class": "noun", "items": ["king", "man", "woman", "queen"], "category": "human"},
        {"family": "gender_role_swap", "kind": "quadruplet", "word_class": "noun", "items": ["prince", "man", "woman", "princess"], "category": "human"},
        {"family": "gender_role_swap", "kind": "quadruplet", "word_class": "noun", "items": ["actor", "man", "woman", "actress"], "category": "human"},
        {"family": "gender_role_swap", "kind": "quadruplet", "word_class": "noun", "items": ["waiter", "man", "woman", "waitress"], "category": "human"},
        {"family": "gender_role_swap", "kind": "quadruplet", "word_class": "noun", "items": ["hero", "man", "woman", "heroine"], "category": "human"},
        {"family": "profession_role_swap", "kind": "quadruplet", "word_class": "noun", "items": ["teacher", "student", "mentor", "apprentice"], "category": "human"},
        {"family": "profession_role_swap", "kind": "quadruplet", "word_class": "noun", "items": ["doctor", "patient", "lawyer", "client"], "category": "human"},
        {"family": "profession_role_swap", "kind": "quadruplet", "word_class": "noun", "items": ["captain", "crew", "teacher", "class"], "category": "human"},
        {"family": "adjective_polarity", "kind": "triplet", "word_class": "adjective", "items": ["hot", "cold", "warm"], "category": "nature"},
        {"family": "adjective_polarity", "kind": "triplet", "word_class": "adjective", "items": ["bright", "dark", "dim"], "category": "celestial"},
        {"family": "adjective_polarity", "kind": "triplet", "word_class": "adjective", "items": ["sweet", "bitter", "sour"], "category": "food"},
        {"family": "adjective_polarity", "kind": "triplet", "word_class": "adjective", "items": ["clean", "dirty", "pure"], "category": "object"},
        {"family": "adjective_polarity", "kind": "triplet", "word_class": "adjective", "items": ["young", "old", "ancient"], "category": "human"},
        {"family": "adjective_degree", "kind": "triplet", "word_class": "adjective", "items": ["small", "smaller", "smallest"], "category": "object"},
        {"family": "adjective_degree", "kind": "triplet", "word_class": "adjective", "items": ["fast", "faster", "fastest"], "category": "vehicle"},
        {"family": "adjective_degree", "kind": "triplet", "word_class": "adjective", "items": ["deep", "deeper", "deepest"], "category": "nature"},
        {"family": "adjective_degree", "kind": "triplet", "word_class": "adjective", "items": ["near", "nearer", "nearest"], "category": "object"},
        {"family": "verb_antonym", "kind": "triplet", "word_class": "verb", "items": ["open", "close", "lock"], "category": "action"},
        {"family": "verb_antonym", "kind": "triplet", "word_class": "verb", "items": ["create", "destroy", "build"], "category": "action"},
        {"family": "verb_process_chain", "kind": "quadruplet", "word_class": "verb", "items": ["plan", "start", "move", "finish"], "category": "action"},
        {"family": "verb_process_chain", "kind": "quadruplet", "word_class": "verb", "items": ["think", "reason", "compare", "solve"], "category": "action"},
        {"family": "abstract_duality", "kind": "triplet", "word_class": "abstract_noun", "items": ["justice", "law", "mercy"], "category": "abstract"},
        {"family": "abstract_duality", "kind": "triplet", "word_class": "abstract_noun", "items": ["order", "chaos", "balance"], "category": "abstract"},
        {"family": "abstract_duality", "kind": "triplet", "word_class": "abstract_noun", "items": ["meaning", "value", "purpose"], "category": "abstract"},
        {"family": "abstract_duality", "kind": "triplet", "word_class": "abstract_noun", "items": ["truth", "belief", "evidence"], "category": "abstract"},
        {"family": "abstract_duality", "kind": "triplet", "word_class": "abstract_noun", "items": ["freedom", "power", "responsibility"], "category": "abstract"},
        {"family": "adverb_manner", "kind": "triplet", "word_class": "adverb", "items": ["quickly", "slowly", "carefully"], "category": "action"},
        {"family": "adverb_manner", "kind": "triplet", "word_class": "adverb", "items": ["formally", "casually", "logically"], "category": "abstract"},
        {"family": "adverb_manner", "kind": "triplet", "word_class": "adverb", "items": ["openly", "secretly", "publicly"], "category": "human"},
        {"family": "adverb_manner", "kind": "triplet", "word_class": "adverb", "items": ["precisely", "roughly", "approximately"], "category": "tech"},
        {"family": "adverb_manner", "kind": "triplet", "word_class": "adverb", "items": ["gently", "firmly", "softly"], "category": "action"},
        {"family": "protocol_role", "kind": "quadruplet", "word_class": "concept", "items": ["protocol", "client", "thread", "algorithm"], "category": "tech"},
        {"family": "protocol_role", "kind": "quadruplet", "word_class": "concept", "items": ["teacher", "engineer", "captain", "miner"], "category": "human"},
        {"family": "protocol_role", "kind": "quadruplet", "word_class": "concept", "items": ["create", "help", "explore", "watch"], "category": "action"},
    ]


def build_inventory_relations(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    by_category: Dict[str, List[str]] = {}
    for row in rows:
        term = str(row.get("term", "")).strip()
        category = str(row.get("category", "")).strip()
        if term and category:
            by_category.setdefault(category, []).append(term)

    records: List[Dict[str, object]] = []
    for category, terms in sorted(by_category.items()):
        ordered = sorted(dict.fromkeys(terms))
        for index in range(0, max(0, len(ordered) - 2), 2):
            items = ordered[index : index + 3]
            if len(items) == 3:
                records.append(
                    {
                        "family": "category_instance_triplet",
                        "kind": "triplet",
                        "word_class": "noun" if category not in {"abstract", "action"} else ("abstract_noun" if category == "abstract" else "verb"),
                        "items": [category] + items[:2],
                        "category": category,
                    }
                )
        for index in range(0, max(0, len(ordered) - 3), 3):
            items = ordered[index : index + 4]
            if len(items) == 4:
                records.append(
                    {
                        "family": "category_instance_quadruplet",
                        "kind": "quadruplet",
                        "word_class": "noun" if category not in {"abstract", "action"} else ("abstract_noun" if category == "abstract" else "verb"),
                        "items": [category] + items[:3],
                        "category": category,
                    }
                )
    return records


def family_priors(family: str) -> Dict[str, float]:
    priors = {
        "gender_role_swap": {"linear": 0.95, "bundle": 0.20, "symmetry": 0.95, "transport": 0.20},
        "profession_role_swap": {"linear": 0.82, "bundle": 0.42, "symmetry": 0.85, "transport": 0.45},
        "adjective_polarity": {"linear": 0.78, "bundle": 0.35, "symmetry": 0.82, "transport": 0.30},
        "adjective_degree": {"linear": 0.74, "bundle": 0.42, "symmetry": 0.78, "transport": 0.36},
        "verb_antonym": {"linear": 0.62, "bundle": 0.48, "symmetry": 0.72, "transport": 0.44},
        "verb_process_chain": {"linear": 0.35, "bundle": 0.82, "symmetry": 0.28, "transport": 0.88},
        "abstract_duality": {"linear": 0.55, "bundle": 0.62, "symmetry": 0.56, "transport": 0.65},
        "adverb_manner": {"linear": 0.52, "bundle": 0.66, "symmetry": 0.46, "transport": 0.72},
        "protocol_role": {"linear": 0.18, "bundle": 0.95, "symmetry": 0.22, "transport": 0.96},
        "category_instance_triplet": {"linear": 0.18, "bundle": 0.92, "symmetry": 0.18, "transport": 0.90},
        "category_instance_quadruplet": {"linear": 0.20, "bundle": 0.94, "symmetry": 0.20, "transport": 0.92},
    }
    return priors.get(family, {"linear": 0.5, "bundle": 0.5, "symmetry": 0.5, "transport": 0.5})


def class_pressure(word_class: str, protocol_gap: float, abstract_penalty: float) -> float:
    if word_class == "adjective":
        return 0.72
    if word_class == "verb":
        return 0.68
    if word_class == "adverb":
        return 0.76
    if word_class == "abstract_noun":
        return clamp01(0.62 + abstract_penalty)
    return clamp01(0.40 + protocol_gap)


def classify_relation_record(
    record: Dict[str, object],
    axis_specificity: float,
    hierarchy_gain: float,
    decoupling: float,
    protocol_gap: float,
    category_metrics: Dict[str, Dict[str, float]],
    abstract_penalty: float,
) -> Dict[str, object]:
    family = str(record["family"])
    category = str(record.get("category", ""))
    word_class = str(record.get("word_class", "concept"))
    priors = family_priors(family)
    category_row = category_metrics.get(
        category,
        {"strict_positive_pair_ratio": 0.0, "mean_union_joint_adv": 0.0, "mean_union_synergy_joint": 0.0},
    )
    closure_support = clamp01(
        0.5
        + 3.0 * safe_float(category_row.get("mean_union_joint_adv"))
        + 2.0 * safe_float(category_row.get("mean_union_synergy_joint"))
        + safe_float(category_row.get("strict_positive_pair_ratio"))
    )
    local_linear_score = clamp01(
        0.38 * axis_specificity * priors["linear"]
        + 0.28 * priors["symmetry"]
        + 0.18 * closure_support
        + 0.16 * (1.0 - class_pressure(word_class, protocol_gap, abstract_penalty))
    )
    path_bundle_score = clamp01(
        0.34 * hierarchy_gain
        + 0.26 * decoupling * priors["bundle"]
        + 0.20 * priors["transport"]
        + 0.20 * class_pressure(word_class, protocol_gap, abstract_penalty)
    )
    if local_linear_score - path_bundle_score > 0.08:
        interpretation = "local_linear"
    elif path_bundle_score - local_linear_score > 0.08:
        interpretation = "path_bundle"
    else:
        interpretation = "hybrid"
    out = dict(record)
    out["local_linear_score"] = local_linear_score
    out["path_bundle_score"] = path_bundle_score
    out["interpretation"] = interpretation
    return out


def count_by(rows: Sequence[Dict[str, object]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def summarize(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    local_rows = [row for row in records if row["interpretation"] == "local_linear"]
    bundle_rows = [row for row in records if row["interpretation"] == "path_bundle"]
    hybrid_rows = [row for row in records if row["interpretation"] == "hybrid"]
    return {
        "record_type": "stage56_large_relation_atlas_summary",
        "group_count": len(records),
        "counts_by_interpretation": count_by(records, "interpretation"),
        "counts_by_word_class": count_by(records, "word_class"),
        "counts_by_family": count_by(records, "family"),
        "mean_local_linear_score": average([safe_float(row["local_linear_score"]) for row in records]),
        "mean_path_bundle_score": average([safe_float(row["path_bundle_score"]) for row in records]),
        "top_local_linear_examples": sorted(local_rows, key=lambda row: safe_float(row["local_linear_score"]), reverse=True)[:12],
        "top_path_bundle_examples": sorted(bundle_rows, key=lambda row: safe_float(row["path_bundle_score"]), reverse=True)[:12],
        "hybrid_examples": sorted(hybrid_rows, key=lambda row: abs(safe_float(row["local_linear_score"]) - safe_float(row["path_bundle_score"])))[:12],
    }


def write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Stage56 大规模关系图谱块",
        "",
        f"- group_count: {int(summary['group_count'])}",
        f"- counts_by_interpretation: {summary['counts_by_interpretation']}",
        f"- counts_by_word_class: {summary['counts_by_word_class']}",
        "",
        "## 最强局部线性样例",
    ]
    for row in summary["top_local_linear_examples"]:
        lines.append(
            f"- {row['family']} / {row['word_class']} / {row['items']}: "
            f"linear={safe_float(row['local_linear_score']):.4f}, bundle={safe_float(row['path_bundle_score']):.4f}"
        )
    lines.extend(["", "## 最强路径束样例"])
    for row in summary["top_path_bundle_examples"]:
        lines.append(
            f"- {row['family']} / {row['word_class']} / {row['items']}: "
            f"linear={safe_float(row['local_linear_score']):.4f}, bundle={safe_float(row['path_bundle_score']):.4f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a large triplet/quadruplet relation atlas and classify local linear vs path-bundle regimes")
    ap.add_argument("--inventory-csv", default=str(ROOT / "tests" / "codex_temp" / "stage56_icspb_expanded_inventory_20260318_1525" / "items.csv"))
    ap.add_argument("--triplet-json", default=str(ROOT / "tempdata" / "deepseek7b_triplet_probe_20260306_150637" / "apple_king_queen_triplet_probe.json"))
    ap.add_argument("--apple-dossier-json", default=str(ROOT / "tempdata" / "deepseek7b_apple_encoding_law_dossier_20260306_223055" / "apple_multiaxis_encoding_law_dossier.json"))
    ap.add_argument("--discovery-summary-json", default=str(ROOT / "tempdata" / "stage56_mass_term_large_seq_20260318_1540" / "discovery_summary.json"))
    ap.add_argument("--protocol-summary-json", default=str(ROOT / "tests" / "codex_temp" / "stage56_protocol_role_encoding_block_20260318_2218" / "summary.json"))
    ap.add_argument("--output-dir", default=str(ROOT / "tests" / "codex_temp" / "stage56_large_relation_atlas_20260318"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    inventory_rows = read_csv_rows(Path(args.inventory_csv))
    triplet_json = read_json(Path(args.triplet_json))
    apple_dossier = read_json(Path(args.apple_dossier_json))
    discovery_summary = read_json(Path(args.discovery_summary_json))
    protocol_summary = read_json(Path(args.protocol_summary_json))

    triplet_metrics = dict(triplet_json.get("metrics", {}))
    apple_metrics = dict(apple_dossier.get("metrics", {}))
    axis_specificity = safe_float(triplet_metrics.get("axis_specificity_index"))
    hierarchy_gain = clamp01(safe_float(apple_metrics.get("apple_meso_to_macro_jaccard_mean")) - safe_float(apple_metrics.get("apple_micro_to_meso_jaccard_mean")))
    decoupling = clamp01(safe_float(apple_metrics.get("cross_dim_decoupling_index")))
    protocol_gap = clamp01(safe_float(protocol_summary.get("focus_mean_protocol_role_pressure")) - safe_float(protocol_summary.get("anchor_mean_protocol_role_pressure")))
    category_metrics = parse_category_metrics(discovery_summary)
    abstract_penalty = clamp01(abs(safe_float(category_metrics.get("abstract", {}).get("mean_union_synergy_joint", 0.0))))

    records = build_manual_relation_specs() + build_inventory_relations(inventory_rows)
    classified = [
        classify_relation_record(row, axis_specificity, hierarchy_gain, decoupling, protocol_gap, category_metrics, abstract_penalty)
        for row in records
    ]
    summary = summarize(classified)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "relation_groups.jsonl", classified)
    write_report(out_dir / "REPORT.md", summary)
    print(json.dumps({"output_dir": str(out_dir), "group_count": summary["group_count"], "counts_by_interpretation": summary["counts_by_interpretation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
