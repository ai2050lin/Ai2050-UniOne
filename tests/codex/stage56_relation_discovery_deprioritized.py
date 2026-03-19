from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def category_signal(
    category: str,
    shared_closure_categories: Iterable[str],
    weak_frontier_categories: Iterable[str],
) -> float:
    shared = set(shared_closure_categories)
    weak = set(weak_frontier_categories)
    if category in shared:
        return 1.0
    if category in weak:
        return 0.0
    return 0.5


def word_class_bias(word_class: str) -> Dict[str, float]:
    table = {
        "noun": {"local": 0.22, "bundle": 0.04, "hybrid": 0.04},
        "adjective": {"local": 0.18, "bundle": 0.08, "hybrid": 0.06},
        "verb": {"local": 0.08, "bundle": 0.18, "hybrid": 0.10},
        "adverb": {"local": 0.04, "bundle": 0.20, "hybrid": 0.12},
        "abstract_noun": {"local": 0.05, "bundle": 0.22, "hybrid": 0.10},
        "concept": {"local": 0.03, "bundle": 0.24, "hybrid": 0.10},
    }
    return table.get(word_class, {"local": 0.08, "bundle": 0.08, "hybrid": 0.08})


def discover_relation_mode(
    row: Dict[str, object],
    shared_closure_categories: Iterable[str],
    weak_frontier_categories: Iterable[str],
) -> Dict[str, object]:
    local_score = safe_float(row.get("local_linear_score"))
    bundle_score = safe_float(row.get("path_bundle_score"))
    word_class = str(row.get("word_class", "concept"))
    category = str(row.get("category", ""))
    item_count = len(list(row.get("items", [])))
    margin = local_score - bundle_score
    certainty = abs(margin)
    category_strength = category_signal(category, shared_closure_categories, weak_frontier_categories)
    bias = word_class_bias(word_class)

    discovered_local = clamp01(
        0.68 * local_score
        + 0.16 * max(margin, 0.0)
        + 0.10 * bias["local"]
        + 0.06 * (1.0 if item_count >= 4 else 0.0)
    )
    discovered_bundle = clamp01(
        0.62 * bundle_score
        + 0.18 * max(-margin, 0.0)
        + 0.12 * bias["bundle"]
        + 0.08 * category_strength
    )
    discovered_hybrid = clamp01(
        0.45 * (1.0 - certainty)
        + 0.25 * min(local_score, bundle_score)
        + 0.15 * bias["hybrid"]
        + 0.15 * (0.5 - abs(category_strength - 0.5))
    )

    discovered_scores = {
        "discovered_local_patch": discovered_local,
        "discovered_path_bundle": discovered_bundle,
        "discovered_control_hybrid": discovered_hybrid,
    }
    discovered_mode = max(discovered_scores.items(), key=lambda item: (item[1], item[0]))[0]
    prior_projection = {
        "local_linear": "discovered_local_patch",
        "path_bundle": "discovered_path_bundle",
        "hybrid": "discovered_control_hybrid",
    }.get(str(row.get("interpretation", "")), "")
    enriched = dict(row)
    enriched.update(
        {
            "margin_without_family": margin,
            "certainty_without_family": certainty,
            "category_strength": category_strength,
            "discovered_scores": discovered_scores,
            "discovered_mode": discovered_mode,
            "prior_projection": prior_projection,
            "agrees_with_prior_interpretation": bool(prior_projection == discovered_mode),
        }
    )
    return enriched


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    counts = Counter(str(row.get("discovered_mode", "")) for row in rows)
    by_word_class: Dict[str, Counter[str]] = defaultdict(Counter)
    by_category: Dict[str, Counter[str]] = defaultdict(Counter)
    disagreements: List[Dict[str, object]] = []
    certainty_values: List[float] = []
    agreement_count = 0

    for row in rows:
        word_class = str(row.get("word_class", ""))
        category = str(row.get("category", ""))
        discovered_mode = str(row.get("discovered_mode", ""))
        by_word_class[word_class][discovered_mode] += 1
        by_category[category][discovered_mode] += 1
        certainty_values.append(safe_float(row.get("certainty_without_family")))
        if bool(row.get("agrees_with_prior_interpretation")):
            agreement_count += 1
        else:
            disagreements.append(
                {
                    "items": list(row.get("items", [])),
                    "family": row.get("family", ""),
                    "word_class": word_class,
                    "category": category,
                    "prior_interpretation": row.get("interpretation", ""),
                    "discovered_mode": discovered_mode,
                    "margin_without_family": safe_float(row.get("margin_without_family")),
                }
            )
    disagreements.sort(key=lambda item: abs(safe_float(item.get("margin_without_family"))), reverse=True)
    return {
        "record_type": "stage56_relation_discovery_deprioritized_summary",
        "group_count": len(rows),
        "counts_by_discovered_mode": dict(counts),
        "prior_agreement_ratio": float(agreement_count / len(rows)) if rows else 0.0,
        "mean_certainty_without_family": float(sum(certainty_values) / len(certainty_values)) if certainty_values else 0.0,
        "word_class_frontier": {
            word_class: counter.most_common(1)[0][0] for word_class, counter in sorted(by_word_class.items()) if counter
        },
        "category_frontier": {
            category: counter.most_common(1)[0][0] for category, counter in sorted(by_category.items()) if counter
        },
        "top_disagreements": disagreements[:20],
    }


def write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Stage56 关系图谱去先验化块",
        "",
        f"- group_count: {summary.get('group_count', 0)}",
        f"- prior_agreement_ratio: {summary.get('prior_agreement_ratio', 0.0):.4f}",
        f"- mean_certainty_without_family: {summary.get('mean_certainty_without_family', 0.0):.4f}",
        "",
        "## discovered_mode",
    ]
    for key, value in dict(summary.get("counts_by_discovered_mode", {})).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## word_class_frontier"])
    for key, value in dict(summary.get("word_class_frontier", {})).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## top_disagreements"])
    for row in list(summary.get("top_disagreements", []))[:10]:
        items = "/".join(str(x) for x in row.get("items", []))
        lines.append(
            f"- {items}: prior={row.get('prior_interpretation', '')}, discovered={row.get('discovered_mode', '')}, margin={safe_float(row.get('margin_without_family')):.4f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Discover relation modes with reduced family prior dependence")
    ap.add_argument(
        "--relation-groups-jsonl",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_large_relation_atlas_20260318_2251" / "relation_groups.jsonl"),
    )
    ap.add_argument(
        "--unified-atlas-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_multimodel_language_unified_atlas_20260318_2252" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_relation_discovery_deprioritized_20260318"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    relation_rows = read_jsonl(Path(args.relation_groups_jsonl))
    unified_summary = read_json(Path(args.unified_atlas_summary_json))
    shared_closure = list(unified_summary.get("shared_closure_categories", []))
    weak_frontier = list(dict(unified_summary.get("global_category_frontier", {})).get("bottom", []))
    discovered_rows = [
        discover_relation_mode(row, shared_closure_categories=shared_closure, weak_frontier_categories=weak_frontier)
        for row in relation_rows
    ]
    summary = summarize(discovered_rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "relation_groups_deprioritized.jsonl", discovered_rows)
    write_report(out_dir / "REPORT.md", summary)
    print(json.dumps({"output_dir": str(out_dir), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
