from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from stage56_mass_scan_io import row_term, scan_term_rows


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MASS_JSONS = [
    "tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed101/mass_noun_encoding_scan.json",
    "tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed202/mass_noun_encoding_scan.json",
    "tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed303/mass_noun_encoding_scan.json",
    "tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed404/mass_noun_encoding_scan.json",
    "tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed505/mass_noun_encoding_scan.json",
]

DEFAULT_MACRO_GROUPS = {
    "living_system": ["animal", "fruit", "human", "nature"],
    "artifact_system": ["object", "tech", "vehicle"],
    "environment_system": ["weather", "celestial", "nature"],
    "consumable_system": ["food", "fruit"],
}


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def safe_stdev(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def normalize_key(text: object) -> str:
    return str(text or "").strip().lower()


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa = {int(x) for x in a}
    sb = {int(x) for x in b}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def normalize_layer_profile(dist: Dict[str, object], n_layers: int) -> List[float]:
    values = [0.0 for _ in range(max(1, n_layers))]
    for key, value in (dist or {}).items():
        try:
            idx = int(key)
        except Exception:
            continue
        if 0 <= idx < len(values):
            values[idx] = float(value)
    total = sum(values)
    if total <= 0:
        return values
    return [value / total for value in values]


def layer_peak_band(profile: Sequence[float]) -> str:
    if not profile:
        return "unknown"
    peak_idx = max(range(len(profile)), key=lambda idx: profile[idx])
    if peak_idx < len(profile) / 3:
        return "early"
    if peak_idx < (2 * len(profile)) / 3:
        return "middle"
    return "late"


def build_macro_sets(category_prototypes: Dict[str, set[int]], macro_groups: Dict[str, Sequence[str]]) -> Dict[str, set[int]]:
    out: Dict[str, set[int]] = {}
    for macro_name, categories in macro_groups.items():
        merged: set[int] = set()
        for category in categories:
            merged |= set(category_prototypes.get(normalize_key(category), set()))
        out[macro_name] = merged
    return out


def macro_set_excluding_category(
    macro_name: str,
    category: str,
    category_prototypes: Dict[str, set[int]],
    macro_groups: Dict[str, Sequence[str]],
) -> set[int]:
    merged: set[int] = set()
    for candidate in macro_groups.get(macro_name, []):
        candidate_key = normalize_key(candidate)
        if candidate_key == normalize_key(category):
            continue
        merged |= set(category_prototypes.get(candidate_key, set()))
    return merged


def load_seed_payload(path: Path, macro_groups: Dict[str, Sequence[str]]) -> Dict[str, object]:
    obj = read_json(path)
    noun_rows = scan_term_rows(obj)
    noun_map = {normalize_key(row_term(row)): row for row in noun_rows}
    category_prototypes = {
        normalize_key(category): {int(x) for x in row.get("prototype_top_indices", [])}
        for category, row in (obj.get("category_prototypes") or {}).items()
    }
    macro_sets = build_macro_sets(category_prototypes, macro_groups)
    n_layers = int(((obj.get("config") or {}).get("n_layers")) or 28)
    return {
        "path": str(path),
        "seed_tag": path.parent.name,
        "noun_rows": noun_rows,
        "noun_map": noun_map,
        "category_prototypes": category_prototypes,
        "macro_sets": macro_sets,
        "n_layers": n_layers,
    }


def compute_seed_anchor_rows(seed_payload: Dict[str, object], macro_groups: Dict[str, Sequence[str]]) -> List[Dict[str, object]]:
    noun_rows: List[Dict[str, object]] = list(seed_payload["noun_rows"])
    noun_map: Dict[str, Dict[str, object]] = dict(seed_payload["noun_map"])
    category_prototypes: Dict[str, set[int]] = dict(seed_payload["category_prototypes"])
    macro_sets: Dict[str, set[int]] = dict(seed_payload["macro_sets"])
    n_layers = int(seed_payload["n_layers"])

    rows = []
    by_category: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in noun_rows:
        by_category[normalize_key(row.get("category"))].append(row)

    for row in noun_rows:
        noun = normalize_key(row.get("noun"))
        category = normalize_key(row.get("category"))
        signature = {int(x) for x in row.get("signature_top_indices", [])}
        category_proto = set(category_prototypes.get(category, set()))
        same_category_rows = [other for other in by_category.get(category, []) if normalize_key(other.get("noun")) != noun]
        other_rows = [other for other in noun_rows if normalize_key(other.get("category")) != category]
        same_values = [
            jaccard(signature, {int(x) for x in other.get("signature_top_indices", [])})
            for other in same_category_rows
        ]
        cross_values = [
            jaccard(signature, {int(x) for x in other.get("signature_top_indices", [])})
            for other in other_rows
        ]
        category_macro_candidates = [
            macro_name
            for macro_name, categories in macro_groups.items()
            if category in {normalize_key(value) for value in categories}
        ]
        best_macro_name = ""
        best_macro_jaccard = 0.0
        for macro_name in category_macro_candidates:
            macro_set = macro_set_excluding_category(macro_name, category, category_prototypes, macro_groups)
            score = jaccard(signature, macro_set)
            if score > best_macro_jaccard:
                best_macro_jaccard = score
                best_macro_name = macro_name
        layer_profile = normalize_layer_profile(row.get("signature_layer_distribution", {}), n_layers)
        rows.append(
            {
                "seed_tag": str(seed_payload["seed_tag"]),
                "noun": noun,
                "category": category,
                "signature_size": len(signature),
                "noun_to_category_jaccard": jaccard(signature, category_proto),
                "noun_to_best_macro_jaccard": best_macro_jaccard,
                "best_macro_name": best_macro_name,
                "same_category_mean_jaccard": safe_mean(same_values),
                "cross_category_mean_jaccard": safe_mean(cross_values),
                "same_cross_margin": float(safe_mean(same_values) - safe_mean(cross_values)),
                "layer_peak_band": layer_peak_band(layer_profile),
                "layer_peak_value": max(layer_profile) if layer_profile else 0.0,
            }
        )
    return rows


def aggregate_category_rows(seed_payloads: Sequence[Dict[str, object]], anchor_rows: Sequence[Dict[str, object]], macro_groups: Dict[str, Sequence[str]]) -> List[Dict[str, object]]:
    by_category_seed: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in anchor_rows:
        by_category_seed[(str(row["seed_tag"]), str(row["category"]))].append(row)

    rows = []
    for seed_payload in seed_payloads:
        category_prototypes: Dict[str, set[int]] = dict(seed_payload["category_prototypes"])
        macro_sets: Dict[str, set[int]] = dict(seed_payload["macro_sets"])
        for category, proto_set in sorted(category_prototypes.items()):
            anchor_subset = by_category_seed.get((str(seed_payload["seed_tag"]), category), [])
            macro_candidates = [
                macro_name
                for macro_name, categories in macro_groups.items()
                if category in {normalize_key(value) for value in categories}
            ]
            best_macro_name = ""
            best_macro_jaccard = 0.0
            for macro_name in macro_candidates:
                score = jaccard(proto_set, macro_set_excluding_category(macro_name, category, category_prototypes, macro_groups))
                if score > best_macro_jaccard:
                    best_macro_jaccard = score
                    best_macro_name = macro_name
            rows.append(
                {
                    "seed_tag": str(seed_payload["seed_tag"]),
                    "category": category,
                    "category_to_best_macro_jaccard": best_macro_jaccard,
                    "best_macro_name": best_macro_name,
                    "mean_noun_to_category_jaccard": safe_mean([float(item["noun_to_category_jaccard"]) for item in anchor_subset]),
                    "mean_same_cross_margin": safe_mean([float(item["same_cross_margin"]) for item in anchor_subset]),
                    "mean_layer_peak_value": safe_mean([float(item["layer_peak_value"]) for item in anchor_subset]),
                    "noun_count": len(anchor_subset),
                }
            )
    return rows


def aggregate_cross_seed_rows(anchor_rows: Sequence[Dict[str, object]], seed_payloads: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    noun_seed_signature: Dict[str, List[Tuple[str, set[int], str]]] = defaultdict(list)
    payload_by_seed = {str(payload["seed_tag"]): payload for payload in seed_payloads}
    for row in anchor_rows:
        seed_tag = str(row["seed_tag"])
        noun = str(row["noun"])
        seed_payload = payload_by_seed[seed_tag]
        raw_row = dict(seed_payload["noun_map"])[noun]
        signature = {int(x) for x in raw_row.get("signature_top_indices", [])}
        noun_seed_signature[noun].append((seed_tag, signature, str(row["layer_peak_band"])))

    out = []
    for noun, items in sorted(noun_seed_signature.items()):
        pairwise = [jaccard(left[1], right[1]) for left, right in combinations(items, 2)]
        bands = [item[2] for item in items]
        band_counter = Counter(bands)
        category = normalize_key(payload_by_seed[items[0][0]]["noun_map"][noun].get("category"))
        out.append(
            {
                "noun": noun,
                "category": category,
                "seed_count": len(items),
                "cross_seed_signature_jaccard_mean": safe_mean(pairwise),
                "cross_seed_signature_jaccard_std": safe_stdev(pairwise),
                "dominant_layer_peak_band": band_counter.most_common(1)[0][0] if band_counter else "unknown",
                "layer_peak_band_agreement_ratio": float(max(band_counter.values()) / max(1, len(items))),
            }
        )
    return out


def aggregate_global_summary(anchor_rows: Sequence[Dict[str, object]], category_rows: Sequence[Dict[str, object]], cross_seed_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    positive_margin_rows = [row for row in anchor_rows if float(row["same_cross_margin"]) > 0.0]
    category_macro_rows = [row for row in category_rows if float(row["category_to_best_macro_jaccard"]) > float(row["mean_noun_to_category_jaccard"])]
    categories = sorted({str(row["category"]) for row in anchor_rows})
    layer_band_counter = Counter(str(row["layer_peak_band"]) for row in anchor_rows)
    mean_cross_seed_signature_jaccard = safe_mean([float(row["cross_seed_signature_jaccard_mean"]) for row in cross_seed_rows])
    mean_layer_peak_band_agreement_ratio = safe_mean([float(row["layer_peak_band_agreement_ratio"]) for row in cross_seed_rows])
    return {
        "seed_count": len(sorted({str(row["seed_tag"]) for row in anchor_rows})),
        "category_count": len(categories),
        "noun_record_count": len(anchor_rows),
        "unique_noun_count": len(sorted({str(row["noun"]) for row in anchor_rows})),
        "mean_noun_to_category_jaccard": safe_mean([float(row["noun_to_category_jaccard"]) for row in anchor_rows]),
        "mean_noun_to_best_macro_jaccard": safe_mean([float(row["noun_to_best_macro_jaccard"]) for row in anchor_rows]),
        "mean_same_category_jaccard": safe_mean([float(row["same_category_mean_jaccard"]) for row in anchor_rows]),
        "mean_cross_category_jaccard": safe_mean([float(row["cross_category_mean_jaccard"]) for row in anchor_rows]),
        "mean_same_cross_margin": safe_mean([float(row["same_cross_margin"]) for row in anchor_rows]),
        "positive_same_cross_margin_ratio": float(len(positive_margin_rows) / max(1, len(anchor_rows))),
        "mean_category_to_best_macro_jaccard": safe_mean([float(row["category_to_best_macro_jaccard"]) for row in category_rows]),
        "macro_stronger_than_micro_category_ratio": float(len(category_macro_rows) / max(1, len(category_rows))),
        "mean_cross_seed_signature_jaccard": mean_cross_seed_signature_jaccard,
        "mean_layer_peak_band_agreement_ratio": mean_layer_peak_band_agreement_ratio,
        "seed_degeneracy_warning": bool(mean_cross_seed_signature_jaccard >= 0.999999 and mean_layer_peak_band_agreement_ratio >= 0.999999),
        "layer_peak_band_distribution": dict(sorted(layer_band_counter.items())),
    }


def build_hypotheses(global_summary: Dict[str, object]) -> List[Dict[str, object]]:
    return [
        {
            "id": "H1_same_category_separation_positive",
            "rule": "positive_same_cross_margin_ratio > 0.95",
            "pass": bool(float(global_summary["positive_same_cross_margin_ratio"]) > 0.95),
        },
        {
            "id": "H2_category_to_macro_stronger_than_noun_to_category",
            "rule": "macro_stronger_than_micro_category_ratio > 0.6",
            "pass": bool(float(global_summary["macro_stronger_than_micro_category_ratio"]) > 0.6),
        },
        {
            "id": "H3_cross_seed_signature_has_stability",
            "rule": "mean_cross_seed_signature_jaccard > 0.15",
            "pass": bool(float(global_summary["mean_cross_seed_signature_jaccard"]) > 0.15),
        },
        {
            "id": "H4_same_category_mean_exceeds_cross_category_mean",
            "rule": "mean_same_category_jaccard > mean_cross_category_jaccard",
            "pass": bool(float(global_summary["mean_same_category_jaccard"]) > float(global_summary["mean_cross_category_jaccard"])),
        },
        {
            "id": "H5_seed_diversity_is_nontrivial",
            "rule": "seed_degeneracy_warning == False",
            "pass": bool(not global_summary["seed_degeneracy_warning"]),
        },
    ]


def write_report(
    path: Path,
    global_summary: Dict[str, object],
    hypotheses: Sequence[Dict[str, object]],
    category_rows: Sequence[Dict[str, object]],
    cross_seed_rows: Sequence[Dict[str, object]],
) -> None:
    category_mean: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in category_rows:
        category = str(row["category"])
        category_mean[category]["micro"].append(float(row["mean_noun_to_category_jaccard"]))
        category_mean[category]["macro"].append(float(row["category_to_best_macro_jaccard"]))
        category_mean[category]["margin"].append(float(row["mean_same_cross_margin"]))

    noun_stability_rows = sorted(
        cross_seed_rows,
        key=lambda row: (float(row["cross_seed_signature_jaccard_mean"]), float(row["layer_peak_band_agreement_ratio"])),
        reverse=True,
    )

    lines = [
        "# ICSPB 大样本概念规律扫描报告",
        "",
        "## 全局指标",
        f"- seed_count: {global_summary['seed_count']}",
        f"- category_count: {global_summary['category_count']}",
        f"- noun_record_count: {global_summary['noun_record_count']}",
        f"- unique_noun_count: {global_summary['unique_noun_count']}",
        f"- mean_noun_to_category_jaccard: {global_summary['mean_noun_to_category_jaccard']:.6f}",
        f"- mean_noun_to_best_macro_jaccard: {global_summary['mean_noun_to_best_macro_jaccard']:.6f}",
        f"- mean_same_category_jaccard: {global_summary['mean_same_category_jaccard']:.6f}",
        f"- mean_cross_category_jaccard: {global_summary['mean_cross_category_jaccard']:.6f}",
        f"- mean_same_cross_margin: {global_summary['mean_same_cross_margin']:.6f}",
        f"- positive_same_cross_margin_ratio: {global_summary['positive_same_cross_margin_ratio']:.6f}",
        f"- mean_category_to_best_macro_jaccard: {global_summary['mean_category_to_best_macro_jaccard']:.6f}",
        f"- macro_stronger_than_micro_category_ratio: {global_summary['macro_stronger_than_micro_category_ratio']:.6f}",
        f"- mean_cross_seed_signature_jaccard: {global_summary['mean_cross_seed_signature_jaccard']:.6f}",
        f"- mean_layer_peak_band_agreement_ratio: {global_summary['mean_layer_peak_band_agreement_ratio']:.6f}",
        "",
        "## 类别均值",
    ]
    for category in sorted(category_mean):
        lines.append(
            "- "
            f"{category}: micro={safe_mean(category_mean[category]['micro']):.6f}, "
            f"macro={safe_mean(category_mean[category]['macro']):.6f}, "
            f"margin={safe_mean(category_mean[category]['margin']):.6f}"
        )
    lines.extend(["", "## 跨种子稳定性 Top10"])
    for row in noun_stability_rows[:10]:
        lines.append(
            "- "
            f"{row['noun']} / {row['category']} / stability={row['cross_seed_signature_jaccard_mean']:.6f} "
            f"/ band={row['dominant_layer_peak_band']} / agreement={row['layer_peak_band_agreement_ratio']:.6f}"
        )
    lines.extend(["", "## 假设判定"])
    for row in hypotheses:
        lines.append(f"- {row['id']}: {'PASS' if row['pass'] else 'FAIL'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a large-scale ICSPB concept law scan over multi-seed mass noun results")
    ap.add_argument("--mass-json", action="append", default=[])
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_icspb_large_scale_concept_law_scan_20260318"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    mass_jsons = [Path(item) for item in (args.mass_json or DEFAULT_MASS_JSONS)]
    seed_payloads = [load_seed_payload(path, DEFAULT_MACRO_GROUPS) for path in mass_jsons]
    anchor_rows = []
    for payload in seed_payloads:
        anchor_rows.extend(compute_seed_anchor_rows(payload, DEFAULT_MACRO_GROUPS))
    category_rows = aggregate_category_rows(seed_payloads, anchor_rows, DEFAULT_MACRO_GROUPS)
    cross_seed_rows = aggregate_cross_seed_rows(anchor_rows, seed_payloads)
    global_summary = aggregate_global_summary(anchor_rows, category_rows, cross_seed_rows)
    hypotheses = build_hypotheses(global_summary)

    payload = {
        "record_type": "stage56_icspb_large_scale_concept_law_scan",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": float(time.time() - t0),
        "config": {
            "mass_jsons": [str(path) for path in mass_jsons],
            "macro_groups": DEFAULT_MACRO_GROUPS,
        },
        "global_summary": global_summary,
        "hypotheses": hypotheses,
        "category_rows": category_rows,
        "cross_seed_rows": cross_seed_rows,
        "anchor_rows": anchor_rows,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir / "REPORT.md", global_summary, hypotheses, category_rows, cross_seed_rows)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "summary_json": str(out_dir / "summary.json"),
                "report_md": str(out_dir / "REPORT.md"),
                "noun_record_count": global_summary["noun_record_count"],
                "category_count": global_summary["category_count"],
                "seed_count": global_summary["seed_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
