from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
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


def average(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def find_category_rows(rows: Sequence[Dict[str, object]], category: str) -> List[Dict[str, object]]:
    return [row for row in rows if str(row.get("category", "")) == category]


def load_selected_vocab_candidates(path: Path, terms: Sequence[str]) -> Dict[str, Dict[str, object]]:
    targets = set(terms)
    out: Dict[str, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            term = str(row.get("term", ""))
            if term in targets:
                out[term] = row
            if len(out) == len(targets):
                break
    return out


def build_adjective_probe(micro_json: Dict[str, object], apple_compare: Dict[str, object]) -> Dict[str, object]:
    apple = dict(micro_json.get("concepts", {})).get("apple", {})
    role_subsets = dict(apple.get("role_subsets", {}))
    entity = dict(role_subsets.get("entity", {}))
    size = dict(role_subsets.get("size", {}))
    weight = dict(role_subsets.get("weight", {}))
    causal = dict(apple_compare.get("causal_ablation", {}))
    attribute_drop = average([safe_float(size.get("drop_target")), safe_float(weight.get("drop_target"))])
    anchor_drop = safe_float(entity.get("drop_target"))
    causal_delta = abs(safe_float(causal.get("apple_attr", {}).get("delta")))
    fiber_index = attribute_drop / max(anchor_drop, 1e-9)
    return {
        "word_class": "adjective",
        "probe_strength": "direct",
        "primary_mechanism": "modifier_fiber",
        "fiber_index": fiber_index,
        "causal_delta_abs": causal_delta,
        "evidence": {
            "size_drop_target": safe_float(size.get("drop_target")),
            "weight_drop_target": safe_float(weight.get("drop_target")),
            "entity_drop_target": anchor_drop,
            "apple_attr_ablation_delta": safe_float(causal.get("apple_attr", {}).get("delta")),
        },
        "judgement": "adjective_as_fiber" if fiber_index > 10.0 and causal_delta > 0.0 else "weak",
    }


def build_verb_probe(category_rows: Sequence[Dict[str, object]], protocol_summary: Dict[str, object]) -> Dict[str, object]:
    action_rows = find_category_rows(category_rows, "action")
    per_category = dict(protocol_summary.get("per_category", {}))
    action_block = dict(per_category.get("action", {}))
    strict_positive = average([safe_float(row.get("strict_positive_pair_ratio")) for row in action_rows])
    synergy = average([safe_float(row.get("mean_union_synergy_joint")) for row in action_rows])
    union = average([safe_float(row.get("mean_union_joint_adv")) for row in action_rows])
    return {
        "word_class": "verb",
        "probe_strength": "direct",
        "primary_mechanism": "transport_operator_with_anchor_residue",
        "transport_index": union - min(synergy, 0.0),
        "closure_ratio": strict_positive,
        "protocol_pressure": safe_float(action_block.get("mean_protocol_role_pressure")),
        "judgement": "verb_transport" if union > 0.0 and strict_positive >= 0.3 else "weak",
        "evidence": {
            "mean_union_joint_adv": union,
            "mean_union_synergy_joint": synergy,
            "strict_positive_pair_ratio": strict_positive,
            "encoding_class": str(action_block.get("encoding_class", "")),
        },
    }


def build_abstract_probe(category_rows: Sequence[Dict[str, object]], protocol_summary: Dict[str, object]) -> Dict[str, object]:
    abstract_rows = find_category_rows(category_rows, "abstract")
    per_category = dict(protocol_summary.get("per_category", {}))
    block = dict(per_category.get("human", {}))
    top_terms = sorted({str(row.get("top_instance_term", "")) for row in abstract_rows if row.get("top_instance_term")})
    strict_positive = average([safe_float(row.get("strict_positive_pair_ratio")) for row in abstract_rows])
    synergy = average([safe_float(row.get("mean_union_synergy_joint")) for row in abstract_rows])
    divergence = len(top_terms) / max(1, len(abstract_rows))
    return {
        "word_class": "abstract_noun",
        "probe_strength": "direct",
        "primary_mechanism": "relation_bundle",
        "abstraction_index": abs(synergy) + divergence,
        "closure_ratio": strict_positive,
        "top_terms": top_terms,
        "human_protocol_reference": safe_float(block.get("mean_protocol_role_pressure")),
        "judgement": "abstract_bundle" if divergence >= 1.0 and synergy < 0.0 else "weak",
        "evidence": {
            "mean_union_synergy_joint": synergy,
            "strict_positive_pair_ratio": strict_positive,
            "model_term_divergence": divergence,
        },
    }


def build_adverb_probe(vocab_candidates: Dict[str, Dict[str, object]], multidim_json: Dict[str, object]) -> Dict[str, object]:
    margins = [safe_float(row.get("margin")) for row in vocab_candidates.values()]
    spreads = [safe_float(row.get("spread")) for row in vocab_candidates.values()]
    top_categories = sorted({str(row.get("top_category", "")) for row in vocab_candidates.values() if row.get("top_category")})
    style_block = dict(dict(multidim_json.get("dimensions", {})).get("style", {}))
    return {
        "word_class": "adverb",
        "probe_strength": "semi_direct",
        "primary_mechanism": "control_axis_modifier",
        "ambiguity_index": 1.0 - average(margins),
        "spread_index": average(spreads),
        "style_prompt_delta_l2": safe_float(style_block.get("mean_pair_delta_l2")),
        "top_categories": top_categories,
        "judgement": "adverb_control_modifier" if average(margins) < 0.02 and safe_float(style_block.get("mean_pair_delta_l2")) > 500.0 else "weak",
        "evidence": {
            "mean_margin": average(margins),
            "mean_spread": average(spreads),
            "selected_terms": sorted(vocab_candidates.keys()),
        },
    }


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "record_type": "stage56_wordclass_causal_probe_summary",
        "probe_count": len(rows),
        "probes": {str(row["word_class"]): row for row in rows},
        "strong_classes": [str(row["word_class"]) for row in rows if str(row.get("judgement", "")).endswith(("fiber", "transport", "bundle", "modifier"))],
    }


def write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Stage56 词类因果探针块",
        "",
        f"- probe_count: {int(summary['probe_count'])}",
        f"- strong_classes: {summary['strong_classes']}",
        "",
    ]
    for word_class, row in dict(summary["probes"]).items():
        lines.append(f"- {word_class}: mechanism={row['primary_mechanism']}, judgement={row['judgement']}, probe_strength={row['probe_strength']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build independent causal probes for adjectives, verbs, adverbs, and abstract nouns")
    ap.add_argument("--micro-json", default=str(ROOT / "tempdata" / "deepseek7b_micro_causal_apple_banana_20260301_210442" / "micro_causal_encoding_graph_results.json"))
    ap.add_argument("--apple-compare-json", default=str(ROOT / "tempdata" / "deepseek7b_apple_100_compare_20260301_204141" / "apple_100_concepts_compare_results.json"))
    ap.add_argument("--discovery-per-category-jsonl", default=str(ROOT / "tempdata" / "stage56_mass_term_large_seq_20260318_1540" / "discovery_per_category.jsonl"))
    ap.add_argument("--protocol-summary-json", default=str(ROOT / "tests" / "codex_temp" / "stage56_protocol_role_encoding_block_20260318_2218" / "summary.json"))
    ap.add_argument("--multidim-json", default=str(ROOT / "tempdata" / "deepseek7b_multidim_encoding_probe_v2_specific" / "multidim_encoding_probe.json"))
    ap.add_argument("--vocab-jsonl", default=str(ROOT / "tempdata" / "deepseek7b_tokenizer_vocab_expander_1500_20260317" / "all_candidates.jsonl"))
    ap.add_argument("--adverb-terms", default="quickly,slowly,carefully,formally,logically")
    ap.add_argument("--output-dir", default=str(ROOT / "tests" / "codex_temp" / "stage56_wordclass_causal_probe_20260318"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    micro_json = read_json(Path(args.micro_json))
    apple_compare = read_json(Path(args.apple_compare_json))
    category_rows = read_jsonl(Path(args.discovery_per_category_jsonl))
    protocol_summary = read_json(Path(args.protocol_summary_json))
    multidim_json = read_json(Path(args.multidim_json))
    adverb_terms = [item.strip() for item in str(args.adverb_terms).split(",") if item.strip()]
    vocab_candidates = load_selected_vocab_candidates(Path(args.vocab_jsonl), adverb_terms)

    probes = [
        build_adjective_probe(micro_json, apple_compare),
        build_verb_probe(category_rows, protocol_summary),
        build_abstract_probe(category_rows, protocol_summary),
        build_adverb_probe(vocab_candidates, multidim_json),
    ]
    summary = summarize(probes)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "probes.jsonl", probes)
    write_report(out_dir / "REPORT.md", summary)
    print(json.dumps({"output_dir": str(out_dir), "strong_classes": summary["strong_classes"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
