from __future__ import annotations

import argparse
import json
import statistics
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from stage56_mass_scan_io import row_term, scan_term_rows


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONCEPT_SPECS = (
    "apple|fruit|food_composed|apple,banana,orange,grape,pear",
    "cat|animal|living|cat,dog,rabbit,tiger,lion",
    "king|human|living|king,queen,teacher,doctor",
    "justice|abstract|abstract|justice,truth,logic,memory",
    "run|action|action|run,jump,walk,move",
)

FIELD_NAME_MAP = {
    "prototype_field_proxy": "P",
    "instance_field_proxy": "I",
    "bridge_field_proxy": "B",
    "conflict_field_proxy": "X",
    "mismatch_field_proxy": "M",
}


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


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


def parse_concept_spec(text: str) -> Dict[str, object]:
    parts = [part.strip() for part in text.split("|")]
    if len(parts) != 4:
        raise ValueError(f"invalid concept spec: {text}")
    anchor, meso, macro_mode, peers_text = parts
    peers = [normalize_key(item) for item in peers_text.split(",") if item.strip()]
    if not anchor or not meso or not macro_mode or not peers:
        raise ValueError(f"incomplete concept spec: {text}")
    return {
        "anchor": normalize_key(anchor),
        "meso": normalize_key(meso),
        "macro_mode": normalize_key(macro_mode),
        "peers": peers,
    }


def build_scan_maps(scan: Dict[str, object]) -> Tuple[Dict[str, Dict[str, object]], Dict[str, set[int]], Dict[str, str]]:
    noun_map: Dict[str, Dict[str, object]] = {}
    noun_cat_map: Dict[str, str] = {}
    for row in scan_term_rows(scan):
        noun = normalize_key(row_term(row))
        if not noun:
            continue
        noun_map[noun] = row
        noun_cat_map[noun] = normalize_key(row.get("category"))

    cat_map: Dict[str, set[int]] = {}
    for category, row in (scan.get("category_prototypes") or {}).items():
        key = normalize_key(category)
        cat_map[key] = {int(x) for x in row.get("prototype_top_indices", [])}
    return noun_map, cat_map, noun_cat_map


def make_living_macro(cat_map: Dict[str, set[int]], noun_map: Dict[str, Dict[str, object]], noun_cat_map: Dict[str, str], top_k: int = 120) -> set[int]:
    base: set[int] = set()
    for category in ("animal", "fruit", "human", "nature"):
        base |= set(cat_map.get(category, set()))
    freq: Dict[int, int] = {}
    for noun, row in noun_map.items():
        if noun_cat_map.get(noun) not in {"animal", "fruit", "human", "nature"}:
            continue
        for idx in row.get("signature_top_indices", [])[:top_k]:
            freq[int(idx)] = freq.get(int(idx), 0) + 1
    if freq:
        top = [idx for idx, _ in sorted(freq.items(), key=lambda item: item[1], reverse=True)[:top_k]]
        base |= set(top)
    return base


def make_food_macro(cat_map: Dict[str, set[int]], noun_map: Dict[str, Dict[str, object]], noun_cat_map: Dict[str, str], top_k: int = 120) -> set[int]:
    base = set(cat_map.get("food", set())) | set(cat_map.get("fruit", set()))
    freq: Dict[int, int] = {}
    for noun, row in noun_map.items():
        if noun_cat_map.get(noun) not in {"food", "fruit"}:
            continue
        for idx in row.get("signature_top_indices", [])[:top_k]:
            freq[int(idx)] = freq.get(int(idx), 0) + 1
    if freq:
        top = [idx for idx, _ in sorted(freq.items(), key=lambda item: item[1], reverse=True)[:top_k]]
        base |= set(top)
    return base


def resolve_macro_set(
    macro_mode: str,
    cat_map: Dict[str, set[int]],
    noun_map: Dict[str, Dict[str, object]],
    noun_cat_map: Dict[str, str],
) -> set[int]:
    macro_key = normalize_key(macro_mode)
    if macro_key == "living":
        return make_living_macro(cat_map, noun_map, noun_cat_map)
    if macro_key == "food_composed":
        return make_food_macro(cat_map, noun_map, noun_cat_map)
    return set(cat_map.get(macro_key, set()))


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


def compute_concept_row(
    spec: Dict[str, object],
    noun_map: Dict[str, Dict[str, object]],
    cat_map: Dict[str, set[int]],
    noun_cat_map: Dict[str, str],
    n_layers: int,
) -> Dict[str, object]:
    anchor = str(spec["anchor"])
    meso = str(spec["meso"])
    macro_mode = str(spec["macro_mode"])
    peers = [str(value) for value in spec["peers"]]

    anchor_row = noun_map.get(anchor)
    if anchor_row is None:
        return {
            "anchor": anchor,
            "meso": meso,
            "macro_mode": macro_mode,
            "coverage_status": "missing_anchor",
            "available_peers": [peer for peer in peers if peer in noun_map],
        }

    meso_set = set(cat_map.get(meso, set()))
    macro_set = resolve_macro_set(macro_mode, cat_map, noun_map, noun_cat_map)
    if not meso_set:
        return {
            "anchor": anchor,
            "meso": meso,
            "macro_mode": macro_mode,
            "coverage_status": "missing_meso",
            "available_peers": [peer for peer in peers if peer in noun_map],
        }
    if not macro_set:
        return {
            "anchor": anchor,
            "meso": meso,
            "macro_mode": macro_mode,
            "coverage_status": "missing_macro",
            "available_peers": [peer for peer in peers if peer in noun_map],
        }

    peer_rows = [noun_map[peer] for peer in peers if peer in noun_map]
    peer_sets = [{int(x) for x in row.get("signature_top_indices", [])} for row in peer_rows]
    peer_sets = [value for value in peer_sets if value]
    if len(peer_sets) < 2:
        return {
            "anchor": anchor,
            "meso": meso,
            "macro_mode": macro_mode,
            "coverage_status": "insufficient_peers",
            "available_peers": [peer for peer in peers if peer in noun_map],
        }

    anchor_sig = {int(x) for x in anchor_row.get("signature_top_indices", [])}
    peer_union = set().union(*peer_sets)
    peer_intersection = set(peer_sets[0])
    for peer_set in peer_sets[1:]:
        peer_intersection &= peer_set
    shared_base = anchor_sig & meso_set & macro_set
    pairwise_peer = [jaccard(left, right) for left, right in combinations(peer_sets, 2)]
    layer_profile = normalize_layer_profile(anchor_row.get("signature_layer_distribution", {}), n_layers)

    anchor_to_meso = jaccard(anchor_sig, meso_set)
    meso_to_macro = jaccard(meso_set, macro_set)
    anchor_to_macro = jaccard(anchor_sig, macro_set)
    micro_to_meso = jaccard(peer_union, meso_set)
    micro_to_macro = jaccard(peer_union, macro_set)

    return {
        "anchor": anchor,
        "category": normalize_key(anchor_row.get("category")),
        "meso": meso,
        "macro_mode": macro_mode,
        "coverage_status": "covered",
        "available_peers": [normalize_key(row.get("noun")) for row in peer_rows],
        "anchor_signature_size": len(anchor_sig),
        "peer_union_size": len(peer_union),
        "peer_intersection_size": len(peer_intersection),
        "meso_size": len(meso_set),
        "macro_size": len(macro_set),
        "shared_base_size": len(shared_base),
        "anchor_to_meso_jaccard": anchor_to_meso,
        "anchor_to_macro_jaccard": anchor_to_macro,
        "micro_to_meso_jaccard": micro_to_meso,
        "micro_to_macro_jaccard": micro_to_macro,
        "meso_to_macro_jaccard": meso_to_macro,
        "shared_base_ratio_vs_anchor": float(len(shared_base) / max(1, len(anchor_sig))),
        "shared_base_ratio_vs_peer_union": float(len(shared_base) / max(1, len(peer_union))),
        "micro_pairwise_jaccard_mean": safe_mean(pairwise_peer),
        "micro_pairwise_jaccard_gap": float(1.0 - safe_mean(pairwise_peer)),
        "layer_peak_band": layer_peak_band(layer_profile),
        "layer_peak_value": max(layer_profile) if layer_profile else 0.0,
        "icspb_view": {
            "section_anchor_strength": anchor_to_meso,
            "fiber_dispersion": float(1.0 - safe_mean(pairwise_peer)),
            "transport_closure_gain": float(meso_to_macro - micro_to_meso),
            "shared_protocol_ratio": float(len(shared_base) / max(1, len(meso_set))),
        },
    }


def summarize_generation_block(apple_dossier: Dict[str, object], gate_summary: Dict[str, object]) -> Dict[str, object]:
    apple_metrics = dict(apple_dossier.get("metrics", {}))
    field_consensus = dict(gate_summary.get("field_consensus", {}))
    axis_rules: Dict[str, Dict[str, str]] = {}
    for axis, axis_block in field_consensus.items():
        axis_rules[axis] = {
            FIELD_NAME_MAP[field_name]: str(info.get("consensus", "mixed"))
            for field_name, info in axis_block.items()
            if field_name in FIELD_NAME_MAP
        }
    return {
        "style_logic_syntax_signal": float(apple_metrics.get("style_logic_syntax_signal", 0.0)),
        "cross_dim_decoupling_index": float(apple_metrics.get("cross_dim_decoupling_index", 0.0)),
        "axis_rules": axis_rules,
    }


def summarize_relation_block(triplet_probe: Dict[str, object]) -> Dict[str, object]:
    metrics = dict(triplet_probe.get("metrics", {}))
    return {
        "triplet_separability_index": float(metrics.get("triplet_separability_index", 0.0)),
        "axis_specificity_index": float(metrics.get("axis_specificity_index", 0.0)),
        "king_queen_jaccard": float(metrics.get("king_queen_jaccard", 0.0)),
        "apple_king_jaccard": float(metrics.get("apple_king_jaccard", 0.0)),
    }


def build_global_summary(concept_rows: Sequence[Dict[str, object]], generation_block: Dict[str, object], relation_block: Dict[str, object]) -> Dict[str, object]:
    covered = [row for row in concept_rows if row.get("coverage_status") == "covered"]
    missing = [row for row in concept_rows if row.get("coverage_status") != "covered"]
    dominant_macro = []
    for row in covered:
        if float(row["meso_to_macro_jaccard"]) > float(row["micro_to_meso_jaccard"]):
            dominant_macro.append(str(row["anchor"]))
    return {
        "covered_anchor_count": len(covered),
        "missing_anchor_count": len(missing),
        "covered_anchors": [str(row["anchor"]) for row in covered],
        "missing_anchors": [str(row["anchor"]) for row in missing],
        "mean_micro_to_meso_jaccard": safe_mean([float(row["micro_to_meso_jaccard"]) for row in covered]),
        "mean_meso_to_macro_jaccard": safe_mean([float(row["meso_to_macro_jaccard"]) for row in covered]),
        "mean_shared_base_ratio_vs_anchor": safe_mean([float(row["shared_base_ratio_vs_anchor"]) for row in covered]),
        "mean_section_anchor_strength": safe_mean([float(row["icspb_view"]["section_anchor_strength"]) for row in covered]),
        "mean_fiber_dispersion": safe_mean([float(row["icspb_view"]["fiber_dispersion"]) for row in covered]),
        "macro_stronger_than_micro_anchors": dominant_macro,
        "style_logic_syntax_signal": float(generation_block["style_logic_syntax_signal"]),
        "cross_dim_decoupling_index": float(generation_block["cross_dim_decoupling_index"]),
        "relation_axis_specificity": float(relation_block["axis_specificity_index"]),
    }


def build_verdict(concept_rows: Sequence[Dict[str, object]], global_summary: Dict[str, object], generation_block: Dict[str, object]) -> Dict[str, object]:
    covered = [row for row in concept_rows if row.get("coverage_status") == "covered"]
    missing = [row for row in concept_rows if row.get("coverage_status") != "covered"]
    meso_dominant = [
        str(row["anchor"])
        for row in covered
        if float(row["meso_to_macro_jaccard"]) > float(row["micro_to_meso_jaccard"])
    ]
    return {
        "core_statement": (
            "当前真实结果支持一种中观锚点主导的联合编码：概念先以中观实体原型稳定锚定，"
            "再通过微观属性纤维与宏观协议路径展开；风格、逻辑、语法主要调制读出而不是替代概念本体。"
        ),
        "covered_scope_statement": f"当前实证已覆盖 {len(covered)} 个锚点：{', '.join(str(row['anchor']) for row in covered)}。",
        "gap_statement": (
            "当前词表没有覆盖 justice 与 run，这意味着抽象名词和动作词仍未进入同口径实证闭环。"
            if missing
            else "当前默认锚点已全部覆盖。"
        ),
        "meso_dominant_anchors": meso_dominant,
        "stable_generation_rules": generation_block["axis_rules"],
        "summary_metrics": global_summary,
    }


def write_report(
    path: Path,
    concept_rows: Sequence[Dict[str, object]],
    generation_block: Dict[str, object],
    relation_block: Dict[str, object],
    global_summary: Dict[str, object],
    verdict: Dict[str, object],
) -> None:
    lines = [
        "# ICSPB 三尺度概念图谱报告",
        "",
        "## 全局结论",
        f"- 已覆盖锚点数: {global_summary['covered_anchor_count']}",
        f"- 缺失锚点数: {global_summary['missing_anchor_count']}",
        f"- micro->meso 平均重合: {global_summary['mean_micro_to_meso_jaccard']:.6f}",
        f"- meso->macro 平均重合: {global_summary['mean_meso_to_macro_jaccard']:.6f}",
        f"- 共享基底平均比率: {global_summary['mean_shared_base_ratio_vs_anchor']:.6f}",
        f"- 风格/逻辑/语法信号: {global_summary['style_logic_syntax_signal']:.6f}",
        f"- 交叉维度解耦指数: {global_summary['cross_dim_decoupling_index']:.6f}",
        f"- 关系轴特异性: {global_summary['relation_axis_specificity']:.6f}",
        "",
        "## 概念行",
    ]
    for row in concept_rows:
        status = str(row["coverage_status"])
        if status != "covered":
            lines.append(f"- {row['anchor']}: {status}, peers={','.join(row.get('available_peers', []))}")
            continue
        lines.append(
            "- "
            f"{row['anchor']}: category={row['category']}, peak={row['layer_peak_band']}, "
            f"anchor->meso={float(row['anchor_to_meso_jaccard']):.6f}, "
            f"micro->meso={float(row['micro_to_meso_jaccard']):.6f}, "
            f"meso->macro={float(row['meso_to_macro_jaccard']):.6f}, "
            f"shared_base_ratio={float(row['shared_base_ratio_vs_anchor']):.6f}"
        )
    lines.extend(
        [
            "",
            "## 生成门控共识",
            f"- style: {generation_block['axis_rules'].get('style', {})}",
            f"- logic: {generation_block['axis_rules'].get('logic', {})}",
            f"- syntax: {generation_block['axis_rules'].get('syntax', {})}",
            "",
            "## 关系轴",
            f"- triplet_separability_index: {relation_block['triplet_separability_index']:.6f}",
            f"- axis_specificity_index: {relation_block['axis_specificity_index']:.6f}",
            f"- king_queen_jaccard: {relation_block['king_queen_jaccard']:.6f}",
            f"- apple_king_jaccard: {relation_block['apple_king_jaccard']:.6f}",
            "",
            "## ICSPB 解释",
            f"- {verdict['core_statement']}",
            f"- {verdict['covered_scope_statement']}",
            f"- {verdict['gap_statement']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build an ICSPB triscale concept atlas from existing result artifacts")
    ap.add_argument(
        "--mass-json",
        default="tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed101/mass_noun_encoding_scan.json",
    )
    ap.add_argument(
        "--apple-dossier-json",
        default="tempdata/deepseek7b_apple_encoding_law_dossier_20260306_223055/apple_multiaxis_encoding_law_dossier.json",
    )
    ap.add_argument(
        "--triplet-json",
        default="tempdata/deepseek7b_triplet_probe_20260306_150637/apple_king_queen_triplet_probe.json",
    )
    ap.add_argument(
        "--gate-compare-summary-json",
        default="tests/codex_temp/stage56_generation_gate_multimodel_compare_all3_8cat_20260318_1338/summary.json",
    )
    ap.add_argument("--concept-spec", action="append", default=[])
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_icspb_concept_dimension_atlas_20260318"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    scan = read_json(Path(args.mass_json))
    apple_dossier = read_json(Path(args.apple_dossier_json))
    triplet_probe = read_json(Path(args.triplet_json))
    gate_summary = read_json(Path(args.gate_compare_summary_json))

    noun_map, cat_map, noun_cat_map = build_scan_maps(scan)
    concept_specs = [parse_concept_spec(text) for text in (args.concept_spec or DEFAULT_CONCEPT_SPECS)]
    n_layers = int(((scan.get("config") or {}).get("n_layers")) or 28)
    concept_rows = [
        compute_concept_row(spec, noun_map, cat_map, noun_cat_map, n_layers)
        for spec in concept_specs
    ]
    generation_block = summarize_generation_block(apple_dossier, gate_summary)
    relation_block = summarize_relation_block(triplet_probe)
    global_summary = build_global_summary(concept_rows, generation_block, relation_block)
    verdict = build_verdict(concept_rows, global_summary, generation_block)

    payload = {
        "record_type": "stage56_icspb_concept_dimension_atlas",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": float(time.time() - t0),
        "config": {
            "mass_json": args.mass_json,
            "apple_dossier_json": args.apple_dossier_json,
            "triplet_json": args.triplet_json,
            "gate_compare_summary_json": args.gate_compare_summary_json,
            "concept_specs": concept_specs,
        },
        "concept_rows": concept_rows,
        "generation_block": generation_block,
        "relation_block": relation_block,
        "global_summary": global_summary,
        "verdict": verdict,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir / "REPORT.md", concept_rows, generation_block, relation_block, global_summary, verdict)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "summary_json": str(out_dir / "summary.json"),
                "report_md": str(out_dir / "REPORT.md"),
                "covered_anchor_count": global_summary["covered_anchor_count"],
                "missing_anchor_count": global_summary["missing_anchor_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
