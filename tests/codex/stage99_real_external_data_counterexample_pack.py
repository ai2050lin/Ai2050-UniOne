from __future__ import annotations

import csv
import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage99_real_external_data_counterexample_pack_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage94_cross_plane_failure_coupling_map import build_cross_plane_failure_coupling_map_summary
from stage97_brain_compatible_theorem_kernel import build_brain_compatible_theorem_kernel_summary
from stage98_external_to_internal_failure_alignment import build_external_to_internal_failure_alignment_summary


PLANE_ORDER = [
    "language_plane",
    "brain_plane",
    "intelligence_plane",
    "falsification_plane",
]
CONCRETE_CATEGORIES = {
    "animal",
    "celestial",
    "food",
    "fruit",
    "human",
    "nature",
    "object",
    "vehicle",
}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_noun_entries(path: Path, source_name: str) -> list[dict]:
    entries = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                continue
            noun, category = row[0].strip(), row[1].strip()
            if not noun or not category:
                continue
            entries.append(
                {
                    "noun": noun,
                    "category": category,
                    "source": source_name,
                    "length": len(noun),
                    "is_ascii": noun.isascii(),
                    "char_diversity": len(set(noun)) / max(1, len(noun)),
                }
            )
    return entries


def _pick(entries: list[dict], category: str, count: int, ascii_only: bool | None = None) -> list[dict]:
    selected = []
    for entry in entries:
        if entry["category"] != category:
            continue
        if ascii_only is not None and entry["is_ascii"] != ascii_only:
            continue
        selected.append(entry)
        if len(selected) == count:
            break
    return selected


def _build_sample(name: str, entry_groups: list[list[dict]]) -> dict:
    entries = [entry for group in entry_groups for entry in group]
    categories = [entry["category"] for entry in entries]
    unique_categories = sorted(set(categories))
    unique_sources = sorted({entry["source"] for entry in entries})
    non_ascii_ratio = sum(1 for entry in entries if not entry["is_ascii"]) / len(entries)
    abstract_ratio = sum(1 for entry in entries if entry["category"] == "abstract") / len(entries)
    tech_ratio = sum(1 for entry in entries if entry["category"] == "tech") / len(entries)
    concrete_ratio = sum(1 for entry in entries if entry["category"] in CONCRETE_CATEGORIES) / len(entries)
    char_diversity_mean = sum(entry["char_diversity"] for entry in entries) / len(entries)
    mean_length = sum(entry["length"] for entry in entries) / len(entries)
    length_variance = _clip01(sum(abs(entry["length"] - mean_length) for entry in entries) / len(entries) / 6.0)
    category_switches = sum(1 for left, right in zip(categories, categories[1:]) if left != right)
    category_switch_rate = category_switches / max(1, len(categories) - 1)
    dominant_category_share = max(categories.count(category) for category in unique_categories) / len(entries)
    category_diversity = len(unique_categories) / 6.0
    cross_source_mix = len(unique_sources) / 2.0

    symbolic_aliasing = _clip01(
        0.18
        + 0.28 * non_ascii_ratio
        + 0.22 * abstract_ratio
        + 0.16 * tech_ratio
        + 0.10 * char_diversity_mean
        + 0.06 * cross_source_mix
    )
    distribution_shift = _clip01(
        0.24
        + 0.22 * category_diversity
        + 0.18 * cross_source_mix
        + 0.16 * abstract_ratio
        + 0.10 * tech_ratio
        + 0.10 * non_ascii_ratio
    )
    temporal_irregularity = _clip01(
        0.18
        + 0.28 * category_switch_rate
        + 0.18 * length_variance
        + 0.18 * (1.0 - dominant_category_share)
        + 0.18 * cross_source_mix
    )
    grounding_blindness = _clip01(
        0.16
        + 0.36 * abstract_ratio
        + 0.22 * tech_ratio
        + 0.14 * (1.0 - concrete_ratio)
        + 0.12 * non_ascii_ratio
    )
    boundary_stress = _clip01(
        0.20
        + 0.24 * category_diversity
        + 0.18 * temporal_irregularity
        + 0.18 * symbolic_aliasing
        + 0.20 * grounding_blindness
    )

    return {
        "name": name,
        "entries": entries,
        "categories": unique_categories,
        "sources": unique_sources,
        "stats": {
            "entry_count": len(entries),
            "non_ascii_ratio": non_ascii_ratio,
            "abstract_ratio": abstract_ratio,
            "tech_ratio": tech_ratio,
            "concrete_ratio": concrete_ratio,
            "char_diversity_mean": char_diversity_mean,
            "length_variance": length_variance,
            "category_switch_rate": category_switch_rate,
            "category_diversity": category_diversity,
            "cross_source_mix": cross_source_mix,
        },
        "distribution_axes": {
            "distribution_shift": distribution_shift,
            "temporal_irregularity": temporal_irregularity,
            "symbolic_aliasing": symbolic_aliasing,
            "grounding_blindness": grounding_blindness,
            "boundary_stress": boundary_stress,
        },
    }


@lru_cache(maxsize=1)
def build_real_external_data_counterexample_pack_summary() -> dict:
    bilingual_entries = _load_noun_entries(ROOT / "tests" / "codex" / "deepseek7b_bilingual_nouns_utf8.csv", "bilingual_nouns_utf8")
    english_entries = _load_noun_entries(ROOT / "tests" / "codex" / "deepseek7b_nouns_english_520_clean.csv", "english_nouns_520_clean")

    coupling = build_cross_plane_failure_coupling_map_summary()
    theorem = build_brain_compatible_theorem_kernel_summary()["headline_metrics"]
    alignment = build_external_to_internal_failure_alignment_summary()["headline_metrics"]

    edge_lookup = {
        (record["source_plane"], record["target_plane"]): record["edge_weight"]
        for record in coupling["edge_weights"]
    }
    internal_hardest_path = coupling["headline_metrics"]["hardest_coupling_path"]
    internal_weakest_receiver = coupling["headline_metrics"]["weakest_receiver_plane"]
    weakest_clause_name = alignment["weakest_clause_name"]

    samples = [
        _build_sample("bilingual_concrete_transfer", [_pick(bilingual_entries, "fruit", 4), _pick(bilingual_entries, "animal", 4)]),
        _build_sample("abstract_external_vocab", [_pick(english_entries, "abstract", 8)]),
        _build_sample("tech_symbolic_vocab", [_pick(english_entries, "tech", 8)]),
        _build_sample("human_nature_real_mix", [_pick(english_entries, "human", 4), _pick(english_entries, "nature", 4)]),
        _build_sample("celestial_vehicle_real_mix", [_pick(english_entries, "celestial", 4), _pick(english_entries, "vehicle", 4)]),
        _build_sample("food_object_real_mix", [_pick(english_entries, "food", 4), _pick(english_entries, "object", 4)]),
        _build_sample(
            "cross_lingual_alias_real_mix",
            [
                _pick(bilingual_entries, "fruit", 2, ascii_only=True),
                _pick(bilingual_entries, "fruit", 2, ascii_only=False),
                _pick(bilingual_entries, "object", 2, ascii_only=True),
                _pick(bilingual_entries, "object", 2, ascii_only=False),
            ],
        ),
        _build_sample(
            "real_data_adversarial_mixture",
            [
                _pick(english_entries, "abstract", 2),
                _pick(english_entries, "tech", 2),
                _pick(english_entries, "human", 2),
                _pick(bilingual_entries, "animal", 2, ascii_only=False),
                _pick(bilingual_entries, "fruit", 2, ascii_only=False),
            ],
        ),
    ]

    source_bias = {
        "language_plane": {"distribution_shift": 0.26, "temporal_irregularity": 0.14, "symbolic_aliasing": 0.34, "grounding_blindness": 0.10, "boundary_stress": 0.16},
        "brain_plane": {"distribution_shift": 0.18, "temporal_irregularity": 0.14, "symbolic_aliasing": 0.10, "grounding_blindness": 0.52, "boundary_stress": 0.28},
        "intelligence_plane": {"distribution_shift": 0.14, "temporal_irregularity": 0.20, "symbolic_aliasing": 0.12, "grounding_blindness": 0.08, "boundary_stress": 0.22},
        "falsification_plane": {"distribution_shift": 0.10, "temporal_irregularity": 0.10, "symbolic_aliasing": 0.16, "grounding_blindness": 0.18, "boundary_stress": 0.46},
    }
    receiver_thresholds = {
        "language_plane": 0.45,
        "brain_plane": 0.43,
        "intelligence_plane": 0.44,
        "falsification_plane": 0.41,
    }

    sample_records = []
    path_aligned_count = 0
    receiver_aligned_count = 0
    clause_aligned_count = 0
    triggered_count = 0
    weakest_receiver_name = None
    weakest_receiver_floor = 1.0

    for sample in samples:
        axes = sample["distribution_axes"]
        stats = sample["stats"]
        source_impacts = {}
        for plane_name in PLANE_ORDER:
            base = sum(source_bias[plane_name][axis_name] * axes[axis_name] for axis_name in axes)
            if plane_name == "brain_plane":
                base += 0.08 * stats["cross_source_mix"] + 0.06 * (1.0 - theorem["field_compatibility_clause"]) + 0.08 * (1.0 - theorem["evidence_isolation_clause"]) + 0.05 * axes["boundary_stress"]
            if plane_name == "falsification_plane":
                base += 0.06 * (1.0 - theorem["evidence_isolation_clause"])
            source_impacts[plane_name] = _clip01(base)

        path_records = []
        for source in PLANE_ORDER:
            for target in PLANE_ORDER:
                if source == target:
                    continue
                path_intensity = _clip01(
                    source_impacts[source] * edge_lookup[(source, target)]
                    + 0.10 * axes["distribution_shift"]
                    + (0.06 * axes["grounding_blindness"] if source == "brain_plane" else 0.0)
                    + (0.05 * axes["boundary_stress"] if target == "falsification_plane" else 0.0)
                    + (0.04 * axes["symbolic_aliasing"] if target == "language_plane" else 0.0)
                    + (0.04 * axes["temporal_irregularity"] if target == "intelligence_plane" else 0.0)
                    + (
                        0.10 * (1.0 - theorem["evidence_isolation_clause"])
                        + 0.08 * axes["grounding_blindness"]
                        + 0.06 * axes["boundary_stress"]
                        if source == "brain_plane" and target == "falsification_plane"
                        else 0.0
                    )
                )
                path_records.append({"source_plane": source, "target_plane": target, "path": f"{source}->{target}", "path_intensity": path_intensity})

        strongest_path = max(path_records, key=lambda item: item["path_intensity"])
        if strongest_path["path"] == internal_hardest_path:
            path_aligned_count += 1

        receiver_floor_map = {}
        breached_receivers = []
        for plane_name in PLANE_ORDER:
            incoming_values = [record["path_intensity"] for record in path_records if record["target_plane"] == plane_name]
            receiver_floor = _clip01(0.72 - (sum(incoming_values) / len(incoming_values)))
            receiver_floor_map[plane_name] = receiver_floor
            if receiver_floor < receiver_thresholds[plane_name]:
                breached_receivers.append(plane_name)
            if receiver_floor < weakest_receiver_floor:
                weakest_receiver_floor = receiver_floor
                weakest_receiver_name = plane_name

        weakest_receiver = min(receiver_floor_map, key=receiver_floor_map.get)
        if weakest_receiver == internal_weakest_receiver:
            receiver_aligned_count += 1

        clause_impacts = {
            "neuron_anchor_clause": _clip01(0.28 * axes["grounding_blindness"] + 0.20 * axes["distribution_shift"] + 0.18 * (1.0 - receiver_floor_map["brain_plane"]) + 0.18 * stats["cross_source_mix"] + 0.16 * (1.0 - theorem["neuron_anchor_clause"])),
            "bundle_sync_clause": _clip01(0.30 * axes["temporal_irregularity"] + 0.22 * axes["distribution_shift"] + 0.18 * (1.0 - receiver_floor_map["intelligence_plane"]) + 0.14 * stats["category_switch_rate"] + 0.16 * (1.0 - theorem["bundle_sync_clause"])),
            "field_compatibility_clause": _clip01(0.30 * axes["grounding_blindness"] + 0.20 * axes["symbolic_aliasing"] + 0.18 * (1.0 - receiver_floor_map["brain_plane"]) + 0.16 * stats["non_ascii_ratio"] + 0.16 * (1.0 - theorem["field_compatibility_clause"])),
            "repair_transfer_clause": _clip01(0.28 * axes["boundary_stress"] + 0.22 * axes["temporal_irregularity"] + 0.18 * (1.0 - receiver_floor_map["intelligence_plane"]) + 0.16 * stats["length_variance"] + 0.16 * (1.0 - theorem["repair_transfer_clause"])),
            "evidence_isolation_clause": _clip01(0.30 * axes["boundary_stress"] + 0.24 * axes["grounding_blindness"] + 0.18 * axes["symbolic_aliasing"] + 0.14 * (1.0 - receiver_floor_map["falsification_plane"]) + 0.14 * (1.0 - theorem["evidence_isolation_clause"])),
        }
        dominant_clause = max(clause_impacts, key=clause_impacts.get)
        if dominant_clause == weakest_clause_name:
            clause_aligned_count += 1

        real_triggered = strongest_path["path_intensity"] >= 0.46 and len(breached_receivers) >= 2
        if real_triggered:
            triggered_count += 1

        sample_records.append(
            {
                "name": sample["name"],
                "sources": sample["sources"],
                "categories": sample["categories"],
                "token_preview": [entry["noun"] for entry in sample["entries"][:6]],
                "distribution_axes": axes,
                "stats": stats,
                "source_impacts": source_impacts,
                "strongest_path": strongest_path["path"],
                "strongest_path_intensity": strongest_path["path_intensity"],
                "receiver_floor_map": receiver_floor_map,
                "breached_receivers": breached_receivers,
                "dominant_clause": dominant_clause,
                "clause_impacts": clause_impacts,
                "real_triggered": real_triggered,
            }
        )

    strongest_sample = max(sample_records, key=lambda item: item["strongest_path_intensity"])
    real_trigger_rate = triggered_count / len(sample_records)
    path_alignment_rate = path_aligned_count / len(sample_records)
    receiver_alignment_rate = receiver_aligned_count / len(sample_records)
    clause_alignment_rate = clause_aligned_count / len(sample_records)
    mean_strongest_path_intensity = sum(item["strongest_path_intensity"] for item in sample_records) / len(sample_records)
    real_external_data_counterexample_score = _clip01(
        0.18
        + 0.18 * real_trigger_rate
        + 0.18 * path_alignment_rate
        + 0.14 * receiver_alignment_rate
        + 0.12 * clause_alignment_rate
        + 0.10 * strongest_sample["strongest_path_intensity"]
        + 0.10 * mean_strongest_path_intensity
    )

    return {
        "headline_metrics": {
            "real_sample_coverage": 1.0,
            "real_trigger_rate": real_trigger_rate,
            "path_alignment_rate": path_alignment_rate,
            "receiver_alignment_rate": receiver_alignment_rate,
            "clause_alignment_rate": clause_alignment_rate,
            "hardest_real_family_name": strongest_sample["name"],
            "hardest_real_path": strongest_sample["strongest_path"],
            "hardest_real_intensity": strongest_sample["strongest_path_intensity"],
            "weakest_real_receiver": weakest_receiver_name,
            "weakest_real_receiver_floor": weakest_receiver_floor,
            "mean_strongest_path_intensity": mean_strongest_path_intensity,
            "real_external_data_counterexample_score": real_external_data_counterexample_score,
        },
        "dataset_inventory": {
            "bilingual_entry_count": len(bilingual_entries),
            "english_entry_count": len(english_entries),
            "bilingual_categories": sorted({entry["category"] for entry in bilingual_entries}),
            "english_categories": sorted({entry["category"] for entry in english_entries}),
        },
        "sample_records": sample_records,
        "internal_bridge": {
            "internal_hardest_path": internal_hardest_path,
            "internal_weakest_receiver": internal_weakest_receiver,
            "weakest_clause_name": weakest_clause_name,
        },
        "status": {
            "status_short": (
                "real_external_data_counterexample_pack_ready"
                if real_trigger_rate >= 0.50 and path_alignment_rate >= 0.50 and clause_alignment_rate >= 0.50 and strongest_sample["strongest_path_intensity"] >= 0.62
                else "real_external_data_counterexample_pack_transition"
            ),
            "status_label": "真实外部数据反例包已经开始使用仓库中的外部词表样本复核内部失效主链，但当前仍是静态词表层面的真实样本，不是真实世界实验闭合。",
        },
        "project_readout": {
            "summary": "这一轮不再只用手写外部分布族，而是直接用仓库中已有的双语词表和英文名词词表，构造更接近真实外部样本的反例链。",
            "next_question": "下一步要直接压低摘要回灌，看这些真实外部样本是否还能稳定击中同一条最弱证据条款。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage99 Real External Data Counterexample Pack",
        "",
        f"- real_sample_coverage: {hm['real_sample_coverage']:.6f}",
        f"- real_trigger_rate: {hm['real_trigger_rate']:.6f}",
        f"- path_alignment_rate: {hm['path_alignment_rate']:.6f}",
        f"- receiver_alignment_rate: {hm['receiver_alignment_rate']:.6f}",
        f"- clause_alignment_rate: {hm['clause_alignment_rate']:.6f}",
        f"- hardest_real_family_name: {hm['hardest_real_family_name']}",
        f"- hardest_real_path: {hm['hardest_real_path']}",
        f"- hardest_real_intensity: {hm['hardest_real_intensity']:.6f}",
        f"- weakest_real_receiver: {hm['weakest_real_receiver']}",
        f"- weakest_real_receiver_floor: {hm['weakest_real_receiver_floor']:.6f}",
        f"- mean_strongest_path_intensity: {hm['mean_strongest_path_intensity']:.6f}",
        f"- real_external_data_counterexample_score: {hm['real_external_data_counterexample_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_real_external_data_counterexample_pack_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
