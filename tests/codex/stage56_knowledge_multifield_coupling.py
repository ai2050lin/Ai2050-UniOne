from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAXONOMY_PATH = (
    ROOT / "tests" / "codex_temp" / "stage56_multicategory_strong_weak_taxonomy_20260318" / "cases.jsonl"
)


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def safe_std(values: Sequence[float]) -> float:
    return float(np.std(values)) if values else 0.0


def compute_case_fields(row: Dict[str, object]) -> Dict[str, float]:
    proto = float(row["stage6_reference"]["proto_joint_adv"])
    inst = float(row["stage6_reference"]["instance_joint_adv"])
    union = float(row["stage6_reference"]["union_joint_adv"])
    synergy = float(row["stage6_reference"].get("union_synergy_joint", 0.0))
    best_strong = float(row["best_strong"]["metrics"]["joint_adv_mean"])
    best_weak = float(row["best_weak"]["metrics"]["joint_adv_mean"])
    best_mixed = float(row["best_mixed"]["metrics"]["joint_adv_mean"])

    bridge = max(0.0, synergy)
    conflict = max(0.0, -synergy)
    weak_support = max(0.0, best_weak)
    route_mismatch = max(0.0, max(proto, inst) - union)
    synergy_gap = float(best_mixed - union)
    core_field = max(0.0, best_strong)
    cooperation_surplus = max(0.0, union - max(proto, inst))
    balance = float(proto + inst + weak_support + bridge + cooperation_surplus - conflict - route_mismatch)
    coupling_energy = float(abs(proto) + abs(inst) + abs(union) + bridge + conflict + route_mismatch)
    bridge_ratio = float(bridge / (bridge + conflict + 1e-8))
    return {
        "prototype_field": float(proto),
        "instance_field": float(inst),
        "union_field": float(union),
        "synergy_field": float(synergy),
        "core_field": float(core_field),
        "weak_support_field": float(weak_support),
        "bridge_field": float(bridge),
        "conflict_field": float(conflict),
        "route_mismatch_field": float(route_mismatch),
        "cooperation_surplus_field": float(cooperation_surplus),
        "synergy_gap_field": float(synergy_gap),
        "multifield_balance": float(balance),
        "coupling_energy": float(coupling_energy),
        "bridge_ratio": float(bridge_ratio),
    }


def classify_regime(fields: Dict[str, float]) -> str:
    bridge = float(fields["bridge_field"])
    conflict = float(fields["conflict_field"])
    mismatch = float(fields["route_mismatch_field"])
    proto = float(fields["prototype_field"])
    inst = float(fields["instance_field"])
    if bridge > conflict and mismatch > 0.0:
        return "bridge_compensated_mismatch"
    if conflict > bridge and mismatch > 0.0:
        return "conflict_locked_mismatch"
    if bridge > 0.0 and proto >= 0.0 and inst >= 0.0:
        return "cooperative_multifield"
    if proto < 0.0 and inst < 0.0 and bridge <= 0.0:
        return "dual_route_collapse"
    if proto >= 0.0 and inst < 0.0 and bridge > 0.0:
        return "proto_bridge_rescue"
    return "mixed_transitional"


def dominant_field(fields: Dict[str, float]) -> str:
    candidates = {
        "prototype_field": abs(float(fields["prototype_field"])),
        "instance_field": abs(float(fields["instance_field"])),
        "bridge_field": abs(float(fields["bridge_field"])),
        "conflict_field": abs(float(fields["conflict_field"])),
        "route_mismatch_field": abs(float(fields["route_mismatch_field"])),
    }
    return max(candidates.items(), key=lambda kv: (kv[1], kv[0]))[0]


def aggregate_block(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    field_names = [
        "prototype_field",
        "instance_field",
        "union_field",
        "synergy_field",
        "core_field",
        "weak_support_field",
        "bridge_field",
        "conflict_field",
        "route_mismatch_field",
        "cooperation_surplus_field",
        "synergy_gap_field",
        "multifield_balance",
        "coupling_energy",
        "bridge_ratio",
    ]
    out: Dict[str, object] = {"case_count": len(rows)}
    for name in field_names:
        values = [float(row["fields"][name]) for row in rows]
        out[f"mean_{name}"] = safe_mean(values)
        out[f"std_{name}"] = safe_std(values)
    out["regime_counts"] = {
        regime: int(sum(1 for row in rows if row["regime"] == regime))
        for regime in sorted({row["regime"] for row in rows})
    }
    out["dominant_field_counts"] = {
        field: int(sum(1 for row in rows if row["dominant_field"] == field))
        for field in sorted({row["dominant_field"] for row in rows})
    }
    return out


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, summary: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# Stage56 Knowledge Multifield Coupling Report",
        "",
        f"- Case count: {summary['case_count']}",
        f"- Regimes: {summary['regime_counts']}",
        f"- Dominant fields: {summary['dominant_field_counts']}",
        f"- Mean prototype field: {summary['mean_prototype_field']:.6f}",
        f"- Mean instance field: {summary['mean_instance_field']:.6f}",
        f"- Mean bridge field: {summary['mean_bridge_field']:.6f}",
        f"- Mean conflict field: {summary['mean_conflict_field']:.6f}",
        f"- Mean route mismatch field: {summary['mean_route_mismatch_field']:.6f}",
        "",
        "## Top Balance Cases",
    ]
    top_rows = sorted(
        rows,
        key=lambda row: (
            float(row["fields"]["multifield_balance"]),
            float(row["fields"]["bridge_field"]),
        ),
        reverse=True,
    )[:20]
    for row in top_rows:
        lines.append(
            "- "
            f"{row['group_label']} / {row['category']} / proto={row['prototype_term']} / inst={row['instance_term']} "
            f"/ regime={row['regime']} / dominant={row['dominant_field']} "
            f"/ P={row['fields']['prototype_field']:.6f}"
            f" / I={row['fields']['instance_field']:.6f}"
            f" / B={row['fields']['bridge_field']:.6f}"
            f" / X={row['fields']['conflict_field']:.6f}"
            f" / M={row['fields']['route_mismatch_field']:.6f}"
            f" / balance={row['fields']['multifield_balance']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze multifield coupling from taxonomy cases")
    ap.add_argument("--taxonomy-cases", default=str(DEFAULT_TAXONOMY_PATH))
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_knowledge_multifield_coupling_20260318"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy_path = Path(args.taxonomy_cases)
    rows = read_jsonl(taxonomy_path)
    case_rows: List[Dict[str, object]] = []
    for row in rows:
        fields = compute_case_fields(row)
        case_rows.append(
            {
                "record_type": "stage56_knowledge_multifield_case",
                "group_label": str(row["group_label"]),
                "model_id": str(row["model_id"]),
                "category": str(row["category"]),
                "prototype_term": str(row["prototype_term"]),
                "instance_term": str(row["instance_term"]),
                "case_role": str(row["case_role"]),
                "dominant_structure": str(row["dominant_structure"]),
                "fields": fields,
                "regime": classify_regime(fields),
                "dominant_field": dominant_field(fields),
            }
        )

    summary = aggregate_block(case_rows)
    summary["record_type"] = "stage56_knowledge_multifield_summary"
    summary["created_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["proxy_definition"] = {
        "P": "stage6 proto_joint_adv",
        "I": "stage6 instance_joint_adv",
        "B": "max(stage6 union_synergy_joint, 0)",
        "X": "max(-stage6 union_synergy_joint, 0)",
        "M": "max(max(proto_joint_adv, instance_joint_adv) - union_joint_adv, 0)",
    }
    summary["regime_counts"] = dict(sorted(
        (regime, int(sum(1 for row in case_rows if row["regime"] == regime)))
        for regime in sorted({row["regime"] for row in case_rows})
    ))
    summary["dominant_field_counts"] = dict(sorted(
        (field, int(sum(1 for row in case_rows if row["dominant_field"] == field)))
        for field in sorted({row["dominant_field"] for row in case_rows})
    ))
    summary["per_model"] = {
        model_id: aggregate_block([row for row in case_rows if row["model_id"] == model_id])
        for model_id in sorted({row["model_id"] for row in case_rows})
    }
    summary["per_category"] = {
        category: aggregate_block([row for row in case_rows if row["category"] == category])
        for category in sorted({row["category"] for row in case_rows})
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "cases.jsonl", case_rows)
    write_report(out_dir / "REPORT.md", summary, case_rows)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "summary": str(out_dir / "summary.json"),
                "case_count": len(case_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
