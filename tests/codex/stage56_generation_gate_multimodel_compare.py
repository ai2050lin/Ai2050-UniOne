from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_generation_gate_coupling import AXES, FIELD_PROXY_NAMES, direction_label  # noqa: E402


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def parse_csv_arg(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def load_case_rows(input_paths: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for input_path in input_paths:
        path = Path(input_path)
        cases_path = path / "cases.jsonl" if path.is_dir() else path
        rows.extend(read_jsonl(cases_path))
    return rows


def filter_rows(
    rows: Sequence[Dict[str, object]],
    model_ids: Sequence[str],
    group_labels: Sequence[str],
    categories: Sequence[str],
) -> List[Dict[str, object]]:
    allowed_models = {value for value in model_ids if value}
    allowed_groups = {value for value in group_labels if value}
    allowed_categories = {value for value in categories if value}
    selected = []
    for row in rows:
        if allowed_models and str(row["model_id"]) not in allowed_models:
            continue
        if allowed_groups and str(row["group_label"]) not in allowed_groups:
            continue
        if allowed_categories and str(row["category"]) not in allowed_categories:
            continue
        selected.append(row)
    return selected


def aggregate_axis_block(rows: Sequence[Dict[str, object]], axis: str) -> Dict[str, object]:
    mean_deltas = {
        field_name: safe_mean([float(row["axis_gate_summary"]["axes"][axis]["deltas"][field_name]) for row in rows])
        for field_name in FIELD_PROXY_NAMES
    }
    scale = max((abs(value) for value in mean_deltas.values()), default=0.0)
    normalized_mean_deltas = {
        field_name: (float(mean_deltas[field_name] / scale) if scale > 0.0 else 0.0)
        for field_name in FIELD_PROXY_NAMES
    }
    return {
        "case_count": len(rows),
        "mean_deltas": mean_deltas,
        "scale_max_abs": float(scale),
        "normalized_mean_deltas": normalized_mean_deltas,
        "dominant_field": max(
            FIELD_PROXY_NAMES,
            key=lambda field_name: (abs(mean_deltas[field_name]), field_name),
        ),
        "direction_signature": {
            field_name: direction_label(mean_deltas[field_name])
            for field_name in FIELD_PROXY_NAMES
        },
    }


def aggregate_model_block(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    categories = sorted({str(row["category"]) for row in rows})
    return {
        "case_count": len(rows),
        "categories": categories,
        "per_axis": {
            axis: aggregate_axis_block(rows, axis)
            for axis in AXES
        },
        "per_category": {
            category: {
                axis: aggregate_axis_block([row for row in rows if str(row["category"]) == category], axis)
                for axis in AXES
            }
            for category in categories
        },
    }


def build_field_consensus(summary: Dict[str, object]) -> Dict[str, object]:
    per_model = summary["per_model"]
    out = {}
    for axis in AXES:
        axis_out = {}
        for field_name in FIELD_PROXY_NAMES:
            signatures = {
                model_id: str(model_block["per_axis"][axis]["direction_signature"][field_name])
                for model_id, model_block in per_model.items()
            }
            non_neutral = [value for value in signatures.values() if value != "neutral"]
            consensus = "mixed"
            if non_neutral and len(set(non_neutral)) == 1 and len(non_neutral) == len(signatures):
                consensus = non_neutral[0]
            elif not non_neutral:
                consensus = "neutral"
            axis_out[field_name] = {
                "consensus": consensus,
                "per_model": signatures,
            }
        out[axis] = axis_out
    return out


def write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Stage56 Generation Gate Multimodel Compare Report",
        "",
        f"- Case count: {summary['case_count']}",
        f"- Model count: {summary['model_count']}",
    ]
    for axis in AXES:
        lines.append(f"- {axis} consensus:")
        for field_name in FIELD_PROXY_NAMES:
            consensus = summary["field_consensus"][axis][field_name]["consensus"]
            lines.append(f"  - {field_name}: {consensus}")
    lines.extend(["", "## Per Model"])
    for model_id, model_block in summary["per_model"].items():
        lines.append(f"- {model_id}: cases={model_block['case_count']} / categories={','.join(model_block['categories'])}")
        for axis in AXES:
            axis_block = model_block["per_axis"][axis]
            deltas = axis_block["mean_deltas"]
            norm = axis_block["normalized_mean_deltas"]
            lines.append(
                "  - "
                f"{axis}: P={deltas['prototype_field_proxy']:.6f}, "
                f"I={deltas['instance_field_proxy']:.6f}, "
                f"B={deltas['bridge_field_proxy']:.6f}, "
                f"X={deltas['conflict_field_proxy']:.6f}, "
                f"M={deltas['mismatch_field_proxy']:.6f}"
            )
            lines.append(
                "  - "
                f"{axis} norm: P={norm['prototype_field_proxy']:.3f}, "
                f"I={norm['instance_field_proxy']:.3f}, "
                f"B={norm['bridge_field_proxy']:.3f}, "
                f"X={norm['conflict_field_proxy']:.3f}, "
                f"M={norm['mismatch_field_proxy']:.3f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare generation gate outputs across models")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--model-ids", default="")
    ap.add_argument("--group-labels", default="")
    ap.add_argument("--categories", default="")
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_generation_gate_multimodel_compare_20260318"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    rows = load_case_rows(args.inputs)
    rows = filter_rows(
        rows,
        model_ids=parse_csv_arg(args.model_ids),
        group_labels=parse_csv_arg(args.group_labels),
        categories=parse_csv_arg(args.categories),
    )
    per_model = {
        model_id: aggregate_model_block([row for row in rows if str(row["model_id"]) == model_id])
        for model_id in sorted({str(row["model_id"]) for row in rows})
    }
    summary = {
        "record_type": "stage56_generation_gate_multimodel_compare_summary",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": float(time.time() - t0),
        "case_count": len(rows),
        "model_count": len(per_model),
        "per_model": per_model,
    }
    summary["field_consensus"] = build_field_consensus(summary)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "REPORT.md", summary)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "summary": str(out_dir / "summary.json"),
                "case_count": len(rows),
                "model_count": len(per_model),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
