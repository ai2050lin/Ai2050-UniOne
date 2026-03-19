from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def category_key(row: Dict[str, object]) -> Tuple[str, str]:
    return (str(row.get("model_id", "")), str(row.get("category", "")))


def build_rows(design_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in design_rows:
        grouped.setdefault(category_key(row), []).append(dict(row))

    out: List[Dict[str, object]] = []
    for row in design_rows:
        peers = grouped.get(category_key(row), [])
        family_patch_raw = mean(
            [
                mean([safe_float(peer.get("atlas_static_proxy")) for peer in peers]),
                mean([safe_float(peer.get("frontier_dynamic_proxy")) for peer in peers]),
            ]
        )
        concept_offset_raw = mean(
            [
                abs(safe_float(row.get("atlas_static_proxy")) - mean([safe_float(peer.get("atlas_static_proxy")) for peer in peers])),
                abs(safe_float(row.get("offset_static_proxy")) - mean([safe_float(peer.get("offset_static_proxy")) for peer in peers])),
                abs(safe_float(row.get("frontier_dynamic_proxy")) - mean([safe_float(peer.get("frontier_dynamic_proxy")) for peer in peers])),
            ]
        )
        out.append(
            {
                **dict(row),
                "family_patch_raw": family_patch_raw,
                "concept_offset_raw": concept_offset_raw,
            }
        )
    return out


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "record_type": "stage56_family_patch_offset_raw_chain_summary",
        "row_count": len(rows),
        "mean_family_patch_raw": mean([safe_float(row.get("family_patch_raw")) for row in rows]),
        "mean_concept_offset_raw": mean([safe_float(row.get("concept_offset_raw")) for row in rows]),
        "main_judgment": (
            "当前第一版 family patch / concept offset 原始链已经从全局摘要推进到类别内局部结构对照，"
            "比此前的静态代理更接近静态本体层。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    return (
        "# Stage56 family patch / concept offset 原始链摘要\n\n"
        f"- row_count: {summary.get('row_count', 0)}\n"
        f"- mean_family_patch_raw: {safe_float(summary.get('mean_family_patch_raw')):+.6f}\n"
        f"- mean_concept_offset_raw: {safe_float(summary.get('mean_concept_offset_raw')):+.6f}\n"
        f"- main_judgment: {summary.get('main_judgment', '')}\n"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a rawer family patch / concept offset chain from sample design rows")
    ap.add_argument(
        "--design-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_fullsample_regression_runner_20260319" / "design_rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_family_patch_offset_raw_chain_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    design_rows = list(read_json(Path(args.design_rows_json)).get("rows", []))
    rows = build_rows(design_rows)
    summary = build_summary(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
