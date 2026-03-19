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


def build_static_raw_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(category_key(row), []).append(dict(row))

    out: List[Dict[str, object]] = []
    for row in rows:
        key = category_key(row)
        peers = grouped.get(key, [])
        category_atlas_mean = mean([safe_float(peer.get("atlas_static_proxy")) for peer in peers])
        category_offset_mean = mean([safe_float(peer.get("offset_static_proxy")) for peer in peers])
        atlas_raw_proxy = mean(
            [
                safe_float(row.get("atlas_static_proxy")),
                category_atlas_mean,
            ]
        )
        offset_raw_proxy = abs(safe_float(row.get("offset_static_proxy")) - category_offset_mean)
        out.append(
            {
                **dict(row),
                "atlas_raw_proxy": atlas_raw_proxy,
                "offset_raw_proxy": offset_raw_proxy,
                "category_atlas_mean": category_atlas_mean,
                "category_offset_mean": category_offset_mean,
            }
        )
    return out


def build_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return build_static_raw_rows(rows)


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "record_type": "stage56_static_raw_chain_summary",
        "row_count": len(rows),
        "mean_atlas_raw_proxy": mean([safe_float(row.get("atlas_raw_proxy")) for row in rows]),
        "mean_offset_raw_proxy": mean([safe_float(row.get("offset_raw_proxy")) for row in rows]),
        "main_judgment": (
            "当前第一版原始静态链把静态项从全局摘要推进到类别内对照，"
            "让 Atlas_static（静态图册项）和 Offset_static（静态偏移项）更接近原始局部结构。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    return (
        "# Stage56 静态项原始估计链摘要\n\n"
        f"- row_count: {summary.get('row_count', 0)}\n"
        f"- mean_atlas_raw_proxy: {safe_float(summary.get('mean_atlas_raw_proxy')):+.6f}\n"
        f"- mean_offset_raw_proxy: {safe_float(summary.get('mean_offset_raw_proxy')):+.6f}\n"
        f"- main_judgment: {summary.get('main_judgment', '')}\n"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a rawer static-estimate chain from sample-level design rows")
    ap.add_argument(
        "--design-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_fullsample_regression_runner_20260319" / "design_rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_static_raw_chain_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    design_rows = list(read_json(Path(args.design_rows_json)).get("rows", []))
    raw_rows = build_static_raw_rows(design_rows)
    summary = build_summary(raw_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(json.dumps({"rows": raw_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(raw_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
