#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

import torch

from stage119_gpt2_embedding_full_vocab_scan import load_embedding_weight
from stage121_adverb_gate_bridge_probe import ensure_stage119_rows


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE119_OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage119_gpt2_embedding_full_vocab_scan_20260323"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage166_category_fiber_map_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE166_CATEGORY_FIBER_MAP_REPORT.md"

CATEGORY_SPECS = {
    "fruit": {"groups": {"meso_fruit"}},
    "animal": {"groups": {"meso_animal"}},
    "tool": {"groups": {"meso_object", "meso_tech"}},
    "vehicle": {"groups": {"meso_vehicle"}},
    "abstract": {"groups": {"macro_abstract", "macro_system"}},
}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def l2_unit(vec: torch.Tensor) -> torch.Tensor:
    return vec / torch.linalg.norm(vec).clamp_min(1e-8)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(l2_unit(a.float()), l2_unit(b.float())).item())


def build_summary(rows: List[Dict[str, object]], embed_weight: torch.Tensor) -> Dict[str, object]:
    category_rows: List[Dict[str, object]] = []
    centroids: Dict[str, torch.Tensor] = {}
    for category_name, spec in CATEGORY_SPECS.items():
        selected = [
            row for row in rows
            if str(row["lexical_type"]) == "noun" and str(row["group"]) in spec["groups"]
        ]
        selected.sort(
            key=lambda row: (
                float(row.get("effective_encoding_score", 0.0)),
                float(row.get("lexical_type_score", 0.0)),
            ),
            reverse=True,
        )
        limited = selected[:2048]
        centroid = l2_unit(torch.stack([embed_weight[int(row["token_id"])] for row in limited], dim=0).mean(dim=0))
        centroids[category_name] = centroid
        category_rows.append(
            {
                "category_name": category_name,
                "group_names": sorted(spec["groups"]),
                "member_count": len(selected),
                "sample_count": len(limited),
            }
        )

    edge_rows: List[Dict[str, object]] = []
    category_names = list(CATEGORY_SPECS.keys())
    for idx, name_a in enumerate(category_names):
        for name_b in category_names[idx + 1 :]:
            similarity = cosine(centroids[name_a], centroids[name_b])
            edge_rows.append(
                {
                    "category_a": name_a,
                    "category_b": name_b,
                    "centroid_similarity": similarity,
                }
            )
    edge_rows.sort(key=lambda row: float(row["centroid_similarity"]), reverse=True)

    strongest_edge = edge_rows[0]
    weakest_edge = edge_rows[-1]
    mean_edge_similarity = mean(float(row["centroid_similarity"]) for row in edge_rows)
    cross_category_separation = mean(1.0 - ((float(row["centroid_similarity"]) + 1.0) / 2.0) for row in edge_rows)
    category_fiber_score = clamp01(
        0.45 * (0.5 + 0.5 * mean_edge_similarity)
        + 0.55 * cross_category_separation
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage166_category_fiber_map",
        "title": "类别纤维图",
        "status_short": "category_fiber_map_ready",
        "category_count": len(category_rows),
        "edge_count": len(edge_rows),
        "mean_edge_similarity": mean_edge_similarity,
        "cross_category_separation": cross_category_separation,
        "category_fiber_score": category_fiber_score,
        "strongest_edge_name": f"{strongest_edge['category_a']}->{strongest_edge['category_b']}",
        "weakest_edge_name": f"{weakest_edge['category_a']}->{weakest_edge['category_b']}",
        "category_rows": category_rows,
        "edge_rows": edge_rows,
    }


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage166: 类别纤维图",
        "",
        "## 核心结果",
        f"- 类别数: {summary['category_count']}",
        f"- 边数: {summary['edge_count']}",
        f"- 平均边相似度: {summary['mean_edge_similarity']:.4f}",
        f"- 跨类分离度: {summary['cross_category_separation']:.4f}",
        f"- 类别纤维分数: {summary['category_fiber_score']:.4f}",
        f"- 最强边: {summary['strongest_edge_name']}",
        f"- 最弱边: {summary['weakest_edge_name']}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force and SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8-sig"))
    _, rows = ensure_stage119_rows(STAGE119_OUTPUT_DIR)
    embed_weight = load_embedding_weight()
    summary = build_summary(rows, embed_weight)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="类别纤维图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
