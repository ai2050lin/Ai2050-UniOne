#!/usr/bin/env python
"""
Apple / King / Queen triplet structure probe (offline).

Goal:
- Use existing mass_noun_encoding_scan.json to test whether concept encoding
  shows a structured split between:
  1) entity identity (apple vs king/queen),
  2) category anchors (fruit vs human),
  3) relation-like axis (king <-> queen).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from agi_research_result_schema import build_result_payload, write_result_bundle


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb) / max(1, len(sa | sb)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a) + 1e-12)
    nb = float(np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / (na * nb))


def layer_vec(dist: Dict, n_layers: int = 28) -> np.ndarray:
    v = np.zeros(n_layers, dtype=np.float32)
    if isinstance(dist, dict):
        for k, val in dist.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if 0 <= idx < n_layers:
                v[idx] = float(val)
    s = float(v.sum())
    if s > 0:
        v /= s
    return v


def signature_hash_vec(sig: Sequence[int], dim: int = 4096) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for x in sig:
        h = (int(x) * 2654435761) % dim
        v[h] += 1.0
    v /= float(np.linalg.norm(v) + 1e-12)
    return v


def normalize_key(s: str) -> str:
    return str(s or "").strip().lower()


def pick_first(data: Dict[str, Dict], aliases: Sequence[str]) -> Optional[Dict]:
    for name in aliases:
        if normalize_key(name) in data:
            return data[normalize_key(name)]
    return None


def pick_proto(category_prototypes: Dict, aliases: Sequence[str]) -> Optional[Dict]:
    for c in aliases:
        if normalize_key(c) in category_prototypes:
            return category_prototypes[normalize_key(c)]
    return None


def load_scan(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_probe(scan_json: Path) -> Dict:
    scan = load_scan(scan_json)
    rows = scan.get("noun_records") or []
    noun_map = {}
    for r in rows:
        key = normalize_key(r.get("noun", ""))
        if key:
            noun_map[key] = r

    apple = pick_first(noun_map, ["apple", "苹果"])
    king = pick_first(noun_map, ["king", "国王"])
    queen = pick_first(noun_map, ["queen", "王后"])
    if not apple or not king or not queen:
        raise RuntimeError("缺少 apple/king/queen 中的至少一个概念，无法执行三联探测。")

    cat_proto = {
        normalize_key(k): v for k, v in (scan.get("category_prototypes") or {}).items()
    }
    fruit_proto = pick_proto(cat_proto, ["fruit", "水果"])
    human_proto = pick_proto(cat_proto, ["human", "person", "人类", "人"])

    a_sig = apple.get("signature_top_indices") or []
    k_sig = king.get("signature_top_indices") or []
    q_sig = queen.get("signature_top_indices") or []

    a_l = layer_vec(apple.get("signature_layer_distribution") or {})
    k_l = layer_vec(king.get("signature_layer_distribution") or {})
    q_l = layer_vec(queen.get("signature_layer_distribution") or {})

    a_v = signature_hash_vec(a_sig)
    k_v = signature_hash_vec(k_sig)
    q_v = signature_hash_vec(q_sig)
    gender_axis = q_v - k_v
    gender_axis /= float(np.linalg.norm(gender_axis) + 1e-12)

    apple_axis_proj = abs(cosine(a_v, gender_axis))
    king_axis_proj = abs(cosine(k_v, gender_axis))
    queen_axis_proj = abs(cosine(q_v, gender_axis))

    if fruit_proto:
        fruit_sig = fruit_proto.get("prototype_top_indices") or []
        apple_fruit_shared = jaccard(a_sig, fruit_sig)
    else:
        apple_fruit_shared = 0.0

    if human_proto:
        human_sig = human_proto.get("prototype_top_indices") or []
        king_human_shared = jaccard(k_sig, human_sig)
        queen_human_shared = jaccard(q_sig, human_sig)
        apple_human_shared = jaccard(a_sig, human_sig)
    else:
        king_human_shared = 0.0
        queen_human_shared = 0.0
        apple_human_shared = 0.0

    jk = jaccard(a_sig, k_sig)
    jq = jaccard(a_sig, q_sig)
    kq = jaccard(k_sig, q_sig)

    metrics = {
        "apple_king_jaccard": jk,
        "apple_queen_jaccard": jq,
        "king_queen_jaccard": kq,
        "apple_king_layer_cosine": cosine(a_l, k_l),
        "apple_queen_layer_cosine": cosine(a_l, q_l),
        "king_queen_layer_cosine": cosine(k_l, q_l),
        "apple_axis_projection_abs": apple_axis_proj,
        "king_axis_projection_abs": king_axis_proj,
        "queen_axis_projection_abs": queen_axis_proj,
        "apple_fruit_shared_ratio": apple_fruit_shared,
        "apple_human_shared_ratio": apple_human_shared,
        "king_human_shared_ratio": king_human_shared,
        "queen_human_shared_ratio": queen_human_shared,
        "triplet_separability_index": float(kq - (jk + jq) / 2.0),
        "axis_specificity_index": float(((king_axis_proj + queen_axis_proj) / 2.0) - apple_axis_proj),
    }
    return metrics


def render_md(payload: Dict) -> str:
    m = payload["metrics"]
    lines = [
        "# Apple/King/Queen 三联结构探测报告",
        "",
        "## 核心指标",
        f"- king-queen 重叠: {m['king_queen_jaccard']:.4f}",
        f"- apple-king 重叠: {m['apple_king_jaccard']:.4f}",
        f"- apple-queen 重叠: {m['apple_queen_jaccard']:.4f}",
        f"- 三联可分离指数: {m['triplet_separability_index']:.4f}",
        f"- 轴特异性指数: {m['axis_specificity_index']:.4f}",
        "",
        "## 解释",
        "- 若三联可分离指数 > 0，说明 king/queen 更共享结构，而 apple 更偏离人类角色轴。",
        "- 若轴特异性指数 > 0，说明 king/queen 对关系轴投影强于 apple。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek7B apple-king-queen triplet probe (offline)")
    ap.add_argument(
        "--mass-json",
        default="tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed101/mass_noun_encoding_scan.json",
    )
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_triplet_probe_{ts}")

    metrics = run_probe(Path(args.mass_json))
    payload = build_result_payload(
        experiment_id="triplet_probe_apple_king_queen_v1",
        title="Apple/King/Queen 三联结构探测",
        config={"mass_json": args.mass_json},
        metrics=metrics,
        hypotheses=[
            {
                "id": "H_triplet_separable",
                "rule": "triplet_separability_index > 0",
                "pass": bool(metrics["triplet_separability_index"] > 0),
            },
            {
                "id": "H_axis_specific",
                "rule": "axis_specificity_index > 0",
                "pass": bool(metrics["axis_specificity_index"] > 0),
            },
            {
                "id": "H_category_anchor",
                "rule": "apple_fruit_shared_ratio > apple_human_shared_ratio",
                "pass": bool(metrics["apple_fruit_shared_ratio"] > metrics["apple_human_shared_ratio"]),
            },
        ],
        notes=["离线结构探测，后续需在线钩子/消融因果实验复核。"],
    )

    paths = write_result_bundle(
        out_dir=out_dir,
        base_name="apple_king_queen_triplet_probe",
        payload=payload,
        report_md=render_md(payload),
    )
    print(f"[OK] {paths['json']}")
    print(f"[OK] {paths['md']}")


if __name__ == "__main__":
    main()

