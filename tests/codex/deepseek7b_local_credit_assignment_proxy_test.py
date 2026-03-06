#!/usr/bin/env python
"""
硬伤攻坚 3：全局信用分配的局部化代理测试。

说明：
- 离线读取 mass_noun_encoding_scan.json。
- 用类别原型与概念签名评估：
  1) 局部子集是否能恢复全局类别判别（local sufficiency）
  2) 局部子集对跨类别是否具区分性（local selectivity）
  3) 局部-全局一致性（local-global consistency）
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from agi_research_result_schema import build_result_payload, write_result_bundle


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb) / max(1, len(sa | sb)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a) + 1e-12)
    nb = float(np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / (na * nb))


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def topk_subset(sig: Sequence[int], k: int) -> List[int]:
    return [int(x) for x in list(sig)[: max(1, k)]]


def run_test(mass_json: Path, *, topk_local: int, max_nouns: int, seed: int) -> Dict:
    random.seed(seed)
    np.random.seed(seed)

    data = load_json(mass_json)
    rows = (data.get("noun_records") or [])[:max_nouns]
    if not rows:
        raise RuntimeError("未找到 noun_records，无法执行局部信用分配测试。")

    category_proto = data.get("category_prototypes") or {}
    by_cat: Dict[str, List[Dict]] = {}
    for row in rows:
        cat = str(row.get("category", "unknown")).strip().lower()
        by_cat.setdefault(cat, []).append(row)

    local_suff = []
    local_sel = []
    local_global_consistency = []

    categories = sorted(by_cat.keys())
    for cat, cat_rows in by_cat.items():
        proto = (category_proto.get(cat) or {}).get("prototype_top_indices") or []
        if not proto:
            continue
        neg_cats = [c for c in categories if c != cat and (category_proto.get(c) or {}).get("prototype_top_indices")]
        for row in cat_rows:
            sig = row.get("signature_top_indices") or []
            local = topk_subset(sig, topk_local)

            # 局部充分性：局部子集与本类原型重叠
            s_pos = jaccard(local, proto)
            local_suff.append(s_pos)

            # 局部选择性：与它类原型重叠应低
            neg_scores = []
            for nc in neg_cats:
                nproto = (category_proto.get(nc) or {}).get("prototype_top_indices") or []
                if nproto:
                    neg_scores.append(jaccard(local, nproto))
            s_neg = float(np.mean(neg_scores) if neg_scores else 0.0)
            local_sel.append(max(0.0, s_pos - s_neg))

            # 局部-全局一致性：层分布方向是否一致
            v_local = layer_vec(row.get("signature_layer_distribution") or {})
            v_proto = layer_vec((category_proto.get(cat) or {}).get("prototype_layer_distribution") or {})
            local_global_consistency.append(cosine(v_local, v_proto))

    metrics = {
        "local_subset_size": int(topk_local),
        "local_sufficiency_mean": float(np.mean(local_suff) if local_suff else 0.0),
        "local_sufficiency_std": float(np.std(local_suff) if local_suff else 0.0),
        "local_selectivity_mean": float(np.mean(local_sel) if local_sel else 0.0),
        "local_selectivity_std": float(np.std(local_sel) if local_sel else 0.0),
        "local_global_consistency_mean": float(np.mean(local_global_consistency) if local_global_consistency else 0.0),
        "local_global_consistency_std": float(np.std(local_global_consistency) if local_global_consistency else 0.0),
        "n_categories": len(categories),
        "n_samples": len(rows),
    }
    return metrics


def render_md(payload: Dict) -> str:
    m = payload["metrics"]
    lines = [
        "# 局部信用分配代理测试报告",
        "",
        "## 关键指标",
        f"- 局部子集大小: {m['local_subset_size']}",
        f"- 局部充分性: {m['local_sufficiency_mean']:.4f} ± {m['local_sufficiency_std']:.4f}",
        f"- 局部选择性: {m['local_selectivity_mean']:.4f} ± {m['local_selectivity_std']:.4f}",
        f"- 局部-全局一致性: {m['local_global_consistency_mean']:.4f} ± {m['local_global_consistency_std']:.4f}",
        "",
        "## 解释",
        "- 若局部充分性与选择性都偏低，说明全局信用分配仍难以通过局部子集近似复现。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Local credit assignment proxy test (offline)")
    ap.add_argument(
        "--mass-json",
        default="tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed101/mass_noun_encoding_scan.json",
    )
    ap.add_argument("--topk-local", type=int, default=24)
    ap.add_argument("--max-nouns", type=int, default=120)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_local_credit_proxy_{ts}")

    metrics = run_test(
        Path(args.mass_json),
        topk_local=args.topk_local,
        max_nouns=args.max_nouns,
        seed=args.seed,
    )

    payload = build_result_payload(
        experiment_id="hard_problem_local_credit_assignment_v1",
        title="局部信用分配代理测试",
        config={
            "mass_json": args.mass_json,
            "topk_local": args.topk_local,
            "max_nouns": args.max_nouns,
            "seed": args.seed,
        },
        metrics=metrics,
        hypotheses=[
            {
                "id": "H_local_sufficient",
                "rule": "local_sufficiency_mean > 0.20",
                "pass": bool(metrics["local_sufficiency_mean"] > 0.20),
            },
            {
                "id": "H_local_selective",
                "rule": "local_selectivity_mean > 0.08",
                "pass": bool(metrics["local_selectivity_mean"] > 0.08),
            },
            {
                "id": "H_local_global_consistent",
                "rule": "local_global_consistency_mean > 0.55",
                "pass": bool(metrics["local_global_consistency_mean"] > 0.55),
            },
        ],
        notes=["该测试是全局信用分配局部化的代理指标，不等同于完整训练时梯度追踪。"],
    )

    paths = write_result_bundle(
        out_dir=out_dir,
        base_name="local_credit_assignment_proxy_test",
        payload=payload,
        report_md=render_md(payload),
    )
    print(f"[OK] {paths['json']}")
    print(f"[OK] {paths['md']}")


if __name__ == "__main__":
    main()

