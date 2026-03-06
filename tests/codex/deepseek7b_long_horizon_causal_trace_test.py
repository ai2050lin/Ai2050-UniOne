#!/usr/bin/env python
"""
硬伤攻坚 2：长程因果链路测试（A->B->C...）。

说明：
- 离线读取 mass_noun_encoding_scan.json。
- 在签名子空间上构造多跳链路，评估：
  1) 多跳可恢复率
  2) 中间层衰减
  3) 跨层传输稳定性
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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a) + 1e-12)
    nb = float(np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / (na * nb))


def load_mass_scan(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def signature_to_vec(sig: Sequence[int], dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for x in sig:
        h = (int(x) * 2654435761) % dim
        v[h] += 1.0
    v /= float(np.linalg.norm(v) + 1e-12)
    return v


def layer_profile_to_vec(dist: Dict, n_layers: int = 28) -> np.ndarray:
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


def build_vocab(mass: Dict, top_sig_k: int, max_concepts: int) -> Dict[str, Dict]:
    rows = mass.get("noun_records") or []
    out: Dict[str, Dict] = {}
    for row in rows[:max_concepts]:
        noun = str(row.get("noun", "")).strip().lower()
        sig = (row.get("signature_top_indices") or [])[:top_sig_k]
        if not noun or not sig:
            continue
        out[noun] = {
            "sig": np.asarray(sig, dtype=np.int64),
            "layer_profile": row.get("signature_layer_distribution") or {},
        }
    return out


def build_chain(concepts: List[str], length: int, rng: random.Random) -> List[str]:
    if len(concepts) < length:
        return concepts[:]
    picks = rng.sample(concepts, k=length)
    return picks


def run_test(
    mass_json: Path,
    *,
    seed: int,
    vec_dim: int,
    top_sig_k: int,
    max_concepts: int,
    chain_len: int,
    n_chains: int,
) -> Dict:
    random.seed(seed)
    np.random.seed(seed)

    mass = load_mass_scan(mass_json)
    vocab = build_vocab(mass, top_sig_k=top_sig_k, max_concepts=max_concepts)
    concepts = sorted(vocab.keys())
    if len(concepts) < max(12, chain_len):
        raise RuntimeError("可用概念不足，无法执行长程因果链路测试。")

    vecs = {k: signature_to_vec(v["sig"], vec_dim) for k, v in vocab.items()}
    layers = {k: layer_profile_to_vec(v["layer_profile"]) for k, v in vocab.items()}

    rng = random.Random(seed)
    hop_recover = []
    hop_layer_stability = []

    for _ in range(n_chains):
        chain = build_chain(concepts, chain_len, rng)
        # v_trace 模拟 A->B->C... 累积状态
        v_trace = vecs[chain[0]].copy()
        l_trace = layers[chain[0]].copy()
        for hi in range(1, len(chain)):
            node = chain[hi]
            v_target = vecs[node]
            l_target = layers[node]

            # 简化的链路更新：保留历史 + 注入当前节点
            v_trace = 0.72 * v_trace + 0.28 * v_target
            v_trace /= float(np.linalg.norm(v_trace) + 1e-12)

            l_trace = 0.68 * l_trace + 0.32 * l_target
            l_trace /= float(np.linalg.norm(l_trace) + 1e-12)

            hop_recover.append(cosine(v_trace, v_target))
            hop_layer_stability.append(cosine(l_trace, l_target))

    # 前跳 vs 后跳衰减
    h = max(1, len(hop_recover))
    early = hop_recover[: max(1, h // 3)]
    late = hop_recover[-max(1, h // 3) :]
    early_mean = float(np.mean(early))
    late_mean = float(np.mean(late))
    decay = float(early_mean - late_mean)

    return {
        "chain_length": chain_len,
        "n_chains": n_chains,
        "hop_recovery_mean": float(np.mean(hop_recover) if hop_recover else 0.0),
        "hop_recovery_std": float(np.std(hop_recover) if hop_recover else 0.0),
        "layer_transport_stability_mean": float(np.mean(hop_layer_stability) if hop_layer_stability else 0.0),
        "layer_transport_stability_std": float(np.std(hop_layer_stability) if hop_layer_stability else 0.0),
        "early_recovery_mean": early_mean,
        "late_recovery_mean": late_mean,
        "long_horizon_decay": decay,
        "n_concepts": len(concepts),
    }


def render_md(payload: Dict) -> str:
    m = payload["metrics"]
    lines = [
        "# 长程因果链路测试报告",
        "",
        "## 关键指标",
        f"- 多跳恢复均值: {m['hop_recovery_mean']:.4f} ± {m['hop_recovery_std']:.4f}",
        f"- 层传输稳定性: {m['layer_transport_stability_mean']:.4f} ± {m['layer_transport_stability_std']:.4f}",
        f"- 前段恢复: {m['early_recovery_mean']:.4f}",
        f"- 后段恢复: {m['late_recovery_mean']:.4f}",
        f"- 长程衰减: {m['long_horizon_decay']:.4f}",
        "",
        "## 解释",
        "- 衰减越小，表示链路跨跳保持能力越强；衰减偏大说明长程信用传输仍是硬伤。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Long-horizon causal trace test (offline)")
    ap.add_argument(
        "--mass-json",
        default="tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed101/mass_noun_encoding_scan.json",
    )
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--vec-dim", type=int, default=4096)
    ap.add_argument("--top-sig-k", type=int, default=120)
    ap.add_argument("--max-concepts", type=int, default=100)
    ap.add_argument("--chain-len", type=int, default=6)
    ap.add_argument("--n-chains", type=int, default=160)
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_long_horizon_trace_{ts}")

    metrics = run_test(
        Path(args.mass_json),
        seed=args.seed,
        vec_dim=args.vec_dim,
        top_sig_k=args.top_sig_k,
        max_concepts=args.max_concepts,
        chain_len=args.chain_len,
        n_chains=args.n_chains,
    )

    payload = build_result_payload(
        experiment_id="hard_problem_long_horizon_trace_v1",
        title="长程因果链路压力测试",
        config={
            "mass_json": args.mass_json,
            "seed": args.seed,
            "vec_dim": args.vec_dim,
            "top_sig_k": args.top_sig_k,
            "max_concepts": args.max_concepts,
            "chain_len": args.chain_len,
            "n_chains": args.n_chains,
        },
        metrics=metrics,
        hypotheses=[
            {
                "id": "H_long_horizon_decay",
                "rule": "long_horizon_decay < 0.08",
                "pass": bool(metrics["long_horizon_decay"] < 0.08),
            },
            {
                "id": "H_transport_stable",
                "rule": "layer_transport_stability_mean > 0.70",
                "pass": bool(metrics["layer_transport_stability_mean"] > 0.70),
            },
        ],
        notes=["用于定位多跳链路中的长程衰减与层间传输稳定性。"],
    )

    paths = write_result_bundle(
        out_dir=out_dir,
        base_name="long_horizon_causal_trace_test",
        payload=payload,
        report_md=render_md(payload),
    )
    print(f"[OK] {paths['json']}")
    print(f"[OK] {paths['md']}")


if __name__ == "__main__":
    main()

