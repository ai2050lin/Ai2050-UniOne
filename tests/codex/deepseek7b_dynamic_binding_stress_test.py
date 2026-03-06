#!/usr/bin/env python
"""
硬伤攻坚 1：动态绑定（Dynamic Binding）压力测试。

说明：
- 默认离线读取已有 mass_noun_encoding_scan.json，不做模型前向推理。
- 使用角色绑定（subject / relation / object）构造组合表示。
- 评估角色交换错误率、碰撞率与绑定稳定性。
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


def signature_to_vec(sig: Sequence[int], dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for x in sig:
        h = (int(x) * 1315423911) % dim
        v[h] += 1.0
    n = float(np.linalg.norm(v) + 1e-12)
    return v / n


def circular_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    fa = np.fft.rfft(a)
    fb = np.fft.rfft(b)
    c = np.fft.irfft(fa * fb, n=a.shape[0]).astype(np.float32)
    n = float(np.linalg.norm(c) + 1e-12)
    return c / n


def load_mass_scan(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_vocab(mass: Dict, top_sig_k: int, max_concepts: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    rows = mass.get("noun_records") or []
    for row in rows[:max_concepts]:
        noun = str(row.get("noun", "")).strip().lower()
        sig = (row.get("signature_top_indices") or [])[:top_sig_k]
        if noun and sig:
            out[noun] = np.asarray(sig, dtype=np.int64)
    return out


def pick_triples(concepts: List[str], n_samples: int, seed: int) -> List[Tuple[str, str, str]]:
    rng = random.Random(seed)
    rel_candidates = ["eat", "see", "chase", "teach", "build", "carry"]
    triples = []
    for _ in range(n_samples):
        s = rng.choice(concepts)
        o = rng.choice(concepts)
        r = rng.choice(rel_candidates)
        triples.append((s, r, o))
    return triples


def decode_role(phrase_vec: np.ndarray, role_vec: np.ndarray, concept_vecs: Dict[str, np.ndarray]) -> str:
    probe = circular_bind(role_vec, phrase_vec)
    best = ""
    best_score = -1e9
    for name, vec in concept_vecs.items():
        score = cosine(probe, vec)
        if score > best_score:
            best_score = score
            best = name
    return best


def run_test(
    mass_json: Path,
    *,
    seed: int,
    vec_dim: int,
    top_sig_k: int,
    max_concepts: int,
    n_samples: int,
) -> Dict:
    random.seed(seed)
    np.random.seed(seed)

    mass = load_mass_scan(mass_json)
    vocab_sig = build_vocab(mass, top_sig_k=top_sig_k, max_concepts=max_concepts)
    concepts = sorted(vocab_sig.keys())
    if len(concepts) < 12:
        raise RuntimeError("可用概念不足，无法执行动态绑定测试。")

    concept_vecs = {k: signature_to_vec(v, vec_dim) for k, v in vocab_sig.items()}

    role_subj = np.random.normal(size=vec_dim).astype(np.float32)
    role_obj = np.random.normal(size=vec_dim).astype(np.float32)
    role_rel = np.random.normal(size=vec_dim).astype(np.float32)
    for rv in (role_subj, role_obj, role_rel):
        rv /= float(np.linalg.norm(rv) + 1e-12)

    triples = pick_triples(concepts, n_samples=n_samples, seed=seed)
    subject_ok = 0
    object_ok = 0
    swap_error = 0

    for s, _r, o in triples:
        s_vec = concept_vecs[s]
        o_vec = concept_vecs[o]
        rel_dummy = np.ones(vec_dim, dtype=np.float32)

        phrase = (
            circular_bind(role_subj, s_vec)
            + circular_bind(role_obj, o_vec)
            + circular_bind(role_rel, rel_dummy)
        ).astype(np.float32)
        phrase /= float(np.linalg.norm(phrase) + 1e-12)

        pred_s = decode_role(phrase, role_subj, concept_vecs)
        pred_o = decode_role(phrase, role_obj, concept_vecs)

        subject_ok += int(pred_s == s)
        object_ok += int(pred_o == o)
        if pred_s == o or pred_o == s:
            swap_error += 1

    total = max(1, len(triples))
    subject_acc = float(subject_ok / total)
    object_acc = float(object_ok / total)
    role_swap_error_rate = float(swap_error / total)
    binding_stability = float((subject_acc + object_acc) / 2.0 - role_swap_error_rate)
    collision_rate = float(1.0 - ((subject_acc + object_acc) / 2.0))

    return {
        "subject_decode_accuracy": subject_acc,
        "object_decode_accuracy": object_acc,
        "role_swap_error_rate": role_swap_error_rate,
        "collision_rate_top1": collision_rate,
        "binding_stability_index": binding_stability,
        "n_concepts": len(concepts),
        "n_samples": len(triples),
    }


def render_md(payload: Dict) -> str:
    m = payload["metrics"]
    lines = [
        "# 动态绑定硬伤压力测试报告",
        "",
        "## 关键指标",
        f"- 主语解码准确率: {m['subject_decode_accuracy']:.4f}",
        f"- 宾语解码准确率: {m['object_decode_accuracy']:.4f}",
        f"- 角色交换错误率: {m['role_swap_error_rate']:.4f}",
        f"- Top1 碰撞率: {m['collision_rate_top1']:.4f}",
        f"- 绑定稳定性指数: {m['binding_stability_index']:.4f}",
        "",
        "## 解释",
        "- 若角色交换错误率持续偏高，说明静态向量叠加不足以承载复杂动态绑定。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Dynamic binding stress test (offline)")
    ap.add_argument(
        "--mass-json",
        default="tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed101/mass_noun_encoding_scan.json",
    )
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--vec-dim", type=int, default=4096)
    ap.add_argument("--top-sig-k", type=int, default=120)
    ap.add_argument("--max-concepts", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=240)
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_dynamic_binding_stress_{ts}")

    metrics = run_test(
        Path(args.mass_json),
        seed=args.seed,
        vec_dim=args.vec_dim,
        top_sig_k=args.top_sig_k,
        max_concepts=args.max_concepts,
        n_samples=args.n_samples,
    )

    payload = build_result_payload(
        experiment_id="hard_problem_dynamic_binding_v1",
        title="动态绑定硬伤压力测试",
        config={
            "mass_json": args.mass_json,
            "seed": args.seed,
            "vec_dim": args.vec_dim,
            "top_sig_k": args.top_sig_k,
            "max_concepts": args.max_concepts,
            "n_samples": args.n_samples,
        },
        metrics=metrics,
        hypotheses=[
            {
                "id": "H_bind_role_separable",
                "rule": "role_swap_error_rate < 0.20",
                "pass": bool(metrics["role_swap_error_rate"] < 0.20),
            },
            {
                "id": "H_bind_stable",
                "rule": "binding_stability_index > 0.50",
                "pass": bool(metrics["binding_stability_index"] > 0.50),
            },
        ],
        notes=["用于定位动态绑定硬伤，属于结构层压力测试，不代表最终语义上限。"],
    )

    paths = write_result_bundle(
        out_dir=out_dir,
        base_name="dynamic_binding_stress_test",
        payload=payload,
        report_md=render_md(payload),
    )
    print(f"[OK] {paths['json']}")
    print(f"[OK] {paths['md']}")


if __name__ == "__main__":
    main()
