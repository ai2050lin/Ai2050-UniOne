#!/usr/bin/env python3
"""Validate multi-axis encoding laws on a synthetic compositional code model.

Axes:
- generation axes: style / logic / syntax
- concept axes: micro / meso / macro

Outputs JSON metrics for separability, compositional transport, and apple-centric stability.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


STYLE = ["chat", "paper"]
LOGIC = ["definition", "causal", "comparison"]
SYNTAX = ["declarative", "question", "attribute"]
MESO = ["apple", "banana", "king", "queen", "car"]
MICRO = ["red", "sweet", "round", "heavy", "male", "female"]
MACRO = ["eat", "trade", "justice", "infinite"]


@dataclass
class AxisMetric:
    axis: str
    accuracy: float
    n_classes: int


@dataclass
class AppleMetric:
    micro_context_stability: float
    meso_margin_apple_vs_banana: float
    macro_transport_consistency: float


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def random_codebook(rng: np.random.Generator, labels: Sequence[str], d: int) -> Dict[str, np.ndarray]:
    mat = rng.normal(0.0, 1.0, size=(len(labels), d))
    mat = np.array([unit(x) for x in mat])
    return {k: mat[i] for i, k in enumerate(labels)}


def encode_sample(
    style: str,
    logic: str,
    syntax: str,
    meso: str,
    micros: Sequence[str],
    macros: Sequence[str],
    cb: Dict[str, Dict[str, np.ndarray]],
    eta_bind: float,
    noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    v_style = cb["style"][style]
    v_logic = cb["logic"][logic]
    v_syntax = cb["syntax"][syntax]
    v_meso = cb["meso"][meso]

    v_micro = np.zeros_like(v_meso)
    for m in micros:
        v_micro += cb["micro"][m]
    if len(micros) > 0:
        v_micro /= float(len(micros))

    v_macro = np.zeros_like(v_meso)
    for g in macros:
        v_macro += cb["macro"][g]
    if len(macros) > 0:
        v_macro /= float(len(macros))

    # Factorized base code.
    h = 0.9 * v_style + 1.0 * v_logic + 1.0 * v_syntax + 1.3 * v_meso + 1.1 * v_micro + 1.1 * v_macro

    # Nonlinear binding terms emulate learned mixed features in MLP blocks.
    h += eta_bind * (v_meso * v_micro)
    h += 0.5 * eta_bind * (v_logic * v_syntax)
    h += 0.35 * eta_bind * (v_macro * v_meso)

    if noise > 0.0:
        h += rng.normal(0.0, noise, size=h.shape)

    return unit(h)


def nearest_centroid_accuracy(X: np.ndarray, y: List[str], labels: Sequence[str]) -> float:
    idx = np.arange(len(y))
    split = int(0.7 * len(y))
    tr, te = idx[:split], idx[split:]

    centroids: Dict[str, np.ndarray] = {}
    for lab in labels:
        mask = [i for i in tr if y[i] == lab]
        if not mask:
            continue
        centroids[lab] = unit(np.mean(X[mask], axis=0))

    if len(te) == 0:
        return 0.0

    correct = 0
    for i in te:
        sims = {lab: cosine(X[i], c) for lab, c in centroids.items()}
        pred = max(sims.items(), key=lambda kv: kv[1])[0]
        if pred == y[i]:
            correct += 1
    return correct / len(te)


def generate_dataset(n: int, d: int, eta_bind: float, noise: float, seed: int) -> Tuple[np.ndarray, Dict[str, List[str]], Dict[str, Dict[str, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    cb = {
        "style": random_codebook(rng, STYLE, d),
        "logic": random_codebook(rng, LOGIC, d),
        "syntax": random_codebook(rng, SYNTAX, d),
        "meso": random_codebook(rng, MESO, d),
        "micro": random_codebook(rng, MICRO, d),
        "macro": random_codebook(rng, MACRO, d),
    }

    # Force a stable gender direction for analogy-like behavior.
    g_dir = unit(cb["micro"]["female"] - cb["micro"]["male"])
    cb["meso"]["queen"] = unit(cb["meso"]["king"] + 0.7 * g_dir)

    X = []
    y_style: List[str] = []
    y_logic: List[str] = []
    y_syntax: List[str] = []
    y_meso: List[str] = []

    for _ in range(n):
        st = STYLE[rng.integers(0, len(STYLE))]
        lg = LOGIC[rng.integers(0, len(LOGIC))]
        sy = SYNTAX[rng.integers(0, len(SYNTAX))]
        me = MESO[rng.integers(0, len(MESO))]

        if me == "apple":
            micro_pool = ["red", "sweet", "round"]
            macro_pool = ["eat", "trade", "justice"]
        elif me == "banana":
            micro_pool = ["sweet", "round"]
            macro_pool = ["eat", "trade"]
        elif me in ("king", "queen"):
            micro_pool = ["male"] if me == "king" else ["female"]
            macro_pool = ["justice", "trade"]
        else:
            micro_pool = ["heavy", "round"]
            macro_pool = ["trade", "infinite"]

        m_cnt = 2 if len(micro_pool) >= 2 else 1
        g_cnt = 2 if len(macro_pool) >= 2 else 1
        micros = list(rng.choice(micro_pool, size=m_cnt, replace=False))
        macros = list(rng.choice(macro_pool, size=g_cnt, replace=False))

        h = encode_sample(st, lg, sy, me, micros, macros, cb, eta_bind=eta_bind, noise=noise, rng=rng)
        X.append(h)
        y_style.append(st)
        y_logic.append(lg)
        y_syntax.append(sy)
        y_meso.append(me)

    arr = np.asarray(X, dtype=np.float64)
    labels = {
        "style": y_style,
        "logic": y_logic,
        "syntax": y_syntax,
        "meso": y_meso,
    }
    return arr, labels, cb


def apple_metrics(cb: Dict[str, Dict[str, np.ndarray]], d: int, eta_bind: float, noise: float, seed: int) -> AppleMetric:
    rng = np.random.default_rng(seed + 99)

    # Micro context stability: same apple micro attrs under style/logic/syntax variation.
    refs: List[np.ndarray] = []
    for st in STYLE:
        for lg in LOGIC:
            for sy in SYNTAX:
                refs.append(
                    encode_sample(
                        st,
                        lg,
                        sy,
                        "apple",
                        ["red", "sweet"],
                        ["eat", "trade"],
                        cb,
                        eta_bind=eta_bind,
                        noise=noise,
                        rng=rng,
                    )
                )
    center = unit(np.mean(refs, axis=0))
    stabilities = [cosine(x, center) for x in refs]
    micro_context_stability = float(np.mean(stabilities))

    # Meso margin apple vs banana in matched context.
    ctx = ("paper", "definition", "declarative", ["sweet", "round"], ["eat", "trade"])
    a = encode_sample(ctx[0], ctx[1], ctx[2], "apple", ctx[3], ctx[4], cb, eta_bind=eta_bind, noise=noise, rng=rng)
    b = encode_sample(ctx[0], ctx[1], ctx[2], "banana", ctx[3], ctx[4], cb, eta_bind=eta_bind, noise=noise, rng=rng)
    aa = cosine(a, cb["meso"]["apple"])
    ba = cosine(b, cb["meso"]["apple"])
    meso_margin_apple_vs_banana = float(aa - ba)

    # Macro transport consistency: (verb+noun)-noun should align across nouns for same verb.
    def transport(noun: str, macro: str) -> np.ndarray:
        base = encode_sample("paper", "causal", "declarative", noun, ["round"], ["trade"], cb, eta_bind=eta_bind, noise=noise, rng=rng)
        with_macro = encode_sample("paper", "causal", "declarative", noun, ["round"], ["trade", macro], cb, eta_bind=eta_bind, noise=noise, rng=rng)
        return unit(with_macro - base)

    eat_dirs = [transport(n, "eat") for n in ["apple", "banana", "car"]]
    cons = [cosine(eat_dirs[0], eat_dirs[1]), cosine(eat_dirs[0], eat_dirs[2]), cosine(eat_dirs[1], eat_dirs[2])]
    macro_transport_consistency = float(np.mean(cons))

    return AppleMetric(
        micro_context_stability=micro_context_stability,
        meso_margin_apple_vs_banana=meso_margin_apple_vs_banana,
        macro_transport_consistency=macro_transport_consistency,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-axis encoding law probe")
    parser.add_argument("--samples", type=int, default=2400)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--eta-bind", type=float, default=0.16)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260306)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    X, labels, cb = generate_dataset(
        n=args.samples,
        d=args.dim,
        eta_bind=args.eta_bind,
        noise=args.noise,
        seed=args.seed,
    )

    axis_results: List[AxisMetric] = [
        AxisMetric("style", nearest_centroid_accuracy(X, labels["style"], STYLE), len(STYLE)),
        AxisMetric("logic", nearest_centroid_accuracy(X, labels["logic"], LOGIC), len(LOGIC)),
        AxisMetric("syntax", nearest_centroid_accuracy(X, labels["syntax"], SYNTAX), len(SYNTAX)),
        AxisMetric("meso", nearest_centroid_accuracy(X, labels["meso"], MESO), len(MESO)),
    ]

    # Analogy-like local linearity check.
    v_king = cb["meso"]["king"]
    v_queen = cb["meso"]["queen"]
    v_male = cb["micro"]["male"]
    v_female = cb["micro"]["female"]
    gender_offset_consistency = cosine(unit(v_queen - v_king), unit(v_female - v_male))

    apple = apple_metrics(cb, args.dim, args.eta_bind, args.noise, args.seed)

    payload = {
        "config": {
            "samples": args.samples,
            "dim": args.dim,
            "eta_bind": args.eta_bind,
            "noise": args.noise,
            "seed": args.seed,
        },
        "axis_separability": [asdict(x) for x in axis_results],
        "analogy_local_linearity": {
            "gender_offset_consistency": float(gender_offset_consistency),
        },
        "apple_three_level_metrics": asdict(apple),
        "theory_checklist": {
            "law_A_disentangled_bind_on_demand": bool(apple.micro_context_stability > 0.75),
            "law_B_dynamic_relation_transport": bool(apple.macro_transport_consistency > 0.25),
            "law_C_entity_cluster_separability": bool(apple.meso_margin_apple_vs_banana > 0.1),
        },
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
