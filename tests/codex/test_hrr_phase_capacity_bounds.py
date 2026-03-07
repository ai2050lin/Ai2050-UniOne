#!/usr/bin/env python3
"""Empirical sanity checks for HRR capacity scaling and phase-gating integral."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np


def cconv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.fft.ifft(np.fft.fft(x) * np.fft.fft(y))


def unitary_key(d: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * math.pi, size=d)
    k_hat = np.exp(1j * theta)
    return np.fft.ifft(k_hat)


def key_inverse(k: np.ndarray) -> np.ndarray:
    return np.fft.ifft(np.conj(np.fft.fft(k)))


def random_content(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(0.0, 1.0, size=d)
    n = np.linalg.norm(v)
    if n == 0:
        return random_content(d, rng)
    return v / n


def score(r: np.ndarray, c: np.ndarray) -> float:
    return float(np.real(np.vdot(r, c)))


@dataclass
class HRRGridResult:
    d: int
    m: int
    n_dict: int
    trials: int
    error_rate: float
    mean_margin: float


@dataclass
class PhaseCaseResult:
    case: str
    numeric_gate: float
    analytic_gate: float
    abs_diff: float


def simulate_hrr_grid(
    d_values: List[int],
    m_values: List[int],
    n_dict: int,
    trials: int,
    seed: int,
) -> List[HRRGridResult]:
    rng = np.random.default_rng(seed)
    out: List[HRRGridResult] = []

    for d in d_values:
        for m in m_values:
            errors = 0
            margins: List[float] = []

            for _ in range(trials):
                vs = [random_content(d, rng) for _ in range(m)]
                ks = [unitary_key(d, rng) for _ in range(m)]

                y = np.zeros(d, dtype=np.complex128)
                for i in range(m):
                    y += cconv(vs[i], ks[i])

                r = cconv(y, key_inverse(ks[0]))
                s_true = score(r, vs[0])

                distractors = [random_content(d, rng) for _ in range(n_dict - 1)]
                s_neg_max = max(score(r, dvec) for dvec in distractors)

                margins.append(s_true - s_neg_max)
                if s_neg_max >= s_true:
                    errors += 1

            out.append(
                HRRGridResult(
                    d=d,
                    m=m,
                    n_dict=n_dict,
                    trials=trials,
                    error_rate=errors / trials,
                    mean_margin=float(np.mean(margins)),
                )
            )

    return out


def fit_c_constant(results: List[HRRGridResult]) -> float:
    eps = 1e-9
    c_values: List[float] = []
    for r in results:
        p = min(max(r.error_rate, 0.5 / r.trials), 1.0 - 0.5 / r.trials)
        rhs = p / max(r.n_dict - 1, 1)
        if rhs <= 0:
            continue
        c = -math.log(max(rhs, eps)) * (r.m / r.d)
        c_values.append(c)
    if not c_values:
        return 0.0
    return float(np.median(c_values))


def bound_prediction(result: HRRGridResult, c_const: float) -> float:
    pred = (result.n_dict - 1) * math.exp(-c_const * (result.d / result.m))
    return float(min(max(pred, 0.0), 1.0))


def phase_gate_numeric(a1: float, a2: float, f1: float, f2: float, p1: float, p2: float, t: float, sr: int = 20000) -> float:
    ts = np.linspace(0.0, t, int(sr * t), endpoint=False)
    s1 = a1 * np.cos(2.0 * math.pi * f1 * ts + p1)
    s2 = a2 * np.cos(2.0 * math.pi * f2 * ts + p2)
    return float(np.mean(s1 * s2))


def _int_cos(alpha: float, beta: float, t: float) -> float:
    if abs(alpha) < 1e-12:
        return t * math.cos(beta)
    return (math.sin(alpha * t + beta) - math.sin(beta)) / alpha


def phase_gate_analytic(a1: float, a2: float, f1: float, f2: float, p1: float, p2: float, t: float) -> float:
    w1 = 2.0 * math.pi * f1
    w2 = 2.0 * math.pi * f2
    delta = p1 - p2
    sigma = p1 + p2
    term1 = _int_cos(w1 - w2, delta, t)
    term2 = _int_cos(w1 + w2, sigma, t)
    return float((a1 * a2 / (2.0 * t)) * (term1 + term2))


def phase_cases() -> List[PhaseCaseResult]:
    a1, a2 = 1.0, 1.0
    t = 0.25
    cases = [
        ("sync_same_freq", 40.0, 40.0, 0.0, 0.0),
        ("phase_shift_pi_over_2", 40.0, 40.0, 0.0, math.pi / 2.0),
        ("freq_mismatch", 40.0, 47.0, 0.0, 0.0),
    ]

    out: List[PhaseCaseResult] = []
    for name, f1, f2, p1, p2 in cases:
        num = phase_gate_numeric(a1, a2, f1, f2, p1, p2, t)
        ana = phase_gate_analytic(a1, a2, f1, f2, p1, p2, t)
        out.append(
            PhaseCaseResult(
                case=name,
                numeric_gate=num,
                analytic_gate=ana,
                abs_diff=abs(num - ana),
            )
        )
    return out


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="HRR/phase rigorous bound sanity checks")
    parser.add_argument("--d-values", default="256,512,1024,2048")
    parser.add_argument("--m-values", default="8,16,32,64")
    parser.add_argument("--n-dict", type=int, default=256)
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260306)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    d_values = parse_int_list(args.d_values)
    m_values = parse_int_list(args.m_values)

    hrr_results = simulate_hrr_grid(
        d_values=d_values,
        m_values=m_values,
        n_dict=args.n_dict,
        trials=args.trials,
        seed=args.seed,
    )
    c_const = fit_c_constant(hrr_results)

    hrr_payload: List[Dict[str, float]] = []
    for r in hrr_results:
        row = asdict(r)
        row["predicted_bound"] = bound_prediction(r, c_const)
        hrr_payload.append(row)

    phase_payload = [asdict(x) for x in phase_cases()]

    payload = {
        "config": {
            "d_values": d_values,
            "m_values": m_values,
            "n_dict": args.n_dict,
            "trials": args.trials,
            "seed": args.seed,
        },
        "fit": {"c_constant_median": c_const},
        "hrr_grid": hrr_payload,
        "phase_cases": phase_payload,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
