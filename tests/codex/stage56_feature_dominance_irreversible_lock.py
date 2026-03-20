from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_irreversible_lock_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_dominance_irreversible_lock_summary() -> dict:
    irrev = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_irreversibility_20260320" / "summary.json"
    )
    circuit_v6 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_near_direct_v6_20260320" / "summary.json"
    )

    hi = irrev["headline_metrics"]
    hc = circuit_v6["headline_metrics"]

    lock_gain = hi["irreversible_gain"] + 0.4 * hi["irreversible_margin"]
    lock_gap = 0.6 * hi["irreversible_gap"] + 0.4 * hc["direct_gate_v6"]
    lock_margin = lock_gain - lock_gap
    lock_ratio = lock_gain / max(lock_gap, 1e-9)

    return {
        "headline_metrics": {
            "lock_gain": lock_gain,
            "lock_gap": lock_gap,
            "lock_margin": lock_margin,
            "lock_ratio": lock_ratio,
        },
        "lock_equation": {
            "gain_term": "G_lock2 = irreversible_gain + 0.4 * irreversible_margin",
            "gap_term": "P_lock2 = 0.6 * irreversible_gap + 0.4 * direct_gate_v6",
            "margin_term": "M_lock2 = G_lock2 - P_lock2",
            "ratio_term": "R_lock2 = G_lock2 / P_lock2",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征提取不可逆锁死报告",
        "",
        f"- lock_gain: {hm['lock_gain']:.6f}",
        f"- lock_gap: {hm['lock_gap']:.6f}",
        f"- lock_margin: {hm['lock_margin']:.6f}",
        f"- lock_ratio: {hm['lock_ratio']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_dominance_irreversible_lock_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
