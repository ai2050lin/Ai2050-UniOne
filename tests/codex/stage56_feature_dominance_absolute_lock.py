from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_absolute_lock_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_dominance_absolute_lock_summary() -> dict:
    lock = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_irreversible_lock_20260320" / "summary.json"
    )
    circuit_v7 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_near_direct_v7_20260320" / "summary.json"
    )

    hl = lock["headline_metrics"]
    hc = circuit_v7["headline_metrics"]

    absolute_gain = hl["lock_gain"] + 0.35 * hl["lock_margin"]
    absolute_gap = 0.55 * hl["lock_gap"] + 0.45 * hc["direct_gate_v7"]
    absolute_margin = absolute_gain - absolute_gap
    absolute_ratio = absolute_gain / max(absolute_gap, 1e-9)

    return {
        "headline_metrics": {
            "absolute_gain": absolute_gain,
            "absolute_gap": absolute_gap,
            "absolute_margin": absolute_margin,
            "absolute_ratio": absolute_ratio,
        },
        "absolute_lock_equation": {
            "gain_term": "G_abs = lock_gain + 0.35 * lock_margin",
            "gap_term": "P_abs = 0.55 * lock_gap + 0.45 * direct_gate_v7",
            "margin_term": "M_abs = G_abs - P_abs",
            "ratio_term": "R_abs = G_abs / P_abs",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征提取绝对锁死报告",
        "",
        f"- absolute_gain: {hm['absolute_gain']:.6f}",
        f"- absolute_gap: {hm['absolute_gap']:.6f}",
        f"- absolute_margin: {hm['absolute_margin']:.6f}",
        f"- absolute_ratio: {hm['absolute_ratio']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_dominance_absolute_lock_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
