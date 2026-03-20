from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_icspb_unification_high_closure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_icspb_unification_high_closure_summary() -> dict:
    reinforced = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_icspb_unification_reinforcement_20260320" / "summary.json"
    )
    stable = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_kernel_stability_strengthening_20260320" / "summary.json"
    )

    hr = reinforced["headline_metrics"]
    hs = stable["headline_metrics"]

    object_unification_high = min(1.0, hr["object_unification_reinforced"] + 0.30 * hs["update_retention_stable"])
    transport_unification_high = min(
        1.0,
        hr["transport_unification_reinforced"] + 0.40 * hs["transport_kernel_stability_stable"] * (1.0 - hr["transport_unification_reinforced"]),
    )
    remap_closure_high = math.sqrt(object_unification_high * transport_unification_high)
    support_gap_high = 1.0 - remap_closure_high
    unification_high_stability = (
        object_unification_high + transport_unification_high + remap_closure_high
    ) / 3.0
    high_closure_gain = unification_high_stability - hr["unification_stability_reinforced"]

    return {
        "headline_metrics": {
            "object_unification_high": object_unification_high,
            "transport_unification_high": transport_unification_high,
            "remap_closure_high": remap_closure_high,
            "support_gap_high": support_gap_high,
            "unification_high_stability": unification_high_stability,
            "high_closure_gain": high_closure_gain,
        },
        "high_closure_equation": {
            "object_term": "U_object_high = U_object_plus + c_u * U_keep_star",
            "transport_term": "U_transport_high = U_transport_plus + c_t * K_keep_star * (1 - U_transport_plus)",
            "closure_term": "C_unify_high = sqrt(U_object_high * U_transport_high)",
            "gap_term": "G_unify_high = 1 - C_unify_high",
            "stability_term": "S_unify_high = mean(U_object_high, U_transport_high, C_unify_high)",
        },
        "project_readout": {
            "summary": "统一收口高闭合块开始把对象层和运输层同时推向高闭合区，不再只要求中等稳定，而是要求收口核和弱通道一起抬升。",
            "next_question": "下一步要验证高闭合后的统一核，是否能跨更多版本稳定保持，而不是在 v36 附近局部最优。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 统一收口高闭合报告",
        "",
        f"- object_unification_high: {hm['object_unification_high']:.6f}",
        f"- transport_unification_high: {hm['transport_unification_high']:.6f}",
        f"- remap_closure_high: {hm['remap_closure_high']:.6f}",
        f"- support_gap_high: {hm['support_gap_high']:.6f}",
        f"- unification_high_stability: {hm['unification_high_stability']:.6f}",
        f"- high_closure_gain: {hm['high_closure_gain']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_icspb_unification_high_closure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
