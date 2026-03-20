from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_icspb_unification_reinforcement_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_icspb_unification_reinforcement_summary() -> dict:
    unify = _load_json(ROOT / "tests" / "codex_temp" / "stage56_icspb_unification_closure_20260320" / "summary.json")
    retention_plus = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_kernel_retention_reinforcement_20260320" / "summary.json"
    )

    hu = unify["headline_metrics"]
    hr = retention_plus["headline_metrics"]

    object_unification_reinforced = min(
        1.0,
        hu["object_unification_strength"] + 0.35 * hr["update_retention_reinforced"] * (1.0 - hu["object_unification_strength"]),
    )
    transport_unification_reinforced = min(
        1.0,
        hu["transport_unification_strength"]
        + 0.45 * hr["transport_kernel_stability_reinforced"] * (1.0 - hu["transport_unification_strength"]),
    )
    remap_closure_reinforced = math.sqrt(object_unification_reinforced * transport_unification_reinforced)
    support_gap_reinforced = 1.0 - remap_closure_reinforced
    unification_stability_reinforced = (
        object_unification_reinforced + transport_unification_reinforced + remap_closure_reinforced
    ) / 3.0
    unification_gain = unification_stability_reinforced - hu["closure_stability"]

    return {
        "headline_metrics": {
            "object_unification_reinforced": object_unification_reinforced,
            "transport_unification_reinforced": transport_unification_reinforced,
            "remap_closure_reinforced": remap_closure_reinforced,
            "support_gap_reinforced": support_gap_reinforced,
            "unification_stability_reinforced": unification_stability_reinforced,
            "unification_gain": unification_gain,
        },
        "reinforcement_equation": {
            "object_term": "U_object_plus = U_object + a * U_keep_plus * (1 - U_object)",
            "transport_term": "U_transport_plus = U_transport + b * K_keep_plus * (1 - U_transport)",
            "closure_term": "C_unify_plus = sqrt(U_object_plus * U_transport_plus)",
            "gap_term": "G_unify_plus = 1 - C_unify_plus",
            "stability_term": "S_unify_plus = mean(U_object_plus, U_transport_plus, C_unify_plus)",
        },
        "project_readout": {
            "summary": "统一收口强化块把旧框架对象层和运输层同时抬高，不再只要求能映射，而是要求统一之后还能维持更高的收口稳定度。",
            "next_question": "下一步要验证强化后的统一收口，是否能在新主核里长期保持，而不是随版本推进再次回落。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 统一收口强化报告",
        "",
        f"- object_unification_reinforced: {hm['object_unification_reinforced']:.6f}",
        f"- transport_unification_reinforced: {hm['transport_unification_reinforced']:.6f}",
        f"- remap_closure_reinforced: {hm['remap_closure_reinforced']:.6f}",
        f"- support_gap_reinforced: {hm['support_gap_reinforced']:.6f}",
        f"- unification_stability_reinforced: {hm['unification_stability_reinforced']:.6f}",
        f"- unification_gain: {hm['unification_gain']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_icspb_unification_reinforcement_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
