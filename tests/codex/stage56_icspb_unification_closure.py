from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_icspb_unification_closure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_icspb_unification_closure_summary() -> dict:
    remap = _load_json(ROOT / "tests" / "codex_temp" / "stage56_icspb_object_remapping_20260320" / "summary.json")
    transport = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_bridge_reintegration_20260320" / "summary.json"
    )

    hr = remap["headline_metrics"]
    ht = transport["headline_metrics"]

    object_unification_strength = (
        hr["family_patch_alignment"] + hr["concept_offset_alignment"] + hr["attribute_fiber_alignment"]
    ) / 3.0
    transport_unification_strength = (
        ht["protocol_bridge_strength"]
        + ht["stage_transport_strength"]
        + ht["successor_alignment_strength"]
    ) / 3.0
    remap_closure_core = math.sqrt(hr["remap_consistency"] * transport_unification_strength)
    support_gap_reduced = 1.0 - remap_closure_core
    closure_stability = (object_unification_strength + transport_unification_strength + remap_closure_core) / 3.0

    return {
        "headline_metrics": {
            "object_unification_strength": object_unification_strength,
            "transport_unification_strength": transport_unification_strength,
            "remap_closure_core": remap_closure_core,
            "support_gap_reduced": support_gap_reduced,
            "closure_stability": closure_stability,
        },
        "closure_equation": {
            "object_term": "U_object = mean(A_family, A_offset, A_fiber)",
            "transport_term": "U_transport = mean(G_bridge, G_stage, G_successor)",
            "closure_term": "C_unify = sqrt(C_remap * U_transport)",
            "gap_term": "G_unify = 1 - C_unify",
            "stability_term": "S_unify = mean(U_object, U_transport, C_unify)",
        },
        "project_readout": {
            "summary": "旧版 ICSPB 对象和当前形成机制主线已经不只是方向对齐，而是开始进入统一收口。当前最关键的是对象层统一强度、运输层统一强度和总闭合核三者开始同时变成可量化对象。",
            "next_question": "下一步要检验这种统一收口是否能在主核里长期稳定保留，而不是只在一次回并时成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 旧框架与新框架统一收口报告",
        "",
        f"- object_unification_strength: {hm['object_unification_strength']:.6f}",
        f"- transport_unification_strength: {hm['transport_unification_strength']:.6f}",
        f"- remap_closure_core: {hm['remap_closure_core']:.6f}",
        f"- support_gap_reduced: {hm['support_gap_reduced']:.6f}",
        f"- closure_stability: {hm['closure_stability']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_icspb_unification_closure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
