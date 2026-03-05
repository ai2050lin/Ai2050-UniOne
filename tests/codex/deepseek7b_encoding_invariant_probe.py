#!/usr/bin/env python
"""
Probe candidate parameter-level encoding invariants from existing triaxial outputs.

Inputs:
- triaxial_param_structure.json
- optional falsifiable report JSON

Outputs:
- encoding_invariant_probe.json
- ENCODING_INVARIANT_PROBE_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


AXES = ("micro_attr", "same_type", "super_type")


def _safe_mean(xs: Sequence[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def _safe_std(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return float(statistics.stdev(xs))


def _jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def _p95(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(float(x) for x in xs)
    idx = int(round((len(ys) - 1) * 0.95))
    idx = max(0, min(idx, len(ys) - 1))
    return float(ys[idx])


def _extract_axis_struct(axis_obj: Dict[str, object]) -> Dict[str, object]:
    subset = axis_obj.get("causal_subset") or []
    neurons = [int(x["flat_index"]) for x in subset if isinstance(x, dict) and "flat_index" in x]
    layers = [int(x["layer"]) for x in subset if isinstance(x, dict) and "layer" in x]

    gate_dims: Set[int] = set()
    up_dims: Set[int] = set()
    down_dims: Set[int] = set()
    gate_up_alignment: List[float] = []
    for ps in axis_obj.get("param_signatures") or []:
        if not isinstance(ps, dict):
            continue
        for row in ((ps.get("gate_dims") or {}).get("pos") or []):
            if isinstance(row, dict) and "dim" in row:
                gate_dims.add(int(row["dim"]))
        for row in ((ps.get("gate_dims") or {}).get("neg") or []):
            if isinstance(row, dict) and "dim" in row:
                gate_dims.add(int(row["dim"]))
        for row in ((ps.get("up_dims") or {}).get("pos") or []):
            if isinstance(row, dict) and "dim" in row:
                up_dims.add(int(row["dim"]))
        for row in ((ps.get("up_dims") or {}).get("neg") or []):
            if isinstance(row, dict) and "dim" in row:
                up_dims.add(int(row["dim"]))
        for row in ((ps.get("down_dims") or {}).get("pos") or []):
            if isinstance(row, dict) and "dim" in row:
                down_dims.add(int(row["dim"]))
        for row in ((ps.get("down_dims") or {}).get("neg") or []):
            if isinstance(row, dict) and "dim" in row:
                down_dims.add(int(row["dim"]))
        if "gate_up_alignment" in ps:
            gate_up_alignment.append(float(ps["gate_up_alignment"]))

    layer_counts = Counter(layers)
    peak_layer = max(layer_counts, key=layer_counts.get) if layer_counts else None
    peak_layer_count = int(layer_counts.get(peak_layer, 0)) if peak_layer is not None else 0
    layer_concentration = float(peak_layer_count / len(layers)) if layers else 0.0

    return {
        "neurons": neurons,
        "layers": layers,
        "layer_counts": dict(sorted(layer_counts.items())),
        "peak_layer": int(peak_layer) if peak_layer is not None else None,
        "layer_concentration": layer_concentration,
        "gate_dims": sorted(gate_dims),
        "up_dims": sorted(up_dims),
        "down_dims": sorted(down_dims),
        "gate_up_alignment_mean": _safe_mean(gate_up_alignment),
        "gate_up_alignment_p95": _p95(gate_up_alignment),
    }


def build_invariants(
    triaxial: Dict[str, object],
    falsifiable: Dict[str, object] | None,
) -> Dict[str, object]:
    concept_axes = triaxial.get("concept_axes") or {}
    comparisons = triaxial.get("comparisons") or {}
    group_shared = (comparisons.get("group_shared") or {}) if isinstance(comparisons, dict) else {}

    per_concept = {}
    within_concept_axis_gap = []
    layer_focus_by_axis: Dict[str, List[float]] = defaultdict(list)
    axis_gate_dim_size: Dict[str, List[int]] = defaultdict(list)
    axis_alignment: Dict[str, List[float]] = defaultdict(list)

    for concept, info in concept_axes.items():
        if not isinstance(info, dict):
            continue
        axes = info.get("axes") or {}
        parsed = {}
        for ax in AXES:
            axis_obj = axes.get(ax) or {}
            axv = _extract_axis_struct(axis_obj)
            parsed[ax] = axv
            layer_focus_by_axis[ax].append(float(axv["layer_concentration"]))
            axis_gate_dim_size[ax].append(len(axv["gate_dims"]))
            axis_alignment[ax].append(float(axv["gate_up_alignment_mean"]))

        for i in range(len(AXES)):
            for j in range(i + 1, len(AXES)):
                a1 = AXES[i]
                a2 = AXES[j]
                within_concept_axis_gap.append({
                    "concept": concept,
                    "pair": f"{a1}__{a2}",
                    "neuron_jaccard": _jaccard(parsed[a1]["neurons"], parsed[a2]["neurons"]),
                    "layer_jaccard": _jaccard(parsed[a1]["layers"], parsed[a2]["layers"]),
                    "gate_dim_jaccard": _jaccard(parsed[a1]["gate_dims"], parsed[a2]["gate_dims"]),
                    "down_dim_jaccard": _jaccard(parsed[a1]["down_dims"], parsed[a2]["down_dims"]),
                })
        per_concept[concept] = {
            "group": info.get("group", "unknown"),
            "axes": parsed,
        }

    axis_isolation = {}
    for pair in ["micro_attr__same_type", "micro_attr__super_type", "same_type__super_type"]:
        rows = [x for x in within_concept_axis_gap if x["pair"] == pair]
        axis_isolation[pair] = {
            "n": len(rows),
            "neuron_jaccard_mean": _safe_mean([float(x["neuron_jaccard"]) for x in rows]),
            "layer_jaccard_mean": _safe_mean([float(x["layer_jaccard"]) for x in rows]),
            "gate_dim_jaccard_mean": _safe_mean([float(x["gate_dim_jaccard"]) for x in rows]),
            "down_dim_jaccard_mean": _safe_mean([float(x["down_dim_jaccard"]) for x in rows]),
        }

    shared_dims_stats = {}
    for group, gobj in group_shared.items():
        if not isinstance(gobj, dict):
            continue
        row = {}
        for ax in AXES:
            dims = (((gobj.get(ax) or {}).get("common_gate_input_dims")) or [])
            row[ax] = {
                "count": len(dims),
                "dims": [int(x) for x in dims],
            }
        shared_dims_stats[group] = row

    group_super_overlap = {}
    groups = sorted(shared_dims_stats.keys())
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = groups[i]
            g2 = groups[j]
            d1 = shared_dims_stats[g1]["super_type"]["dims"]
            d2 = shared_dims_stats[g2]["super_type"]["dims"]
            group_super_overlap[f"{g1}__{g2}"] = {
                "jaccard": _jaccard(d1, d2),
                "overlap_dims": sorted(list(set(d1) & set(d2))),
            }

    invariants = []
    for pair, stat in axis_isolation.items():
        invariants.append({
            "name": f"axis_isolation_{pair}",
            "metric": "gate_dim_jaccard_mean",
            "value": float(stat["gate_dim_jaccard_mean"]),
            "criterion": "<= 0.08",
            "decision": "pass" if float(stat["gate_dim_jaccard_mean"]) <= 0.08 else "fail",
            "note": "轴间参数维重叠应低，支持分工编码。",
        })

    for group, stat in shared_dims_stats.items():
        micro_count = int(stat["micro_attr"]["count"])
        same_count = int(stat["same_type"]["count"])
        super_count = int(stat["super_type"]["count"])
        invariants.append({
            "name": f"group_super_shared_backbone_{group}",
            "metric": "super_type_count_minus_max_other",
            "value": float(super_count - max(micro_count, same_count)),
            "criterion": "> 0",
            "decision": "pass" if super_count > max(micro_count, same_count) else "fail",
            "note": "组内上级轴共享维应多于属性轴/同类轴。",
        })

    if falsifiable:
        metrics = falsifiable.get("metrics") or {}
        seq_z = (((metrics.get("causal_seq_z") or {}).get("mean")) or 0.0)
        overall = (((metrics.get("overall_score") or {}).get("mean")) or 0.0)
        invariants.append({
            "name": "global_causal_strength_ready",
            "metric": "causal_seq_z_mean",
            "value": float(seq_z),
            "criterion": ">= 1.96",
            "decision": "pass" if float(seq_z) >= 1.96 else "fail",
            "note": "序列因果强度是否达到强证据区间。",
        })
        invariants.append({
            "name": "global_mechanism_score_ready",
            "metric": "overall_score_mean",
            "value": float(overall),
            "criterion": ">= 0.42",
            "decision": "pass" if float(overall) >= 0.42 else "fail",
            "note": "机制证据总分是否跨过可用阈值。",
        })

    pass_count = sum(1 for x in invariants if x["decision"] == "pass")
    fail_count = sum(1 for x in invariants if x["decision"] == "fail")
    readiness = float(pass_count / max(1, len(invariants)))

    return {
        "summary": {
            "n_concepts": len(per_concept),
            "axis_isolation_pairs": len(within_concept_axis_gap),
            "invariant_pass": pass_count,
            "invariant_fail": fail_count,
            "readiness_score": readiness,
        },
        "layer_focus_by_axis": {
            ax: {
                "mean": _safe_mean(vals),
                "std": _safe_std(vals),
                "samples": [float(v) for v in vals],
            }
            for ax, vals in layer_focus_by_axis.items()
        },
        "gate_dim_size_by_axis": {
            ax: {
                "mean": _safe_mean([float(v) for v in vals]),
                "std": _safe_std([float(v) for v in vals]),
                "samples": [int(v) for v in vals],
            }
            for ax, vals in axis_gate_dim_size.items()
        },
        "gate_up_alignment_by_axis": {
            ax: {
                "mean": _safe_mean(vals),
                "std": _safe_std(vals),
                "samples": [float(v) for v in vals],
            }
            for ax, vals in axis_alignment.items()
        },
        "axis_isolation": axis_isolation,
        "group_shared_dims": shared_dims_stats,
        "group_super_overlap": group_super_overlap,
        "invariants": invariants,
        "per_concept": per_concept,
    }


def build_markdown(
    payload: Dict[str, object],
    triaxial_path: Path,
    falsifiable_path: Path | None,
) -> str:
    s = payload.get("summary") or {}
    lines = []
    lines.append("# 编码不变量探针报告")
    lines.append("")
    lines.append("## 输入")
    lines.append(f"- 三轴结构: `{triaxial_path.as_posix()}`")
    lines.append(f"- 可证伪报告: `{falsifiable_path.as_posix() if falsifiable_path else '未提供'}`")
    lines.append("")
    lines.append("## 总览")
    lines.append(f"- 概念数: `{s.get('n_concepts', 0)}`")
    lines.append(f"- 轴隔离比较数: `{s.get('axis_isolation_pairs', 0)}`")
    lines.append(f"- 不变量通过/失败: `{s.get('invariant_pass', 0)}` / `{s.get('invariant_fail', 0)}`")
    lines.append(f"- 阶段就绪度(readiness): `{float(s.get('readiness_score', 0.0)):.4f}`")
    lines.append("")

    lines.append("## 轴间隔离（越低越接近分工编码）")
    axis_iso = payload.get("axis_isolation") or {}
    for pair, row in axis_iso.items():
        lines.append(
            f"- `{pair}`: gate_dim_jaccard_mean={float(row.get('gate_dim_jaccard_mean', 0.0)):.4f}, "
            f"neuron_jaccard_mean={float(row.get('neuron_jaccard_mean', 0.0)):.4f}, "
            f"layer_jaccard_mean={float(row.get('layer_jaccard_mean', 0.0)):.4f}"
        )
    lines.append("")

    lines.append("## 组内共享骨架（按super_type）")
    shared = payload.get("group_shared_dims") or {}
    for group, row in shared.items():
        st = row.get("super_type") or {}
        ma = row.get("micro_attr") or {}
        sa = row.get("same_type") or {}
        lines.append(
            f"- `{group}`: super={int(st.get('count', 0))}, micro={int(ma.get('count', 0))}, same={int(sa.get('count', 0))}, "
            f"super_dims={st.get('dims', [])}"
        )
    lines.append("")

    lines.append("## 判定")
    for item in payload.get("invariants") or []:
        lines.append(
            f"- `{item.get('name')}`: {item.get('decision')} "
            f"(metric={item.get('metric')}, value={float(item.get('value', 0.0)):.4f}, criterion={item.get('criterion')})"
        )
    lines.append("")

    lines.append("## 解释")
    lines.append("- 该探针重点检查“参数维共享骨架 + 轴间低重叠”是否同时成立。")
    lines.append("- 若局部不变量成立但全局强证据未过阈值，意味着编码结构线索已出现，但统一原理仍需更强因果验证。")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--triaxial-json",
        default="tempdata/deepseek7b_triaxial_param_structure_v1/triaxial_param_structure.json",
    )
    parser.add_argument(
        "--falsifiable-json",
        default="tempdata/deepseek7b_stage_falsifiable_report_v3_bilingual.json",
    )
    parser.add_argument("--output-dir", default="tempdata/deepseek7b_encoding_invariant_probe_v1")
    args = parser.parse_args()

    triaxial_path = Path(args.triaxial_json)
    falsifiable_path = Path(args.falsifiable_json) if args.falsifiable_json else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    triaxial = json.loads(triaxial_path.read_text(encoding="utf-8"))
    falsifiable = None
    if falsifiable_path and falsifiable_path.exists():
        falsifiable = json.loads(falsifiable_path.read_text(encoding="utf-8"))

    payload = build_invariants(triaxial, falsifiable)
    payload["input"] = {
        "triaxial_json": triaxial_path.as_posix(),
        "falsifiable_json": falsifiable_path.as_posix() if falsifiable_path else "",
    }

    json_path = out_dir / "encoding_invariant_probe.json"
    md_path = out_dir / "ENCODING_INVARIANT_PROBE_REPORT.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(payload, triaxial_path, falsifiable_path), encoding="utf-8")

    print(json.dumps({"json": json_path.as_posix(), "markdown": md_path.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()

