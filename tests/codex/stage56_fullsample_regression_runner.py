from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def pair_key(row: Dict[str, object]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("model_id", "")),
        str(row.get("category", "")),
        str(row.get("prototype_term", "")),
        str(row.get("instance_term", "")),
    )


def build_design_rows(
    pair_density_rows: Sequence[Dict[str, object]],
    complete_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    density_group: Dict[Tuple[str, str, str, str], Dict[str, Dict[str, object]]] = {}
    for row in pair_density_rows:
        key = pair_key(row)
        if isinstance(row.get("axes"), dict):
            axes_block = density_group.setdefault(key, {})
            for axis_name, axis_payload in dict(row.get("axes", {})).items():
                if isinstance(axis_payload, dict):
                    axes_block[str(axis_name)] = dict(axis_payload)
        else:
            density_group.setdefault(key, {})[str(row.get("axis", ""))] = dict(row)

    component_group: Dict[Tuple[str, str, str, str], Dict[str, Dict[str, object]]] = {}
    for row in complete_rows:
        component_group.setdefault(pair_key(row), {})[str(row.get("component_label", ""))] = dict(row)

    design_rows: List[Dict[str, object]] = []
    all_keys = sorted(set(density_group) | set(component_group))
    for key in all_keys:
        axis_block = density_group.get(key, {})
        component_block = component_group.get(key, {})
        any_complete = next(iter(component_block.values()), {})
        if not any_complete:
            continue

        def axis_value(axis_name: str, feature_name: str) -> float:
            return safe_float(dict(axis_block.get(axis_name, {})).get(feature_name))

        def component_value(component_name: str, feature_name: str) -> float:
            return safe_float(dict(component_block.get(component_name, {})).get(feature_name))

        atlas_static_proxy = mean(
            [
                axis_value("style", "role_alignment_compaction"),
                axis_value("logic", "role_alignment_compaction"),
                axis_value("syntax", "role_alignment_compaction"),
                axis_value("style", "role_alignment_coverage"),
                axis_value("logic", "role_alignment_coverage"),
                axis_value("syntax", "role_alignment_coverage"),
            ]
        )
        offset_static_proxy = mean(
            [
                axis_value("style", "role_asymmetry_compaction_l1"),
                axis_value("logic", "role_asymmetry_compaction_l1"),
                axis_value("syntax", "role_asymmetry_compaction_l1"),
                axis_value("style", "role_asymmetry_coverage_l1"),
                axis_value("logic", "role_asymmetry_coverage_l1"),
                axis_value("syntax", "role_asymmetry_coverage_l1"),
            ]
        )
        frontier_dynamic_proxy = mean(
            [
                axis_value("style", "pair_compaction_middle_mean"),
                axis_value("logic", "pair_compaction_middle_mean"),
                axis_value("syntax", "pair_compaction_middle_mean"),
                axis_value("style", "pair_coverage_middle_mean"),
                axis_value("logic", "pair_coverage_middle_mean"),
                axis_value("syntax", "pair_coverage_middle_mean"),
            ]
        )
        design_rows.append(
            {
                "model_id": key[0],
                "category": key[1],
                "prototype_term": key[2],
                "instance_term": key[3],
                "atlas_static_proxy": atlas_static_proxy,
                "offset_static_proxy": offset_static_proxy,
                "frontier_dynamic_proxy": frontier_dynamic_proxy,
                "logic_prototype_proxy": component_value("logic_prototype", "weight"),
                "logic_fragile_bridge_proxy": component_value("logic_fragile_bridge", "weight"),
                "syntax_constraint_conflict_proxy": component_value("syntax_constraint_conflict", "weight"),
                "window_hidden_proxy": mean(
                    [
                        component_value("logic_prototype", "hidden_window_center"),
                        component_value("logic_fragile_bridge", "hidden_window_center"),
                        component_value("syntax_constraint_conflict", "hidden_window_center"),
                    ]
                ),
                "window_mlp_proxy": mean(
                    [
                        component_value("logic_prototype", "mlp_window_center"),
                        component_value("logic_fragile_bridge", "mlp_window_center"),
                        component_value("syntax_constraint_conflict", "mlp_window_center"),
                    ]
                ),
                "style_control_proxy": axis_value("style", "pair_compaction_middle_mean"),
                "logic_control_proxy": axis_value("logic", "pair_compaction_middle_mean"),
                "syntax_control_proxy": axis_value("syntax", "pair_compaction_middle_mean"),
                "union_joint_adv": safe_float(any_complete.get("union_joint_adv")),
                "union_synergy_joint": safe_float(any_complete.get("union_synergy_joint")),
                "strict_positive_synergy": 1.0 if bool(any_complete.get("strict_positive_synergy")) else 0.0,
            }
        )
    return design_rows


def transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(col) for col in zip(*matrix)] if matrix else []


def matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    if not a or not b:
        return []
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    out = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            aik = safe_float(a[i][k])
            for j in range(cols):
                out[i][j] += aik * safe_float(b[k][j])
    return out


def inverse(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    n = len(matrix)
    aug = [list(row) + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = col
        while pivot < n and abs(aug[pivot][col]) < 1e-12:
            pivot += 1
        if pivot == n:
            raise ValueError("matrix is singular")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        aug[col] = [value / pivot_value for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [value - factor * aug[col][idx] for idx, value in enumerate(aug[row])]
    return [row[n:] for row in aug]


def fit_linear_regression(rows: Sequence[Dict[str, object]], feature_names: Sequence[str], target_name: str) -> Dict[str, object]:
    if not rows:
        return {"target_name": target_name, "weights": {}, "row_count": 0}
    x_matrix: List[List[float]] = []
    y_matrix: List[List[float]] = []
    for row in rows:
        x_matrix.append([1.0] + [safe_float(row.get(feature)) for feature in feature_names])
        y_matrix.append([safe_float(row.get(target_name))])
    xt = transpose(x_matrix)
    xtx = matmul(xt, x_matrix)
    ridge = 1e-6
    for i in range(len(xtx)):
        xtx[i][i] += ridge
    xty = matmul(xt, y_matrix)
    beta = matmul(inverse(xtx), xty)
    labels = ["intercept"] + list(feature_names)
    weights = {label: safe_float(beta[idx][0]) for idx, label in enumerate(labels)}
    return {
        "target_name": target_name,
        "weights": weights,
        "row_count": len(rows),
    }


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    feature_names = [
        "atlas_static_proxy",
        "offset_static_proxy",
        "frontier_dynamic_proxy",
        "logic_prototype_proxy",
        "logic_fragile_bridge_proxy",
        "syntax_constraint_conflict_proxy",
        "window_hidden_proxy",
        "window_mlp_proxy",
        "style_control_proxy",
        "logic_control_proxy",
        "syntax_control_proxy",
    ]
    fits = [
        fit_linear_regression(rows, feature_names, "union_joint_adv"),
        fit_linear_regression(rows, feature_names, "union_synergy_joint"),
        fit_linear_regression(rows, feature_names, "strict_positive_synergy"),
    ]
    return {
        "record_type": "stage56_fullsample_regression_runner_summary",
        "row_count": len(rows),
        "feature_names": feature_names,
        "fits": fits,
        "main_judgment": (
            "当前第一版全样本回归器已经具备样本级设计矩阵和最小线性回归能力。"
            "后续只要恢复实跑环境，就可以把现有摘要层系统推进到真正的样本级拟合。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 全样本回归落地摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        f"- row_count: {summary.get('row_count', 0)}",
        "",
        "## Fits",
    ]
    for row in list(summary.get("fits", [])):
        row = dict(row)
        lines.append(f"- target: {row.get('target_name', '')}")
        for key, value in dict(row.get("weights", {})).items():
            lines.append(f"  {key}: {safe_float(value):+.6f}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the first sample-level regression landing for the unified master equation")
    ap.add_argument(
        "--pair-density-jsonl",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_pair_density_tensor_field_20260319_1512" / "joined_rows.jsonl"),
    )
    ap.add_argument(
        "--complete-joined-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_complete_highdim_field_20260319_1645" / "joined_rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_fullsample_regression_runner_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pair_density_rows = read_jsonl(Path(args.pair_density_jsonl))
    complete_rows = list(read_json(Path(args.complete_joined_json)).get("rows", []))
    design_rows = build_design_rows(pair_density_rows, complete_rows)
    summary = build_summary(design_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "design_rows.json").write_text(json.dumps({"rows": design_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(design_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
