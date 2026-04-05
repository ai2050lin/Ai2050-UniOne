import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "research" / "gpt5" / "data" / "visualization" / "apple_switch_mechanism_view_v1.json"

STAGE438_PATH = ROOT / "tests" / "codex_temp" / "stage438_apple_neuron_role_3d_view_20260402" / "summary.json"
STAGE448_QWEN_PATH = ROOT / "tests" / "codex_temp" / "stage448_apple_switch_layer_scan_and_neuron_counts_20260403" / "qwen3_cpu" / "summary.json"
STAGE448_DEEPSEEK_PATH = ROOT / "tests" / "codex_temp" / "stage448_apple_switch_layer_scan_and_neuron_counts_20260403" / "deepseek7b_cpu" / "summary.json"
STAGE479_PATH = ROOT / "tests" / "codex_temp" / "stage479_apple_switch_mixed_circuit_search_20260403" / "summary.json"
STAGE482_PATH = ROOT / "tests" / "codex_temp" / "stage482_apple_switch_direction_tracking_20260403" / "summary.json"
STAGE484_PATH = ROOT / "tests" / "codex_temp" / "stage484_apple_switch_signed_residual_basis_20260403" / "summary.json"
STAGE485_PATH = ROOT / "tests" / "codex_temp" / "stage485_cpu_gpu_consistency_compare_20260403" / "summary.json"


def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_concept_key(value: str) -> str:
    return str(value or "").strip().lower()


def scene_layer_index(layer_index: int, layer_count: int) -> int:
    if layer_count <= 1:
        return 0
    return int(round((float(layer_index) / float(layer_count - 1)) * 27.0))


def safe_role_score(role: str) -> float:
    role_map = {
        "anchor_neuron": 1.0,
        "main_booster_1": 0.92,
        "main_booster_2": 0.9,
        "skeleton_head_1": 0.92,
        "skeleton_head_2": 0.9,
        "bridge_head": 0.82,
        "heldout_booster": 0.56,
    }
    return role_map.get(role, 0.68)


def build_single_score_map(single_scores):
    return {row["candidate_id"]: row for row in (single_scores or [])}


def build_unit_map(units):
    return {row["unit_id"]: row for row in (units or [])}


def build_stability_map(unit_rows):
    return {row["unit_id"]: row for row in (unit_rows or [])}


def max_or_one(values):
    numeric = [abs(float(v)) for v in values if v is not None]
    return max(numeric) if numeric else 1.0


def find_stage438_model(summary_438, model_key: str):
    for row in summary_438.get("model_results", []):
        if row.get("model_key") == model_key:
            return row
    raise KeyError(f"stage438 missing model: {model_key}")


def find_stage448_model(summary_448):
    rows = summary_448.get("model_results", [])
    if not rows:
        raise KeyError("stage448 missing model_results")
    return rows[0]


def make_layer_summary(stage448_model):
    scan_by_layer = {
        row["layer_index"]: row
        for row in stage448_model.get("layer_scan", [])
    }
    count_by_layer = {
        row["layer_index"]: row
        for row in stage448_model.get("per_layer_counts", [])
    }
    all_layers = sorted(set(scan_by_layer) | set(count_by_layer))
    rows = []
    for layer_index in all_layers:
        scan = scan_by_layer.get(layer_index, {})
        count = count_by_layer.get(layer_index, {})
        rows.append(
            {
                "layer_index": layer_index,
                "switch_prob_drop": scan.get("switch_prob_drop", 0.0),
                "control_prob_drop": scan.get("control_prob_drop", 0.0),
                "excess_switch_drop": scan.get("excess_switch_drop", 0.0),
                "shared_active_neuron_count": count.get("shared_active_neuron_count", 0),
                "fruit_only_neuron_count": count.get("fruit_only_neuron_count", 0),
                "brand_only_neuron_count": count.get("brand_only_neuron_count", 0),
                "active_jaccard": count.get("active_jaccard", 0.0),
                "process_label": make_process_label(scan, count),
            }
        )
    return rows


def make_process_label(scan_row, count_row):
    excess = float(scan_row.get("excess_switch_drop", 0.0) or 0.0)
    jaccard = float(count_row.get("active_jaccard", 0.0) or 0.0)
    if jaccard >= 0.12:
        return "共享底座层"
    if excess >= 0.02:
        return "强切换敏感层"
    if excess >= 0.008:
        return "中等切换传播层"
    if excess <= -0.002:
        return "反向校正或回填层"
    return "弱传播或过渡层"


def build_core_units(model_key, stage479_model, stage482_model, stage484_model, stage485_model, actual_layer_count):
    single_score_map = build_single_score_map(stage479_model.get("single_scores"))
    unit_482_map = build_unit_map(stage482_model.get("units"))
    stability_map = build_stability_map(stage485_model.get("unit_rows"))
    units_484 = stage484_model.get("units", [])

    support_strength_max = max_or_one(
        [
            abs(min(0.0, float(row["tracking"].get("late_mean_signed_contrast_switch_coupling", 0.0) or 0.0)))
            for row in units_484
        ]
    )
    causal_max = max_or_one(
        [
            float(single_score_map.get(row["unit_id"], {}).get("effect", {}).get("utility", 0.0) or 0.0)
            for row in units_484
        ]
    )
    stability_diff_max = max_or_one(
        [
            float(stability_map.get(row["unit_id"], {}).get("late_mean_signed_contrast_switch_coupling_abs_diff", 0.0) or 0.0)
            for row in units_484
        ]
    )

    final_subset_ids = set(stage479_model.get("final_effect", {}).get("subset_ids", []))
    if not final_subset_ids:
        final_subset_ids = {row.get("candidate_id") for row in stage479_model.get("final_subset", [])}
    final_subset_ids.discard(None)

    core_units = []
    for row in units_484:
        unit_id = row["unit_id"]
        tracking = row.get("tracking", {})
        late_mean = float(tracking.get("late_mean_signed_contrast_switch_coupling", 0.0) or 0.0)
        single_row = single_score_map.get(unit_id, {})
        effect = single_row.get("effect", {})
        utility = float(effect.get("utility", 0.0) or 0.0)
        stability_row = stability_map.get(unit_id, {})
        peak_match = (
            stability_row.get("forward_peak_layer_cpu") == stability_row.get("forward_peak_layer_gpu")
            and stability_row.get("reverse_peak_layer_cpu") == stability_row.get("reverse_peak_layer_gpu")
        )
        stability_score = clamp01(
            1.0 - (float(stability_row.get("late_mean_signed_contrast_switch_coupling_abs_diff", 0.0) or 0.0) / stability_diff_max)
        )
        if peak_match:
            stability_score = clamp01(stability_score * 0.7 + 0.3)

        signed_process_score = clamp01(abs(min(0.0, late_mean)) / support_strength_max)
        causal_score = clamp01(utility / causal_max)
        role_score = safe_role_score(str(row.get("role", "")))
        effective_score = (
            0.20 * role_score
            + 0.35 * causal_score
            + 0.35 * signed_process_score
            + 0.10 * stability_score
        )

        direction_label = "正向支撑"
        if late_mean > 0:
            direction_label = "反向校正"
        elif abs(late_mean) < 1e-9:
            direction_label = "中性或未定"

        track_482 = unit_482_map.get(unit_id, {}).get("tracking", {})
        layer_rows = []
        rows_484 = tracking.get("layer_rows", [])
        rows_482 = {
            int(item["layer_index"]): item
            for item in track_482.get("layer_rows", [])
        }
        for item in rows_484:
            idx = int(item["layer_index"])
            direction_value = float(item.get("signed_contrast_switch_coupling", 0.0) or 0.0)
            layer_rows.append(
                {
                    "layer_index": idx,
                    "scene_layer_index": scene_layer_index(idx, actual_layer_count),
                    "signed_contrast_switch_coupling": direction_value,
                    "signed_pc1_switch_coupling": float(item.get("signed_pc1_switch_coupling", 0.0) or 0.0),
                    "pc1_explained_variance_ratio": float(item.get("pc1_explained_variance_ratio", 0.0) or 0.0),
                    "relative_separation_drop": float(rows_482.get(idx, {}).get("relative_drop", 0.0) or 0.0),
                    "direction_label": (
                        "正向推进削弱"
                        if direction_value < -1e-9
                        else "反向推进增强"
                        if direction_value > 1e-9
                        else "中性"
                    ),
                }
            )

        core_units.append(
            {
                "unit_id": unit_id,
                "model_key": model_key,
                "kind": row.get("kind"),
                "role": row.get("role"),
                "display_name": unit_id,
                "actual_layer_index": int(row.get("layer_index", 0) or 0),
                "scene_layer_index": scene_layer_index(int(row.get("layer_index", 0) or 0), actual_layer_count),
                "head_index": row.get("head_index"),
                "neuron_index": row.get("neuron_index"),
                "is_final_circuit_member": unit_id in final_subset_ids,
                "causal_effect": {
                    "search_drop": float(effect.get("search_drop", 0.0) or 0.0),
                    "heldout_drop": float(effect.get("heldout_drop", 0.0) or 0.0),
                    "control_abs_shift": float(effect.get("control_abs_shift", 0.0) or 0.0),
                    "utility": utility,
                },
                "signed_effect": {
                    "late_mean_signed_contrast_switch_coupling": late_mean,
                    "late_mean_signed_pc1_switch_coupling": float(tracking.get("late_mean_signed_pc1_switch_coupling", 0.0) or 0.0),
                    "forward_peak_layer": tracking.get("forward_peak_layer"),
                    "reverse_peak_layer": tracking.get("reverse_peak_layer"),
                    "forward_peak_signed_contrast_switch_coupling": float(tracking.get("forward_peak_signed_contrast_switch_coupling", 0.0) or 0.0),
                    "reverse_peak_signed_contrast_switch_coupling": float(tracking.get("reverse_peak_signed_contrast_switch_coupling", 0.0) or 0.0),
                    "direction_label": direction_label,
                },
                "stability": {
                    "late_mean_abs_diff": float(stability_row.get("late_mean_signed_contrast_switch_coupling_abs_diff", 0.0) or 0.0),
                    "forward_peak_abs_diff": float(stability_row.get("forward_peak_signed_contrast_switch_coupling_abs_diff", 0.0) or 0.0),
                    "reverse_peak_abs_diff": float(stability_row.get("reverse_peak_signed_contrast_switch_coupling_abs_diff", 0.0) or 0.0),
                    "peak_layer_match": peak_match,
                    "stability_score": stability_score,
                },
                "scores": {
                    "role_score": role_score,
                    "causal_score": causal_score,
                    "signed_process_score": signed_process_score,
                    "stability_score": stability_score,
                    "effective_score": effective_score,
                },
                "process_timeline": layer_rows,
            }
        )

    core_units.sort(key=lambda item: item["scores"]["effective_score"], reverse=True)
    return core_units


def make_model_payload(
    model_key,
    model_name,
    stage438_model,
    stage448_model,
    stage479_model,
    stage482_model,
    stage484_model,
    stage485_model,
):
    actual_layer_count = int(stage484_model.get("layer_count", stage448_model.get("layer_top_k", 28)) or 28)
    core_units = build_core_units(
        model_key=model_key,
        stage479_model=stage479_model,
        stage482_model=stage482_model,
        stage484_model=stage484_model,
        stage485_model=stage485_model,
        actual_layer_count=actual_layer_count,
    )
    return {
        "model_key": model_key,
        "model_name": model_name,
        "actual_layer_count": actual_layer_count,
        "scene_layer_count": 28,
        "best_sensitive_layer": stage448_model.get("best_sensitive_layer"),
        "best_shared_layer": stage448_model.get("best_shared_layer"),
        "best_split_layer": stage448_model.get("best_split_layer"),
        "global_counts": stage448_model.get("global_counts", {}),
        "role_summary": stage438_model.get("summary", {}),
        "layer_summary": make_layer_summary(stage448_model),
        "effective_circuit": {
            "final_subset": stage479_model.get("final_subset", []),
            "final_effect": stage479_model.get("final_effect", {}),
            "final_subset_kind_counts": stage479_model.get("final_subset_kind_counts", {}),
        },
        "core_units": core_units,
    }


def main():
    stage438 = read_json(STAGE438_PATH)
    stage448_qwen = read_json(STAGE448_QWEN_PATH)
    stage448_deepseek = read_json(STAGE448_DEEPSEEK_PATH)
    stage479 = read_json(STAGE479_PATH)
    stage482 = read_json(STAGE482_PATH)
    stage484 = read_json(STAGE484_PATH)
    stage485 = read_json(STAGE485_PATH)

    payload = {
        "schema_version": "apple_switch_mechanism_view.v1",
        "concept": "apple",
        "title": "苹果切换机制统一可视化资产",
        "generated_from": {
            "stage438": str(STAGE438_PATH),
            "stage448_qwen3": str(STAGE448_QWEN_PATH),
            "stage448_deepseek7b": str(STAGE448_DEEPSEEK_PATH),
            "stage479": str(STAGE479_PATH),
            "stage482": str(STAGE482_PATH),
            "stage484": str(STAGE484_PATH),
            "stage485": str(STAGE485_PATH),
        },
        "aggregate_stability": stage485.get("aggregate", {}),
        "models": {
            "qwen3": make_model_payload(
                model_key="qwen3",
                model_name=stage484["models"]["qwen3"]["model_name"],
                stage438_model=find_stage438_model(stage438, "qwen3"),
                stage448_model=find_stage448_model(stage448_qwen),
                stage479_model=stage479["models"]["qwen3"],
                stage482_model=stage482["models"]["qwen3"],
                stage484_model=stage484["models"]["qwen3"],
                stage485_model=stage485["models"]["qwen3"],
            ),
            "deepseek7b": make_model_payload(
                model_key="deepseek7b",
                model_name=stage484["models"]["deepseek7b"]["model_name"],
                stage438_model=find_stage438_model(stage438, "deepseek7b"),
                stage448_model=find_stage448_model(stage448_deepseek),
                stage479_model=stage479["models"]["deepseek7b"],
                stage482_model=stage482["models"]["deepseek7b"],
                stage484_model=stage484["models"]["deepseek7b"],
                stage485_model=stage485["models"]["deepseek7b"],
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    summary = {
        "output_path": str(OUTPUT_PATH),
        "models": {
            model_key: {
                "core_unit_count": len(model_payload["core_units"]),
                "best_sensitive_layer": model_payload["best_sensitive_layer"]["layer_index"],
                "effective_circuit_size": len(model_payload["effective_circuit"]["final_subset"]),
            }
            for model_key, model_payload in payload["models"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
