from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage103_native_brain_anchor_search_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage92_brain_grounding_counterexample_pack import build_brain_grounding_counterexample_pack_summary
from stage101_brain_evidence_joint_closure import build_brain_evidence_joint_closure_summary
from stage102_real_world_falsification_bridge import build_real_world_falsification_bridge_summary


PRIMARY_PROBE = ROOT / "tempdata" / "deepseek7b_multidim_encoding_probe_20260305_220444" / "multidim_encoding_probe.json"
MULTISEED_ROOT = ROOT / "tempdata" / "deepseek7b_multidim_multiseed_v1"
DIMENSIONS = ["style", "logic", "syntax"]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def build_native_brain_anchor_search_summary() -> dict:
    primary = _load_json(PRIMARY_PROBE)
    multiseed_paths = sorted(MULTISEED_ROOT.glob("seed*/probe/multidim_encoding_probe.json"))
    multiseed = [_load_json(path) for path in multiseed_paths]

    brain = build_brain_grounding_counterexample_pack_summary()["headline_metrics"]
    joint = build_brain_evidence_joint_closure_summary()["headline_metrics"]
    bridge = build_real_world_falsification_bridge_summary()["headline_metrics"]

    generic_count = Counter()
    generic_seed_sets: dict[tuple[int, int], set[str]] = defaultdict(set)
    generic_dim_sets: dict[tuple[int, int], set[str]] = defaultdict(set)
    generic_strengths: dict[tuple[int, int], list[float]] = defaultdict(list)

    specific_count = Counter()
    specific_seed_sets: dict[tuple[int, int], set[str]] = defaultdict(set)
    specific_dim_sets: dict[tuple[int, int], set[str]] = defaultdict(set)
    specific_scores: dict[tuple[int, int], list[float]] = defaultdict(list)

    dominant_layers_by_dim: dict[str, list[int]] = {dimension: [] for dimension in DIMENSIONS}

    for path, probe in zip(multiseed_paths, multiseed):
        seed_name = path.parent.parent.name
        for dimension in DIMENSIONS:
            dim = probe["dimensions"][dimension]
            layer_profile = dim["layer_profile_abs_delta_norm"]
            dominant_layer = max(range(len(layer_profile)), key=lambda idx: layer_profile[idx])
            dominant_layers_by_dim[dimension].append(dominant_layer)

            for item in dim["generic_top_neurons"][:10]:
                key = (item["layer"], item["neuron"])
                generic_count[key] += 1
                generic_seed_sets[key].add(seed_name)
                generic_dim_sets[key].add(dimension)
                generic_strengths[key].append(item["mean_abs_delta"])

            for item in dim["specific_top_neurons"][:10]:
                key = (item["layer"], item["neuron"])
                specific_count[key] += 1
                specific_seed_sets[key].add(seed_name)
                specific_dim_sets[key].add(dimension)
                specific_scores[key].append(item["specific_score"])

    shared_anchor_candidates = []
    for key, count in generic_count.items():
        dims = generic_dim_sets[key]
        if len(dims) < 2:
            continue
        layer, neuron = key
        seed_recurrence = len(generic_seed_sets[key]) / max(1, len(multiseed))
        mean_strength = statistics.mean(generic_strengths[key])
        candidate_strength = _clip01(
            0.42 * seed_recurrence
            + 0.26 * (count / (len(multiseed) * 2.0))
            + 0.18 * _clip01(mean_strength / 20.0)
            + 0.14 * (len(dims) / 3.0)
        )
        shared_anchor_candidates.append(
            {
                "layer": layer,
                "neuron": neuron,
                "seed_count": len(generic_seed_sets[key]),
                "dimension_names": sorted(dims),
                "mean_abs_delta": mean_strength,
                "candidate_strength": candidate_strength,
            }
        )
    shared_anchor_candidates.sort(key=lambda item: (-item["candidate_strength"], -item["seed_count"], item["layer"], item["neuron"]))

    specific_anchor_candidates = []
    for key, count in specific_count.items():
        dims = specific_dim_sets[key]
        if len(dims) != 1:
            continue
        layer, neuron = key
        dimension = next(iter(dims))
        seed_recurrence = len(specific_seed_sets[key]) / max(1, len(multiseed))
        mean_specific_score = statistics.mean(specific_scores[key])
        candidate_strength = _clip01(
            0.44 * seed_recurrence
            + 0.28 * (count / len(multiseed))
            + 0.28 * _clip01(mean_specific_score / 10.0)
        )
        specific_anchor_candidates.append(
            {
                "dimension": dimension,
                "layer": layer,
                "neuron": neuron,
                "seed_count": len(specific_seed_sets[key]),
                "mean_specific_score": mean_specific_score,
                "candidate_strength": candidate_strength,
            }
        )
    specific_anchor_candidates.sort(
        key=lambda item: (-item["candidate_strength"], item["dimension"], -item["seed_count"], item["layer"], item["neuron"])
    )

    dimension_specific_best = {}
    for dimension in DIMENSIONS:
        by_dim = [item for item in specific_anchor_candidates if item["dimension"] == dimension]
        if by_dim:
            dimension_specific_best[dimension] = by_dim[0]

    generic_seed_recurrence_strength = _clip01(
        statistics.mean(item["seed_count"] / len(multiseed) for item in shared_anchor_candidates[:5])
    ) if shared_anchor_candidates else 0.0
    dimension_specific_anchor_strength = _clip01(
        statistics.mean(item["candidate_strength"] for item in dimension_specific_best.values())
    ) if dimension_specific_best else 0.0

    layer_stability_by_dim = {}
    for dimension, layer_list in dominant_layers_by_dim.items():
        mode_layer, mode_count = Counter(layer_list).most_common(1)[0]
        layer_stability_by_dim[dimension] = {
            "mode_layer": mode_layer,
            "stability": mode_count / len(layer_list),
        }
    layer_anchor_stability = statistics.mean(item["stability"] for item in layer_stability_by_dim.values())

    anchor_ambiguity_penalty = _clip01(
        statistics.mean(len(item["dimension_names"]) / 3.0 for item in shared_anchor_candidates[:5])
    ) if shared_anchor_candidates else 0.0
    closure_bridge_support = _clip01(
        0.28 * joint["brain_evidence_joint_closure_score"]
        + 0.24 * (1.0 - joint["brain_evidence_joint_closure_gap"])
        + 0.24 * bridge["real_world_falsification_bridge_score"]
        + 0.24 * bridge["multiseed_probe_stability"]
    )

    native_brain_anchor_search_score = _clip01(
        0.24 * generic_seed_recurrence_strength
        + 0.24 * dimension_specific_anchor_strength
        + 0.18 * layer_anchor_stability
        + 0.16 * closure_bridge_support
        + 0.10 * brain["brain_grounding_residual"]
        + 0.08 * (1.0 - anchor_ambiguity_penalty)
    )

    weakness_map = {
        "generic_recurrence_gap": 1.0 - generic_seed_recurrence_strength,
        "specific_anchor_gap": 1.0 - dimension_specific_anchor_strength,
        "layer_stability_gap": 1.0 - layer_anchor_stability,
        "anchor_ambiguity_gap": anchor_ambiguity_penalty,
        "closure_bridge_gap": 1.0 - closure_bridge_support,
    }
    weakest_anchor_mode_name, weakest_anchor_mode_pressure = max(weakness_map.items(), key=lambda item: item[1])

    primary_anchor_preview = {}
    for dimension in DIMENSIONS:
        top = primary["dimensions"][dimension]["top_neurons"][:5]
        primary_anchor_preview[dimension] = [
            {
                "layer": item["layer"],
                "neuron": item["neuron"],
                "mean_abs_delta": item["mean_abs_delta"],
            }
            for item in top
        ]

    return {
        "headline_metrics": {
            "generic_seed_recurrence_strength": generic_seed_recurrence_strength,
            "dimension_specific_anchor_strength": dimension_specific_anchor_strength,
            "layer_anchor_stability": layer_anchor_stability,
            "anchor_ambiguity_penalty": anchor_ambiguity_penalty,
            "closure_bridge_support": closure_bridge_support,
            "weakest_anchor_mode_name": weakest_anchor_mode_name,
            "weakest_anchor_mode_pressure": weakest_anchor_mode_pressure,
            "native_brain_anchor_search_score": native_brain_anchor_search_score,
        },
        "layer_stability_by_dim": layer_stability_by_dim,
        "shared_anchor_candidates": shared_anchor_candidates[:12],
        "specific_anchor_candidates": specific_anchor_candidates[:12],
        "primary_anchor_preview": primary_anchor_preview,
        "status": {
            "status_short": (
                "native_brain_anchor_search_ready"
                if native_brain_anchor_search_score >= 0.58
                and dimension_specific_anchor_strength >= 0.55
                else "native_brain_anchor_search_transition"
            ),
            "status_label": "原生脑锚点搜索已经找到跨随机种子重复出现的候选，但当前仍主要停留在候选层，还没有到原生定理层。",
        },
        "project_readout": {
            "summary": "这一轮不再只说脑编码弱链在哪里，而是开始明确寻找哪些层、哪些神经元在多维度和多随机种子下反复出现，适合作为原生脑锚点候选。",
            "next_question": "下一步要把这些候选锚点接到真实任务失败案例和真实激活轨迹上，确认它们不是探针专用产物。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage103 Native Brain Anchor Search",
        "",
        f"- generic_seed_recurrence_strength: {hm['generic_seed_recurrence_strength']:.6f}",
        f"- dimension_specific_anchor_strength: {hm['dimension_specific_anchor_strength']:.6f}",
        f"- layer_anchor_stability: {hm['layer_anchor_stability']:.6f}",
        f"- anchor_ambiguity_penalty: {hm['anchor_ambiguity_penalty']:.6f}",
        f"- closure_bridge_support: {hm['closure_bridge_support']:.6f}",
        f"- weakest_anchor_mode_name: {hm['weakest_anchor_mode_name']}",
        f"- weakest_anchor_mode_pressure: {hm['weakest_anchor_mode_pressure']:.6f}",
        f"- native_brain_anchor_search_score: {hm['native_brain_anchor_search_score']:.6f}",
        f"- status_short: {summary['status']['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_native_brain_anchor_search_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
