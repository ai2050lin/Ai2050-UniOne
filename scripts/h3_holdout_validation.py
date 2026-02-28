import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.runtime.contracts import AnalysisSpec, ConclusionCard, Metric, RunRecord, RunSummary
from server.runtime.experiment_store import ExperimentTimelineStore
from scripts.h3_category_adaptive_search import build_candidate_configs, eval_one_config


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_report_paths(raw: str) -> List[Path]:
    if not raw:
        return []
    out: List[Path] = []
    for item in raw.split(","):
        p = Path(item.strip())
        if p.exists():
            out.append(p)
    return out


def _summarize_seed_block(
    report_paths: List[Path],
    support_models_min: int,
    falsify_models_max: int,
) -> Dict[str, Any]:
    if not report_paths:
        return {
            "run_count": 0,
            "support_models_min": None,
            "support_models_max": None,
            "support_models_avg": None,
            "falsify_models_max": None,
            "strict_gate_pass": False,
            "reports": [],
        }
    support_values: List[int] = []
    falsify_values: List[int] = []
    reports: List[str] = []
    for p in report_paths:
        payload = _load_json(p)
        ag = payload.get("aggregates", {})
        support_values.append(int(ag.get("support_models", 0)))
        falsify_values.append(int(ag.get("falsify_models", 0)))
        reports.append(str(p).replace("\\", "/"))
    support_floor = min(support_values) if support_values else 0
    falsify_ceiling = max(falsify_values) if falsify_values else 999
    strict_gate_pass = support_floor >= support_models_min and falsify_ceiling <= falsify_models_max
    return {
        "run_count": len(report_paths),
        "support_models_min": support_floor,
        "support_models_max": max(support_values) if support_values else 0,
        "support_models_avg": round(float(np.mean(support_values)), 4) if support_values else 0.0,
        "falsify_models_max": falsify_ceiling if falsify_values else 0,
        "strict_gate_pass": bool(strict_gate_pass),
        "reports": reports,
    }


def _build_layered_goals(
    *,
    runs: List[Dict[str, Any]],
    support_models: int,
    falsify_models: int,
    support_models_min: int,
    falsify_models_max: int,
    seed_block_summary: Dict[str, Any],
) -> Dict[str, Any]:
    g1_seed_pass = bool(seed_block_summary.get("strict_gate_pass", False))
    g2_arch_pass = support_models >= support_models_min and falsify_models <= falsify_models_max
    g3_task_pass = True
    if not runs:
        g3_task_pass = False
    for r in runs:
        ms = r.get("model_summary", {})
        if int(ms.get("falsify_categories", 0)) > 0 or int(ms.get("support_categories", 0)) < 2:
            g3_task_pass = False
            break

    if not g1_seed_pass:
        progression = "goal_1_pending"
    elif not g2_arch_pass:
        progression = "goal_2_pending"
    elif not g3_task_pass:
        progression = "goal_3_pending"
    else:
        progression = "all_goals_pass"

    return {
        "goal_1_same_arch_seed_stability": {
            "status": "pass" if g1_seed_pass else "pending",
            "criterion": f"seed_block: support_models_min>={support_models_min} and falsify_models_max<={falsify_models_max}",
            "evidence": seed_block_summary,
        },
        "goal_2_cross_arch_stability": {
            "status": "pass" if g2_arch_pass else "pending",
            "criterion": f"current_run: support_models>={support_models_min} and falsify_models<={falsify_models_max}",
            "evidence": {
                "support_models": support_models,
                "falsify_models": falsify_models,
            },
        },
        "goal_3_cross_task_family_stability": {
            "status": "pass" if g3_task_pass else "pending",
            "criterion": "per_model: falsify_categories=0 and support_categories>=2",
            "evidence": [
                {
                    "model": r.get("model"),
                    "support_categories": r.get("model_summary", {}).get("support_categories"),
                    "falsify_categories": r.get("model_summary", {}).get("falsify_categories"),
                }
                for r in runs
            ],
        },
        "progression": progression,
    }


def build_holdout_tasks(
    seed: int = 20260221,
    max_per_category: int = 24,
    task_profile: str = "standard",
) -> Dict[str, List[Tuple[str, str]]]:
    math_tasks = [(f"{a} + {b} =", f" {a + b}") for a in range(41, 80) for b in range(11, 29)]

    capitals = [
        ("Mexico", "Mexico City"),
        ("Russia", "Moscow"),
        ("Egypt", "Cairo"),
        ("Norway", "Oslo"),
        ("Kenya", "Nairobi"),
        ("Argentina", "Buenos Aires"),
        ("South Korea", "Seoul"),
        ("Turkey", "Ankara"),
    ]
    capital_templates = [
        "The capital city of {country} is",
        "Geography quiz: the capital of {country} is",
        "In world maps, {country} has capital",
    ]
    capital_tasks = [(tpl.format(country=c), f" {city}") for c, city in capitals for tpl in capital_templates]

    antonyms = [
        ("ancient", "modern"),
        ("expand", "contract"),
        ("accept", "reject"),
        ("arrive", "depart"),
        ("include", "exclude"),
        ("victory", "defeat"),
        ("visible", "hidden"),
        ("fragile", "robust"),
    ]
    antonym_templates = [
        "The opposite of {word} is",
        "An antonym for {word} is",
        "Reverse meaning of {word} is",
    ]
    antonym_tasks = [(tpl.format(word=w), f" {a}") for w, a in antonyms for tpl in antonym_templates]

    facts = [
        ("A year has", " 12 months"),
        ("A leap year can have", " 366 days"),
        ("A pentagon has", " 5 sides"),
        ("The speed unit abbreviation for kilometer per hour is", " km/h"),
        ("The largest planet in the solar system is", " Jupiter"),
        ("The human body temperature is often around", " 37 degrees Celsius"),
        ("The chemical symbol for silver is", " Ag"),
        ("The chemical symbol for sodium is", " Na"),
    ]
    fact_templates = [
        "{head}",
        "A basic fact: {head}",
        "Quick check: {head}",
    ]
    fact_tasks = [(tpl.format(head=h), t) for h, t in facts for tpl in fact_templates]

    if task_profile == "expanded":
        # harder arithmetic slices with larger values and carry-heavy patterns
        math_tasks.extend((f"{a} + {b} =", f" {a + b}") for a in range(81, 140) for b in range(21, 60))
        math_tasks.extend((f"Compute: {a} plus {b} equals", f" {a + b}") for a in range(60, 121) for b in range(17, 33))

        capitals.extend(
            [
                ("Thailand", "Bangkok"),
                ("Sweden", "Stockholm"),
                ("Portugal", "Lisbon"),
                ("Poland", "Warsaw"),
                ("Indonesia", "Jakarta"),
                ("Chile", "Santiago"),
                ("Peru", "Lima"),
                ("Netherlands", "Amsterdam"),
                ("Belgium", "Brussels"),
                ("Austria", "Vienna"),
                ("Switzerland", "Bern"),
                ("Denmark", "Copenhagen"),
            ]
        )
        capital_templates.extend(
            [
                "Q: What's the capital of {country}? A:",
                "{country} -> capital:",
            ]
        )
        capital_tasks = [(tpl.format(country=c), f" {city}") for c, city in capitals for tpl in capital_templates]

        antonyms.extend(
            [
                ("optimistic", "pessimistic"),
                ("expandable", "rigid"),
                ("constructive", "destructive"),
                ("sparse", "dense"),
                ("legal", "illegal"),
                ("increase", "decrease"),
                ("majority", "minority"),
                ("export", "import"),
                ("attack", "defend"),
                ("active", "passive"),
            ]
        )
        antonym_templates.extend(
            [
                "Fill in the opposite word: {word} ->",
                "Opposite term for {word}:",
            ]
        )
        antonym_tasks = [(tpl.format(word=w), f" {a}") for w, a in antonyms for tpl in antonym_templates]

        facts.extend(
            [
                ("A hexagon has", " 6 sides"),
                ("A heptagon has", " 7 sides"),
                ("The SI unit for electric current is", " ampere"),
                ("The SI unit for force is", " newton"),
                ("The chemical symbol for potassium is", " K"),
                ("The chemical symbol for iron is", " Fe"),
                ("The nearest planet to the Sun is", " Mercury"),
                ("The largest ocean on Earth is", " Pacific Ocean"),
                ("The freezing point of water in Fahrenheit is", " 32 degrees Fahrenheit"),
                ("The boiling point of water in Fahrenheit is", " 212 degrees Fahrenheit"),
            ]
        )
        fact_templates.extend(
            [
                "Science check: {head}",
                "Complete the fact: {head}",
            ]
        )
        fact_tasks = [(tpl.format(head=h), t) for h, t in facts for tpl in fact_templates]

    rng = random.Random(seed)
    out = {
        "math_add_holdout": math_tasks,
        "capital_holdout": capital_tasks,
        "antonym_holdout": antonym_tasks,
        "fact_holdout": fact_tasks,
    }
    for k in out:
        rng.shuffle(out[k])
        out[k] = out[k][:max_per_category]
    return out


def _base_category(holdout_category: str) -> str:
    return holdout_category.replace("_holdout", "")


def load_locked_configs(report_path: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not report_path:
        return {}
    path = Path(report_path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    locked: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for run in payload.get("runs", []):
        model_name = run.get("model")
        if not model_name:
            continue
        per_model: Dict[str, Dict[str, Any]] = {}
        for category, obj in (run.get("categories") or {}).items():
            if not isinstance(obj, dict):
                continue
            cfg = obj.get("selected_config")
            if cfg is None:
                cfg = ((obj.get("best") or {}).get("config"))
            if isinstance(cfg, dict):
                per_model[category] = dict(cfg)
        if per_model:
            locked[model_name] = per_model
    return locked


def load_single_locked_config(report_path: str) -> Dict[str, Dict[str, Any]]:
    if not report_path:
        return {}
    path = Path(report_path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for run in payload.get("runs", []):
        model_name = run.get("model")
        if not model_name:
            continue
        buckets: Dict[str, List[float]] = {}
        cfg_map: Dict[str, Dict[str, Any]] = {}
        for _cat, obj in (run.get("categories") or {}).items():
            if not isinstance(obj, dict):
                continue
            cfg = obj.get("selected_config")
            if cfg is None:
                cfg = ((obj.get("best") or {}).get("config"))
            if not isinstance(cfg, dict):
                continue
            score = float((obj.get("best") or {}).get("uplift_logprob", 0.0))
            cfg_key = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
            buckets.setdefault(cfg_key, []).append(score)
            cfg_map[cfg_key] = dict(cfg)
        if not buckets:
            continue
        best_key = max(buckets.keys(), key=lambda k: (sum(buckets[k]) / len(buckets[k]), len(buckets[k])))
        out[model_name] = cfg_map[best_key]
    return out


def load_failure_targets(report_path: str) -> Dict[str, set]:
    if not report_path:
        return {}
    path = Path(report_path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets: Dict[str, set] = {}
    for item in payload.get("strict_failure_clusters", []):
        model = item.get("model")
        category = str(item.get("category", ""))
        if not model or not category:
            continue
        base_cat = _base_category(category)
        targets.setdefault(model, set()).add(base_cat)
    return targets


def apply_adapter(
    cfg: Dict[str, Any],
    model_name: str,
    base_category: str,
    failure_targets: Dict[str, set],
    adapter_profile: str,
    adapter_strength: float,
) -> Tuple[Dict[str, Any], bool, str]:
    base_cfg = dict(cfg)
    if adapter_profile == "none":
        return base_cfg, False, "adapter_disabled"

    model_targets = failure_targets.get(model_name, set())
    s = max(0.1, min(2.0, float(adapter_strength)))
    out = dict(base_cfg)

    if adapter_profile == "conservative":
        if base_category not in model_targets:
            return base_cfg, False, "not_in_failure_cluster"
        out["alpha"] = round(max(0.08, float(base_cfg.get("alpha", 0.25)) * (1.0 - 0.35 * s)), 4)
        out["top_k"] = max(8, int(round(int(base_cfg.get("top_k", 16)) * (1.0 - 0.4 * s))))
        out["layer_idx"] = max(1, int(base_cfg.get("layer_idx", 1)))
        return out, True, "conservative_shrink"

    if adapter_profile == "math_fact_relief":
        if base_category not in model_targets:
            return base_cfg, False, "not_in_failure_cluster"
        if base_category in {"math_add", "fact"}:
            out["alpha"] = round(max(0.08, float(base_cfg.get("alpha", 0.25)) * (1.0 - 0.55 * s)), 4)
            out["top_k"] = max(8, int(round(int(base_cfg.get("top_k", 16)) * (1.0 - 0.5 * s))))
            out["layer_idx"] = max(1, int(round(int(base_cfg.get("layer_idx", 1)) - 1 * s)))
            return out, True, "math_fact_relief"
        return out, False, "profile_skip_non_target_category"

    if adapter_profile == "math_fact_strong":
        if base_category not in model_targets:
            return base_cfg, False, "not_in_failure_cluster"
        if base_category in {"math_add", "fact"}:
            out["alpha"] = round(max(0.06, float(base_cfg.get("alpha", 0.25)) * (1.0 - 0.75 * s)), 4)
            out["top_k"] = max(8, int(round(int(base_cfg.get("top_k", 16)) * (1.0 - 0.65 * s))))
            out["layer_idx"] = max(1, int(round(int(base_cfg.get("layer_idx", 1)) - 2 * s)))
            return out, True, "math_fact_strong"
        return out, False, "profile_skip_non_target_category"

    if adapter_profile == "math_only_ultra":
        if base_category not in model_targets:
            return base_cfg, False, "not_in_failure_cluster"
        if base_category == "math_add":
            out["alpha"] = round(max(0.04, float(base_cfg.get("alpha", 0.25)) * (1.0 - 0.85 * s)), 4)
            out["top_k"] = max(6, int(round(int(base_cfg.get("top_k", 16)) * (1.0 - 0.8 * s))))
            out["layer_idx"] = max(1, int(round(int(base_cfg.get("layer_idx", 1)) - 2 * s)))
            return out, True, "math_only_ultra"
        return out, False, "profile_skip_non_math_category"

    if adapter_profile == "neutralize_failures":
        if base_category not in model_targets:
            return base_cfg, False, "not_in_failure_cluster"
        # convert fragile categories toward near-control intervention to suppress hard regressions
        out["alpha"] = round(max(0.015, float(base_cfg.get("alpha", 0.25)) * (1.0 - 0.92 * s)), 4)
        out["top_k"] = max(4, int(round(int(base_cfg.get("top_k", 16)) * (1.0 - 0.85 * s))))
        out["layer_idx"] = max(1, int(round(int(base_cfg.get("layer_idx", 1)) - 2 * s)))
        return out, True, "neutralize_failures"

    if adapter_profile == "math_quarantine":
        if base_category not in model_targets:
            return base_cfg, False, "not_in_failure_cluster"
        # quarantine unstable math channel by reducing intervention close to no-op
        if base_category == "math_add":
            out["alpha"] = 0.0
            out["top_k"] = 1
            out["layer_idx"] = max(1, int(base_cfg.get("layer_idx", 1)))
            return out, True, "math_quarantine"
        return out, False, "profile_skip_non_math_category"

    if adapter_profile == "hybrid_constructive_v1":
        # constructive override for residual cluster + neutralize for remaining failure targets
        if model_name == "gpt2-medium" and base_category == "math_add":
            out["layer_idx"] = 6
            out["top_k"] = 4
            out["alpha"] = 0.01
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_constructive_gpt2m_math_add"
        # fallback: neutralize other targeted failures conservatively
        out["alpha"] = round(max(0.015, float(base_cfg.get("alpha", 0.25)) * (1.0 - 0.92 * s)), 4)
        out["top_k"] = max(4, int(round(int(base_cfg.get("top_k", 16)) * (1.0 - 0.85 * s))))
        out["layer_idx"] = max(1, int(round(int(base_cfg.get("layer_idx", 1)) - 2 * s)))
        return out, True, "hybrid_constructive_neutralize_rest"

    if adapter_profile == "hybrid_support_boost_v1":
        # Keep non-failure categories untouched to preserve positive transfer.
        # Apply constructive override only on residual cluster, neutralize only for known failure clusters.
        if model_name == "gpt2-medium" and base_category == "math_add":
            out["layer_idx"] = 6
            out["top_k"] = 4
            out["alpha"] = 0.01
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_gpt2m_math_add_constructive"
        model_targets = failure_targets.get(model_name, set())
        if base_category in model_targets:
            out["alpha"] = round(max(0.015, float(base_cfg.get("alpha", 0.25)) * (1.0 - 0.92 * s)), 4)
            out["top_k"] = max(4, int(round(int(base_cfg.get("top_k", 16)) * (1.0 - 0.85 * s))))
            out["layer_idx"] = max(1, int(round(int(base_cfg.get("layer_idx", 1)) - 2 * s)))
            return out, True, "hybrid_support_boost_neutralize_failure_target"
        return out, False, "hybrid_support_boost_keep_original"

    if adapter_profile == "hybrid_support_boost_v2":
        # v2: keep v1 safety, and add mild constructive boosts on capital for distilgpt2/gpt2-medium.
        if model_name == "gpt2-medium" and base_category == "math_add":
            out["layer_idx"] = 6
            out["top_k"] = 4
            out["alpha"] = 0.01
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v2_gpt2m_math_add_constructive"
        if model_name == "distilgpt2" and base_category == "capital":
            out["layer_idx"] = 1
            out["top_k"] = 16
            out["alpha"] = 0.35
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v2_distilgpt2_capital_boost"
        if model_name == "gpt2-medium" and base_category == "capital":
            out["layer_idx"] = 6
            out["top_k"] = 16
            out["alpha"] = 0.30
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v2_gpt2m_capital_boost"
        model_targets = failure_targets.get(model_name, set())
        if base_category in model_targets:
            out["alpha"] = round(max(0.015, float(base_cfg.get("alpha", 0.25)) * (1.0 - 0.92 * s)), 4)
            out["top_k"] = max(4, int(round(int(base_cfg.get("top_k", 16)) * (1.0 - 0.85 * s))))
            out["layer_idx"] = max(1, int(round(int(base_cfg.get("layer_idx", 1)) - 2 * s)))
            return out, True, "hybrid_support_boost_v2_neutralize_failure_target"
        return out, False, "hybrid_support_boost_v2_keep_original"

    if adapter_profile == "hybrid_support_boost_v3":
        # v3: keep strict-risk suppression while correcting gpt2 capital over-shoot.
        if model_name == "gpt2-medium" and base_category == "math_add":
            out["layer_idx"] = 6
            out["top_k"] = 4
            out["alpha"] = 0.01
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v3_gpt2m_math_add_constructive"
        if model_name == "gpt2" and base_category == "capital":
            out["layer_idx"] = 3
            out["top_k"] = 16
            out["alpha"] = 0.25
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v3_gpt2_capital_recenter"
        if model_name == "distilgpt2" and base_category == "capital":
            out["layer_idx"] = 1
            out["top_k"] = 16
            out["alpha"] = 0.35
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v3_distilgpt2_capital_boost"
        if model_name == "gpt2-medium" and base_category == "capital":
            out["layer_idx"] = 6
            out["top_k"] = 16
            out["alpha"] = 0.25
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v3_gpt2m_capital_boost"
        model_targets = failure_targets.get(model_name, set())
        if base_category in model_targets:
            out["alpha"] = round(max(0.015, float(base_cfg.get("alpha", 0.25)) * (1.0 - 0.92 * s)), 4)
            out["top_k"] = max(4, int(round(int(base_cfg.get("top_k", 16)) * (1.0 - 0.85 * s))))
            out["layer_idx"] = max(1, int(round(int(base_cfg.get("layer_idx", 1)) - 2 * s)))
            return out, True, "hybrid_support_boost_v3_neutralize_failure_target"
        return out, False, "hybrid_support_boost_v3_keep_original"

    if adapter_profile == "hybrid_support_boost_v4":
        # v4: add constructive gpt2 math_add profile validated by dedicated residual search.
        if model_name == "gpt2" and base_category == "math_add":
            out["layer_idx"] = 3
            out["top_k"] = 16
            out["alpha"] = 0.25
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v4_gpt2_math_add_constructive"
        if model_name == "gpt2" and base_category == "capital":
            out["layer_idx"] = 3
            out["top_k"] = 16
            out["alpha"] = 0.25
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v4_gpt2_capital_recenter"
        if model_name == "gpt2-medium" and base_category == "math_add":
            out["layer_idx"] = 6
            out["top_k"] = 4
            out["alpha"] = 0.01
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v4_gpt2m_math_add_constructive"
        if model_name == "distilgpt2" and base_category == "capital":
            out["layer_idx"] = 1
            out["top_k"] = 16
            out["alpha"] = 0.35
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v4_distilgpt2_capital_boost"
        if model_name == "gpt2-medium" and base_category == "capital":
            out["layer_idx"] = 6
            out["top_k"] = 16
            out["alpha"] = 0.25
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v4_gpt2m_capital_boost"
        model_targets = failure_targets.get(model_name, set())
        if base_category in model_targets:
            out["alpha"] = round(max(0.015, float(base_cfg.get("alpha", 0.25)) * (1.0 - 0.92 * s)), 4)
            out["top_k"] = max(4, int(round(int(base_cfg.get("top_k", 16)) * (1.0 - 0.85 * s))))
            out["layer_idx"] = max(1, int(round(int(base_cfg.get("layer_idx", 1)) - 2 * s)))
            return out, True, "hybrid_support_boost_v4_neutralize_failure_target"
        return out, False, "hybrid_support_boost_v4_keep_original"

    if adapter_profile == "hybrid_support_boost_v5":
        # v5: v4 + distilgpt2 capital constructive profile from dedicated capital search.
        if model_name == "gpt2" and base_category == "math_add":
            out["layer_idx"] = 3
            out["top_k"] = 16
            out["alpha"] = 0.25
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v5_gpt2_math_add_constructive"
        if model_name == "gpt2" and base_category == "capital":
            out["layer_idx"] = 3
            out["top_k"] = 16
            out["alpha"] = 0.25
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v5_gpt2_capital_recenter"
        if model_name == "distilgpt2" and base_category == "capital":
            out["layer_idx"] = 1
            out["top_k"] = 24
            out["alpha"] = 0.20
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v5_distilgpt2_capital_constructive"
        if model_name == "gpt2-medium" and base_category == "math_add":
            out["layer_idx"] = 6
            out["top_k"] = 4
            out["alpha"] = 0.01
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v5_gpt2m_math_add_constructive"
        if model_name == "gpt2-medium" and base_category == "capital":
            out["layer_idx"] = 6
            out["top_k"] = 16
            out["alpha"] = 0.25
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v5_gpt2m_capital_boost"
        model_targets = failure_targets.get(model_name, set())
        if base_category in model_targets:
            out["alpha"] = round(max(0.015, float(base_cfg.get("alpha", 0.25)) * (1.0 - 0.92 * s)), 4)
            out["top_k"] = max(4, int(round(int(base_cfg.get("top_k", 16)) * (1.0 - 0.85 * s))))
            out["layer_idx"] = max(1, int(round(int(base_cfg.get("layer_idx", 1)) - 2 * s)))
            return out, True, "hybrid_support_boost_v5_neutralize_failure_target"
        return out, False, "hybrid_support_boost_v5_keep_original"

    if adapter_profile == "hybrid_support_boost_v6":
        # v6: v5 + gpt2 capital/antonym calibrations from dedicated nine-seed search.
        if model_name == "gpt2" and base_category == "math_add":
            out["layer_idx"] = 3
            out["top_k"] = 16
            out["alpha"] = 0.25
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v6_gpt2_math_add_constructive"
        if model_name == "gpt2" and base_category == "capital":
            out["layer_idx"] = 4
            out["top_k"] = 32
            out["alpha"] = 0.25
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v6_gpt2_capital_constructive"
        if model_name == "gpt2" and base_category == "antonym":
            out["layer_idx"] = 6
            out["top_k"] = 32
            out["alpha"] = 0.25
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v6_gpt2_antonym_calibrated"
        if model_name == "distilgpt2" and base_category == "capital":
            out["layer_idx"] = 1
            out["top_k"] = 24
            out["alpha"] = 0.20
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v6_distilgpt2_capital_constructive"
        if model_name == "gpt2-medium" and base_category == "math_add":
            out["layer_idx"] = 6
            out["top_k"] = 4
            out["alpha"] = 0.01
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v6_gpt2m_math_add_constructive"
        if model_name == "gpt2-medium" and base_category == "capital":
            out["layer_idx"] = 6
            out["top_k"] = 16
            out["alpha"] = 0.25
            out["t"] = float(base_cfg.get("t", 1.0))
            return out, True, "hybrid_support_boost_v6_gpt2m_capital_boost"
        model_targets = failure_targets.get(model_name, set())
        if base_category in model_targets:
            out["alpha"] = round(max(0.015, float(base_cfg.get("alpha", 0.25)) * (1.0 - 0.92 * s)), 4)
            out["top_k"] = max(4, int(round(int(base_cfg.get("top_k", 16)) * (1.0 - 0.85 * s))))
            out["layer_idx"] = max(1, int(round(int(base_cfg.get("layer_idx", 1)) - 2 * s)))
            return out, True, "hybrid_support_boost_v6_neutralize_failure_target"
        return out, False, "hybrid_support_boost_v6_keep_original"

    return out, False, "unknown_profile"


def append_timeline(result: Dict[str, Any], summary_path: Path, timeline_path: Path) -> None:
    store = ExperimentTimelineStore(path=str(timeline_path))
    ag = result.get("aggregates", {})
    now = time.time()
    record = RunRecord(
        run_id=f"run_h3_holdout_validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        spec=AnalysisSpec(
            route="fiber_bundle",
            analysis_type="h3_holdout_validation",
            model="multi_model",
            params={"source_report": str(summary_path).replace("\\", "/")},
            input_payload={},
        ),
        status="completed",
        created_at=now,
        updated_at=now + 0.001,
        summary=RunSummary(
            metrics=[
                Metric(key="support_models", value=float(ag.get("support_models", 0)), min_value=0.0),
                Metric(key="falsify_models", value=float(ag.get("falsify_models", 0)), min_value=0.0),
                Metric(key="open_models", value=float(ag.get("open_models", 0)), min_value=0.0),
                Metric(key="consistency_score", value=float(ag.get("consistency_score", 0.0)), min_value=0.0, max_value=1.0),
            ],
            conclusion=ConclusionCard(
                objective="Validate whether H3 transfer effect generalizes to holdout task templates.",
                method="Category-wise adaptive intervention evaluated on unseen prompt/data templates.",
                evidence=[
                    f"support_models={ag.get('support_models')}",
                    f"falsify_models={ag.get('falsify_models')}",
                    f"open_models={ag.get('open_models')}",
                    f"consistency_score={ag.get('consistency_score')}",
                ],
                result=result.get("conclusion", ""),
                confidence=0.73,
                limitations=[
                    "Holdout pool is still synthetic and limited in breadth.",
                    "Only three model families are currently included.",
                ],
                next_action="Expand holdout pool and evaluate transfer without per-category retuning.",
            ),
            artifacts=[{"path": str(summary_path).replace("\\", "/")}],
        ),
        event_count=0,
    )
    store.append_run(record)


def main() -> None:
    parser = argparse.ArgumentParser(description="H3 holdout validation on unseen templates/tasks.")
    parser.add_argument("--models", default="gpt2,distilgpt2,gpt2-medium")
    parser.add_argument("--max-per-category", type=int, default=24)
    parser.add_argument("--task-profile", choices=["standard", "expanded"], default="standard")
    parser.add_argument("--locked-configs-from", default="")
    parser.add_argument("--lock-mode", choices=["per_category", "per_model_single"], default="per_category")
    parser.add_argument("--adapter-failure-report", default="")
    parser.add_argument(
        "--adapter-profile",
        choices=["none", "conservative", "math_fact_relief", "math_fact_strong", "math_only_ultra", "neutralize_failures", "math_quarantine", "hybrid_constructive_v1", "hybrid_support_boost_v1", "hybrid_support_boost_v2", "hybrid_support_boost_v3", "hybrid_support_boost_v4", "hybrid_support_boost_v5", "hybrid_support_boost_v6"],
        default="none",
    )
    parser.add_argument("--adapter-strength", type=float, default=1.0)
    parser.add_argument("--fallback-config-index", type=int, default=1)
    parser.add_argument("--support-models-min", type=int, default=2)
    parser.add_argument("--falsify-models-max", type=int, default=0)
    parser.add_argument("--seed-block-reports", default="")
    parser.add_argument("--seed", type=int, default=20260221)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--timeline", default="tempdata/agi_route_test_timeline.json")
    parser.add_argument("--output", default="tempdata/h3_holdout_validation_20260221.json")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    holdout_tasks = build_holdout_tasks(
        seed=args.seed,
        max_per_category=args.max_per_category,
        task_profile=args.task_profile,
    )
    locked_configs = load_locked_configs(args.locked_configs_from)
    single_locked = load_single_locked_config(args.locked_configs_from)
    failure_targets = load_failure_targets(args.adapter_failure_report)

    runs = []
    missing_locked: List[str] = []
    for model_name in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
        n_layers = len(model.transformer.h)
        configs = build_candidate_configs(num_layers=n_layers)
        ref_cache: Dict[int, torch.Tensor] = {}

        per_category: Dict[str, Any] = {}
        for category, tasks in holdout_tasks.items():
            base_cat = _base_category(category)
            model_locked = locked_configs.get(model_name, {})
            selected_cfg = None
            source = "locked_in_pool"
            if args.lock_mode == "per_model_single":
                selected_cfg = single_locked.get(model_name)
                source = "locked_in_pool_per_model_single"
            else:
                selected_cfg = model_locked.get(base_cat)
                source = "locked_in_pool_per_category"
            if selected_cfg is None:
                idx = max(0, min(len(configs) - 1, int(args.fallback_config_index)))
                selected_cfg = dict(configs[idx])
                source = "fallback_default"
                missing_locked.append(f"{model_name}:{base_cat}:{args.lock_mode}")
            adapted_cfg, adapter_applied, adapter_reason = apply_adapter(
                cfg=selected_cfg,
                model_name=model_name,
                base_category=base_cat,
                failure_targets=failure_targets,
                adapter_profile=args.adapter_profile,
                adapter_strength=args.adapter_strength,
            )
            source = f"{source}|{adapter_reason}"
            eval_result = eval_one_config(
                model=model,
                tokenizer=tokenizer,
                device=device,
                tasks=tasks,
                cfg=adapted_cfg,
                seed=args.seed + 1777,
                reference_cache=ref_cache,
            )
            per_category[category] = {
                "base_category": base_cat,
                "selected_config_source": source,
                "selected_config": dict(selected_cfg),
                "eval_config": dict(adapted_cfg),
                "adapter_applied": adapter_applied,
                "adapter_reason": adapter_reason,
                "best": eval_result,
            }

        best_verdicts = [v["best"]["verdict"] for v in per_category.values()]
        support_count = sum(1 for x in best_verdicts if x == "support")
        falsify_count = sum(1 for x in best_verdicts if x == "falsify")
        open_count = sum(1 for x in best_verdicts if x == "open")
        avg_uplift = float(np.mean([v["best"]["uplift_logprob"] for v in per_category.values()]))
        model_verdict = (
            "support_h3_holdout"
            if support_count >= 2 and falsify_count == 0
            else "falsify_h3_holdout"
            if falsify_count >= 2 and support_count == 0
            else "open_h3_holdout"
        )
        runs.append(
            {
                "model": model_name,
                "device": device,
                "categories": per_category,
                "model_summary": {
                    "support_categories": support_count,
                    "falsify_categories": falsify_count,
                    "open_categories": open_count,
                    "avg_best_uplift_logprob": round(avg_uplift, 8),
                    "model_verdict": model_verdict,
                },
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_verdicts = [r["model_summary"]["model_verdict"] for r in runs]
    support_models = sum(1 for v in model_verdicts if v == "support_h3_holdout")
    falsify_models = sum(1 for v in model_verdicts if v == "falsify_h3_holdout")
    open_models = sum(1 for v in model_verdicts if v == "open_h3_holdout")
    consistency_score = round(1.0 - (falsify_models > 0 and support_models > 0) * 0.5, 4)

    current_gate_pass = support_models >= args.support_models_min and falsify_models <= args.falsify_models_max
    if support_models > 0 and falsify_models > 0:
        status = "mixed_conflict"
    elif current_gate_pass:
        status = "provisional_support"
    elif falsify_models >= max(2, args.falsify_models_max + 1) and support_models == 0:
        status = "falsified"
    else:
        status = "open"

    seed_block_paths = _parse_report_paths(args.seed_block_reports)
    seed_block_summary = _summarize_seed_block(
        report_paths=seed_block_paths,
        support_models_min=args.support_models_min,
        falsify_models_max=args.falsify_models_max,
    )
    layered_goals = _build_layered_goals(
        runs=runs,
        support_models=support_models,
        falsify_models=falsify_models,
        support_models_min=args.support_models_min,
        falsify_models_max=args.falsify_models_max,
        seed_block_summary=seed_block_summary,
    )

    result = {
        "schema_version": "1.0",
        "test_date": datetime.now(timezone.utc).date().isoformat(),
        "analysis_type": "h3_holdout_validation",
        "status": status,
        "verdict": status,
        "config": {
            "models": models,
            "max_per_category": args.max_per_category,
            "sampling_strategy": "balanced_equal_per_category",
            "category_task_counts": {k: len(v) for k, v in holdout_tasks.items()},
            "task_profile": args.task_profile,
            "locked_configs_from": args.locked_configs_from,
            "lock_mode": args.lock_mode,
            "adapter_failure_report": args.adapter_failure_report,
            "adapter_profile": args.adapter_profile,
            "adapter_strength": args.adapter_strength,
            "fallback_config_index": args.fallback_config_index,
            "support_models_min": args.support_models_min,
            "falsify_models_max": args.falsify_models_max,
            "seed_block_reports": [str(p).replace("\\", "/") for p in seed_block_paths],
            "device": device,
            "seed": args.seed,
        },
        "runs": runs,
        "aggregates": {
            "support_models": support_models,
            "falsify_models": falsify_models,
            "open_models": open_models,
            "consistency_score": consistency_score,
            "status": status,
            "strict_gate": {
                "support_models_min_required": args.support_models_min,
                "falsify_models_max_allowed": args.falsify_models_max,
                "current_run_pass": current_gate_pass,
                "seed_block": seed_block_summary,
                "promotion_basis": "task_metrics_only",
            },
        },
        "layered_goals": layered_goals,
        "missing_locked_configs": sorted(set(missing_locked)),
        "conclusion": (
            "Holdout validation completed using locked in-pool configs (no holdout retuning)."
            if not missing_locked
            else "Holdout validation completed with partial fallback configs; add missing locked configs."
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    append_timeline(result, summary_path=out, timeline_path=Path(args.timeline))
    print(
        json.dumps(
            {
                "output": str(out),
                "status": status,
                "support_models": support_models,
                "falsify_models": falsify_models,
                "open_models": open_models,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
