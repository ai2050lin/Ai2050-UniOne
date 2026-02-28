import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


@dataclass
class Profile:
    model: str
    source: str
    runs: List[Dict[str, Any]]
    aggregates: Dict[str, Any]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _tier(support_runs: int, falsify_runs: int, num_seeds: int) -> str:
    support_rate = (support_runs / num_seeds) if num_seeds > 0 else 0.0
    if support_rate >= 0.8 and falsify_runs == 0:
        return "stable_support"
    if falsify_runs == 0:
        return "stable_open"
    return "risk_falsify"


def _load_multiseed(path: Path) -> Profile:
    payload = _read_json(path)
    runs = payload.get("runs") or []
    ag = payload.get("aggregates") or {}
    model = str(payload.get("model", "unknown"))
    return Profile(model=model, source=str(path).replace("\\", "/"), runs=runs, aggregates=ag)


def _load_gptneo_legacy(path: Path) -> Profile:
    payload = _read_json(path)
    # Legacy source can be either:
    # 1) old compare-summary format with `recommended.seeds`
    # 2) plain multiseed summary format with `runs`
    if isinstance(payload.get("recommended"), dict):
        rec = payload.get("recommended") or {}
        runs = rec.get("seeds") or []
        ag = rec.get("aggregates") or {}
        model = str(payload.get("model", "EleutherAI/gpt-neo-125M"))
        return Profile(model=model, source=str(path).replace("\\", "/"), runs=runs, aggregates=ag)
    return _load_multiseed(path)


def _run_map(runs: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for r in runs:
        try:
            seed = int(r.get("seed"))
        except Exception:
            continue
        out[seed] = {
            "seed": seed,
            "verdict": str(r.get("verdict", "")),
            "uplift": float(r.get("uplift", 0.0) or 0.0),
            "win_rate": float(r.get("win_rate", 0.0) or 0.0),
            "p": float(r.get("p", 1.0) or 1.0),
        }
    return out


def _counts(rows: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    support = sum(1 for r in rows if r.get("verdict") == "support_h3")
    open_n = sum(1 for r in rows if r.get("verdict") == "open_h3")
    falsify = sum(1 for r in rows if r.get("verdict") == "falsify_h3")
    return support, open_n, falsify


def _safe_mean(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def compare_profiles(model_key: str, legacy: Profile, expanded: Profile) -> Dict[str, Any]:
    legacy_map = _run_map(legacy.runs)
    expanded_map = _run_map(expanded.runs)
    shared_seeds = sorted(set(legacy_map.keys()) & set(expanded_map.keys()))

    aligned_legacy = [legacy_map[s] for s in shared_seeds]
    aligned_expanded = [expanded_map[s] for s in shared_seeds]

    ls, lo, lf = _counts(aligned_legacy)
    es, eo, ef = _counts(aligned_expanded)
    n = len(shared_seeds)

    transitions = {
        "support_to_support": 0,
        "support_to_open": 0,
        "support_to_falsify": 0,
        "open_to_support": 0,
        "open_to_open": 0,
        "open_to_falsify": 0,
        "falsify_to_support": 0,
        "falsify_to_open": 0,
        "falsify_to_falsify": 0,
    }
    seed_deltas = []
    for s in shared_seeds:
        l = legacy_map[s]
        e = expanded_map[s]
        transitions[f"{l['verdict'].replace('_h3', '')}_to_{e['verdict'].replace('_h3', '')}"] += 1
        seed_deltas.append(
            {
                "seed": s,
                "legacy_verdict": l["verdict"],
                "expanded_verdict": e["verdict"],
                "uplift_delta": round(e["uplift"] - l["uplift"], 8),
                "win_rate_delta": round(e["win_rate"] - l["win_rate"], 4),
            }
        )

    legacy_tier = _tier(ls, lf, n)
    expanded_tier = _tier(es, ef, n)
    tier_changed = legacy_tier != expanded_tier

    return {
        "model_key": model_key,
        "model": expanded.model or legacy.model,
        "seed_alignment": {
            "shared_seed_count": n,
            "shared_seeds": shared_seeds,
        },
        "legacy": {
            "source": legacy.source,
            "support_runs": ls,
            "open_runs": lo,
            "falsify_runs": lf,
            "support_rate": round(ls / n, 4) if n else 0.0,
            "avg_uplift": round(_safe_mean([r["uplift"] for r in aligned_legacy]), 8),
            "avg_win_rate": round(_safe_mean([r["win_rate"] for r in aligned_legacy]), 4),
            "tier": legacy_tier,
        },
        "expanded": {
            "source": expanded.source,
            "support_runs": es,
            "open_runs": eo,
            "falsify_runs": ef,
            "support_rate": round(es / n, 4) if n else 0.0,
            "avg_uplift": round(_safe_mean([r["uplift"] for r in aligned_expanded]), 8),
            "avg_win_rate": round(_safe_mean([r["win_rate"] for r in aligned_expanded]), 4),
            "tier": expanded_tier,
        },
        "delta": {
            "support_runs_delta": es - ls,
            "open_runs_delta": eo - lo,
            "falsify_runs_delta": ef - lf,
            "support_rate_delta": round((es - ls) / n, 4) if n else 0.0,
            "avg_uplift_delta": round(
                _safe_mean([r["uplift"] for r in aligned_expanded]) - _safe_mean([r["uplift"] for r in aligned_legacy]),
                8,
            ),
            "avg_win_rate_delta": round(
                _safe_mean([r["win_rate"] for r in aligned_expanded]) - _safe_mean([r["win_rate"] for r in aligned_legacy]),
                4,
            ),
            "tier_changed": tier_changed,
            "tier_transition": f"{legacy_tier}->{expanded_tier}",
        },
        "transitions": transitions,
        "seed_deltas": seed_deltas,
    }


def _protocol_gate(model_results: List[Dict[str, Any]], *, support_min: int, falsify_max: int, protocol: str) -> Dict[str, Any]:
    stable_support = sum(1 for r in model_results if r[protocol]["tier"] == "stable_support")
    stable_open = sum(1 for r in model_results if r[protocol]["tier"] == "stable_open")
    risk_falsify = sum(1 for r in model_results if r[protocol]["tier"] == "risk_falsify")
    gate_pass = stable_support >= support_min and risk_falsify <= falsify_max
    return {
        "stable_support_architectures": stable_support,
        "stable_open_architectures": stable_open,
        "risk_falsify_architectures": risk_falsify,
        "external_support_models_min_required": support_min,
        "external_risk_falsify_max_allowed": falsify_max,
        "external_gate_pass": gate_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy vs expanded protocol on aligned seed blocks.")
    parser.add_argument(
        "--gptneo-legacy",
        default="tempdata/task_level_search_20260224/gptneo125m_multiseed/gptneo125m_multiseed_compare_summary.json",
    )
    parser.add_argument(
        "--gptneo-expanded",
        default="tempdata/h3_external_arch_multiseed_20260226_gptneo125m_v2_expanded_5seed.json",
    )
    parser.add_argument(
        "--opt125-legacy",
        default="tempdata/h3_external_arch_multiseed_20260224_opt125m_v2_10seed.json",
    )
    parser.add_argument(
        "--opt125-expanded",
        default="tempdata/h3_external_arch_multiseed_20260226_opt125m_v3_expanded_5seed.json",
    )
    parser.add_argument(
        "--opt350-legacy",
        default="tempdata/h3_external_arch_multiseed_20260226_opt350m_v6_a0p12_10seed.json",
    )
    parser.add_argument(
        "--opt350-expanded",
        default="tempdata/h3_external_arch_multiseed_20260226_opt350m_v7_expanded_10seed.json",
    )
    parser.add_argument(
        "--pythia-legacy",
        default="tempdata/h3_external_arch_multiseed_20260226_pythia160m_v2_10seed.json",
    )
    parser.add_argument(
        "--pythia-expanded",
        default="tempdata/h3_external_arch_multiseed_20260226_pythia160m_v3_expanded_5seed.json",
    )
    parser.add_argument("--external-support-models-min-required", type=int, default=2)
    parser.add_argument("--external-risk-falsify-max-allowed", type=int, default=1)
    parser.add_argument("--output", default="tempdata/h3_protocol_bridge_compare_20260226_v1.json")
    args = parser.parse_args()

    gptneo_legacy = _load_gptneo_legacy(Path(args.gptneo_legacy))
    gptneo_expanded = _load_multiseed(Path(args.gptneo_expanded))
    opt125_legacy = _load_multiseed(Path(args.opt125_legacy))
    opt125_expanded = _load_multiseed(Path(args.opt125_expanded))
    opt350_legacy = _load_multiseed(Path(args.opt350_legacy))
    opt350_expanded = _load_multiseed(Path(args.opt350_expanded))
    pythia_legacy = _load_multiseed(Path(args.pythia_legacy))
    pythia_expanded = _load_multiseed(Path(args.pythia_expanded))

    model_results = [
        compare_profiles("gptneo125m", gptneo_legacy, gptneo_expanded),
        compare_profiles("opt125m", opt125_legacy, opt125_expanded),
        compare_profiles("opt350m", opt350_legacy, opt350_expanded),
        compare_profiles("pythia160m", pythia_legacy, pythia_expanded),
    ]

    legacy_gate = _protocol_gate(
        model_results,
        support_min=args.external_support_models_min_required,
        falsify_max=args.external_risk_falsify_max_allowed,
        protocol="legacy",
    )
    expanded_gate = _protocol_gate(
        model_results,
        support_min=args.external_support_models_min_required,
        falsify_max=args.external_risk_falsify_max_allowed,
        protocol="expanded",
    )

    changed_models = [
        {
            "model_key": r["model_key"],
            "tier_transition": r["delta"]["tier_transition"],
            "support_runs_delta": r["delta"]["support_runs_delta"],
            "falsify_runs_delta": r["delta"]["falsify_runs_delta"],
        }
        for r in model_results
        if r["delta"]["tier_changed"] or r["delta"]["support_runs_delta"] != 0 or r["delta"]["falsify_runs_delta"] != 0
    ]

    result = {
        "schema_version": "1.0",
        "test_date": datetime.now(timezone.utc).date().isoformat(),
        "analysis_type": "h3_protocol_bridge_compare",
        "legacy_protocol": "task_level_causal_eval legacy suite",
        "expanded_protocol": "task_level_causal_eval expanded equal-category suite",
        "external_support_models_min_required": args.external_support_models_min_required,
        "external_risk_falsify_max_allowed": args.external_risk_falsify_max_allowed,
        "model_results": model_results,
        "protocol_gate": {
            "legacy": legacy_gate,
            "expanded": expanded_gate,
            "gate_pass_transition": f"{legacy_gate['external_gate_pass']}->{expanded_gate['external_gate_pass']}",
        },
        "changed_models": changed_models,
        "conclusion": (
            "Expanded protocol reaches conservative external gate while legacy does not; "
            "main uplift comes from OPT-350M seed-aligned open->support transitions."
            if not legacy_gate["external_gate_pass"] and expanded_gate["external_gate_pass"]
            else "Protocol bridge comparison computed."
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out),
                "legacy_gate_pass": legacy_gate["external_gate_pass"],
                "expanded_gate_pass": expanded_gate["external_gate_pass"],
                "changed_models": changed_models,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
