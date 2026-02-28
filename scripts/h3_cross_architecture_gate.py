import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_runs_from_followup(payload: Dict[str, Any]) -> Tuple[int, int, int, float, float]:
    ms = (payload.get("multiseed_compare") or {}).get("recommended_aggregates") or {}
    return (
        int(ms.get("support_runs", 0)),
        int(ms.get("open_runs", 0)),
        int(ms.get("falsify_runs", 0)),
        float(ms.get("avg_uplift", 0.0) or 0.0),
        float(ms.get("avg_win_rate", 0.0) or 0.0),
    )


def _extract_runs_from_multiseed(payload: Dict[str, Any]) -> Tuple[int, int, int, float, float]:
    ag = payload.get("aggregates") or {}
    return (
        int(ag.get("support_runs", 0)),
        int(ag.get("open_runs", 0)),
        int(ag.get("falsify_runs", 0)),
        float(ag.get("avg_uplift", 0.0) or 0.0),
        float(ag.get("avg_win_rate", 0.0) or 0.0),
    )


def _tier(support_runs: int, falsify_runs: int, num_seeds: int) -> str:
    support_rate = (support_runs / num_seeds) if num_seeds > 0 else 0.0
    if support_rate >= 0.8 and falsify_runs == 0:
        return "stable_support"
    if falsify_runs == 0:
        return "stable_open"
    return "risk_falsify"


def build_gate_summary(
    *,
    gptneo_followup: Path,
    opt125_multiseed: Path,
    opt350_multiseed: Path,
    pythia_multiseed: Path,
    external_support_models_min_required: int,
    external_risk_falsify_max_allowed: int,
    version: str,
) -> Dict[str, Any]:
    sources = {
        "gptneo125m": (gptneo_followup, "followup"),
        "opt125m": (opt125_multiseed, "multiseed"),
        "opt350m": (opt350_multiseed, "multiseed"),
        "pythia160m": (pythia_multiseed, "multiseed"),
    }

    architectures: List[Dict[str, Any]] = []
    missing: List[str] = []
    for arch, (path, mode) in sources.items():
        if not path.exists():
            missing.append(f"{arch}:{str(path).replace(chr(92), '/')}")
            continue
        payload = _load_json(path)
        if mode == "followup":
            followup_ag = (payload.get("multiseed_compare") or {}).get("recommended_aggregates") or {}
            if followup_ag:
                support_runs, open_runs, falsify_runs, avg_uplift, avg_win_rate = _extract_runs_from_followup(payload)
                num_seeds = int(followup_ag.get("num_seeds", 0))
                source_mode = "recommended_conservative_multiseed"
            else:
                # Compat path: allow passing a plain multiseed summary for GPT-Neo.
                support_runs, open_runs, falsify_runs, avg_uplift, avg_win_rate = _extract_runs_from_multiseed(payload)
                num_seeds = int((payload.get("aggregates") or {}).get("num_seeds", 0))
                source_mode = "bestpoint_multiseed"
        else:
            support_runs, open_runs, falsify_runs, avg_uplift, avg_win_rate = _extract_runs_from_multiseed(payload)
            num_seeds = int((payload.get("aggregates") or {}).get("num_seeds", 0))
            source_mode = "bestpoint_multiseed"

        tier = _tier(support_runs=support_runs, falsify_runs=falsify_runs, num_seeds=num_seeds)
        support_rate = round((support_runs / num_seeds), 4) if num_seeds > 0 else 0.0
        falsify_rate = round((falsify_runs / num_seeds), 4) if num_seeds > 0 else 0.0
        architectures.append(
            {
                "architecture": arch,
                "source": str(path).replace("\\", "/"),
                "source_mode": source_mode,
                "num_seeds": num_seeds,
                "support_runs": support_runs,
                "open_runs": open_runs,
                "falsify_runs": falsify_runs,
                "support_rate": support_rate,
                "falsify_rate": falsify_rate,
                "avg_uplift": round(avg_uplift, 8),
                "avg_win_rate": round(avg_win_rate, 4),
                "tier": tier,
            }
        )

    stable_support = sum(1 for item in architectures if item["tier"] == "stable_support")
    stable_open = sum(1 for item in architectures if item["tier"] == "stable_open")
    risk_falsify = sum(1 for item in architectures if item["tier"] == "risk_falsify")
    external_gate_pass = (
        stable_support >= int(external_support_models_min_required)
        and risk_falsify <= int(external_risk_falsify_max_allowed)
    )

    return {
        "schema_version": "1.0",
        "test_date": datetime.now(timezone.utc).date().isoformat(),
        "analysis_type": "h3_cross_architecture_gate",
        "version": version,
        "architectures": architectures,
        "aggregates": {
            "num_architectures": len(architectures),
            "stable_support_architectures": stable_support,
            "stable_open_architectures": stable_open,
            "risk_falsify_architectures": risk_falsify,
            "external_support_models_min_required": int(external_support_models_min_required),
            "external_risk_falsify_max_allowed": int(external_risk_falsify_max_allowed),
            "external_gate_pass": bool(external_gate_pass),
        },
        "missing_inputs": missing,
        "conclusion": (
            "External architecture gate passes conservatively."
            if external_gate_pass
            else "External architecture gate not yet passed."
        ),
        "artifacts": [str(item[0]).replace("\\", "/") for item in sources.values()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build H3 cross-architecture gate summary.")
    parser.add_argument(
        "--gptneo-followup",
        default="tempdata/h3_external_arch_followup_20260224_gptneo125m.json",
    )
    parser.add_argument(
        "--opt125-multiseed",
        default="tempdata/h3_external_arch_multiseed_20260224_opt125m_v2_10seed.json",
    )
    parser.add_argument(
        "--opt350-multiseed",
        default="tempdata/h3_external_arch_multiseed_20260226_opt350m_v6_a0p12_10seed.json",
    )
    parser.add_argument(
        "--pythia-multiseed",
        default="tempdata/h3_external_arch_multiseed_20260226_pythia160m_v2_10seed.json",
    )
    parser.add_argument("--external-support-models-min-required", type=int, default=2)
    parser.add_argument("--external-risk-falsify-max-allowed", type=int, default=1)
    parser.add_argument("--version", default="v4")
    parser.add_argument("--output", default="tempdata/h3_cross_architecture_gate_20260226_v4.json")
    args = parser.parse_args()

    summary = build_gate_summary(
        gptneo_followup=Path(args.gptneo_followup),
        opt125_multiseed=Path(args.opt125_multiseed),
        opt350_multiseed=Path(args.opt350_multiseed),
        pythia_multiseed=Path(args.pythia_multiseed),
        external_support_models_min_required=args.external_support_models_min_required,
        external_risk_falsify_max_allowed=args.external_risk_falsify_max_allowed,
        version=args.version,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out),
                "external_gate_pass": summary["aggregates"]["external_gate_pass"],
                "stable_support_architectures": summary["aggregates"]["stable_support_architectures"],
                "risk_falsify_architectures": summary["aggregates"]["risk_falsify_architectures"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
