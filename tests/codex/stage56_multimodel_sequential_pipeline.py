from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"

DEFAULT_MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen/Qwen3-4B",
]

MODEL_TAGS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "deepseek_7b",
    "Qwen/Qwen3-4B": "qwen3_4b",
    "Qwen/Qwen2.5-7B": "qwen2p5_7b",
    "zai-org/GLM-4-9B-Chat-HF": "glm4_9b_chat_hf",
}
STEP_SUCCESS_FILES = {
    "stage1_three_pool": "summary.json",
    "stage2_focus_builder": "focus_manifest.json",
    "stage2_focus_cleanup": "cleaned_focus_manifest.json",
    "stage3_causal_closure": "summary.json",
    "stage4_minimal_circuit": "summary.json",
    "stage5_prototype": "summary.json",
    "stage5_instance": "summary.json",
    "stage6_prototype_instance_decomposition": "summary.json",
}


def model_tag(model_id: str) -> str:
    tag = MODEL_TAGS.get(model_id)
    if tag is not None:
        return tag
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in model_id)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "model"


def build_stage_dirs(output_root: Path, model_id: str) -> Dict[str, Path]:
    base = output_root / model_tag(model_id)
    return {
        "model_root": base,
        "stage1": base / "stage1_three_pool",
        "stage2": base / "stage2_focus",
        "stage2_cleanup": base / "stage2_focus_cleanup",
        "stage3": base / "stage3_causal_closure",
        "stage4": base / "stage4_minimal_circuit",
        "stage5_prototype": base / "stage5_prototype",
        "stage5_instance": base / "stage5_instance",
        "stage6": base / "stage6_prototype_instance_decomposition",
    }


def make_step(name: str, command: Sequence[str], output_dir: Path) -> Dict[str, object]:
    return {
        "name": name,
        "command": list(command),
        "output_dir": str(output_dir),
    }


def build_command_plan(args: argparse.Namespace, model_id: str, stage_dirs: Dict[str, Path]) -> List[Dict[str, object]]:
    python_exe = args.python_exe or sys.executable
    stage1_items_file = args.items_file
    stage2_focus_manifest = stage_dirs["stage2"] / "focus_manifest.json"
    stage2_focus_manifest_for_stage3 = (
        stage_dirs["stage2_cleanup"] / "cleaned_focus_manifest.json"
        if bool(args.use_stage2_cleanup)
        else stage2_focus_manifest
    )
    stage2_families = stage_dirs["stage1"] / "families.jsonl"
    stage2_closure = stage_dirs["stage1"] / "closure_candidates.jsonl"
    stage3_summary = stage_dirs["stage3"] / "summary.json"
    stage3_baselines = stage_dirs["stage3"] / "baselines.jsonl"
    stage3_interventions = stage_dirs["stage3"] / "interventions.jsonl"
    stage4_results = stage_dirs["stage4"] / "results.jsonl"
    stage5_proto_candidates = stage_dirs["stage5_prototype"] / "candidates.jsonl"
    stage5_instance_candidates = stage_dirs["stage5_instance"] / "candidates.jsonl"

    plan = [
        make_step(
            "stage1_three_pool",
            [
                python_exe,
                str(CODEX_DIR / "deepseek7b_three_pool_structure_scan.py"),
                "--model-id",
                model_id,
                "--items-file",
                stage1_items_file,
                "--max-items",
                str(args.max_items),
                "--survey-per-category",
                str(args.survey_per_category),
                "--deep-per-category",
                str(args.deep_per_category),
                "--closure-per-category",
                str(args.closure_per_category),
                "--dtype",
                args.dtype,
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--progress-every",
                str(args.progress_every),
                "--output-dir",
                str(stage_dirs["stage1"]),
            ],
            stage_dirs["stage1"],
        ),
        make_step(
            "stage2_focus_builder",
            [
                python_exe,
                str(CODEX_DIR / "deepseek7b_stage2_focus_builder.py"),
                "--source-items-file",
                stage1_items_file,
                "--records-file",
                str(stage_dirs["stage1"] / "records.jsonl"),
                "--closure-candidates-file",
                str(stage2_closure),
                "--anchors-per-category",
                str(args.anchors_per_category),
                "--challengers-per-category",
                str(args.challengers_per_category),
                "--supports-per-category",
                str(args.supports_per_category),
                "--output-dir",
                str(stage_dirs["stage2"]),
            ],
            stage_dirs["stage2"],
        ),
    ]
    if bool(args.use_stage2_cleanup):
        plan.append(
            make_step(
                "stage2_focus_cleanup",
                [
                    python_exe,
                    str(CODEX_DIR / "deepseek7b_stage2_focus_cleanup.py"),
                    "--focus-manifest-file",
                    str(stage2_focus_manifest),
                    "--source-items-file",
                    args.cleanup_source_items_file,
                    "--seed-file",
                    args.cleanup_seed_file,
                    "--candidate-metadata-file",
                    args.cleanup_candidate_metadata_file,
                    "--records-file",
                    str(stage_dirs["stage1"] / "records.jsonl"),
                    "--closure-candidates-file",
                    str(stage2_closure),
                    "--output-dir",
                    str(stage_dirs["stage2_cleanup"]),
                ],
                stage_dirs["stage2_cleanup"],
            )
        )
    plan.extend([
        make_step(
            "stage3_causal_closure",
            [
                python_exe,
                str(CODEX_DIR / "deepseek7b_stage3_causal_closure.py"),
                "--model-id",
                model_id,
                "--dtype",
                args.dtype,
                "--device",
                args.device,
                "--focus-manifest",
                str(stage2_focus_manifest_for_stage3),
                "--stage2-families",
                str(stage2_families),
                "--stage2-closure",
                str(stage2_closure),
                "--family-count",
                str(args.family_count),
                "--terms-per-family",
                str(args.terms_per_family),
                "--shared-k",
                str(args.shared_k),
                "--specific-k",
                str(args.specific_k),
                "--signature-top-k",
                str(args.signature_top_k),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(stage_dirs["stage3"]),
            ],
            stage_dirs["stage3"],
        ),
        make_step(
            "stage4_minimal_circuit",
            [
                python_exe,
                str(CODEX_DIR / "deepseek7b_stage4_minimal_circuit_search.py"),
                "--model-id",
                model_id,
                "--dtype",
                args.dtype,
                "--device",
                args.device,
                "--focus-manifest",
                str(stage2_focus_manifest_for_stage3),
                "--stage2-families",
                str(stage2_families),
                "--stage3-summary",
                str(stage3_summary),
                "--stage3-baselines",
                str(stage3_baselines),
                "--stage3-interventions",
                str(stage3_interventions),
                "--source-kinds",
                args.source_kinds,
                "--subset-sizes",
                args.subset_sizes,
                "--signature-top-k",
                str(args.signature_top_k),
                "--margin-preserve-threshold",
                str(args.margin_preserve_threshold),
                "--global-common-max-fraction",
                str(args.global_common_max_fraction),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(stage_dirs["stage4"]),
            ],
            stage_dirs["stage4"],
        ),
        make_step(
            "stage5_prototype",
            [
                python_exe,
                str(CODEX_DIR / "deepseek7b_stage5_readout_coupled_search.py"),
                "--model-id",
                model_id,
                "--dtype",
                args.dtype,
                "--device",
                args.device,
                "--stage2-families",
                str(stage2_families),
                "--stage3-summary",
                str(stage3_summary),
                "--stage3-baselines",
                str(stage3_baselines),
                "--stage4-results",
                str(stage4_results),
                "--max-candidates",
                str(args.stage5_max_candidates),
                "--per-category-limit",
                str(args.stage5_per_category_limit),
                "--max-neurons-per-candidate",
                str(args.stage5_max_neurons_per_candidate),
                "--max-neurons-per-layer",
                str(args.stage5_max_neurons_per_layer),
                "--signature-top-k",
                str(args.signature_top_k),
                "--score-alpha",
                str(args.score_alpha),
                "--candidate-overlap-penalty",
                str(args.candidate_overlap_penalty),
                "--max-candidate-overlap",
                str(args.max_candidate_overlap),
                "--margin-adv-threshold",
                str(args.stage5_margin_adv_threshold),
                "--margin-adv-penalty",
                str(args.stage5_margin_adv_penalty),
                "--lane-mode",
                "prototype",
                "--prototype-term-mode",
                args.stage5_prototype_term_mode,
                "--seed",
                str(args.seed),
                "--output-dir",
                str(stage_dirs["stage5_prototype"]),
            ]
            + (["--disable-prototype-proxy"] if args.stage5_disable_prototype_proxy else [])
            + (["--require-category-coverage"] if args.require_category_coverage else []),
            stage_dirs["stage5_prototype"],
        ),
        make_step(
            "stage5_instance",
            [
                python_exe,
                str(CODEX_DIR / "deepseek7b_stage5_readout_coupled_search.py"),
                "--model-id",
                model_id,
                "--dtype",
                args.dtype,
                "--device",
                args.device,
                "--stage2-families",
                str(stage2_families),
                "--stage3-summary",
                str(stage3_summary),
                "--stage3-baselines",
                str(stage3_baselines),
                "--stage4-results",
                str(stage4_results),
                "--max-candidates",
                str(args.stage5_max_candidates),
                "--per-category-limit",
                str(args.stage5_per_category_limit),
                "--max-neurons-per-candidate",
                str(args.stage5_max_neurons_per_candidate),
                "--max-neurons-per-layer",
                str(args.stage5_max_neurons_per_layer),
                "--signature-top-k",
                str(args.signature_top_k),
                "--score-alpha",
                str(args.score_alpha),
                "--candidate-overlap-penalty",
                str(args.candidate_overlap_penalty),
                "--max-candidate-overlap",
                str(args.max_candidate_overlap),
                "--margin-adv-threshold",
                str(args.stage5_margin_adv_threshold),
                "--margin-adv-penalty",
                str(args.stage5_margin_adv_penalty),
                "--lane-mode",
                "instance",
                "--seed",
                str(args.seed),
                "--output-dir",
                str(stage_dirs["stage5_instance"]),
            ]
            + (["--require-category-coverage"] if args.require_category_coverage else []),
            stage_dirs["stage5_instance"],
        ),
        make_step(
            "stage6_prototype_instance_decomposition",
            [
                python_exe,
                str(CODEX_DIR / "deepseek7b_stage6_prototype_instance_decomposition.py"),
                "--model-id",
                model_id,
                "--dtype",
                args.dtype,
                "--device",
                args.device,
                "--stage2-families",
                str(stage2_families),
                "--stage3-summary",
                str(stage3_summary),
                "--stage3-baselines",
                str(stage3_baselines),
                "--prototype-candidates",
                str(stage5_proto_candidates),
                "--instance-candidates",
                str(stage5_instance_candidates),
                "--max-instance-terms-per-category",
                str(args.stage6_max_instance_terms_per_category),
                "--signature-top-k",
                str(args.signature_top_k),
                "--score-alpha",
                str(args.score_alpha),
                "--strict-synergy-threshold",
                str(args.stage6_strict_synergy_threshold),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(stage_dirs["stage6"]),
            ],
            stage_dirs["stage6"],
        ),
    ])
    return plan


def build_execution_plan(args: argparse.Namespace) -> List[Dict[str, object]]:
    output_root = Path(args.output_root)
    plan: List[Dict[str, object]] = []
    for model_id in args.models:
        stage_dirs = build_stage_dirs(output_root, model_id)
        for step in build_command_plan(args, model_id, stage_dirs):
            plan.append(
                {
                    "model_id": model_id,
                    "model_tag": model_tag(model_id),
                    **step,
                }
            )
    return plan


def run_step(step: Dict[str, object], dry_run: bool) -> Dict[str, object]:
    command = [str(x) for x in step["command"]]
    output_dir = Path(str(step["output_dir"]))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    if dry_run:
        return {
            "name": str(step["name"]),
            "model_id": str(step["model_id"]),
            "output_dir": str(output_dir),
            "returncode": 0,
            "runtime_sec": 0.0,
            "dry_run": True,
            "command": command,
        }
    log_path = output_dir / "step.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log_file.write(json.dumps({"command": command}, ensure_ascii=False) + "\n")
        proc = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_file.write(f"\n=== end {time.strftime('%Y-%m-%d %H:%M:%S')} / returncode={proc.returncode} ===\n")
    return {
        "name": str(step["name"]),
        "model_id": str(step["model_id"]),
        "output_dir": str(output_dir),
        "log_file": str(log_path),
        "returncode": int(proc.returncode),
        "runtime_sec": float(time.time() - started_at),
        "dry_run": False,
        "command": command,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_models(raw: str) -> List[str]:
    models = [x.strip() for x in raw.split(",") if x.strip()]
    if not models:
        raise ValueError("at least one model_id is required")
    return models


def main() -> None:
    ap = argparse.ArgumentParser(description="Run stage1-to-stage6 sequentially for DeepSeek/Qwen models without concurrent CUDA load")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--python-exe", default="")
    ap.add_argument("--items-file", default="tests/codex/deepseek7b_nouns_english_500plus.csv")
    ap.add_argument("--output-root", default="tempdata/stage56_multimodel_sequential")
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--survey-per-category", type=int, default=24)
    ap.add_argument("--deep-per-category", type=int, default=8)
    ap.add_argument("--closure-per-category", type=int, default=3)
    ap.add_argument("--anchors-per-category", type=int, default=2)
    ap.add_argument("--challengers-per-category", type=int, default=2)
    ap.add_argument("--supports-per-category", type=int, default=2)
    ap.add_argument("--use-stage2-cleanup", action="store_true")
    ap.add_argument(
        "--cleanup-source-items-file",
        default="tempdata/deepseek7b_tokenizer_vocab_expander_1500_20260317/combined_seed_plus_expanded.csv",
    )
    ap.add_argument("--cleanup-seed-file", default="tests/codex/deepseek7b_nouns_english_520_clean.csv")
    ap.add_argument(
        "--cleanup-candidate-metadata-file",
        default="tempdata/deepseek7b_tokenizer_vocab_expander_1500_20260317/all_candidates.jsonl",
    )
    ap.add_argument("--family-count", type=int, default=4)
    ap.add_argument("--terms-per-family", type=int, default=4)
    ap.add_argument("--shared-k", type=int, default=48)
    ap.add_argument("--specific-k", type=int, default=24)
    ap.add_argument("--signature-top-k", type=int, default=256)
    ap.add_argument("--source-kinds", default="family_shared,combined")
    ap.add_argument("--subset-sizes", default="48,32,24,16,12,8,6,4")
    ap.add_argument("--margin-preserve-threshold", type=float, default=0.8)
    ap.add_argument("--global-common-max-fraction", type=float, default=1.1)
    ap.add_argument("--stage5-max-candidates", type=int, default=6)
    ap.add_argument("--stage5-per-category-limit", type=int, default=2)
    ap.add_argument("--stage5-max-neurons-per-candidate", type=int, default=12)
    ap.add_argument("--stage5-max-neurons-per-layer", type=int, default=4)
    ap.add_argument("--stage5-prototype-term-mode", choices=["any", "category_only"], default="any")
    ap.add_argument("--stage5-disable-prototype-proxy", action="store_true")
    ap.add_argument("--stage5-margin-adv-threshold", type=float, default=0.0)
    ap.add_argument("--stage5-margin-adv-penalty", type=float, default=0.0)
    ap.add_argument("--stage6-max-instance-terms-per-category", type=int, default=2)
    ap.add_argument("--stage6-strict-synergy-threshold", type=float, default=0.0)
    ap.add_argument("--score-alpha", type=float, default=256.0)
    ap.add_argument("--candidate-overlap-penalty", type=float, default=0.15)
    ap.add_argument("--max-candidate-overlap", type=float, default=0.80)
    ap.add_argument("--require-category-coverage", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.models = parse_models(args.models)

    plan = build_execution_plan(args)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "plan.json", {"models": args.models, "steps": plan})

    results: List[Dict[str, object]] = []
    for step in plan:
        if bool(args.resume):
            marker_name = STEP_SUCCESS_FILES.get(str(step["name"]))
            marker_path = Path(str(step["output_dir"])) / marker_name if marker_name else None
            if marker_path is not None and marker_path.exists():
                result = {
                    "name": str(step["name"]),
                    "model_id": str(step["model_id"]),
                    "output_dir": str(step["output_dir"]),
                    "returncode": 0,
                    "runtime_sec": 0.0,
                    "dry_run": False,
                    "skipped": True,
                    "reason": "resume_marker_exists",
                    "marker_file": str(marker_path),
                    "command": [str(x) for x in step["command"]],
                }
                results.append(result)
                print(json.dumps({"event": "skip_step", **result}, ensure_ascii=False))
                continue
        print(
            json.dumps(
                {
                    "event": "start_step",
                    "model_id": step["model_id"],
                    "step": step["name"],
                    "output_dir": step["output_dir"],
                    "command": step["command"],
                },
                ensure_ascii=False,
            )
        )
        result = run_step(step, dry_run=bool(args.dry_run))
        results.append(result)
        print(json.dumps({"event": "end_step", **result}, ensure_ascii=False))
        if int(result["returncode"]) != 0:
            write_json(output_root / "run_summary.json", {"models": args.models, "steps": results, "success": False})
            raise SystemExit(int(result["returncode"]))

    write_json(output_root / "run_summary.json", {"models": args.models, "steps": results, "success": True})


if __name__ == "__main__":
    main()
