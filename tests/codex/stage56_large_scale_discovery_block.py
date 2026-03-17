from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[2]
INVENTORY_BUILDER = ROOT / "tests" / "codex" / "stage56_large_scale_discovery_inventory.py"
PIPELINE = ROOT / "tests" / "codex" / "stage56_multimodel_sequential_pipeline.py"
AGGREGATOR = ROOT / "tests" / "codex" / "stage56_large_scale_discovery_aggregator.py"

DEFAULT_OUTPUT_ROOT = ROOT / "tempdata" / "stage56_large_scale_discovery"
DEFAULT_ITEMS_FILE = ROOT / "tests" / "codex_temp" / "stage56_large_scale_discovery_items.csv"
DEFAULT_MANIFEST_FILE = ROOT / "tests" / "codex_temp" / "stage56_large_scale_discovery_manifest.json"
DEFAULT_REPORT_FILE = ROOT / "tests" / "codex_temp" / "stage56_large_scale_discovery_report.md"


def build_inventory_command(args: argparse.Namespace) -> List[str]:
    python_exe = args.python_exe or sys.executable
    command = [
        python_exe,
        str(INVENTORY_BUILDER),
        "--source-file",
        args.source_file,
        "--categories",
        args.categories,
        "--terms-per-category",
        str(args.terms_per_category),
        "--seed",
        str(args.seed),
        "--output-file",
        str(Path(args.items_file)),
        "--manifest-file",
        str(Path(args.inventory_manifest_file)),
        "--report-file",
        str(Path(args.inventory_report_file)),
    ]
    if args.require_category_word:
        command.append("--require-category-word")
    return command


def build_pipeline_command(args: argparse.Namespace) -> List[str]:
    python_exe = args.python_exe or sys.executable
    command = [
        python_exe,
        str(PIPELINE),
        "--models",
        args.models,
        "--items-file",
        str(Path(args.items_file)),
        "--output-root",
        str(Path(args.output_root)),
        "--survey-per-category",
        str(args.survey_per_category),
        "--deep-per-category",
        str(args.deep_per_category),
        "--closure-per-category",
        str(args.closure_per_category),
        "--anchors-per-category",
        str(args.anchors_per_category),
        "--challengers-per-category",
        str(args.challengers_per_category),
        "--supports-per-category",
        str(args.supports_per_category),
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
        "--subset-sizes",
        args.subset_sizes,
        "--stage5-max-candidates",
        str(args.stage5_max_candidates),
        "--stage5-per-category-limit",
        str(args.stage5_per_category_limit),
        "--stage5-max-neurons-per-candidate",
        str(args.stage5_max_neurons_per_candidate),
        "--stage5-max-neurons-per-layer",
        str(args.stage5_max_neurons_per_layer),
        "--stage5-prototype-term-mode",
        args.stage5_prototype_term_mode,
        "--stage5-margin-adv-threshold",
        str(args.stage5_margin_adv_threshold),
        "--stage5-margin-adv-penalty",
        str(args.stage5_margin_adv_penalty),
        "--stage6-max-instance-terms-per-category",
        str(args.stage6_max_instance_terms_per_category),
        "--stage6-strict-synergy-threshold",
        str(args.stage6_strict_synergy_threshold),
        "--score-alpha",
        str(args.score_alpha),
        "--candidate-overlap-penalty",
        str(args.candidate_overlap_penalty),
        "--max-candidate-overlap",
        str(args.max_candidate_overlap),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--progress-every",
        str(args.progress_every),
        "--require-category-coverage",
    ]
    if args.stage5_disable_prototype_proxy:
        command.append("--stage5-disable-prototype-proxy")
    if args.use_stage2_cleanup:
        command.append("--use-stage2-cleanup")
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        command.append("--dry-run")
    return command


def build_aggregator_command(args: argparse.Namespace) -> List[str]:
    python_exe = args.python_exe or sys.executable
    output_root = Path(args.output_root)
    return [
        python_exe,
        str(AGGREGATOR),
        "--output-root",
        str(output_root),
        "--summary-file",
        str(output_root / "discovery_summary.json"),
        "--report-file",
        str(output_root / "DISCOVERY_REPORT.md"),
        "--per-model-file",
        str(output_root / "discovery_per_model.jsonl"),
        "--per-category-file",
        str(output_root / "discovery_per_category.jsonl"),
    ]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a discovery-oriented large-scale stage56 block and aggregate patterns")
    ap.add_argument("--models", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,Qwen/Qwen3-4B")
    ap.add_argument("--python-exe", default="")
    ap.add_argument("--source-file", default="tests/codex/deepseek7b_nouns_english_520_clean.csv")
    ap.add_argument("--categories", default="")
    ap.add_argument("--terms-per-category", type=int, default=9)
    ap.add_argument("--require-category-word", action="store_true")
    ap.add_argument("--items-file", default=str(DEFAULT_ITEMS_FILE))
    ap.add_argument("--inventory-manifest-file", default=str(DEFAULT_MANIFEST_FILE))
    ap.add_argument("--inventory-report-file", default=str(DEFAULT_REPORT_FILE))
    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--survey-per-category", type=int, default=9)
    ap.add_argument("--deep-per-category", type=int, default=6)
    ap.add_argument("--closure-per-category", type=int, default=4)
    ap.add_argument("--anchors-per-category", type=int, default=2)
    ap.add_argument("--challengers-per-category", type=int, default=3)
    ap.add_argument("--supports-per-category", type=int, default=2)
    ap.add_argument("--use-stage2-cleanup", action="store_true")
    ap.add_argument("--family-count", type=int, default=10)
    ap.add_argument("--terms-per-family", type=int, default=4)
    ap.add_argument("--shared-k", type=int, default=48)
    ap.add_argument("--specific-k", type=int, default=24)
    ap.add_argument("--signature-top-k", type=int, default=256)
    ap.add_argument("--subset-sizes", default="48,32,24,16,12,8,6,4")
    ap.add_argument("--stage5-max-candidates", type=int, default=30)
    ap.add_argument("--stage5-per-category-limit", type=int, default=3)
    ap.add_argument("--stage5-max-neurons-per-candidate", type=int, default=12)
    ap.add_argument("--stage5-max-neurons-per-layer", type=int, default=4)
    ap.add_argument("--stage5-prototype-term-mode", choices=["any", "category_only"], default="any")
    ap.add_argument("--stage5-disable-prototype-proxy", action="store_true")
    ap.add_argument("--stage5-margin-adv-threshold", type=float, default=0.0)
    ap.add_argument("--stage5-margin-adv-penalty", type=float, default=0.02)
    ap.add_argument("--stage6-max-instance-terms-per-category", type=int, default=3)
    ap.add_argument("--stage6-strict-synergy-threshold", type=float, default=0.0)
    ap.add_argument("--score-alpha", type=float, default=256.0)
    ap.add_argument("--candidate-overlap-penalty", type=float, default=0.15)
    ap.add_argument("--max-candidate-overlap", type=float, default=1.0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--inventory-only", action="store_true")
    return ap.parse_args()


def run_command(command: List[str]) -> int:
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main() -> None:
    args = parse_args()
    inventory_rc = run_command(build_inventory_command(args))
    if inventory_rc != 0:
        raise SystemExit(inventory_rc)
    if args.inventory_only:
        return
    pipeline_rc = run_command(build_pipeline_command(args))
    if pipeline_rc != 0:
        raise SystemExit(pipeline_rc)
    raise SystemExit(run_command(build_aggregator_command(args)))


if __name__ == "__main__":
    main()
