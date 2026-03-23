#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage119_gpt2_embedding_full_vocab_scan import OUTPUT_DIR as STAGE119_OUTPUT_DIR
from wordclass_neuron_basic_probe_lib import run_wordclass_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage126_verb_neuron_basic_probe_20260323"


def run_analysis(
    *,
    input_dir: Path = STAGE119_OUTPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
):
    return run_wordclass_analysis(
        target_lexical_type="verb",
        experiment_id="stage126_verb_neuron_basic_probe",
        title="Verb 神经元基础探针",
        status_short="gpt2_verb_neuron_probe_ready",
        input_dir=input_dir,
        output_dir=output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verb 神经元基础探针")
    parser.add_argument("--input-dir", default=str(STAGE119_OUTPUT_DIR), help="Stage119 输出目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage126 输出目录")
    args = parser.parse_args()

    summary = run_analysis(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "dominant_general_layer_index": summary["dominant_general_layer_index"],
                "wordclass_neuron_basic_probe_score": summary["wordclass_neuron_basic_probe_score"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
