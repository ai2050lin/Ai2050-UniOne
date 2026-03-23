#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage191_shared_neuron_overlap_tracing import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["same_block_candidate_score"] > 0.3
    assert summary["cross_stage_overlap_count"] >= 1
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
