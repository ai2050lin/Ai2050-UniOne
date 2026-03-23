#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage190_cross_model_neuron_role_consensus import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["role_count"] == 5
    assert summary["strongest_role_name"] == "条件场角色"
    assert summary["weakest_role_name"] == "副词动态选路角色"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
