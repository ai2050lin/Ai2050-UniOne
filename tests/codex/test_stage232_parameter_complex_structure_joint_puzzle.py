#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage232_parameter_complex_structure_joint_puzzle import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["piece_count"] == 5
    assert summary["model_name"] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    assert summary["deepseek_direct_verdict"] == "theory_transfer_weak"
    assert summary["top_gap_name"] == "复杂处理结构仍弱于参数级边界与来源保真拼图"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
