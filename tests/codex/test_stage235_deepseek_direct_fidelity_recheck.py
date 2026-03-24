#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage235_deepseek_direct_fidelity_recheck import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["piece_count"] == 5
    assert summary["model_name"] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    assert summary["support_14b"] is True
    assert summary["top_gap_name"] == "DeepSeek复杂处理后段仍然偏弱"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
