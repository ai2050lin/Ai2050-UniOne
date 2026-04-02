# -*- coding: utf-8 -*-
"""
Stage468-471 批量运行脚本
========================
依次运行 Stage468-471 的 Qwen3 和 DeepSeek 测试。

用法:
  python run_stage468_471_batch.py

注意：
- 每个模型测试完后会释放GPU内存
- 总共8次测试，预计耗时40-80分钟
- 结果保存在 tests/codex_temp/ 目录下
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "tests" / "codex"

TESTS = [
    ("stage468_hyperbolic_embedding.py", "双曲嵌入验证"),
    ("stage469_geodesic_arithmetic.py", "测地线算术"),
    ("stage470_large_scale_graph.py", "大规模概念图谱"),
    ("stage471_raw_activation.py", "原始激活空间分析"),
]

MODELS = ["qwen3", "deepseek"]

def main():
    print("=" * 60)
    print("Stage468-471 批量测试")
    print("=" * 60)

    total_start = time.time()
    results = []

    for script, desc in TESTS:
        for model in MODELS:
            print(f"\n{'='*60}")
            print(f"运行: {desc} ({model})")
            print(f"{'='*60}")

            script_path = SCRIPTS_DIR / script
            if not script_path.exists():
                print(f"  错误: 脚本不存在 {script_path}")
                results.append((desc, model, "SKIPPED", "脚本不存在"))
                continue

            t0 = time.time()
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path), model],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30分钟超时
                )

                elapsed = time.time() - t0
                if result.returncode == 0:
                    status = "SUCCESS"
                    detail = f"耗时 {elapsed:.0f}s"
                else:
                    status = "FAILED"
                    detail = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr

                results.append((desc, model, status, detail))
                print(f"  状态: {status} ({elapsed:.0f}s)")

                # 输出最后几行
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-10:]:
                        print(f"  {line}")
                if status == "FAILED" and result.stderr:
                    err_lines = result.stderr.strip().split('\n')
                    for line in err_lines[-5:]:
                        print(f"  [ERR] {line}")

            except subprocess.TimeoutExpired:
                elapsed = time.time() - t0
                results.append((desc, model, "TIMEOUT", f"超过30分钟"))
                print(f"  超时 ({elapsed:.0f}s)")

            except Exception as e:
                results.append((desc, model, "ERROR", str(e)))
                print(f"  异常: {e}")

    # 汇总
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"全部测试完成，总耗时: {total_elapsed/60:.1f}分钟")
    print(f"{'='*60}")
    for desc, model, status, detail in results:
        print(f"  {desc} ({model}): {status}")
        if status != "SUCCESS":
            print(f"    → {detail[:200]}")

    # 保存汇总
    import json
    summary = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "total_time_min": total_elapsed / 60,
        "results": [{"test": d, "model": m, "status": s} for d, m, s, _ in results],
    }
    summary_path = PROJECT_ROOT / "tests" / "codex_temp" / "stage468_471_batch_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总保存至: {summary_path}")


if __name__ == "__main__":
    main()
