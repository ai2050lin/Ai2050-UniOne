# -*- coding: utf-8 -*-
"""
DeepSeek14B 全量测试运行器
在本地终端执行: python tests/codex_temp/run_all_deepseek14b_tests.py
预计总时间: 15-30分钟（4个stage, deepseek-r1:14b 已加载在GPU上）
"""
import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = PROJECT_ROOT / "tests" / "codex"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp"

# 要执行的测试文件列表（按顺序）
TEST_SCRIPTS = [
    {
        "stage": "Stage238",
        "name": "DeepSeek14B 直测处理链探针",
        "script": TESTS_DIR / "stage238_deepseek14b_direct_chain_probe.py",
        "args": ["--force"],
    },
    {
        "stage": "Stage241",
        "name": "DeepSeek14B 长链复杂处理复核",
        "script": TESTS_DIR / "stage241_deepseek14b_long_chain_probe.py",
        "args": ["--force"],
    },
    {
        "stage": "Stage244",
        "name": "DeepSeek14B 高压长链复核",
        "script": TESTS_DIR / "stage244_deepseek14b_stress_long_chain_probe.py",
        "args": ["--force"],
    },
    {
        "stage": "Stage256",
        "name": "DeepSeek14B 多方向翻译直测",
        "script": TESTS_DIR / "stage256_deepseek14b_multidirection_translation_probe.py",
        "args": ["--force"],
    },
]


def main():
    print(f"\n{'='*70}")
    print(f"  DeepSeek14B 全量测试运行器")
    print(f"  模型: deepseek-r1:14b (已加载)")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  测试数量: {len(TEST_SCRIPTS)}")
    print(f"{'='*70}\n")

    # 预检查
    print("[预检查] 验证 Ollama 服务...")
    r = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10)
    if "deepseek-r1:14b" not in r.stdout:
        print("[错误] deepseek-r1:14b 未在运行，请先启动: ollama run deepseek-r1:14b")
        sys.exit(1)
    print("[预检查] Ollama 服务正常, deepseek-r1:14b 已加载\n")

    all_results = []
    total_start = time.time()

    for i, test in enumerate(TEST_SCRIPTS):
        script_path = test["script"]
        if not script_path.exists():
            print(f"[跳过] {test['stage']}: 文件不存在 {script_path}")
            continue

        print(f"\n{'='*70}")
        print(f"  [{i+1}/{len(TEST_SCRIPTS)}] {test['stage']}: {test['name']}")
        print(f"  脚本: {script_path.name}")
        print(f"  开始: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}\n")

        stage_start = time.time()
        try:
            cmd = [sys.executable, str(script_path)] + test["args"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,  # 每个 stage 最多10分钟
                cwd=str(TESTS_DIR),
            )
            elapsed = time.time() - stage_start

            if result.returncode == 0:
                print(f"[完成] {test['stage']} - 耗时 {elapsed:.0f}秒")

                # 尝试读取生成的 summary.json
                summary = None
                for d in OUTPUT_DIR.iterdir():
                    if d.is_dir() and test["stage"][5:] in d.name.lower():
                        sf = d / "summary.json"
                        if sf.exists():
                            summary = json.loads(sf.read_text(encoding="utf-8-sig"))
                            break

                all_results.append({
                    "stage": test["stage"],
                    "name": test["name"],
                    "status": "PASS",
                    "elapsed_seconds": round(elapsed),
                    "summary": summary,
                })

                if summary:
                    score = summary.get("direct_chain_score") or \
                            summary.get("long_chain_score") or \
                            summary.get("stress_score") or \
                            summary.get("review_score") or \
                            summary.get("behavior_score") or \
                            summary.get("score") or "N/A"
                    correct = summary.get("correct_count", "N/A")
                    total = summary.get("probe_count", "N/A")
                    print(f"  分数: {score}, 正确: {correct}/{total}")
            else:
                print(f"[失败] {test['stage']} - 耗时 {elapsed:.0f}秒")
                print(f"  错误: {result.stderr[:500]}")
                all_results.append({
                    "stage": test["stage"],
                    "name": test["name"],
                    "status": "FAIL",
                    "elapsed_seconds": round(elapsed),
                    "error": result.stderr[:500],
                })

        except subprocess.TimeoutExpired:
            elapsed = time.time() - stage_start
            print(f"[超时] {test['stage']} - {elapsed:.0f}秒后超时")
            all_results.append({
                "stage": test["stage"],
                "name": test["name"],
                "status": "TIMEOUT",
                "elapsed_seconds": round(elapsed),
            })
        except Exception as e:
            elapsed = time.time() - stage_start
            print(f"[异常] {test['stage']} - {e}")
            all_results.append({
                "stage": test["stage"],
                "name": test["name"],
                "status": "ERROR",
                "elapsed_seconds": round(elapsed),
                "error": str(e),
            })

    total_elapsed = time.time() - total_start

    # 总结
    print(f"\n\n{'='*70}")
    print(f"  测试总结")
    print(f"{'='*70}")
    print(f"  总耗时: {total_elapsed:.0f}秒 ({total_elapsed/60:.1f}分钟)")
    print(f"  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    for r in all_results:
        status_icon = {"PASS": "✓", "FAIL": "✗", "TIMEOUT": "⏰", "ERROR": "!"}.get(r["status"], "?")
        print(f"  {status_icon} {r['stage']}: {r['name']} - {r['status']} ({r['elapsed_seconds']}s)")

    passed = sum(1 for r in all_results if r["status"] == "PASS")
    print(f"\n  通过: {passed}/{len(all_results)}")

    # 保存汇总
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "deepseek-r1:14b",
        "total_elapsed_seconds": round(total_elapsed),
        "results": all_results,
    }
    report_file = OUTPUT_DIR / f"deepseek14b_all_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  报告已保存: {report_file}")


if __name__ == "__main__":
    main()
