# -*- coding: utf-8 -*-
"""DeepSeek14B 快速验证脚本 - 简化探针，适配实际运行"""
import json
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime

MODEL_NAME = "deepseek-r1:14b"
TIMEOUT = 600  # 10分钟超时

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

# 简化探针集
PROBES = [
    {
        "probe_name": "水果义触发",
        "prompt": "只输出一个词(fruit或brand): 'I ate an apple after washing the fruit' -- apple是fruit还是brand?",
        "expected": "fruit",
    },
    {
        "probe_name": "品牌义触发",
        "prompt": "只输出一个词(fruit或brand): 'I bought an Apple laptop and updated the device' -- Apple是fruit还是brand?",
        "expected": "brand",
    },
    {
        "probe_name": "代词回指1",
        "prompt": "只输出一个词(apple或laptop): 'Tom sliced the apple and the laptop. It became sweet.' -- It指什么?",
        "expected": "apple",
    },
    {
        "probe_name": "代词回指2",
        "prompt": "只输出一个词(apple或laptop): 'Tom sliced the apple and the laptop. It has a keyboard.' -- It指什么?",
        "expected": "laptop",
    },
    {
        "probe_name": "中文水果义",
        "prompt": "只输出水果或科技: '我吃了一个苹果，味道很甜' -- 苹果是什么意思?",
        "expected": "水果",
    },
    {
        "probe_name": "中文品牌义",
        "prompt": "只输出水果或科技: '我买了一台苹果手机，运行很快' -- 苹果是什么意思?",
        "expected": "科技",
    },
]


def clean_output(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    # 优先取最后一个像答案的短行
    for line in reversed(lines):
        lower = line.lower()
        if lower in {"fruit", "brand", "apple", "laptop", "水果", "科技"}:
            return lower
    return lines[-1]


def normalize_label(raw: str, expected: str) -> str:
    lower = raw.lower()
    if expected in ("fruit", "brand"):
        if "fruit" in lower:
            return "fruit"
        if "brand" in lower:
            return "brand"
    if expected in ("apple", "laptop"):
        if "apple" in lower:
            return "apple"
        if "laptop" in lower:
            return "laptop"
    if expected in ("水果", "科技"):
        if "水果" in raw:
            return "水果"
        if "科技" in raw or "技术" in raw:
            return "科技"
    return lower


def run_prompt(prompt: str) -> str:
    print(f"    调用模型中...", end=" ", flush=True)
    completed = subprocess.run(
        ["ollama", "run", MODEL_NAME, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT,
    )
    combined = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
    cleaned = clean_output(combined)
    print(f"完成")
    return cleaned


def main():
    print(f"\n{'='*60}")
    print(f"  DeepSeek14B 快速验证测试")
    print(f"  模型: {MODEL_NAME}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    rows = []
    hit_count = 0

    for i, probe in enumerate(PROBES):
        name = probe["probe_name"]
        prompt = probe["prompt"]
        expected = probe["expected"]

        print(f"[{i+1}/{len(PROBES)}] {name}")
        print(f"  Prompt: {prompt[:60]}...")

        try:
            raw = run_prompt(prompt)
            predicted = normalize_label(raw, expected)
            is_correct = predicted == expected
        except subprocess.TimeoutExpired:
            raw = "TIMEOUT"
            predicted = "timeout"
            is_correct = False
            print("    超时!")
        except Exception as e:
            raw = f"ERROR: {e}"
            predicted = "error"
            is_correct = False
            print(f"    错误: {e}")

        if is_correct:
            hit_count += 1

        status = "PASS" if is_correct else "FAIL"
        print(f"  结果: {status} (expected={expected}, predicted={predicted}, raw={raw[:80]})")
        print()

        rows.append({
            "probe_name": name,
            "expected": expected,
            "predicted": predicted,
            "is_correct": is_correct,
            "raw_output": raw[:200],
        })

    score = hit_count / len(rows)
    weakest = next((r["probe_name"] for r in rows if not r["is_correct"]), "无")

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "deepseek14b_quick_test",
        "title": "DeepSeek14B 快速验证测试",
        "model_name": MODEL_NAME,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "probe_count": len(rows),
        "correct_count": hit_count,
        "score": score,
        "weakest_probe": weakest,
        "probe_rows": rows,
    }

    print(f"{'='*60}")
    print(f"  测试总结")
    print(f"{'='*60}")
    print(f"  总数: {len(rows)}, 正确: {hit_count}, 分数: {score:.2f}")
    print(f"  最弱探针: {weakest}")
    print()

    # 保存结果
    output_dir = Path(__file__).parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"deepseek14b_quick_test_{ts}.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  结果已保存: {out_file}")

    return summary


if __name__ == "__main__":
    main()
