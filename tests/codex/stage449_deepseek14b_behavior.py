# -*- coding: utf-8 -*-
"""
Stage449: DeepSeek-14B 行为测试（使用Ollama API）
通过行为探针验证AGI编码机制理论

目标：验证DeepSeek-14B对不同词性的处理能力
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests

# ==================== 配置 ====================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage449_deepseek14b_behavior_20260331"

OLLAMA_API_URL = "http://localhost:11454/api/generate"
MODEL_NAME = "deepseek-r1:14b"

# ==================== 测试探针 ====================

# 1. 词性分类探针
POS_CLASSIFICATION_PROBES = [
    # 名词探针
    {
        "id": "noun_apple",
        "pos": "noun",
        "prompt": "只输出一个词：apple（这是名词还是动词？）",
        "expected_pattern": r"noun",
    },
    {
        "id": "noun_chair", 
        "pos": "noun",
        "prompt": "只输出一个词：chair（这是名词还是形容词？）",
        "expected_pattern": r"noun",
    },
    # 动词探针
    {
        "id": "verb_run",
        "pos": "verb",
        "prompt": "只输出一个词：run（这是名词还是动词？）",
        "expected_pattern": r"verb",
    },
    {
        "id": "verb_think",
        "pos": "verb", 
        "prompt": "只输出一个词：think（这是动词还是名词？）",
        "expected_pattern": r"verb",
    },
    # 形容词探针
    {
        "id": "adj_beautiful",
        "pos": "adjective",
        "prompt": "只输出一个词：beautiful（这是形容词还是名词？）",
        "expected_pattern": r"adj",
    },
    {
        "id": "adj_big",
        "pos": "adjective",
        "prompt": "只输出一个词：big（这是什么词性？）",
        "expected_pattern": r"adj",
    },
    # 副词探针
    {
        "id": "adv_quickly",
        "pos": "adverb",
        "prompt": "只输出一个词：quickly（这是什么词性？）",
        "expected_pattern": r"adv",
    },
    {
        "id": "adv_always",
        "pos": "adverb",
        "prompt": "只输出一个词：always（这是什么词性？）",
        "expected_pattern": r"adv",
    },
]

# 2. 语义一致性探针
SEMANTIC_CONSISTENCY_PROBES = [
    # 水果一致性
    {
        "id": "fruit_apple_context1",
        "pos": "noun",
        "context": "I ate a fresh apple from the tree. The fruit was sweet.",
        "word": "apple",
        "expected": "fruit_sense",
        "prompt": "只输出一个词（fruit或brand）：apple在句子'I ate a fresh apple from the tree'中是什么意思？"
    },
    {
        "id": "fruit_apple_context2",
        "pos": "noun",
        "context": "I bought a new Apple laptop for coding. The brand is expensive.",
        "word": "Apple",
        "expected": "brand_sense",
        "prompt": "只输出一个词（fruit或brand）：Apple在句子'I bought a new Apple laptop'中是什么意思？"
    },
    # 动词时态
    {
        "id": "verb_past",
        "pos": "verb",
        "context": "Yesterday he walked to school.",
        "word": "walked",
        "expected": "past",
        "prompt": "只输出一个词（past/present/future）：walked是什么时态？"
    },
    {
        "id": "verb_future",
        "pos": "verb",
        "context": "Tomorrow she will fly to Paris.",
        "word": "will fly",
        "expected": "future",
        "prompt": "只输出一个词（past/present/future）：will fly是什么时态？"
    },
]

# 3. 上下文依赖探针
CONTEXT_DEPENDENCY_PROBES = [
    # 词义消歧
    {
        "id": "bank_river",
        "word": "bank",
        "context1": "I sat on the bank of the river and watched the water flow.",
        "context2": "I went to the bank to withdraw money.",
        "expected_diff": True,
        "prompt1": "只输出一个词（river/money）：bank在句子'I sat on the bank of the river'中是什么意思？",
        "prompt2": "只输出一个词（river/money）：bank在句子'I went to the bank'中是什么意思？",
    },
    # 代词指代
    {
        "id": "pronoun_coref",
        "word": "it",
        "context1": "The book is on the table. It is very old.",
        "context2": "The car is in the garage. It is very fast.",
        "expected_diff": False,
        "prompt1": "只输出一个词（book/table）：It在句子'The book is on the table. It is very old.'中指什么？",
        "prompt2": "只输出一个词（car/garage）：It在句子'The car is in the garage. It is very fast.'中指什么？",
    },
]

# 4. 层叠推理探针
MULTI_HOP_PROBES = [
    {
        "id": "transitive_inference",
        "prompt": "如果A大于B，B大于C。请问A和C哪个更大？只输出A、B或C。",
        "expected": "A",
    },
    {
        "id": "composition_logic",
        "prompt": "所有鸟都会飞。企鹅是鸟。请问企鹅会飞吗？只输出会或不会。",
        "expected": "不会",
    },
    {
        "id": "negation",
        "prompt": "我不不吃苹果。请问我吃苹果了吗？只回答吃或不吃。",
        "expected": "吃",
    },
]


# ==================== Ollama客户端 ====================

def check_ollama_connection() -> bool:
    """检查Ollama服务是否可用"""
    try:
        response = requests.get("http://localhost:11454/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            print(f"[OK] Ollama服务正常")
            print(f"     可用模型: {model_names}")
            if MODEL_NAME in model_names:
                print(f"     模型 '{MODEL_NAME}' 已安装")
                return True
            else:
                print(f"     [WARNING] 模型 '{MODEL_NAME}' 未安装")
                return False
        return False
    except Exception as e:
        print(f"[ERROR] Ollama连接失败: {e}")
        return False


def query_ollama(prompt: str, timeout: int = 120) -> str:
    """向Ollama发送查询"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 50},
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.Timeout:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {e}]"


# ==================== 探针执行 ====================

def clean_response(text: str) -> str:
    """清理模型响应"""
    text = text.strip()
    # 移除思考过程（如果有）
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0].strip()
    return text


def extract_answer(text: str, expected_type: str = "any") -> str:
    """从响应中提取答案"""
    text = clean_response(text)
    text_lower = text.lower()

    if expected_type == "fruit_or_brand":
        if "fruit" in text_lower and "brand" not in text_lower:
            return "fruit"
        if "brand" in text_lower and "fruit" not in text_lower:
            return "brand"
        return text_lower[:20]

    if expected_type == "past_present_future":
        if "past" in text_lower:
            return "past"
        if "future" in text_lower:
            return "future"
        if "present" in text_lower:
            return "present"
        return text_lower[:20]

    if expected_type == "a_b_c":
        if " a " in text_lower or text_lower.startswith("a"):
            return "A"
        if " b " in text_lower or text_lower.startswith("b"):
            return "B"
        if " c " in text_lower or text_lower.startswith("c"):
            return "C"
        return text_lower[:20]

    if expected_type == "will_not":
        if "不" in text or "不会" in text_lower or "no" in text_lower:
            return "不会"
        if "会" in text or "会" in text_lower or "yes" in text_lower:
            return "会"
        return text_lower[:20]

    if expected_type == "eat_or_not":
        if "吃" in text and "不吃" not in text_lower:
            return "吃"
        if "不吃" in text_lower:
            return "不吃"
        if "eat" in text_lower and "not eat" not in text_lower:
            return "eat"
        if "not eat" in text_lower:
            return "not eat"
        return text_lower[:20]

    return text_lower[:50]


def run_pos_classification_probe(probe: Dict) -> Dict:
    """运行词性分类探针"""
    response = query_ollama(probe["prompt"])
    answer = extract_answer(response)

    matched = bool(re.search(probe["expected_pattern"], answer, re.IGNORECASE))

    return {
        "probe_id": probe["id"],
        "pos": probe["pos"],
        "prompt": probe["prompt"],
        "response": clean_response(response),
        "extracted_answer": answer,
        "expected_pattern": probe["expected_pattern"],
        "matched": matched,
    }


def run_semantic_consistency_probe(probe: Dict) -> Dict:
    """运行语义一致性探针"""
    response = query_ollama(probe["prompt"])
    answer = extract_answer(response, "fruit_or_brand" if "fruit" in probe["id"] else "past_present_future")

    matched = probe["expected"] in answer.lower() or answer.lower() in probe["expected"].lower()

    return {
        "probe_id": probe["id"],
        "pos": probe["pos"],
        "context": probe.get("context", ""),
        "word": probe.get("word", ""),
        "expected": probe["expected"],
        "response": clean_response(response),
        "extracted_answer": answer,
        "matched": matched,
    }


def run_context_dependency_probe(probe: Dict) -> Dict:
    """运行上下文依赖探针"""
    response1 = query_ollama(probe["prompt1"])
    response2 = query_ollama(probe["prompt2"])

    answer1 = clean_response(response1)
    answer2 = clean_response(response2)

    # 检查两个答案是否不同（说明上下文被考虑）
    answers_different = answer1.lower() != answer2.lower()

    return {
        "probe_id": probe["id"],
        "word": probe["word"],
        "context1": probe["context1"],
        "context2": probe["context2"],
        "response1": answer1,
        "response2": answer2,
        "answers_different": answers_different,
        "expected_diff": probe["expected_diff"],
        "context_sensitive": answers_different == probe["expected_diff"],
    }


def run_multi_hop_probe(probe: Dict) -> Dict:
    """运行多层推理探针"""
    response = query_ollama(probe["prompt"])

    expected = probe["expected"]
    matched = expected.lower() in response.lower()

    return {
        "probe_id": probe["id"],
        "prompt": probe["prompt"],
        "expected": expected,
        "response": clean_response(response),
        "matched": matched,
    }


# ==================== 主分析流程 ====================

def run_behavior_analysis() -> Dict:
    """运行完整行为分析"""
    print(f"\n{'='*60}")
    print(f"  Stage449: DeepSeek-14B 行为测试")
    print(f"{'='*60}")
    print(f"  模型: {MODEL_NAME}")
    print(f"  API: {OLLAMA_API_URL}")
    print(f"{'='*60}\n")

    start_time = time.time()

    results = {
        "model_name": MODEL_NAME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pos_classification": [],
        "semantic_consistency": [],
        "context_dependency": [],
        "multi_hop_reasoning": [],
    }

    # 1. 词性分类测试
    print(f"\n[1/4] 词性分类测试 ({len(POS_CLASSIFICATION_PROBES)} probes)...")
    pos_correct = 0
    for i, probe in enumerate(POS_CLASSIFICATION_PROBES):
        print(f"    [{i+1}/{len(POS_CLASSIFICATION_PROBES)}] {probe['id']}...", end=" ", flush=True)
        result = run_pos_classification_probe(probe)
        results["pos_classification"].append(result)
        if result["matched"]:
            print(f"[OK] {result['extracted_answer']}")
            pos_correct += 1
        else:
            print(f"[FAIL] {result['extracted_answer']}")
        time.sleep(0.5)  # 避免过快请求

    # 2. 语义一致性测试
    print(f"\n[2/4] 语义一致性测试 ({len(SEMANTIC_CONSISTENCY_PROBES)} probes)...")
    sem_correct = 0
    for i, probe in enumerate(SEMANTIC_CONSISTENCY_PROBES):
        print(f"    [{i+1}/{len(SEMANTIC_CONSISTENCY_PROBES)}] {probe['id']}...", end=" ", flush=True)
        result = run_semantic_consistency_probe(probe)
        results["semantic_consistency"].append(result)
        if result["matched"]:
            print(f"[OK] {result['extracted_answer']}")
            sem_correct += 1
        else:
            print(f"[FAIL] {result['extracted_answer']}")
        time.sleep(0.5)

    # 3. 上下文依赖测试
    print(f"\n[3/4] 上下文依赖测试 ({len(CONTEXT_DEPENDENCY_PROBES)} probes)...")
    ctx_correct = 0
    for i, probe in enumerate(CONTEXT_DEPENDENCY_PROBES):
        print(f"    [{i+1}/{len(CONTEXT_DEPENDENCY_PROBES)}] {probe['id']}...", end=" ", flush=True)
        result = run_context_dependency_probe(probe)
        results["context_dependency"].append(result)
        if result["context_sensitive"]:
            print(f"[OK] 上下文敏感")
            ctx_correct += 1
        else:
            print(f"[FAIL] 不敏感")
        time.sleep(0.5)

    # 4. 多跳推理测试
    print(f"\n[4/4] 多跳推理测试 ({len(MULTI_HOP_PROBES)} probes)...")
    hop_correct = 0
    for i, probe in enumerate(MULTI_HOP_PROBES):
        print(f"    [{i+1}/{len(MULTI_HOP_PROBES)}] {probe['id']}...", end=" ", flush=True)
        result = run_multi_hop_probe(probe)
        results["multi_hop_reasoning"].append(result)
        if result["matched"]:
            print(f"[OK] {result['response']}")
            hop_correct += 1
        else:
            print(f"[FAIL] {result['response']}")
        time.sleep(0.5)

    elapsed = time.time() - start_time

    # 汇总统计
    total_probes = len(POS_CLASSIFICATION_PROBES) + len(SEMANTIC_CONSISTENCY_PROBES) + \
                   len(CONTEXT_DEPENDENCY_PROBES) + len(MULTI_HOP_PROBES)
    total_correct = pos_correct + sem_correct + ctx_correct + hop_correct

    results["summary"] = {
        "elapsed_seconds": elapsed,
        "pos_accuracy": pos_correct / max(1, len(POS_CLASSIFICATION_PROBES)),
        "semantic_accuracy": sem_correct / max(1, len(SEMANTIC_CONSISTENCY_PROBES)),
        "context_accuracy": ctx_correct / max(1, len(CONTEXT_DEPENDENCY_PROBES)),
        "multi_hop_accuracy": hop_correct / max(1, len(MULTI_HOP_PROBES)),
        "overall_accuracy": total_correct / max(1, total_probes),
        "total_probes": total_probes,
        "total_correct": total_correct,
    }

    return results


def build_report(results: Dict) -> str:
    """生成Markdown报告"""
    summary = results["summary"]

    lines = [
        "# Stage449: DeepSeek-14B 行为测试报告",
        "",
        "## 实验配置",
        f"- 时间: {results['timestamp']}",
        f"- 模型: {results['model_name']}",
        f"- 运行时间: {summary['elapsed_seconds']:.2f}秒",
        "",
        "## 测试结果汇总",
        "",
        f"| 测试类型 | 正确数 | 总数 | 准确率 |",
        f"|----------|--------|------|--------|",
        f"| 词性分类 | {sum(1 for r in results['pos_classification'] if r['matched'])} | {len(results['pos_classification'])} | {summary['pos_accuracy']:.1%} |",
        f"| 语义一致性 | {sum(1 for r in results['semantic_consistency'] if r['matched'])} | {len(results['semantic_consistency'])} | {summary['semantic_accuracy']:.1%} |",
        f"| 上下文依赖 | {sum(1 for r in results['context_dependency'] if r['context_sensitive'])} | {len(results['context_dependency'])} | {summary['context_accuracy']:.1%} |",
        f"| 多跳推理 | {sum(1 for r in results['multi_hop_reasoning'] if r['matched'])} | {len(results['multi_hop_reasoning'])} | {summary['multi_hop_accuracy']:.1%} |",
        f"| **总计** | **{summary['total_correct']}** | **{summary['total_probes']}** | **{summary['overall_accuracy']:.1%}** |",
        "",
        "## 详细结果",
        "",
    ]

    # 词性分类详情
    lines.extend(["### 词性分类", ""])
    for r in results["pos_classification"]:
        status = "[OK]" if r["matched"] else "[FAIL]"
        lines.append(f"- {status} **{r['pos']}**: {r['extracted_answer']}")
    lines.append("")

    # 语义一致性详情
    lines.extend(["### 语义一致性", ""])
    for r in results["semantic_consistency"]:
        status = "[OK]" if r["matched"] else "[FAIL]"
        lines.append(f"- {status} {r['word']} -> {r['extracted_answer']} (期望: {r['expected']})")
    lines.append("")

    # 上下文依赖详情
    lines.extend(["### 上下文依赖", ""])
    for r in results["context_dependency"]:
        status = "[OK]" if r["context_sensitive"] else "[FAIL]"
        lines.append(f"- {status} {r['word']}: ctx1={r['response1'][:20]}, ctx2={r['response2'][:20]}")
    lines.append("")

    # 多跳推理详情
    lines.extend(["### 多跳推理", ""])
    for r in results["multi_hop_reasoning"]:
        status = "[OK]" if r["matched"] else "[FAIL]"
        lines.append(f"- {status} {r['response'][:30]} (期望: {r['expected']})")

    return "\n".join(lines)


def save_outputs(results: Dict, output_dir: Path):
    """保存结果"""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    report_path = output_dir / "REPORT.md"

    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    report_path.write_text(build_report(results), encoding="utf-8-sig")

    print(f"\n[OK] 结果已保存到: {output_dir}")


def main():
    print(f"\n{'='*60}")
    print(f"  Stage449: DeepSeek-14B 行为测试")
    print(f"{'='*60}\n")

    # 检查Ollama连接
    if not check_ollama_connection():
        print("\n[ERROR] Ollama服务不可用或模型未安装")
        print("请确保运行: ollama serve")
        return

    # 运行分析
    results = run_behavior_analysis()

    # 保存结果
    save_outputs(results, OUTPUT_DIR)

    # 打印摘要
    summary = results["summary"]
    print(f"\n{'='*60}")
    print("  测试完成!")
    print(f"{'='*60}")
    print(f"模型: {results['model_name']}")
    print(f"总探针数: {summary['total_probes']}")
    print(f"总正确数: {summary['total_correct']}")
    print(f"总体准确率: {summary['overall_accuracy']:.1%}")
    print(f"运行时间: {summary['elapsed_seconds']:.2f}秒")
    print()
    print(f"词性分类准确率: {summary['pos_accuracy']:.1%}")
    print(f"语义一致性准确率: {summary['semantic_accuracy']:.1%}")
    print(f"上下文依赖准确率: {summary['context_accuracy']:.1%}")
    print(f"多跳推理准确率: {summary['multi_hop_accuracy']:.1%}")


if __name__ == "__main__":
    main()