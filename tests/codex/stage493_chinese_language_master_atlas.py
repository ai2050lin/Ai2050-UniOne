#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from stage492_chinese_pattern_route_atlas import (
    MAX_MIXED_SUBSET,
    TOP_CONTEXTS,
    TOP_HEAD_CANDIDATES,
    TOP_NEURON_CANDIDATES,
    OUTPUT_DIR as STAGE492_OUTPUT_DIR,  # noqa: F401
    ensure_dir,
    free_model,
    get_model_device,
    load_model_bundle,
    run_pattern,
    summarize_model_patterns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage493_chinese_language_master_atlas_20260403"
)
MODEL_KEYS = ["qwen3", "deepseek7b"]

PATTERNS = [
    {
        "pattern_key": "apple",
        "family": "concrete_noun",
        "word": "苹果",
        "prefix_char": "苹",
        "target_char": "果",
        "contexts": ["新鲜的苹", "我今天吃了苹", "这是一个苹", "他刚买了苹", "苹果的苹可以写成苹"],
    },
    {
        "pattern_key": "grape",
        "family": "concrete_noun",
        "word": "葡萄",
        "prefix_char": "葡",
        "target_char": "萄",
        "contexts": ["一串葡", "新鲜的葡", "我喜欢吃葡", "她洗了一串葡", "葡萄的葡可以写成葡", "紫色的葡"],
    },
    {
        "pattern_key": "butterfly",
        "family": "concrete_noun",
        "word": "蝴蝶",
        "prefix_char": "蝴",
        "target_char": "蝶",
        "contexts": ["一只蝴", "那只蝴", "我画了一只蝴", "彩色的蝴", "蝴蝶的蝴可以写成蝴"],
    },
    {
        "pattern_key": "self_reflexive",
        "family": "pronoun",
        "word": "自己",
        "prefix_char": "自",
        "target_char": "己",
        "contexts": ["保护自", "为了自", "自己的自可以写成自", "她只顾自", "先写自"],
    },
    {
        "pattern_key": "here",
        "family": "locative",
        "word": "这里",
        "prefix_char": "这",
        "target_char": "里",
        "contexts": ["就放在这", "站在这", "这里的这可以写成这", "看这", "留在这"],
    },
    {
        "pattern_key": "there",
        "family": "locative",
        "word": "那里",
        "prefix_char": "那",
        "target_char": "里",
        "contexts": ["坐在那", "放在那", "那里的那可以写成那", "走到那", "停在那"],
    },
    {
        "pattern_key": "today",
        "family": "time",
        "word": "今天",
        "prefix_char": "今",
        "target_char": "天",
        "contexts": ["到了今", "今天的今可以写成今", "从今", "直到今", "就在今"],
    },
    {
        "pattern_key": "tomorrow",
        "family": "time",
        "word": "明天",
        "prefix_char": "明",
        "target_char": "天",
        "contexts": ["等到明", "明天的明可以写成明", "就在明", "约在明", "到了明"],
    },
    {
        "pattern_key": "evening",
        "family": "time",
        "word": "晚上",
        "prefix_char": "晚",
        "target_char": "上",
        "contexts": ["到了晚", "晚上的晚可以写成晚", "今天晚", "就在晚", "每到晚"],
    },
    {
        "pattern_key": "some",
        "family": "quantity",
        "word": "一些",
        "prefix_char": "一",
        "target_char": "些",
        "contexts": ["还有一", "一些的一可以写成一", "买了一", "准备一", "再来一"],
    },
    {
        "pattern_key": "two_units",
        "family": "quantity",
        "word": "两个",
        "prefix_char": "两",
        "target_char": "个",
        "contexts": ["还有两", "两个的两可以写成两", "就差两", "准备两", "买了两"],
    },
    {
        "pattern_key": "several_times",
        "family": "quantity",
        "word": "几次",
        "prefix_char": "几",
        "target_char": "次",
        "contexts": ["去了几", "几次的几可以写成几", "来了几", "问了几", "试了几"],
    },
    {
        "pattern_key": "because",
        "family": "connective",
        "word": "因为",
        "prefix_char": "因",
        "target_char": "为",
        "contexts": ["因为的因可以写成因", "这里只写因", "句子先写因", "常见连词因", "解释时先写因"],
    },
    {
        "pattern_key": "but",
        "family": "connective",
        "word": "但是",
        "prefix_char": "但",
        "target_char": "是",
        "contexts": ["但是的但可以写成但", "这里只写但", "转折先写但", "句子里用但", "说到但"],
    },
    {
        "pattern_key": "if",
        "family": "connective",
        "word": "如果",
        "prefix_char": "如",
        "target_char": "果",
        "contexts": ["如果的如可以写成如", "这里只写如", "句子先写如", "常见连词如", "条件句里先写如"],
    },
    {
        "pattern_key": "although",
        "family": "connective",
        "word": "虽然",
        "prefix_char": "虽",
        "target_char": "然",
        "contexts": ["虽然的虽可以写成虽", "这里只写虽", "转折词里常见虽", "句子先写虽", "我们常写虽"],
    },
    {
        "pattern_key": "together",
        "family": "fixed_phrase",
        "word": "一起",
        "prefix_char": "一",
        "target_char": "起",
        "contexts": ["一起的一可以写成一", "我们一", "大家一", "想要一", "放在一"],
    },
    {
        "pattern_key": "for_example",
        "family": "fixed_phrase",
        "word": "比如",
        "prefix_char": "比",
        "target_char": "如",
        "contexts": ["比如的比可以写成比", "这里只写比", "举例时写比", "先说比", "常见表达比"],
    },
    {
        "pattern_key": "finally",
        "family": "fixed_phrase",
        "word": "终于",
        "prefix_char": "终",
        "target_char": "于",
        "contexts": ["终于的终可以写成终", "最后终", "事情终", "现在终", "结果终"],
    },
]


def build_report(summary: dict) -> str:
    lines = ["# stage493 中文语言模式总图谱", ""]
    lines.append("## 总结")
    lines.append("")
    lines.append(
        f"- 本轮覆盖模式数：`{len(PATTERNS)}`，家族包括 `concrete_noun / pronoun / locative / time / quantity / connective / fixed_phrase`。"
    )
    lines.append(
        f"- 路线搜索参数：`TOP_CONTEXTS={TOP_CONTEXTS}`，`TOP_HEAD_CANDIDATES={TOP_HEAD_CANDIDATES}`，`TOP_NEURON_CANDIDATES={TOP_NEURON_CANDIDATES}`，`MAX_MIXED_SUBSET={MAX_MIXED_SUBSET}`。"
    )
    lines.append("")
    for model_key in MODEL_KEYS:
        ms = summary["models"][model_key]
        lines.append(f"### {model_key}")
        lines.append("")
        lines.append(f"- 使用 CUDA（图形处理器）: `{ms['used_cuda']}`")
        lines.append(f"- 加载模式: `{ms.get('load_mode')}`")
        lines.append(f"- 总体拓扑统计: `{json.dumps(ms['model_summary']['overall_topology_counts'], ensure_ascii=False)}`")
        for family, family_row in ms["model_summary"]["family_summary"].items():
            lines.append(
                f"- `{family}`: 平均基线概率 `{family_row['mean_baseline_prob']:.4f}`，"
                f"平均最终下降 `{family_row['mean_final_drop']:.4f}`，"
                f"拓扑 `{json.dumps(family_row['topology_counts'], ensure_ascii=False)}`"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="建立扩展版中文语言模式总图谱")
    parser.add_argument("--prefer-cuda", action="store_true", help="优先使用 CUDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    summary = {
        "stage": "stage493_chinese_language_master_atlas",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prefer_cuda": bool(args.prefer_cuda),
        "pattern_count": len(PATTERNS),
        "models": {},
    }
    for model_key in MODEL_KEYS:
        model = None
        tokenizer = None
        try:
            model, tokenizer = load_model_bundle(model_key, prefer_cuda=args.prefer_cuda)
            pattern_rows = [run_pattern(model, tokenizer, pattern) for pattern in PATTERNS]
            summary["models"][model_key] = {
                "used_cuda": bool(get_model_device(model).type == "cuda"),
                "load_mode": "prefer_cuda" if args.prefer_cuda else "cpu",
                "patterns": pattern_rows,
                "model_summary": summarize_model_patterns(pattern_rows),
            }
        except NotImplementedError as exc:
            if model is not None:
                free_model(model)
            model, tokenizer = load_model_bundle(model_key, prefer_cuda=False)
            pattern_rows = [run_pattern(model, tokenizer, pattern) for pattern in PATTERNS]
            summary["models"][model_key] = {
                "used_cuda": bool(get_model_device(model).type == "cuda"),
                "load_mode": "cpu_fallback_after_meta",
                "fallback_reason": str(exc),
                "patterns": pattern_rows,
                "model_summary": summarize_model_patterns(pattern_rows),
            }
        finally:
            if model is not None:
                free_model(model)
    summary["elapsed_seconds"] = float(time.time() - started)
    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "REPORT.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_report(summary), encoding="utf-8")
    print(f"summary written to {summary_path}")
    print(f"report written to {report_path}")


if __name__ == "__main__":
    main()
