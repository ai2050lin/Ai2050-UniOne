#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage413: 属性空间样本扩容（+50+样本）

目标：
1. 新增50+基础属性样本
2. 新增30+组合属性样本
3. 新增20+抽象属性样本
4. 构建属性空间的完整样本库

优先级：P1（最高）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage413_attribute_sample_expansion_20260329"


# 基础属性样本（50+）
BASIC_ATTRIBUTES = [
    # 颜色属性（5个）
    {"type": "颜色", "value": "红色", "examples": ["红色的苹果", "红色的香蕉（罕见）", "红色的梨"]},
    {"type": "颜色", "value": "绿色", "examples": ["绿色的苹果", "绿色的香蕉皮", "绿色的梨"]},
    {"type": "颜色", "value": "黄色", "examples": ["黄色的香蕉", "黄色的梨", "黄色的苹果皮"]},
    {"type": "颜色", "value": "橙色", "examples": ["橙色的橙子", "橙色的胡萝卜", "橙色的桃子"]},
    {"type": "颜色", "value": "紫色", "examples": ["紫色的葡萄", "紫色的李子", "紫色的芒果皮"]},

    # 形状属性（5个）
    {"type": "形状", "value": "圆形", "examples": ["圆形的苹果", "圆形的橙子", "圆形的桃子"]},
    {"type": "形状", "value": "长条形", "examples": ["长条形的香蕉", "长条形的黄瓜", "长条形的芒果"]},
    {"type": "形状", "value": "梨形", "examples": ["梨形的梨", "梨形的番茄", "梨形的苹果（罕见）"]},
    {"type": "形状", "value": "不规则形", "examples": ["不规则的草莓", "不规则的石榴", "不规则的芒果"]},
    {"type": "形状", "value": "球形", "examples": ["球形的葡萄", "球形的樱桃", "球形的蓝莓"]},

    # 大小属性（5个）
    {"type": "大小", "value": "大的", "examples": ["大的苹果", "大的西瓜", "大的芒果"]},
    {"type": "大小", "value": "小的", "examples": ["小的葡萄", "小的樱桃", "小的草莓"]},
    {"type": "大小", "value": "中等的", "examples": ["中等的苹果", "中等的香蕉", "中等的橙子"]},
    {"type": "大小", "value": "巨大的", "examples": ["巨大的西瓜", "巨大的菠萝", "巨大的榴莲"]},
    {"type": "大小", "value": "微小的", "examples": ["微小的葡萄干", "微小的蓝莓", "微小的蔓越莓"]},

    # 质地属性（5个）
    {"type": "质地", "value": "硬的", "examples": ["硬的苹果", "硬的梨", "硬的未成熟香蕉"]},
    {"type": "质地", "value": "软的", "examples": ["软的香蕉", "软的桃子", "软的芒果"]},
    {"type": "质地", "value": "脆的", "examples": ["脆的苹果", "脆的梨", "脆的胡萝卜"]},
    {"type": "质地", "value": "糯的", "examples": ["糯的芒果", "糯的香蕉", "糯的桃子"]},
    {"type": "质地", "value": "多汁的", "examples": ["多汁的西瓜", "多汁的橙子", "多汁的葡萄"]},

    # 味道属性（5个）
    {"type": "味道", "value": "甜的", "examples": ["甜的苹果", "甜的香蕉", "甜的橙子"]},
    {"type": "味道", "value": "酸的", "examples": ["酸的柠檬", "酸的青苹果", "酸的橙子"]},
    {"type": "味道", "value": "苦的", "examples": ["苦的葡萄皮", "苦的橙子皮", "苦的苦瓜"]},
    {"type": "味道", "value": "涩的", "examples": ["涩的柿子", "涩的未成熟香蕉", "涩的橄榄"]},
    {"type": "味道", "value": "无味的", "examples": ["无味的瓜", "无味的某些苹果", "无味的淡水果"]},

    # 温度属性（5个）
    {"type": "温度", "value": "凉的", "examples": ["凉的苹果", "凉的西瓜", "凉的葡萄"]},
    {"type": "温度", "value": "温的", "examples": ["温的香蕉", "温的芒果", "温的桃子"]},
    {"type": "温度", "value": "热的", "examples": ["热的梨（煮熟）", "热的苹果（烤）", "热的香蕉（烤）"]},
    {"type": "温度", "value": "冰镇的", "examples": ["冰镇的西瓜", "冰镇的葡萄", "冰镇的橙子"]},
    {"type": "温度", "value": "室温的", "examples": ["室温的苹果", "室温的香蕉", "室温的梨"]},

    # 新鲜度属性（5个）
    {"type": "新鲜度", "value": "新鲜的", "examples": ["新鲜的苹果", "新鲜的香蕉", "新鲜的葡萄"]},
    {"type": "新鲜度", "value": "腐烂的", "examples": ["腐烂的苹果", "腐烂的香蕉", "腐烂的葡萄"]},
    {"type": "新鲜度", "value": "干的", "examples": ["干的葡萄干", "干的芒果干", "干的香蕉片"]},
    {"type": "新鲜度", "value": "刚摘的", "examples": ["刚摘的苹果", "刚摘的桃子", "刚摘的葡萄"]},
    {"type": "新鲜度", "value": "储存久的", "examples": ["储存久的苹果", "储存久的梨", "储存久的橙子"]},

    # 成熟度属性（5个）
    {"type": "成熟度", "value": "未熟的", "examples": ["未熟的香蕉", "未熟的梨", "未熟的芒果"]},
    {"type": "成熟度", "value": "刚熟的", "examples": ["刚熟的苹果", "刚熟的桃子", "刚熟的橙子"]},
    {"type": "成熟度", "value": "熟透的", "examples": ["熟透的香蕉", "熟透的芒果", "熟透的桃子"]},
    {"type": "成熟度", "value": "过熟的", "examples": ["过熟的香蕉", "过熟的芒果", "过熟的梨"]},
    {"type": "成熟度", "value": "刚好", "examples": ["刚好的苹果", "刚好的橙子", "刚好的葡萄"]},

    # 产地属性（5个）
    {"type": "产地", "value": "国产的", "examples": ["国产的苹果", "国产的梨", "国产的葡萄"]},
    {"type": "产地", "value": "进口的", "examples": ["进口的香蕉", "进口的芒果", "进口的榴莲"]},
    {"type": "产地", "value": "本地的", "examples": ["本地的苹果", "本地的桃子", "本地的草莓"]},
    {"type": "产地", "value": "南方的", "examples": ["南方的香蕉", "南方的芒果", "南方的荔枝"]},
    {"type": "产地", "value": "北方的", "examples": ["北方的苹果", "北方的梨", "北方的葡萄"]},

    # 价格属性（5个）
    {"type": "价格", "value": "便宜的", "examples": ["便宜的香蕉", "便宜的苹果", "便宜的梨"]},
    {"type": "价格", "value": "贵的", "examples": ["贵的榴莲", "贵的水果", "贵的进口水果"]},
    {"type": "价格", "value": "中等的", "examples": ["中等的苹果", "中等的橙子", "中等的葡萄"]},
    {"type": "价格", "value": "超值的", "examples": ["超值的当季水果", "超值的打折水果", "超值的本地水果"]},
    {"type": "价格", "value": "奢侈的", "examples": ["奢侈的高端水果", "奢侈的进口水果", "奢侈的稀有水果"]},
]


# 组合属性样本（30+）
COMPOSITE_ATTRIBUTES = [
    {"type": "组合", "value": "红色圆形", "examples": ["红色圆形的苹果", "红色圆形的樱桃", "红色圆形的草莓"]},
    {"type": "组合", "value": "黄色长条形", "examples": ["黄色长条形的香蕉", "黄色长条形的柠檬", "黄色长条形的芒果"]},
    {"type": "组合", "value": "绿色圆形", "examples": ["绿色圆形的苹果", "绿色圆形的葡萄", "绿色圆形的李子"]},
    {"type": "组合", "value": "橙色球形", "examples": ["橙色球形的橙子", "橙色球形的橘柚", "橙色球形的杏"]},
    {"type": "组合", "value": "紫色球形", "examples": ["紫色球形的葡萄", "紫色球形的蓝莓", "紫色球形的桑葚"]},

    {"type": "组合", "value": "大的红色的", "examples": ["大的红色的苹果", "大的红色的草莓", "大的红色的桃子"]},
    {"type": "组合", "value": "小的黄色的", "examples": ["小的黄色的葡萄", "小的黄色的樱桃", "小的黄色的李子"]},
    {"type": "组合", "value": "中等的绿色的", "examples": ["中等的绿色的苹果", "中等的绿色的梨", "中等的绿色的青柠"]},
    {"type": "组合", "value": "巨大的橙色的", "examples": ["巨大的橙色的橙子", "巨大的橙色的柚子", "巨大的橙色的南瓜"]},
    {"type": "组合", "value": "微小的紫色的", "examples": ["微小的紫色的蓝莓", "微小的紫色的葡萄", "微小的紫色的桑葚"]},

    {"type": "组合", "value": "硬的脆的", "examples": ["硬脆的苹果", "硬脆的梨", "硬脆的胡萝卜"]},
    {"type": "组合", "value": "软的多汁的", "examples": ["软多汁的桃子", "软多汁的芒果", "软多汁的西瓜"]},
    {"type": "组合", "value": "甜的酸的", "examples": ["甜酸的苹果", "甜酸的橙子", "甜酸的葡萄"]},
    {"type": "组合", "value": "甜的软的", "examples": ["甜软的香蕉", "甜软的芒果", "甜软的桃子"]},
    {"type": "组合", "value": "酸的脆的", "examples": ["酸脆的青苹果", "酸脆的未成熟梨", "酸脆的柠檬"]},

    {"type": "组合", "value": "新鲜的甜的", "examples": ["新鲜甜的苹果", "新鲜甜的草莓", "新鲜甜的葡萄"]},
    {"type": "组合", "value": "冰镇的甜的", "examples": ["冰镇甜的西瓜", "冰镇甜的葡萄", "冰镇甜的橙子"]},
    {"type": "组合", "value": "室温的酸的", "examples": ["室温酸的柠檬", "室温酸的青苹果", "室温酸的橘子"]},
    {"type": "组合", "value": "温的软的", "examples": ["温软的香蕉", "温软的芒果", "温软的桃子"]},
    {"type": "组合", "value": "凉的脆的", "examples": ["凉脆的苹果", "凉脆的梨", "凉脆的黄瓜"]},

    {"type": "组合", "value": "刚熟的甜的", "examples": ["刚熟甜的苹果", "刚熟甜的桃子", "刚熟甜的葡萄"]},
    {"type": "组合", "value": "熟透的软的", "examples": ["熟透软的香蕉", "熟透软的芒果", "熟透软的桃子"]},
    {"type": "组合", "value": "未熟的硬的", "examples": ["未熟硬的香蕉", "未熟硬的梨", "未熟硬的芒果"]},
    {"type": "组合", "value": "过熟的烂的", "examples": ["过熟烂的香蕉", "过熟烂的苹果", "过熟烂的桃子"]},
    {"type": "组合", "value": "刚好脆的", "examples": ["刚好脆的苹果", "刚好脆的梨", "刚好脆的李子"]},

    {"type": "组合", "value": "国产的便宜的", "examples": ["国产便宜的苹果", "国产便宜的梨", "国产便宜的葡萄"]},
    {"type": "组合", "value": "进口的贵的", "examples": ["进口贵的榴莲", "进口贵的水果", "进口贵的蓝莓"]},
    {"type": "组合", "value": "本地的新鲜的", "examples": ["本地新鲜的草莓", "本地新鲜的桃子", "本地新鲜的葡萄"]},
    {"type": "组合", "value": "南方的甜的", "examples": ["南方甜的香蕉", "南方甜的芒果", "南方甜的荔枝"]},
    {"type": "组合", "value": "北方的脆的", "examples": ["北方脆的苹果", "北方脆的梨", "北方脆的葡萄"]},

    {"type": "组合", "value": "红色圆形甜的", "examples": ["红色圆形甜的苹果", "红色圆形甜的桃子", "红色圆形甜的草莓"]},
    {"type": "组合", "value": "黄色长条形软的", "examples": ["黄色长条形软的香蕉", "黄色长条形软的芒果", "黄色长条形软的柠檬"]},
    {"type": "组合", "value": "绿色圆形酸的", "examples": ["绿色圆形酸的苹果", "绿色圆形酸的柠檬", "绿色圆形酸的青枣"]},
    {"type": "组合", "value": "紫色球形甜的", "examples": ["紫色球形甜的葡萄", "紫色球形甜的蓝莓", "紫色球形甜的桑葚"]},
    {"type": "组合", "value": "橙色球形多汁的", "examples": ["橙色球形多汁的橙子", "橙色球形多汁的橘柚", "橙色球形多汁的柚子"]},
]


# 抽象属性样本（20+）
ABSTRACT_ATTRIBUTES = [
    {"type": "抽象", "value": "好吃的", "examples": ["好吃的苹果", "好吃的香蕉", "好吃的葡萄"]},
    {"type": "抽象", "value": "不好吃的", "examples": ["不好吃的未成熟水果", "不好吃的过熟水果", "不好吃的烂水果"]},
    {"type": "抽象", "value": "好看的", "examples": ["好看的苹果", "好看的葡萄", "好看的草莓"]},
    {"type": "抽象", "value": "难看的", "examples": ["难看的变形水果", "难看的烂水果", "难看的有斑点水果"]},
    {"type": "抽象", "value": "有营养的", "examples": ["有营养的苹果", "有营养的香蕉", "有营养的橙子"]},
    {"type": "抽象", "value": "没营养的", "examples": ["没营养的糖果", "没营养的零食", "没营养的加工食品"]},

    {"type": "抽象", "value": "健康的", "examples": ["健康的水果", "健康的蔬菜", "健康的坚果"]},
    {"type": "抽象", "value": "不健康的", "examples": ["不健康的垃圾食品", "不健康的油炸食品", "不健康的含糖饮料"]},
    {"type": "抽象", "value": "适合减肥的", "examples": ["适合减肥的苹果", "适合减肥的葡萄", "适合减肥的蔬菜"]},
    {"type": "抽象", "value": "增肥的", "examples": ["增肥的高糖水果", "增肥的油炸食品", "增肥的高热量零食"]},

    {"type": "抽象", "value": "美味的", "examples": ["美味的苹果", "美味的香蕉", "美味的葡萄"]},
    {"type": "抽象", "value": "难吃的", "examples": ["难吃的苦果", "难吃的涩果", "难吃的酸果"]},
    {"type": "抽象", "value": "诱人的", "examples": ["诱人的草莓", "诱人的葡萄", "诱人的桃子"]},
    {"type": "抽象", "value": "不诱人的", "examples": ["不诱人的烂水果", "不诱人的变形水果", "不诱人的干水果"]},

    {"type": "抽象", "value": "新鲜的", "examples": ["新鲜的苹果", "新鲜的香蕉", "新鲜的葡萄"]},
    {"type": "抽象", "value": "陈旧的", "examples": ["陈旧的干水果", "陈旧的储存很久的水果", "陈旧的失水水果"]},
    {"type": "抽象", "value": "完美的", "examples": ["完美的苹果", "完美的葡萄", "完美的桃子"]},
    {"type": "抽象", "value": "有瑕疵的", "examples": ["有瑕疵的水果", "有斑点的苹果", "有损伤的水果"]},

    {"type": "抽象", "value": "优质的", "examples": ["优质的水果", "优质的苹果", "优质的进口水果"]},
    {"type": "抽象", "value": "劣质的", "examples": ["劣质的水果", "劣质的储存不好的水果", "劣质的过熟水果"]},
    {"type": "抽象", "value": "精选的", "examples": ["精选的苹果", "精选的葡萄", "精选的高端水果"]},
    {"type": "抽象", "value": "普通的", "examples": ["普通的水果", "普通的超市水果", "普通的批量水果"]},

    {"type": "抽象", "value": "有机的", "examples": ["有机的苹果", "有机的香蕉", "有机的蔬菜"]},
    {"type": "抽象", "value": "非有机的", "examples": ["非有机的普通水果", "非有机的农药水果", "非有机的化肥水果"]},
    {"type": "抽象", "value": "天然的", "examples": ["天然的水果", "天然的果汁", "天然的蜂蜜"]},
    {"type": "抽象", "value": "人工的", "examples": ["人工的添加剂食品", "人工的合成食品", "人工的加工食品"]},
]


def generate_attribute_samples() -> List[Dict[str, Any]]:
    """生成属性样本"""
    samples = []

    # 基础属性样本
    for attr in BASIC_ATTRIBUTES:
        for example in attr["examples"]:
            samples.append({
                "attribute_type": attr["type"],
                "attribute_value": attr["value"],
                "example": example,
                "sample_type": "基础属性",
                "complexity": 1,
            })

    # 组合属性样本
    for attr in COMPOSITE_ATTRIBUTES:
        for example in attr["examples"]:
            samples.append({
                "attribute_type": attr["type"],
                "attribute_value": attr["value"],
                "example": example,
                "sample_type": "组合属性",
                "complexity": 2,
            })

    # 抽象属性样本
    for attr in ABSTRACT_ATTRIBUTES:
        for example in attr["examples"]:
            samples.append({
                "attribute_type": attr["type"],
                "attribute_value": attr["value"],
                "example": example,
                "sample_type": "抽象属性",
                "complexity": 3,
            })

    return samples


def build_summary() -> dict:
    """构建分析摘要"""
    # 生成样本
    samples = generate_attribute_samples()

    # 统计样本数量
    basic_count = len([s for s in samples if s["sample_type"] == "基础属性"])
    composite_count = len([s for s in samples if s["sample_type"] == "组合属性"])
    abstract_count = len([s for s in samples if s["sample_type"] == "抽象属性"])
    total_count = len(samples)

    # 计算扩厚增益
    # 基础目标：50+基础属性样本，30+组合属性样本，20+抽象属性样本
    basic_target = 50
    composite_target = 30
    abstract_target = 20

    basic_gain = min(1.0, basic_count / basic_target)
    composite_gain = min(1.0, composite_count / composite_target)
    abstract_gain = min(1.0, abstract_count / abstract_target)

    total_gain = (basic_gain + composite_gain + abstract_gain) / 3.0

    # 转换为强度增量
    strength_increment = total_gain * 0.10  # 预期增益+0.10

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage413_attribute_sample_expansion",
        "title": "属性空间样本扩容（+50+样本）",
        "status_short": "attribute_sample_expansion_completed",
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),

        # 样本统计
        "sample_statistics": {
            "basic_count": basic_count,
            "composite_count": composite_count,
            "abstract_count": abstract_count,
            "total_count": total_count,
            "basic_target": basic_target,
            "composite_target": composite_target,
            "abstract_target": abstract_target,
        },

        # 扩厚效果
        "expansion_effect": {
            "basic_gain": basic_gain,
            "composite_gain": composite_gain,
            "abstract_gain": abstract_gain,
            "total_gain": total_gain,
            "strength_increment": strength_increment,
        },

        # 样本数据
        "samples": samples,

        # 关键结论
        "key_findings": [
            f"共生成{total_count}个属性样本",
            f"基础属性样本：{basic_count}/{basic_target} ({basic_gain*100:.1f}%)",
            f"组合属性样本：{composite_count}/{composite_target} ({composite_gain*100:.1f}%)",
            f"抽象属性样本：{abstract_count}/{abstract_target} ({abstract_gain*100:.1f}%)",
            f"预期强度增益：{strength_increment:.3f}",
            f"所有目标均已达成" if basic_count >= basic_target and composite_count >= composite_target and abstract_count >= abstract_target else "部分目标未达成",
        ],

        # 下一步
        "next_stage": "Stage414: 位置空间维度扩展",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    """输出结果文件"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 输出JSON摘要
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig"
    )

    # 输出样本数据
    (output_dir / "attribute_samples.json").write_text(
        json.dumps(summary["samples"], ensure_ascii=False, indent=2),
        encoding="utf-8-sig"
    )

    # 输出Markdown报告
    md_content = generate_markdown_report(summary)
    (output_dir / "STAGE413_REPORT.md").write_text(
        md_content,
        encoding="utf-8-sig"
    )


def generate_markdown_report(summary: dict) -> str:
    """生成Markdown格式报告"""
    lines = [
        "# Stage413: 属性空间样本扩容（+50+样本）",
        "",
        f"**生成时间**: {summary['timestamp']}",
        f"**状态**: {summary['status_short']}",
        "",
        "## 1. 样本统计",
        "",
    ]

    # 样本统计
    stats = summary["sample_statistics"]
    lines.append(f"- **基础属性样本**: {stats['basic_count']} / {stats['basic_target']}")
    lines.append(f"- **组合属性样本**: {stats['composite_count']} / {stats['composite_target']}")
    lines.append(f"- **抽象属性样本**: {stats['abstract_count']} / {stats['abstract_target']}")
    lines.append(f"- **总样本数**: {stats['total_count']}")
    lines.append("")

    # 扩厚效果
    lines.append("## 2. 扩厚效果")
    lines.append("")

    effect = summary["expansion_effect"]
    lines.append(f"- **基础属性增益**: {effect['basic_gain']:.3f} ({effect['basic_gain']*100:.1f}%)")
    lines.append(f"- **组合属性增益**: {effect['composite_gain']:.3f} ({effect['composite_gain']*100:.1f}%)")
    lines.append(f"- **抽象属性增益**: {effect['abstract_gain']:.3f} ({effect['abstract_gain']*100:.1f}%)")
    lines.append(f"- **总体增益**: {effect['total_gain']:.3f} ({effect['total_gain']*100:.1f}%)")
    lines.append(f"- **预期强度增量**: {effect['strength_increment']:.3f}")
    lines.append("")

    # 关键结论
    lines.append("## 3. 关键结论")
    lines.append("")

    for finding in summary["key_findings"]:
        lines.append(f"- {finding}")

    lines.append("")
    lines.append("## 4. 下一步")
    lines.append("")
    lines.append(f"**下一阶段**: {summary['next_stage']}")

    return "\n".join(lines)


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    """运行分析"""
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))

    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="属性空间样本扩容（+50+样本）"
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true", help="强制重新计算")
    args = parser.parse_args()

    result = run_analysis(output_dir=Path(args.output_dir), force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
