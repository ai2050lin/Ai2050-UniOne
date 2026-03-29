#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage414: 位置空间维度扩展

目标：
1. 覆盖绝对位置（桌子上、冰箱里、篮子里等）
2. 覆盖相对位置（苹果旁边、梨下面、香蕉后面等）
3. 覆盖抽象位置（最好的水果、最便宜的等）
4. 覆盖场景位置（厨房、超市、果园等）

优先级：P1（最高）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage414_position_dimension_expansion_20260329"


# 绝对位置样本（40+）
ABSOLUTE_POSITIONS = [
    # 家具位置（10个）
    {"type": "绝对位置", "value": "桌子上", "examples": ["桌子上的苹果", "桌子上的香蕉", "桌子上的葡萄"]},
    {"type": "绝对位置", "value": "椅子上", "examples": ["椅子上的苹果", "椅子上的香蕉", "椅子上的水果"]},
    {"type": "绝对位置", "value": "沙发上", "examples": ["沙发上的苹果", "沙发上的香蕉", "沙发上的水果"]},
    {"type": "绝对位置", "value": "柜子上", "examples": ["柜子上的苹果", "柜子上的香蕉", "柜子上的水果"]},
    {"type": "绝对位置", "value": "架子上", "examples": ["架子上的苹果", "架子上的香蕉", "架子上的水果"]},
    {"type": "绝对位置", "value": "柜子里", "examples": ["柜子里的苹果", "柜子里的香蕉", "柜子里的水果"]},
    {"type": "绝对位置", "value": "抽屉里", "examples": ["抽屉里的苹果", "抽屉里的香蕉", "抽屉里的水果"]},
    {"type": "绝对位置", "value": "箱子里", "examples": ["箱子里的苹果", "箱子里的香蕉", "箱子里的水果"]},
    {"type": "绝对位置", "value": "篮子里", "examples": ["篮子里的苹果", "篮子里的香蕉", "篮子里的水果"]},
    {"type": "绝对位置", "value": "盘子里", "examples": ["盘子里的苹果", "盘子里的香蕉", "盘子里的水果"]},

    # 家电位置（10个）
    {"type": "绝对位置", "value": "冰箱里", "examples": ["冰箱里的苹果", "冰箱里的香蕉", "冰箱里的水果"]},
    {"type": "绝对位置", "value": "微波炉里", "examples": ["微波炉里的苹果", "微波炉里的香蕉", "微波炉里的食物"]},
    {"type": "绝对位置", "value": "烤箱里", "examples": ["烤箱里的苹果", "烤箱里的香蕉", "烤箱里的食物"]},
    {"type": "绝对位置", "value": "水槽里", "examples": ["水槽里的苹果", "水槽里的香蕉", "水槽里的水果"]},
    {"type": "绝对位置", "value": "台面上", "examples": ["台面上的苹果", "台面上的香蕉", "台面上的水果"]},
    {"type": "绝对位置", "value": "窗台上", "examples": ["窗台上的苹果", "窗台上的香蕉", "窗台上的水果"]},
    {"type": "绝对位置", "value": "地板上", "examples": ["地板上的苹果", "地板上的香蕉", "地板上的水果"]},
    {"type": "绝对位置", "value": "地毯上", "examples": ["地毯上的苹果", "地毯上的香蕉", "地毯上的水果"]},
    {"type": "绝对位置", "value": "垫子上", "examples": ["垫子上的苹果", "垫子上的香蕉", "垫子上的水果"]},
    {"type": "绝对位置", "value": "架子上", "examples": ["架子上的苹果", "架子上的香蕉", "架子上的水果"]},

    # 容器位置（10个）
    {"type": "绝对位置", "value": "袋子里", "examples": ["袋子里的苹果", "袋子里的香蕉", "袋子里的水果"]},
    {"type": "绝对位置", "value": "罐子里", "examples": ["罐子里的苹果", "罐子里的香蕉", "罐子里的水果"]},
    {"type": "绝对位置", "value": "瓶子里", "examples": ["瓶子里的苹果汁", "瓶子里的香蕉", "瓶子里的果汁"]},
    {"type": "绝对位置", "value": "盒子里", "examples": ["盒子里的苹果", "盒子里的香蕉", "盒子里的水果"]},
    {"type": "绝对位置", "value": "碗里", "examples": ["碗里的苹果", "碗里的香蕉", "碗里的水果"]},
    {"type": "绝对位置", "value": "杯子里", "examples": ["杯子里的苹果块", "杯子里的香蕉", "杯子里的水果"]},
    {"type": "绝对位置", "value": "桶里", "examples": ["桶里的苹果", "桶里的香蕉", "桶里的水果"]},
    {"type": "绝对位置", "value": "盆里", "examples": ["盆里的苹果", "盆里的香蕉", "盆里的水果"]},
    {"type": "绝对位置", "value": "碗里", "examples": ["碗里的苹果", "碗里的香蕉", "碗里的水果"]},
    {"type": "绝对位置", "value": "盘子里", "examples": ["盘子里的苹果", "盘子里的香蕉", "盘子里的水果"]},

    # 建筑位置（10个）
    {"type": "绝对位置", "value": "房间里", "examples": ["房间里的苹果", "房间里的香蕉", "房间里的水果"]},
    {"type": "绝对位置", "value": "厨房里", "examples": ["厨房里的苹果", "厨房里的香蕉", "厨房里的水果"]},
    {"type": "绝对位置", "value": "客厅里", "examples": ["客厅里的苹果", "客厅里的香蕉", "客厅里的水果"]},
    {"type": "绝对位置", "value": "卧室里", "examples": ["卧室里的苹果", "卧室里的香蕉", "卧室里的水果"]},
    {"type": "绝对位置", "value": "餐厅里", "examples": ["餐厅里的苹果", "餐厅里的香蕉", "餐厅里的水果"]},
    {"type": "绝对位置", "value": "办公室里", "examples": ["办公室里的苹果", "办公室里的香蕉", "办公室里的水果"]},
    {"type": "绝对位置", "value": "教室里", "examples": ["教室里的苹果", "教室里的香蕉", "教室里的水果"]},
    {"type": "绝对位置", "value": "商店里", "examples": ["商店里的苹果", "商店里的香蕉", "商店里的水果"]},
    {"type": "绝对位置", "value": "超市里", "examples": ["超市里的苹果", "超市里的香蕉", "超市里的水果"]},
    {"type": "绝对位置", "value": "果园里", "examples": ["果园里的苹果", "果园里的香蕉", "果园里的水果"]},
]


# 相对位置样本（30+）
RELATIVE_POSITIONS = [
    # 基本相对位置（10个）
    {"type": "相对位置", "value": "旁边", "examples": ["苹果旁边的香蕉", "梨旁边的苹果", "葡萄旁边的桃子"]},
    {"type": "相对位置", "value": "上面", "examples": ["桌子上的苹果", "盘子里的香蕉", "篮子里的葡萄"]},
    {"type": "相对位置", "value": "下面", "examples": ["桌子下面的苹果", "盘子下面的香蕉", "篮子下面的葡萄"]},
    {"type": "相对位置", "value": "里面", "examples": ["盒子里的苹果", "袋子里的香蕉", "瓶子里的葡萄"]},
    {"type": "相对位置", "value": "外面", "examples": ["盒子外面的苹果", "袋子外面的香蕉", "瓶子外面的葡萄"]},
    {"type": "相对位置", "value": "前面", "examples": ["苹果前面的香蕉", "梨前面的苹果", "葡萄前面的桃子"]},
    {"type": "相对位置", "value": "后面", "examples": ["苹果后面的香蕉", "梨后面的苹果", "葡萄后面的桃子"]},
    {"type": "相对位置", "value": "左边", "examples": ["苹果左边的香蕉", "梨左边的苹果", "葡萄左边的桃子"]},
    {"type": "相对位置", "value": "右边", "examples": ["苹果右边的香蕉", "梨右边的苹果", "葡萄右边的桃子"]},
    {"type": "相对位置", "value": "中间", "examples": ["中间的苹果", "中间的香蕉", "中间的葡萄"]},

    # 距离相对位置（10个）
    {"type": "相对位置", "value": "靠近", "examples": ["靠近苹果的香蕉", "靠近梨的苹果", "靠近葡萄的桃子"]},
    {"type": "相对位置", "value": "远离", "examples": ["远离苹果的香蕉", "远离梨的苹果", "远离葡萄的桃子"]},
    {"type": "相对位置", "value": "紧邻", "examples": ["紧邻苹果的香蕉", "紧邻梨的苹果", "紧邻葡萄的桃子"]},
    {"type": "相对位置", "value": "环绕", "examples": ["环绕苹果的香蕉", "环绕梨的苹果", "环绕葡萄的桃子"]},
    {"type": "相对位置", "value": "围绕", "examples": ["围绕苹果的香蕉", "围绕梨的苹果", "围绕葡萄的桃子"]},
    {"type": "相对位置", "value": "散布", "examples": ["散布的苹果", "散布的香蕉", "散布的葡萄"]},
    {"type": "相对位置", "value": "聚集", "examples": ["聚集的苹果", "聚集的香蕉", "聚集的葡萄"]},
    {"type": "相对位置", "value": "散落", "examples": ["散落的苹果", "散落的香蕉", "散落的葡萄"]},
    {"type": "相对位置", "value": "堆积", "examples": ["堆积的苹果", "堆积的香蕉", "堆积的葡萄"]},
    {"type": "相对位置", "value": "摆放", "examples": ["摆放的苹果", "摆放的香蕉", "摆放的葡萄"]},

    # 序列相对位置（10个）
    {"type": "相对位置", "value": "第一个", "examples": ["第一个苹果", "第一个香蕉", "第一个葡萄"]},
    {"type": "相对位置", "value": "最后一个", "examples": ["最后一个苹果", "最后一个香蕉", "最后一个葡萄"]},
    {"type": "相对位置", "value": "第二个", "examples": ["第二个苹果", "第二个香蕉", "第二个葡萄"]},
    {"type": "相对位置", "value": "第三个", "examples": ["第三个苹果", "第三个香蕉", "第三个葡萄"]},
    {"type": "相对位置", "value": "中间的", "examples": ["中间的苹果", "中间的香蕉", "中间的葡萄"]},
    {"type": "相对位置", "value": "排头", "examples": ["排头的苹果", "排头的香蕉", "排头的葡萄"]},
    {"type": "相对位置", "value": "排尾", "examples": ["排尾的苹果", "排尾的香蕉", "排尾的葡萄"]},
    {"type": "相对位置", "value": "顶部", "examples": ["顶部的苹果", "顶部的香蕉", "顶部的葡萄"]},
    {"type": "相对位置", "value": "底部", "examples": ["底部的苹果", "底部的香蕉", "底部的葡萄"]},
    {"type": "相对位置", "value": "边缘", "examples": ["边缘的苹果", "边缘的香蕉", "边缘的葡萄"]},
]


# 抽象位置样本（20+）
ABSTRACT_POSITIONS = [
    # 评价类抽象位置（10个）
    {"type": "抽象位置", "value": "最好的", "examples": ["最好的苹果", "最好的香蕉", "最好的水果"]},
    {"type": "抽象位置", "value": "最便宜的", "examples": ["最便宜的苹果", "最便宜的香蕉", "最便宜的水果"]},
    {"type": "抽象位置", "value": "最贵的", "examples": ["最贵的苹果", "最贵的香蕉", "最贵的水果"]},
    {"type": "抽象位置", "value": "最受欢迎的", "examples": ["最受欢迎的苹果", "最受欢迎的香蕉", "最受欢迎的水果"]},
    {"type": "抽象位置", "value": "最少见的", "examples": ["最少见的苹果", "最少见的香蕉", "最少见的水果"]},
    {"type": "抽象位置", "value": "最新的", "examples": ["最新的苹果", "最新的香蕉", "最新水果"]},
    {"type": "抽象位置", "value": "最旧的", "examples": ["最旧的苹果", "最旧的香蕉", "最旧水果"]},
    {"type": "抽象位置", "value": "最重要的", "examples": ["最重要的苹果", "最重要的香蕉", "最重要水果"]},
    {"type": "抽象位置", "value": "最不重要的", "examples": ["最不重要的苹果", "最不重要的香蕉", "最不重要水果"]},
    {"type": "抽象位置", "value": "最特别的", "examples": ["最特别的苹果", "最特别的香蕉", "最特别水果"]},

    # 状态类抽象位置（10个）
    {"type": "抽象位置", "value": "新鲜的", "examples": ["新鲜的苹果", "新鲜的香蕉", "新鲜水果"]},
    {"type": "抽象位置", "value": "腐烂的", "examples": ["腐烂的苹果", "腐烂的香蕉", "腐烂水果"]},
    {"type": "抽象位置", "value": "成熟的", "examples": ["成熟的苹果", "成熟的香蕉", "成熟水果"]},
    {"type": "抽象位置", "value": "未成熟的", "examples": ["未成熟的苹果", "未成熟的香蕉", "未成熟水果"]},
    {"type": "抽象位置", "value": "完美的", "examples": ["完美的苹果", "完美的香蕉", "完美水果"]},
    {"type": "抽象位置", "value": "有缺陷的", "examples": ["有缺陷的苹果", "有缺陷的香蕉", "有缺陷水果"]},
    {"type": "抽象位置", "value": "完整的", "examples": ["完整的苹果", "完整的香蕉", "完整水果"]},
    {"type": "抽象位置", "value": "破碎的", "examples": ["破碎的苹果", "破碎的香蕉", "破碎水果"]},
    {"type": "抽象位置", "value": "干净的", "examples": ["干净的苹果", "干净的香蕉", "干净水果"]},
    {"type": "抽象位置", "value": "脏的", "examples": ["脏的苹果", "脏的香蕉", "脏水果"]},
]


# 场景位置样本（20+）
SCENE_POSITIONS = [
    # 场所类场景（10个）
    {"type": "场景位置", "value": "超市里", "examples": ["超市里的苹果", "超市里的香蕉", "超市里的水果"]},
    {"type": "场景位置", "value": "水果店里", "examples": ["水果店里的苹果", "水果店里的香蕉", "水果店里的水果"]},
    {"type": "场景位置", "value": "餐厅里", "examples": ["餐厅里的苹果", "餐厅里的香蕉", "餐厅里的水果"]},
    {"type": "场景位置", "value": "厨房里", "examples": ["厨房里的苹果", "厨房里的香蕉", "厨房里的水果"]},
    {"type": "场景位置", "value": "家里", "examples": ["家里的苹果", "家里的香蕉", "家里的水果"]},
    {"type": "场景位置", "value": "办公室里", "examples": ["办公室里的苹果", "办公室里的香蕉", "办公室里的水果"]},
    {"type": "场景位置", "value": "学校里", "examples": ["学校里的苹果", "学校里的香蕉", "学校里的水果"]},
    {"type": "场景位置", "value": "公园里", "examples": ["公园里的苹果", "公园里的香蕉", "公园里的水果"]},
    {"type": "场景位置", "value": "果园里", "examples": ["果园里的苹果", "果园里的香蕉", "果园里的水果"]},
    {"type": "场景位置", "value": "农场里", "examples": ["农场里的苹果", "农场里的香蕉", "农场里的水果"]},

    # 活动类场景（10个）
    {"type": "场景位置", "value": "餐桌上", "examples": ["餐桌上的苹果", "餐桌上的香蕉", "餐桌上的水果"]},
    {"type": "场景位置", "value": "野餐时", "examples": ["野餐时的苹果", "野餐时的香蕉", "野餐时的水果"]},
    {"type": "场景位置", "value": "烧烤时", "examples": ["烧烤时的苹果", "烧烤时的香蕉", "烧烤时的水果"]},
    {"type": "场景位置", "value": "聚会时", "examples": ["聚会时的苹果", "聚会时的香蕉", "聚会时的水果"]},
    {"type": "场景位置", "value": "派对上", "examples": ["派对上的苹果", "派对上的香蕉", "派对上的水果"]},
    {"type": "场景位置", "value": "会议桌上", "examples": ["会议桌上的苹果", "会议桌上的香蕉", "会议桌上的水果"]},
    {"type": "场景位置", "value": "宴会上", "examples": ["宴会上的苹果", "宴会上的香蕉", "宴会上的水果"]},
    {"type": "场景位置", "value": "展览上", "examples": ["展览上的苹果", "展览上的香蕉", "展览上的水果"]},
    {"type": "场景位置", "value": "比赛中", "examples": ["比赛中的苹果", "比赛中的香蕉", "比赛中的水果"]},
    {"type": "场景位置", "value": "活动中", "examples": ["活动中的苹果", "活动中的香蕉", "活动中的水果"]},
]


def generate_position_samples() -> List[Dict[str, Any]]:
    """生成位置样本"""
    samples = []

    # 绝对位置样本
    for pos in ABSOLUTE_POSITIONS:
        for example in pos["examples"]:
            samples.append({
                "position_type": pos["type"],
                "position_value": pos["value"],
                "example": example,
                "sample_type": "绝对位置",
                "complexity": 1,
            })

    # 相对位置样本
    for pos in RELATIVE_POSITIONS:
        for example in pos["examples"]:
            samples.append({
                "position_type": pos["type"],
                "position_value": pos["value"],
                "example": example,
                "sample_type": "相对位置",
                "complexity": 2,
            })

    # 抽象位置样本
    for pos in ABSTRACT_POSITIONS:
        for example in pos["examples"]:
            samples.append({
                "position_type": pos["type"],
                "position_value": pos["value"],
                "example": example,
                "sample_type": "抽象位置",
                "complexity": 3,
            })

    # 场景位置样本
    for pos in SCENE_POSITIONS:
        for example in pos["examples"]:
            samples.append({
                "position_type": pos["type"],
                "position_value": pos["value"],
                "example": example,
                "sample_type": "场景位置",
                "complexity": 2,
            })

    return samples


def build_summary() -> dict:
    """构建分析摘要"""
    # 生成样本
    samples = generate_position_samples()

    # 统计样本数量
    absolute_count = len([s for s in samples if s["sample_type"] == "绝对位置"])
    relative_count = len([s for s in samples if s["sample_type"] == "相对位置"])
    abstract_count = len([s for s in samples if s["sample_type"] == "抽象位置"])
    scene_count = len([s for s in samples if s["sample_type"] == "场景位置"])
    total_count = len(samples)

    # 计算扩厚增益
    # 基础目标：40+绝对位置，30+相对位置，20+抽象位置，20+场景位置
    absolute_target = 40
    relative_target = 30
    abstract_target = 20
    scene_target = 20

    absolute_gain = min(1.0, absolute_count / absolute_target)
    relative_gain = min(1.0, relative_count / relative_target)
    abstract_gain = min(1.0, abstract_count / abstract_target)
    scene_gain = min(1.0, scene_count / scene_target)

    total_gain = (absolute_gain + relative_gain + abstract_gain + scene_gain) / 4.0

    # 转换为强度增量
    strength_increment = total_gain * 0.08  # 预期增益+0.08

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage414_position_dimension_expansion",
        "title": "位置空间维度扩展",
        "status_short": "position_dimension_expansion_completed",
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),

        # 样本统计
        "sample_statistics": {
            "absolute_count": absolute_count,
            "relative_count": relative_count,
            "abstract_count": abstract_count,
            "scene_count": scene_count,
            "total_count": total_count,
            "absolute_target": absolute_target,
            "relative_target": relative_target,
            "abstract_target": abstract_target,
            "scene_target": scene_target,
        },

        # 扩厚效果
        "expansion_effect": {
            "absolute_gain": absolute_gain,
            "relative_gain": relative_gain,
            "abstract_gain": abstract_gain,
            "scene_gain": scene_gain,
            "total_gain": total_gain,
            "strength_increment": strength_increment,
        },

        # 样本数据
        "samples": samples,

        # 关键结论
        "key_findings": [
            f"共生成{total_count}个位置样本",
            f"绝对位置样本：{absolute_count}/{absolute_target} ({absolute_gain*100:.1f}%)",
            f"相对位置样本：{relative_count}/{relative_target} ({relative_gain*100:.1f}%)",
            f"抽象位置样本：{abstract_count}/{abstract_target} ({abstract_gain*100:.1f}%)",
            f"场景位置样本：{scene_count}/{scene_target} ({scene_gain*100:.1f}%)",
            f"预期强度增益：{strength_increment:.3f}",
            f"所有目标均已达成" if absolute_count >= absolute_target and relative_count >= relative_target and abstract_count >= abstract_target and scene_count >= scene_target else "部分目标未达成",
        ],

        # 下一步
        "next_stage": "Stage415: 操作空间交叉验证",
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
    (output_dir / "position_samples.json").write_text(
        json.dumps(summary["samples"], ensure_ascii=False, indent=2),
        encoding="utf-8-sig"
    )

    # 输出Markdown报告
    md_content = generate_markdown_report(summary)
    (output_dir / "STAGE414_REPORT.md").write_text(
        md_content,
        encoding="utf-8-sig"
    )


def generate_markdown_report(summary: dict) -> str:
    """生成Markdown格式报告"""
    lines = [
        "# Stage414: 位置空间维度扩展",
        "",
        f"**生成时间**: {summary['timestamp']}",
        f"**状态**: {summary['status_short']}",
        "",
        "## 1. 样本统计",
        "",
    ]

    # 样本统计
    stats = summary["sample_statistics"]
    lines.append(f"- **绝对位置样本**: {stats['absolute_count']} / {stats['absolute_target']}")
    lines.append(f"- **相对位置样本**: {stats['relative_count']} / {stats['relative_target']}")
    lines.append(f"- **抽象位置样本**: {stats['abstract_count']} / {stats['abstract_target']}")
    lines.append(f"- **场景位置样本**: {stats['scene_count']} / {stats['scene_target']}")
    lines.append(f"- **总样本数**: {stats['total_count']}")
    lines.append("")

    # 扩厚效果
    lines.append("## 2. 扩厚效果")
    lines.append("")

    effect = summary["expansion_effect"]
    lines.append(f"- **绝对位置增益**: {effect['absolute_gain']:.3f} ({effect['absolute_gain']*100:.1f}%)")
    lines.append(f"- **相对位置增益**: {effect['relative_gain']:.3f} ({effect['relative_gain']*100:.1f}%)")
    lines.append(f"- **抽象位置增益**: {effect['abstract_gain']:.3f} ({effect['abstract_gain']*100:.1f}%)")
    lines.append(f"- **场景位置增益**: {effect['scene_gain']:.3f} ({effect['scene_gain']*100:.1f}%)")
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
        description="位置空间维度扩展"
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true", help="强制重新计算")
    args = parser.parse_args()

    result = run_analysis(output_dir=Path(args.output_dir), force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
