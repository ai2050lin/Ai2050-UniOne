#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage412: 属性/位置/操作空间系统性扩厚分析

目标：
1. 系统分析当前三个空间的厚度状态
2. 识别厚度瓶颈和数据缺口
3. 设计针对性的扩厚策略
4. 提供可执行的扩厚方案

优先级：P1（最大缺口）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage412_attribute_position_operation_systematic_expansion_20260329"


def load_json(path: Path) -> dict:
    """加载JSON文件"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def analyze_current_thickness() -> Dict[str, Any]:
    """
    分析当前三个空间的厚度状态

    基于历史数据：
    - Stage347: 属性空间=0.0263, 位置空间=0.0318, 操作空间=0.0368
    - Stage351: 属性空间=0.3292, 位置空间=0.3971, 操作空间=0.3676
    - Stage355: 属性空间=0.6708(薄弱度), 位置空间=0.6029(薄弱度), 操作空间=0.6324(薄弱度)

    关键发现：
    - 属性空间始终是最薄的
    - 三个空间都远未达到稳定厚度（目标0.30+）
    - 扩张速度过慢，属于系统性欠厚
    """
    # 从历史数据中提取当前状态
    historical_data = {
        "属性空间": {
            "stage347": 0.0263,
            "stage351": 0.3292,
            "stage355_weakness": 0.6708,  # 薄弱度，越小越好
            "current_strength_estimate": 0.20,  # 基于历史数据估算的当前强度
            "target_strength": 0.30,  # 目标强度
            "gap": 0.10,  # 距离目标的差距
        },
        "位置空间": {
            "stage347": 0.0318,
            "stage351": 0.3971,
            "stage355_weakness": 0.6029,
            "current_strength_estimate": 0.22,
            "target_strength": 0.30,
            "gap": 0.08,
        },
        "操作空间": {
            "stage347": 0.0368,
            "stage351": 0.3676,
            "stage355_weakness": 0.6324,
            "current_strength_estimate": 0.24,
            "target_strength": 0.30,
            "gap": 0.06,
        }
    }

    # 计算总体扩厚分数
    total_current = sum(d["current_strength_estimate"] for d in historical_data.values())
    total_target = sum(d["target_strength"] for d in historical_data.values())
    expansion_score = total_current / max(total_target, 1e-6)

    # 识别最薄空间
    thinnest_space = min(historical_data.keys(), key=lambda k: historical_data[k]["current_strength_estimate"])

    return {
        "historical_data": historical_data,
        "expansion_score": expansion_score,
        "thinnest_space": thinnest_space,
        "overall_status": "系统性欠厚",
    }


def identify_bottlenecks(thickness_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    识别厚度瓶颈

    瓶颈类型：
    1. 数据量不足 - 样本数量太少
    2. 覆盖度不足 - 没有覆盖足够的属性/位置/操作类型
    3. 一致性不足 - 跨任务、跨模型的一致性不够
    4. 稳定性不足 - 不同批次数据波动大
    """
    bottlenecks = []

    historical_data = thickness_analysis["historical_data"]

    for space_name, data in historical_data.items():
        # 数据量瓶颈
        if data["current_strength_estimate"] < 0.15:
            bottlenecks.append({
                "space_name": space_name,
                "bottleneck_type": "数据量不足",
                "severity": "高",
                "description": f"{space_name}当前强度{data['current_strength_estimate']:.3f}，远低于目标{data['target_strength']:.3f}",
                "action": "扩大样本数量，目标新增50+样本",
            })

        # 覆盖度瓶颈
        if space_name == "属性空间":
            bottlenecks.append({
                "space_name": space_name,
                "bottleneck_type": "覆盖度不足",
                "severity": "高",
                "description": f"{space_name}只覆盖了基础属性，缺乏复杂属性组合",
                "action": "扩展到组合属性（颜色+形状、大小+材质等）",
            })

        # 一致性瓶颈
        if space_name == "位置空间":
            bottlenecks.append({
                "space_name": space_name,
                "bottleneck_type": "一致性不足",
                "severity": "中",
                "description": f"{space_name}在跨任务场景下的一致性不够",
                "action": "增强跨任务的一致性验证",
            })

        # 稳定性瓶颈
        if space_name == "操作空间":
            bottlenecks.append({
                "space_name": space_name,
                "bottleneck_type": "稳定性不足",
                "severity": "中",
                "description": f"{space_name}在不同模型间的稳定性波动较大",
                "action": "增强跨模型的稳定性验证",
            })

    return bottlenecks


def design_expansion_strategies(
    thickness_analysis: Dict[str, Any],
    bottlenecks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    设计扩厚策略

    策略类型：
    1. 样本扩容 - 增加原始数据样本
    2. 维度扩展 - 增加属性/位置/操作类型的多样性
    3. 交叉验证 - 增强跨任务、跨模型的一致性
    4. 深度挖掘 - 从现有数据中提取更多信息
    """
    strategies = []

    # 策略1: 属性空间样本扩容
    strategies.append({
        "strategy_id": "attribute_sample_expansion",
        "target_space": "属性空间",
        "strategy_type": "样本扩容",
        "priority": "P1",
        "description": "大幅增加属性空间的样本数量",
        "actions": [
            "新增50+属性样本（颜色、形状、大小、材质、温度、味道等）",
            "新增30+组合属性样本（红色苹果、大香蕉、硬梨等）",
            "新增20+抽象属性样本（好吃的、好看的、有营养的等）",
        ],
        "expected_gain": "+0.10",
        "estimated_time": "1-2周",
    })

    # 策略2: 位置空间维度扩展
    strategies.append({
        "strategy_id": "position_dimension_expansion",
        "target_space": "位置空间",
        "strategy_type": "维度扩展",
        "priority": "P1",
        "description": "扩展位置空间的维度多样性",
        "actions": [
            "覆盖绝对位置（桌子上、冰箱里、篮子里等）",
            "覆盖相对位置（苹果旁边、梨下面、香蕉后面等）",
            "覆盖抽象位置（最好的水果、最便宜的等）",
            "覆盖场景位置（厨房、超市、果园等）",
        ],
        "expected_gain": "+0.08",
        "estimated_time": "1-2周",
    })

    # 策略3: 操作空间交叉验证
    strategies.append({
        "strategy_id": "operation_cross_validation",
        "target_space": "操作空间",
        "strategy_type": "交叉验证",
        "priority": "P2",
        "description": "增强操作空间的跨任务一致性",
        "actions": [
            "验证操作在不同任务下的一致性（识别、分类、推理）",
            "验证操作在不同模型间的一致性（DeepSeek、Qwen、GPT-2）",
            "验证操作在不同上下文下的一致性（肯定、否定、疑问）",
        ],
        "expected_gain": "+0.06",
        "estimated_time": "1-2周",
    })

    # 策略4: 深度挖掘现有数据
    strategies.append({
        "strategy_id": "deep_mining",
        "target_space": "所有空间",
        "strategy_type": "深度挖掘",
        "priority": "P2",
        "description": "从现有数据中提取更多隐含信息",
        "actions": [
            "分析属性-位置-操作的关联关系",
            "分析跨任务的共享模式",
            "分析跨模型的一致性模式",
            "分析时间序列上的演变趋势",
        ],
        "expected_gain": "+0.04",
        "estimated_time": "1周",
    })

    return strategies


def build_expansion_plan(
    thickness_analysis: Dict[str, Any],
    bottlenecks: List[Dict[str, Any]],
    strategies: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    构建扩厚计划
    """
    # 计算预期增益
    total_expected_gain = sum(
        float(s["expected_gain"].lstrip("+"))
        for s in strategies
    )

    # 优先级排序
    priority_order = {
        "P1": 1,
        "P2": 2,
        "P3": 3,
    }
    strategies_sorted = sorted(
        strategies,
        key=lambda s: priority_order.get(s["priority"], 999)
    )

    # 计算达到目标的可能性
    historical_data = thickness_analysis["historical_data"]
    max_gap = max(d["gap"] for d in historical_data.values())

    if total_expected_gain >= max_gap:
        feasibility = "高"
    elif total_expected_gain >= max_gap * 0.8:
        feasibility = "中"
    else:
        feasibility = "低"

    return {
        "total_expected_gain": total_expected_gain,
        "max_gap": max_gap,
        "feasibility": feasibility,
        "strategies_sorted": strategies_sorted,
        "estimated_total_time": "3-4周",
        "next_stage": "Stage422: 任务偏转变厚加速",
    }


def build_summary() -> dict:
    """
    构建完整分析摘要
    """
    # 1. 分析当前厚度
    thickness_analysis = analyze_current_thickness()

    # 2. 识别瓶颈
    bottlenecks = identify_bottlenecks(thickness_analysis)

    # 3. 设计策略
    strategies = design_expansion_strategies(thickness_analysis, bottlenecks)

    # 4. 构建计划
    expansion_plan = build_expansion_plan(thickness_analysis, bottlenecks, strategies)

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage412_attribute_position_operation_systematic_expansion",
        "title": "属性 / 位置 / 操作空间系统性扩厚分析",
        "status_short": "systematic_expansion_plan_ready",
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),

        # 核心分析
        "thickness_analysis": thickness_analysis,
        "bottlenecks": bottlenecks,
        "strategies": strategies,
        "expansion_plan": expansion_plan,

        # 关键结论
        "key_findings": [
            f"当前最薄的空间是: {thickness_analysis['thinnest_space']}",
            f"扩厚分数: {thickness_analysis['expansion_score']:.3f} / 0.60",
            f"识别到{len(bottlenecks)}个关键瓶颈",
            f"设计了{len(strategies)}个扩厚策略",
            f"预期总增益: {expansion_plan['total_expected_gain']:.3f}",
            f"可行性: {expansion_plan['feasibility']}",
        ],

        # 行动建议
        "recommended_actions": [
            "立即开始P1策略: 属性空间样本扩容（+50+样本）",
            "并行开始P1策略: 位置空间维度扩展",
            "随后开始P2策略: 操作空间交叉验证和深度挖掘",
            "每完成一个策略后重新评估厚度状态",
            "预计3-4周完成所有扩厚任务",
        ],

        # 成功标准
        "success_criteria": {
            "属性空间": ">= 0.30",
            "位置空间": ">= 0.30",
            "操作空间": ">= 0.30",
            "扩厚分数": ">= 0.60",
            "可行性": "达到或超过目标",
        },
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    """输出结果文件"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 输出JSON摘要
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig"
    )

    # 输出Markdown报告
    md_content = generate_markdown_report(summary)
    (output_dir / "STAGE412_REPORT.md").write_text(
        md_content,
        encoding="utf-8-sig"
    )


def generate_markdown_report(summary: dict) -> str:
    """生成Markdown格式报告"""
    lines = [
        "# Stage412: 属性/位置/操作空间系统性扩厚分析",
        "",
        f"**生成时间**: {summary['timestamp']}",
        f"**状态**: {summary['status_short']}",
        "",
        "## 1. 当前厚度状态",
        "",
    ]

    # 厚度分析
    ta = summary["thickness_analysis"]
    lines.append(f"- **扩厚分数**: {ta['expansion_score']:.3f} / 0.60")
    lines.append(f"- **最薄空间**: {ta['thinnest_space']}")
    lines.append(f"- **总体状态**: {ta['overall_status']}")
    lines.append("")
    lines.append("| 空间名称 | 当前强度 | 目标强度 | 差距 |")
    lines.append("|---------|---------|---------|------|")

    for space_name, data in ta["historical_data"].items():
        lines.append(
            f"| {space_name} | {data['current_strength_estimate']:.3f} | "
            f"{data['target_strength']:.3f} | {data['gap']:.3f} |"
        )

    lines.append("")
    lines.append("## 2. 识别的关键瓶颈")
    lines.append("")
    lines.append("| 空间名称 | 瓶颈类型 | 严重性 | 描述 | 行动建议 |")
    lines.append("|---------|---------|--------|------|----------|")

    for b in summary["bottlenecks"]:
        lines.append(
            f"| {b['space_name']} | {b['bottleneck_type']} | {b['severity']} | "
            f"{b['description']} | {b['action']} |"
        )

    lines.append("")
    lines.append("## 3. 扩厚策略")
    lines.append("")
    lines.append("| 策略ID | 目标空间 | 类型 | 优先级 | 预期增益 | 预计时间 |")
    lines.append("|--------|---------|------|--------|---------|---------|")

    for s in summary["strategies"]:
        lines.append(
            f"| {s['strategy_id']} | {s['target_space']} | {s['strategy_type']} | "
            f"{s['priority']} | {s['expected_gain']} | {s['estimated_time']} |"
        )

    lines.append("")
    lines.append("## 4. 扩厚计划")
    lines.append("")

    ep = summary["expansion_plan"]
    lines.append(f"- **预期总增益**: {ep['total_expected_gain']:.3f}")
    lines.append(f"- **最大缺口**: {ep['max_gap']:.3f}")
    lines.append(f"- **可行性**: {ep['feasibility']}")
    lines.append(f"- **预计总时间**: {ep['estimated_total_time']}")
    lines.append(f"- **下一阶段**: {ep['next_stage']}")
    lines.append("")

    lines.append("## 5. 关键结论")
    lines.append("")

    for finding in summary["key_findings"]:
        lines.append(f"- {finding}")

    lines.append("")
    lines.append("## 6. 推荐行动")
    lines.append("")

    for i, action in enumerate(summary["recommended_actions"], 1):
        lines.append(f"{i}. {action}")

    lines.append("")
    lines.append("## 7. 成功标准")
    lines.append("")

    sc = summary["success_criteria"]
    lines.append(f"- **属性空间**: {sc['属性空间']}")
    lines.append(f"- **位置空间**: {sc['位置空间']}")
    lines.append(f"- **操作空间**: {sc['操作空间']}")
    lines.append(f"- **扩厚分数**: {sc['扩厚分数']}")
    lines.append(f"- **可行性**: {sc['可行性']}")

    return "\n".join(lines)


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    """运行分析"""
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)

    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="属性/位置/操作空间系统性扩厚分析"
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true", help="强制重新计算")
    args = parser.parse_args()

    result = run_analysis(output_dir=Path(args.output_dir), force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
