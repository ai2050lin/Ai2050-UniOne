from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tests" / "codex_temp" / "stage396_research_asset_neuron_display_scheme_20260325"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
MANIFEST_PATH = OUTPUT_DIR / "research_asset_neuron_display_manifest.json"


def build_manifest() -> dict:
    return {
        "scheme_name": "research_asset_neuron_display_scheme",
        "date": "2026-03-25",
        "base_principle": "所有神经元显示都建立在当前28个layer主视图之上",
        "reuse_from_existing_system": [
            "研究资产文件选择",
            "导入并映射到3D按钮链",
            "研究资产预览面板",
            "原始JSON查看",
            "理论说明与3D关注点说明",
        ],
        "system_pipeline": [
            {
                "id": "asset_import",
                "label": "研究资产导入",
                "goal": "复用当前研究资产与3D映射的文件导入入口",
                "source_inputs": [
                    "mass noun 扫描结果",
                    "多维编码探针",
                    "硬伤实验结果",
                    "统一解码结果",
                    "层级原始场景导出",
                ],
                "ui_actions": [
                    "选择文件",
                    "刷新列表",
                    "导入并映射到3D",
                ],
            },
            {
                "id": "runtime_mapping",
                "label": "神经元运行映射",
                "goal": "把研究资产里的原始数据压到当前28个layer主视图",
                "required_fields": [
                    "layer_index",
                    "neuron",
                    "dim_index",
                    "activation_value",
                    "parameter_ids",
                    "source_stage",
                ],
                "fallback_fields": [
                    "早层",
                    "中层",
                    "后层",
                    "source_dim_index",
                    "carrier_dim",
                    "bias_dim",
                ],
                "mapping_rules": [
                    "显式layer_index优先",
                    "显式layer_index缺失时才使用早中后粗带",
                    "神经元点必须附着在当前layer上",
                    "抽象结构不能替代神经元点",
                ],
            },
            {
                "id": "neuron_rendering",
                "label": "参数级神经元显示",
                "goal": "在layer基础上显示真实神经元和参数位",
                "visible_elements": [
                    "有效神经元点",
                    "参数位节点",
                    "参数位标签",
                    "神经元到参数位连线",
                ],
                "display_priority": [
                    "layer骨架",
                    "神经元点",
                    "参数位节点",
                    "联动连线",
                ],
            },
            {
                "id": "analysis_overlay",
                "label": "高级分析叠加",
                "goal": "在神经元可视基础上叠加当前研发进展中的结构分析",
                "overlays": [
                    "共享承载",
                    "偏置偏转",
                    "逐层放大",
                    "语义角色",
                ],
                "rules": [
                    "默认关闭",
                    "只能叠加",
                    "不能替代layer主视图",
                ],
            },
            {
                "id": "detail_panel",
                "label": "右侧数据分析面板",
                "goal": "点击神经元后直接显示基于数据的分析结果",
                "panel_sections": [
                    "节点基本信息",
                    "parameter_ids",
                    "原始行",
                    "来源阶段",
                    "输出目录",
                    "研究资产原始JSON片段",
                ],
            },
        ],
        "button_design": {
            "buttons": [
                {
                    "id": "start_animation",
                    "label": "开始动画",
                    "effect": [
                        "先点亮全部28个layer骨架",
                        "再按当前研究资产的layer顺序点亮有效神经元",
                        "再显示神经元到参数位的联动线",
                        "最后叠加共享承载和偏置偏转等高级分析层",
                    ],
                },
                {
                    "id": "end_animation",
                    "label": "结束动画",
                    "effect": [
                        "停止所有脉冲和流动线",
                        "保留当前神经元与参数位静态状态",
                        "恢复到可阅读的分析视图",
                    ],
                },
                {
                    "id": "replay_animation",
                    "label": "重新播放",
                    "effect": [
                        "重新从第1层开始播放",
                        "清空中间脉冲轨迹",
                        "重新按当前资产顺序回放",
                    ],
                },
            ],
        },
        "main_display_modes": [
            {
                "id": "runtime_focus",
                "label": "运行机制模式",
                "description": "优先看layer和神经元怎么运行",
            },
            {
                "id": "parameter_focus",
                "label": "参数定位模式",
                "description": "优先看参数位、参数清单和来源阶段",
            },
            {
                "id": "analysis_focus",
                "label": "分析叠加模式",
                "description": "在运行机制已可见的前提下叠加共享承载、偏置偏转和逐层放大",
            },
        ],
        "visual_priority": [
            "先layer",
            "再神经元",
            "再参数位",
            "最后高级分析层",
        ],
    }


def build_summary(manifest: dict) -> dict:
    return {
        "stage": "stage396_research_asset_neuron_display_scheme",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scheme_score": 1.0,
        "pipeline_count": len(manifest["system_pipeline"]),
        "button_count": len(manifest["button_design"]["buttons"]),
        "display_mode_count": len(manifest["main_display_modes"]),
        "summary_rows": [
            {"part": "基础显示", "value": "当前28个layer主视图不变"},
            {"part": "导入入口", "value": "复用研究资产与3D映射现有导入链"},
            {"part": "神经元主线", "value": "layer -> 神经元 -> 参数位 -> 分析叠加"},
            {"part": "动画按钮", "value": "开始动画 / 结束动画 / 重新播放"},
            {"part": "分析面板", "value": "节点基本信息 + 参数位 + 原始行 + 原始JSON"},
        ],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    summary = build_summary(manifest)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
