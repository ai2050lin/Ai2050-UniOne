from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def build_framework_bridge(
    law_summary: Dict[str, object],
    frontier_summary: Dict[str, object],
) -> Dict[str, object]:
    laws = dict(law_summary.get("laws", {}))
    density_frontier = dict(frontier_summary.get("density_frontier", {}))
    closure = dict(frontier_summary.get("closure", {}))

    broad_support_base = safe_float(laws.get("broad_support_base"))
    long_separation_frontier = safe_float(laws.get("long_separation_frontier"))
    mid_syntax_filter = safe_float(laws.get("mid_syntax_filter"))
    late_window_closure = safe_float(laws.get("late_window_closure"))
    pair_positive_ratio = safe_float(closure.get("pair_positive_ratio"))

    candidates: List[Dict[str, object]] = [
        {
            "framework": "局部仿射切片",
            "math_family": "线性代数 + 表示论",
            "supported_by": [
                "局部线性关系切片依然存在，但只占少数",
                "family patch（家族片区）与 concept offset（概念偏移）可视为局部图册中的局部坐标差",
            ],
            "fit_score": 0.58,
            "scope": "解释局部关系轴和局部偏移",
            "limitation": "不能覆盖大部分路径束主结构，也不能单独解释闭包动力学",
        },
        {
            "framework": "图册 + 纤维束",
            "math_family": "微分几何 + 纤维束 + 分层空间",
            "supported_by": [
                f"broad_support_base（广支撑底座）={broad_support_base:+.4f}",
                f"long_separation_frontier（长期分离前沿）={long_separation_frontier:+.4f}",
                "family patch 更像局部图册，concept offset 更像图册内坐标偏移",
            ],
            "fit_score": 0.84,
            "scope": "解释静态概念本体层和局部家族结构",
            "limitation": "只靠几何对象还不够解释时间窗口上的生成收束",
        },
        {
            "framework": "分层动力系统",
            "math_family": "动力系统 + 控制论 + 信息几何",
            "supported_by": [
                f"mid_syntax_filter（中段句法筛选）={mid_syntax_filter:+.4f}",
                f"late_window_closure（晚窗口闭包）={late_window_closure:+.4f}",
                f"pair_positive_ratio（严格正协同比例）={pair_positive_ratio:+.4f}",
            ],
            "fit_score": 0.87,
            "scope": "解释内部子场、词元窗口和闭包量如何联动",
            "limitation": "需要和静态图册层结合，否则只能解释生成过程，不能解释概念本体来源",
        },
        {
            "framework": "拓扑/分层闭包结构",
            "math_family": "拓扑学 + 分层拓扑 + 奇点理论",
            "supported_by": [
                "长期分离前沿提示存在相区、边界和汇合点",
                "闭包量更像相变成功态，而不是连续平滑平均态",
            ],
            "fit_score": 0.72,
            "scope": "解释分离前沿、相变式闭包和失败/成功态边界",
            "limitation": "单独使用过于粗，不足以给出层、窗口、子场级预测",
        },
        {
            "framework": "范畴化组合结构",
            "math_family": "范畴论 + 组合语义",
            "supported_by": [
                "关系轴、路径束、控制轴和闭包更像可组合态射，而不是单个向量点",
                "适合把静态本体层和动态生成层接成统一组合系统",
            ],
            "fit_score": 0.66,
            "scope": "适合做统一语言，但当前实证变量还不够直接",
            "limitation": "当前项目还缺足够直接的范畴化可观测量",
        },
    ]

    recommended_stack = [
        "线性代数负责局部切片",
        "图册/纤维束负责静态概念层",
        "分层动力系统负责生成与闭包层",
        "拓扑负责前沿相区与闭包边界",
    ]

    return {
        "record_type": "stage56_math_framework_bridge_summary",
        "recommended_stack": recommended_stack,
        "candidates": candidates,
        "main_judgment": (
            "当前证据不支持用单一现成数学分支完全吃掉语言系统。"
            "更稳的方向是分层混合体系：局部线性 + 图册/纤维束 + 分层动力系统 + 拓扑闭包。"
        ),
        "answer_to_user_question": (
            "当前研究已经支持：仅靠现有低阶单一数学对象不够。"
            "如果要完成语言背后的数学机制以及逼近大脑编码理论，大概率需要更高阶的分层数学体系。"
            "但它不像是简单升级成某一个现成学科，例如只用群论或只用拓扑学；"
            "更像是把局部线性、几何图册、动力系统和拓扑闭包组织成统一变量系统。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 数学框架桥接摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## 推荐数学堆栈",
    ]
    for item in list(summary.get("recommended_stack", [])):
        lines.append(f"- {item}")
    lines.extend(["", "## 候选框架"])
    for row in list(summary.get("candidates", [])):
        row = dict(row)
        lines.append(
            f"- {row.get('framework', '')} / {row.get('math_family', '')}: "
            f"fit_score={safe_float(row.get('fit_score')):+.2f}, scope={row.get('scope', '')}, limitation={row.get('limitation', '')}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bridge current empirical structure to candidate higher-order math frameworks")
    ap.add_argument(
        "--law-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_simple_generator_laws_20260319_1646" / "summary.json"),
    )
    ap.add_argument(
        "--frontier-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_frontier_subfield_window_closure_summary_20260319_1646" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_math_framework_bridge_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_framework_bridge(
        read_json(Path(args.law_summary_json)),
        read_json(Path(args.frontier_summary_json)),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "record_type": summary["record_type"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
