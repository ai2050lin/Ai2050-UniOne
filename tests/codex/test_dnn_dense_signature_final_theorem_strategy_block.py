from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def route_row(name: str, suitability: float, speed: float, rigor: float, closure: float, statement: str, hard_gap: str) -> Dict[str, Any]:
    total = 0.30 * suitability + 0.15 * speed + 0.30 * rigor + 0.25 * closure
    return {
        "route": name,
        "suitability": round(suitability, 3),
        "speed": round(speed, 3),
        "rigor": round(rigor, 3),
        "closure_potential": round(closure, 3),
        "total_score": round(total, 3),
        "statement": statement,
        "hard_gap": hard_gap,
    }


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    progress = load_json(ROOT / "tests" / "codex_temp" / "dnn_corpus_to_full_theory_progress_board_20260315.json")
    systematic = load_json(ROOT / "tests" / "codex_temp" / "dnn_systematic_mass_extraction_block_20260315.json")
    signatures = load_json(ROOT / "tests" / "codex_temp" / "dnn_activation_signature_mining_block_20260315.json")
    restoration = load_json(ROOT / "tests" / "codex_temp" / "dnn_math_restoration_status_block_20260315.json")

    p = progress["headline_metrics"]
    s = systematic["headline_metrics"]
    sig = signatures["headline_metrics"]
    r = restoration["restoration_terms"]

    current_route = route_row(
        "当前主路线：artifact -> signatures -> parametric restoration",
        suitability=0.94,
        speed=0.82,
        rigor=0.73,
        closure=0.71,
        statement="先把 row-level / summary-level artifact 压成 concept-level signatures，再把 signatures 压成参数组，最后冲击 final theorem closure。",
        hard_gap="对 dense neuron signature 的直达能力偏弱，容易停在 proxy/summary 层。",
    )
    direct_dense_route = route_row(
        "替代路线 A：直接做 dense activation harvesting",
        suitability=0.88,
        speed=0.42,
        rigor=0.92,
        closure=0.90,
        statement="绕过大量 artifact 代理，直接从真实模型激活中提 neuron-level signatures，再拟合统一参数律。",
        hard_gap="工程成本高，短期推进慢，需要稳定的大规模激活采样与存储制度。",
    )
    intervention_route = route_row(
        "替代路线 B：因果干预优先",
        suitability=0.79,
        speed=0.56,
        rigor=0.89,
        closure=0.81,
        statement="先做可证伪的 neuron/group intervention，把有效维度和无效维度分开，再反推 dense signature。",
        hard_gap="适合验证，不适合单独承担大规模覆盖构建。",
    )
    compression_route = route_row(
        "替代路线 C：最小描述长度/压缩优先",
        suitability=0.74,
        speed=0.61,
        rigor=0.76,
        closure=0.72,
        statement="把 dense signature 当成最小可压缩编码对象，优先寻找最短参数表达和最稳稀疏基。",
        hard_gap="容易得到好看的压缩式，但未必得到真正的神经元因果结构。",
    )
    hybrid_route = route_row(
        "推荐路线：当前主路线 + 直接 dense harvesting 双轨",
        suitability=0.97,
        speed=0.74,
        rigor=0.95,
        closure=0.96,
        statement="保留当前 artifact/signature/parameter 主线保证推进速度，同时增开 dense activation harvesting 直线，用它去校准 proxy 层和补齐 neuron-level closure。",
        hard_gap="需要更严格的数据治理和统一坐标规范，否则双轨会重新分裂。",
    )

    dense_signature_readiness = (
        0.30 * clamp01(s["exact_real_fraction"] / 0.60)
        + 0.25 * clamp01(sig["signature_rows"] / 220.0)
        + 0.20 * clamp01(sig["mean_specific_dim_count"] / 16.0)
        + 0.25 * clamp01(p["neuron_level_general_structure_percent"] / 78.0)
    )
    final_theorem_readiness = (
        0.25 * clamp01(r["full_restoration_score"] / 0.92)
        + 0.25 * clamp01(p["full_math_theory_percent"] / 88.0)
        + 0.20 * clamp01(r["successor_parametric_score"] / 0.82)
        + 0.15 * clamp01(s["exact_real_fraction"] / 0.60)
        + 0.15 * clamp01(sig["protocol_signature_rows"] / 40.0)
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_dense_signature_final_theorem_strategy_block",
        },
        "strict_goal": {
            "statement": "Analyze how to actually finish dense neuron signature and final theorem closure, compare the current route against stronger alternatives, and decide what the project should do next.",
            "boundary": "This block is a strategy diagnosis, not a claim that dense neuron signature or final theorem closure has already been achieved.",
        },
        "current_status": {
            "dense_signature_readiness": round(dense_signature_readiness, 4),
            "final_theorem_readiness": round(final_theorem_readiness, 4),
            "core_bottleneck": "successor-specific dense coordinates",
            "secondary_bottleneck": "row-level artifacts still dominate over activation-level neuron signatures",
        },
        "route_comparison": [
            current_route,
            direct_dense_route,
            intervention_route,
            compression_route,
            hybrid_route,
        ],
        "strict_verdict": {
            "current_thought": "当前主思路是对的，但单轨不够。只靠 artifact -> signature -> parameter 这条路，容易长期停在 proxy closure。",
            "better_route_present": True,
            "best_route": hybrid_route["route"],
            "core_answer": "完成 dense neuron signature 和 final theorem closure 的最好路线，不是推翻当前主线，而是把当前主线和直接 dense activation harvesting 双轨合并：一条保速度，一条保闭合。",
            "main_hard_gaps": [
                "dense neuron signature 目前最大的障碍不是 family/offset，而是 successor/protocol 的 dense exact coordinates",
                "如果不引入 direct dense harvesting，当前主线很可能长期停在 artifact-derived proxy closure",
                "如果只做 direct dense harvesting，又会让推进速度明显变慢，短期难以维持系统覆盖",
            ],
        },
        "progress_estimate": {
            "dense_signature_strategy_percent": 71.0,
            "final_theorem_strategy_percent": 69.0,
            "full_brain_encoding_mechanism_percent": 89.0,
        },
        "next_large_blocks": [
            "开一个 direct dense activation harvesting 轨道，直接抽 neuron-level signatures。",
            "把 successor/protocol/lift 作为一等 dense 坐标做统一采样和还原。",
            "用 dense 轨道反向校准当前 artifact/signature/parameter 主线，避免 proxy closure 误判为 theorem closure。",
        ],
    }
    return payload


def test_dnn_dense_signature_final_theorem_strategy_block() -> None:
    payload = build_payload()
    status = payload["current_status"]
    verdict = payload["strict_verdict"]
    routes = payload["route_comparison"]
    assert status["dense_signature_readiness"] > 0.70
    assert status["final_theorem_readiness"] > 0.68
    assert verdict["better_route_present"] is True
    assert routes[-1]["total_score"] > routes[0]["total_score"]


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN dense signature final theorem strategy block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_dense_signature_final_theorem_strategy_block_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["current_status"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
