from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage87_evidence_independence_audit_20260322"
TARGET_FILES = {
    "stage71": ROOT / "tests" / "codex" / "stage71_first_principles_unification.py",
    "stage73": ROOT / "tests" / "codex" / "stage73_falsifiability_boundary_hardening.py",
    "stage80": ROOT / "tests" / "codex" / "stage80_intelligence_closure_failure_map.py",
    "stage81": ROOT / "tests" / "codex" / "stage81_forward_backward_unification.py",
    "stage82": ROOT / "tests" / "codex" / "stage82_novelty_generalization_repair.py",
    "stage83": ROOT / "tests" / "codex" / "stage83_forward_backward_theorem_kernel.py",
    "stage84": ROOT / "tests" / "codex" / "stage84_falsifiable_computation_core.py",
}
TEST_FILES = [
    ROOT / "tests" / "codex" / "test_stage71_first_principles_unification.py",
    ROOT / "tests" / "codex" / "test_stage73_falsifiability_boundary_hardening.py",
    ROOT / "tests" / "codex" / "test_stage81_forward_backward_unification.py",
    ROOT / "tests" / "codex" / "test_stage82_novelty_generalization_repair.py",
    ROOT / "tests" / "codex" / "test_stage84_falsifiable_computation_core.py",
]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _extract_stage_deps(text: str) -> set[str]:
    return {
        match.group(1)
        for match in re.finditer(r"from (stage\d+[a-z_0-9]*) import build_", text)
    }


def _has_path(graph: dict[str, set[str]], start: str, target: str) -> bool:
    queue = deque([start])
    seen = set()
    while queue:
        node = queue.popleft()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        queue.extend(graph.get(node, set()) - seen)
    return False


@lru_cache(maxsize=1)
def build_evidence_independence_audit_summary() -> dict:
    graph: dict[str, set[str]] = {}
    hardcoded_scenario_hits = 0
    handcrafted_law_hits = 0
    for name, path in TARGET_FILES.items():
        text = path.read_text(encoding="utf-8")
        graph[name] = _extract_stage_deps(text)
        hardcoded_scenario_hits += text.count("scenarios = [") + text.count("scenario = {") + text.count("DEFAULT_NOVELTY_SCENARIO")
        handcrafted_law_hits += text.count("laws = {") + text.count("DEFAULT_NOVELTY_LAWS")

    stage71_deps = graph["stage71"]
    independent_evidence_chain_exists = len(stage71_deps) <= 2
    summary_backfeed_risk = _clip01(len(stage71_deps) / 8.0)

    backfeed_paths = []
    if _has_path(graph, "stage71", "stage80") and ("stage80" in graph["stage81"] or "stage80" in graph["stage82"]):
        backfeed_paths.append("stage71 -> stage81/stage82 -> stage80")
    if _has_path(graph, "stage71", "stage73") and _has_path(graph, "stage84", "stage73"):
        backfeed_paths.append("stage71 -> stage84 -> stage73")
    if _has_path(graph, "stage84", "stage82") and _has_path(graph, "stage82", "stage81"):
        backfeed_paths.append("stage84 -> stage82 -> stage81")

    threshold_asserts = 0
    structural_asserts = 0
    for path in TEST_FILES:
        text = path.read_text(encoding="utf-8")
        threshold_asserts += len(re.findall(r"assert .*[><=]", text))
        structural_asserts += len(re.findall(r"assert .* in |assert .*==|assert len\(", text))
    threshold_only_test_ratio = threshold_asserts / max(1, threshold_asserts + structural_asserts)

    document_text = (ROOT / "research" / "gpt5" / "docs" / "AGI_GPT5_ICSPB.md").read_text(encoding="utf-8")
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    documentation_frontier_claim = "前沿区" in document_text or "前沿区" in readme_text
    code_transition_claim = "first_principles_unification_transition" in TARGET_FILES["stage71"].read_text(encoding="utf-8")
    documentation_consistency = not (documentation_frontier_claim and code_transition_claim)

    audit_checks = [
        {
            "name": "independent_evidence_chain",
            "risk_level": "high" if not independent_evidence_chain_exists else "medium",
            "passed": independent_evidence_chain_exists,
            "detail": "高层结论当前仍依赖较多下层摘要块。",
        },
        {
            "name": "evidence_backfeed",
            "risk_level": "high" if backfeed_paths else "medium",
            "passed": not backfeed_paths,
            "detail": "检测到多条摘要回灌路径。" if backfeed_paths else "未发现明显回灌路径。",
        },
        {
            "name": "synthetic_falsification",
            "risk_level": "medium",
            "passed": False,
            "detail": "已去掉固定差值构造，但反例仍是内部设定样本。",
        },
        {
            "name": "hardcoded_scenario_dominance",
            "risk_level": "high" if hardcoded_scenario_hits + handcrafted_law_hits >= 4 else "medium",
            "passed": False,
            "detail": "最坏场景与候选律仍主要来自人工设定。",
        },
        {
            "name": "candidate_margin_strength",
            "risk_level": "high",
            "passed": False,
            "detail": "最优律领先幅度仍然较小，需要继续扩大优势边际。",
        },
        {
            "name": "perturbation_counterfactual_coverage",
            "risk_level": "medium",
            "passed": threshold_only_test_ratio < 0.65,
            "detail": "已开始补扰动与结构测试，但外部基线仍缺失。",
        },
        {
            "name": "documentation_consistency",
            "risk_level": "medium" if not documentation_consistency else "low",
            "passed": documentation_consistency,
            "detail": "文档与代码状态口径需要同步。" if not documentation_consistency else "文档与代码口径一致。",
        },
        {
            "name": "review_cost",
            "risk_level": "medium",
            "passed": False,
            "detail": "同进程缓存已降低重复调用成本，但跨进程强审计仍然偏重。",
        },
    ]

    high_risk_count = sum(1 for item in audit_checks if item["risk_level"] == "high" and not item["passed"])
    evidence_independence_score = _clip01(
        0.30 * float(independent_evidence_chain_exists)
        + 0.20 * float(not backfeed_paths)
        + 0.20 * float(documentation_consistency)
        + 0.15 * (1.0 - min(1.0, threshold_only_test_ratio))
        + 0.15 * (1.0 - min(1.0, (hardcoded_scenario_hits + handcrafted_law_hits) / 10.0))
    )

    return {
        "headline_metrics": {
            "independent_evidence_chain_exists": independent_evidence_chain_exists,
            "summary_backfeed_risk": summary_backfeed_risk,
            "threshold_only_test_ratio": threshold_only_test_ratio,
            "hardcoded_scenario_hits": hardcoded_scenario_hits,
            "handcrafted_law_hits": handcrafted_law_hits,
            "documentation_consistency": documentation_consistency,
            "high_risk_count": high_risk_count,
            "evidence_independence_score": evidence_independence_score,
        },
        "dependency_graph": {name: sorted(deps) for name, deps in graph.items()},
        "backfeed_paths": backfeed_paths,
        "audit_checks": audit_checks,
        "status": {
            "status_short": (
                "evidence_independence_audit_high_risk"
                if high_risk_count >= 3
                else "evidence_independence_audit_transition"
            ),
            "status_label": "证据独立性审计已经把主要风险显式钉出来，但当前理论仍存在明显回灌与内部构造成分。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage87 Evidence Independence Audit",
        "",
        f"- independent_evidence_chain_exists: {hm['independent_evidence_chain_exists']}",
        f"- summary_backfeed_risk: {hm['summary_backfeed_risk']:.6f}",
        f"- threshold_only_test_ratio: {hm['threshold_only_test_ratio']:.6f}",
        f"- hardcoded_scenario_hits: {hm['hardcoded_scenario_hits']}",
        f"- handcrafted_law_hits: {hm['handcrafted_law_hits']}",
        f"- documentation_consistency: {hm['documentation_consistency']}",
        f"- high_risk_count: {hm['high_risk_count']}",
        f"- evidence_independence_score: {hm['evidence_independence_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_evidence_independence_audit_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
