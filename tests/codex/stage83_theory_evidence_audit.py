from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage83_theory_evidence_audit_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage82_novelty_generalization_repair import build_novelty_generalization_repair_summary


STAGE71_PATH = ROOT / "tests" / "codex" / "stage71_first_principles_unification.py"
STAGE73_PATH = ROOT / "tests" / "codex" / "stage73_falsifiability_boundary_hardening.py"
STAGE80_PATH = ROOT / "tests" / "codex" / "stage80_intelligence_closure_failure_map.py"
STAGE82_PATH = ROOT / "tests" / "codex" / "stage82_novelty_generalization_repair.py"
AUDIT_TEST_PATHS = [
    ROOT / "tests" / "codex" / "test_stage71_first_principles_unification.py",
    ROOT / "tests" / "codex" / "test_stage73_falsifiability_boundary_hardening.py",
    ROOT / "tests" / "codex" / "test_stage81_forward_backward_unification.py",
    ROOT / "tests" / "codex" / "test_stage82_novelty_generalization_repair.py",
]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _count_stage_imports(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.startswith("from stage") and " import build_" in line)


def _assigned_container_length(path: Path, target_name: str) -> int:
    tree = ast.parse(_read_text(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == target_name:
                    if isinstance(node.value, ast.List):
                        return len(node.value.elts)
                    if isinstance(node.value, ast.Dict):
                        return len(node.value.keys)
    raise ValueError(f"target {target_name} not found in {path}")


def _roundtrip_only_test_flag(path: Path) -> bool:
    text = _read_text(path)
    return (
        "write_text(json.dumps(summary" in text
        and "saved = json.loads(out_path.read_text" in text
        and 'status["status_short"]' in text
    )


def build_theory_evidence_audit_summary() -> dict:
    stage71_text = _read_text(STAGE71_PATH)
    stage73_text = _read_text(STAGE73_PATH)

    stage71_summary_dependency_fan_in = _count_stage_imports(stage71_text)
    stage80_hardcoded_scenario_count = _assigned_container_length(STAGE80_PATH, "scenarios")
    stage82_hardcoded_law_count = _assigned_container_length(STAGE82_PATH, "laws")

    derived_falsification_flag = (
        "synthetic_mismatch_support = _clip01(shared_state_support - 0.24)" in stage73_text
        and '"trigger_demonstrated": synthetic_mismatch_support < shared_state_support' in stage73_text
    )
    status_label_mismatch_flag = (
        "first_principles_unification_frontier" in stage71_text
        and "仍处在第一性原理统一过渡区" in stage71_text
    )
    roundtrip_only_test_count = sum(1 for path in AUDIT_TEST_PATHS if _roundtrip_only_test_flag(path))

    start = time.perf_counter()
    stage82_summary = build_novelty_generalization_repair_summary()
    stage82_runtime_seconds = time.perf_counter() - start

    law_rank = sorted(
        (
            {
                "law_name": law_name,
                "repaired_novelty_score": metrics["repaired_novelty_score"],
                "failure_after": metrics["failure_after"],
            }
            for law_name, metrics in stage82_summary["law_results"].items()
        ),
        key=lambda item: item["repaired_novelty_score"],
        reverse=True,
    )
    best_law_margin = law_rank[0]["repaired_novelty_score"] - law_rank[1]["repaired_novelty_score"]
    best_law_fragility_flag = best_law_margin < 0.005

    evidence_independence_score = _clip01(
        0.64
        - 0.02 * max(0, stage71_summary_dependency_fan_in - 8)
        - 0.18 * float(derived_falsification_flag)
        - 0.12 * float(best_law_fragility_flag)
        - 0.08 * float(status_label_mismatch_flag)
    )
    test_strength_score = _clip01(
        0.60
        - 0.11 * roundtrip_only_test_count
        - 0.08 * float(derived_falsification_flag)
        - 0.06 * float(stage82_runtime_seconds > 20.0)
    )
    theory_correctness_confidence = _clip01(
        0.26 * evidence_independence_score
        + 0.22 * test_strength_score
        + 0.18 * (1.0 - float(best_law_fragility_flag))
        + 0.16 * (1.0 - float(derived_falsification_flag))
        + 0.18 * (1.0 - min(1.0, stage82_runtime_seconds / 90.0))
    )

    if theory_correctness_confidence >= 0.74 and evidence_independence_score >= 0.70 and test_strength_score >= 0.68:
        audit_status_short = "theory_evidence_hardened"
    elif theory_correctness_confidence >= 0.56:
        audit_status_short = "theory_evidence_transition"
    else:
        audit_status_short = "unproven_explanatory_framework"

    findings = [
        "高层统一分数对下层摘要分数有明显回灌，证据独立性不足。",
        "可判伪边界包含脚本内构造出的 synthetic mismatch，不属于强外部反例。",
        "Stage82 的最优律虽然仍是 sqrt，但与第二名的优势很小，当前结论脆弱。",
        "现有测试多为阈值断言加落盘回读，主要证明脚本稳定，不足以证明理论正确。",
    ]

    return {
        "headline_metrics": {
            "stage71_summary_dependency_fan_in": stage71_summary_dependency_fan_in,
            "stage80_hardcoded_scenario_count": stage80_hardcoded_scenario_count,
            "stage82_hardcoded_law_count": stage82_hardcoded_law_count,
            "stage82_runtime_seconds": stage82_runtime_seconds,
            "stage82_best_law_name": law_rank[0]["law_name"],
            "stage82_best_law_margin": best_law_margin,
            "roundtrip_only_test_count": roundtrip_only_test_count,
            "derived_falsification_flag": derived_falsification_flag,
            "best_law_fragility_flag": best_law_fragility_flag,
            "status_label_mismatch_flag": status_label_mismatch_flag,
            "evidence_independence_score": evidence_independence_score,
            "test_strength_score": test_strength_score,
            "theory_correctness_confidence": theory_correctness_confidence,
        },
        "law_rank": law_rank,
        "audit_findings": findings,
        "status": {
            "status_short": audit_status_short,
            "status_label": "当前理论更像解释框架而非已被强证据锁定的第一性原理定理体系。",
        },
        "project_readout": {
            "summary": "这一轮不是继续给理论加分，而是反过来审计证据独立性、判伪真实性、最优律稳健性和测试强度。",
            "next_question": "下一步应把 synthetic mismatch 改成真正外部生成的反例，并增加参数扰动与顺序打乱测试，看 sqrt 优势是否仍然成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage83 Theory Evidence Audit",
        "",
        f"- stage71_summary_dependency_fan_in: {hm['stage71_summary_dependency_fan_in']}",
        f"- stage80_hardcoded_scenario_count: {hm['stage80_hardcoded_scenario_count']}",
        f"- stage82_hardcoded_law_count: {hm['stage82_hardcoded_law_count']}",
        f"- stage82_runtime_seconds: {hm['stage82_runtime_seconds']:.6f}",
        f"- stage82_best_law_name: {hm['stage82_best_law_name']}",
        f"- stage82_best_law_margin: {hm['stage82_best_law_margin']:.6f}",
        f"- roundtrip_only_test_count: {hm['roundtrip_only_test_count']}",
        f"- derived_falsification_flag: {hm['derived_falsification_flag']}",
        f"- best_law_fragility_flag: {hm['best_law_fragility_flag']}",
        f"- status_label_mismatch_flag: {hm['status_label_mismatch_flag']}",
        f"- evidence_independence_score: {hm['evidence_independence_score']:.6f}",
        f"- test_strength_score: {hm['test_strength_score']:.6f}",
        f"- theory_correctness_confidence: {hm['theory_correctness_confidence']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_theory_evidence_audit_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
