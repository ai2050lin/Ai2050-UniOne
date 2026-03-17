from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class DnnThousandNounFamilyPatchOffsetProgram:
    root: Path
    large_inventory: Dict[str, Any]
    dense_schema: Dict[str, Any]
    specific_manifest: Dict[str, Any]

    @classmethod
    def from_repo(cls, root: Path) -> "DnnThousandNounFamilyPatchOffsetProgram":
        temp = root / "tests" / "codex_temp"
        return cls(
            root=root,
            large_inventory=load_json(temp / "theory_track_large_scale_concept_inventory_analysis_20260312.json"),
            dense_schema=load_json(temp / "dnn_joint_dense_export_schema_block_20260315.json"),
            specific_manifest=load_json(temp / "dnn_family_specific_dense_target_manifest_block_20260316.json"),
        )

    def summary(self) -> Dict[str, Any]:
        mass_scan_paths = sorted(
            self.root.glob("tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed*/mass_noun_encoding_scan.json")
        )
        scan_counts: List[int] = []
        for path in mass_scan_paths:
            payload = load_json(path)
            noun_records = payload.get("noun_records", [])
            scan_counts.append(len(noun_records))

        current_scan_count = len(mass_scan_paths)
        current_noun_total = sum(scan_counts)
        current_unique_capacity = max(scan_counts) if scan_counts else 0
        inventory_concepts = int(self.large_inventory["headline_metrics"]["num_concepts"])
        thousand_target = 3000
        batch_size = max(1, current_unique_capacity or 120)
        projected_batches = int((thousand_target + batch_size - 1) // batch_size)
        schema_ready = float(self.dense_schema["headline_metrics"]["schema_ready_score"])
        specific_gap = float(self.specific_manifest["headline_metrics"]["closure_gap"])
        specific_uplift = float(self.specific_manifest["headline_metrics"]["projected_dense_uplift"])

        thousand_scale_readiness = min(
            1.0,
            0.30 * min(1.0, current_noun_total / 600.0)
            + 0.20 * min(1.0, inventory_concepts / 384.0)
            + 0.25 * schema_ready
            + 0.25 * specific_uplift,
        )

        metric_lines_cn = [
            f"（当前mass noun扫描批次数）current_mass_scan_count = {current_scan_count:.4f}",
            f"（当前mass noun累计记录数）current_mass_noun_total = {current_noun_total:.4f}",
            f"（当前单批名词容量）current_unique_capacity = {current_unique_capacity:.4f}",
            f"（当前概念库存规模）inventory_concepts = {inventory_concepts:.4f}",
            f"（3000名词目标批次数）projected_batches_for_3000 = {projected_batches:.4f}",
            f"（数千名词计划准备度）thousand_scale_readiness = {thousand_scale_readiness:.4f}",
        ]

        return {
            "metric_lines_cn": metric_lines_cn,
            "program": {
                "current_base": {
                    "mass_scan_paths": [str(path.relative_to(self.root)) for path in mass_scan_paths],
                    "current_mass_scan_count": current_scan_count,
                    "current_mass_noun_total": current_noun_total,
                    "current_unique_capacity": current_unique_capacity,
                    "inventory_concepts": inventory_concepts,
                },
                "thousand_noun_target": {
                    "target_nouns": thousand_target,
                    "estimated_batch_size": batch_size,
                    "projected_batches": projected_batches,
                },
                "why_this_can_help": [
                    "数千名词会让 family patch 的统计更稳，因为每个 family 不再只靠少量示例支撑。",
                    "数千名词会更容易暴露 concept offset 的共性，比如哪些偏移总是稀疏、哪些方向总是共享。",
                    "数千名词还能把语言里的家族结构、属性重用和关系差异，从局部例子推进成全局编码规律。",
                ],
                "what_it_will_not_auto_solve": [
                    "光把名词规模拉大，不会自动解决 successor exact closure。",
                    "光有更多名词，也不等于已经得到 dense neuron-level exact theorem。",
                    "如果没有统一 dense export，数千名词很容易停在更大规模的 summary，而不是更强的定理闭合。",
                ],
                "execution_blocks": [
                    {
                        "block": "阶段一：把现有 120 名词批次扩成稳定多批次计划",
                        "content": "以现有 n=120 的 mass noun 机制扫描为标准批次，先完成 3000 名词的批次切分和 family 覆盖计划。",
                    },
                    {
                        "block": "阶段二：把数千名词接到 family-to-specific dense target 上",
                        "content": "不是只做 summary，而是把高价值名词批次接入 specific_dense_signature 的统一 schema。",
                    },
                    {
                        "block": "阶段三：用数千名词重算全局编码规律",
                        "content": "重算 family patch 稳定性、offset 稀疏性、跨 family 共享 scaffold、属性方向复用，再回灌数学定理候选。",
                    },
                ],
            },
            "strict_conclusion": {
                "core_answer": "Yes. An analysis over thousands of nouns is feasible and likely very valuable for discovering more universal family patch and concept offset laws. But it only helps crack language coding if the scaling is tied to unified dense export rather than staying at summary level.",
                "main_hard_gaps": [
                    "current evidence is still much stronger at summary/signature level than dense neuron level",
                    "successor exact closure will still remain open even if noun coverage scales up",
                    "thousand-scale data without exact tensor export can increase quantity without improving theorem closure enough",
                ],
            },
        }
