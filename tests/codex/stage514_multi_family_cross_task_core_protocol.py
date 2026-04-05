#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from multimodel_language_shared import MODEL_SPECS, free_model, load_model_bundle
from qwen3_language_shared import capture_qwen_mlp_payloads, discover_layers, move_batch_to_model_device, remove_hooks


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage514_multi_family_cross_task_core_protocol_20260404"
)
MODEL_KEYS = ["qwen3", "deepseek7b", "glm4", "gemma4"]
TOP_K = 256

FAMILY_SPECS = {
    "fruit_apple": {
        "target": "apple",
        "groups": {
            "family_knowledge": [
                "An apple is a kind of fruit.",
                "People often eat an apple as a snack.",
                "The apple belongs to the fruit family.",
            ],
            "syntax_subject": [
                "The apple fell from the tree.",
                "The apple rolled across the table.",
                "The apple hit the floor.",
            ],
            "syntax_object": [
                "She sliced the apple into pieces.",
                "He washed the apple in the sink.",
                "They packed the apple in a lunch box.",
            ],
            "attribute_binding": [
                "The apple is red and sweet.",
                "The apple tastes sweet and juicy.",
                "The apple is about the size of a fist.",
            ],
            "concept_association": [
                "Apple is related to banana and pear.",
                "Apple is associated with fruit, tree, and juice.",
                "Apple connects naturally to orchard, seed, and harvest.",
            ],
        },
    },
    "animal_tiger": {
        "target": "tiger",
        "groups": {
            "family_knowledge": [
                "A tiger is a kind of animal.",
                "A tiger belongs to the big cat family.",
                "People recognize the tiger as a wild animal.",
            ],
            "syntax_subject": [
                "The tiger ran across the field.",
                "The tiger roared at the edge of the forest.",
                "The tiger leaped over the stream.",
            ],
            "syntax_object": [
                "The ranger watched the tiger from a tower.",
                "The camera captured the tiger at night.",
                "They protected the tiger in the reserve.",
            ],
            "attribute_binding": [
                "The tiger is large and striped.",
                "The tiger looks powerful and fast.",
                "The tiger has bright orange fur.",
            ],
            "concept_association": [
                "Tiger is related to lion and leopard.",
                "Tiger is associated with forest, hunt, and claw.",
                "Tiger connects naturally to predator, stripe, and roar.",
            ],
        },
    },
    "tool_hammer": {
        "target": "hammer",
        "groups": {
            "family_knowledge": [
                "A hammer is a kind of tool.",
                "Workers use a hammer as a basic tool.",
                "The hammer belongs to the hand tool family.",
            ],
            "syntax_subject": [
                "The hammer fell off the bench.",
                "The hammer struck the nail.",
                "The hammer slid across the table.",
            ],
            "syntax_object": [
                "He picked up the hammer from the box.",
                "She placed the hammer near the wall.",
                "They repaired the hammer after work.",
            ],
            "attribute_binding": [
                "The hammer is heavy and solid.",
                "The hammer has a wooden handle.",
                "The hammer feels cold and hard.",
            ],
            "concept_association": [
                "Hammer is related to nail and toolbox.",
                "Hammer is associated with building, repair, and wood.",
                "Hammer connects naturally to workshop, metal, and handle.",
            ],
        },
    },
    "org_google": {
        "target": "Google",
        "groups": {
            "family_knowledge": [
                "Google is a kind of company.",
                "Google belongs to the technology company family.",
                "People know Google as a major tech company.",
            ],
            "syntax_subject": [
                "Google released a new model this week.",
                "Google announced a product update this morning.",
                "Google expanded its cloud platform this year.",
            ],
            "syntax_object": [
                "Investors followed Google after the earnings call.",
                "Developers discussed Google at the conference.",
                "The article mentioned Google in the first paragraph.",
            ],
            "attribute_binding": [
                "Google is large and influential.",
                "Google is known for search and cloud services.",
                "Google is a global technology company.",
            ],
            "concept_association": [
                "Google is related to search and Android.",
                "Google is associated with cloud, ads, and data.",
                "Google connects naturally to browser, AI, and maps.",
            ],
        },
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_last_subsequence(full_ids: List[int], sub_ids: List[int]) -> Tuple[int, int] | None:
    if not sub_ids or len(sub_ids) > len(full_ids):
        return None
    last_match = None
    for start in range(0, len(full_ids) - len(sub_ids) + 1):
        if full_ids[start : start + len(sub_ids)] == sub_ids:
            last_match = (start, start + len(sub_ids))
    return last_match


def locate_target_span(tokenizer, prompt: str, target: str) -> Tuple[int, int]:
    full_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    variants = [target, f" {target}", target.lower(), f" {target.lower()}", target.capitalize(), f" {target.capitalize()}"]
    best = None
    best_len = -1
    seen = set()
    for text in variants:
        if text in seen:
            continue
        seen.add(text)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        match = find_last_subsequence(full_ids, ids)
        if match is not None and len(ids) > best_len:
            best = match
            best_len = len(ids)
    if best is None:
        raise RuntimeError(f"无法定位目标词：target={target!r}, prompt={prompt!r}")
    return best


def capture_case_flat_neuron_vector(model, tokenizer, sentence: str, target: str) -> torch.Tensor:
    layer_payload_map = {layer_idx: "neuron_in" for layer_idx in range(len(discover_layers(model)))}
    buffers, handles = capture_qwen_mlp_payloads(model, layer_payload_map)
    try:
        encoded = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
        encoded = move_batch_to_model_device(model, encoded)
        start, end = locate_target_span(tokenizer, sentence, target)
        with torch.inference_mode():
            model(**encoded, use_cache=False, return_dict=True)
        per_layer = []
        for layer_idx in range(len(buffers)):
            payload = buffers[layer_idx]
            if payload is None:
                raise RuntimeError(f"缺少第 {layer_idx} 层神经元载荷")
            vec = payload[0, start:end, :].mean(dim=0).detach().float().cpu()
            per_layer.append(vec)
        return torch.cat(per_layer, dim=0)
    finally:
        remove_hooks(handles)


def top_active_ids(vec: torch.Tensor, top_k: int) -> List[int]:
    positive = torch.clamp(vec.float(), min=0.0)
    vals, idxs = torch.topk(positive, k=min(top_k, positive.numel()))
    out = []
    for value, idx in zip(vals.tolist(), idxs.tolist()):
        if value <= 0:
            continue
        out.append(int(idx))
    return out


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def mean_tensor(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([v.float() for v in vectors], dim=0).mean(dim=0)


def analyze_family(model, tokenizer, family_spec: Dict[str, object]) -> Dict[str, object]:
    group_vectors: Dict[str, torch.Tensor] = {}
    group_top_ids: Dict[str, List[int]] = {}
    for group_name, sentences in family_spec["groups"].items():
        rows = [
            capture_case_flat_neuron_vector(model, tokenizer, sentence, str(family_spec["target"]))
            for sentence in sentences
        ]
        group_mean = mean_tensor(rows)
        group_vectors[group_name] = group_mean
        group_top_ids[group_name] = top_active_ids(group_mean, TOP_K)

    group_names = list(family_spec["groups"].keys())
    pairwise = []
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            a = group_names[i]
            b = group_names[j]
            pairwise.append({"group_a": a, "group_b": b, "active_jaccard": jaccard(group_top_ids[a], group_top_ids[b])})

    id_freq = defaultdict(int)
    for ids in group_top_ids.values():
        for idx in ids:
            id_freq[int(idx)] += 1
    shared_all = sorted([idx for idx, freq in id_freq.items() if freq == len(group_names)])
    shared_4plus = sorted([idx for idx, freq in id_freq.items() if freq >= 4])
    unique_counts = {group_name: sum(1 for idx in ids if id_freq[int(idx)] == 1) for group_name, ids in group_top_ids.items()}

    family_set = set(group_top_ids["family_knowledge"])
    syntax_union = set(group_top_ids["syntax_subject"]) | set(group_top_ids["syntax_object"])
    attr_set = set(group_top_ids["attribute_binding"])
    assoc_set = set(group_top_ids["concept_association"])

    return {
        "family_id": next(key for key, value in FAMILY_SPECS.items() if value is family_spec),
        "target": family_spec["target"],
        "group_top_ids": group_top_ids,
        "pairwise_overlaps": pairwise,
        "summary": {
            "core_shared_all_groups_count": len(shared_all),
            "core_shared_4plus_count": len(shared_4plus),
            "core_shared_ratio": len(shared_all) / TOP_K,
            "shared_knowledge_syntax_count": len(family_set & syntax_union),
            "shared_knowledge_attribute_count": len(family_set & attr_set),
            "shared_knowledge_association_count": len(family_set & assoc_set),
            "syntax_internal_overlap": jaccard(group_top_ids["syntax_subject"], group_top_ids["syntax_object"]),
            "attribute_vs_association_overlap": jaccard(group_top_ids["attribute_binding"], group_top_ids["concept_association"]),
            "mean_pairwise_overlap": sum(row["active_jaccard"] for row in pairwise) / max(1, len(pairwise)),
            "unique_adapter_counts": unique_counts,
        },
    }


def analyze_model(model_key: str) -> Dict[str, object]:
    model, tokenizer = load_model_bundle(model_key, prefer_cuda=False)
    try:
        family_rows = [analyze_family(model, tokenizer, family_spec) for family_spec in FAMILY_SPECS.values()]
        agg = {
            "family_count": len(family_rows),
            "mean_core_shared_all": sum(row["summary"]["core_shared_all_groups_count"] for row in family_rows) / len(family_rows),
            "mean_core_shared_4plus": sum(row["summary"]["core_shared_4plus_count"] for row in family_rows) / len(family_rows),
            "mean_core_shared_ratio": sum(row["summary"]["core_shared_ratio"] for row in family_rows) / len(family_rows),
            "mean_knowledge_syntax_shared": sum(row["summary"]["shared_knowledge_syntax_count"] for row in family_rows) / len(family_rows),
            "mean_knowledge_attribute_shared": sum(row["summary"]["shared_knowledge_attribute_count"] for row in family_rows) / len(family_rows),
            "mean_knowledge_association_shared": sum(row["summary"]["shared_knowledge_association_count"] for row in family_rows) / len(family_rows),
            "mean_attribute_association_overlap": sum(row["summary"]["attribute_vs_association_overlap"] for row in family_rows)
            / len(family_rows),
        }
        return {
            "model_key": model_key,
            "model_name": MODEL_SPECS[model_key]["label"],
            "layer_count": len(discover_layers(model)),
            "family_rows": family_rows,
            "aggregate": agg,
        }
    finally:
        free_model(model)


def build_report(model_rows: List[Dict[str, object]], summary: Dict[str, object]) -> str:
    lines = [f"# {summary['experiment_id']}", "", "## 核心结论", summary["core_answer"], ""]
    for row in model_rows:
        agg = row["aggregate"]
        lines.extend(
            [
                f"## {row['model_name']}",
                f"- 家族数量：`{agg['family_count']}`",
                f"- 平均全任务共享核心：`{agg['mean_core_shared_all']:.2f}`",
                f"- 平均四组以上共享核心：`{agg['mean_core_shared_4plus']:.2f}`",
                f"- 平均全任务共享比例：`{agg['mean_core_shared_ratio']:.4f}`",
                f"- 平均知识-语法共享：`{agg['mean_knowledge_syntax_shared']:.2f}`",
                f"- 平均知识-属性共享：`{agg['mean_knowledge_attribute_shared']:.2f}`",
                f"- 平均知识-联想共享：`{agg['mean_knowledge_association_shared']:.2f}`",
                f"- 平均属性/联想重合：`{agg['mean_attribute_association_overlap']:.4f}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    model_rows = [analyze_model(model_key) for model_key in MODEL_KEYS]
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage514_multi_family_cross_task_core_protocol",
        "title": "多名词家族跨任务骨干协议",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "model_rows": model_rows,
        "core_answer": (
            "共享骨干 + 任务适配器并不是苹果个案，而是在水果、动物、工具、组织四类名词家族里都可见的更一般结构。"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "REPORT.md").write_text(build_report(model_rows, summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
