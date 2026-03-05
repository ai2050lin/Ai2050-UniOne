#!/usr/bin/env python
"""
Cross-dimension causal ablation for style/logic/syntax encoding subsets.

输入：
- multidim_encoding_probe.json（含每个维度的 specific_top_neurons）

输出：
- multidim_causal_ablation.json
- MULTIDIM_CAUSAL_ABLATION_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DIMENSIONS = ["style", "logic", "syntax"]


def build_contrast_pairs() -> Dict[str, List[Dict[str, str]]]:
    return {
        "style": [
            {
                "id": "style_chat_vs_paper_1",
                "a": "User: What is an apple?\nAssistant: An apple is a",
                "b": "In formal academic writing, an apple is a",
            },
            {
                "id": "style_chat_vs_paper_2",
                "a": "User: Explain gravity in one sentence.\nAssistant: Gravity is",
                "b": "In concise scientific prose, gravity is",
            },
            {
                "id": "style_casual_vs_formal_1",
                "a": "Tell me quickly: a cat is a",
                "b": "Provide a formal definition: a cat is a",
            },
        ],
        "logic": [
            {
                "id": "logic_valid_vs_invalid_1",
                "a": "Premise: All fruits are edible. Premise: Apple is a fruit. Therefore apple is",
                "b": "Premise: All fruits are edible. Premise: Apple is a fruit. Therefore apple is not",
            },
            {
                "id": "logic_valid_vs_invalid_2",
                "a": "Premise: All mammals are animals. Premise: Dog is a mammal. Therefore dog is an",
                "b": "Premise: All mammals are animals. Premise: Dog is a mammal. Therefore dog is not an",
            },
            {
                "id": "logic_consistent_vs_contradict_1",
                "a": "Statement A: Every triangle has three sides. Statement B: This shape is a triangle. Conclusion: it has",
                "b": "Statement A: Every triangle has three sides. Statement B: This shape is a triangle. Conclusion: it does not have",
            },
        ],
        "syntax": [
            {
                "id": "syntax_active_vs_passive_1",
                "a": "The scientist solved the problem. The result was",
                "b": "The problem was solved by the scientist. The result was",
            },
            {
                "id": "syntax_active_vs_passive_2",
                "a": "The dog chased the cat in the yard. Then it",
                "b": "The cat was chased by the dog in the yard. Then it",
            },
            {
                "id": "syntax_clause_order_1",
                "a": "Because it was raining, the match was canceled. This decision was",
                "b": "The match was canceled because it was raining. This decision was",
            },
        ],
    }


class GateCollector:
    def __init__(self, model):
        self.layers = list(model.model.layers)
        self.buffers: List[torch.Tensor | None] = [None for _ in self.layers]
        self.handles = []
        for li, layer in enumerate(self.layers):
            self.handles.append(layer.mlp.gate_proj.register_forward_hook(self._mk_hook(li)))

    def _mk_hook(self, layer_idx: int):
        def _hook(_module, _inputs, output):
            self.buffers[layer_idx] = output[0, -1, :].detach().float().cpu()
            return output

        return _hook

    def reset(self):
        for i in range(len(self.buffers)):
            self.buffers[i] = None

    def get_flat(self) -> np.ndarray:
        miss = [i for i, x in enumerate(self.buffers) if x is None]
        if miss:
            raise RuntimeError(f"Missing hook outputs for layers: {miss}")
        return np.concatenate([x.numpy() for x in self.buffers if x is not None], axis=0).astype(np.float32)

    def close(self):
        for h in self.handles:
            h.remove()


class GateAblator:
    def __init__(self, model, flat_indices: Sequence[int], d_ff: int, ablate_all_positions: bool):
        self.handles = []
        by_layer: Dict[int, List[int]] = defaultdict(list)
        for idx in flat_indices:
            ii = int(idx)
            by_layer[ii // d_ff].append(ii % d_ff)

        for li, arr in by_layer.items():
            module = model.model.layers[li].mlp.gate_proj
            idx_cpu = torch.tensor(sorted(set(arr)), dtype=torch.long)

            def _mk(local_idx_cpu, all_pos: bool):
                def _hook(_module, _inputs, output):
                    out = output.clone()
                    idx = local_idx_cpu.to(out.device)
                    if all_pos:
                        out[:, :, idx] = 0.0
                    else:
                        out[:, -1, idx] = 0.0
                    return out

                return _hook

            self.handles.append(module.register_forward_hook(_mk(idx_cpu, ablate_all_positions)))

    def close(self):
        for h in self.handles:
            h.remove()


def load_model(model_id: str, dtype_name: str, local_files_only: bool):
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def run_prompt(model, tok, text: str):
    device = next(model.parameters()).device
    inp = tok(text, return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.inference_mode():
        return model(**inp, use_cache=False, return_dict=True)


def pair_delta_l2(model, tok, collector: GateCollector, a: str, b: str) -> float:
    collector.reset()
    run_prompt(model, tok, a)
    va = collector.get_flat()
    collector.reset()
    run_prompt(model, tok, b)
    vb = collector.get_flat()
    return float(np.linalg.norm(va - vb))


def build_report(payload: Dict[str, object]) -> str:
    lines = [
        "# 多维编码交叉消融报告",
        "",
        "## 抑制矩阵（行=消融维度，列=被测维度）",
    ]
    m = payload["suppression_matrix_mean"]
    for row_dim in DIMENSIONS:
        row = m[row_dim]
        lines.append(
            f"- {row_dim}: style={row['style']:.4f}, logic={row['logic']:.4f}, syntax={row['syntax']:.4f}"
        )
    lines.extend(["", "## 对角优势"])
    for k, v in payload["diagonal_advantage"].items():
        lines.append(f"- {k}: {v:.4f}")
    lines.extend(
        [
            "",
            "## 解释",
            "- 对角值越高，表示该维子集对本维对照差分抑制越强。",
            "- 若对角优势显著为正，支持维度级因果特异性。",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument(
        "--probe-json",
        default="tempdata/deepseek7b_multidim_encoding_probe_v2_specific/multidim_encoding_probe.json",
    )
    parser.add_argument("--top-n", type=int, default=64)
    parser.add_argument("--ablate-all-positions", action="store_true", default=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_multidim_causal_ablation_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = json.loads(Path(args.probe_json).read_text(encoding="utf-8"))
    d_ff = int(probe["runtime_config"]["d_ff"])

    subsets: Dict[str, List[int]] = {}
    for dim in DIMENSIONS:
        rows = probe["dimensions"][dim].get("specific_top_neurons") or []
        subsets[dim] = [int(x["flat_index"]) for x in rows[: max(1, int(args.top_n))]]

    pairs_from_probe = {}
    for dim in DIMENSIONS:
        rows = (((probe.get("dimensions") or {}).get(dim) or {}).get("pairs") or [])
        norm_rows = []
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("a"), str) and isinstance(row.get("b"), str):
                norm_rows.append({"id": str(row.get("id", "")), "a": row["a"], "b": row["b"]})
        if norm_rows:
            pairs_from_probe[dim] = norm_rows
    pairs = pairs_from_probe if len(pairs_from_probe) == len(DIMENSIONS) else build_contrast_pairs()
    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    collector = GateCollector(model)
    try:
        baseline = {}
        for dim in DIMENSIONS:
            vals = []
            for p in pairs[dim]:
                vals.append(pair_delta_l2(model, tok, collector, p["a"], p["b"]))
            baseline[dim] = vals

        suppression_matrix = {r: {c: [] for c in DIMENSIONS} for r in DIMENSIONS}
        for ab_dim in DIMENSIONS:
            ab = GateAblator(
                model,
                subsets[ab_dim],
                d_ff=d_ff,
                ablate_all_positions=bool(args.ablate_all_positions),
            )
            try:
                for ev_dim in DIMENSIONS:
                    for i, p in enumerate(pairs[ev_dim]):
                        b = baseline[ev_dim][i]
                        a = pair_delta_l2(model, tok, collector, p["a"], p["b"])
                        if b <= 1e-12:
                            sup = 0.0
                        else:
                            sup = (b - a) / b
                        suppression_matrix[ab_dim][ev_dim].append(float(sup))
            finally:
                ab.close()

        suppression_mean = {
            r: {c: float(np.mean(suppression_matrix[r][c])) if suppression_matrix[r][c] else 0.0 for c in DIMENSIONS}
            for r in DIMENSIONS
        }
        diagonal_adv = {}
        for dim in DIMENSIONS:
            diag = suppression_mean[dim][dim]
            off = [suppression_mean[dim][x] for x in DIMENSIONS if x != dim]
            diagonal_adv[dim] = float(diag - float(np.mean(off)))

        result = {
            "model_id": args.model_id,
            "probe_json": args.probe_json,
            "top_n": int(args.top_n),
            "ablate_all_positions": bool(args.ablate_all_positions),
            "baseline_pair_delta_l2": baseline,
            "suppression_matrix_raw": suppression_matrix,
            "suppression_matrix_mean": suppression_mean,
            "diagonal_advantage": diagonal_adv,
        }

        json_path = out_dir / "multidim_causal_ablation.json"
        md_path = out_dir / "MULTIDIM_CAUSAL_ABLATION_REPORT.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(build_report(result), encoding="utf-8")
        print(json.dumps({"json": json_path.as_posix(), "markdown": md_path.as_posix()}, ensure_ascii=False))
    finally:
        collector.close()


if __name__ == "__main__":
    main()
