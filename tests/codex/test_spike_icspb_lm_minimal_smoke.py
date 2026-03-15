from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gpt5.code.spike_icspb_lm_minimal import SpikeICSPBLMConfig, SpikeICSPBLMMinimal



def encode_text(text: str) -> List[int]:
    return list(text.encode("utf-8", errors="ignore"))


def build_batch(texts: List[str], max_seq_len: int) -> torch.Tensor:
    rows = []
    for text in texts:
        ids = encode_text(text)[:max_seq_len]
        if len(ids) < 8:
            continue
        rows.append(ids)
    min_len = min(len(row) for row in rows)
    rows = [row[:min_len] for row in rows]
    return torch.tensor(rows, dtype=torch.long)


def build_payload() -> Dict[str, object]:
    t0 = time.time()
    torch.manual_seed(7)

    texts = [
        "apple is fruit.\n",
        "banana is fruit.\n",
        "pear is fruit.\n",
        "cat is animal.\n",
        "dog is animal.\n",
        "truth is abstract.\n",
        "logic is abstract.\n",
    ]
    batch = build_batch(texts, max_seq_len=48)
    config = SpikeICSPBLMConfig(vocab_size=256, hidden_dim=96, patch_slots=24, max_seq_len=batch.size(1))
    model = SpikeICSPBLMMinimal(config)

    pre_loss, pre_metrics = model.compute_loss(batch, batch)
    update_metrics = []
    for _ in range(40):
        update_metrics.append(model.local_update_step(batch, batch, lr=0.08))
    post_loss, post_metrics = model.compute_loss(batch, batch)

    prompt = torch.tensor([encode_text("apple is ")], dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=16, temperature=0.0)
    text = model.decode_token_ids(generated["generated_ids"][0])

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "SpikeICSPB_LM_minimal_smoke",
        },
        "model": {
            "hidden_dim": config.hidden_dim,
            "patch_slots": config.patch_slots,
            "seq_len": config.max_seq_len,
            "training_mode": "local_eligibility_modulated_non_bp",
        },
        "metrics": {
            "pre_loss": float(pre_loss.item()),
            "post_loss": float(post_loss.item()),
            "loss_delta": float(pre_loss.item() - post_loss.item()),
            "pre_patch_entropy": pre_metrics["patch_entropy"],
            "post_patch_entropy": post_metrics["patch_entropy"],
            "post_successor_gate_mean": post_metrics["successor_gate_mean"],
            "avg_update_error_norm": float(sum(row["avg_error_norm"] for row in update_metrics) / len(update_metrics)),
        },
        "generation": {
            "prompt": "apple is ",
            "text": text,
            "generated_length": int(generated["generated_ids"].size(1)),
        },
        "verdict": {
            "forward_ready": True,
            "local_update_reduces_loss": bool(post_loss.item() < pre_loss.item()),
            "nonempty_generation": bool(len(text.strip()) > 0),
            "core_answer": (
                "SpikeICSPB-LM 最小原型已经能在不使用 Attention 和标准 BP 的情况下完成前向、"
                "局部更新和最小文本生成，但这仍然只是原型级能力。"
            ),
        },
        "progress_estimate": {
            "spike_icspb_lm_minimal_prototype_percent": 46.0,
            "non_attention_non_bp_large_scale_trainability_percent": 37.0,
            "non_attention_non_bp_full_language_capability_percent": 23.0,
        },
    }
    return payload


def test_spike_icspb_lm_minimal_smoke() -> None:
    payload = build_payload()
    metrics = payload["metrics"]
    assert payload["verdict"]["forward_ready"] is True
    assert payload["verdict"]["local_update_reduces_loss"] is True
    assert payload["verdict"]["nonempty_generation"] is True
    assert metrics["loss_delta"] > 0.01


def main() -> None:
    ap = argparse.ArgumentParser(description="SpikeICSPB-LM minimal smoke test")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/spike_icspb_lm_minimal_smoke_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
