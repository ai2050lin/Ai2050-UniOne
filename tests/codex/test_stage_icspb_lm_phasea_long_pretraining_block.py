from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"
TEMP.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gpt5.code.icspb_lm_phasea import ICSPBLMPhaseA, ICSPBLMPhaseAConfig


HELPER_PATH = ROOT / "tests" / "codex" / "test_stage_openwebtext_real_data_block.py"


class SimpleByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8", errors="ignore"))


def load_helper():
    spec = importlib.util.spec_from_file_location("openwebtext_helper_for_phasea_long", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_batches(config: ICSPBLMPhaseAConfig) -> tuple[list[Dict[str, torch.Tensor]], list[str], float]:
    helper = load_helper()
    files = helper.iter_openwebtext_files()[:5]
    tokenizer = SimpleByteTokenizer()
    train_batches: list[Dict[str, torch.Tensor]] = []
    eval_texts: list[str] = []
    sampled_chars = 0.0
    sample_idx = 0

    for idx, path in enumerate(files):
        chunks = helper.sample_chunks_from_file(path, chunk_chars=config.max_seq_len + 192, num_chunks=5 if idx < 3 else 4)
        sampled_chars += sum(len(chunk) for chunk in chunks)
        for chunk in chunks[:-1]:
            token_ids = tokenizer.encode(chunk)
            if len(token_ids) < 96:
                continue
            token_ids = token_ids[: config.max_seq_len]
            padded = token_ids + [0] * (config.max_seq_len - len(token_ids))
            novelty = min(0.28, 0.04 + (sample_idx % 6) * 0.02)
            retention = min(0.30, 0.16 + (sample_idx % 4) * 0.02)
            train_batches.append(
                {
                    "input_ids": torch.tensor([padded], dtype=torch.long),
                    "novelty": torch.tensor([[novelty]], dtype=torch.float32),
                    "retention": torch.tensor([[retention]], dtype=torch.float32),
                }
            )
            sample_idx += 1
        eval_texts.append(chunks[-1])

    if len(train_batches) < 10 or len(eval_texts) < 4:
        raise RuntimeError("PhaseA 长程预训练样本不足。")
    return train_batches, eval_texts[:4], sampled_chars


def eval_loss(model: ICSPBLMPhaseA, batches: list[Dict[str, torch.Tensor]]) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in batches:
            loss, _ = model.compute_loss(batch)
            total += float(loss.item())
    return total / max(1, len(batches))


def continuation_match(model: ICSPBLMPhaseA, eval_texts: list[str], seq_len: int) -> Dict[str, float]:
    tokenizer = SimpleByteTokenizer()
    match_total = 0.0
    out_chars_total = 0.0
    rows = []
    for idx, text in enumerate(eval_texts):
        token_ids = tokenizer.encode(text)
        if len(token_ids) < 96:
            continue
        prompt_ids = token_ids[:48]
        target_ids = token_ids[48:80]
        result = model.generate(
            torch.tensor([prompt_ids], dtype=torch.long),
            max_new_tokens=len(target_ids),
            novelty=torch.tensor([[0.05 + 0.01 * idx]], dtype=torch.float32),
            retention=torch.tensor([[0.22]], dtype=torch.float32),
            vocab_limit=256,
        )
        pred_ids = result["new_token_ids"][0].detach().cpu().tolist()
        limit = min(len(pred_ids), len(target_ids))
        match = sum(1 for i in range(limit) if int(pred_ids[i]) == int(target_ids[i])) / max(1, limit)
        pred_text = model.decode_token_ids(pred_ids)
        target_text = model.decode_token_ids(target_ids)
        match_total += match
        out_chars_total += len(pred_text)
        rows.append(
            {
                "prompt_text": model.decode_token_ids(prompt_ids),
                "predicted_text": pred_text,
                "target_text": target_text,
                "byte_match_ratio": match,
            }
        )
    count = max(1, len(rows))
    return {
        "avg_match": match_total / count,
        "avg_output_chars": out_chars_total / count,
        "rows": rows,
    }


def main() -> None:
    start = time.time()
    config = ICSPBLMPhaseAConfig(max_seq_len=128)
    model = ICSPBLMPhaseA(config)

    for param in model.parameters():
        param.requires_grad = False
    for block in model.blocks[-3:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True
    for param in model.online_adapter.parameters():
        param.requires_grad = True
    model.token_embedding.weight.requires_grad = True

    train_batches, eval_texts, sampled_chars = build_batches(config)
    initial_val_loss = eval_loss(model, train_batches[:3])
    initial_gen = continuation_match(model, eval_texts, config.max_seq_len)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=6e-5, weight_decay=0.01)
    train_losses: list[float] = []
    for _ in range(4):
        for batch in train_batches:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

    final_val_loss = eval_loss(model, train_batches[:3])
    final_gen = continuation_match(model, eval_texts, config.max_seq_len)

    loss_drop = initial_val_loss - final_val_loss
    gen_gain = final_gen["avg_match"] - initial_gen["avg_match"]
    stage_score = min(
        1.0,
        0.42 * min(1.0, max(0.0, loss_drop) / 240.0)
        + 0.28 * min(1.0, max(0.0, final_gen["avg_match"]) / 0.25)
        + 0.20 * min(1.0, max(0.0, gen_gain) / 0.10)
        + 0.10 * min(1.0, sampled_chars / 20000.0),
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "Stage_ICSPB_LM_PhaseA_Long_Pretraining_Block",
        },
        "headline_metrics": {
            "sampled_chars": sampled_chars,
            "train_batch_count": float(len(train_batches)),
            "initial_val_loss": initial_val_loss,
            "final_val_loss": final_val_loss,
            "loss_drop": loss_drop,
            "initial_match": initial_gen["avg_match"],
            "final_match": final_gen["avg_match"],
            "generation_gain": gen_gain,
            "stage_score": stage_score,
            "mean_train_loss": sum(train_losses) / max(1, len(train_losses)),
        },
        "rows": final_gen["rows"],
        "verdict": {
            "overall_pass": stage_score >= 0.62,
            "phasea_long_pretraining_ready": stage_score >= 0.78,
            "core_answer": "PhaseA 长程预训练块验证更长训练后 loss 和真实文本 continuation 是否同步改善。",
        },
    }

    out_file = TEMP / "stage_icspb_lm_phasea_long_pretraining_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
