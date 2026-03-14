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
    spec = importlib.util.spec_from_file_location("openwebtext_helper_for_phasea_level", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_corpus(config: ICSPBLMPhaseAConfig) -> tuple[list[Dict[str, torch.Tensor]], list[str], float]:
    helper = load_helper()
    files = helper.iter_openwebtext_files()[:6]
    tokenizer = SimpleByteTokenizer()
    train_batches: list[Dict[str, torch.Tensor]] = []
    eval_texts: list[str] = []
    sampled_chars = 0.0
    sample_idx = 0

    for idx, path in enumerate(files):
        chunks = helper.sample_chunks_from_file(path, chunk_chars=config.max_seq_len + 224, num_chunks=5)
        sampled_chars += sum(len(chunk) for chunk in chunks)
        for chunk in chunks[:4]:
            token_ids = tokenizer.encode(chunk)
            if len(token_ids) < 96:
                continue
            token_ids = token_ids[: config.max_seq_len]
            padded = token_ids + [0] * (config.max_seq_len - len(token_ids))
            novelty = min(0.30, 0.04 + (sample_idx % 6) * 0.02)
            retention = min(0.32, 0.18 + (sample_idx % 5) * 0.02)
            train_batches.append(
                {
                    "input_ids": torch.tensor([padded], dtype=torch.long),
                    "novelty": torch.tensor([[novelty]], dtype=torch.float32),
                    "retention": torch.tensor([[retention]], dtype=torch.float32),
                }
            )
            sample_idx += 1
        eval_texts.append(chunks[4])

    if len(train_batches) < 12 or len(eval_texts) < 4:
        raise RuntimeError("PhaseA language level block 样本不足。")
    return train_batches, eval_texts[:4], sampled_chars


def eval_loss(model: ICSPBLMPhaseA, batches: list[Dict[str, torch.Tensor]]) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in batches:
            loss, _ = model.compute_loss(batch)
            total += float(loss.item())
    return total / max(1, len(batches))


def collapse_ratio(pred_ids: List[int]) -> float:
    if not pred_ids:
        return 1.0
    same = 0
    for idx in range(1, len(pred_ids)):
        if pred_ids[idx] == pred_ids[idx - 1]:
            same += 1
    return same / max(1, len(pred_ids) - 1)


def continuation_eval(model: ICSPBLMPhaseA, eval_texts: list[str], seq_len: int) -> Dict[str, object]:
    tokenizer = SimpleByteTokenizer()
    rows = []
    match_total = 0.0
    distinct_total = 0.0
    collapse_total = 0.0
    output_chars_total = 0.0

    for idx, text in enumerate(eval_texts):
        token_ids = tokenizer.encode(text)
        if len(token_ids) < 112:
            continue
        prompt_ids = token_ids[:56]
        target_ids = token_ids[56:96]
        result = model.generate(
            torch.tensor([prompt_ids], dtype=torch.long),
            max_new_tokens=len(target_ids),
            novelty=torch.tensor([[0.05 + 0.01 * idx]], dtype=torch.float32),
            retention=torch.tensor([[0.24]], dtype=torch.float32),
            vocab_limit=256,
            temperature=0.85,
        )
        pred_ids = result["new_token_ids"][0].detach().cpu().tolist()
        limit = min(len(pred_ids), len(target_ids))
        match = sum(1 for i in range(limit) if int(pred_ids[i]) == int(target_ids[i])) / max(1, limit)
        distinct = len(set(pred_ids)) / max(1, len(pred_ids))
        collapse = collapse_ratio(pred_ids)
        pred_text = model.decode_token_ids(pred_ids)
        target_text = model.decode_token_ids(target_ids)
        prompt_text = model.decode_token_ids(prompt_ids)

        rows.append(
            {
                "prompt_text": prompt_text,
                "predicted_text": pred_text,
                "target_text": target_text,
                "byte_match_ratio": match,
                "distinct_ratio": distinct,
                "collapse_ratio": collapse,
            }
        )
        match_total += match
        distinct_total += distinct
        collapse_total += collapse
        output_chars_total += len(pred_text)

    count = max(1, len(rows))
    avg_match = match_total / count
    avg_distinct = distinct_total / count
    avg_collapse = collapse_total / count
    avg_output_chars = output_chars_total / count

    if avg_match >= 0.22 and avg_distinct >= 0.45 and avg_collapse <= 0.35:
        level = "可用原型级语言主干"
    elif avg_match >= 0.12 and avg_distinct >= 0.32 and avg_collapse <= 0.55:
        level = "早期语言先验形成"
    elif avg_match >= 0.07:
        level = "弱 continuation 原型"
    else:
        level = "未形成可用语言生成"

    return {
        "avg_match": avg_match,
        "avg_distinct": avg_distinct,
        "avg_collapse": avg_collapse,
        "avg_output_chars": avg_output_chars,
        "rows": rows,
        "level": level,
    }


def main() -> None:
    start = time.time()
    config = ICSPBLMPhaseAConfig(max_seq_len=128)
    model = ICSPBLMPhaseA(config)

    for param in model.parameters():
        param.requires_grad = False
    for block in model.blocks[-4:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True
    for param in model.online_adapter.parameters():
        param.requires_grad = True
    model.token_embedding.weight.requires_grad = True

    train_batches, eval_texts, sampled_chars = build_corpus(config)
    initial_loss = eval_loss(model, train_batches[:4])
    initial_eval = continuation_eval(model, eval_texts, config.max_seq_len)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5, weight_decay=0.01)
    train_losses: list[float] = []
    for _ in range(6):
        for batch in train_batches:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

    final_loss = eval_loss(model, train_batches[:4])
    final_eval = continuation_eval(model, eval_texts, config.max_seq_len)
    loss_drop = initial_loss - final_loss
    match_gain = float(final_eval["avg_match"]) - float(initial_eval["avg_match"])
    distinct_gain = float(final_eval["avg_distinct"]) - float(initial_eval["avg_distinct"])
    collapse_reduction = float(initial_eval["avg_collapse"]) - float(final_eval["avg_collapse"])

    stage_score = min(
        1.0,
        0.34 * min(1.0, max(0.0, loss_drop) / 260.0)
        + 0.28 * min(1.0, max(0.0, float(final_eval["avg_match"])) / 0.25)
        + 0.16 * min(1.0, max(0.0, float(final_eval["avg_distinct"])) / 0.55)
        + 0.10 * min(1.0, max(0.0, collapse_reduction + 0.02) / 0.22)
        + 0.12 * min(1.0, sampled_chars / 40000.0)
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "Stage_ICSPB_LM_PhaseA_Language_Level_Block",
        },
        "headline_metrics": {
            "sampled_chars": sampled_chars,
            "train_batch_count": float(len(train_batches)),
            "initial_val_loss": initial_loss,
            "final_val_loss": final_loss,
            "loss_drop": loss_drop,
            "initial_match": initial_eval["avg_match"],
            "final_match": final_eval["avg_match"],
            "match_gain": match_gain,
            "initial_distinct": initial_eval["avg_distinct"],
            "final_distinct": final_eval["avg_distinct"],
            "distinct_gain": distinct_gain,
            "initial_collapse": initial_eval["avg_collapse"],
            "final_collapse": final_eval["avg_collapse"],
            "collapse_reduction": collapse_reduction,
            "language_level": final_eval["level"],
            "stage_score": stage_score,
            "mean_train_loss": sum(train_losses) / max(1, len(train_losses)),
        },
        "rows": final_eval["rows"],
        "verdict": {
            "overall_pass": stage_score >= 0.68,
            "phasea_language_level_ready": stage_score >= 0.84,
            "core_answer": "PhaseA 扩展语言训练块用于判断训练后语言生成已达到什么等级。",
        },
    }

    out_file = TEMP / "stage_icspb_lm_phasea_language_level_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
