#!/usr/bin/env python
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_CACHE_ROOT = Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B")
OUTPUT_DIR = Path(r"D:\develop\TransformerLens-main\tests\codex_temp")


def resolve_snapshot_path() -> Path:
    ref_file = MODEL_CACHE_ROOT / "refs" / "main"
    if not ref_file.exists():
        raise FileNotFoundError(f"Missing ref file: {ref_file}")
    revision = ref_file.read_text(encoding="utf-8").strip()
    snapshot = MODEL_CACHE_ROOT / "snapshots" / revision
    if not snapshot.exists():
        raise FileNotFoundError(f"Missing snapshot dir: {snapshot}")
    required = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing = [name for name in required if not (snapshot / name).exists()]
    if missing:
        raise FileNotFoundError(f"Snapshot incomplete, missing: {missing}")
    shard_count = len(list(snapshot.glob("model-*.safetensors")))
    if shard_count < 2:
        raise FileNotFoundError(f"Expected at least 2 safetensor shards, got {shard_count}")
    return snapshot


def cuda_info() -> dict:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        info.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "total_memory_gb": round(props.total_memory / 1024**3, 2),
                "free_memory_gb_before_load": round(free_bytes / 1024**3, 2),
                "total_memory_gb_from_mem_get_info": round(total_bytes / 1024**3, 2),
            }
        )
    return info


def main() -> int:
    ts = time.strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"deepseek_cuda_runtime_status_{ts}.json"

    result = {
        "timestamp": ts,
        "model_cache_root": str(MODEL_CACHE_ROOT),
        "snapshot_path": None,
        "status": "started",
        "cuda": cuda_info(),
        "load_seconds": None,
        "generate_seconds": None,
        "hf_device_map": None,
        "prompt": "Write one short sentence answering: what is 2+2?",
        "generated_text": None,
        "error": None,
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in torch")

        snapshot_path = resolve_snapshot_path()
        result["snapshot_path"] = str(snapshot_path)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        load_start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            snapshot_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        model.eval()
        result["load_seconds"] = round(time.time() - load_start, 3)
        result["hf_device_map"] = {str(k): str(v) for k, v in getattr(model, "hf_device_map", {}).items()}

        inputs = tokenizer(result["prompt"], return_tensors="pt")
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

        gen_start = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        result["generate_seconds"] = round(time.time() - gen_start, 3)
        result["generated_text"] = tokenizer.decode(outputs[0], skip_special_tokens=True)

        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        result["cuda"].update(
            {
                "free_memory_gb_after_run": round(free_bytes / 1024**3, 2),
                "peak_reserved_gb": round(torch.cuda.max_memory_reserved(0) / 1024**3, 2),
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated(0) / 1024**3, 2),
                "total_memory_gb_after_run": round(total_bytes / 1024**3, 2),
            }
        )
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
