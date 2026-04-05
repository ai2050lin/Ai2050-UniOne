#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage600_601_602: 旋转方向功能标注 + 扩大生成测试集 + Gemma4架构差异分析
每次运行一个模型以避免GPU内存问题。

用法: python stage600_601_602_rotation_quality_arch.py [qwen3|deepseek7b|glm4|gemma4]
如果不传参数则运行全部（顺序执行，每个模型间充分清理）。
"""

from __future__ import annotations
import sys, json, time, gc, torch, os
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    load_model_bundle, free_model, discover_layers
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def safe_get_device(model):
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        pass
    try:
        return next(model.model.parameters()).device
    except (StopIteration, AttributeError):
        pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def move_to_device(batch, model):
    device = safe_get_device(model)
    if hasattr(batch, 'to'):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
    return batch


def get_arch_info(model, model_key):
    info = {}
    if hasattr(model, 'config'):
        cfg = model.config
        info["config_num_heads"] = getattr(cfg, 'num_attention_heads', None)
        info["config_num_kv_heads"] = getattr(cfg, 'num_key_value_heads', None)
        info["config_head_dim"] = getattr(cfg, 'head_dim', None)
        info["config_hidden"] = getattr(cfg, 'hidden_size', None)
        info["config_num_layers"] = getattr(cfg, 'num_hidden_layers', None)
        info["config_intermediate"] = getattr(cfg, 'intermediate_size', None)
        info["has_gqa"] = (getattr(cfg, 'num_key_value_heads', None) is not None and
                           getattr(cfg, 'num_key_value_heads', None) != getattr(cfg, 'num_attention_heads', None))
        h = getattr(cfg, 'hidden_size', 0)
        nh = getattr(cfg, 'num_attention_heads', 0)
        info["hidden_dim"] = h
        if h > 0 and nh > 0:
            info["head_dim"] = h // nh
    return info


def compute_semantic_probes(tokenizer, model):
    sem_probes = {}
    word_senses = {
        "bank": ["river", "money", "financial", "water", "loan"],
        "apple": ["fruit", "tree", "orchard", "iPhone", "technology"],
        "plant": ["factory", "manufacturing", "green", "leaf", "botanical"],
        "spring": ["season", "warm", "water", "flow", "source"],
        "nail": ["hammer", "metal", "finger", "manicure", "sharp"],
    }
    unembed = None
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        unembed = model.lm_head.weight.data.float().cpu()
    elif hasattr(model, 'get_output_embeddings'):
        oe = model.get_output_embeddings()
        if oe is not None:
            unembed = oe.weight.data.float().cpu()
    if unembed is None:
        return word_senses, None
    for word, senses in word_senses.items():
        sem_probes[word] = {}
        for sense in senses:
            ids = tokenizer.encode(sense, add_special_tokens=False)
            if ids:
                vecs = [unembed[i].float() for i in ids if i < unembed.shape[0]]
                if vecs:
                    sem_probes[word][sense] = torch.stack(vecs).mean(dim=0)
    return word_senses, sem_probes


# ============ Stage600 ============

def run_stage600(model, tokenizer, model_key):
    print(f"\n  --- Stage600: 旋转方向功能标注 ---")
    t0 = time.time()
    layers = discover_layers(model)
    n_layers = len(layers)
    device = safe_get_device(model)

    disamb_pairs = [
        ("The river bank was muddy.", "The bank approved the loan.", "bank"),
        ("She ate a red apple.", "Apple released the iPhone.", "apple"),
        ("The factory plant employs workers.", "She watered the plant.", "plant"),
        ("The hot spring resort.", "Spring is beautiful.", "spring"),
        ("He hit the nail with a hammer.", "She painted her fingernail.", "nail"),
    ]

    # Find peak disambiguation layer
    peak_layers = {}
    for s1, s2, word in disamb_pairs:
        max_disamb = 0
        best_layer = 0
        for s in [s1, s2]:
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=128)
            enc = move_to_device(enc, model)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            for li, h in enumerate(out.hidden_states):
                hv = h[0, -1, :].float().cpu()
                if s == s1:
                    if word not in peak_layers:
                        peak_layers[word] = {}
                    peak_layers[word]["ctx1"] = peak_layers[word].get("ctx1", {})
                    peak_layers[word]["ctx1"][li] = hv
                else:
                    peak_layers[word]["ctx2"] = peak_layers[word].get("ctx2", {})
                    peak_layers[word]["ctx2"][li] = hv
        for li in range(n_layers):
            if li in peak_layers[word].get("ctx1", {}) and li in peak_layers[word].get("ctx2", {}):
                d = 1 - cos(peak_layers[word]["ctx1"][li], peak_layers[word]["ctx2"][li])
                if d > max_disamb:
                    max_disamb = d
                    best_layer = li
        peak_layers[word]["peak_layer"] = best_layer
        peak_layers[word]["peak_disamb"] = max_disamb

    word_senses, sem_probes = compute_semantic_probes(tokenizer, model)

    rotation_analysis = {}
    for word in peak_layers:
        if word not in ["bank", "apple", "plant", "spring", "nail"]:
            continue
        pl = peak_layers[word]
        peak_l = pl["peak_layer"]
        ctx1_peak = pl["ctx1"][peak_l]
        ctx2_peak = pl["ctx2"][peak_l]
        d_peak = ctx1_peak - ctx2_peak
        d_peak_norm = F.normalize(d_peak, dim=0)

        offsets = {}
        for pct in [25, 50, 75]:
            target_l = min(peak_l + int((n_layers - peak_l) * pct / 100), n_layers - 1)
            if target_l <= peak_l:
                target_l = min(peak_l + 1, n_layers - 1)
            ctx1_t = pl["ctx1"][target_l]
            ctx2_t = pl["ctx2"][target_l]
            d_target = ctx1_t - ctx2_t
            d_target_norm = F.normalize(d_target, dim=0)
            dir_cos = cos(d_peak_norm, d_target_norm)
            proj_parallel = torch.dot(d_target, d_peak_norm)
            v_parallel = proj_parallel * d_peak_norm
            v_orthogonal = d_target - v_parallel
            total_energy = torch.norm(d_target).item() ** 2
            parallel_energy = torch.norm(v_parallel).item() ** 2
            orth_energy = torch.norm(v_orthogonal).item() ** 2
            semantic_alignment = {}
            if sem_probes and word in sem_probes:
                for sense_name, sense_vec in sem_probes[word].items():
                    sense_norm = F.normalize(sense_vec, dim=0)
                    orth_cos = cos(v_orthogonal, sense_norm)
                    semantic_alignment[sense_name] = round(orth_cos, 4)
            offsets[f"+{pct}%"] = {
                "target_layer": target_l,
                "dir_cos": round(dir_cos, 4),
                "total_energy": round(total_energy, 2),
                "parallel_energy": round(parallel_energy, 2),
                "orth_energy": round(orth_energy, 2),
                "orth_ratio": round(orth_energy / max(total_energy, 1e-10), 4),
                "semantic_alignment": semantic_alignment,
            }
        rotation_analysis[word] = {
            "peak_layer": peak_l,
            "peak_disamb": round(pl["peak_disamb"], 4),
            "offsets": offsets,
        }

    elapsed = time.time() - t0
    print(f"  Stage600 done in {elapsed:.1f}s")
    return {"rotation_analysis": rotation_analysis}


# ============ Stage601 ============

def run_stage601(model, tokenizer, model_key):
    print(f"\n  --- Stage601: 扩大生成测试集 ---")
    t0 = time.time()

    gen_tasks = [
        {"word": "bank", "ctx": "river", "prompt": "The river bank was very", "expected_contains": ["muddy", "wet", "steep", "eroded", "sandy", "grass", "overgrown", "rocky"], "template": 1},
        {"word": "bank", "ctx": "finance", "prompt": "The bank approved the loan for", "expected_contains": ["the", "a", "their", "her", "his", "business", "mortgage"], "template": 1},
        {"word": "bank", "ctx": "river", "prompt": "I walked along the river bank and saw", "expected_contains": ["the", "a", "many", "some", "birds", "fish", "water"], "template": 2},
        {"word": "bank", "ctx": "finance", "prompt": "She deposited her savings at the bank", "expected_contains": ["yesterday", "today", "last", "every", "and", "to", "for"], "template": 2},
        {"word": "apple", "ctx": "fruit", "prompt": "She ate a red apple for", "expected_contains": ["breakfast", "lunch", "dinner", "dessert", "a", "snack", "the"], "template": 1},
        {"word": "apple", "ctx": "tech", "prompt": "Apple released the new iPhone", "expected_contains": ["today", "yesterday", "last", "in", "at", "with", "during"], "template": 1},
        {"word": "apple", "ctx": "fruit", "prompt": "The apple pie was delicious and", "expected_contains": ["warm", "sweet", "fresh", "hot", "golden", "homemade"], "template": 2},
        {"word": "apple", "ctx": "tech", "prompt": "Apple announced the new feature at", "expected_contains": ["the", "their", "WWDC", "event", "conference", "a"], "template": 2},
        {"word": "plant", "ctx": "factory", "prompt": "The factory plant employs hundreds of", "expected_contains": ["workers", "people", "employees", "staff", "men", "local"], "template": 1},
        {"word": "plant", "ctx": "botany", "prompt": "She watered the plant in the", "expected_contains": ["garden", "pot", "room", "yard", "corner", "window"], "template": 1},
        {"word": "plant", "ctx": "factory", "prompt": "The power plant generates electricity using", "expected_contains": ["coal", "natural", "solar", "wind", "nuclear", "water", "gas"], "template": 2},
        {"word": "plant", "ctx": "botany", "prompt": "The plant grew tall in the sunny", "expected_contains": ["garden", "field", "yard", "spot", "corner", "area"], "template": 2},
        {"word": "spring", "ctx": "season", "prompt": "Spring is the most beautiful", "expected_contains": ["season", "time", "of", "period", "month"], "template": 1},
        {"word": "spring", "ctx": "water", "prompt": "The hot spring resort attracts many", "expected_contains": ["tourists", "visitors", "people", "guests", "travelers"], "template": 1},
        {"word": "spring", "ctx": "season", "prompt": "In spring the flowers begin to", "expected_contains": ["bloom", "grow", "open", "bud", "appear", "blossom"], "template": 2},
        {"word": "spring", "ctx": "water", "prompt": "The spring water was cold and", "expected_contains": ["clear", "fresh", "clean", "cold", "pure", "refreshing"], "template": 2},
        {"word": "nail", "ctx": "hardware", "prompt": "He hit the nail with a heavy", "expected_contains": ["hammer", "tool", "mallet", "blow", "strike"], "template": 1},
        {"word": "nail", "ctx": "body", "prompt": "She painted her fingernail bright", "expected_contains": ["red", "pink", "blue", "green", "white", "purple"], "template": 1},
        {"word": "nail", "ctx": "hardware", "prompt": "The nail was driven into the wooden", "expected_contains": ["board", "wall", "plank", "beam", "frame", "post"], "template": 2},
        {"word": "nail", "ctx": "body", "prompt": "She went to the salon to get a nail", "expected_contains": ["treatment", "polish", "art", "trim", "manicure", "design"], "template": 2},
        {"word": "bat", "ctx": "animal", "prompt": "The bat flew out of the dark", "expected_contains": ["cave", "night", "sky", "dark", "attic", "hole"], "template": 1},
        {"word": "bat", "ctx": "sports", "prompt": "He swung the baseball bat and hit", "expected_contains": ["the", "a", "home", "hard", "and", "it"], "template": 1},
        {"word": "bat", "ctx": "animal", "prompt": "The bat hung upside down in the", "expected_contains": ["cave", "tree", "attic", "dark", "ceiling", "roof"], "template": 2},
        {"word": "bat", "ctx": "sports", "prompt": "The baseball bat was made of", "expected_contains": ["wood", "aluminum", "metal", "ash", "maple"], "template": 2},
        {"word": "match", "ctx": "fire", "prompt": "He struck a match to light the", "expected_contains": ["candle", "fire", "lamp", "stove", "room", "cigarette"], "template": 1},
        {"word": "match", "ctx": "sports", "prompt": "The football match was exciting and", "expected_contains": ["thrilling", "intense", "close", "great", "fun", "amazing"], "template": 1},
        {"word": "match", "ctx": "fire", "prompt": "She used a match to start the", "expected_contains": ["fire", "campfire", "stove", "candle", "grill"], "template": 2},
        {"word": "match", "ctx": "sports", "prompt": "They watched the tennis match at the", "expected_contains": ["stadium", "arena", "court", "club", "center"], "template": 2},
        {"word": "rock", "ctx": "geology", "prompt": "The rock was very hard and", "expected_contains": ["heavy", "solid", "rough", "dense", "old", "smooth"], "template": 1},
        {"word": "rock", "ctx": "music", "prompt": "The rock concert was loud and", "expected_contains": ["exciting", "amazing", "wild", "incredible", "epic", "great"], "template": 1},
        {"word": "rock", "ctx": "geology", "prompt": "The rock formation was created by", "expected_contains": ["erosion", "volcanic", "millions", "ancient", "natural", "weather"], "template": 2},
        {"word": "rock", "ctx": "music", "prompt": "He played rock music on his", "expected_contains": ["guitar", "band", "drums", "piano", "radio", "speakers"], "template": 2},
        {"word": "light", "ctx": "illumination", "prompt": "The light in the room was very", "expected_contains": ["bright", "dim", "warm", "soft", "harsh", "faint"], "template": 1},
        {"word": "light", "ctx": "weight", "prompt": "The suitcase was very light and", "expected_contains": ["easy", "small", "compact", "simple", "portable", "convenient"], "template": 1},
        {"word": "light", "ctx": "illumination", "prompt": "She turned on the light to see", "expected_contains": ["the", "what", "if", "her", "better", "clearly"], "template": 2},
        {"word": "light", "ctx": "weight", "prompt": "The fabric was light and comfortable for", "expected_contains": ["summer", "warm", "hot", "everyday", "travel", "running"], "template": 2},
        {"word": "ring", "ctx": "jewelry", "prompt": "The diamond ring sparkled in the", "expected_contains": ["light", "sun", "moon", "dark", "candle", "store"], "template": 1},
        {"word": "ring", "ctx": "sound", "prompt": "The phone ring interrupted the", "expected_contains": ["meeting", "call", "conversation", "class", "silence", "quiet"], "template": 1},
        {"word": "ring", "ctx": "jewelry", "prompt": "She wore a gold ring on her", "expected_contains": ["finger", "hand", "left", "ring", "wedding"], "template": 2},
        {"word": "ring", "ctx": "sound", "prompt": "The doorbell ring startled the", "expected_contains": ["dog", "cat", "baby", "family", "children", "everyone"], "template": 2},
    ]

    results = []
    correct = 0
    total = 0
    for task in gen_tasks:
        prompt = task["prompt"]
        expected = task["expected_contains"]
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        enc = move_to_device(enc, model)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=8, do_sample=False, temperature=0.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        gen_words = generated.split()[:3]
        gen_text = " ".join(gen_words)
        is_correct = False
        matched_word = None
        for exp in expected:
            for gw in gen_words:
                if gw.lower().startswith(exp.lower()[:3]) or exp.lower().startswith(gw.lower()[:3]):
                    is_correct = True
                    matched_word = exp
                    break
            if is_correct:
                break
        if is_correct:
            correct += 1
        total += 1
        results.append({
            "word": task["word"], "ctx": task["ctx"], "template": task["template"],
            "prompt": prompt, "generated": gen_text, "expected": expected,
            "matched": matched_word, "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0
    word_acc = {}
    for task in gen_tasks:
        w = task["word"]
        if w not in word_acc:
            word_acc[w] = {"correct": 0, "total": 0}
        word_acc[w]["total"] += 1
    for r in results:
        if r["correct"]:
            word_acc[r["word"]]["correct"] += 1

    elapsed = time.time() - t0
    print(f"  Stage601 done: {correct}/{total} = {accuracy:.1%}, {elapsed:.1f}s")
    return {
        "accuracy": round(accuracy, 4), "correct": correct, "total": total,
        "per_word": {w: f"{v['correct']}/{v['total']}" for w, v in word_acc.items()},
        "details": results,
    }


# ============ Stage602 ============

def run_stage602(model, tokenizer, model_key):
    print(f"\n  --- Stage602: 架构差异分析 ---")
    t0 = time.time()
    layers = discover_layers(model)
    n_layers = len(layers)
    device = safe_get_device(model)
    arch = get_arch_info(model, model_key)

    disamb_pairs = [
        ("The river bank was muddy.", "The bank approved the loan.", "bank"),
        ("She ate a red apple.", "Apple released the iPhone.", "apple"),
        ("The factory plant employs workers.", "She watered the plant.", "plant"),
        ("The hot spring resort.", "Spring is beautiful.", "spring"),
        ("He hit the nail with a hammer.", "She painted her fingernail.", "nail"),
        ("The bat flew out of the dark cave.", "He swung the baseball bat and hit it.", "bat"),
        ("He struck a match to light the candle.", "The football match was exciting.", "match"),
        ("The rock was very hard and heavy.", "The rock concert was loud.", "rock"),
    ]

    layer_data = []
    for s1, s2, word in disamb_pairs:
        h1s, h2s = [], []
        for s in [s1, s2]:
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=128)
            enc = move_to_device(enc, model)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            for h in out.hidden_states:
                hv = h[0, -1, :].float().cpu()
                if s == s1:
                    h1s.append(hv)
                else:
                    h2s.append(hv)
        for li in range(n_layers):
            disamb = 1 - cos(h1s[li], h2s[li])
            d0 = h1s[0] - h2s[0]
            d0_norm = F.normalize(d0, dim=0)
            dli = h1s[li] - h2s[li]
            dli_norm = F.normalize(dli, dim=0)
            dir_cos_l0 = cos(d0_norm, dli_norm)
            mag_ratio = torch.norm(dli).item() / max(torch.norm(d0).item(), 1e-10)
            h_norm = torch.norm(h1s[li]).item()
            layer_data.append({
                "word": word, "layer": li, "disamb": round(disamb, 6),
                "dir_cos_l0": round(dir_cos_l0, 6), "mag_ratio": round(mag_ratio, 4),
                "h_norm": round(h_norm, 2),
            })

    layer_summary = {}
    for ld in layer_data:
        li = ld["layer"]
        if li not in layer_summary:
            layer_summary[li] = {"disamb": [], "dir_cos_l0": [], "mag_ratio": [], "h_norm": []}
        layer_summary[li]["disamb"].append(ld["disamb"])
        layer_summary[li]["dir_cos_l0"].append(ld["dir_cos_l0"])
        layer_summary[li]["mag_ratio"].append(ld["mag_ratio"])
        layer_summary[li]["h_norm"].append(ld["h_norm"])

    layer_avg = {}
    for li in sorted(layer_summary.keys()):
        ls = layer_summary[li]
        layer_avg[str(li)] = {
            "disamb": round(np.mean(ls["disamb"]), 6),
            "dir_cos_l0": round(np.mean(ls["dir_cos_l0"]), 6),
            "mag_ratio": round(np.mean(ls["mag_ratio"]), 4),
            "h_norm": round(np.mean(ls["h_norm"]), 2),
        }

    peak_layer = max(layer_avg.items(), key=lambda x: x[1]["disamb"])
    best_dir_layer = max(
        [(k, v) for k, v in layer_avg.items() if int(k) > 0],
        key=lambda x: x[1]["dir_cos_l0"]
    )
    worst_dir_layer = min(
        [(k, v) for k, v in layer_avg.items() if int(k) > 0],
        key=lambda x: x[1]["dir_cos_l0"]
    )
    final_disamb = layer_avg[str(n_layers - 1)]["disamb"]

    elapsed = time.time() - t0
    print(f"  Stage602 done: peak=L{peak_layer[0]}({peak_layer[1]['disamb']:.4f}), final={final_disamb:.4f}, {elapsed:.1f}s")
    return {
        "architecture": arch, "n_layers": n_layers,
        "peak_layer": int(peak_layer[0]), "peak_disamb": round(peak_layer[1]["disamb"], 4),
        "best_dir_layer": int(best_dir_layer[0]), "best_dir_cos": round(best_dir_layer[1]["dir_cos_l0"], 4),
        "worst_dir_layer": int(worst_dir_layer[0]), "worst_dir_cos": round(worst_dir_layer[1]["dir_cos_l0"], 4),
        "final_disamb": round(final_disamb, 4),
        "layer_avg": layer_avg,
    }


# ============ Main ============

MODEL_KEYS = ["qwen3", "deepseek7b", "glm4", "gemma4"]


def run_single_model(mk):
    """Run all stages for a single model, return results dict."""
    print(f"\n{'='*60}")
    print(f"  Loading {mk}...")
    print(f"{'='*60}")
    t0 = time.time()
    model, tokenizer = load_model_bundle(mk)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    try:
        s600 = run_stage600(model, tokenizer, mk)
        s601 = run_stage601(model, tokenizer, mk)
        s602 = run_stage602(model, tokenizer, mk)
        result = {"stage600": s600, "stage601": s601, "stage602": s602}
    except Exception as e:
        import traceback
        print(f"  ERROR in {mk}: {e}")
        traceback.print_exc()
        result = {"error": str(e)}
    finally:
        free_model(model)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return result


def main():
    # If command line arg provided, run only that model
    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        if target not in MODEL_KEYS:
            print(f"Unknown model: {target}. Use one of: {MODEL_KEYS}")
            return
        models_to_run = [target]
    else:
        models_to_run = MODEL_KEYS

    # Load existing results if available
    combined_path = OUTPUT_DIR / f"stage600_601_602_combined_{TIMESTAMP}.json"
    combined = {"timestamp": TIMESTAMP, "models": {}}

    # Check if there's a recent combined file to resume from
    existing_files = sorted(OUTPUT_DIR.glob("stage600_601_602_combined_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if existing_files and len(sys.argv) == 1:
        try:
            with open(existing_files[0], "r", encoding="utf-8") as f:
                prev = json.load(f)
            combined = prev
            combined_path = existing_files[0]
            print(f"Resuming from {existing_files[0].name}")
        except:
            pass

    for mk in models_to_run:
        # Skip if already completed successfully
        if mk in combined["models"] and "error" not in combined["models"][mk]:
            print(f"\n  Skipping {mk} (already completed)")
            continue

        result = run_single_model(mk)
        combined["models"][mk] = result

        # Save after each model
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"  Results saved to {combined_path}")

        # Wait between models for GPU cooldown
        if mk != models_to_run[-1]:
            time.sleep(5)

    print(f"\nAll done. Results: {combined_path}")


if __name__ == "__main__":
    main()
