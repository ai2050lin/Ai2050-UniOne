#!/usr/bin/env python3
"""
P53: Single Token Contribution Analysis (Stage699)

P50-P52 confirmed:
- delta-h directions carry semantic info (P50)
- Linear accumulation: logits = sum(delta_h_l @ U.T) (P52, INV-354)

P53 asks: HOW does changing a single token affect delta-h at each layer?

Method:
1. Use template sentences with "slots" (e.g., "The [animal] sat on the [furniture]")
2. For each slot, try N alternative tokens
3. Measure delta-h change at each layer: ||delta_h(token_A) - delta_h(token_B)||
4. Also measure the direction of change in logit space

Key questions:
- Are early layers sensitive to syntax tokens (determiners, prepositions)?
- Are later layers sensitive to content tokens (nouns, verbs)?
- Is there a "critical position" where token replacement has maximum impact?
- Does the magnitude of delta-h change correlate with semantic distance of the replacement?

This tells us which tokens control which layers — essential for building the prediction model (P54).
"""
import sys, math, time, gc, json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path
from collections import defaultdict

# === Config ===
OUTPUT_DIR = _Path(f"tests/glm5_temp/stage699_token_contribution_{time.strftime('%Y%m%d_%H%M')}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}

# Template texts with [SLOT] markers
# Each template has a description, the template string, and alternatives for each slot
TEMPLATES = [
    {
        "id": "T1",
        "desc": "Animal-action template",
        "template": "The {slot0} jumped over the {slot1}",
        "slots": {
            0: ["cat", "dog", "fox", "deer", "wolf"],
            1: ["fence", "wall", "gate", "rock", "hedge"],
        },
    },
    {
        "id": "T2",
        "desc": "Person-profession template",
        "template": "The {slot0} works as a {slot1} in the city",
        "slots": {
            0: ["doctor", "teacher", "artist", "engineer", "lawyer"],
            1: ["hospital", "school", "studio", "factory", "court"],
        },
    },
    {
        "id": "T3",
        "desc": "Emotion-adjective template",
        "template": "She felt {slot0} after reading the {slot1} book",
        "slots": {
            0: ["happy", "sad", "angry", "excited", "confused"],
            1: ["long", "short", "old", "new", "strange"],
        },
    },
    {
        "id": "T4",
        "desc": "Location-activity template",
        "template": "We went to the {slot0} to {slot1} last weekend",
        "slots": {
            0: ["beach", "mountain", "park", "museum", "library"],
            1: ["swim", "hike", "walk", "learn", "read"],
        },
    },
]


class Logger:
    def __init__(self, path):
        self.path = _Path(path)
        self.f = open(self.path, "w", encoding="utf-8")
    def __call__(self, msg):
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()
    def close(self):
        self.f.close()


def load_model(model_name):
    """Load model and tokenizer using HuggingFace."""
    model_path = MODEL_MAP[model_name]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def get_delta_h(model, tokenizer, text):
    """Extract per-layer delta-h = h_l - h_{l-1} for all layers."""
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple of (1, seq_len, d_model)
    # Get last token states
    states = [hs[0, -1, :].float().cpu() for hs in hidden_states]
    num_layers = len(states) - 1
    deltas = []
    for l in range(1, num_layers + 1):
        deltas.append(states[l] - states[l - 1])
    return torch.stack(deltas)  # (num_layers, d_model)


def get_logits(model, tokenizer, text):
    """Get logit distribution for the text."""
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        outputs = model(tokens)
    logits = outputs.logits[0, -1, :].float().cpu()  # (vocab_size,)
    return logits


def analyze_template(model, tokenizer, template, log):
    """Analyze single token contribution for one template."""
    tpl_str = template["template"]
    slots = template["slots"]
    tpl_id = template["id"]

    log(f"  Template {tpl_id}: {template['desc']}")
    log(f"    Pattern: {tpl_str}")

    results = {
        "template_id": tpl_id,
        "desc": template["desc"],
        "slot_analyses": [],
    }

    for slot_idx, alternatives in slots.items():
        log(f"\n    Slot {slot_idx} ({len(alternatives)} alternatives):")

        # Compute delta-h for each alternative
        all_deltas = []
        all_logits = []
        slot_texts = []

        for alt in alternatives:
            text = tpl_str.replace("{slot" + str(slot_idx) + "}", alt)
            # Also fill other slots with their first alternative
            for other_idx, other_alts in slots.items():
                if other_idx != slot_idx:
                    text = text.replace("{slot" + str(other_idx) + "}", other_alts[0])

            slot_texts.append(text)
            deltas = get_delta_h(model, tokenizer, text)  # (num_layers, d_model)
            logits = get_logits(model, tokenizer, text)  # (vocab_size,)
            all_deltas.append(deltas)
            all_logits.append(logits)

        num_layers = all_deltas[0].shape[0]

        # Compute pairwise delta-h distances for each layer
        layer_sensitivities = []
        for l in range(num_layers):
            pairwise_dists = []
            pairwise_cos_dists = []
            for i in range(len(alternatives)):
                for j in range(i + 1, len(alternatives)):
                    d_i = all_deltas[i][l]
                    d_j = all_deltas[j][l]
                    # L2 distance of delta-h
                    dist = torch.norm(d_i - d_j).item()
                    # Cosine distance of delta-h
                    cos = F.cosine_similarity(d_i.unsqueeze(0), d_j.unsqueeze(0)).item()
                    pairwise_dists.append(dist)
                    pairwise_cos_dists.append(1.0 - cos)
            layer_sensitivities.append({
                "layer": l,
                "mean_l2_dist": float(np.mean(pairwise_dists)),
                "std_l2_dist": float(np.std(pairwise_dists)),
                "max_l2_dist": float(np.max(pairwise_dists)),
                "mean_cos_dist": float(np.mean(pairwise_cos_dists)),
                "max_cos_dist": float(np.max(pairwise_cos_dists)),
            })

        # Compute logit divergence (KL divergence between alternative logits)
        logit_divergences = []
        for i in range(len(alternatives)):
            for j in range(i + 1, len(alternatives)):
                p = F.softmax(all_logits[i], dim=-1)
                q = F.softmax(all_logits[j], dim=-1)
                kl = F.kl_div(q.log(), p, reduction='sum').item()
                logit_divergences.append(kl)
        mean_kl = float(np.mean(logit_divergences))

        # Top token changes
        top1_tokens = []
        for i, logits in enumerate(all_logits):
            top1_idx = logits.argmax().item()
            top1_tokens.append(tokenizer.decode([top1_idx]).strip())

        # Find most sensitive layer
        most_sensitive = max(layer_sensitivities, key=lambda x: x["mean_cos_dist"])
        least_sensitive = min(layer_sensitivities, key=lambda x: x["mean_cos_dist"])

        slot_result = {
            "slot_idx": slot_idx,
            "alternatives": alternatives,
            "num_pairs": len(alternatives) * (len(alternatives) - 1) // 2,
            "mean_kl_divergence": mean_kl,
            "top1_tokens": top1_tokens,
            "top1_agreement": sum(1 for t in top1_tokens[1:] if t == top1_tokens[0]) / len(top1_tokens) * 100,
            "most_sensitive_layer": most_sensitive["layer"],
            "most_sensitive_cos_dist": most_sensitive["mean_cos_dist"],
            "least_sensitive_layer": least_sensitive["layer"],
            "least_sensitive_cos_dist": least_sensitive["mean_cos_dist"],
            "layer_sensitivities": layer_sensitivities,
        }

        results["slot_analyses"].append(slot_result)

        log(f"      KL divergence: {mean_kl:.4f}")
        log(f"      Top1 agreement: {slot_result['top1_agreement']:.1f}%")
        log(f"      Top1 tokens: {top1_tokens}")
        log(f"      Most sensitive layer: L{most_sensitive['layer']} (cos_dist={most_sensitive['mean_cos_dist']:.4f})")
        log(f"      Least sensitive layer: L{least_sensitive['layer']} (cos_dist={least_sensitive['mean_cos_dist']:.4f})")

    return results


def cross_template_analysis(all_results, log):
    """Analyze patterns across all templates."""
    log("\n\n" + "=" * 60)
    log("Cross-Template Analysis")
    log("=" * 60)

    for model_name, model_results in all_results.items():
        log(f"\n--- {model_name} ---")

        # Collect most/least sensitive layers per slot
        early_sensitive_count = 0
        mid_sensitive_count = 0
        late_sensitive_count = 0

        for tpl_result in model_results["templates"]:
            for slot_result in tpl_result["slot_analyses"]:
                layer = slot_result["most_sensitive_layer"]
                num_layers = len(slot_result["layer_sensitivities"])
                third = num_layers // 3
                if layer < third:
                    early_sensitive_count += 1
                elif layer < 2 * third:
                    mid_sensitive_count += 1
                else:
                    late_sensitive_count += 1

        total = early_sensitive_count + mid_sensitive_count + late_sensitive_count
        log(f"  Sensitive layer distribution (n={total}):")
        log(f"    Early (0-{num_layers//3}): {early_sensitive_count} ({early_sensitive_count/total*100:.0f}%)")
        log(f"    Mid ({num_layers//3}-{2*num_layers//3}): {mid_sensitive_count} ({mid_sensitive_count/total*100:.0f}%)")
        log(f"    Late ({2*num_layers//3}-{num_layers}): {late_sensitive_count} ({late_sensitive_count/total*100:.0f}%)")

        # Summary: which slot position is most impactful?
        slot0_kls = []
        slot1_kls = []
        for tpl_result in model_results["templates"]:
            for i, slot_result in enumerate(tpl_result["slot_analyses"]):
                if slot_result["slot_idx"] == 0:
                    slot0_kls.append(slot_result["mean_kl_divergence"])
                else:
                    slot1_kls.append(slot_result["mean_kl_divergence"])

        if slot0_kls:
            log(f"  Slot 0 (content word) mean KL: {np.mean(slot0_kls):.4f}")
        if slot1_kls:
            log(f"  Slot 1 (content word) mean KL: {np.mean(slot1_kls):.4f}")


def main():
    log = Logger(OUTPUT_DIR / "results.log")
    log(f"P53: Single Token Contribution Analysis")
    log(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Output: {OUTPUT_DIR}")

    all_model_results = {}

    for model_name in MODEL_MAP:
        log(f"\n{'='*60}")
        log(f"Processing model: {model_name}")
        log(f"{'='*60}")
        t0 = time.time()

        try:
            model, tokenizer = load_model(model_name)
            log(f"Loaded in {time.time()-t0:.1f}s")

            model_results = {"model": model_name, "templates": []}

            for template in TEMPLATES:
                tpl_result = analyze_template(model, tokenizer, template, log)
                model_results["templates"].append(tpl_result)

            all_model_results[model_name] = model_results
            log(f"\n{model_name} done in {time.time()-t0:.1f}s")

            # Save per-model JSON
            safe_result = {
                "model": model_name,
                "templates": [],
            }
            for tpl in model_results["templates"]:
                safe_tpl = {
                    "template_id": tpl["template_id"],
                    "desc": tpl["desc"],
                    "slot_analyses": [],
                }
                for slot in tpl["slot_analyses"]:
                    safe_slot = {k: v for k, v in slot.items() if k != "layer_sensitivities"}
                    safe_slot["layer_sensitivities_summary"] = [
                        {"layer": s["layer"], "mean_cos_dist": s["mean_cos_dist"], "mean_l2_dist": s["mean_l2_dist"]}
                        for s in slot["layer_sensitivities"]
                    ]
                    safe_tpl["slot_analyses"].append(safe_slot)
                safe_result["templates"].append(safe_tpl)

            json_path = OUTPUT_DIR / f"results_{model_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(safe_result, f, indent=2, ensure_ascii=False)

            # Free GPU memory
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            log(f"ERROR processing {model_name}: {e}")
            import traceback
            log(traceback.format_exc())

    # Cross-template analysis
    cross_template_analysis(all_model_results, log)

    # Final summary
    log(f"\n\n{'='*60}")
    log("P53 COMPLETE - Summary")
    log(f"{'='*60}")

    for model_name, mr in all_model_results.items():
        log(f"\n{model_name}:")
        for tpl in mr["templates"]:
            for slot in tpl["slot_analyses"]:
                log(f"  {tpl['template_id']} slot{slot['slot_idx']}: "
                    f"KL={slot['mean_kl_divergence']:.4f}, "
                    f"top1_agree={slot['top1_agreement']:.0f}%, "
                    f"sensitive_L{slot['most_sensitive_layer']}")

    log(f"\nTotal time: {time.strftime('%H:%M:%S')}")
    log.close()
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
