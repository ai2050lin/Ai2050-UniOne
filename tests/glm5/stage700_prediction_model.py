#!/usr/bin/env python3
"""
P54: Text → Delta-H → Logits Prediction Model (Stage700)

P52 proved: logits = sum(delta_h_l @ U.T) exactly (cos>0.995, INV-354)
P53 showed: different tokens cause different delta-h at different layers

P54 asks: Can we PREDICT delta-h from text features?
If yes → we can predict logits from text alone (generation model).

Method:
1. Collect training data: (text, delta_h_l) pairs for many texts
2. Extract text features: token embeddings (avg, max, pos-weighted), length, syntactic features
3. Train linear regression: text_features → delta_h_l for each layer
4. Evaluate: predicted_logits = sum(predicted_delta_h_l @ U.T) vs actual logits
5. Success criterion: top-1 accuracy > 70% or Pearson r > 0.8

This is the CRITICAL test of the generation route.
"""
import sys, math, time, gc, json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, accuracy_score

OUTPUT_DIR = _Path(f"tests/glm5_temp/stage700_prediction_model_{time.strftime('%Y%m%d_%H%M')}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}

# Large text set for training + testing
TEXTS = [
    # Category 1: Animals (10)
    "The cat sat on the mat.", "A dog chased the ball across the yard.",
    "The bird flew over the tall tree.", "Fish swim in the deep ocean.",
    "The horse galloped through the meadow.", "A rabbit hopped into the garden.",
    "The lion roared at the sunset.", "Dolphins play in the warm waves.",
    "The eagle soared above the mountain peak.", "A wolf howled at the moon.",
    # Category 2: Science (10)
    "Water boils at one hundred degrees Celsius.", "Light travels at a constant speed.",
    "Atoms contain protons and neutrons.", "Gravity pulls objects toward Earth.",
    "Photosynthesis converts sunlight to energy.", "DNA contains genetic instructions.",
    "Electrons orbit the atomic nucleus.", "Chemical reactions transform matter.",
    "The Earth orbits around the Sun.", "Sound waves travel through air.",
    # Category 3: Geography (10)
    "Paris is the capital of France.", "The Amazon is the longest river.",
    "Mount Everest is the highest peak.", "Tokyo is a large city in Japan.",
    "The Sahara is a vast desert.", "Australia is both a country and continent.",
    "The Nile flows through Egypt.", "Greenland has the largest ice sheet.",
    "China has the most people in the world.", "The Pacific is the biggest ocean.",
    # Category 4: Emotions (10)
    "She felt happy after the good news.", "He was sad about the loss.",
    "The movie made me laugh out loud.", "I feel anxious before exams.",
    "They were excited for the trip.", "She looked surprised at the gift.",
    "The news made him angry.", "We feel grateful for your help.",
    "He seemed confused by the instructions.", "The children were delighted.",
    # Category 5: Food (10)
    "Fresh bread smells wonderful in the morning.", "Chocolate cake is delicious.",
    "Sushi is a popular Japanese dish.", "The soup was hot and tasty.",
    "Coffee helps me stay awake.", "Fruit salad is healthy and sweet.",
    "Pasta is a traditional Italian food.", "The steak was cooked perfectly.",
    "Ice cream melts quickly in summer.", "Rice is a staple food in Asia.",
    # Category 6: Technology (10)
    "Artificial intelligence learns from data.", "The internet connects people worldwide.",
    "Computers process information quickly.", "Smartphones changed how we communicate.",
    "Cloud computing stores data remotely.", "Algorithms solve complex problems.",
    "Machine learning models improve with training.", "Social media has billions of users.",
    "Blockchain ensures secure transactions.", "Quantum computing may revolutionize science.",
    # Category 7: Music (10)
    "The piano has eighty eight keys.", "Music theory studies harmony and melody.",
    "Beethoven composed nine symphonies.", "Jazz originated in New Orleans.",
    "The guitar is a popular instrument.", "Orchestras perform classical music.",
    "Rhythm is essential to dance music.", "Singing requires breath control.",
    "Drums provide the beat for the band.", "Violins produce beautiful high notes.",
    # Category 8: Sports (10)
    "Soccer is the most popular sport globally.", "Basketball requires teamwork and skill.",
    "Swimming is a great full body workout.", "Tennis matches can last for hours.",
    "The Olympics happen every four years.", "Running improves cardiovascular health.",
    "Cycling is an efficient way to travel.", "Baseball is known as Americas pastime.",
    "Golf requires precision and patience.", "Figure skating combines art and athletics.",
    # Category 9: Philosophy (10)
    "Socrates asked many deep questions.", "Ethics studies right and wrong behavior.",
    "Descartes said I think therefore I am.", "Kant wrote about moral duty.",
    "Existentialism focuses on individual freedom.", "Logic is the foundation of reasoning.",
    "Epistemology asks what knowledge is.", "Aristotle taught about virtue ethics.",
    "Plato described the allegory of the cave.", "Utilitarianism seeks the greatest good.",
]


class Logger:
    def __init__(self, path):
        self.path = _Path(path)
        self.f = open(self.path, "w", encoding="utf-8")
    def __call__(self, msg):
        safe = msg.encode('utf-8', errors='replace').decode('utf-8')
        print(safe)
        self.f.write(safe + "\n")
        self.f.flush()
    def close(self):
        self.f.close()


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = MODEL_MAP[model_name]
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


def get_delta_h_and_logits(model, tokenizer, text):
    """Get per-layer delta-h and final logits for a text."""
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=64).to(model.device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    states = [hs[0, -1, :].float().cpu() for hs in hidden_states]
    num_layers = len(states) - 1

    deltas = []
    for l in range(1, num_layers + 1):
        deltas.append(states[l] - states[l - 1])
    delta_h = torch.stack(deltas)  # (num_layers, d_model)

    logits = outputs.logits[0, -1, :].float().cpu()  # (vocab_size,)
    return delta_h, logits


def extract_text_features(model, tokenizer, text):
    """Extract features from text that could predict delta-h.

    Features:
    1. Average token embedding (d_model dim)
    2. Last token embedding (d_model dim)
    3. Max-pooled token embedding (d_model dim)
    4. Text length (normalized)
    5. Token count
    Total: 3*d_model + 2
    """
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=64).to(model.device)
    with torch.no_grad():
        embed_out = model.get_input_embeddings()(tokens)  # (1, seq_len, d_model)

    embeds = embed_out[0].float().cpu()  # (seq_len, d_model)
    seq_len = embeds.shape[0]
    d_model = embeds.shape[1]

    avg_embed = embeds.mean(dim=0)  # (d_model,)
    last_embed = embeds[-1, :]  # (d_model,)
    max_embed = embeds.max(dim=0).values  # (d_model,)

    features = torch.cat([
        avg_embed,
        last_embed,
        max_embed,
        torch.tensor([seq_len / 64.0]),  # normalized length
        torch.tensor([float(seq_len)]),
    ])
    return features.numpy()  # (3*d_model + 2,)


def main():
    log = Logger(OUTPUT_DIR / "results.log")
    log(f"P54: Text → Delta-H → Logits Prediction Model")
    log(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Output: {OUTPUT_DIR}")
    log(f"Texts: {len(TEXTS)}")

    np.random.seed(42)
    indices = np.random.permutation(len(TEXTS))
    train_idx = indices[:70]  # 70 training
    test_idx = indices[70:]   # 30 testing

    log(f"Train: {len(train_idx)} texts, Test: {len(test_idx)} texts")

    all_model_results = {}

    for model_name in MODEL_MAP:
        log(f"\n{'='*60}")
        log(f"Processing model: {model_name}")
        log(f"{'='*60}")
        t0 = time.time()

        try:
            model, tokenizer = load_model(model_name)
            log(f"Loaded in {time.time()-t0:.1f}s")

            # Step 1: Collect data
            log("  Step 1: Collecting delta-h and text features...")
            train_features = []
            train_delta_h = []
            train_logits = []
            test_features = []
            test_delta_h = []
            test_logits = []
            actual_top1_tokens = []

            for idx in train_idx:
                text = TEXTS[idx]
                features = extract_text_features(model, tokenizer, text)
                delta_h, logits = get_delta_h_and_logits(model, tokenizer, text)
                train_features.append(features)
                train_delta_h.append(delta_h.numpy())
                train_logits.append(logits.numpy())

            for idx in test_idx:
                text = TEXTS[idx]
                features = extract_text_features(model, tokenizer, text)
                delta_h, logits = get_delta_h_and_logits(model, tokenizer, text)
                test_features.append(features)
                test_delta_h.append(delta_h.numpy())
                test_logits.append(logits.numpy())
                top1 = tokenizer.decode([logits.argmax().item()]).strip()
                actual_top1_tokens.append(top1)

            train_features = np.array(train_features)
            train_delta_h = np.array(train_delta_h)  # (70, num_layers, d_model)
            train_logits = np.array(train_logits)
            test_features = np.array(test_features)
            test_delta_h = np.array(test_delta_h)    # (30, num_layers, d_model)
            test_logits = np.array(test_logits)

            num_layers = train_delta_h.shape[1]
            d_model = train_delta_h.shape[2]
            feat_dim = train_features.shape[1]

            log(f"  Features: {feat_dim}D, Layers: {num_layers}, d_model: {d_model}")

            # Step 2: Train per-layer linear regression
            log("  Step 2: Training per-layer Ridge regression...")
            models_per_layer = {}
            pred_delta_h_test = np.zeros_like(test_delta_h)

            for l in range(num_layers):
                X_train = train_features
                Y_train = train_delta_h[:, l, :]  # (70, d_model)
                X_test = test_features

                reg = Ridge(alpha=1.0)
                reg.fit(X_train, Y_train)
                models_per_layer[l] = reg

                Y_pred = reg.predict(X_test)
                pred_delta_h_test[:, l, :] = Y_pred

                train_r2 = reg.score(X_train, Y_train)
                test_r2 = r2_score(test_delta_h[:, l, :], Y_pred)
                train_cos = np.mean([
                    np.dot(Y_train[i], reg.predict(X_train[i:i+1])[0]) /
                    (np.linalg.norm(Y_train[i]) * np.linalg.norm(reg.predict(X_train[i:i+1])[0]) + 1e-10)
                    for i in range(0, len(Y_train), 10)
                ])

                if l in [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]:
                    log(f"    L{l}: train_r2={train_r2:.4f}, test_r2={test_r2:.4f}")

            # Step 3: Compute predicted logits via linear accumulation
            log("  Step 3: Computing predicted logits...")
            with torch.no_grad():
                unembed = model.get_output_embeddings()
                unembed_weight = unembed.weight.float().cpu()  # (vocab_size, d_model)

                # For each test text
                top1_accuracies = []
                top5_accuracies = []
                margin_correlations = []
                logit_correlations = []
                pred_top1_tokens = []

                for i in range(len(test_idx)):
                    # Actual logits
                    actual_logits = test_logits[i]

                    # Predicted logits via linear accumulation
                    pred_logits = np.zeros(actual_logits.shape)
                    for l in range(num_layers):
                        delta_l = pred_delta_h_test[i, l, :]
                        delta_logits = unembed_weight.numpy() @ delta_l  # (vocab_size,)
                        pred_logits += delta_logits

                    # Compare
                    actual_top1 = np.argmax(actual_logits)
                    pred_top1 = np.argmax(pred_logits)
                    actual_top5 = set(np.argsort(actual_logits)[-5:])
                    pred_top5 = set(np.argsort(pred_logits)[-5:])

                    top1_accuracies.append(1 if actual_top1 == pred_top1 else 0)
                    top5_accuracies.append(1 if pred_top1 in actual_top5 else 0)

                    actual_margin = actual_logits[actual_top1] - np.sort(actual_logits)[-2]
                    pred_margin = pred_logits[pred_top1] - np.sort(pred_logits)[-2]
                    margin_correlations.append((actual_margin, pred_margin))

                    # Pearson correlation of full logit vectors (sampled)
                    sample_idx = np.random.choice(len(actual_logits), min(1000, len(actual_logits)), replace=False)
                    r = np.corrcoef(actual_logits[sample_idx], pred_logits[sample_idx])[0, 1]
                    logit_correlations.append(r)

                    pred_token = tokenizer.decode([int(pred_top1)]).strip()
                    pred_top1_tokens.append(pred_token)

            mean_top1 = np.mean(top1_accuracies)
            mean_top5 = np.mean(top5_accuracies)
            mean_logit_r = np.mean(logit_correlations)
            margins = np.array(margin_correlations)
            margin_r = np.corrcoef(margins[:, 0], margins[:, 1])[0, 1] if len(margins) > 2 else 0

            log(f"\n  ===== RESULTS for {model_name} =====")
            log(f"  Top-1 accuracy: {mean_top1:.2%}")
            log(f"  Top-5 accuracy: {mean_top5:.2%}")
            log(f"  Mean logit Pearson r: {mean_logit_r:.4f}")
            log(f"  Margin Pearson r: {margin_r:.4f}")

            # Delta-h prediction quality
            all_pred_flat = pred_delta_h_test.flatten()
            all_true_flat = test_delta_h.flatten()
            delta_r2 = r2_score(all_true_flat, all_pred_flat)
            log(f"  Delta-h overall R2: {delta_r2:.4f}")

            # Sample predictions
            log(f"\n  Sample predictions:")
            for i in range(min(5, len(test_idx))):
                log(f"    '{TEXTS[test_idx[i]][:40]}...'")
                log(f"      Actual top1: '{actual_top1_tokens[i]}'")
                log(f"      Pred top1:   '{pred_top1_tokens[i]}'")

            # Success criterion
            success = mean_top1 > 0.70 or mean_logit_r > 0.8
            log(f"\n  SUCCESS CRITERION: {'PASS' if success else 'FAIL'}")
            log(f"    (Need top1>70% or r>0.8, got top1={mean_top1:.0%}, r={mean_logit_r:.4f})")

            model_result = {
                "model": model_name,
                "num_layers": num_layers,
                "d_model": d_model,
                "feat_dim": feat_dim,
                "top1_accuracy": float(mean_top1),
                "top5_accuracy": float(mean_top5),
                "logit_pearson_r": float(mean_logit_r),
                "margin_pearson_r": float(margin_r),
                "delta_h_r2": float(delta_r2),
                "success": bool(success),
                "sample_predictions": [
                    {"text": TEXTS[test_idx[i]][:50], "actual": actual_top1_tokens[i], "predicted": pred_top1_tokens[i]}
                    for i in range(min(10, len(test_idx)))
                ],
            }
            all_model_results[model_name] = model_result

            # Save JSON
            json_path = OUTPUT_DIR / f"results_{model_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(model_result, f, indent=2, ensure_ascii=False)

            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            log(f"  {model_name} done in {time.time()-t0:.1f}s")

        except Exception as e:
            log(f"ERROR: {e}")
            import traceback
            log(traceback.format_exc())

    # Final summary
    log(f"\n\n{'='*60}")
    log("P54 COMPLETE - Final Summary")
    log(f"{'='*60}")

    for name, res in all_model_results.items():
        status = "PASS" if res["success"] else "FAIL"
        log(f"  {name}: top1={res['top1_accuracy']:.0%}, r={res['logit_pearson_r']:.4f}, "
            f"delta_r2={res['delta_h_r2']:.4f} [{status}]")

    log(f"\nTotal time: {time.strftime('%H:%M:%S')}")
    log.close()
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
