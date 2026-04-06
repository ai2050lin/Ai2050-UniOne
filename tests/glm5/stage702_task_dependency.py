#!/usr/bin/env python3
"""
P58: Task-Dependency Verification for avg_lp ≈ log(margin) - C (Falsifying Proposition 4)

Proposition 4: "avg_lp ≈ log(margin) - 1.5 is a universal equation"
- Current evidence: verified on 4 models with general English text
- Falsification condition: if C varies >1.0 across tasks → not universal

Method:
1. Test on 5 task types:
   - General English (baseline)
   - Python code
   - Mathematical reasoning
   - Chinese text
   - Formal/logical text
2. For each text, compute avg_logprob, margin, and C = log(margin) - avg_logprob
3. If C varies >1.0 across tasks → P4 FALSIFIED
4. If C stays within 1.5±0.3 → P4 confirmed
"""
import sys, math, time, gc, json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path

OUTPUT_DIR = _Path(f"tests/glm5_temp/stage702_task_dependency_{time.strftime('%Y%m%d_%H%M')}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}

TEXTS_BY_TASK = {
    "general_english": [
        "The cat sat on the mat.", "A dog chased the ball across the yard.",
        "Paris is the capital of France.", "She felt happy after the good news.",
        "Fresh bread smells wonderful.", "Soccer is the most popular sport.",
        "The piano has eighty eight keys.", "Water boils at one hundred degrees.",
        "Artificial intelligence learns from data.", "Philosophy asks deep questions.",
    ],
    "python_code": [
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "import numpy as np; arr = np.array([1, 2, 3]); print(arr.mean())",
        "class DataLoader: def __init__(self, data): self.data = data",
        "for i in range(len(items)): result.append(process(items[i]))",
        "df.groupby('category').agg({'value': ['mean', 'std', 'count']})",
        "async def fetch_data(url): async with aiohttp.ClientSession() as s:",
        "model = tf.keras.Sequential([tf.keras.layers.Dense(128, activation='relu')])",
        "with open('data.json', 'r') as f: data = json.load(f)",
        "from collections import defaultdict; counter = defaultdict(int)",
        "try: result = int(user_input) except ValueError: result = 0",
    ],
    "math_reasoning": [
        "Let x be a positive integer. Prove that x squared plus x is always even.",
        "The derivative of sine x with respect to x equals cosine x.",
        "By the fundamental theorem of calculus, the integral from a to b of f prime equals f of b minus f of a.",
        "The sum of an infinite geometric series with ratio r where absolute r less than 1 equals a over one minus r.",
        "Euler proved that e raised to the power of i pi plus one equals zero.",
        "The binomial coefficient n choose k equals n factorial over k factorial times n minus k factorial.",
        "Bayes theorem states that the probability of A given B equals P of B given A times P of A over P of B.",
        "A matrix is invertible if and only if its determinant is non-zero.",
        "The Pythagorean theorem: a squared plus b squared equals c squared for any right triangle.",
        "The limit as n approaches infinity of one plus one over n to the power n equals e.",
    ],
    "chinese_text": [
        "人工智能正在改变我们的生活方式。",
        "机器学习是人工智能的一个核心分支。",
        "深度神经网络可以自动提取特征。",
        "自然语言处理让计算机理解人类语言。",
        "知识图谱构建了语义之间的关系网络。",
        "大语言模型展现了强大的语言理解能力。",
        "强化学习通过奖励信号优化决策策略。",
        "计算机视觉技术在自动驾驶中广泛应用。",
        "量子计算有潜力解决传统计算无法处理的问题。",
        "数据驱动的科学方法正在改变研究领域。",
    ],
    "formal_logic": [
        "For all x, if P of x implies Q of x, and P of a is true, then Q of a is true.",
        "The contrapositive of P implies Q is not Q implies not P, which is logically equivalent.",
        "A statement and its negation cannot both be true: this is the law of non-contradiction.",
        "Modus ponens: from P and P implies Q, conclude Q.",
        "Existential instantiation: if there exists x such that P of x, let c be such an element.",
        "A universal quantifier distributes over conjunction: for all x, P of x and Q of x, iff for all x P of x and for all x Q of x.",
        "Proof by contradiction: assume not P, derive a contradiction, therefore P is true.",
        "The identity law: P or not P is a tautology, always true.",
        "De Morgan's laws: not P and Q equals not P or not Q, and not P or Q equals not P and not Q.",
        "A implies B is equivalent to not A or B by material implication.",
    ],
}


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


def compute_avg_logprob_and_margin(model, tokenizer, text):
    """Compute avg_logprob and margin for a text."""
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=64).to(model.device)

    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    # Average log probability (over all tokens except first)
    log_probs = F.log_softmax(logits, dim=-1)
    total_logprob = 0.0
    count = 0
    for i in range(1, tokens.shape[1]):
        token_id = tokens[0, i].item()
        total_logprob += log_probs[i - 1, token_id].item()
        count += 1

    avg_logprob = total_logprob / count if count > 0 else 0.0

    # Margin: top1 logit - top2 logit at last position
    last_logits = logits[-1, :]
    sorted_logits = torch.sort(last_logits, descending=True).values
    margin = (sorted_logits[0] - sorted_logits[1]).item()

    return avg_logprob, margin


def main():
    log = Logger(OUTPUT_DIR / "results.log")
    log(f"P58: Task-Dependency Verification (avg_lp = log(margin) - C)")
    log(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Output: {OUTPUT_DIR}")

    all_results = {}

    for model_name in MODEL_MAP:
        log(f"\n{'='*60}")
        log(f"Processing model: {model_name}")
        log(f"{'='*60}")
        t0 = time.time()

        try:
            model, tokenizer = load_model(model_name)
            log(f"Loaded in {time.time()-t0:.1f}s")

            task_results = {}

            for task_name, texts in TEXTS_BY_TASK.items():
                log(f"\n  Task: {task_name} ({len(texts)} texts)")

                cs = []
                avg_logprobs = []
                margins = []

                for text in texts:
                    try:
                        avg_lp, margin = compute_avg_logprob_and_margin(model, tokenizer, text)

                        if margin > 0 and not math.isnan(avg_lp) and not math.isinf(avg_lp):
                            c = math.log(margin) - avg_lp
                            cs.append(c)
                            avg_logprobs.append(avg_lp)
                            margins.append(margin)
                    except Exception as e:
                        log(f"    Error: {e}")

                if len(cs) >= 3:
                    mean_c = np.mean(cs)
                    std_c = np.std(cs)
                    median_c = np.median(cs)
                    mean_lp = np.mean(avg_logprobs)
                    mean_margin = np.mean(margins)

                    # Pearson r between avg_lp and log(margin)
                    log_margins = [math.log(m) for m in margins]
                    r = np.corrcoef(avg_logprobs, log_margins)[0, 1]

                    task_results[task_name] = {
                        "n_valid": len(cs),
                        "mean_C": float(mean_c),
                        "std_C": float(std_c),
                        "median_C": float(median_c),
                        "mean_avg_lp": float(mean_lp),
                        "mean_margin": float(mean_margin),
                        "pearson_r": float(r),
                        "C_values": [float(c) for c in cs],
                    }

                    log(f"    C = {mean_c:.3f} +/- {std_c:.3f} (median={median_c:.3f})")
                    log(f"    avg_lp = {mean_lp:.3f}, margin = {mean_margin:.3f}")
                    log(f"    Pearson r(avg_lp, log_margin) = {r:.4f}")
                else:
                    log(f"    Only {len(cs)} valid texts, skipping")
                    task_results[task_name] = {"n_valid": len(cs), "error": "too few valid texts"}

            # Cross-task comparison
            valid_tasks = {k: v for k, v in task_results.items() if "mean_C" in v}
            if len(valid_tasks) >= 2:
                c_values = [v["mean_C"] for v in valid_tasks.values()]
                c_range = max(c_values) - min(c_values)
                c_std = np.std(c_values)

                log(f"\n  Cross-task C values:")
                for task, res in valid_tasks.items():
                    log(f"    {task}: C = {res['mean_C']:.3f}")
                log(f"  C range: {c_range:.3f}")
                log(f"  C std across tasks: {c_std:.3f}")

                if c_range > 1.0:
                    verdict = "P4 FALSIFIED - C varies >1.0 across tasks"
                elif c_range < 0.3:
                    verdict = "P4 CONFIRMED - C is universal across tasks"
                else:
                    verdict = "P4 PARTIALLY CONFIRMED - C varies moderately"
                log(f"  Verdict: {verdict}")
            else:
                verdict = "insufficient data"
                c_range = 0

            model_result = {
                "model": model_name,
                "tasks": task_results,
                "verdict": verdict,
                "c_range": float(c_range) if len(valid_tasks) >= 2 else None,
            }
            all_results[model_name] = model_result

            # Save JSON
            json_path = OUTPUT_DIR / f"results_{model_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(model_result, f, indent=2, ensure_ascii=False, default=str)

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
    log("P58 COMPLETE - Task-Dependency Summary")
    log(f"{'='*60}")

    for name, res in all_results.items():
        log(f"\n  {name}: verdict={res.get('verdict', 'N/A')}")
        if "tasks" in res:
            for task, data in res["tasks"].items():
                if "mean_C" in data:
                    log(f"    {task}: C={data['mean_C']:.3f} +/- {data['std_C']:.3f}")

    log(f"\nTotal time: {time.strftime('%H:%M:%S')}")
    log.close()
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
