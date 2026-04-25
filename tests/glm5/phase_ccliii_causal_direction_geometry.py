"""
Phase CCLIII: Causal Direction Geometry Analysis
==================================================
CCLII found: feature causal efficacy increases with depth, 
but PC alignment DECREASES with depth.

This paradox suggests PC is not the right framework for understanding
causal directions. This test investigates the GEOMETRY of causal directions:

1. Causal Consistency: Is the tense direction the SAME across different inputs?
   - CR = ||mean(d_i)||^2 / mean(||d_i||^2)
   - CR ~ 1: all inputs use the same causal direction (concentrated)
   - CR ~ 0: different inputs use different causal directions (distributed)

2. Cross-Feature Orthogonality: Are tense and question directions orthogonal?
   - Do different features use the same or different subspaces?

3. Causal Dimension Structure: 
   - SVD of per-sample feature directions
   - How many dimensions are needed to explain 90% of causal direction variance?
   - Does this change with depth?

4. Per-Sample Causal Efficacy Variation:
   - For test sentences, measure tense shift per sample
   - Is efficacy correlated with alignment between per-sample and mean direction?

Key predictions:
- Deep layers should have higher CR (more consistent causal direction)
- Tense and question directions should become more orthogonal with depth
- Fewer dimensions needed to explain causal variance in deep layers
"""

import argparse
import json
import os
import numpy as np
from datetime import datetime
from scipy.stats import spearmanr

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_CONFIGS = {
    'qwen3': {
        'name': 'Qwen3-4B',
        'path': 'D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c',
        'n_layers': 36,
        'd_model': 2560,
        'dtype': 'bf16',
    },
    'deepseek7b': {
        'name': 'DeepSeek-R1-Distill-Qwen-7B',
        'path': 'D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'n_layers': 28,
        'd_model': 3584,
        'dtype': '8bit',
    },
    'glm4': {
        'name': 'GLM4-9B-Chat',
        'path': 'D:/develop/model/hub/models--zai-org--glm-4-9b-chat-hf/download',
        'n_layers': 40,
        'd_model': 4096,
        'dtype': '8bit',
    }
}

# 25 tense pairs for consistency analysis
TENSE_PAIRS = [
    ("She walks to school", "She walked to school"),
    ("He runs in the park", "He ran in the park"),
    ("They play football", "They played football"),
    ("The cat sleeps on the mat", "The cat slept on the mat"),
    ("She sings beautifully", "She sang beautifully"),
    ("He writes a letter", "He wrote a letter"),
    ("The bird flies south", "The bird flew south"),
    ("She reads the book", "She read the book"),
    ("They build houses", "They built houses"),
    ("The wind blows hard", "The wind blew hard"),
    ("I think about it", "I thought about it"),
    ("We drink coffee", "We drank coffee"),
    ("He drives fast", "He drove fast"),
    ("She knows the answer", "She knew the answer"),
    ("The dog barks loudly", "The dog barked loudly"),
    ("They grow vegetables", "They grew vegetables"),
    ("I feel happy today", "I felt happy today"),
    ("She brings lunch", "She brought lunch"),
    ("He catches the ball", "He caught the ball"),
    ("The bell rings twice", "The bell rang twice"),
    ("We sit together", "We sat together"),
    ("She tells stories", "She told stories"),
    ("He finds the key", "He found the key"),
    ("The river flows fast", "The river flowed fast"),
    ("They stand outside", "They stood outside"),
]

# 25 question pairs
QUESTION_PAIRS = [
    ("The sky is blue", "Is the sky blue?"),
    ("Cats like fish", "Do cats like fish?"),
    ("She is happy", "Is she happy?"),
    ("The door is open", "Is the door open?"),
    ("He works hard", "Does he work hard?"),
    ("Water boils at 100 degrees", "Does water boil at 100 degrees?"),
    ("The car is red", "Is the car red?"),
    ("They live nearby", "Do they live nearby?"),
    ("The book is long", "Is the book long?"),
    ("She speaks French", "Does she speak French?"),
    ("The sun is bright", "Is the sun bright?"),
    ("Birds can fly", "Can birds fly?"),
    ("The room is cold", "Is the room cold?"),
    ("He plays guitar", "Does he play guitar?"),
    ("The food is ready", "Is the food ready?"),
    ("They know the truth", "Do they know the truth?"),
    ("The test is hard", "Is the test hard?"),
    ("She loves music", "Does she love music?"),
    ("The lake is deep", "Is the lake deep?"),
    ("He runs every day", "Does he run every day?"),
    ("The train is late", "Is the train late?"),
    ("We need more time", "Do we need more time?"),
    ("The cake is sweet", "Is the cake sweet?"),
    ("She reads a lot", "Does she read a lot?"),
    ("The movie is good", "Is the movie good?"),
]

# Test sentences for efficacy analysis
TEST_SENTENCES_PRESENT = [
    "She walks to school every day",
    "He runs in the park each morning",
    "The cat sleeps on the mat quietly",
    "They play football on Sundays",
    "The bird flies above the trees",
    "She sings in the choir weekly",
    "He writes reports for work",
    "The dog chases its tail",
    "We eat lunch at noon",
    "The wind blows through the window",
    "I read books every evening",
    "She drives to the office",
    "He swims in the pool daily",
    "They dance at the party",
    "The sun shines bright today",
]

TEST_SENTENCES_PAST = [
    "She walked to school every day",
    "He ran in the park each morning",
    "The cat slept on the mat quietly",
    "They played football on Sundays",
    "The bird flew above the trees",
    "She sang in the choir weekly",
    "He wrote reports for work",
    "The dog chased its tail",
    "We ate lunch at noon",
    "The wind blew through the window",
    "I read books every evening",
    "She drove to the office",
    "He swam in the pool daily",
    "They danced at the party",
    "The sun shone bright today",
]


def load_model(model_key, device='cuda'):
    config = MODEL_CONFIGS[model_key]
    path = config['path']
    dtype = config['dtype']

    print(f"  Loading: {config['name']} ({dtype})")

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dtype == '8bit':
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, local_files_only=True
        )
    elif dtype == 'bf16':
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map="auto",
            trust_remote_code=True, local_files_only=True
        )

    model.eval()
    return model, tokenizer


def get_target_layer(model, layer_idx):
    model_type = model.config.model_type
    if model_type in ['qwen2', 'qwen3']:
        return model.model.layers[layer_idx]
    elif model_type in ['chatglm', 'glm4']:
        return model.transformer.encoder.layers[layer_idx]
    else:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers[layer_idx]
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'encoder'):
            return model.transformer.encoder.layers[layer_idx]
        else:
            raise ValueError(f"Cannot find layers for model type {model_type}")


def get_hidden_state(model, tokenizer, target_layer, sentence):
    """Get hidden state at target layer for last token."""
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    seq_len = attention_mask.sum().item()
    last_pos = seq_len - 1

    hidden_state = [None]

    def capture_hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        hidden_state[0] = h[0, last_pos, :].detach().cpu().float().clone()
        return output

    h = target_layer.register_forward_hook(capture_hook)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    h.remove()

    return hidden_state[0].numpy()


def get_hidden_and_logits(model, tokenizer, target_layer, sentence):
    """Get hidden state and logits."""
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    seq_len = attention_mask.sum().item()
    last_pos = seq_len - 1

    hidden_state = [None]

    def capture_hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        hidden_state[0] = h[0, last_pos, :].detach().cpu().float().clone()
        return output

    h = target_layer.register_forward_hook(capture_hook)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, last_pos, :].detach().cpu().float()
    h.remove()

    return hidden_state[0].numpy(), logits.numpy()


def intervene_and_get_logits(model, tokenizer, target_layer, sentence, perturbation_vec):
    """Perturb hidden state at target layer and get resulting logits."""
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    seq_len = attention_mask.sum().item()
    last_pos = seq_len - 1

    perturb_tensor = torch.tensor(perturbation_vec, dtype=torch.float32, device=model.device)

    def perturb_hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        perturbed = h.clone()
        perturbed[0, last_pos, :] = perturbed[0, last_pos, :].float() + perturb_tensor.to(perturbed.dtype)
        if isinstance(output, tuple):
            return (perturbed,) + output[1:]
        return perturbed

    h = target_layer.register_forward_hook(perturb_hook)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, last_pos, :].detach().cpu().float()
    h.remove()

    return logits.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_tense', type=int, default=25)
    parser.add_argument('--n_question', type=int, default=25)
    parser.add_argument('--n_test', type=int, default=15)
    parser.add_argument('--eps_rel', type=float, default=0.1)
    args = parser.parse_args()

    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]

    out_dir = f"results/causal_fiber/{model_key}_ccliii"
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, 'run.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f"=== Phase CCLIII: Causal Direction Geometry Analysis ===")
    log(f"Model: {cfg['name']}")
    log(f"n_tense: {args.n_tense}, n_question: {args.n_question}, n_test: {args.n_test}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    model, tokenizer = load_model(model_key)
    log(f"Model loaded successfully")

    n_layers = cfg['n_layers']
    test_layers = [
        max(1, n_layers // 8),
        max(2, n_layers // 4),
        max(3, n_layers // 2),
        max(4, 3 * n_layers // 4),
        max(5, n_layers - 2),
    ]
    test_layers = sorted(set([l for l in test_layers if 1 <= l < n_layers]))

    results = {}

    for layer_idx in test_layers:
        target_layer = get_target_layer(model, layer_idx)
        log(f"\n=== Layer {layer_idx} ===")

        # ============================================================
        # Part 1: Causal Consistency - Tense
        # ============================================================
        log(f"  Part 1: Collecting tense pair activations...")

        tense_diffs = []
        tense_acts_pos = []
        tense_acts_neg = []

        for i, (present, past) in enumerate(TENSE_PAIRS[:args.n_tense]):
            h_present = get_hidden_state(model, tokenizer, target_layer, present)
            h_past = get_hidden_state(model, tokenizer, target_layer, past)
            diff = h_past - h_present
            tense_diffs.append(diff)
            tense_acts_pos.append(h_present)
            tense_acts_neg.append(h_past)

        tense_diffs = np.array(tense_diffs)  # [n_tense, d_model]
        tense_acts_pos = np.array(tense_acts_pos)
        tense_acts_neg = np.array(tense_acts_neg)

        # Mean causal direction
        tense_mean = tense_diffs.mean(axis=0)
        tense_mean_norm = np.linalg.norm(tense_mean)

        # Per-sample norms
        tense_diff_norms = np.linalg.norm(tense_diffs, axis=1)

        # Causal Consistency Ratio
        # CR = ||mean(d_i)||^2 / mean(||d_i||^2)
        # CR=1 means perfect consistency, CR=1/N means random directions
        cr_tense = tense_mean_norm**2 / (np.mean(tense_diff_norms**2) + 1e-10)

        # Per-sample alignment with mean direction
        tense_alignments = []
        for i in range(len(tense_diffs)):
            if tense_diff_norms[i] > 1e-10:
                cos = np.dot(tense_diffs[i], tense_mean) / (tense_diff_norms[i] * tense_mean_norm + 1e-10)
                tense_alignments.append(cos)
            else:
                tense_alignments.append(0.0)
        tense_mean_alignment = np.mean(tense_alignments)

        log(f"    Tense CR: {cr_tense:.4f}, mean alignment: {tense_mean_alignment:.4f}")

        # ============================================================
        # Part 2: Causal Consistency - Question
        # ============================================================
        log(f"  Part 2: Collecting question pair activations...")

        quest_diffs = []
        for stmt, quest in QUESTION_PAIRS[:args.n_question]:
            h_stmt = get_hidden_state(model, tokenizer, target_layer, stmt)
            h_quest = get_hidden_state(model, tokenizer, target_layer, quest)
            diff = h_quest - h_stmt
            quest_diffs.append(diff)

        quest_diffs = np.array(quest_diffs)
        quest_mean = quest_diffs.mean(axis=0)
        quest_mean_norm = np.linalg.norm(quest_mean)
        quest_diff_norms = np.linalg.norm(quest_diffs, axis=1)

        cr_quest = quest_mean_norm**2 / (np.mean(quest_diff_norms**2) + 1e-10)

        quest_alignments = []
        for i in range(len(quest_diffs)):
            if quest_diff_norms[i] > 1e-10:
                cos = np.dot(quest_diffs[i], quest_mean) / (quest_diff_norms[i] * quest_mean_norm + 1e-10)
                quest_alignments.append(cos)
            else:
                quest_alignments.append(0.0)
        quest_mean_alignment = np.mean(quest_alignments)

        log(f"    Question CR: {cr_quest:.4f}, mean alignment: {quest_mean_alignment:.4f}")

        # ============================================================
        # Part 3: Cross-Feature Orthogonality
        # ============================================================
        if tense_mean_norm > 1e-10 and quest_mean_norm > 1e-10:
            cos_tense_quest = np.dot(tense_mean, quest_mean) / (tense_mean_norm * quest_mean_norm)
        else:
            cos_tense_quest = 0.0

        # Also: per-sample cross-alignment
        cross_alignments = []
        for i in range(min(len(tense_diffs), len(quest_diffs))):
            tn = np.linalg.norm(tense_diffs[i])
            qn = np.linalg.norm(quest_diffs[i])
            if tn > 1e-10 and qn > 1e-10:
                cross_alignments.append(np.dot(tense_diffs[i], quest_diffs[i]) / (tn * qn))
        cross_alignment_mean = np.mean(cross_alignments) if cross_alignments else 0.0

        log(f"    Tense-Quest cos: {cos_tense_quest:.4f}, per-sample cross: {cross_alignment_mean:.4f}")

        # ============================================================
        # Part 4: SVD of causal direction matrix
        # ============================================================
        log(f"  Part 4: SVD of causal directions...")

        # Stack all causal directions (tense + question)
        all_diffs = np.vstack([tense_diffs, quest_diffs])  # [n_pairs*2, d_model]

        # Center
        all_diffs_centered = all_diffs - all_diffs.mean(axis=0, keepdims=True)

        n_samples, d_model = all_diffs_centered.shape

        # SVD
        if n_samples < d_model:
            gram = all_diffs_centered @ all_diffs_centered.T
            eigvals, eigvecs = np.linalg.eigh(gram)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvals = np.maximum(eigvals, 0)
            # Convert to singular values
            svs = np.sqrt(eigvals)
            # Variance explained
            total_var = np.sum(svs**2)
            cumvar = np.cumsum(svs**2) / (total_var + 1e-10)
        else:
            U, svs, Vt = np.linalg.svd(all_diffs_centered, full_matrices=False)
            total_var = np.sum(svs**2)
            cumvar = np.cumsum(svs**2) / (total_var + 1e-10)

        # Effective causal dimensionality
        # Number of dims needed for 90% of variance
        dim_90 = np.searchsorted(cumvar, 0.9) + 1 if len(cumvar) > 0 else 1
        dim_50 = np.searchsorted(cumvar, 0.5) + 1 if len(cumvar) > 0 else 1

        # Participation ratio of causal directions
        pr_causal = np.sum(svs**2)**2 / (np.sum(svs**4) + 1e-10)

        log(f"    Causal dim(50%): {dim_50}, dim(90%): {dim_90}, PR: {pr_causal:.2f}")

        # ============================================================
        # Part 5: Separate SVD for tense vs question
        # ============================================================
        # Tense directions SVD
        tense_centered = tense_diffs - tense_diffs.mean(axis=0, keepdims=True)
        if len(tense_centered) > 1:
            gram_t = tense_centered @ tense_centered.T
            eigvals_t, _ = np.linalg.eigh(gram_t)
            eigvals_t = np.sort(eigvals_t)[::-1]
            eigvals_t = np.maximum(eigvals_t, 0)
            svs_t = np.sqrt(eigvals_t)
            pr_tense = np.sum(svs_t**2)**2 / (np.sum(svs_t**4) + 1e-10)
            total_t = np.sum(svs_t**2)
            cumvar_t = np.cumsum(svs_t**2) / (total_t + 1e-10)
            dim_t90 = np.searchsorted(cumvar_t, 0.9) + 1 if len(cumvar_t) > 0 else 1
        else:
            pr_tense = 1.0
            dim_t90 = 1

        # Question directions SVD
        quest_centered = quest_diffs - quest_diffs.mean(axis=0, keepdims=True)
        if len(quest_centered) > 1:
            gram_q = quest_centered @ quest_centered.T
            eigvals_q, _ = np.linalg.eigh(gram_q)
            eigvals_q = np.sort(eigvals_q)[::-1]
            eigvals_q = np.maximum(eigvals_q, 0)
            svs_q = np.sqrt(eigvals_q)
            pr_quest = np.sum(svs_q**2)**2 / (np.sum(svs_q**4) + 1e-10)
            total_q = np.sum(svs_q**2)
            cumvar_q = np.cumsum(svs_q**2) / (total_q + 1e-10)
            dim_q90 = np.searchsorted(cumvar_q, 0.9) + 1 if len(cumvar_q) > 0 else 1
        else:
            pr_quest = 1.0
            dim_q90 = 1

        log(f"    Tense PR: {pr_tense:.2f}, dim90: {dim_t90}")
        log(f"    Quest PR: {pr_quest:.2f}, dim90: {dim_q90}")

        # ============================================================
        # Part 6: Per-sample causal efficacy vs alignment
        # ============================================================
        log(f"  Part 6: Causal efficacy analysis...")

        # Normalize tense direction for perturbation
        if tense_mean_norm > 1e-10:
            tense_dir_unit = tense_mean / tense_mean_norm
        else:
            tense_dir_unit = np.zeros(d_model)

        # Per-sample efficacy
        per_sample_efficacy = []
        per_sample_alignment = []

        for i in range(min(args.n_test, len(TEST_SENTENCES_PRESENT))):
            present = TEST_SENTENCES_PRESENT[i]
            past = TEST_SENTENCES_PAST[i]

            # Get clean and past logits
            _, logits_clean = get_hidden_and_logits(model, tokenizer, target_layer, present)
            _, logits_past = get_hidden_and_logits(model, tokenizer, target_layer, past)

            logit_diff_clean = logits_past - logits_clean

            # Get activation norm for perturbation magnitude
            h_clean = get_hidden_state(model, tokenizer, target_layer, present)
            h_norm = np.linalg.norm(h_clean)
            eps = args.eps_rel * h_norm

            # Perturb along mean tense direction
            logits_plus = intervene_and_get_logits(
                model, tokenizer, target_layer, present, eps * tense_dir_unit)

            diff = logits_plus - logits_clean
            if np.linalg.norm(diff) > 1e-10 and np.linalg.norm(logit_diff_clean) > 1e-10:
                efficacy = np.dot(diff, logit_diff_clean) / (np.linalg.norm(diff) * np.linalg.norm(logit_diff_clean))
            else:
                efficacy = 0.0

            # Per-sample alignment with mean direction
            h_past_act = get_hidden_state(model, tokenizer, target_layer, past)
            sample_dir = h_past_act - h_clean
            sample_norm = np.linalg.norm(sample_dir)
            if sample_norm > 1e-10 and tense_mean_norm > 1e-10:
                alignment = np.dot(sample_dir, tense_mean) / (sample_norm * tense_mean_norm)
            else:
                alignment = 0.0

            per_sample_efficacy.append(efficacy)
            per_sample_alignment.append(alignment)

        mean_efficacy = np.mean(per_sample_efficacy)
        mean_alignment = np.mean(per_sample_alignment)

        # Correlation between alignment and efficacy
        if len(per_sample_efficacy) > 3:
            corr_align_efficacy, _ = spearmanr(per_sample_alignment, per_sample_efficacy)
        else:
            corr_align_efficacy = 0.0

        log(f"    Mean efficacy: {mean_efficacy:.4f}, mean alignment: {mean_alignment:.4f}")
        log(f"    Corr(alignment, efficacy): {corr_align_efficacy:.3f}")

        # ============================================================
        # Part 7: Random baseline efficacy
        # ============================================================
        rand_efficacies = []
        for i in range(min(5, args.n_test)):
            present = TEST_SENTENCES_PRESENT[i]
            _, logits_clean = get_hidden_and_logits(model, tokenizer, target_layer, present)
            _, logits_past = get_hidden_and_logits(model, tokenizer, target_layer, TEST_SENTENCES_PAST[i])
            logit_diff_clean = logits_past - logits_clean

            h_clean = get_hidden_state(model, tokenizer, target_layer, present)
            h_norm = np.linalg.norm(h_clean)
            eps = args.eps_rel * h_norm

            # 5 random directions
            for _ in range(3):
                rand_dir = np.random.randn(d_model)
                rand_dir = rand_dir / np.linalg.norm(rand_dir)
                logits_plus = intervene_and_get_logits(
                    model, tokenizer, target_layer, present, eps * rand_dir)
                diff = logits_plus - logits_clean
                if np.linalg.norm(diff) > 1e-10 and np.linalg.norm(logit_diff_clean) > 1e-10:
                    eff = np.dot(diff, logit_diff_clean) / (np.linalg.norm(diff) * np.linalg.norm(logit_diff_clean))
                else:
                    eff = 0.0
                rand_efficacies.append(abs(eff))

        mean_rand_efficacy = np.mean(rand_efficacies)
        efficacy_ratio = abs(mean_efficacy) / (mean_rand_efficacy + 1e-10)

        log(f"    Random efficacy: {mean_rand_efficacy:.4f}, Feat/Rand ratio: {efficacy_ratio:.3f}")

        # Store results
        results[layer_idx] = {
            'cr_tense': float(cr_tense),
            'cr_quest': float(cr_quest),
            'alignment_tense': float(tense_mean_alignment),
            'alignment_quest': float(quest_mean_alignment),
            'cos_tense_quest': float(cos_tense_quest),
            'cross_alignment_mean': float(cross_alignment_mean),
            'dim_90': int(dim_90),
            'dim_50': int(dim_50),
            'pr_causal': float(pr_causal),
            'pr_tense': float(pr_tense),
            'pr_quest': float(pr_quest),
            'dim_t90': int(dim_t90),
            'dim_q90': int(dim_q90),
            'mean_efficacy': float(mean_efficacy),
            'mean_alignment': float(mean_alignment),
            'corr_align_eff': float(corr_align_efficacy),
            'mean_rand_efficacy': float(mean_rand_efficacy),
            'efficacy_ratio': float(efficacy_ratio),
        }

    # ============================================================
    # Analysis
    # ============================================================
    log(f"\n{'='*70}")
    log(f"ANALYSIS")
    log(f"{'='*70}")

    layers = sorted(results.keys())
    layer_indices = list(range(len(layers)))

    log(f"\n  Summary Table:")
    log(f"  {'L':>4} {'CR_T':>6} {'CR_Q':>6} {'Cos_TQ':>7} {'PR_c':>6} {'dim90':>5} "
        f"{'Eff':>6} {'E/R':>6} {'Corr':>6}")

    for l in layers:
        r = results[l]
        log(f"  {l:>4} {r['cr_tense']:>6.3f} {r['cr_quest']:>6.3f} {r['cos_tense_quest']:>7.4f} "
            f"{r['pr_causal']:>6.2f} {r['dim_90']:>5d} "
            f"{r['mean_efficacy']:>6.3f} {r['efficacy_ratio']:>6.2f} {r['corr_align_eff']:>6.3f}")

    # Depth correlations
    cr_t = [results[l]['cr_tense'] for l in layers]
    cr_q = [results[l]['cr_quest'] for l in layers]
    cos_tq = [results[l]['cos_tense_quest'] for l in layers]
    pr_c = [results[l]['pr_causal'] for l in layers]
    dim90 = [results[l]['dim_90'] for l in layers]
    eff = [results[l]['mean_efficacy'] for l in layers]
    eff_ratio = [results[l]['efficacy_ratio'] for l in layers]
    align_eff = [results[l]['corr_align_eff'] for l in layers]
    pr_t = [results[l]['pr_tense'] for l in layers]
    pr_q = [results[l]['pr_quest'] for l in layers]

    corr_cr_t, _ = spearmanr(layer_indices, cr_t) if len(layers) > 2 else (0, 1)
    corr_cr_q, _ = spearmanr(layer_indices, cr_q) if len(layers) > 2 else (0, 1)
    corr_cos_tq, _ = spearmanr(layer_indices, cos_tq) if len(layers) > 2 else (0, 1)
    corr_pr_c, _ = spearmanr(layer_indices, pr_c) if len(layers) > 2 else (0, 1)
    corr_dim90, _ = spearmanr(layer_indices, dim90) if len(layers) > 2 else (0, 1)
    corr_eff, _ = spearmanr(layer_indices, eff) if len(layers) > 2 else (0, 1)
    corr_eff_ratio, _ = spearmanr(layer_indices, eff_ratio) if len(layers) > 2 else (0, 1)
    corr_pr_t, _ = spearmanr(layer_indices, pr_t) if len(layers) > 2 else (0, 1)
    corr_pr_q, _ = spearmanr(layer_indices, pr_q) if len(layers) > 2 else (0, 1)

    log(f"\n  Depth Correlations:")
    log(f"    Corr(L, CR_tense)     = {corr_cr_t:+.3f}  {'↑' if corr_cr_t > 0.3 else '↓' if corr_cr_t < -0.3 else '~'}")
    log(f"    Corr(L, CR_quest)     = {corr_cr_q:+.3f}  {'↑' if corr_cr_q > 0.3 else '↓' if corr_cr_q < -0.3 else '~'}")
    log(f"    Corr(L, cos_TQ)       = {corr_cos_tq:+.3f}  {'→orth' if corr_cos_tq < -0.3 else '→parallel' if corr_cos_tq > 0.3 else '~'}")
    log(f"    Corr(L, PR_causal)    = {corr_pr_c:+.3f}  {'↑more dims' if corr_pr_c > 0.3 else '↓fewer dims' if corr_pr_c < -0.3 else '~'}")
    log(f"    Corr(L, dim_90)       = {corr_dim90:+.3f}  {'↑more dims' if corr_dim90 > 0.3 else '↓fewer dims' if corr_dim90 < -0.3 else '~'}")
    log(f"    Corr(L, efficacy)     = {corr_eff:+.3f}  {'↑' if corr_eff > 0.3 else '↓' if corr_eff < -0.3 else '~'}")
    log(f"    Corr(L, eff_ratio)    = {corr_eff_ratio:+.3f}  {'↑' if corr_eff_ratio > 0.3 else '↓' if corr_eff_ratio < -0.3 else '~'}")
    log(f"    Corr(L, PR_tense)     = {corr_pr_t:+.3f}")
    log(f"    Corr(L, PR_quest)     = {corr_pr_q:+.3f}")

    # Key predictions
    log(f"\n  PREDICTIONS:")
    log(f"    {'SUPPORTED' if corr_cr_t > 0.3 else 'NOT SUPPORTED'}: "
        f"CR_tense increases with depth (Corr={corr_cr_t:+.3f})")
    log(f"    {'SUPPORTED' if corr_cr_q > 0.3 else 'NOT SUPPORTED'}: "
        f"CR_quest increases with depth (Corr={corr_cr_q:+.3f})")
    log(f"    {'SUPPORTED' if corr_cos_tq < -0.3 else 'NOT SUPPORTED'}: "
        f"Tense-Quest become more orthogonal with depth (Corr={corr_cos_tq:+.3f})")
    log(f"    {'SUPPORTED' if corr_eff_ratio > 0.3 else 'NOT SUPPORTED'}: "
        f"Causal efficacy ratio increases with depth (Corr={corr_eff_ratio:+.3f})")

    n_supported = sum([
        corr_cr_t > 0.3,
        corr_cr_q > 0.3,
        corr_cos_tq < -0.3,
        corr_eff_ratio > 0.3,
    ])

    log(f"\n  INTERPRETATION:")
    if n_supported >= 3:
        log(f"  => Causal Direction Standardization model STRONGLY SUPPORTED!")
        log(f"  => Deep layers have more consistent, orthogonal, and effective causal directions")
    elif n_supported >= 2:
        log(f"  => Causal Direction Standardization model PARTIALLY supported")
    elif n_supported >= 1:
        log(f"  => Weak evidence for causal direction standardization")
    else:
        log(f"  => Causal Direction Standardization model NOT supported")
        log(f"  => Causal directions remain distributed/variable across depth")

    # Save results
    save_data = {
        'model': model_key,
        'layers': layers,
        'correlations': {
            'cr_tense_vs_depth': float(corr_cr_t),
            'cr_quest_vs_depth': float(corr_cr_q),
            'cos_tq_vs_depth': float(corr_cos_tq),
            'pr_causal_vs_depth': float(corr_pr_c),
            'dim90_vs_depth': float(corr_dim90),
            'efficacy_vs_depth': float(corr_eff),
            'efficacy_ratio_vs_depth': float(corr_eff_ratio),
            'pr_tense_vs_depth': float(corr_pr_t),
            'pr_quest_vs_depth': float(corr_pr_q),
        },
        'layer_results': {str(k): v for k, v in results.items()},
    }

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    del model
    torch.cuda.empty_cache()

    log(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Results saved to {out_dir}/results.json")


if __name__ == '__main__':
    main()
