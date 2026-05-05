"""
CCML Phase 46: Force Field Reconstruction — From Statistics to Structure
========================================================================

User's decisive critique of Phase 45:
  "You're at the dangerous edge of overfitting to experimental phenomena.
   Your biggest risk is mistaking statistical phenomena for structural laws."

5 Fatal Hard Injuries identified:
  1. PCA ≠ semantic subspace (PCA maximizes variance, not separation)
  2. h_base decomposition is coordinate-dependent (not gauge-fixed)
  3. Δh dim≈16 is likely sample artifact (N=20, max dim=N-1=19)
  4. Tangential ≠ manifold dynamics (need to prove Δh ⟂ normal space)
  5. next-token ≠ semantic label (W_U analysis is misaligned)

Core mission: Reconstruct the FORCE FIELD F_l
  h_{l+1} = h_l + F_l(h_l)

  What type of system is F_l?
  A. Linear: Δh ≈ A_l h + b_l
  B. Gradient: Δh ≈ -∇V(h) → Δh·h < 0
  C. Rotation: Δh ≈ Ω(h)h (Ω antisymmetric) → Δh·h ≈ 0
  D. Hamiltonian-like: conservative rotation
  E. Mixed: attention + MLP

Experiments:
  46A: Large Sample Validation (200+ sentences)
    - Test if Δh dim90≈16 is real or sample artifact
    - If dim90 stays at 16 with N=200 → intrinsic structure
    - If dim90 grows with N → sample limitation

  46B: Linear Operator Fitting — Δh ≈ A_l h + b_l
    - Fit linear model at each layer
    - Measure rank(A_l), eigenvalue spectrum
    - R² of fit → how linear is F_l?

  46C: Gradient vs Rotation Field Test
    - Compute Δh·h at each layer for each sentence
    - If Δh·h < 0 → gradient field (energy minimization)
    - If Δh·h ≈ 0 → rotation field (conservative dynamics)
    - If mixed → complex dynamics

  46D: Supervised Semantic Subspace (LDA)
    - Fix hard injury 1: use LDA instead of PCA
    - LDA finds discriminative directions, not max-variance
    - Compare LDA dimensions with PCA dimensions

Usage:
  python ccml_phase46_force_field.py --model deepseek7b --exp 1
  python ccml_phase46_force_field.py --model deepseek7b --exp 2
  python ccml_phase46_force_field.py --model deepseek7b --exp 3
  python ccml_phase46_force_field.py --model deepseek7b --exp 4
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import torch
import numpy as np
import gc
import time
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from model_utils import (load_model, get_layers, get_model_info, release_model,
                          safe_decode, collect_layer_outputs, get_W_U)

# ===== Large sentence pool (200+ sentences) =====
# We need MANY sentences to validate dimensionality estimates
LARGE_SENTENCE_POOL = [
    # Animals (30)
    "The cat sat on the mat", "A dog ran through the park", "The bird flew over the tree",
    "My fish swam in the pond", "The horse galloped across the field", "A rabbit hopped through the garden",
    "The elephant drank from the river", "The snake slithered under the rock", "The whale dove into the ocean",
    "The bear climbed the mountain", "The lion roared in the savanna", "The penguin waddled on the ice",
    "The deer grazed in the meadow", "The fox sneaked through the forest", "The owl hooted in the night",
    "The dolphin jumped over the waves", "The tiger hunted in the jungle", "The eagle soared above the clouds",
    "The turtle crawled along the beach", "The monkey swung from tree to tree",
    "A small kitten chased the ball", "The old hound slept by the fire", "Wild wolves howled at the moon",
    "The parrot repeated every word", "The shark circled the boat slowly", "The butterfly landed on the flower",
    "The squirrel gathered nuts quickly", "The crocodile waited in the river", "The hummingbird hovered near the bloom",
    "The cheetah sprinted across the plain",
    # Science (30)
    "The scientist discovered a new element", "Researchers found evidence of dark matter",
    "The experiment confirmed the hypothesis", "Chemists synthesized a novel compound",
    "Physicists measured the particle decay", "Biologists identified a new species",
    "The telescope observed distant galaxies", "The microscope revealed cell structures",
    "Mathematicians proved the conjecture", "Engineers designed a new circuit",
    "The laboratory tested the sample carefully", "The computer simulated the reaction",
    "Geneticists mapped the chromosome sequence", "Astronomers detected the exoplanet signal",
    "The reactor produced stable fusion energy", "Geologists dated the rock formation",
    "The algorithm optimized the parameters", "The sensor recorded the temperature change",
    "Climate models predicted the warming trend", "The catalyst accelerated the reaction rate",
    "Neuroscientists studied the brain activity", "The satellite captured the hurricane image",
    "Pharmacologists developed the new vaccine", "The robot performed the delicate surgery",
    "Ecologists monitored the coral reef health", "The database stored the genomic information",
    "Quantum computers solved the optimization", "The spectrograph analyzed the light spectrum",
    "Volcanologists predicted the eruption timing", "The nanomaterial exhibited unusual properties",
    # Emotion (30)
    "She felt happy about the news", "He was angry at the decision", "They felt sad about the loss",
    "The child was excited for the trip", "She felt worried about the exam", "He was disgusted by the smell",
    "The woman felt proud of her work", "The man was surprised by the result",
    "She felt calm after the meditation", "He was confused by the instructions",
    "The student felt nervous before the test", "The mother felt relieved after the call",
    "He was embarrassed by the mistake", "She felt grateful for the help",
    "The player was frustrated by the loss", "The artist felt inspired by the sunset",
    "He was lonely in the new city", "She felt hopeful about the future",
    "The worker was exhausted after the shift", "The child felt scared of the dark",
    "She was jealous of her success", "He felt guilty about the accident",
    "The old man was nostalgic for the past", "She felt peaceful in the garden",
    "He was bored during the lecture", "The fan was thrilled by the concert",
    "She felt anxious about the interview", "He was disappointed by the movie",
    "The couple felt content with their life", "The leader felt confident about the plan",
    # Actions (30)
    "She walked to the store yesterday", "He cooked dinner for the family", "They built a house last year",
    "She wrote a letter to her friend", "He drove the car to work", "They planted trees in the garden",
    "She painted the wall blue", "He fixed the broken window", "They cleaned the entire house",
    "She played the piano beautifully", "He read the book carefully", "They danced at the party",
    "She sang a song for the audience", "He ran five miles this morning", "They swam across the lake",
    "She taught the class effectively", "He designed the new building", "They researched the topic thoroughly",
    "She organized the conference", "He managed the project successfully", "They discussed the problem",
    "She translated the document", "He programmed the application", "They manufactured the product",
    "She delivered the presentation", "He repaired the old engine", "They investigated the crime",
    "She composed the music piece", "He directed the film crew", "They negotiated the contract",
    # Descriptions (30)
    "The sky was blue and clear", "The room was dark and cold", "The water was warm and calm",
    "The mountain was tall and steep", "The flower was red and beautiful", "The road was long and winding",
    "The building was old and crumbling", "The lake was still and peaceful", "The forest was dense and green",
    "The desert was vast and empty", "The city was busy and noisy", "The garden was small but lovely",
    "The ocean was deep and mysterious", "The valley was wide and fertile", "The castle was ancient and grand",
    "The river was fast and dangerous", "The island was remote and quiet", "The bridge was long and narrow",
    "The cave was dark and damp", "The hill was gentle and rolling", "The sunset was orange and pink",
    "The snow was white and powdery", "The rain was heavy and cold", "The wind was strong and gusty",
    "The fog was thick and grey", "The ice was smooth and slippery", "The sand was soft and warm",
    "The rock was hard and rough", "The metal was shiny and cold", "The wood was dark and polished",
    # Questions (30)
    "What is the meaning of life?", "How does the brain process language?", "Why do people make mistakes?",
    "When will the rain stop falling?", "Where can I find the answer?", "Who discovered the new method?",
    "Which path leads to success?", "Can machines understand emotions?", "Should we explore the universe?",
    "Is knowledge always beneficial?", "What causes the seasons to change?", "How do birds navigate long distances?",
    "Why is the sky blue at noon?", "When did civilization first begin?", "Where do rivers find their source?",
    "Who wrote the first novel?", "Which element is the most abundant?", "Can we travel faster than light?",
    "Should everyone vote in elections?", "Is art subjective or objective?", "What determines intelligence?",
    "How do vaccines provide immunity?", "Why do dreams feel so real?", "When will AI surpass humans?",
    "Where does gravity come from?", "Who invented the printing press?", "Which planet has the most moons?",
    "Can we live on other planets?", "Should science have boundaries?", "Is time travel theoretically possible?",
    # Negation (20)
    "The cat did not sit on the mat", "She never walked to the store", "He was not happy about the news",
    "They could not find the answer", "The experiment did not confirm it", "She had no reason to complain",
    "He never said anything wrong", "They did not understand the problem", "The system was not working properly",
    "She could not believe the result", "No one expected the outcome", "The project was never completed",
    "He did not want to leave early", "They were not ready for the test", "The machine did not function correctly",
    "She never visited that place", "The answer was not obvious", "He could not solve the equation",
    "They did not notice the change", "The weather was not cooperating",
]

# Assign categories
LARGE_CATEGORIES = (
    ["animals"] * 30 + ["science"] * 30 + ["emotion"] * 30 + ["actions"] * 30 +
    ["descriptions"] * 30 + ["questions"] * 30 + ["negation"] * 20
)


def get_all_hidden_states(model, tokenizer, device, text, n_layers, d_model):
    """Get hidden states at ALL layers. Returns: [n_layers, d_model]"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).detach()
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    layers = get_layers(model)
    captured = {}

    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook

    hooks = []
    for li in range(n_layers):
        hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))

    with torch.no_grad():
        try:
            _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                     attention_mask=attention_mask)
        except Exception as e:
            pass

    for h in hooks:
        h.remove()

    result = np.zeros((n_layers, d_model), dtype=np.float32)
    for li in range(n_layers):
        key = f"L{li}"
        if key in captured:
            result[li] = captured[key][0, -1, :].numpy()

    del captured
    torch.cuda.empty_cache()
    return result


# ============================================================================
# 46A: Large Sample Validation — Is Δh dim≈16 real?
# ============================================================================

def run_46A(model_name):
    """
    THE most critical validation: test if Δh dim90≈16 is real or sample artifact.
    
    If N=200 and dim90 stays ≈16 → intrinsic structure
    If N=200 and dim90 grows → sample limitation
    """
    print(f"\n{'='*70}")
    print("46A: Large Sample Validation — Is Δh dim≈16 Real?")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  Model: {model_name}, layers={n_layers}, d_model={d_model}")

    # We'll collect trajectories at sampled layers
    # Key layers: early, mid, late (skip final 2 to avoid LN artifact)
    sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-3, n_layers-1]
    sample_layers = sorted(set([l for l in sample_layers if 0 <= l < n_layers]))
    print(f"  Sample layers: {sample_layers}")

    # Progressive collection: first N sentences, then more
    test_sizes = [20, 50, 100, 200]
    all_sentences = LARGE_SENTENCE_POOL[:200]
    all_cats = LARGE_CATEGORIES[:200]

    print(f"\n  Total sentences available: {len(all_sentences)}")

    # Collect all hidden states first
    print("\n  Collecting hidden states for all sentences...")
    t0 = time.time()

    # {layer: [h_1, h_2, ...]} for ALL sentences
    layer_hidden_all = {li: [] for li in sample_layers}
    layer_h_all = {li: [] for li in sample_layers}  # full trajectory for Δh

    # Also need adjacent layers for Δh computation
    delta_layers = []
    for li in sample_layers:
        if li + 1 < n_layers:
            delta_layers.append(li)

    for i, sent in enumerate(all_sentences):
        hs = get_all_hidden_states(model, tokenizer, device, sent, n_layers, d_model)
        for li in sample_layers:
            layer_hidden_all[li].append(hs[li])
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/200 sentences ({time.time()-t0:.1f}s)")

    print(f"  Collection complete: {time.time()-t0:.1f}s")

    # ===== Analysis: Progressive dimensionality =====
    print("\n  ===== Progressive Dimensionality Analysis =====")
    print(f"  Testing if Δh dim90 depends on sample size N")

    for li in sample_layers[:-1]:  # Skip final layer (LN artifact)
        if li + 1 >= n_layers:
            continue

        print(f"\n  --- Layer {li} → {li+1} (Δh) ---")
        print(f"  {'N':>6} {'dim90':>6} {'dim50':>6} {'dim95':>6} {'top1%':>8} {'top3%':>8} {'top10%':>8}")

        for N in test_sizes:
            # Compute Δh for first N sentences
            delta_h_list = []
            for i in range(N):
                # Need h at layer li and li+1
                # We stored at sample_layers, but may not have li+1
                # Recompute from stored data
                # Actually we need to recompute for adjacent layers
                pass

        # We need to recompute Δh properly - collect h at li and li+1
        # Since we only stored at sample_layers, we need adjacent pairs
        # Let's re-collect for adjacent layers specifically

    # ===== Refined approach: collect at adjacent pairs =====
    print("\n  ===== Refined: Collecting at adjacent layer pairs =====")

    # Pick 3 representative layers for Δh analysis
    delta_test_layers = [n_layers//4, n_layers//2, 3*n_layers//4]
    # Make sure li+1 is also valid
    delta_test_layers = [l for l in delta_test_layers if l + 1 < n_layers - 2]  # avoid final LN

    print(f"  Δh test layers: {delta_test_layers}")

    # Collect h at layer li and li+1 for each test layer
    delta_data = {li: {'h_l': [], 'h_lp1': []} for li in delta_test_layers}

    for i, sent in enumerate(all_sentences):
        hs = get_all_hidden_states(model, tokenizer, device, sent, n_layers, d_model)
        for li in delta_test_layers:
            delta_data[li]['h_l'].append(hs[li])
            delta_data[li]['h_lp1'].append(hs[li+1])
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/200 sentences ({time.time()-t0:.1f}s)")

    print(f"  Δh data collection complete: {time.time()-t0:.1f}s")

    # Now do progressive analysis
    print("\n  ===== Progressive Δh Dimensionality =====")

    results_by_layer = {}

    for li in delta_test_layers:
        h_l_all = np.array(delta_data[li]['h_l'])      # [N, d_model]
        h_lp1_all = np.array(delta_data[li]['h_lp1'])  # [N, d_model]
        delta_h_all = h_lp1_all - h_l_all               # [N, d_model]

        print(f"\n  --- Layer {li} → {li+1} ---")
        print(f"  {'N':>6} {'dim90':>6} {'dim50':>6} {'dim95':>6} {'top1%':>8} {'top3%':>8} {'top10%':>8}")

        layer_results = []

        for N in test_sizes:
            delta_h = delta_h_all[:N]
            centered = delta_h - delta_h.mean(axis=0)

            U, s, Vt = np.linalg.svd(centered, full_matrices=False)
            total_var = np.sum(s**2)

            if total_var < 1e-10:
                continue

            cum_var = np.cumsum(s**2) / total_var
            dim90 = int(np.searchsorted(cum_var, 0.90) + 1)
            dim50 = int(np.searchsorted(cum_var, 0.50) + 1)
            dim95 = int(np.searchsorted(cum_var, 0.95) + 1)
            top1 = s[0]**2 / total_var * 100
            top3 = np.sum(s[:3]**2) / total_var * 100
            top10 = np.sum(s[:min(10, len(s))]**2) / total_var * 100

            print(f"  {N:>6} {dim90:>6} {dim50:>6} {dim95:>6} {top1:>7.1f}% {top3:>7.1f}% {top10:>7.1f}%")

            layer_results.append({
                'N': N, 'dim90': dim90, 'dim50': dim50, 'dim95': dim95,
                'top1_pct': top1, 'top3_pct': top3, 'top10_pct': top10,
            })

        results_by_layer[li] = layer_results

        # Check: does dim90 grow with N?
        dim90s = [r['dim90'] for r in layer_results]
        if len(dim90s) >= 3:
            if dim90s[-1] > dim90s[0] * 1.5:
                print(f"  ⚠️ dim90 GROWS with N → likely sample artifact")
            else:
                print(f"  ✅ dim90 STABLE with N → likely intrinsic structure")

    # ===== Also test h_sem dimensionality with large N =====
    print("\n  ===== h_sem Dimensionality vs N =====")
    final_li = n_layers - 3  # avoid final LN
    print(f"  Using layer {final_li}")

    # Collect h at this layer
    h_at_layer = []
    for i, sent in enumerate(all_sentences):
        hs = get_all_hidden_states(model, tokenizer, device, sent, n_layers, d_model)
        h_at_layer.append(hs[final_li])
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/200 sentences")

    h_at_layer = np.array(h_at_layer)  # [200, d_model]

    print(f"  {'N':>6} {'dim90_h':>8} {'dim90_hsem':>10} {'dim50_hsem':>10} {'top1_hsem':>10}")

    for N in test_sizes:
        H = h_at_layer[:N]
        h_base = H.mean(axis=0)
        H_sem = H - h_base[np.newaxis, :]

        # Full H centered
        U_h, s_h, _ = np.linalg.svd(H - H.mean(axis=0), full_matrices=False)
        tv_h = np.sum(s_h**2)
        dim90_h = int(np.searchsorted(np.cumsum(s_h**2)/tv_h, 0.90) + 1) if tv_h > 1e-10 else 0

        # h_sem
        U_s, s_s, _ = np.linalg.svd(H_sem, full_matrices=False)
        tv_s = np.sum(s_s**2)
        if tv_s > 1e-10:
            cum_s = np.cumsum(s_s**2) / tv_s
            dim90_s = int(np.searchsorted(cum_s, 0.90) + 1)
            dim50_s = int(np.searchsorted(cum_s, 0.50) + 1)
            top1_s = s_s[0]**2 / tv_s * 100
        else:
            dim90_s, dim50_s, top1_s = 0, 0, 0

        print(f"  {N:>6} {dim90_h:>8} {dim90_s:>10} {dim50_s:>10} {top1_s:>9.1f}%")

    # ===== Summary =====
    print("\n  ===== 46A SUMMARY =====")
    print(f"  Key question: Does Δh dim90 stay ≈16 when N increases?")
    print(f"  If YES → intrinsic structure, ~16-dim tangent dynamics")
    print(f"  If NO → sample artifact, real dim is much higher")

    for li, res in results_by_layer.items():
        dim90s = [r['dim90'] for r in res]
        Ns = [r['N'] for r in res]
        print(f"  Layer {li}: dim90 = {dim90s} at N = {Ns}")

    # Save
    out = {
        'model': model_name,
        'delta_results': {str(k): v for k, v in results_by_layer.items()},
    }
    out_path = f"tests/glm5_temp/phase46A_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    release_model(model)
    return out


# ============================================================================
# 46B: Linear Operator Fitting — Δh ≈ A_l h + b_l
# ============================================================================

def run_46B(model_name):
    """
    Fit linear model: Δh_l = A_l · h_l + b_l
    
    Key questions:
    1. How well does the linear model fit? (R²)
    2. What is the effective rank of A_l?
    3. What is the eigenvalue spectrum of A_l?
    """
    print(f"\n{'='*70}")
    print("46B: Linear Operator Fitting — Δh ≈ A h + b")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  Model: {model_name}, layers={n_layers}, d_model={d_model}")

    # Use 100 sentences for fitting
    N_fit = 100
    sentences = LARGE_SENTENCE_POOL[:N_fit]

    # Test at several layers (avoid final 2 for LN artifact)
    test_layers = list(range(0, n_layers - 2, max(1, n_layers // 8)))
    test_layers = [l for l in test_layers if l + 1 < n_layers - 2]
    print(f"  Test layers: {test_layers}")

    # Collect h_l and h_{l+1}
    print("\n  Collecting hidden states...")
    t0 = time.time()

    # {layer: {'h_l': [N, d], 'h_lp1': [N, d]}}
    layer_data = {}

    for li in test_layers:
        layer_data[li] = {'h_l': [], 'h_lp1': []}

    for i, sent in enumerate(sentences):
        hs = get_all_hidden_states(model, tokenizer, device, sent, n_layers, d_model)
        for li in test_layers:
            layer_data[li]['h_l'].append(hs[li])
            layer_data[li]['h_lp1'].append(hs[li + 1])
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{N_fit} ({time.time()-t0:.1f}s)")

    print(f"  Collection complete: {time.time()-t0:.1f}s")

    # ===== Fit linear model at each layer =====
    print("\n  ===== Linear Model Fitting =====")
    print(f"  {'Layer':>6} {'R²_mean':>8} {'R²_std':>8} {'rank_A':>8} {'||A||_F':>10} {'||b||':>10} "
          f"{'top_eig':>10} {'eig_sign':>10}")

    for li in test_layers:
        h_l = np.array(layer_data[li]['h_l'])      # [N, d_model]
        h_lp1 = np.array(layer_data[li]['h_lp1'])  # [N, d_model]
        delta_h = h_lp1 - h_l                       # [N, d_model]

        N = h_l.shape[0]
        d = h_l.shape[1]

        # Fit: delta_h = A @ h_l + b
        # Stack: [h_l, 1] @ [A; b]^T = delta_h
        # Use ridge regression for stability
        H_aug = np.hstack([h_l, np.ones((N, 1))])  # [N, d+1]

        # Fit per output dimension
        r2_list = []
        A_cols = []
        b_vals = []

        for j in range(min(d, d_model)):  # fit each output dim
            y = delta_h[:, j]
            clf = Ridge(alpha=1.0)
            clf.fit(H_aug, y)
            y_pred = clf.predict(H_aug)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = 1 - ss_res / max(ss_tot, 1e-20)
            r2_list.append(r2)
            A_cols.append(clf.coef_[:d])  # first d coefficients = row of A
            b_vals.append(clf.coef_[d])   # last coefficient = b

        A = np.array(A_cols)  # [d, d]
        b = np.array(b_vals)  # [d]

        # Effective rank of A
        U_A, s_A, Vt_A = np.linalg.svd(A, full_matrices=False)
        total_energy = np.sum(s_A**2)
        if total_energy > 1e-10:
            cum_energy = np.cumsum(s_A**2) / total_energy
            rank90 = int(np.searchsorted(cum_energy, 0.90) + 1)
        else:
            rank90 = 0

        # Eigenvalue spectrum of A
        # A is [d, d], compute eigenvalues
        # For stability, use SVD-based approximation
        # eigenvalues of A ≈ complex, but we check the real parts
        eig_A = np.linalg.eigvals(A[:min(50, d), :min(50, d)])
        top_eig = np.max(np.abs(eig_A))
        mean_real = np.mean(np.real(eig_A))
        n_positive = np.sum(np.real(eig_A) > 0)
        n_negative = np.sum(np.real(eig_A) < 0)

        A_norm = np.linalg.norm(A, 'fro')
        b_norm = np.linalg.norm(b)

        print(f"  {li:>6} {np.mean(r2_list):>8.4f} {np.std(r2_list):>8.4f} {rank90:>8} "
              f"{A_norm:>10.2f} {b_norm:>10.2f} {top_eig:>10.4f} "
              f"+{n_positive}/-{n_negative}")

        # Save A's singular values for spectrum analysis
        if li == test_layers[len(test_layers)//2]:  # middle layer
            sv_A = s_A[:50]
            print(f"    A singular values (top 20): {sv_A[:20].round(4)}")

    # ===== Also test: how much of Δh is explained by h_l itself? =====
    print("\n  ===== Δh vs h_l correlation =====")
    print(f"  {'Layer':>6} {'Mean cos(Δh, h)':>16} {'Mean Δh·h':>12} {'Mean ||Δh||':>12} {'Mean ||h||':>12}")

    for li in test_layers:
        h_l = np.array(layer_data[li]['h_l'])
        h_lp1 = np.array(layer_data[li]['h_lp1'])
        delta_h = h_lp1 - h_l

        # cos(Δh, h) for each sentence
        cos_dh_h = []
        dot_dh_h = []
        for i in range(len(h_l)):
            d_norm = np.linalg.norm(delta_h[i])
            h_norm = np.linalg.norm(h_l[i])
            if d_norm > 1e-10 and h_norm > 1e-10:
                cos_dh_h.append(np.dot(delta_h[i], h_l[i]) / (d_norm * h_norm))
                dot_dh_h.append(np.dot(delta_h[i], h_l[i]))

        mean_cos = np.mean(cos_dh_h) if cos_dh_h else 0
        mean_dot = np.mean(dot_dh_h) if dot_dh_h else 0

        print(f"  {li:>6} {mean_cos:>16.6f} {mean_dot:>12.2f} "
              f"{np.mean(np.linalg.norm(delta_h, axis=1)):>12.2f} "
              f"{np.mean(np.linalg.norm(h_l, axis=1)):>12.2f}")

    # ===== Summary =====
    print("\n  ===== 46B SUMMARY =====")
    print(f"  Linear model R²: measures how much of Δh is linearly predictable from h_l")
    print(f"  rank(A): effective rank of the linear operator")
    print(f"  eig sign: distribution of positive/negative eigenvalues")

    release_model(model)
    return {'model': model_name}


# ============================================================================
# 46C: Gradient vs Rotation Field Test
# ============================================================================

def run_46C(model_name):
    """
    THE most decisive test for the nature of F_l:
    
    If Δh·h < 0 → gradient field (energy decreasing)
    If Δh·h ≈ 0 → rotation field (conservative dynamics)
    If Δh·h > 0 → expanding field
    
    Phase 45 showed radial% ≈ 9% (low), suggesting rotation.
    But we need to be precise: Δh·h = ||h_{l+1}||^2 - ||h_l||^2 - ||Δh||^2
    """
    print(f"\n{'='*70}")
    print("46C: Gradient vs Rotation Field Test")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  Model: {model_name}, layers={n_layers}, d_model={d_model}")

    N = 100
    sentences = LARGE_SENTENCE_POOL[:N]

    # Collect full trajectories
    print("\n  Collecting full trajectories...")
    t0 = time.time()

    trajectories = []
    for i, sent in enumerate(sentences):
        hs = get_all_hidden_states(model, tokenizer, device, sent, n_layers, d_model)
        trajectories.append(hs)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{N} ({time.time()-t0:.1f}s)")

    print(f"  Collection complete: {time.time()-t0:.1f}s")

    # ===== Core analysis: Δh·h =====
    print("\n  ===== Δh·h Analysis (Gradient vs Rotation) =====")
    print(f"  {'Layer':>6} {'Mean Δh·h':>12} {'Std Δh·h':>12} {'%negative':>10} "
          f"{'%near0':>10} {'%positive':>10} {'Mean cos':>10}")

    # For each layer, compute Δh·h for all sentences
    dot_products_by_layer = {}

    for l in range(n_layers - 2):  # skip final 2 (LN artifact)
        delta_h = np.array([trajectories[i][l+1] - trajectories[i][l] for i in range(N)])
        h_l = np.array([trajectories[i][l] for i in range(N)])

        # Δh·h for each sentence
        dot_dh_h = np.sum(delta_h * h_l, axis=1)  # [N]

        # cos(Δh, h)
        cos_dh_h = []
        for i in range(N):
            d_norm = np.linalg.norm(delta_h[i])
            h_norm = np.linalg.norm(h_l[i])
            if d_norm > 1e-10 and h_norm > 1e-10:
                cos_dh_h.append(np.dot(delta_h[i], h_l[i]) / (d_norm * h_norm))

        # Classify: negative, near-zero, positive
        n_neg = np.sum(dot_dh_h < -0.01 * np.mean(np.abs(dot_dh_h)))
        n_pos = np.sum(dot_dh_h > 0.01 * np.mean(np.abs(dot_dh_h)))
        n_zero = N - n_neg - n_pos

        dot_products_by_layer[l] = dot_dh_h

        if l % max(1, n_layers // 12) == 0 or l == n_layers - 3:
            print(f"  {l:>6} {np.mean(dot_dh_h):>12.2f} {np.std(dot_dh_h):>12.2f} "
                  f"{n_neg/N*100:>9.1f}% {n_zero/N*100:>9.1f}% {n_pos/N*100:>9.1f}% "
                  f"{np.mean(cos_dh_h):>10.4f}")

    # ===== Decompose Δh into radial and tangential components =====
    print("\n  ===== Radial vs Tangential Decomposition =====")
    print(f"  {'Layer':>6} {'Radial%':>10} {'Tangential%':>14} {'||radial||':>12} {'||tang||':>12}")

    for l in range(n_layers - 2):
        if l % max(1, n_layers // 12) != 0 and l != n_layers - 3:
            continue

        delta_h = np.array([trajectories[i][l+1] - trajectories[i][l] for i in range(N)])
        h_l = np.array([trajectories[i][l] for i in range(N)])

        radial_norms = []
        tang_norms = []

        for i in range(N):
            h_norm = np.linalg.norm(h_l[i])
            d_norm = np.linalg.norm(delta_h[i])
            if h_norm > 1e-10:
                h_unit = h_l[i] / h_norm
                proj = np.dot(delta_h[i], h_unit)
                radial = proj * h_unit
                tangential = delta_h[i] - radial
                radial_norms.append(np.linalg.norm(radial))
                tang_norms.append(np.linalg.norm(tangential))

        if radial_norms:
            r_mean = np.mean(radial_norms)
            t_mean = np.mean(tang_norms)
            total = r_mean + t_mean
            print(f"  {l:>6} {r_mean/total*100:>9.1f}% {t_mean/total*100:>13.1f}% "
                  f"{r_mean:>12.2f} {t_mean:>12.2f}")

    # ===== Key test: ||h_{l+1}||^2 vs ||h_l||^2 =====
    print("\n  ===== Norm Evolution: ||h_{l+1}||² - ||h_l||² =====")
    print(f"  {'Layer':>6} {'Mean Δ||h||²':>14} {'%expanding':>12} {'%contracting':>14} {'%stable':>10}")

    for l in range(n_layers - 2):
        if l % max(1, n_layers // 12) != 0 and l != n_layers - 3:
            continue

        norms_sq_l = np.array([np.linalg.norm(trajectories[i][l])**2 for i in range(N)])
        norms_sq_lp1 = np.array([np.linalg.norm(trajectories[i][l+1])**2 for i in range(N)])
        delta_norm_sq = norms_sq_lp1 - norms_sq_l

        mean_delta = np.mean(delta_norm_sq)
        threshold = 0.01 * np.mean(np.abs(delta_norm_sq))
        n_expand = np.sum(delta_norm_sq > threshold)
        n_contract = np.sum(delta_norm_sq < -threshold)
        n_stable = N - n_expand - n_contract

        print(f"  {l:>6} {mean_delta:>14.2f} {n_expand/N*100:>11.1f}% "
              f"{n_contract/N*100:>13.1f}% {n_stable/N*100:>9.1f}%")

    # ===== Summary =====
    print("\n  ===== 46C SUMMARY =====")

    # Aggregate over middle layers (skip first 3 and last 3)
    mid_layers = list(range(3, n_layers - 4))
    all_cos = []
    all_dot_signs = []

    for l in mid_layers:
        delta_h = np.array([trajectories[i][l+1] - trajectories[i][l] for i in range(N)])
        h_l = np.array([trajectories[i][l] for i in range(N)])
        for i in range(N):
            d_norm = np.linalg.norm(delta_h[i])
            h_norm = np.linalg.norm(h_l[i])
            if d_norm > 1e-10 and h_norm > 1e-10:
                cos_val = np.dot(delta_h[i], h_l[i]) / (d_norm * h_norm)
                all_cos.append(cos_val)
                all_dot_signs.append(np.sign(np.dot(delta_h[i], h_l[i])))

    mean_cos = np.mean(all_cos)
    frac_pos = np.mean([1 for s in all_dot_signs if s > 0])
    frac_neg = np.mean([1 for s in all_dot_signs if s < 0])
    frac_zero = 1 - frac_pos - frac_neg

    print(f"  Mean cos(Δh, h) across middle layers: {mean_cos:.6f}")
    print(f"  Sign of Δh·h: +{frac_pos:.1%} / -{frac_neg:.1%} / 0:{frac_zero:.1%}")

    if abs(mean_cos) < 0.05:
        print(f"\n  ★★★ ROTATION FIELD: Δh·h ≈ 0 → conservative rotation dynamics ★★★")
        print(f"  → Language model dynamics is ROTATION-dominant")
        print(f"  → F_l is approximately an antisymmetric operator")
    elif mean_cos < -0.05:
        print(f"\n  ★★★ GRADIENT FIELD: Δh·h < 0 → energy decreasing ★★★")
        print(f"  → Language model dynamics is gradient-like (energy minimization)")
    elif mean_cos > 0.05:
        print(f"\n  ★★★ EXPANDING FIELD: Δh·h > 0 → norm increasing ★★★")
        print(f"  → Language model dynamics is expanding")
    else:
        print(f"\n  Mixed dynamics: no clear gradient or rotation pattern")

    # Save
    out = {
        'model': model_name,
        'mean_cos_dh_h': float(mean_cos),
        'frac_positive': float(frac_pos),
        'frac_negative': float(frac_neg),
    }
    out_path = f"tests/glm5_temp/phase46C_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    release_model(model)
    return out


# ============================================================================
# 46D: Supervised Semantic Subspace (LDA vs PCA)
# ============================================================================

def run_46D(model_name):
    """
    Fix hard injury 1: use LDA instead of PCA for semantic subspace.
    
    PCA finds max-variance directions (may not be semantic).
    LDA finds max-separation directions (directly semantic).
    
    Compare:
    - PCA dim needed for 90% classification
    - LDA dim needed for 90% classification
    - Overlap between PCA and LDA subspaces
    """
    print(f"\n{'='*70}")
    print("46D: Supervised Semantic Subspace — LDA vs PCA")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  Model: {model_name}, layers={n_layers}, d_model={d_model}")

    # Use categorized sentences
    N = 180  # 30 per category × 6 categories
    sentences = LARGE_SENTENCE_POOL[:N]
    categories = LARGE_CATEGORIES[:N]

    # Only use categories with ≥30 samples
    unique_cats = list(set(categories))
    print(f"  Categories: {unique_cats}")
    print(f"  Samples per category: {min(categories.count(c) for c in unique_cats)}")

    # Collect h at mid-layer
    test_layer = n_layers // 2
    print(f"  Test layer: {test_layer}")

    print("\n  Collecting hidden states...")
    t0 = time.time()

    H = np.zeros((N, d_model), dtype=np.float32)
    for i, sent in enumerate(sentences):
        hs = get_all_hidden_states(model, tokenizer, device, sent, n_layers, d_model)
        H[i] = hs[test_layer]
        if (i + 1) % 30 == 0:
            print(f"    {i+1}/{N} ({time.time()-t0:.1f}s)")

    print(f"  Collection complete: {time.time()-t0:.1f}s")

    # Center
    h_base = H.mean(axis=0)
    H_sem = H - h_base[np.newaxis, :]

    y = np.array(categories)

    # ===== PCA =====
    print("\n  ===== PCA Analysis =====")
    U_pca, s_pca, Vt_pca = np.linalg.svd(H_sem, full_matrices=False)
    total_var = np.sum(s_pca**2)
    cum_var = np.cumsum(s_pca**2) / total_var

    # PCA classification
    from sklearn.linear_model import LogisticRegression
    for n_dim in [3, 5, 10, 20, 30, 50]:
        X_pca = U_pca[:, :n_dim] * s_pca[:n_dim]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        scores = cross_val_score(clf, X_pca, y, cv=5, scoring='accuracy')
        print(f"  PCA-{n_dim:>3}: accuracy = {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # ===== LDA =====
    print("\n  ===== LDA Analysis =====")
    n_lda_components = min(len(unique_cats) - 1, d_model, N - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_lda_components)
    X_lda = lda.fit_transform(H_sem, y)

    print(f"  LDA components available: {n_lda_components}")
    print(f"  LDA explained variance ratio (top 10): {lda.explained_variance_ratio_[:10].round(4)}")

    # LDA classification with increasing dimensions
    for n_dim in [1, 2, 3, 4, 5, min(10, n_lda_components)]:
        X_lda_sub = X_lda[:, :n_dim]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        scores = cross_val_score(clf, X_lda_sub, y, cv=5, scoring='accuracy')
        print(f"  LDA-{n_dim:>3}: accuracy = {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # ===== Compare PCA and LDA subspaces =====
    print("\n  ===== PCA vs LDA Subspace Overlap =====")
    # Project LDA directions onto PCA space
    # LDA scalings define the discriminative directions in original space
    lda_dirs = lda.scalings_  # [d_model, n_components]

    # PCA directions
    pca_dirs = Vt_pca[:50]  # [50, d_model]

    # Subspace angles between top-k PCA and LDA
    for k in [3, 5, 10]:
        if k > n_lda_components or k > 50:
            continue
        pca_sub = pca_dirs[:k].T  # [d_model, k]
        lda_sub = lda_dirs[:, :k]  # [d_model, k]

        # Principal angles
        from scipy.linalg import subspace_angles
        try:
            angles = subspace_angles(pca_sub, lda_sub)
            mean_angle = np.mean(angles) * 180 / np.pi
            print(f"  PCA-{k} vs LDA-{k}: mean angle = {mean_angle:.2f}°")
        except:
            # Compute overlap via projection
            Q_pca, _ = np.linalg.qr(pca_sub)
            Q_lda, _ = np.linalg.qr(lda_sub)
            overlap = np.trace(Q_pca.T @ Q_lda @ Q_lda.T @ Q_pca) / k
            print(f"  PCA-{k} vs LDA-{k}: overlap = {overlap:.4f}")

    # ===== Summary =====
    print("\n  ===== 46D SUMMARY =====")
    print(f"  PCA maximizes variance — may or may not align with semantic directions")
    print(f"  LDA maximizes separation — directly finds discriminative directions")
    print(f"  If PCA and LDA overlap → variance and semantics are aligned")
    print(f"  If they differ → max-variance ≠ max-separation")

    # Key comparison
    lda_5 = lda.explained_variance_ratio_[:5]
    pca_5 = s_pca[:5]**2 / total_var
    print(f"\n  Top-5 variance explained:")
    print(f"  PCA: {pca_5.round(4)}")
    print(f"  LDA: {lda_5.round(4)}")

    release_model(model)
    return {'model': model_name}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 46: Force Field Reconstruction")
    parser.add_argument("--model", type=str, default="deepseek7b",
                       choices=["deepseek7b", "glm4", "qwen3"])
    parser.add_argument("--exp", type=int, default=0,
                       help="Experiment: 1=46A, 2=46B, 3=46C, 4=46D, 0=all")
    args = parser.parse_args()

    if args.exp == 1 or args.exp == 0:
        run_46A(args.model)
        gc.collect()
        torch.cuda.empty_cache()

    if args.exp == 2 or args.exp == 0:
        run_46B(args.model)
        gc.collect()
        torch.cuda.empty_cache()

    if args.exp == 3 or args.exp == 0:
        run_46C(args.model)
        gc.collect()
        torch.cuda.empty_cache()

    if args.exp == 4 or args.exp == 0:
        run_46D(args.model)
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "="*70)
    print("Phase 46 complete!")
    print("="*70)


if __name__ == "__main__":
    main()
