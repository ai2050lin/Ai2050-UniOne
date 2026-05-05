"""
Phase 48: Language Ontology Verification — 语言本体验证
========================================================

用户核心批判（Phase 47→48）:
    1. PCA/LDA ≠ 真实语义坐标（只是"最容易读出的方向"）
    2. "线性可分" ≠ "本体低维"
    3. R²=0.53 被误判（47%未建模）
    4. Attn vs MLP "对抗"可能是伪影
    5. 语法2维结论不可信（N=8太少）

关键认知修正:
    层级1: 实现结构（Transformer/Residual/LayerNorm）
    层级2: 表示结构（PCA/LDA/可分性）
    层级3: 语言本体（组合性/语义关系/语法变换）
    
    ❌ 之前: 用表示结构解释语言本体
    ✅ 现在: 直接验证语言本体的数学结构

三个决定性实验:
    48A: 语义组合律 — z("red cat") ≈ z("red") + z("cat")?
    48B: 语法变换矩阵 — z_passive ≈ T · z_active?
    48C: Δz 动力模式分解 — Δz = Σ c_i · basis_i?
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
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.linalg import orthogonal_procrustes
import warnings
warnings.filterwarnings('ignore')

from model_utils import (load_model, get_layers, get_model_info, release_model,
                          safe_decode, collect_layer_outputs, get_W_U)

# ============================================================
# Sentence Generation for Composition Tests
# ============================================================

def generate_composition_sentences():
    """
    Generate sentences to test semantic composition law.
    
    Key design: For each composition test, we need:
    - Sentence with adjective+noun: "The red cat sat"
    - Sentence with just adjective context: "The red car sat" (adjective alone)
    - Sentence with just noun: "The black cat sat" (noun alone)
    - Sentence with both: "The red cat sat" (composition)
    
    We measure: z(red+cat) vs z(red_context) + z(cat_context) - z(base)
    """
    pairs = []
    
    # Color adjective compositions (8 adjectives × 5 nouns = 40 pairs)
    colors = ['red', 'blue', 'green', 'white', 'black', 'dark', 'bright', 'warm']
    nouns_a = ['cat', 'dog', 'bird', 'horse', 'fish']
    nouns_b = ['car', 'house', 'sky', 'river', 'mountain']
    
    for color in colors:
        for na in nouns_a:
            for nb in nouns_b:
                # Full: "The {color} {na} and the {nb}"
                # Adj-only: "The {color} {nb} and the {na}" - color modifies nb, na is bare
                # Noun-only: "The {na} and the {color} {nb}" - na is bare
                # Both: "The {color} {na} and the {nb}"
                
                # Simpler design: same frame, vary what modifies what
                full = f"The {color} {na} ran quickly"
                adj_context = f"The {color} {nb} ran quickly"  # color with different noun
                noun_context = f"The {na} ran quickly"          # noun without color
                base = f"The {nb} ran quickly"                  # neither
                
                pairs.append({
                    'full': full,           # color + noun
                    'adj_ctx': adj_context, # color (with other noun)
                    'noun_ctx': noun_context, # noun (without color)
                    'base': base,           # neither
                    'color': color,
                    'noun': na,
                    'type': 'color_animal'
                })
    
    # Size adjective compositions (6 × 4 = 24 pairs)
    sizes = ['big', 'small', 'tall', 'short', 'long', 'tiny']
    size_nouns = ['tree', 'building', 'rock', 'bird']
    
    for size in sizes:
        for sn in size_nouns:
            full = f"The {size} {sn} stood there"
            adj_context = f"The {size} house stood there"
            noun_context = f"The {sn} stood there"
            base = f"The house stood there"
            
            pairs.append({
                'full': full,
                'adj_ctx': adj_context,
                'noun_ctx': noun_context,
                'base': base,
                'color': size,
                'noun': sn,
                'type': 'size_object'
            })
    
    # Verb compositions: subject+verb (8 × 4 = 32 pairs)
    subjects = ['cat', 'dog', 'child', 'man', 'woman', 'bird', 'fish', 'horse']
    verbs = ['ran', 'walked', 'jumped', 'slept']
    
    for subj in subjects:
        for verb in verbs:
            full = f"The {subj} {verb} very fast"
            subj_ctx = f"The {subj} slept very quietly"  # different verb
            verb_ctx = f"The person {verb} very fast"     # different subject
            base = f"The person slept very quietly"       # neither
            
            pairs.append({
                'full': full,
                'adj_ctx': subj_ctx,      # subject-only context
                'noun_ctx': verb_ctx,      # verb-only context
                'base': base,
                'color': subj,
                'noun': verb,
                'type': 'subject_verb'
            })
    
    return pairs


def generate_syntax_transform_pairs():
    """
    Generate sentence pairs for syntax transformation test.
    N=25 per type (much larger than Phase 47's N=8).
    """
    pairs = {}
    
    # Active ↔ Passive (25 pairs)
    pairs['active_passive'] = [
        ("The cat chased the mouse", "The mouse was chased by the cat"),
        ("The dog bit the man", "The man was bitten by the dog"),
        ("The wind blew the door open", "The door was blown open by the wind"),
        ("The teacher praised the student", "The student was praised by the teacher"),
        ("The chef cooked the meal", "The meal was cooked by the chef"),
        ("The rain ruined the crops", "The crops were ruined by the rain"),
        ("The artist painted the mural", "The mural was painted by the artist"),
        ("The company built the bridge", "The bridge was built by the company"),
        ("The king ruled the kingdom", "The kingdom was ruled by the king"),
        ("The scientist discovered the element", "The element was discovered by the scientist"),
        ("The river flooded the valley", "The valley was flooded by the river"),
        ("The general commanded the army", "The army was commanded by the general"),
        ("The fire destroyed the forest", "The forest was destroyed by the fire"),
        ("The writer published the novel", "The novel was published by the writer"),
        ("The doctor cured the patient", "The patient was cured by the doctor"),
        ("The police arrested the criminal", "The criminal was arrested by the police"),
        ("The sun warmed the earth", "The earth was warmed by the sun"),
        ("The storm damaged the roof", "The roof was damaged by the storm"),
        ("The boy broke the window", "The window was broken by the boy"),
        ("The girl opened the door", "The door was opened by the girl"),
        ("The snake bit the farmer", "The farmer was bitten by the snake"),
        ("The machine crushed the rock", "The rock was crushed by the machine"),
        ("The wave swept the boat", "The boat was swept by the wave"),
        ("The guard protected the castle", "The castle was protected by the guard"),
        ("The cat caught the fish", "The fish was caught by the cat"),
    ]
    
    # Statement ↔ Question (25 pairs)
    pairs['statement_question'] = [
        ("The sky is blue today", "Is the sky blue today?"),
        ("She likes chocolate cake", "Does she like chocolate cake?"),
        ("They went to the park", "Did they go to the park?"),
        ("He can swim very well", "Can he swim very well?"),
        ("The book is on the table", "Is the book on the table?"),
        ("Water freezes at zero degrees", "Does water freeze at zero degrees?"),
        ("Birds fly south in winter", "Do birds fly south in winter?"),
        ("The project succeeded", "Did the project succeed?"),
        ("The system works correctly", "Does the system work correctly?"),
        ("She speaks three languages", "Does she speak three languages?"),
        ("The train arrives at noon", "Does the train arrive at noon?"),
        ("They live in the city", "Do they live in the city?"),
        ("The movie starts at eight", "Does the movie start at eight?"),
        ("He plays guitar every day", "Does he play guitar every day?"),
        ("The store opens on Monday", "Does the store open on Monday?"),
        ("She writes poetry often", "Does she write poetry often?"),
        ("The river flows north", "Does the river flow north?"),
        ("They eat lunch together", "Do they eat lunch together?"),
        ("The car needs repairs", "Does the car need repairs?"),
        ("He reads books constantly", "Does he read books constantly?"),
        ("The tree grows very tall", "Does the tree grow very tall?"),
        ("She runs every morning", "Does she run every morning?"),
        ("The sun rises in the east", "Does the sun rise in the east?"),
        ("They study hard every night", "Do they study hard every night?"),
        ("The bell rings at nine", "Does the bell ring at nine?"),
    ]
    
    # Singular ↔ Plural (25 pairs)
    pairs['singular_plural'] = [
        ("The cat sat on the mat", "The cats sat on the mat"),
        ("A dog ran in the park", "Some dogs ran in the park"),
        ("This flower is very beautiful", "These flowers are very beautiful"),
        ("The child played with toys", "The children played with toys"),
        ("A bird sang in the tree", "Many birds sang in the tree"),
        ("The book was on the shelf", "The books were on the shelf"),
        ("That house looks very old", "Those houses look very old"),
        ("One student answered correctly", "Several students answered correctly"),
        ("The tree stood in the garden", "The trees stood in the garden"),
        ("A star shone in the sky", "Many stars shone in the sky"),
        ("The rock blocked the path", "The rocks blocked the path"),
        ("A cloud drifted overhead", "Some clouds drifted overhead"),
        ("The river flowed gently", "The rivers flowed gently"),
        ("A mountain rose in the distance", "The mountains rose in the distance"),
        ("The ship sailed across the sea", "The ships sailed across the sea"),
        ("A letter arrived this morning", "Several letters arrived this morning"),
        ("The lamp lit the room", "The lamps lit the room"),
        ("A knife cut the bread", "The knives cut the bread"),
        ("The sheep grazed on the hill", "The sheep grazed on the hill"),
        ("A wolf howled at the moon", "The wolves howled at the moon"),
        ("The goose swam in the pond", "The geese swam in the pond"),
        ("A foot stepped on the grass", "Several feet stepped on the grass"),
        ("The leaf fell from the tree", "The leaves fell from the tree"),
        ("A mouse ran under the table", "The mice ran under the table"),
        ("The person waited patiently", "The people waited patiently"),
    ]
    
    # Tense shift (25 pairs)
    pairs['tense_shift'] = [
        ("The cat sits on the mat", "The cat sat on the mat"),
        ("She walks to school daily", "She walked to school daily"),
        ("He runs very fast today", "He ran very fast yesterday"),
        ("They play in the garden", "They played in the garden"),
        ("The bird flies over trees", "The bird flew over trees"),
        ("Water flows down the hill", "Water flowed down the hill"),
        ("The sun shines very bright", "The sun shone very bright"),
        ("The wind blows from north", "The wind blew from north"),
        ("The dog barks at strangers", "The dog barked at strangers"),
        ("She writes in her journal", "She wrote in her journal"),
        ("He drives to work daily", "He drove to work daily"),
        ("They build houses together", "They built houses together"),
        ("The baby cries very loudly", "The baby cried very loudly"),
        ("Rain falls from the clouds", "Rain fell from the clouds"),
        ("The teacher explains the lesson", "The teacher explained the lesson"),
        ("She reads many books", "She read many books"),
        ("He swims in the ocean", "He swam in the ocean"),
        ("The clock strikes midnight", "The clock struck midnight"),
        ("They sing beautiful songs", "They sang beautiful songs"),
        ("The river carries sediment", "The river carried sediment"),
        ("She speaks with confidence", "She spoke with confidence"),
        ("He draws detailed pictures", "He drew detailed pictures"),
        ("The children laugh happily", "The children laughed happily"),
        ("The train passes through town", "The train passed through town"),
        ("She grows vegetables yearly", "She grew vegetables yearly"),
    ]
    
    return pairs


# ============================================================
# Helper: Collect hidden states using hooks
# ============================================================

def collect_hidden_for_sentences(model, tokenizer, sentences, n_layers, device):
    """Collect hidden states at ALL layers for all sentences, return {layer: [N, d]}"""
    all_hidden = {i: [] for i in range(n_layers)}
    layers = get_layers(model)
    
    for si, sent in enumerate(sentences):
        captured = {}
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[layer_idx] = output[0].detach().float().cpu()
                else:
                    captured[layer_idx] = output.detach().float().cpu()
            return hook
        
        hooks = []
        for li in range(n_layers):
            hooks.append(layers[li].register_forward_hook(make_hook(li)))
        
        tokens = tokenizer(sent, return_tensors='pt', padding=True, truncation=True, max_length=60)
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        
        with torch.no_grad():
            try:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception as e:
                print(f"  Forward failed: {e}")
        
        for h in hooks:
            h.remove()
        
        last_pos = attention_mask.sum(dim=1).item() - 1
        for li in range(n_layers):
            if li in captured:
                h_last = captured[li][0, last_pos, :].numpy()
                all_hidden[li].append(h_last)
            else:
                all_hidden[li].append(np.zeros(all_hidden[0][-1].shape[0]) if all_hidden[0] else np.zeros(4096))
    
    for li in range(n_layers):
        all_hidden[li] = np.array(all_hidden[li])
    
    return all_hidden


# ============================================================
# Experiment 48A: Semantic Composition Law
# ============================================================

def exp_48a(model, tokenizer, model_name, device):
    """
    Test: z(A ∘ B) ≈ z(A) + z(B) - z(base)?
    
    Method:
    1. Collect h for: full (adj+noun), adj_only, noun_only, base
    2. Compute z = U^T (h - h_base_mean) for each
    3. Test: z(full) ≈ z(adj) + z(noun) - z(base)
    4. Measure: ||z(full) - z(adj) - z(noun) + z(base)|| / ||z(full)||
    """
    print(f"\n{'='*70}")
    print(f"48A: Semantic Composition Law — {model_name}")
    print(f"{'='*70}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    # Generate composition test data
    comp_pairs = generate_composition_sentences()
    
    # Subsample to avoid too many (use 40 color + 24 size + 32 verb = 96, cap at 80)
    np.random.seed(42)
    if len(comp_pairs) > 80:
        indices = np.random.choice(len(comp_pairs), 80, replace=False)
        comp_pairs = [comp_pairs[i] for i in indices]
    
    print(f"  Composition pairs: {len(comp_pairs)}")
    print(f"  Types: color_animal, size_object, subject_verb")
    
    # Collect all unique sentences
    all_sentences = []
    sent_to_idx = {}
    for p in comp_pairs:
        for key in ['full', 'adj_ctx', 'noun_ctx', 'base']:
            s = p[key]
            if s not in sent_to_idx:
                sent_to_idx[s] = len(all_sentences)
                all_sentences.append(s)
    
    print(f"  Unique sentences: {len(all_sentences)}")
    
    # Collect hidden states
    print(f"  Collecting hidden states...")
    t0 = time.time()
    all_hidden = collect_hidden_for_sentences(model, tokenizer, all_sentences, n_layers, device)
    print(f"  Done in {time.time()-t0:.1f}s")
    
    # Use middle layer for analysis (most informative per Phase 47)
    # But test across layers to see composition at different depths
    test_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    test_layers = sorted(set([max(1, l) for l in test_layers]))
    
    print(f"\n  Test layers: {test_layers}")
    
    # For each test layer, do the analysis
    results_by_layer = {}
    
    for layer in test_layers:
        H = all_hidden[layer]
        
        # Compute h_base = mean of all base sentences
        base_indices = [sent_to_idx[p['base']] for p in comp_pairs]
        h_base = np.mean(H[base_indices], axis=0)
        
        # h_sem = h - h_base
        H_sem = H - h_base
        
        # PCA on h_sem to get U
        n_pca = min(60, H_sem.shape[0] - 1, H_sem.shape[1])
        pca = PCA(n_components=n_pca)
        pca.fit(H_sem)
        U = pca.components_.T  # [d, n_pca]
        
        # Choose k = dim90
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        k = min(int(np.searchsorted(cumvar, 0.90) + 1), n_pca)
        dim50 = int(np.searchsorted(cumvar, 0.50) + 1)
        
        # Project to z-space
        Z = H_sem @ U[:, :k]  # [N, k]
        
        # Test composition law for each pair
        composition_errors = []
        additive_approx_errors = []
        relative_errors = []
        
        by_type = {'color_animal': [], 'size_object': [], 'subject_verb': []}
        
        for p in comp_pairs:
            z_full = Z[sent_to_idx[p['full']]]
            z_adj = Z[sent_to_idx[p['adj_ctx']]]
            z_noun = Z[sent_to_idx[p['noun_ctx']]]
            z_base = Z[sent_to_idx[p['base']]]
            
            # Composition prediction: z(full) ≈ z(adj) + z(noun) - z(base)
            z_composed = z_adj + z_noun - z_base
            
            # Error
            err = np.linalg.norm(z_full - z_composed)
            norm_full = np.linalg.norm(z_full)
            
            composition_errors.append(err)
            if norm_full > 1e-10:
                relative_errors.append(err / norm_full)
            else:
                relative_errors.append(np.nan)
            
            by_type[p['type']].append(err / max(norm_full, 1e-10))
            
            # Also test: is z(full) closer to z(adj) or z(noun)?
            dist_adj = np.linalg.norm(z_full - z_adj)
            dist_noun = np.linalg.norm(z_full - z_noun)
            dist_composed = err
            additive_approx_errors.append({
                'dist_adj': dist_adj,
                'dist_noun': dist_noun,
                'dist_composed': dist_composed,
                'norm_full': norm_full,
            })
        
        # Summary statistics
        rel_err = np.array([e for e in relative_errors if not np.isnan(e)])
        mean_rel_err = np.mean(rel_err) if len(rel_err) > 0 else float('nan')
        median_rel_err = np.median(rel_err) if len(rel_err) > 0 else float('nan')
        
        # Cosine similarity between z_full and z_composed
        cos_vals = []
        for p in comp_pairs:
            z_full = Z[sent_to_idx[p['full']]]
            z_composed = Z[sent_to_idx[p['adj_ctx']]] + Z[sent_to_idx[p['noun_ctx']]] - Z[sent_to_idx[p['base']]]
            n1 = np.linalg.norm(z_full)
            n2 = np.linalg.norm(z_composed)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_vals.append(np.dot(z_full, z_composed) / (n1 * n2))
        
        # Test: is composition error smaller than just using adj or noun alone?
        adj_only_errors = [e['dist_adj'] / max(e['norm_full'], 1e-10) for e in additive_approx_errors]
        noun_only_errors = [e['dist_noun'] / max(e['norm_full'], 1e-10) for e in additive_approx_errors]
        
        results_by_layer[layer] = {
            'k': k,
            'dim50': dim50,
            'mean_rel_err': mean_rel_err,
            'median_rel_err': median_rel_err,
            'mean_cos': np.mean(cos_vals) if cos_vals else 0,
            'adj_only_rel_err': np.mean(adj_only_errors),
            'noun_only_rel_err': np.mean(noun_only_errors),
            'by_type': {t: np.mean(v) for t, v in by_type.items() if v},
        }
    
    # Print results
    print(f"\n  {'='*60}")
    print(f"  COMPOSITION LAW RESULTS")
    print(f"  {'='*60}")
    print(f"  z(A∘B) ≈ z(A) + z(B) - z(base)?")
    print(f"  {'Layer':>6} {'k':>4} {'dim50':>6} {'rel_err':>10} {'cos(z,z\')':>10} {'adj_only':>10} {'noun_only':>10}")
    print(f"  {'-'*60}")
    
    for layer in test_layers:
        r = results_by_layer[layer]
        print(f"  {layer:>6} {r['k']:>4} {r['dim50']:>6} {r['mean_rel_err']:>10.4f} {r['mean_cos']:>10.4f} {r['adj_only_rel_err']:>10.4f} {r['noun_only_rel_err']:>10.4f}")
    
    # By type breakdown
    print(f"\n  By composition type:")
    print(f"  {'Layer':>6} {'color_animal':>15} {'size_object':>15} {'subject_verb':>15}")
    print(f"  {'-'*55}")
    for layer in test_layers:
        r = results_by_layer[layer]
        ca = r['by_type'].get('color_animal', 0)
        so = r['by_type'].get('size_object', 0)
        sv = r['by_type'].get('subject_verb', 0)
        print(f"  {layer:>6} {ca:>15.4f} {so:>15.4f} {sv:>15.4f}")
    
    # Interpretation
    print(f"\n  {'='*60}")
    print(f"  INTERPRETATION")
    print(f"  {'='*60}")
    
    mid_layer = test_layers[len(test_layers)//2]
    r = results_by_layer[mid_layer]
    
    if r['mean_rel_err'] < 0.3:
        print(f"  ✅ Composition error < 0.3 → ADDITIVE STRUCTURE EXISTS")
    elif r['mean_rel_err'] < 0.6:
        print(f"  ⚠️ Composition error 0.3-0.6 → PARTIAL additivity")
    else:
        print(f"  ❌ Composition error > 0.6 → NOT additive")
    
    if r['mean_cos'] > 0.8:
        print(f"  ✅ Directional consistency (cos > 0.8) → Composition preserves direction")
    elif r['mean_cos'] > 0.5:
        print(f"  ⚠️ Moderate directional consistency (cos 0.5-0.8)")
    else:
        print(f"  ❌ Low directional consistency (cos < 0.5)")
    
    # Key comparison: is composition better than either component alone?
    if r['mean_rel_err'] < r['adj_only_rel_err'] and r['mean_rel_err'] < r['noun_only_rel_err']:
        print(f"  ✅ Composition BETTER than either component alone!")
    elif r['mean_rel_err'] < max(r['adj_only_rel_err'], r['noun_only_rel_err']):
        print(f"  ⚠️ Composition better than one component but not the other")
    else:
        print(f"  ❌ Composition NOT better than components alone")
    
    return results_by_layer


# ============================================================
# Experiment 48B: Syntax = Linear Transformation
# ============================================================

def exp_48b(model, tokenizer, model_name, device):
    """
    Test: z_passive ≈ T · z_active?
    
    Method:
    1. For each syntax pair type, collect z for form A and form B
    2. Fit T = argmin ||Z_B - T · Z_A|| via least squares
    3. Test on held-out pairs
    4. Check: is T consistent across sentences?
    """
    print(f"\n{'='*70}")
    print(f"48B: Syntax = Linear Transformation — {model_name}")
    print(f"{'='*70}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    syntax_pairs = generate_syntax_transform_pairs()
    
    test_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    test_layers = sorted(set([max(1, l) for l in test_layers]))
    
    print(f"  Syntax pair types: {list(syntax_pairs.keys())}")
    print(f"  Pairs per type: {[len(v) for v in syntax_pairs.values()]}")
    print(f"  Test layers: {test_layers}")
    
    results = {}
    
    for syn_type, pairs in syntax_pairs.items():
        print(f"\n  --- {syn_type} ---")
        
        # Collect all sentences
        sent_a = [p[0] for p in pairs]
        sent_b = [p[1] for p in pairs]
        all_sents = sent_a + sent_b
        
        # Collect hidden states
        print(f"  Collecting hidden states for {len(all_sents)} sentences...")
        all_hidden = collect_hidden_for_sentences(model, tokenizer, all_sents, n_layers, device)
        
        for layer in test_layers:
            H = all_hidden[layer]
            
            # Compute z via PCA
            h_base = np.mean(H, axis=0)
            H_sem = H - h_base
            
            n_pca = min(60, H_sem.shape[0] - 1, H_sem.shape[1])
            pca = PCA(n_components=n_pca)
            pca.fit(H_sem)
            U = pca.components_.T
            
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            k = min(int(np.searchsorted(cumvar, 0.90) + 1), n_pca)
            
            Z = H_sem @ U[:, :k]
            
            n_pairs_total = len(pairs)
            Z_A = Z[:n_pairs_total]        # form A (first half)
            Z_B = Z[n_pairs_total:]         # form B (second half)
            
            # Method 1: Global linear transformation T
            # Z_B ≈ T · Z_A, T = [k, k]
            # Fit on 80%, test on 20%
            n_train = int(0.8 * n_pairs_total)
            if n_train < k + 2:
                n_train = n_pairs_total - 2
            n_test = n_pairs_total - n_train
            
            if n_test < 1:
                n_test = 1
                n_train = n_pairs_total - 1
            
            Z_A_train = Z_A[:n_train]
            Z_B_train = Z_B[:n_train]
            Z_A_test = Z_A[n_train:]
            Z_B_test = Z_B[n_train:]
            
            # Fit T using Ridge regression: Z_B_train = T @ Z_A_train
            # Solve: T = Z_B_train @ pinv(Z_A_train)
            ridge = Ridge(alpha=0.1)
            ridge.fit(Z_A_train, Z_B_train)
            Z_B_pred = ridge.predict(Z_A_test)
            
            # R² on test set
            ss_res = np.sum((Z_B_test - Z_B_pred) ** 2)
            ss_tot = np.sum((Z_B_test - np.mean(Z_B_test, axis=0)) ** 2)
            r2_test = 1 - ss_res / max(ss_tot, 1e-20)
            
            # Cosine similarity
            cos_vals = []
            for i in range(len(Z_B_test)):
                n1 = np.linalg.norm(Z_B_test[i])
                n2 = np.linalg.norm(Z_B_pred[i])
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_vals.append(np.dot(Z_B_test[i], Z_B_pred[i]) / (n1 * n2))
            
            mean_cos_test = np.mean(cos_vals) if cos_vals else 0
            
            # Relative error
            rel_errs = []
            for i in range(len(Z_B_test)):
                n1 = np.linalg.norm(Z_B_test[i])
                if n1 > 1e-10:
                    rel_errs.append(np.linalg.norm(Z_B_test[i] - Z_B_pred[i]) / n1)
            mean_rel_err_test = np.mean(rel_errs) if rel_errs else 0
            
            # Method 2: Per-pair delta analysis (does T vary across sentences?)
            # If z_B = T @ z_A, then z_B - z_A should be T @ z_A - z_A = (T-I) @ z_A
            # Check: is delta = z_B - z_A proportional to z_A?
            deltas = Z_B - Z_A
            cos_dz_z = []
            for i in range(n_pairs_total):
                n1 = np.linalg.norm(Z_A[i])
                n2 = np.linalg.norm(deltas[i])
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_dz_z.append(np.dot(deltas[i], Z_A[i]) / (n1 * n2))
            
            mean_cos_dz_z = np.mean(cos_dz_z) if cos_dz_z else 0
            
            # Method 3: Cross-pair consistency test
            # If T is universal, then T @ z_A[0] should predict z_B[0]
            # and T @ z_A[1] should predict z_B[1], etc.
            # Test: is the transformation pair-dependent?
            # Fit T on each individual pair and compare
            if n_pairs_total >= 3:
                # Fit T on pair i, predict pair j
                pair_r2s = []
                for i in range(min(5, n_pairs_total)):
                    # Use all except pair i for training
                    train_idx = [j for j in range(n_pairs_total) if j != i][:max(1, n_pairs_total-1)]
                    test_idx = [i]
                    
                    ridge_cv = Ridge(alpha=0.1)
                    ridge_cv.fit(Z_A[train_idx], Z_B[train_idx])
                    Z_B_pred_cv = ridge_cv.predict(Z_A[test_idx])
                    
                    err = np.linalg.norm(Z_B[test_idx[0]] - Z_B_pred_cv[0])
                    norm = np.linalg.norm(Z_B[test_idx[0]])
                    pair_r2s.append(err / max(norm, 1e-10))
                
                cross_pair_err = np.mean(pair_r2s)
            else:
                cross_pair_err = float('nan')
            
            # Dimensionality of delta space
            pca_delta = PCA()
            pca_delta.fit(deltas)
            cumvar_delta = np.cumsum(pca_delta.explained_variance_ratio_)
            dim50_delta = int(np.searchsorted(cumvar_delta, 0.50) + 1)
            dim90_delta = int(np.searchsorted(cumvar_delta, 0.90) + 1)
            
            key = f"{syn_type}_L{layer}"
            results[key] = {
                'k': k,
                'r2_test': r2_test,
                'cos_test': mean_cos_test,
                'rel_err_test': mean_rel_err_test,
                'cos_dz_z': mean_cos_dz_z,
                'cross_pair_err': cross_pair_err,
                'dim50_delta': dim50_delta,
                'dim90_delta': dim90_delta,
            }
            
            print(f"  L{layer}: k={k}, R²={r2_test:.3f}, cos={mean_cos_test:.3f}, "
                  f"rel_err={mean_rel_err_test:.3f}, cos(Δz,z)={mean_cos_dz_z:.3f}, "
                  f"dim50(Δz)={dim50_delta}, dim90(Δz)={dim90_delta}, "
                  f"cross_pair_err={cross_pair_err:.3f}")
    
    # Print summary
    print(f"\n  {'='*60}")
    print(f"  SYNTAX TRANSFORMATION SUMMARY")
    print(f"  {'='*60}")
    
    # Pick best layer for each syntax type
    for syn_type in syntax_pairs.keys():
        best_r2 = -999
        best_layer = None
        for layer in test_layers:
            key = f"{syn_type}_L{layer}"
            if key in results and results[key]['r2_test'] > best_r2:
                best_r2 = results[key]['r2_test']
                best_layer = layer
        
        if best_layer is not None:
            key = f"{syn_type}_L{best_layer}"
            r = results[key]
            print(f"\n  {syn_type} (best layer {best_layer}):")
            print(f"    R² = {r['r2_test']:.3f}, cos = {r['cos_test']:.3f}")
            print(f"    dim90(Δz) = {r['dim90_delta']}, cross_pair_err = {r['cross_pair_err']:.3f}")
            
            if r['r2_test'] > 0.7:
                print(f"    ✅ STRONG linear transformation!")
            elif r['r2_test'] > 0.4:
                print(f"    ⚠️ Moderate linear transformation")
            else:
                print(f"    ❌ Weak linear transformation")
            
            if r['cross_pair_err'] < 0.3:
                print(f"    ✅ Cross-pair consistent → UNIVERSAL T")
            elif r['cross_pair_err'] < 0.6:
                print(f"    ⚠️ Partially universal T")
            else:
                print(f"    ❌ T is pair-dependent")
    
    return results


# ============================================================
# Experiment 48C: Δz Dynamical Mode Decomposition
# ============================================================

def exp_48c(model, tokenizer, model_name, device):
    """
    Find finite basis for Δz: Δz = Σ c_i · basis_i
    
    Key question: Is there a finite set of "language operations"?
    
    Method:
    1. Collect Δz across all layers and sentences
    2. SVD decomposition to find basis
    3. Test: how many basis vectors needed to reconstruct Δz?
    4. Are the basis vectors shared across layers?
    """
    print(f"\n{'='*70}")
    print(f"48C: Δz Dynamical Mode Decomposition — {model_name}")
    print(f"{'='*70}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    # Use diverse sentences from Phase 47's categories
    from ccml_phase47_semantic_dynamics import generate_diverse_sentences
    sentences, labels = generate_diverse_sentences(n_per_category=30)
    
    # Also add syntax pairs for broader coverage
    syntax_pairs = generate_syntax_transform_pairs()
    for syn_type, pairs in syntax_pairs.items():
        for a, b in pairs[:10]:
            sentences.extend([a, b])
            labels.extend([syn_type + '_a', syn_type + '_b'])
    
    print(f"  Total sentences: {len(sentences)}")
    
    # Collect hidden states
    print(f"  Collecting hidden states...")
    t0 = time.time()
    all_hidden = collect_hidden_for_sentences(model, tokenizer, sentences, n_layers, device)
    print(f"  Done in {time.time()-t0:.1f}s")
    
    # Compute z-space
    # Use layer n_layers//2 as reference for PCA
    ref_layer = n_layers // 2
    H_ref = all_hidden[ref_layer]
    h_base = np.mean(H_ref, axis=0)
    H_sem = H_ref - h_base
    
    n_pca = min(60, H_sem.shape[0] - 1, H_sem.shape[1])
    pca = PCA(n_components=n_pca)
    pca.fit(H_sem)
    U = pca.components_.T
    
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = min(int(np.searchsorted(cumvar, 0.90) + 1), n_pca)
    print(f"  k = {k} (dim90 at layer {ref_layer})")
    
    # Project all layers to z-space
    Z_by_layer = {}
    for layer in range(n_layers):
        H_l = all_hidden[layer]
        Z_by_layer[layer] = (H_l - h_base) @ U[:, :k]
    
    # Compute Δz for all consecutive layer pairs
    dz_by_layer = {}
    for layer in range(1, n_layers):
        dz_by_layer[layer] = Z_by_layer[layer] - Z_by_layer[layer - 1]
    
    # === Analysis 1: Per-layer basis dimensionality ===
    print(f"\n  --- Per-layer Δz basis ---")
    print(f"  {'Layer':>6} {'dim50':>6} {'dim80':>6} {'dim90':>6} {'dim95':>6} {'top1_var':>10}")
    
    layer_dims = {}
    for layer in sorted(dz_by_layer.keys()):
        dz = dz_by_layer[layer]
        pca_dz = PCA()
        pca_dz.fit(dz)
        cumvar_dz = np.cumsum(pca_dz.explained_variance_ratio_)
        
        dim50 = int(np.searchsorted(cumvar_dz, 0.50) + 1)
        dim80 = int(np.searchsorted(cumvar_dz, 0.80) + 1)
        dim90 = int(np.searchsorted(cumvar_dz, 0.90) + 1)
        dim95 = int(np.searchsorted(cumvar_dz, 0.95) + 1)
        top1 = pca_dz.explained_variance_ratio_[0]
        
        layer_dims[layer] = {'dim50': dim50, 'dim80': dim80, 'dim90': dim90, 'dim95': dim95}
        
        if layer % max(1, n_layers // 8) == 0 or layer == n_layers - 1:
            print(f"  {layer:>6} {dim50:>6} {dim80:>6} {dim90:>6} {dim95:>6} {top1:>10.4f}")
    
    # === Analysis 2: Global basis across all layers ===
    print(f"\n  --- Global Δz basis (all layers pooled) ---")
    
    # Pool all Δz across layers
    all_dz = []
    dz_layer_labels = []
    for layer in sorted(dz_by_layer.keys()):
        all_dz.append(dz_by_layer[layer])
        dz_layer_labels.extend([layer] * len(dz_by_layer[layer]))
    
    all_dz = np.vstack(all_dz)
    dz_layer_labels = np.array(dz_layer_labels)
    
    print(f"  Total Δz vectors: {all_dz.shape[0]}")
    
    pca_global = PCA()
    pca_global.fit(all_dz)
    cumvar_global = np.cumsum(pca_global.explained_variance_ratio_)
    
    dim50_g = int(np.searchsorted(cumvar_global, 0.50) + 1)
    dim80_g = int(np.searchsorted(cumvar_global, 0.80) + 1)
    dim90_g = int(np.searchsorted(cumvar_global, 0.90) + 1)
    dim95_g = int(np.searchsorted(cumvar_global, 0.95) + 1)
    
    print(f"  Global basis: dim50={dim50_g}, dim80={dim80_g}, dim90={dim90_g}, dim95={dim95_g}")
    print(f"  Top 10 variance: {pca_global.explained_variance_ratio_[:10].round(4)}")
    
    # === Analysis 3: Per-layer reconstruction with global basis ===
    print(f"\n  --- Per-layer reconstruction with global basis ---")
    
    basis_global = pca_global.components_  # [n_comp, k]
    
    for n_basis in [5, 10, 20, min(dim90_g, basis_global.shape[0])]:
        r2_list = []
        for layer in sorted(dz_by_layer.keys()):
            dz = dz_by_layer[layer]
            B = basis_global[:n_basis].T  # [k, n_basis]
            coeffs = dz @ B  # [N, n_basis]
            dz_recon = coeffs @ B.T  # [N, k]
            
            ss_res = np.sum((dz - dz_recon) ** 2)
            ss_tot = np.sum(dz ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-20)
            r2_list.append(r2)
        
        mean_r2 = np.mean(r2_list)
        print(f"  basis={n_basis:>3}: mean R² = {mean_r2:.3f}")
    
    # === Analysis 4: Are basis modes shared across layers? ===
    print(f"\n  --- Basis sharing across layers ---")
    
    # For each pair of nearby layers, compute subspace overlap
    sample_layers = list(range(2, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    print(f"  {'Layer':>6} {'overlap(L,L+1)':>15} {'overlap(L,L+2)':>15}")
    
    for li in sample_layers:
        if li not in dz_by_layer or (li+1) not in dz_by_layer:
            continue
        
        # Get top-10 PCA subspace for each layer
        pca_li = PCA(n_components=min(10, dz_by_layer[li].shape[0]-1, dz_by_layer[li].shape[1]))
        pca_li.fit(dz_by_layer[li])
        B_li = pca_li.components_  # [10, k]
        
        overlap_1 = 0
        overlap_2 = 0
        
        # Overlap with next layer
        if (li+1) in dz_by_layer:
            pca_li1 = PCA(n_components=min(10, dz_by_layer[li+1].shape[0]-1, dz_by_layer[li+1].shape[1]))
            pca_li1.fit(dz_by_layer[li+1])
            B_li1 = pca_li1.components_
            
            # Subspace overlap = mean |cos(v_i, nearest w_j)|
            cos_matrix = np.abs(B_li @ B_li1.T)  # [10, 10]
            overlap_1 = np.mean(np.max(cos_matrix, axis=1))
        
        # Overlap with layer+2
        if (li+2) in dz_by_layer:
            pca_li2 = PCA(n_components=min(10, dz_by_layer[li+2].shape[0]-1, dz_by_layer[li+2].shape[1]))
            pca_li2.fit(dz_by_layer[li+2])
            B_li2 = pca_li2.components_
            
            cos_matrix2 = np.abs(B_li @ B_li2.T)
            overlap_2 = np.mean(np.max(cos_matrix2, axis=1))
        
        print(f"  {li:>6} {overlap_1:>15.4f} {overlap_2:>15.4f}")
    
    # === Analysis 5: Interpreting top basis modes ===
    print(f"\n  --- Top basis mode interpretation ---")
    
    # Correlate top basis modes with semantic categories
    n_top_modes = min(10, basis_global.shape[0])
    
    # Get z-coefficients on global basis for each sentence
    z_mid = Z_by_layer[ref_layer]
    dz_mid = dz_by_layer[ref_layer]
    
    # Project dz onto top basis modes
    dz_coeffs = dz_mid @ basis_global[:n_top_modes].T  # [N, n_top_modes]
    
    # Check which modes correlate with semantic categories
    unique_labels = list(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    label_indices = np.array([label_to_idx.get(l, 0) for l in labels])
    
    print(f"  Top {n_top_modes} modes vs semantic categories (ANOVA F):")
    
    from scipy.stats import f_oneway
    
    mode_anova = []
    for mi in range(n_top_modes):
        groups = [dz_coeffs[label_indices == li, mi] for li in range(len(unique_labels)) if np.sum(label_indices == li) > 2]
        if len(groups) >= 2:
            try:
                F, p = f_oneway(*groups)
                mode_anova.append((mi, F, p))
            except:
                mode_anova.append((mi, 0, 1))
        else:
            mode_anova.append((mi, 0, 1))
    
    mode_anova.sort(key=lambda x: -x[1])
    for mi, F, p in mode_anova[:5]:
        print(f"    Mode {mi}: F={F:.2f}, p={p:.4f}" + (" ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""))
    
    # === Analysis 6: Minimal generative model test ===
    print(f"\n  --- Minimal generative model test ---")
    
    # Can we predict Δz from z using a small set of basis operations?
    # Model: Δz = Σ α_i(z) · basis_i
    # Simplest: α_i = linear function of z
    
    # Use global basis top dim90_g modes (cap at 30 for efficiency)
    n_basis_use = min(dim90_g, 30, basis_global.shape[0])
    B_use = basis_global[:n_basis_use]  # [n_basis_use, k]
    
    print(f"  Using {n_basis_use} basis modes (from {basis_global.shape[0]} total), k={k}", flush=True)
    print(f"  B_use shape: {B_use.shape}", flush=True)
    
    # Check dimensions of first layer
    first_layer = sorted(dz_by_layer.keys())[0]
    print(f"  dz_by_layer[{first_layer}].shape = {dz_by_layer[first_layer].shape}", flush=True)
    print(f"  Z_by_layer[{first_layer-1}].shape = {Z_by_layer.get(first_layer-1, np.array([])).shape}", flush=True)
    
    # For each layer, fit: coeffs = W @ z + b
    print(f"  {'Layer':>6} {'R²(coeff)':>12} {'R²(dz)':>10}")
    
    for layer in sorted(dz_by_layer.keys()):
        if layer not in Z_by_layer or (layer-1) not in Z_by_layer:
            continue
        
        dz = dz_by_layer[layer]        # [N, k]
        z_prev = Z_by_layer[layer - 1]  # [N, k]
        
        # Project dz onto basis: coefficients [N, n_basis_use]
        dz_c = dz @ B_use.T  # [N, n_basis_use]
        if dz_c.ndim == 1:
            dz_c = dz_c.reshape(-1, 1)
        
        # Fit linear model: dz_c ≈ W @ z_prev + b
        ridge = Ridge(alpha=1.0)
        ridge.fit(z_prev, dz_c)
        dz_c_pred = ridge.predict(z_prev)  # [N, n_basis_use]
        if dz_c_pred.ndim == 1:
            dz_c_pred = dz_c_pred.reshape(-1, 1)
        
        # Reconstruct dz from predicted coefficients
        dz_recon = dz_c_pred @ B_use  # [N, n_basis_use] @ [n_basis_use, k] = [N, k]
        
        # R² for coefficients
        ss_res_c = np.sum((dz_c - dz_c_pred) ** 2)
        ss_tot_c = np.sum((dz_c - np.mean(dz_c, axis=0)) ** 2)
        r2_coeff = 1 - ss_res_c / max(ss_tot_c, 1e-20)
        
        # R² for reconstructed dz
        ss_res_dz = np.sum((dz - dz_recon) ** 2)
        ss_tot_dz = np.sum(dz ** 2)
        r2_dz = 1 - ss_res_dz / max(ss_tot_dz, 1e-20)
        
        if layer % max(1, n_layers // 8) == 0 or layer == n_layers - 1:
            print(f"  {layer:>6} {r2_coeff:>12.3f} {r2_dz:>10.3f}")
    
    print(f"\n  {'='*60}")
    print(f"  DYNAMICAL MODE DECOMPOSITION SUMMARY")
    print(f"  {'='*60}")
    print(f"  Global Δz basis: dim50={dim50_g}, dim90={dim90_g}")
    print(f"  Per-layer Δz basis: dim50≈{np.mean([v['dim50'] for v in layer_dims.values()]):.1f}, "
          f"dim90≈{np.mean([v['dim90'] for v in layer_dims.values()]):.1f}")
    
    if dim90_g <= 20:
        print(f"  ✅ FEW modes needed → Language has finite operation set!")
    elif dim90_g <= 40:
        print(f"  ⚠️ Moderate number of modes")
    else:
        print(f"  ❌ Many modes needed → No simple basis")
    
    return {
        'global_dims': {'dim50': dim50_g, 'dim80': dim80_g, 'dim90': dim90_g, 'dim95': dim95_g},
        'layer_dims': layer_dims,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='deepseek7b',
                        choices=['qwen3', 'glm4', 'deepseek7b'])
    parser.add_argument('--exp', type=int, default=0,
                        help='0=all, 1=48A(composition), 2=48B(syntax), 3=48C(modes)')
    args = parser.parse_args()
    
    model_name = args.model
    
    print(f"\n{'#'*70}")
    print(f"# Phase 48: Language Ontology Verification — {model_name}")
    print(f"{'#'*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    print(f"  Model: {info.name}, Layers: {info.n_layers}, d_model: {info.d_model}")
    
    try:
        if args.exp in [0, 1]:
            exp_48a(model, tokenizer, model_name, device)
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if args.exp in [0, 2]:
            exp_48b(model, tokenizer, model_name, device)
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if args.exp in [0, 3]:
            exp_48c(model, tokenizer, model_name, device)
    finally:
        release_model(model)
    
    print(f"\n{'#'*70}")
    print(f"# Phase 48 Complete")
    print(f"{'#'*70}")


if __name__ == '__main__':
    main()
