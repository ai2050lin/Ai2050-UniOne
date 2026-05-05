"""
Phase 47: Semantic Coordinate Dynamics — 语义坐标空间动力学
============================================================

用户核心洞察: 我们一直在高维h空间做统计, 但语言的本质在z空间!

理论框架:
    h_l(x) = h_base + U · z_l(x) + ε_l
    
    z_l ∈ R^k  (语义坐标, k≈30-50)
    U ∈ R^{d×k} (语义基)
    
    层动力学:
    z_{l+1} = z_l + f_l(z_l)
    
    目标: 找到 f_l 的结构!

基于Phase 46的修正:
    - Δh dim90 ≈ 0.7N (样本假象, 不再依赖)
    - 混合动力学 (前旋转后膨胀, 不是纯切向)
    - dim50 ≈ 30-50 (有效维度, 更可靠)

实验设计:
    47A: 语义坐标恢复 — 提取U和z_l
    47B: z空间动力学 — 拟合z_{l+1} = z_l + f_l(z_l)
    47C: Attention vs MLP分解 — 在z空间中
    47D: 语法→动力模式 — 不同句法的z演化差异
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

from model_utils import (load_model, get_layers, get_model_info, release_model,
                          safe_decode, collect_layer_outputs, get_W_U)

# ============================================================
# Sentence Generation
# ============================================================

def generate_diverse_sentences(n_per_category=40):
    """Generate diverse sentences across semantic categories"""
    templates = {
        'animals': [
            "The cat sat on the mat quietly", "A dog ran through the park today",
            "The bird flew over the tall tree", "Fish swim in the deep ocean",
            "The horse galloped across the field", "A rabbit hopped through the garden",
            "The tiger hunted in the jungle", "Eagles soar above the mountain",
            "The whale dove deep underwater", "A snake slithered through grass",
            "The elephant walked slowly forward", "Butterflies dance among flowers",
            "The lion roared at the sunset", "A fox crept through the forest",
            "The dolphin jumped over waves", "Bears sleep during the winter",
            "The deer grazed in the meadow", "Owls hunt in the dark night",
            "The penguin slid on the ice", "A wolf howled at the moon",
            "The monkey climbed the tall tree", "Turtles move very slowly today",
            "The parrot spoke some words", "A frog jumped into water",
            "The squirrel gathered some nuts", "Geese fly south for winter",
            "The cheetah ran very fast", "A seal swam near shore",
            "The giraffe ate from trees", "Bees buzz around the hive",
            "The zebra crossed the river", "A crab walked on sand",
            "The swan glided on water", "An ant carried heavy food",
            "The hawk circled overhead", "A deer stood very still",
            "The otter played with rocks", "A moth flew toward light",
            "The ram climbed steep rocks", "A toad sat by pond",
        ],
        'science': [
            "The experiment confirmed the hypothesis", "Electrons orbit around the nucleus",
            "The reaction produced new compounds", "Gravity pulls objects downward",
            "Light travels at high speed", "The formula solved the problem",
            "Data supports the conclusion", "Molecules bond through sharing",
            "Energy converts between forms", "The telescope observed distant stars",
            "Chemicals react when heated", "The microscope revealed tiny cells",
            "Forces act in pairs always", "Waves propagate through the medium",
            "The theory explained observations", "Atoms contain protons inside",
            "Voltage drives the current", "The satellite orbited Earth",
            "Elements combine in reactions", "Friction slows moving objects",
            "The algorithm processed data", "Neutrons exist in nuclei",
            "The laser focused the beam", "Osmosis moves water across",
            "The particle accelerated fast", "Catalysts speed up reactions",
            "The gene expressed the trait", "Neurons transmit electric signals",
            "The crystal formed slowly", "Plasma is very hot",
            "The equation balanced perfectly", "Resonance amplifies the wave",
            "The quantum state collapsed", "Entropy increases over time",
            "The spectrum showed absorption", "Diffusion spreads particles out",
            "The variable controlled results", "Mitosis divides the cell",
            "The circuit completed the loop", "Mass bends spacetime around",
        ],
        'emotion': [
            "She felt very happy today", "The news made him sad",
            "They were angry at delay", "Joy filled her whole heart",
            "Fear gripped the small child", "He felt proud of work",
            "The loss caused deep grief", "Love warmed their cold hearts",
            "Anxiety kept her awake", "Hope sustained them through hardship",
            "He felt disgusted by it", "Surprise shocked everyone present",
            "The guilt weighed heavily down", "She felt relieved after",
            "Jealousy consumed his thoughts", "The disappointment was crushing",
            "Gratitude filled her heart", "Shame made him hide away",
            "She felt peaceful and calm", "Loneliness haunted his nights",
            "The excitement was overwhelming", "He felt confused and lost",
            "She experienced deep contentment", "The frustration built up",
            "He felt nostalgic about past", "She was touched by kindness",
            "The sorrow was unbearable", "He felt betrayed by friends",
            "She felt empowered and strong", "The worry never stopped",
            "He experienced pure bliss", "She felt vulnerable and exposed",
            "The bitterness lasted long", "He felt accepted finally",
            "She felt inspired by art", "The despair was total",
            "He felt curious about life", "She experienced real awe",
            "The resentment grew stronger", "He felt truly alive",
        ],
        'actions': [
            "She opened the door slowly", "He kicked the ball hard",
            "They built a house together", "She wrote a long letter",
            "He drove the car fast", "They cooked the meal perfectly",
            "She painted the wall blue", "He fixed the broken machine",
            "They planted seeds in spring", "She sang a beautiful song",
            "He cut the wood precisely", "They cleaned the entire house",
            "She read the whole book", "He threw the ball far",
            "They designed a new bridge", "She played the piano well",
            "He carried the heavy box", "They repaired the old roof",
            "She drew a nice picture", "He pushed the cart forward",
            "They washed the dirty clothes", "She typed the report quickly",
            "He caught the fast ball", "They dug a deep hole",
            "She sewed the torn dress", "He mixed the ingredients together",
            "They painted the fence white", "She baked a chocolate cake",
            "He lifted the weights easily", "They assembled the new desk",
            "She folded the clean laundry", "He poured the hot coffee",
            "They wrapped the small gift", "She tied the knot tight",
            "He ground the fresh pepper", "They arranged the flowers beautifully",
            "She polished the old silver", "He sliced the ripe fruit",
            "They folded the paper carefully", "She tuned the old guitar",
        ],
        'questions': [
            "What is the meaning here?", "How does this machine work?",
            "Where did they go yesterday?", "When will the rain stop?",
            "Why did she leave early?", "Who wrote this famous book?",
            "Which option is better now?", "Can you help me please?",
            "Is this the right way?", "Do you understand the problem?",
            "What time does it start?", "How much does it cost?",
            "Where can I find help?", "When did this happen first?",
            "Why is the sky blue?", "Who knows the correct answer?",
            "Which path should we take?", "Could this be a mistake?",
            "Are they coming today soon?", "Does this make any sense?",
            "What caused the big explosion?", "How long will it take?",
            "Where was the treasure hidden?", "When was this built originally?",
            "Why do birds migrate south?", "Who discovered this old element?",
            "Which method works best here?", "Will this ever be finished?",
            "Have you seen this before?", "Should we try again now?",
            "What determines the final price?", "How far is the station?",
            "Where does this road lead?", "When can we start working?",
            "Why did the project fail?", "Who designed this strange building?",
            "Which theory explains the data?", "May I ask you something?",
            "Must we follow these rules?", "Would you like some tea?",
        ],
        'descriptions': [
            "The sky was clear and blue", "A tall building stood nearby",
            "The water flowed very gently", "Red flowers covered the hill",
            "The old castle looked grand", "Soft music played in background",
            "The path wound through forest", "Bright stars filled the night",
            "The room was quite spacious", "Green leaves rustled in wind",
            "The painting was very colorful", "Cold wind blew from north",
            "The city looked impressive", "The lake was perfectly still",
            "Dense fog covered the valley", "The furniture was quite old",
            "The road stretched far ahead", "The clouds were dark gray",
            "The garden was beautifully kept", "The mountain was snow covered",
            "The bridge was made of stone", "The forest was deeply quiet",
            "The sand was warm and soft", "The cave was completely dark",
            "The sunset was really beautiful", "The river was quite wide",
            "The snow was very deep", "The temple was very ancient",
            "The desert was extremely hot", "The coast was rocky and steep",
            "The valley was lush and green", "The clock tower stood tall",
            "The statue was made of bronze", "The harbor was very busy",
            "The field was golden wheat", "The island was small remote",
            "The cathedral was quite massive", "The glacier was slowly moving",
            "The waterfall was very tall", "The canyon was incredibly deep",
        ],
    }
    
    sentences = []
    labels = []
    for cat, sents in templates.items():
        for s in sents[:n_per_category]:
            sentences.append(s)
            labels.append(cat)
    
    return sentences, labels


def generate_syntax_pairs():
    """Generate sentence pairs with controlled syntax differences"""
    pairs = {
        'active_passive': [
            ("The cat chased the mouse", "The mouse was chased by the cat"),
            ("The dog bit the man", "The man was bitten by the dog"),
            ("The wind blew the door", "The door was blown by the wind"),
            ("The teacher praised the student", "The student was praised by the teacher"),
            ("The chef cooked the meal", "The meal was cooked by the chef"),
            ("The rain ruined the crops", "The crops were ruined by the rain"),
            ("The artist painted the mural", "The mural was painted by the artist"),
            ("The company built the bridge", "The bridge was built by the company"),
        ],
        'statement_question': [
            ("The sky is blue today", "Is the sky blue today?"),
            ("She likes chocolate cake", "Does she like chocolate cake?"),
            ("They went to the park", "Did they go to the park?"),
            ("He can swim very well", "Can he swim very well?"),
            ("The book is on the table", "Is the book on the table?"),
            ("Water freezes at zero degrees", "Does water freeze at zero degrees?"),
            ("Birds fly south in winter", "Do birds fly south in winter?"),
            ("The project succeeded beyond expectations", "Did the project succeed beyond expectations?"),
        ],
        'singular_plural': [
            ("The cat sat on the mat", "The cats sat on the mat"),
            ("A dog ran in the park", "Some dogs ran in the park"),
            ("This flower is very beautiful", "These flowers are very beautiful"),
            ("The child played with toys", "The children played with toys"),
            ("A bird sang in the tree", "Many birds sang in the tree"),
            ("The book was on the shelf", "The books were on the shelf"),
            ("That house looks very old", "Those houses look very old"),
            ("One student answered correctly", "Several students answered correctly"),
        ],
        'tense_shift': [
            ("The cat sits on the mat", "The cat sat on the mat"),
            ("She walks to school daily", "She walked to school daily"),
            ("He runs very fast today", "He ran very fast yesterday"),
            ("They play in the garden", "They played in the garden"),
            ("The bird flies over trees", "The bird flew over trees"),
            ("Water flows down the hill", "Water flowed down the hill"),
            ("The sun shines very bright", "The sun shone very bright"),
            ("The wind blows from north", "The wind blew from north"),
        ],
    }
    return pairs


# ============================================================
# Helper: Collect hidden states using hooks
# ============================================================

def collect_all_layer_hidden(model, tokenizer, sentences, n_layers, device):
    """Collect hidden states at ALL layers for all sentences"""
    all_hidden = {i: [] for i in range(n_layers)}
    layers = get_layers(model)
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                all_hidden[layer_idx].append(output[0].detach().float().cpu())
            else:
                all_hidden[layer_idx].append(output.detach().float().cpu())
        return hook
    
    for sent in sentences:
        # Register hooks
        hooks = []
        for li in range(n_layers):
            hooks.append(layers[li].register_forward_hook(make_hook(li)))
        
        # Forward pass
        tokens = tokenizer(sent, return_tensors='pt', padding=True, truncation=True, max_length=50)
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        
        with torch.no_grad():
            try:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception as e:
                print(f"  Forward failed for '{sent[:30]}...': {e}")
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Extract last non-padding token
        last_pos = attention_mask.sum(dim=1).item() - 1
        for li in range(n_layers):
            if all_hidden[li]:
                h = all_hidden[li][-1]  # Last captured
                h_last = h[0, last_pos, :].numpy()
                all_hidden[li][-1] = h_last  # Replace tensor with numpy
    
    # Convert lists to arrays
    for li in range(n_layers):
        all_hidden[li] = np.array(all_hidden[li])
    
    return all_hidden


def collect_attn_mlp_components(model, tokenizer, sentences, n_layers, device):
    """Collect attention and MLP output contributions separately"""
    layers = get_layers(model)
    
    # We need: h_l (pre-layer), attn_out, h_{l+1} (post-layer)
    # Δh_attn = attn_out (from hook on attention output)
    # Δh_mlp = h_{l+1} - h_l - Δh_attn
    
    results_per_sentence = []
    
    for si, sent in enumerate(sentences):
        layer_outputs = {}
        attn_outputs = {}
        
        def make_layer_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    layer_outputs[layer_idx] = output[0].detach().float().cpu()
                else:
                    layer_outputs[layer_idx] = output.detach().float().cpu()
            return hook
        
        def make_attn_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    attn_outputs[layer_idx] = output[0].detach().float().cpu()
                else:
                    attn_outputs[layer_idx] = output.detach().float().cpu()
            return hook
        
        # Register hooks
        hooks = []
        for li in range(n_layers):
            layer = layers[li]
            hooks.append(layer.register_forward_hook(make_layer_hook(li)))
            
            # Hook attention output
            attn_module = None
            if hasattr(layer, 'self_attn'):
                attn_module = layer.self_attn
            elif hasattr(layer, 'attention'):
                attn_module = layer.attention
            
            if attn_module is not None:
                hooks.append(attn_module.register_forward_hook(make_attn_hook(li)))
        
        # Forward pass
        tokens = tokenizer(sent, return_tensors='pt', padding=True, truncation=True, max_length=50)
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        
        with torch.no_grad():
            try:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            except:
                pass
        
        for h in hooks:
            h.remove()
        
        # Extract last position
        last_pos = attention_mask.sum(dim=1).item() - 1
        
        sent_data = {'layer_h': {}, 'attn_out': {}, 'mlp_out': {}}
        for li in range(n_layers):
            if li in layer_outputs:
                sent_data['layer_h'][li] = layer_outputs[li][0, last_pos, :].numpy()
            if li in attn_outputs:
                sent_data['attn_out'][li] = attn_outputs[li][0, last_pos, :].numpy()
        
        # Compute MLP output = total delta - attn_out
        for li in range(1, n_layers):
            if li in sent_data['layer_h'] and (li-1) in sent_data['layer_h'] and li in sent_data['attn_out']:
                delta_total = sent_data['layer_h'][li] - sent_data['layer_h'][li-1]
                delta_attn = sent_data['attn_out'][li]
                delta_mlp = delta_total - delta_attn
                sent_data['mlp_out'][li] = delta_mlp
        
        results_per_sentence.append(sent_data)
    
    # Aggregate
    all_attn = {}
    all_mlp = {}
    all_h = {}
    
    for si, sent_data in enumerate(results_per_sentence):
        for li in sent_data['layer_h']:
            if li not in all_h:
                all_h[li] = []
            all_h[li].append(sent_data['layer_h'][li])
        for li in sent_data['attn_out']:
            if li not in all_attn:
                all_attn[li] = []
            all_attn[li].append(sent_data['attn_out'][li])
        for li in sent_data['mlp_out']:
            if li not in all_mlp:
                all_mlp[li] = []
            all_mlp[li].append(sent_data['mlp_out'][li])
    
    for li in all_h:
        all_h[li] = np.array(all_h[li])
    for li in all_attn:
        all_attn[li] = np.array(all_attn[li])
    for li in all_mlp:
        all_mlp[li] = np.array(all_mlp[li])
    
    return all_h, all_attn, all_mlp


# ============================================================
# Experiment 47A: Semantic Coordinate Recovery
# ============================================================

def exp_47a(model, tokenizer, model_name, device):
    """Recover semantic coordinates z_l from hidden states"""
    print(f"\n{'='*70}")
    print(f"47A: Semantic Coordinate Recovery — {model_name}")
    print(f"{'='*70}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    # Generate sentences
    sentences, labels = generate_diverse_sentences(n_per_category=40)
    N = len(sentences)
    print(f"N={N} sentences, {len(set(labels))} categories")
    
    # Collect all hidden states
    print("Collecting hidden states at all layers...")
    all_h = collect_all_layer_hidden(model, tokenizer, sentences, n_layers, device)
    
    # Step 1: Compute h_base from final layer
    h_L = all_h[n_layers - 1]
    h_base = h_L.mean(axis=0)
    h_sem = h_L - h_base
    
    print(f"\n||h_base|| = {np.linalg.norm(h_base):.2f}")
    print(f"h_base energy% = {np.dot(h_base, h_base) / np.mean(np.sum(h_L**2, axis=1)) * 100:.1f}%")
    
    # Step 2: PCA on h_sem to find U
    pca_full = PCA()
    pca_full.fit(h_sem)
    
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    dim50 = np.searchsorted(cumvar, 0.50) + 1
    dim80 = np.searchsorted(cumvar, 0.80) + 1
    dim90 = np.searchsorted(cumvar, 0.90) + 1
    dim95 = np.searchsorted(cumvar, 0.95) + 1
    
    print(f"\nPCA on h_sem (N={N}):")
    print(f"  dim50 = {dim50}")
    print(f"  dim80 = {dim80}")
    print(f"  dim90 = {dim90}")
    print(f"  dim95 = {dim95}")
    print(f"  Top 5 variance: {pca_full.explained_variance_ratio_[:5]}")
    
    # Step 3: Choose k for semantic subspace
    # Use dim90 for full coverage (Phase 46 showed dim80 misses critical dynamics)
    k = min(dim90, 50)  # Use dim90, cap at 50 for efficiency
    print(f"\nChosen k = {k} (from dim90={dim90})")
    
    # Extract U (top-k PCA directions)
    U = pca_full.components_[:k].T  # (d, k)
    
    # Step 4: Project ALL layers to z-space
    print(f"\nProjecting all layers to z-space...")
    z_by_layer = {}
    for layer_idx in range(n_layers):
        h_l = all_h[layer_idx]
        z_l = (h_l - h_base) @ U  # (N, k)
        z_by_layer[layer_idx] = z_l
    
    # Step 5: Analyze z evolution
    print(f"\n--- z-space statistics across layers ---")
    print(f"{'Layer':>6} {'||z||':>10} {'||dz||':>10} {'dim50(dz)':>12} {'cos(dz,z)':>12}")
    
    for layer_idx in range(n_layers):
        z_l = z_by_layer[layer_idx]
        z_norm = np.mean(np.linalg.norm(z_l, axis=1))
        
        if layer_idx > 0:
            dz = z_by_layer[layer_idx] - z_by_layer[layer_idx - 1]
            dz_norm = np.mean(np.linalg.norm(dz, axis=1))
            
            # PCA on Δz
            pca_dz = PCA()
            pca_dz.fit(dz)
            cumvar_dz = np.cumsum(pca_dz.explained_variance_ratio_)
            dim50_dz = np.searchsorted(cumvar_dz, 0.50) + 1
            
            # cos(Δz, z)
            cos_vals = []
            for i in range(N):
                dz_i = dz[i]
                z_i = z_l[i]
                n1 = np.linalg.norm(dz_i)
                n2 = np.linalg.norm(z_i)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_vals.append(np.dot(dz_i, z_i) / (n1 * n2))
            mean_cos = np.mean(cos_vals) if cos_vals else 0
            
            if layer_idx % max(1, n_layers // 8) == 0 or layer_idx <= 2 or layer_idx == n_layers - 1:
                print(f"{layer_idx:>6} {z_norm:>10.2f} {dz_norm:>10.2f} {dim50_dz:>12} {mean_cos:>12.4f}")
        else:
            print(f"{layer_idx:>6} {z_norm:>10.2f} {'---':>10} {'---':>12} {'---':>12}")
    
    # Step 6: Semantic classification in z-space
    print(f"\n--- Semantic classification in z-space ---")
    label_array = np.array(labels)
    
    for k_test in [5, 10, 20, k]:
        if k_test > k:
            continue
        z_test = z_by_layer[n_layers - 1][:, :k_test]
        clf = LogisticRegression(max_iter=1000, C=1.0)
        scores = cross_val_score(clf, z_test, label_array, cv=5, scoring='accuracy')
        print(f"  z (k={k_test:>2}): accuracy = {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Compare: LDA in z-space
    lda = LinearDiscriminantAnalysis(n_components=min(5, len(set(labels)) - 1))
    z_lda = lda.fit_transform(z_by_layer[n_layers - 1], label_array)
    clf_lda = LogisticRegression(max_iter=1000, C=1.0)
    scores_lda = cross_val_score(clf_lda, z_lda, label_array, cv=5, scoring='accuracy')
    print(f"  LDA-5: accuracy = {scores_lda.mean():.4f} ± {scores_lda.std():.4f}")
    
    # Step 7: z-space structure
    print(f"\n--- z-space structure ---")
    z_final = z_by_layer[n_layers - 1]
    
    D_z = squareform(pdist(z_final, metric='cosine'))
    within_dists = []
    cross_dists = []
    categories = list(set(labels))
    for i in range(N):
        for j in range(i+1, N):
            if labels[i] == labels[j]:
                within_dists.append(D_z[i, j])
            else:
                cross_dists.append(D_z[i, j])
    
    print(f"  Within-category cosine dist: {np.mean(within_dists):.4f}")
    print(f"  Cross-category cosine dist:  {np.mean(cross_dists):.4f}")
    print(f"  Gap: {np.mean(cross_dists) - np.mean(within_dists):.4f}")
    
    # Step 8: Orthogonality of z coordinates
    print(f"\n--- z coordinate orthogonality ---")
    z_cov = np.corrcoef(z_final.T)
    off_diag = z_cov[np.triu_indices(k, k=1)]
    print(f"  Mean |corr(z_i, z_j)| = {np.mean(np.abs(off_diag)):.4f}")
    print(f"  Max |corr(z_i, z_j)| = {np.max(np.abs(off_diag)):.4f}")
    
    # Save results
    results = {
        'model': model_name,
        'N': N,
        'k': int(k),
        'dim50': int(dim50),
        'dim80': int(dim80),
        'dim90': int(dim90),
        'dim95': int(dim95),
        'h_base_norm': float(np.linalg.norm(h_base)),
        'h_base_energy_pct': float(np.dot(h_base, h_base) / np.mean(np.sum(h_L**2, axis=1)) * 100),
        'within_dist': float(np.mean(within_dists)),
        'cross_dist': float(np.mean(cross_dists)),
        'mean_abs_corr_z': float(np.mean(np.abs(off_diag))),
    }
    
    output_dir = Path('tests/glm5_temp')
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / f'phase47_U_{model_name}.npy', U)
    np.save(output_dir / f'phase47_h_base_{model_name}.npy', h_base)
    
    # Save z_by_layer
    z_data = {str(li): z_by_layer[li].tolist() for li in range(n_layers)}
    with open(output_dir / f'phase47_z_{model_name}.json', 'w') as f:
        json.dump(z_data, f)
    
    # Save labels
    with open(output_dir / f'phase47_labels_{model_name}.json', 'w') as f:
        json.dump(labels, f)
    
    with open(output_dir / f'phase47_results_47a_{model_name}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to tests/glm5_temp/")
    
    return U, h_base, z_by_layer, labels, all_h, k


# ============================================================
# Experiment 47B: z-space Dynamics — Fit f_l
# ============================================================

def exp_47b(model, tokenizer, model_name, device):
    """Fit dynamics in z-space: z_{l+1} = z_l + f_l(z_l)"""
    print(f"\n{'='*70}")
    print(f"47B: z-space Dynamics — Fitting f_l(z_l)")
    print(f"{'='*70}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    # Load 47A data
    temp_dir = Path('tests/glm5_temp')
    U = np.load(temp_dir / f'phase47_U_{model_name}.npy')
    h_base = np.load(temp_dir / f'phase47_h_base_{model_name}.npy')
    with open(temp_dir / f'phase47_z_{model_name}.json', 'r') as f:
        z_data = json.load(f)
    z_by_layer = {int(li): np.array(v) for li, v in z_data.items()}
    
    k = U.shape[1]
    N = z_by_layer[0].shape[0]
    print(f"k={k}, N={N}")
    
    # Fit linear model: Δz = A z + b for each layer
    print(f"\n--- Linear fit: Δz = A z + b ---")
    print(f"{'Layer':>6} {'R²':>8} {'||A||_F':>10} {'||b||':>8} {'rank_eff':>10} {'top_eig':>10} {'neg_eig%':>10} {'imag%':>10}")
    
    A_matrices = {}
    b_vectors = {}
    linear_r2 = {}
    
    for layer_idx in range(1, n_layers):
        z_l = z_by_layer[layer_idx - 1]  # (N, k)
        z_next = z_by_layer[layer_idx]   # (N, k)
        dz = z_next - z_l                # (N, k)
        
        # Fit: dz = A z + b via Ridge regression
        X = np.column_stack([z_l, np.ones(N)])  # (N, k+1)
        Y = dz  # (N, k)
        
        reg = Ridge(alpha=1.0)
        reg.fit(X, Y)
        
        A = reg.coef_[:, :k].T  # (k, k)
        b = reg.coef_[:, k]     # (k,)
        
        A_matrices[layer_idx] = A
        b_vectors[layer_idx] = b
        
        # R²
        Y_pred = reg.predict(X)
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        linear_r2[layer_idx] = r2
        
        # Eigenvalue analysis
        eigenvalues = np.linalg.eigvals(A)
        real_eigs = eigenvalues.real
        imag_eigs = eigenvalues.imag
        
        A_frob = np.linalg.norm(A, 'fro')
        b_norm = np.linalg.norm(b)
        
        # Effective rank
        _, s, _ = np.linalg.svd(A)
        total_var = np.sum(s)
        if total_var > 0:
            cum_s = np.cumsum(s) / total_var
            rank_eff = np.searchsorted(cum_s, 0.90) + 1
        else:
            rank_eff = 0
        
        top_eig = np.max(np.abs(real_eigs))
        neg_frac = np.mean(real_eigs < 0)
        imag_frac = np.mean(np.abs(imag_eigs) > 0.01 * max(np.max(np.abs(real_eigs)), 1e-10))
        
        if layer_idx % max(1, n_layers // 8) == 0 or layer_idx == n_layers - 1:
            print(f"{layer_idx:>6} {r2:>8.4f} {A_frob:>10.4f} {b_norm:>8.4f} {rank_eff:>10} {top_eig:>10.4f} {neg_frac*100:>10.1f}% {imag_frac*100:>10.1f}%")
    
    # Step 2: z-space rotation vs expansion analysis
    print(f"\n--- z-space dynamics type ---")
    print(f"{'Layer':>6} {'cos(dz,z)':>12} {'radial%':>10} {'tangent%':>10} {'Δ||z||²':>10}")
    
    for layer_idx in range(1, n_layers):
        z_l = z_by_layer[layer_idx - 1]
        z_next = z_by_layer[layer_idx]
        dz = z_next - z_l
        
        cos_vals = []
        radial_fracs = []
        tangent_fracs = []
        delta_norm_sq = []
        
        for i in range(N):
            dz_i = dz[i]
            z_i = z_l[i]
            n1 = np.linalg.norm(dz_i)
            n2 = np.linalg.norm(z_i)
            
            if n1 < 1e-10 or n2 < 1e-10:
                continue
            
            cos_val = np.dot(dz_i, z_i) / (n1 * n2)
            
            z_dir = z_i / n2
            radial = np.dot(dz_i, z_dir)
            tangent = np.linalg.norm(dz_i - radial * z_dir)
            
            cos_vals.append(cos_val)
            radial_fracs.append(abs(radial) / n1)
            tangent_fracs.append(tangent / n1)
            delta_norm_sq.append(np.linalg.norm(z_next[i])**2 - np.linalg.norm(z_i)**2)
        
        if layer_idx % max(1, n_layers // 8) == 0 or layer_idx == n_layers - 1:
            print(f"{layer_idx:>6} {np.mean(cos_vals):>12.4f} {np.mean(radial_fracs)*100:>10.1f}% {np.mean(tangent_fracs)*100:>10.1f}% {np.mean(delta_norm_sq):>10.1f}")
    
    # Step 3: Eigenvalue spectrum (detailed)
    print(f"\n--- Eigenvalue spectrum of A (sample layers) ---")
    for layer_idx in [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
        if layer_idx not in A_matrices:
            continue
        A = A_matrices[layer_idx]
        eigenvalues = np.linalg.eigvals(A)
        
        print(f"\n  Layer {layer_idx} (R²={linear_r2[layer_idx]:.4f}):")
        print(f"    Max |Re(λ)| = {np.max(np.abs(eigenvalues.real)):.4f}")
        print(f"    Max |Im(λ)| = {np.max(np.abs(eigenvalues.imag)):.4f}")
        print(f"    Mean Re(λ)  = {np.mean(eigenvalues.real):.4f}")
        re_abs = np.abs(eigenvalues.real)
        im_abs = np.abs(eigenvalues.imag)
        print(f"    |Im|/|Re|   = {np.mean(im_abs) / max(np.mean(re_abs), 1e-10):.4f}")
        print(f"    Re(λ) > 0%  = {np.mean(eigenvalues.real > 0)*100:.1f}%")
        print(f"    |Im(λ)| > 0.01% = {np.mean(np.abs(eigenvalues.imag) > 0.01)*100:.1f}%")
        
        # Top 5 eigenvalues
        top_indices = np.argsort(np.abs(eigenvalues))[::-1][:5]
        print(f"    Top 5 eigenvalues:")
        for idx in top_indices:
            print(f"      {eigenvalues[idx]:.4f}")
    
    # Step 4: Multi-step prediction
    print(f"\n--- Multi-step prediction in z-space ---")
    for step in [1, 2, 3]:
        r2_values = []
        for layer_idx in range(1, n_layers - step):
            z_start = z_by_layer[layer_idx - 1]
            z_target = z_by_layer[layer_idx - 1 + step]
            
            # Compose: z_pred = z_start + Σ (A_l z + b_l)
            z_pred = z_start.copy()
            for s in range(step):
                current_layer = layer_idx + s
                if current_layer in A_matrices:
                    z_pred = z_pred + z_pred @ A_matrices[current_layer].T + b_vectors[current_layer]
            
            ss_res = np.sum((z_target - z_pred) ** 2)
            ss_tot = np.sum((z_target - z_target.mean(axis=0)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            r2_values.append(r2)
        
        if r2_values:
            print(f"  Step={step}: mean R² = {np.mean(r2_values):.4f}, min={np.min(r2_values):.4f}, max={np.max(r2_values):.4f}")
    
    # Step 5: Nonlinearity test
    print(f"\n--- Nonlinearity test ---")
    for layer_idx in [n_layers // 4, n_layers // 2, 3 * n_layers // 4]:
        if layer_idx not in z_by_layer or layer_idx - 1 not in z_by_layer:
            continue
        
        z_l = z_by_layer[layer_idx - 1]
        z_next = z_by_layer[layer_idx]
        dz = z_next - z_l
        
        # Linear features
        X_lin = np.column_stack([z_l, np.ones(N)])
        
        # Quadratic features (top-k_quad)
        k_quad = min(10, k)
        z_quad = z_l[:, :k_quad]
        quad_features = []
        for i in range(k_quad):
            for j in range(i, k_quad):
                quad_features.append(z_quad[:, i] * z_quad[:, j])
        X_quad = np.column_stack(quad_features) if quad_features else np.zeros((N, 0))
        X_full = np.column_stack([X_lin, X_quad])
        
        reg_lin = Ridge(alpha=1.0)
        reg_lin.fit(X_lin, dz)
        r2_lin = reg_lin.score(X_lin, dz)
        
        if X_quad.shape[1] > 0 and X_full.shape[1] < N:
            reg_quad = Ridge(alpha=1.0)
            reg_quad.fit(X_full, dz)
            r2_quad = reg_quad.score(X_full, dz)
        else:
            r2_quad = r2_lin
        
        print(f"  Layer {layer_idx}: Linear R²={r2_lin:.4f}, Quad R²={r2_quad:.4f}, improvement={r2_quad-r2_lin:.4f}")
    
    # Step 6: Summary — Is A rotation-like?
    print(f"\n--- Summary: Is A rotation-like? ---")
    mean_r2 = np.mean(list(linear_r2.values()))
    
    # Check if A is approximately antisymmetric
    antisym_scores = []
    for layer_idx in A_matrices:
        A = A_matrices[layer_idx]
        A_anti = (A - A.T) / 2
        A_sym = (A + A.T) / 2
        anti_norm = np.linalg.norm(A_anti, 'fro')
        sym_norm = np.linalg.norm(A_sym, 'fro')
        total_norm = np.linalg.norm(A, 'fro')
        if total_norm > 1e-10:
            antisym_scores.append(anti_norm / total_norm)
    
    print(f"  Mean linear R² = {mean_r2:.4f}")
    print(f"  Mean antisymmetric fraction = {np.mean(antisym_scores):.4f}")
    print(f"  → {'Rotation-like (antisym dominant)' if np.mean(antisym_scores) > 0.5 else 'Mixed (sym + antisym)' if np.mean(antisym_scores) > 0.3 else 'Expansion-like (sym dominant)'}")
    
    results = {
        'model': model_name,
        'k': int(k),
        'N': N,
        'mean_linear_r2': float(mean_r2),
        'mean_antisym_fraction': float(np.mean(antisym_scores)),
    }
    
    temp_dir = Path('tests/glm5_temp')
    with open(temp_dir / f'phase47_results_47b_{model_name}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return A_matrices, b_vectors, linear_r2


# ============================================================
# Experiment 47C: Attention vs MLP decomposition in z-space
# ============================================================

def exp_47c(model, tokenizer, model_name, device):
    """Decompose attention vs MLP contributions in z-space"""
    print(f"\n{'='*70}")
    print(f"47C: Attention vs MLP Decomposition in z-space — {model_name}")
    print(f"{'='*70}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    temp_dir = Path('tests/glm5_temp')
    U = np.load(temp_dir / f'phase47_U_{model_name}.npy')
    h_base = np.load(temp_dir / f'phase47_h_base_{model_name}.npy')
    k = U.shape[1]
    
    # Use smaller set for decomposition
    sentences, labels = generate_diverse_sentences(n_per_category=10)
    N = len(sentences)
    print(f"N={N} sentences for decomposition")
    
    print("Collecting attn/MLP components...")
    all_h, all_attn, all_mlp = collect_attn_mlp_components(model, tokenizer, sentences, n_layers, device)
    
    # Project to z-space
    z_by_layer = {}
    dz_attn_z = {}
    dz_mlp_z = {}
    dz_total_z = {}
    
    for li in all_h:
        z_by_layer[li] = (all_h[li] - h_base) @ U
    
    for li in all_attn:
        dz_attn_z[li] = all_attn[li] @ U
    for li in all_mlp:
        dz_mlp_z[li] = all_mlp[li] @ U
    for li in all_h:
        if (li-1) in all_h:
            dz_total_z[li] = (all_h[li] - all_h[li-1]) @ U
    
    # Analysis - only on layers that have both attn and mlp data
    common_layers = sorted(set(dz_attn_z.keys()) & set(dz_mlp_z.keys()))
    
    # Analysis
    print(f"\n--- Attn vs MLP in z-space ---")
    print(f"{'Layer':>6} {'||dz_attn||':>12} {'||dz_mlp||':>12} {'attn/total':>12} {'cos(attn,mlp)':>14}")
    
    for layer_idx in common_layers:
        dz_a = dz_attn_z[layer_idx]
        dz_m = dz_mlp_z[layer_idx]
        
        norm_attn = np.mean(np.linalg.norm(dz_a, axis=1))
        norm_mlp = np.mean(np.linalg.norm(dz_m, axis=1))
        norm_total = norm_attn + norm_mlp
        
        attn_frac = norm_attn / norm_total if norm_total > 0 else 0
        
        cos_vals = []
        for i in range(min(len(dz_a), len(dz_m))):
            n1 = np.linalg.norm(dz_a[i])
            n2 = np.linalg.norm(dz_m[i])
            if n1 > 1e-10 and n2 > 1e-10:
                cos_vals.append(np.dot(dz_a[i], dz_m[i]) / (n1 * n2))
        mean_cos = np.mean(cos_vals) if cos_vals else 0
        
        if layer_idx % max(1, n_layers // 8) == 0 or layer_idx == n_layers - 1:
            print(f"{layer_idx:>6} {norm_attn:>12.4f} {norm_mlp:>12.4f} {attn_frac:>12.4f} {mean_cos:>14.4f}")
    
    # Radial vs tangential for each component
    print(f"\n--- Radial vs Tangential for attn/MLP ---")
    print(f"{'Layer':>6} {'attn_rad%':>10} {'attn_tan%':>10} {'mlp_rad%':>10} {'mlp_tan%':>10}")
    
    for layer_idx in common_layers:
        dz_a = dz_attn_z[layer_idx]
        dz_m = dz_mlp_z[layer_idx]
        z_l = z_by_layer.get(layer_idx - 1, None)
        
        if z_l is None:
            continue
        
        attn_rad, attn_tan, mlp_rad, mlp_tan = [], [], [], []
        
        n_samples = min(len(dz_a), len(dz_m), len(z_l))
        for i in range(n_samples):
            z_norm = np.linalg.norm(z_l[i])
            if z_norm < 1e-10:
                continue
            z_dir = z_l[i] / z_norm
            
            # Attention
            if i < len(dz_a) and np.linalg.norm(dz_a[i]) > 1e-10:
                proj = np.dot(dz_a[i], z_dir)
                attn_rad.append(abs(proj) / np.linalg.norm(dz_a[i]))
                attn_tan.append(np.linalg.norm(dz_a[i] - proj * z_dir) / np.linalg.norm(dz_a[i]))
            
            # MLP
            if i < len(dz_m) and np.linalg.norm(dz_m[i]) > 1e-10:
                proj = np.dot(dz_m[i], z_dir)
                mlp_rad.append(abs(proj) / np.linalg.norm(dz_m[i]))
                mlp_tan.append(np.linalg.norm(dz_m[i] - proj * z_dir) / np.linalg.norm(dz_m[i]))
        
        if layer_idx % max(1, n_layers // 8) == 0 or layer_idx == n_layers - 1:
            ar = np.mean(attn_rad)*100 if attn_rad else 0
            at = np.mean(attn_tan)*100 if attn_tan else 0
            mr = np.mean(mlp_rad)*100 if mlp_rad else 0
            mt = np.mean(mlp_tan)*100 if mlp_tan else 0
            print(f"{layer_idx:>6} {ar:>10.1f}% {at:>10.1f}% {mr:>10.1f}% {mt:>10.1f}%")
    
    # dim50 for each component
    print(f"\n--- dim50 for attn/MLP in z-space ---")
    print(f"{'Layer':>6} {'dim50(attn)':>12} {'dim50(mlp)':>12} {'dim50(total)':>12}")
    
    for layer_idx in common_layers:
        dz_a = dz_attn_z[layer_idx]
        dz_m = dz_mlp_z[layer_idx]
        dz_t = dz_total_z.get(layer_idx, None)
        
        def get_dim50(data):
            if len(data) < 3:
                return 0
            pca = PCA()
            pca.fit(data)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            return int(np.searchsorted(cumvar, 0.50) + 1)
        
        dim50_a = get_dim50(dz_a)
        dim50_m = get_dim50(dz_m)
        dim50_t = get_dim50(dz_t) if dz_t is not None else 0
        
        if layer_idx % max(1, n_layers // 8) == 0 or layer_idx == n_layers - 1:
            print(f"{layer_idx:>6} {dim50_a:>12} {dim50_m:>12} {dim50_t:>12}")
    
    results = {
        'model': model_name,
        'N': N,
        'k': int(k),
    }
    
    with open(temp_dir / f'phase47_results_47c_{model_name}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


# ============================================================
# Experiment 47D: Syntax → Dynamical Mode
# ============================================================

def exp_47d(model, tokenizer, model_name, device):
    """Test if different syntax structures produce different dynamical modes"""
    print(f"\n{'='*70}")
    print(f"47D: Syntax → Dynamical Mode — {model_name}")
    print(f"{'='*70}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    temp_dir = Path('tests/glm5_temp')
    U = np.load(temp_dir / f'phase47_U_{model_name}.npy')
    h_base = np.load(temp_dir / f'phase47_h_base_{model_name}.npy')
    k = U.shape[1]
    
    syntax_pairs = generate_syntax_pairs()
    
    for pair_type, pairs in syntax_pairs.items():
        print(f"\n--- Syntax pair type: {pair_type} ---")
        
        z_trajectories_a = []
        z_trajectories_b = []
        
        for sent_a, sent_b in pairs:
            for sent, traj_list in [(sent_a, z_trajectories_a), (sent_b, z_trajectories_b)]:
                # Collect all layer hidden states for this sentence
                all_h = collect_all_layer_hidden(model, tokenizer, [sent], n_layers, device)
                
                z_traj = []
                for li in range(n_layers):
                    h = all_h[li][0]  # First (only) sentence
                    z = (h - h_base) @ U
                    z_traj.append(z)
                
                traj_list.append(np.array(z_traj))
        
        # Analyze Δz trajectories
        print(f"  {'Layer':>6} {'cos(dz_a, dz_b)':>16} {'||ddz||':>10} {'dz_a_rad%':>12} {'dz_b_rad%':>12}")
        
        cos_dz_values = []
        ddz_norms = []
        
        for layer_idx in range(1, n_layers):
            dz_a_list = []
            dz_b_list = []
            
            for i in range(len(pairs)):
                dz_a = z_trajectories_a[i][layer_idx] - z_trajectories_a[i][layer_idx - 1]
                dz_b = z_trajectories_b[i][layer_idx] - z_trajectories_b[i][layer_idx - 1]
                dz_a_list.append(dz_a)
                dz_b_list.append(dz_b)
            
            dz_a_arr = np.array(dz_a_list)
            dz_b_arr = np.array(dz_b_list)
            
            mean_dz_a = dz_a_arr.mean(axis=0)
            mean_dz_b = dz_b_arr.mean(axis=0)
            
            n1 = np.linalg.norm(mean_dz_a)
            n2 = np.linalg.norm(mean_dz_b)
            cos_dz = np.dot(mean_dz_a, mean_dz_b) / (n1 * n2) if n1 > 1e-10 and n2 > 1e-10 else 0
            
            ddz = dz_a_arr - dz_b_arr
            ddz_norm = np.mean(np.linalg.norm(ddz, axis=1))
            
            cos_dz_values.append(cos_dz)
            ddz_norms.append(ddz_norm)
            
            # Radial fractions
            z_a = np.array([z_trajectories_a[i][layer_idx - 1] for i in range(len(pairs))])
            z_b = np.array([z_trajectories_b[i][layer_idx - 1] for i in range(len(pairs))])
            
            def get_mean_radial(dz_arr, z_arr):
                rads = []
                for i in range(len(dz_arr)):
                    n = np.linalg.norm(z_arr[i])
                    if np.linalg.norm(dz_arr[i]) > 1e-10 and n > 1e-10:
                        proj = np.dot(dz_arr[i], z_arr[i] / n)
                        rads.append(abs(proj) / np.linalg.norm(dz_arr[i]))
                return np.mean(rads) if rads else 0
            
            rad_a = get_mean_radial(dz_a_arr, z_a)
            rad_b = get_mean_radial(dz_b_arr, z_b)
            
            if layer_idx % max(1, n_layers // 4) == 0 or layer_idx == n_layers - 1:
                print(f"  {layer_idx:>6} {cos_dz:>16.4f} {ddz_norm:>10.4f} {rad_a*100:>12.1f}% {rad_b*100:>12.1f}%")
        
        # Summary
        print(f"\n  Summary ({pair_type}):")
        print(f"    Mean cos(Δz_a, Δz_b) = {np.mean(cos_dz_values):.4f}")
        print(f"    Mean ||ΔΔz|| = {np.mean(ddz_norms):.4f}")
        early = cos_dz_values[:n_layers//3]
        late = cos_dz_values[2*n_layers//3:]
        if early:
            print(f"    Early layers: cos = {np.mean(early):.4f}")
        if late:
            print(f"    Late layers: cos = {np.mean(late):.4f}")
        
        # ΔΔz structure
        delta_dz_list = []
        for i in range(len(pairs)):
            for layer_idx in range(1, n_layers):
                dz_a = z_trajectories_a[i][layer_idx] - z_trajectories_a[i][layer_idx - 1]
                dz_b = z_trajectories_b[i][layer_idx] - z_trajectories_b[i][layer_idx - 1]
                delta_dz_list.append(dz_a - dz_b)
        
        delta_dz_arr = np.array(delta_dz_list)
        if len(delta_dz_arr) > 2:
            pca_ddz = PCA()
            pca_ddz.fit(delta_dz_arr)
            cumvar_ddz = np.cumsum(pca_ddz.explained_variance_ratio_)
            dim50_ddz = int(np.searchsorted(cumvar_ddz, 0.50) + 1)
            dim90_ddz = int(np.searchsorted(cumvar_ddz, 0.90) + 1)
            print(f"    ΔΔz dim50 = {dim50_ddz}, dim90 = {dim90_ddz}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 47: Semantic Coordinate Dynamics')
    parser.add_argument('--model', type=str, required=True, choices=['deepseek7b', 'glm4', 'qwen3'])
    parser.add_argument('--exp', type=int, required=True, choices=[1, 2, 3, 4],
                       help='Experiment number: 1=47A, 2=47B, 3=47C, 4=47D')
    args = parser.parse_args()
    
    print(f"Device: cuda" if torch.cuda.is_available() else "Device: cpu")
    
    # Load model
    model, tokenizer, device = load_model(args.model)
    model_name = args.model
    
    if args.exp == 1:
        exp_47a(model, tokenizer, model_name, device)
    elif args.exp == 2:
        exp_47b(model, tokenizer, model_name, device)
    elif args.exp == 3:
        exp_47c(model, tokenizer, model_name, device)
    elif args.exp == 4:
        exp_47d(model, tokenizer, model_name, device)
    
    # Cleanup
    release_model(model)
    print("\nDone!")


if __name__ == '__main__':
    main()
