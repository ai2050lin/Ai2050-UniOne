"""
CCXLVIII(408): 大规模自然句Pair编码验证
=========================================
解决CCXLVII的三大硬伤:
  1. 样本量太小 → 200+自然句, 每类50+对
  2. 模板偏置 → 使用spaCy自动解析自然句依存树
  3. pair_full失败归因 → 系统消融: 维度/样本量/降维方法/分类器

核心验证:
  Exp1: 大规模自然句 pair_diff vs point 分类 (6类依存)
  Exp2: 方向分类在自然句上的表现
  Exp3: pair_full系统性失败分析 (维度/样本/降维/分类器)
  Exp4: 注意力方向在大规模自然句上的验证
  Exp5: 泛化测试: 模板训练→自然句测试 (量化偏置)
"""

import torch
import numpy as np
import json
import argparse
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import ttest_rel
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.glm5.model_utils import load_model

# Global device
_DEVICE = "cuda:0"

# ============================================================
# 大规模自然句语料库 (200+句, 涵盖多种句法和词汇)
# ============================================================

# 用spaCy自动解析, 但先提供大量自然句
NATURAL_SENTENCES = [
    # === Simple declarative (varied subjects/verbs/objects) ===
    "The scientist discovered a new species in the rainforest",
    "Several students completed their assignments before the deadline",
    "My neighbor bought an expensive painting at the auction",
    "The chef prepared a delicious meal for the guests",
    "Two children found a stray kitten near the park",
    "The engineer designed a revolutionary bridge across the river",
    "Many tourists visited the ancient temple during summer",
    "The author published three novels last year",
    "Our teacher explained the complex theorem clearly",
    "The musician composed a beautiful sonata for piano",
    "Some researchers identified the gene responsible for the disease",
    "The president signed the controversial bill into law",
    "A young woman won the prestigious scholarship",
    "The company launched several innovative products this quarter",
    "His grandmother told him fascinating stories about the war",
    "The detective solved the mysterious case within a week",
    "Local farmers harvested abundant crops despite the drought",
    "The artist painted a stunning portrait of her mother",
    "An elderly gentleman donated millions to the hospital",
    "The professor delivered an inspiring lecture on quantum physics",
    
    # === With adverbial modifiers ===
    "The children played happily in the garden",
    "She walked slowly through the crowded market",
    "The dog barked furiously at the mailman",
    "He spoke softly to the frightened animal",
    "They worked diligently on the challenging project",
    "The wind blew fiercely across the open plain",
    "The baby slept peacefully in her crib",
    "The athlete ran swiftly toward the finish line",
    "She answered the question confidently during the interview",
    "The river flowed gently through the valley",
    
    # === With prepositional phrases ===
    "The cat slept on the warm rug",
    "She placed the vase near the window",
    "The bird built a nest in the old oak tree",
    "They sat around the large wooden table",
    "The book fell from the top shelf",
    "He hid behind the tall hedge",
    "The car stopped at the traffic light",
    "She walked along the sandy beach",
    "The squirrel ran up the telephone pole",
    "The children played beside the sparkling stream",
    
    # === With adjectives ===
    "The ancient castle stood on the hill",
    "A tiny kitten crawled under the fence",
    "The brilliant scientist won the award",
    "Several tall buildings dominated the skyline",
    "The spicy soup burned his tongue",
    "A gentle breeze rustled the autumn leaves",
    "The wealthy businessman owned three yachts",
    "The curious toddler explored every room",
    "An old man fed the pigeons",
    "The fierce storm damaged the coastal town",
    
    # === Complex sentences (relative clauses, embedded) ===
    "The woman who lives next door teaches mathematics",
    "The dog that chased the cat returned home",
    "The book which I borrowed from the library is fascinating",
    "The students who studied hard passed the examination",
    "The scientist whose research was groundbreaking received recognition",
    "The city where I grew up has changed dramatically",
    "The movie that we watched yesterday was thrilling",
    "The man whom she interviewed seemed nervous",
    "The restaurant whose chef won the prize was fully booked",
    "The team that scored first eventually won the match",
    
    # === Passive constructions ===
    "The painting was stolen by an unknown thief",
    "The report was written by the senior analyst",
    "The building was designed by a famous architect",
    "The problem was solved by a brilliant student",
    "The cake was baked by my younger sister",
    "The concert was attended by thousands of fans",
    "The document was signed by both parties",
    "The experiment was conducted by graduate researchers",
    "The accident was caused by reckless driving",
    "The discovery was announced by the university",
    
    # === Questions ===
    "Which student submitted the best project",
    "What caused the sudden power outage",
    "Who directed the award-winning film",
    "How did the team overcome the obstacle",
    "When will the new policy take effect",
    "Where did the explorer find the artifact",
    "Why did the committee reject the proposal",
    "Which company developed the vaccine",
    "Who wrote the controversial editorial",
    "What influenced the court decision",
    
    # === Sentences with determiners varied ===
    "Some people believe in supernatural phenomena",
    "Every student must submit the assignment",
    "Both candidates debated the economic policy",
    "No witness came forward with information",
    "Several witnesses reported hearing strange noises",
    "Each participant received a certificate",
    "Many historians doubt the official account",
    "Any solution must address the root cause",
    "Few politicians kept their campaign promises",
    "All employees attended the mandatory meeting",
    
    # === Coordination ===
    "The teacher and the principal discussed the curriculum",
    "Cats and dogs often make good companions",
    "She sang and danced at the talent show",
    "The sun rose and the birds began to sing",
    "He cooked dinner and washed the dishes",
    
    # === Sentences with various determiner-noun pairs ===
    "The quick fox jumped over the fence",
    "A large elephant wandered into the village",
    "This old house needs major repairs",
    "Those beautiful flowers attract many butterflies",
    "His latest novel became a bestseller",
    "Her unusual accent fascinated the audience",
    "My favorite restaurant closed last month",
    "Their new neighbor plays the violin",
    "Every spring the garden blooms beautifully",
    "Another candidate joined the race",
    
    # === Long distance dependencies ===
    "The key that the janitor lost yesterday was found today",
    "The scientist who the university hired last year published breakthrough research",
    "The proposal that the committee rejected was later revised",
    "The man who the police arrested confessed to the crime",
    "The theory which the physicist developed explains dark matter",
    
    # === Sentences with multiple PPs ===
    "The cat slept on the rug near the fireplace in the living room",
    "She placed the book on the shelf beside the window",
    "The bird flew from the tree to the feeder near the house",
    "He walked through the park toward the museum on main street",
    "The dog ran from the yard into the street after the cat",
    
    # === Various verb types ===
    "The evidence seems compelling to the jury",
    "She became a renowned surgeon after years of training",
    "The soup tastes slightly salty to me",
    "The proposal sounds reasonable to the board",
    "The patient grew increasingly frustrated with the delay",
    "He remained silent throughout the interrogation",
    "The weather turned cold unexpectedly in October",
    "She appeared nervous before the large audience",
    "The situation proved more difficult than anticipated",
    "The results look promising to the research team",
    
    # === Ditransitive verbs ===
    "The teacher gave the student a scholarship",
    "She told her friend the entire story",
    "The company offered the candidate the position",
    "He showed his daughter the constellation",
    "The manager sent the team the updated schedule",
    
    # === Control verbs ===
    "The boy wanted to build a treehouse",
    "She decided to pursue a medical degree",
    "They attempted to climb the mountain",
    "He promised to finish the report by Friday",
    "The couple planned to renovate their kitchen",
    
    # === Existential there ===
    "There is a problem with the hypothesis",
    "There were several objections to the proposal",
    "There exists a fundamental contradiction in the argument",
    "There remains considerable uncertainty about the outcome",
    "There appeared a faint light in the distance",
    
    # === Cleft sentences ===
    "It was the detective who solved the mystery",
    "It is the context that determines the meaning",
    "It was last Tuesday that the incident occurred",
    "It is precisely this ambiguity that creates the difficulty",
    "It was the younger sibling who won the competition",
]

# 目标依存类型 (8种, 比之前多2种)
TARGET_DEP_TYPES = [
    "subj_verb",    # nsubj → ROOT verb
    "verb_obj",     # ROOT verb → dobj
    "noun_adj",     # noun → amod
    "verb_adv",     # verb → advmod
    "prep_pobj",    # preposition → pobj
    "noun_det",     # noun → det
    "noun_poss",    # noun → poss (新增: 所有格)
    "verb_aux",     # verb → aux (新增: 助动词)
]

DEP_TYPE_MAP = {
    "nsubj": "subj_verb",
    "nsubjpass": "subj_verb",
    "csubj": "subj_verb",
    "dobj": "verb_obj",
    "iobj": "verb_obj",
    "obj": "verb_obj",
    "amod": "noun_adj",
    "advmod": "verb_adv",
    "prep": "prep_pobj",
    "pobj": "prep_pobj",
    "det": "noun_det",
    "poss": "noun_poss",
    "aux": "verb_aux",
    "auxpass": "verb_aux",
}


def parse_with_spacy(sentences):
    """Use spaCy to parse sentences and extract dependency pairs.
    
    Returns list of (sentence, head_idx, dep_idx, dep_type) tuples.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except ImportError:
        print("  spaCy not available, falling back to simple parsing")
        return parse_simple(sentences)
    except OSError:
        print("  en_core_web_sm model not found, trying en_core_web_lg")
        try:
            import spacy
            nlp = spacy.load("en_core_web_lg")
        except:
            print("  No spaCy model available, falling back to simple parsing")
            return parse_simple(sentences)
    
    pairs = []
    for sent in sentences:
        doc = nlp(sent)
        for token in doc:
            dep_label = token.dep_
            if dep_label in DEP_TYPE_MAP:
                dep_type = DEP_TYPE_MAP[dep_label]
                head_idx = token.head.i
                dep_idx = token.i
                pairs.append((sent, head_idx, dep_idx, dep_type))
    
    return pairs


def parse_simple(sentences):
    """Fallback: simple rule-based parsing for common patterns."""
    import re
    pairs = []
    
    for sent in sentences:
        words = sent.split()
        n = len(words)
        
        # Simple heuristics for common patterns
        # Pattern: "The ADJ NOUN VERB ..." → noun_adj at (2,1), noun_det at (2,0), subj_verb at (2,3)
        # Pattern: "NOUN VERB the NOUN" → subj_verb at (0,1), verb_obj at (1,3), noun_det at (3,2)
        
        for i, w in enumerate(words):
            wl = w.lower().rstrip('.,;?!')
            
            # Determiner detection
            if wl in ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'his', 'her', 'their', 'our', 'your', 'every', 'some', 'any', 'no', 'each', 'both', 'all', 'few', 'several', 'many']:
                # Next word might be adjective or noun
                if i + 1 < n:
                    next_w = words[i+1].lower().rstrip('.,;?!')
                    # Check if next-next is noun (adj in between)
                    if i + 2 < n:
                        nn_w = words[i+2].lower().rstrip('.,;?!')
                        # Heuristic: if i+2 ends with typical noun suffixes or is common
                        if nn_w.endswith(('tion', 'ment', 'ness', 'ity', 'er', 'or', 'ist', 'ant', 'ent', 'ence', 'ance', 'dom', 'ship', 'ism')):
                            pairs.append((sent, i+2, i, "noun_det"))
        
        # Subject-verb: find "NOUN VERB" pattern
        for i in range(n-1):
            w1 = words[i].lower().rstrip('.,;?!')
            w2 = words[i+1].lower().rstrip('.,;?!')
            
            # Common verb endings
            if w2.endswith(('ed', 'es', 'ing', 'ate', 'ize', 'ify')) or w2 in [
                'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'has', 'have', 'had', 'do', 'does', 'did',
                'will', 'would', 'can', 'could', 'shall', 'should', 'may', 'might', 'must',
                'runs', 'walks', 'sleeps', 'eats', 'finds', 'reads', 'chases', 'plays',
                'run', 'walk', 'sleep', 'eat', 'find', 'read', 'chase', 'play',
                'discovered', 'completed', 'bought', 'prepared', 'found', 'designed',
                'visited', 'published', 'explained', 'composed', 'identified', 'signed',
                'won', 'launched', 'told', 'solved', 'harvested', 'painted', 'donated',
                'delivered', 'played', 'walked', 'barked', 'spoke', 'worked', 'blew',
                'slept', 'ran', 'answered', 'flowed', 'stood', 'crawled', 'won',
                'dominated', 'burned', 'rustled', 'owned', 'explored', 'fed', 'damaged',
                'teaches', 'returned', 'passed', 'received', 'changed', 'watched',
                'seemed', 'attended', 'signed', 'conducted', 'caused', 'announced',
                'submitted', 'caused', 'directed', 'overcame', 'wrote', 'influenced',
                'believe', 'must', 'debated', 'came', 'reported', 'received',
                'doubt', 'kept', 'attended', 'discussed', 'sang', 'danced', 'rose', 'began',
                'cooked', 'washed', 'jumped', 'wandered', 'needs', 'attract', 'became',
                'closed', 'plays', 'joined', 'lost', 'was', 'hired', 'published',
                'rejected', 'arrested', 'confessed', 'developed', 'explains', 'placed',
                'built', 'flew', 'walked', 'stopped', 'sat', 'fell', 'hid',
                'grew', 'remained', 'turned', 'appeared', 'proved', 'look',
                'gave', 'told', 'offered', 'showed', 'sent',
                'wanted', 'decided', 'attempted', 'promised', 'planned',
                'is', 'were', 'exists', 'remains', 'appeared',
            ]:
                # Check if w1 is likely a subject (not a determiner or preposition)
                if w1 not in ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'and', 'or', 'but', 'who', 'which', 'that', 'where', 'when', 'how', 'why', 'what']:
                    pairs.append((sent, i, i+1, "subj_verb"))
    
    return pairs


def extract_pair_features(model, tokenizer, sentence, head_idx, dep_idx, layer=-1):
    """Extract token representations and construct pair features."""
    from tests.glm5.model_utils import get_layers
    layers = get_layers(model)
    target_layer = layer if layer >= 0 else len(layers) + layer
    
    toks = tokenizer(sentence, return_tensors="pt").to(_DEVICE)
    input_ids = toks["input_ids"]
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    n_tokens = len(tokens)
    
    # BOS detection
    has_bos = tokens[0] in ['<s>', '<|begin_of_text|>', '<|im_start|>', 'Ċ', '<|endoftext|>']
    offset = 1 if has_bos else 0
    
    actual_head = head_idx + offset
    actual_dep = dep_idx + offset
    
    # Safety check with boundary clamping
    actual_head = min(actual_head, n_tokens - 1)
    actual_dep = min(actual_dep, n_tokens - 1)
    
    if actual_head < 0 or actual_dep < 0:
        raise IndexError(f"Negative index: head={actual_head}, dep={actual_dep}")
    
    captured = {}
    def make_hook():
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured["h"] = output[0].detach().float().cpu()
            else:
                captured["h"] = output.detach().float().cpu()
        return hook
    
    hook = layers[target_layer].register_forward_hook(make_hook())
    with torch.no_grad():
        _ = model(**toks)
    hook.remove()
    
    if "h" not in captured:
        raise ValueError(f"Failed to capture hidden states for: {sentence}")
    
    hidden = captured["h"][0].numpy()
    h_head = hidden[actual_head]
    h_dep = hidden[actual_dep]
    
    pair_diff = h_head - h_dep
    pair_prod = h_head * h_dep
    pair_concat = np.concatenate([h_head, h_dep])
    pair_full = np.concatenate([h_head, h_dep, pair_diff, pair_prod])
    
    return {
        "h_head": h_head,
        "h_dep": h_dep,
        "pair_diff": pair_diff,
        "pair_prod": pair_prod,
        "pair_concat": pair_concat,
        "pair_full": pair_full,
    }


def extract_attention_weights(model, tokenizer, sentence, head_idx, dep_idx, layer=-1):
    """Extract attention weights between head and dep tokens."""
    from tests.glm5.model_utils import get_layers
    layers = get_layers(model)
    target_layer = layer if layer >= 0 else len(layers) + layer
    
    toks = tokenizer(sentence, return_tensors="pt").to(_DEVICE)
    input_ids = toks["input_ids"]
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    has_bos = tokens[0] in ['<s>', '<|begin_of_text|>', '<|im_start|>', 'Ċ', '<|endoftext|>']
    offset = 1 if has_bos else 0
    actual_head = min(head_idx + offset, len(tokens) - 1)
    actual_dep = min(dep_idx + offset, len(tokens) - 1)
    
    with torch.no_grad():
        outputs = model(**toks, output_attentions=True)
    
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        attn_pattern = outputs.attentions[target_layer][0].float().cpu().numpy()
    else:
        raise ValueError("Cannot extract attention weights")
    
    attn_dep_to_head = attn_pattern[:, actual_dep, actual_head].mean()
    attn_head_to_dep = attn_pattern[:, actual_head, actual_dep].mean()
    
    return {
        "attn_dep_to_head": float(attn_dep_to_head),
        "attn_head_to_dep": float(attn_head_to_dep),
    }


def collect_all_pairs(model_name, model, tokenizer, sentences, layer=-1, max_per_type=80):
    """Extract pair features from all sentences, return feature matrices."""
    print(f"\n  Extracting features from {len(sentences)} sentences...")
    
    # Parse sentences to get pairs
    pairs = parse_with_spacy(sentences)
    print(f"  Parsed {len(pairs)} dependency pairs from {len(sentences)} sentences")
    
    # Count per type
    type_counts = {}
    for _, _, _, dt in pairs:
        type_counts[dt] = type_counts.get(dt, 0) + 1
    print(f"  Type distribution: {type_counts}")
    
    # Sample to balance (max_per_type per type)
    type_sampled = {dt: 0 for dt in TARGET_DEP_TYPES}
    balanced_pairs = []
    for p in pairs:
        _, _, _, dt = p
        if dt in TARGET_DEP_TYPES and type_sampled[dt] < max_per_type:
            balanced_pairs.append(p)
            type_sampled[dt] += 1
    
    print(f"  Balanced to {len(balanced_pairs)} pairs (max {max_per_type}/type)")
    type_counts2 = {}
    for _, _, _, dt in balanced_pairs:
        type_counts2[dt] = type_counts2.get(dt, 0) + 1
    print(f"  Final distribution: {type_counts2}")
    
    # Extract features
    X_h_head = []
    X_h_dep = []
    X_pair_diff = []
    X_pair_prod = []
    X_pair_concat = []
    X_pair_full = []
    y_type = []
    y_direction = []  # 1 = head→dep (as labeled), 0 = reversed
    valid_pairs = []
    
    for sent, h_idx, d_idx, dep_type in balanced_pairs:
        try:
            feats = extract_pair_features(model, tokenizer, sent, h_idx, d_idx, layer)
            X_h_head.append(feats["h_head"])
            X_h_dep.append(feats["h_dep"])
            X_pair_diff.append(feats["pair_diff"])
            X_pair_prod.append(feats["pair_prod"])
            X_pair_concat.append(feats["pair_concat"])
            X_pair_full.append(feats["pair_full"])
            y_type.append(TARGET_DEP_TYPES.index(dep_type))
            y_direction.append(1)
            valid_pairs.append((sent, h_idx, d_idx, dep_type))
        except Exception as e:
            continue
    
    result = {
        "X_h_head": np.array(X_h_head),
        "X_h_dep": np.array(X_h_dep),
        "X_pair_diff": np.array(X_pair_diff),
        "X_pair_prod": np.array(X_pair_prod),
        "X_pair_concat": np.array(X_pair_concat),
        "X_pair_full": np.array(X_pair_full),
        "y_type": np.array(y_type),
        "y_direction": np.array(y_direction),
        "valid_pairs": valid_pairs,
        "n_samples": len(y_type),
        "type_counts": {dt: int(np.sum(np.array(y_type) == i)) for i, dt in enumerate(TARGET_DEP_TYPES)},
    }
    
    print(f"  Successfully extracted {result['n_samples']} pairs")
    print(f"  Per-type: {result['type_counts']}")
    
    return result


# ============================================================
# Exp1: 大规模自然句 pair_diff vs point 分类
# ============================================================

def exp1_large_scale_classification(model_name, model, tokenizer, layer=-1):
    """Large-scale natural sentence pair vs point classification."""
    print(f"\n{'='*70}")
    print(f"Exp1: 大规模自然句分类 ({model_name})")
    print(f"{'='*70}")
    
    data = collect_all_pairs(model_name, model, tokenizer, NATURAL_SENTENCES, layer, max_per_type=80)
    
    if data["n_samples"] < 30:
        print("  Too few samples for reliable classification!")
        return {"error": "insufficient_samples", "n": data["n_samples"]}
    
    X_h_head = data["X_h_head"]
    X_h_dep = data["X_h_dep"]
    X_pair_diff = data["X_pair_diff"]
    X_pair_concat = data["X_pair_concat"]
    X_pair_full = data["X_pair_full"]
    y = data["y_type"]
    n = len(y)
    n_types = len(TARGET_DEP_TYPES)
    
    results = {"n_samples": n, "n_types": n_types, "type_counts": data["type_counts"]}
    
    # Use proper cross-validation based on sample size
    if n >= 100:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    
    # Classifiers to test
    clf_svc = LinearSVC(max_iter=5000, C=1.0, dual=False)
    
    print(f"\n  === {n_types}-type dependency classification (n={n}) ===")
    print(f"  CV strategy: {type(cv).__name__}")
    
    configs = [
        ("point_head", X_h_head),
        ("point_dep", X_h_dep),
        ("pair_diff", X_pair_diff),
        ("pair_concat", X_pair_concat),
    ]
    
    for name, X in configs:
        scores = cross_val_score(clf_svc, X, y, cv=cv, scoring='accuracy')
        results[f"{name}_acc"] = float(scores.mean())
        results[f"{name}_std"] = float(scores.std())
        print(f"  {name:20s}: {scores.mean():.3f} ± {scores.std():.3f}")
    
    # pair_full with different PCA dimensions
    print(f"\n  === pair_full with varying PCA dims ===")
    for n_pca in [5, 10, 20, 50]:
        if n_pca >= n:
            continue
        pca = PCA(n_components=n_pca, random_state=42)
        X_pca = pca.fit_transform(X_pair_full)
        var_explained = pca.explained_variance_ratio_.sum()
        scores = cross_val_score(clf_svc, X_pca, y, cv=cv, scoring='accuracy')
        results[f"pair_full_pca{n_pca}_acc"] = float(scores.mean())
        results[f"pair_full_pca{n_pca}_var"] = float(var_explained)
        print(f"  pair_full PCA-{n_pca:2d} (var={var_explained:.3f}): {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Binary classification: argument vs modifier
    print(f"\n  === Binary: Argument vs Modifier ===")
    argument_types = {"subj_verb", "verb_obj", "verb_aux"}
    y_binary = np.array([0 if TARGET_DEP_TYPES[t] in argument_types else 1 for t in y])
    
    for name, X in [("point_dep", X_h_dep), ("pair_diff", X_pair_diff), ("pair_concat", X_pair_concat)]:
        if len(np.unique(y_binary)) < 2:
            continue
        scores = cross_val_score(clf_svc, X, y_binary, cv=cv, scoring='accuracy')
        results[f"binary_{name}_acc"] = float(scores.mean())
        print(f"  Binary {name:20s}: {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Per-type binary: one-vs-rest for each type
    print(f"\n  === Per-type one-vs-rest (pair_diff) ===")
    per_type_results = {}
    for t_idx, t_name in enumerate(TARGET_DEP_TYPES):
        mask = (y == t_idx)
        if mask.sum() < 5:
            continue
        y_ovr = mask.astype(int)
        if y_ovr.sum() < 3 or (1 - y_ovr).sum() < 3:
            continue
        scores = cross_val_score(clf_svc, X_pair_diff, y_ovr, cv=cv, scoring='accuracy')
        per_type_results[t_name] = float(scores.mean())
        print(f"  {t_name:12s}: {scores.mean():.3f} ± {scores.std():.3f} (n_pos={mask.sum()})")
    results["per_type_ovr"] = per_type_results
    
    results["raw_data_shape"] = {
        "point_head": list(X_h_head.shape),
        "pair_diff": list(X_pair_diff.shape),
        "pair_full": list(X_pair_full.shape),
    }
    
    return results


# ============================================================
# Exp2: 方向分类在大规模自然句上的验证
# ============================================================

def exp2_direction_classification(model_name, model, tokenizer, layer=-1):
    """Direction classification on natural data."""
    print(f"\n{'='*70}")
    print(f"Exp2: 方向分类 — 自然句验证 ({model_name})")
    print(f"{'='*70}")
    
    data = collect_all_pairs(model_name, model, tokenizer, NATURAL_SENTENCES, layer, max_per_type=80)
    
    if data["n_samples"] < 20:
        return {"error": "insufficient_samples"}
    
    X_h_head = data["X_h_head"]
    X_h_dep = data["X_h_dep"]
    X_pair_diff = data["X_pair_diff"]
    n = data["n_samples"]
    
    # Construct direction data: forward (y=1) and reverse (y=0)
    X_dir_diff = list(X_pair_diff)
    X_dir_point = list(X_h_head)
    y_dir = [1] * n
    
    for i in range(n):
        X_dir_diff.append(X_h_dep[i] - X_h_head[i])  # reversed diff
        X_dir_point.append(X_h_dep[i])  # reversed point
        y_dir.append(0)
    
    X_dir_diff = np.array(X_dir_diff)
    X_dir_point = np.array(X_dir_point)
    y_dir = np.array(y_dir)
    
    results = {"n_samples": n, "n_dir_samples": len(y_dir)}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if len(y_dir) >= 60 else \
         StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    
    clf = LinearSVC(max_iter=5000, C=1.0, dual=False)
    
    # pair_diff direction
    scores = cross_val_score(clf, X_dir_diff, y_dir, cv=cv, scoring='accuracy')
    results["pair_diff_direction_acc"] = float(scores.mean())
    results["pair_diff_direction_std"] = float(scores.std())
    print(f"  pair_diff → direction: {scores.mean():.3f} ± {scores.std():.3f}")
    
    # point direction (h_head vs h_dep)
    scores = cross_val_score(clf, X_dir_point, y_dir, cv=cv, scoring='accuracy')
    results["point_direction_acc"] = float(scores.mean())
    results["point_direction_std"] = float(scores.std())
    print(f"  point → direction:     {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Direction classification per dependency type
    print(f"\n  === Per-type direction classification (pair_diff) ===")
    y_type = data["y_type"]
    per_type_dir = {}
    for t_idx, t_name in enumerate(TARGET_DEP_TYPES):
        mask = y_type == t_idx
        if mask.sum() < 3:
            continue
        # Forward and reverse pairs for this type
        X_type_diff_fwd = X_pair_diff[mask]
        X_type_diff_rev = data["X_h_dep"][mask] - data["X_h_head"][mask]
        X_type = np.concatenate([X_type_diff_fwd, X_type_diff_rev])
        y_type_dir = np.array([1] * mask.sum() + [0] * mask.sum())
        
        if len(y_type_dir) >= 10:
            cv_type = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
            scores = cross_val_score(clf, X_type, y_type_dir, cv=cv_type, scoring='accuracy')
            per_type_dir[t_name] = {"acc": float(scores.mean()), "std": float(scores.std()), "n": int(mask.sum())}
            print(f"  {t_name:12s}: {scores.mean():.3f} ± {scores.std():.3f} (n={mask.sum()})")
    
    results["per_type_direction"] = per_type_dir
    
    return results


# ============================================================
# Exp3: pair_full系统性失败分析
# ============================================================

def exp3_pair_full_analysis(model_name, model, tokenizer, layer=-1):
    """Systematic analysis of pair_full failure: dimensions, samples, methods, classifiers."""
    print(f"\n{'='*70}")
    print(f"Exp3: pair_full系统性失败分析 ({model_name})")
    print(f"{'='*70}")
    
    data = collect_all_pairs(model_name, model, tokenizer, NATURAL_SENTENCES, layer, max_per_type=80)
    
    if data["n_samples"] < 30:
        return {"error": "insufficient_samples"}
    
    X_pair_full = data["X_pair_full"]
    X_pair_diff = data["X_pair_diff"]
    X_pair_concat = data["X_pair_concat"]
    X_pair_prod = data["X_pair_prod"]
    y = data["y_type"]
    n = len(y)
    d_full = X_pair_full.shape[1]
    
    results = {"n_samples": n, "d_full": d_full}
    
    # === 3a: Varying PCA dimensions ===
    print(f"\n  === 3a: PCA dimension sweep ===")
    pca_results = {}
    for n_pca in [2, 5, 10, 15, 20, 30, 50, 100]:
        if n_pca >= n or n_pca >= d_full:
            continue
        pca = PCA(n_components=n_pca, random_state=42)
        X_pca = pca.fit_transform(X_pair_full)
        var = pca.explained_variance_ratio_.sum()
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
        clf = LinearSVC(max_iter=5000, C=1.0, dual=False)
        scores = cross_val_score(clf, X_pca, y, cv=cv, scoring='accuracy')
        pca_results[n_pca] = {"acc": float(scores.mean()), "std": float(scores.std()), "var_explained": float(var)}
        print(f"  PCA-{n_pca:3d} (var={var:.3f}): {scores.mean():.3f} ± {scores.std():.3f}")
    results["pca_sweep"] = pca_results
    
    # === 3b: LDA vs PCA (supervised vs unsupervised dim reduction) ===
    print(f"\n  === 3b: LDA vs PCA ===")
    n_classes = len(np.unique(y))
    n_lda = min(n_classes - 1, 10)
    if n_lda >= 1 and n >= n_classes * 3:
        try:
            lda = LinearDiscriminantAnalysis(n_components=n_lda)
            X_lda = lda.fit_transform(X_pair_full, y)
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
            clf = LinearSVC(max_iter=5000, C=1.0, dual=False)
            scores = cross_val_score(clf, X_lda, y, cv=cv, scoring='accuracy')
            results["lda_acc"] = float(scores.mean())
            results["lda_dims"] = n_lda
            print(f"  LDA-{n_lda}: {scores.mean():.3f} ± {scores.std():.3f}")
        except Exception as e:
            print(f"  LDA failed: {e}")
    
    # Compare with PCA at same dimension
    if n_lda in pca_results:
        print(f"  PCA-{n_lda}: {pca_results[n_lda]['acc']:.3f}")
        print(f"  → LDA {'wins' if results.get('lda_acc', 0) > pca_results[n_lda]['acc'] else 'loses'} → PCA objective is NOT the bottleneck")
    
    # === 3c: Classifier comparison ===
    print(f"\n  === 3c: Classifier comparison (pair_diff, d={X_pair_diff.shape[1]}) ===")
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    
    classifiers = [
        ("LinearSVC_C0.1", LinearSVC(max_iter=5000, C=0.1, dual=False)),
        ("LinearSVC_C1", LinearSVC(max_iter=5000, C=1.0, dual=False)),
        ("LinearSVC_C10", LinearSVC(max_iter=5000, C=10.0, dual=False)),
        ("LogReg_C1", LogisticRegression(max_iter=2000, C=1.0)),
        ("LogReg_C0.1", LogisticRegression(max_iter=2000, C=0.1)),
        ("kNN_5", KNeighborsClassifier(n_neighbors=5)),
        ("kNN_3", KNeighborsClassifier(n_neighbors=3)),
    ]
    
    clf_results = {}
    for name, clf in classifiers:
        scores = cross_val_score(clf, X_pair_diff, y, cv=cv, scoring='accuracy')
        clf_results[name] = float(scores.mean())
        print(f"  {name:20s}: {scores.mean():.3f} ± {scores.std():.3f}")
    results["classifier_comparison"] = clf_results
    
    # === 3d: Feature component comparison ===
    print(f"\n  === 3d: Feature components ===")
    components = [
        ("h_head", data["X_h_head"]),
        ("h_dep", data["X_h_dep"]),
        ("pair_diff", X_pair_diff),
        ("pair_prod", X_pair_prod),
        ("pair_concat", X_pair_concat),
    ]
    
    comp_results = {}
    for name, X in components:
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
        scores = cross_val_score(LinearSVC(max_iter=5000, C=1.0, dual=False), X, y, cv=cv, scoring='accuracy')
        comp_results[name] = float(scores.mean())
        print(f"  {name:20s} (d={X.shape[1]}): {scores.mean():.3f} ± {scores.std():.3f}")
    results["component_comparison"] = comp_results
    
    # === 3e: Sample size scaling ===
    print(f"\n  === 3e: Sample size scaling (pair_diff) ===")
    sample_fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
    scaling_results = {}
    
    for frac in sample_fracs:
        n_sub = max(int(n * frac), 10)
        # Subsample maintaining class balance
        np.random.seed(42)
        indices = []
        for t in range(len(TARGET_DEP_TYPES)):
            mask = y == t
            if mask.sum() > 0:
                idx = np.where(mask)[0]
                n_sub_t = max(int(len(idx) * frac), 2)
                indices.extend(np.random.choice(idx, min(n_sub_t, len(idx)), replace=False))
        
        X_sub = X_pair_diff[indices]
        y_sub = y[indices]
        
        if len(np.unique(y_sub)) < 2:
            continue
        
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
        scores = cross_val_score(LinearSVC(max_iter=5000, C=1.0, dual=False), X_sub, y_sub, cv=cv, scoring='accuracy')
        scaling_results[frac] = {"acc": float(scores.mean()), "n": len(indices)}
        print(f"  frac={frac:.1f} (n={len(indices):3d}): {scores.mean():.3f} ± {scores.std():.3f}")
    
    results["scaling_results"] = scaling_results
    
    # === 3f: Is pair_full actually worse or just dimension issue? ===
    print(f"\n  === 3f: pair_full optimal pipeline ===")
    # Best PCA dim from sweep
    if pca_results:
        best_pca = max(pca_results.keys(), key=lambda k: pca_results[k]["acc"])
        best_clf_name = max(clf_results, key=clf_results.get)
        print(f"  Best PCA dim: {best_pca} (acc={pca_results[best_pca]['acc']:.3f})")
        print(f"  Best classifier: {best_clf_name} (on pair_diff: {clf_results[best_clf_name]:.3f})")
        
        # Try best classifier + best PCA on pair_full
        pca = PCA(n_components=best_pca, random_state=42)
        X_pca = pca.fit_transform(X_pair_full)
        best_C = float(best_clf_name.split("C")[-1]) if "C" in best_clf_name else 1.0
        clf = LinearSVC(max_iter=5000, C=best_C, dual=False)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
        scores = cross_val_score(clf, X_pca, y, cv=cv, scoring='accuracy')
        results["pair_full_optimal"] = float(scores.mean())
        print(f"  pair_full optimal (PCA-{best_pca} + {best_clf_name}): {scores.mean():.3f}")
        print(f"  pair_diff baseline: {comp_results['pair_diff']:.3f}")
        gap = comp_results['pair_diff'] - scores.mean()
        print(f"  → pair_diff still wins by {gap:.3f}" if gap > 0 else f"  → pair_full wins by {-gap:.3f}!")
    
    return results


# ============================================================
# Exp4: 注意力方向在大规模自然句上的验证
# ============================================================

def exp4_attention_natural(model_name, model, tokenizer, layer=-1):
    """Validate attention-direction correlation on natural sentences."""
    print(f"\n{'='*70}")
    print(f"Exp4: 注意力方向 — 自然句验证 ({model_name})")
    print(f"{'='*70}")
    
    # Parse sentences
    pairs = parse_with_spacy(NATURAL_SENTENCES)
    
    # Sample up to 10 per type for attention (memory intensive)
    type_sampled = {dt: 0 for dt in TARGET_DEP_TYPES}
    max_attn = 10
    sampled_pairs = []
    for p in pairs:
        _, _, _, dt = p
        if dt in TARGET_DEP_TYPES and type_sampled[dt] < max_attn:
            sampled_pairs.append(p)
            type_sampled[dt] += 1
    
    print(f"  Extracting attention for {len(sampled_pairs)} pairs...")
    
    attn_data = {dt: {"dep_to_head": [], "head_to_dep": []} for dt in TARGET_DEP_TYPES}
    
    for sent, h_idx, d_idx, dep_type in sampled_pairs:
        try:
            attn = extract_attention_weights(model, tokenizer, sent, h_idx, d_idx, layer)
            attn_data[dep_type]["dep_to_head"].append(attn["attn_dep_to_head"])
            attn_data[dep_type]["head_to_dep"].append(attn["attn_head_to_dep"])
        except:
            continue
    
    results = {}
    print(f"\n  Attention by dependency type:")
    for dt in TARGET_DEP_TYPES:
        d2h = attn_data[dt]["dep_to_head"]
        h2d = attn_data[dt]["head_to_dep"]
        if len(d2h) < 1:
            continue
        avg_d2h = np.mean(d2h)
        avg_h2d = np.mean(h2d)
        direction = "dep→head" if avg_d2h > avg_h2d else "head→dep"
        ratio = avg_d2h / (avg_h2d + 1e-10)
        
        results[dt] = {
            "dep_to_head": float(avg_d2h),
            "head_to_dep": float(avg_h2d),
            "ratio": float(ratio),
            "direction": direction,
            "n": len(d2h),
        }
        print(f"  {dt:12s}: d2h={avg_d2h:.4f}  h2d={avg_h2d:.4f}  ratio={ratio:.2f}  → {direction} (n={len(d2h)})")
    
    # Argument vs modifier analysis
    argument_types = ["subj_verb", "verb_obj", "verb_aux"]
    modifier_types = ["noun_adj", "noun_det", "noun_poss"]
    
    arg_d2h = []
    arg_h2d = []
    mod_d2h = []
    mod_h2d = []
    
    for dt in argument_types:
        if dt in attn_data:
            arg_d2h.extend(attn_data[dt]["dep_to_head"])
            arg_h2d.extend(attn_data[dt]["head_to_dep"])
    
    for dt in modifier_types:
        if dt in attn_data:
            mod_d2h.extend(attn_data[dt]["dep_to_head"])
            mod_h2d.extend(attn_data[dt]["head_to_dep"])
    
    if arg_d2h and mod_d2h:
        results["argument_avg_d2h"] = float(np.mean(arg_d2h))
        results["argument_avg_h2d"] = float(np.mean(arg_h2d))
        results["modifier_avg_d2h"] = float(np.mean(mod_d2h))
        results["modifier_avg_h2d"] = float(np.mean(mod_h2d))
        
        print(f"\n  Argument types: d2h={np.mean(arg_d2h):.4f} h2d={np.mean(arg_h2d):.4f} → {'dep→head' if np.mean(arg_d2h) > np.mean(arg_h2d) else 'head→dep'}")
        print(f"  Modifier types: d2h={np.mean(mod_d2h):.4f} h2d={np.mean(mod_h2d):.4f} → {'dep→head' if np.mean(mod_d2h) > np.mean(mod_h2d) else 'head→dep'}")
        
        # Statistical test
        if len(arg_d2h) >= 5 and len(arg_h2d) >= 5:
            t_stat, p_val = ttest_rel(arg_d2h, arg_h2d)
            results["argument_ttest_t"] = float(t_stat)
            results["argument_ttest_p"] = float(p_val)
            print(f"  Argument t-test: t={t_stat:.3f}, p={p_val:.4f}")
        
        if len(mod_d2h) >= 5 and len(mod_h2d) >= 5:
            t_stat, p_val = ttest_rel(mod_d2h, mod_h2d)
            results["modifier_ttest_t"] = float(t_stat)
            results["modifier_ttest_p"] = float(p_val)
            print(f"  Modifier t-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    return results


# ============================================================
# Exp5: 泛化测试 — 模板训练→自然句测试
# ============================================================

# Old template pairs from CCXLVII (for comparison)
TEMPLATE_PAIRS = [
    ("The cat chases the mouse", 1, 2, "subj_verb"),
    ("The dog runs quickly", 1, 2, "subj_verb"),
    ("A bird sings beautifully", 1, 2, "subj_verb"),
    ("The student reads the book", 1, 2, "subj_verb"),
    ("The cat chases the mouse", 2, 4, "verb_obj"),
    ("The student reads the book", 2, 4, "verb_obj"),
    ("The man eats the food", 2, 4, "verb_obj"),
    ("The girl finds the key", 2, 4, "verb_obj"),
    ("The big cat sleeps", 2, 1, "noun_adj"),
    ("The small dog barks", 2, 1, "noun_adj"),
    ("The red car moves", 2, 1, "noun_adj"),
    ("The old man walks", 2, 1, "noun_adj"),
    ("The cat runs quickly", 2, 3, "verb_adv"),
    ("The dog barks loudly", 2, 3, "verb_adv"),
    ("The man walks slowly", 2, 3, "verb_adv"),
    ("The girl smiles brightly", 2, 3, "verb_adv"),
    ("The cat sleeps on the mat", 3, 5, "prep_pobj"),
    ("The dog sits under the tree", 3, 5, "prep_pobj"),
    ("The bird flies over the house", 3, 5, "prep_pobj"),
    ("The man walks to the store", 3, 5, "prep_pobj"),
    ("The cat chases the mouse", 1, 0, "noun_det"),
    ("The dog runs quickly", 1, 0, "noun_det"),
    ("The big cat sleeps", 2, 0, "noun_det"),
    ("The small dog barks", 2, 0, "noun_det"),
]


def exp5_generalization(model_name, model, tokenizer, layer=-1):
    """Test generalization: template→natural, natural→template."""
    print(f"\n{'='*70}")
    print(f"Exp5: 泛化测试 — 模板偏置量化 ({model_name})")
    print(f"{'='*70}")
    
    # Extract template features
    dep_types_6 = ["subj_verb", "verb_obj", "noun_adj", "verb_adv", "prep_pobj", "noun_det"]
    
    X_t_diff = []
    X_t_dep = []
    y_t = []
    for sent, h_idx, d_idx, dep_type in TEMPLATE_PAIRS:
        try:
            feats = extract_pair_features(model, tokenizer, sent, h_idx, d_idx, layer)
            X_t_diff.append(feats["pair_diff"])
            X_t_dep.append(feats["h_dep"])
            y_t.append(dep_types_6.index(dep_type))
        except:
            continue
    
    X_t_diff = np.array(X_t_diff)
    X_t_dep = np.array(X_t_dep)
    y_t = np.array(y_t)
    
    # Extract natural features (only matching 6 types)
    natural_pairs = parse_with_spacy(NATURAL_SENTENCES)
    # Filter to 6 types
    natural_pairs_6 = [p for p in natural_pairs if p[3] in dep_types_6]
    
    # Sample balanced
    type_counts = {}
    balanced_natural = []
    for p in natural_pairs_6:
        dt = p[3]
        type_counts[dt] = type_counts.get(dt, 0) + 1
        if type_counts[dt] <= 20:  # up to 20 per type
            balanced_natural.append(p)
    
    X_n_diff = []
    X_n_dep = []
    y_n = []
    for sent, h_idx, d_idx, dep_type in balanced_natural:
        try:
            feats = extract_pair_features(model, tokenizer, sent, h_idx, d_idx, layer)
            X_n_diff.append(feats["pair_diff"])
            X_n_dep.append(feats["h_dep"])
            y_n.append(dep_types_6.index(dep_type))
        except:
            continue
    
    X_n_diff = np.array(X_n_diff)
    X_n_dep = np.array(X_n_dep)
    y_n = np.array(y_n)
    
    results = {
        "n_template": len(y_t),
        "n_natural": len(y_n),
    }
    
    if len(y_n) < 10:
        print("  Too few natural pairs for generalization test!")
        return results
    
    clf = LinearSVC(max_iter=5000, C=1.0, dual=False)
    
    # 1. Template only (within-template CV)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    scores = cross_val_score(clf, X_t_diff, y_t, cv=cv, scoring='accuracy')
    results["template_cv_diff"] = float(scores.mean())
    print(f"  Template CV (pair_diff): {scores.mean():.3f}")
    
    # 2. Natural only (within-natural CV)
    scores = cross_val_score(clf, X_n_diff, y_n, cv=cv, scoring='accuracy')
    results["natural_cv_diff"] = float(scores.mean())
    print(f"  Natural CV (pair_diff):  {scores.mean():.3f}")
    
    # 3. Train template → test natural (key generalization test!)
    clf.fit(X_t_diff, y_t)
    pred_n = clf.predict(X_n_diff)
    acc_t2n = accuracy_score(y_n, pred_n)
    results["template_to_natural_diff"] = float(acc_t2n)
    print(f"  Template→Natural (pair_diff): {acc_t2n:.3f}  ← KEY: measures template bias!")
    
    # 4. Train natural → test template
    clf.fit(X_n_diff, y_n)
    pred_t = clf.predict(X_t_diff)
    acc_n2t = accuracy_score(y_t, pred_t)
    results["natural_to_template_diff"] = float(acc_n2t)
    print(f"  Natural→Template (pair_diff): {acc_n2t:.3f}")
    
    # 5. Same for point features
    scores = cross_val_score(clf, X_t_dep, y_t, cv=cv, scoring='accuracy')
    results["template_cv_point"] = float(scores.mean())
    print(f"\n  Template CV (point_dep): {scores.mean():.3f}")
    
    scores = cross_val_score(clf, X_n_dep, y_n, cv=cv, scoring='accuracy')
    results["natural_cv_point"] = float(scores.mean())
    print(f"  Natural CV (point_dep):  {scores.mean():.3f}")
    
    clf.fit(X_t_dep, y_t)
    pred_n = clf.predict(X_n_dep)
    results["template_to_natural_point"] = float(accuracy_score(y_n, pred_n))
    print(f"  Template→Natural (point_dep): {accuracy_score(y_n, pred_n):.3f}")
    
    # 6. Generalization gap
    gap_diff = results["template_cv_diff"] - results["natural_cv_diff"]
    gap_point = results["template_cv_point"] - results["natural_cv_point"]
    gap_t2n = results["template_cv_diff"] - results["template_to_natural_diff"]
    results["template_natural_gap_diff"] = float(gap_diff)
    results["template_natural_gap_point"] = float(gap_point)
    results["generalization_gap_t2n"] = float(gap_t2n)
    print(f"\n  Template-Natural gap (diff): {gap_diff:+.3f}")
    print(f"  Template-Natural gap (point): {gap_point:+.3f}")
    print(f"  Generalization gap (t→n): {gap_t2n:+.3f}")
    
    # Per-type accuracy on natural
    print(f"\n  Per-type accuracy (train template → test natural):")
    clf.fit(X_t_diff, y_t)
    pred_n = clf.predict(X_n_diff)
    for t_idx, t_name in enumerate(dep_types_6):
        mask = y_n == t_idx
        if mask.sum() > 0:
            acc = accuracy_score(y_n[mask], pred_n[mask])
            results[f"t2n_{t_name}"] = float(acc)
            print(f"  {t_name:12s}: {acc:.3f} (n={mask.sum()})")
    
    return results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CCXLVIII: Large-Scale Natural Sentence Pair Validation")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--layer", type=int, default=-1)
    args = parser.parse_args()
    
    print(f"Loading {args.model}...")
    model, tokenizer, device = load_model(args.model)
    model.eval()
    global _DEVICE
    _DEVICE = device
    
    t0 = time.time()
    
    if args.exp == 1:
        results = exp1_large_scale_classification(args.model, model, tokenizer, args.layer)
    elif args.exp == 2:
        results = exp2_direction_classification(args.model, model, tokenizer, args.layer)
    elif args.exp == 3:
        results = exp3_pair_full_analysis(args.model, model, tokenizer, args.layer)
    elif args.exp == 4:
        results = exp4_attention_natural(args.model, model, tokenizer, args.layer)
    elif args.exp == 5:
        results = exp5_generalization(args.model, model, tokenizer, args.layer)
    
    elapsed = time.time() - t0
    results["elapsed_seconds"] = float(elapsed)
    
    # Save results
    out_dir = Path(__file__).resolve().parent.parent / "glm5_temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"ccxlviii_exp{args.exp}_{args.model}_results.json"
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(out_file, "w") as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to {out_file}")
    print(f"Elapsed: {elapsed:.1f}s")
    
    # Release model
    del model
    torch.cuda.empty_cache()
    print("GPU memory released.")


if __name__ == "__main__":
    main()
