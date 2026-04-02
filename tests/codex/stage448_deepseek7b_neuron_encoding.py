# -*- coding: utf-8 -*-
"""
Stage448: DeepSeek-7B 神经元激活分析
基于CUDA，参考stage423_qwen3_deepseek_wordclass_layer_distribution.py

目标：验证AGI编码机制理论在DeepSeek-7B上的表现
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen3_language_shared import (
    PROJECT_ROOT,
    capture_qwen_mlp_payloads,
    discover_layers,
    move_batch_to_model_device,
    remove_hooks,
)

# ==================== 配置 ====================
DEEPSEEK7B_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage448_deepseek7b_neuron_encoding_20260331"

# 词性类别
WORD_CLASSES = ["noun", "adjective", "verb", "adverb", "pronoun", "preposition"]

# 测试词表 - 扩展到每个词性200个单词
TARGET_LIMITS = {
    "noun": 200,
    "adjective": 200,
    "verb": 200,
    "adverb": 200,
    "pronoun": 48,
    "preposition": 64,
}

CONTROL_LIMITS = {
    "noun": 200,
    "adjective": 200,
    "verb": 200,
    "adverb": 200,
    "pronoun": 200,
    "preposition": 200,
}

TOP_FRACTION = 0.01  # 有效神经元比例
EPS = 1e-8

# ==================== 词汇定义 ====================
PRONOUN_WORDS = {
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves", "this", "that", "these",
    "those", "who", "whom", "whose", "which", "what", "someone", "somebody",
    "something", "anyone", "anybody", "anything", "everyone", "everybody",
    "everything", "nobody", "nothing", "another", "other", "others", "each",
    "either", "neither"
}

PREPOSITION_WORDS = {
    "about", "above", "across", "after", "against", "along", "among", "around",
    "as", "at", "before", "behind", "below", "beneath", "beside", "between",
    "beyond", "by", "despite", "down", "during", "except", "for", "from",
    "in", "inside", "into", "like", "near", "of", "off", "on", "onto", "over",
    "past", "per", "since", "through", "throughout", "to", "toward", "towards",
    "under", "underneath", "until", "up", "upon", "via", "with", "within",
    "without"
}

ADVERB_ALLOWLIST = {
    "not", "never", "always", "often", "sometimes", "maybe", "perhaps",
    "quite", "rather", "very", "almost", "already", "still", "then", "thus",
    "therefore", "however", "instead", "together", "apart", "else"
}

ADJECTIVE_BLOCKLIST = {
    "another", "other", "either", "neither", "every", "some", "any", "many",
    "much", "few", "little"
}

# ==================== 扩展测试词汇 ====================
NOUN_TEST_WORDS = [
    "apple", "banana", "orange", "grape", "mango", "peach", "lemon", "cherry",
    "berry", "melon", "kiwi", "plum", "pear", "fig", "date", "lime", "coconut",
    "walnut", "almond", "pistachio", "chair", "table", "window", "door", "floor",
    "ceiling", "wall", "roof", "house", "car", "bike", "train", "plane", "ship",
    "book", "paper", "pencil", "pen", "computer", "phone", "screen", "keyboard",
    "river", "mountain", "ocean", "forest", "desert", "island", "valley", "lake",
    "ocean", "cloud", "storm", "rain", "snow", "wind", "fire", "earth", "stone",
    "metal", "wood", "glass", "cloth", "silk", "cotton", "gold", "silver", "copper",
    "iron", "steel", "diamond", "ruby", "emerald", "pearl", "amber", "ivory",
    "father", "mother", "brother", "sister", "child", "friend", "teacher",
    "doctor", "lawyer", "engineer", "artist", "writer", "singer", "dancer",
    "king", "queen", "prince", "princess", "soldier", "leader", "judge",
    "dog", "cat", "bird", "fish", "horse", "lion", "tiger", "elephant", "whale",
    "shark", "snake", "eagle", "hawk", "wolf", "bear", "monkey", "rabbit",
    "hand", "foot", "head", "heart", "brain", "eye", "face", "voice", "sound",
    "color", "shape", "size", "weight", "length", "width", "height", "depth",
    "time", "year", "month", "week", "day", "hour", "minute", "second", "moment",
    "water", "milk", "juice", "coffee", "tea", "wine", "beer", "bread", "meat",
    "fish", "rice", "wheat", "corn", "potato", "tomato", "onion", "garlic",
    "salt", "sugar", "butter", "cheese", "oil", "vinegar", "honey", "chocolate",
    "garden", "flower", "rose", "tree", "leaf", "grass", "seed", "root", "branch",
    "dream", "thought", "idea", "memory", "story", "song", "dance", "poem",
    "truth", "justice", "peace", "war", "life", "death", "love", "hope", "fear",
    "anger", "joy", " sorrow", "pain", "gain", "loss", "start", "end", "middle"
]

ADJECTIVE_TEST_WORDS = [
    "beautiful", "ugly", "good", "bad", "big", "small", "large", "tiny",
    "huge", "massive", "heavy", "light", "bright", "dark", "hot", "cold",
    "warm", "cool", "new", "old", "young", "ancient", "modern", "fast",
    "slow", "quick", "rapid", "strong", "weak", "powerful", "gentle",
    "hard", "soft", "rough", "smooth", "sharp", "dull", "thick", "thin",
    "wide", "narrow", "deep", "shallow", "high", "low", "tall", "short",
    "loud", "quiet", "noisy", "silent", "sweet", "sour", "bitter", "salty",
    "fresh", "stale", "clean", "dirty", "dry", "wet", "rich", "poor",
    "happy", "sad", "angry", "calm", "nervous", "brave", "scared", "bold",
    "shy", "kind", "cruel", "fair", "unfair", "just", "unjust", "equal",
    "free", "bound", "open", "closed", "empty", "full", "alive", "dead",
    "healthy", "sick", "safe", "dangerous", "easy", "difficult", "simple",
    "complex", "clear", "vague", "plain", "decorated", "round", "square",
    "straight", "curved", "solid", "liquid", "gas", "visible", "invisible"
]

VERB_TEST_WORDS = [
    "run", "walk", "stand", "sit", "lie", "jump", "fly", "swim", "climb",
    "fall", "rise", "drop", "throw", "catch", "hit", "kick", "push", "pull",
    "carry", "lift", "break", "build", "make", "create", "destroy", "fix",
    "repair", "clean", "wash", "cook", "eat", "drink", "sleep", "wake",
    "dream", "think", "know", "believe", "feel", "see", "hear", "smell",
    "taste", "touch", "look", "watch", "read", "write", "speak", "talk",
    "listen", "ask", "answer", "give", "take", "get", "put", "send", "bring",
    "buy", "sell", "pay", "cost", "owe", "borrow", "lend", "share", "keep",
    "hold", "grasp", "release", "move", "stop", "start", "begin", "end",
    "continue", "finish", "complete", "wait", "leave", "arrive", "come",
    "go", "return", "enter", "exit", "escape", "hide", "find", "lose",
    "search", "explore", "discover", "invent", "learn", "teach", "study",
    "practice", "play", "work", "rest", "help", "hurt", "save", "protect"
]

ADVERB_TEST_WORDS = [
    "quickly", "slowly", "fast", "slow", "rapidly", "swiftly", "suddenly",
    "gradually", "slowly", "carefully", "carelessly", "easily", "difficultly",
    "happily", "sadly", "angrily", "calmly", "nervously", "bravely",
    "shyly", "kindly", "cruelly", "fairly", "unfairly", "clearly", "vaguely",
    "loudly", "quietly", "softly", "highly", "lowly", "deeply", "shallowly",
    "warmly", "coldly", "brightly", "darkly", "neatly", "messily", "well",
    "badly", "almost", "nearly", "already", "still", "yet", "soon", "later",
    "now", "then", "here", "there", "somewhere", "anywhere", "nowhere",
    "always", "never", "sometimes", "often", "rarely", "frequently", "ever",
    "perhaps", "maybe", "certainly", "definitely", "probably", "unlikely"
]

# ==================== 工具函数 ====================
def normalize_word(word: str) -> str:
    return word.strip().lower()


def is_ascii_alpha_word(word: str) -> bool:
    return word.isascii() and word.isalpha()


def classify_word(word: str) -> str | None:
    word = normalize_word(word)
    if word in PRONOUN_WORDS:
        return "pronoun"
    if word in PREPOSITION_WORDS:
        return "preposition"
    return None


def candidate_ok(word: str, class_name: str) -> bool:
    word = normalize_word(word)
    if not word or not is_ascii_alpha_word(word):
        return False
    if len(word) < 3 and class_name not in {"pronoun", "preposition"}:
        return False
    return True


def build_target_words(class_name: str, limit: int) -> List[str]:
    """构建目标词列表"""
    words = []
    if class_name == "noun":
        words = NOUN_TEST_WORDS
    elif class_name == "adjective":
        words = ADJECTIVE_TEST_WORDS
    elif class_name == "verb":
        words = VERB_TEST_WORDS
    elif class_name == "adverb":
        words = ADVERB_TEST_WORDS
    elif class_name == "pronoun":
        words = list(PRONOUN_WORDS)
    elif class_name == "preposition":
        words = list(PREPOSITION_WORDS)

    # 去重并过滤
    seen = set()
    result = []
    for w in words:
        w_norm = normalize_word(w)
        if w_norm and w_norm not in seen and candidate_ok(w_norm, class_name):
            seen.add(w_norm)
            result.append(w_norm)
    return result[:limit]


def build_control_words(target_words: List[str], limit: int) -> Dict[str, List[str]]:
    """构建控制组词（其他词性）"""
    target_set = set(target_words)
    result = {cls: [] for cls in WORD_CLASSES}

    for cls in WORD_CLASSES:
        words = build_target_words(cls, limit)
        result[cls] = [w for w in words if w not in target_set][:limit // 5]

    return result


# ==================== 模型加载 ====================
def set_offline_env() -> None:
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_model(model_path: Path, prefer_cuda: bool = True):
    """加载DeepSeek-7B模型"""
    set_offline_env()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    want_cuda = prefer_cuda and torch.cuda.is_available()
    load_kwargs = {
        "pretrained_model_name_or_path": str(model_path),
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "dtype": torch.bfloat16,
    }
    if want_cuda:
        load_kwargs["device_map"] = "auto"
        print(f"    [CUDA] 使用GPU加速")
    else:
        load_kwargs["device_map"] = "cpu"
        load_kwargs["attn_implementation"] = "eager"
        print(f"    [CPU] 使用CPU模式")

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()
    return model, tokenizer


# ==================== 激活捕获 ====================
def mean_token_activation(layer_tensor: Tensor, attention_mask: Tensor) -> Tensor:
    mask = attention_mask.unsqueeze(-1).to(layer_tensor.dtype)
    lengths = attention_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(layer_tensor.dtype)
    return (layer_tensor * mask).sum(dim=1) / lengths


def capture_all_layers(model):
    """捕获所有MLP层的激活"""
    layer_count = len(discover_layers(model))
    layer_payload_map = {layer_idx: "neuron_in" for layer_idx in range(layer_count)}
    return capture_qwen_mlp_payloads(model, layer_payload_map)


# ==================== 统计分析 ====================
def init_stats(layer_count: int, neuron_count: int):
    shape = (layer_count, neuron_count)
    return {
        "target_count": 0,
        "control_count": 0,
        "target_sum": torch.zeros(shape, dtype=torch.float64),
        "target_sumsq": torch.zeros(shape, dtype=torch.float64),
        "control_sum": torch.zeros(shape, dtype=torch.float64),
        "control_sumsq": torch.zeros(shape, dtype=torch.float64),
    }


def update_stats(stats: Dict, sample_tensor: Tensor, is_target: bool):
    prefix = "target" if is_target else "control"
    stats[f"{prefix}_sum"] += sample_tensor.sum(dim=0)
    stats[f"{prefix}_sumsq"] += (sample_tensor * sample_tensor).sum(dim=0)
    stats[f"{prefix}_count"] += int(sample_tensor.shape[0])


def process_batch(model, tokenizer, buffers, batch_words: List[str], stats: Dict, is_target: bool):
    """处理一批词"""
    encoded = tokenizer(
        batch_words,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    attention_mask_cpu = encoded["attention_mask"].cpu()
    encoded = move_batch_to_model_device(model, encoded)

    with torch.inference_mode():
        model(**encoded, use_cache=False, return_dict=True)

    per_layer_rows = []
    for layer_idx in range(len(buffers)):
        layer_tensor = buffers[layer_idx]
        if layer_tensor is None:
            raise RuntimeError(f"Layer {layer_idx} activation capture failed")
        per_layer_rows.append(mean_token_activation(layer_tensor, attention_mask_cpu).to(torch.float64))

    sample_tensor = torch.stack(per_layer_rows, dim=1)
    update_stats(stats, sample_tensor, is_target=is_target)


def run_scan(model, tokenizer, target_words: List[str], control_words: List[str], batch_size: int):
    """运行完整扫描"""
    buffers, handles = capture_all_layers(model)
    first_layer = discover_layers(model)[0]

    if hasattr(first_layer.mlp, "gate_proj"):
        neuron_count = int(first_layer.mlp.gate_proj.out_features)
    else:
        raise RuntimeError("Cannot identify neuron count from MLP structure")

    stats = init_stats(len(buffers), neuron_count)

    try:
        # 处理目标词
        for start in range(0, len(target_words), batch_size):
            batch = target_words[start:start + batch_size]
            process_batch(model, tokenizer, buffers, batch, stats, is_target=True)

        # 处理控制词
        for start in range(0, len(control_words), batch_size):
            batch = control_words[start:start + batch_size]
            process_batch(model, tokenizer, buffers, batch, stats, is_target=False)
    finally:
        remove_hooks(handles)

    return stats


# ==================== 分析函数 ====================
def build_layer_summary(stats: Dict, top_fraction: float = TOP_FRACTION):
    """构建层统计摘要"""
    target_count = float(stats["target_count"])
    control_count = float(stats["control_count"])

    target_mean = stats["target_sum"] / max(target_count, 1.0)
    control_mean = stats["control_sum"] / max(control_count, 1.0)

    target_var = stats["target_sumsq"] / max(target_count, 1.0) - target_mean * target_mean
    control_var = stats["control_sumsq"] / max(control_count, 1.0) - control_mean * control_mean
    pooled_std = torch.sqrt(((target_var.clamp_min(0.0) + control_var.clamp_min(0.0)) / 2.0).clamp_min(EPS))

    effect = (target_mean - control_mean) / pooled_std
    diff = target_mean - control_mean

    # 综合得分
    pos_gap = (target_mean > 0).float() - (control_mean > 0).float()
    score = 0.7 * torch.clamp(effect / 3.0, 0.0, 1.0) + 0.3 * torch.clamp(pos_gap / 0.5, 0.0, 1.0)

    active_mask = (diff > 0) & (score > 0)
    active_scores = score[active_mask]

    if active_scores.numel() == 0:
        threshold = 0.0
        effective_mask = active_mask
    else:
        quantile = max(0.0, min(1.0, 1.0 - top_fraction))
        threshold = float(torch.quantile(active_scores, quantile).item())
        effective_mask = active_mask & (score >= threshold)

    # 找top神经元
    flat_score = score.flatten()
    top_k = min(50, flat_score.numel())
    top_values, top_indices = torch.topk(flat_score, k=top_k)
    neuron_count = score.shape[1]
    top_neurons = []
    for rank, flat_idx in enumerate(top_indices.tolist(), start=1):
        layer_idx = flat_idx // neuron_count
        neuron_idx = flat_idx % neuron_count
        top_neurons.append({
            "rank": rank,
            "layer_index": int(layer_idx),
            "neuron_index": int(neuron_idx),
            "score": float(score[layer_idx, neuron_idx].item()),
            "effect_size": float(effect[layer_idx, neuron_idx].item()),
        })

    # 层统计
    layer_rows = []
    for layer_idx in range(score.shape[0]):
        layer_score = score[layer_idx]
        layer_effective = effective_mask[layer_idx]
        effective_count = int(layer_effective.to(torch.int64).sum().item())
        effective_fraction = effective_count / max(1, score.shape[1])
        effective_score_sum = float((layer_score * layer_effective.to(layer_score.dtype)).sum().item())

        layer_rows.append({
            "layer_index": int(layer_idx),
            "effective_count": effective_count,
            "effective_fraction": float(effective_fraction),
            "effective_score_sum": effective_score_sum,
            "mean_score": float(layer_score.mean().item()),
            "max_score": float(layer_score.max().item()),
        })

    # 排序
    layer_rows_by_count = sorted(layer_rows, key=lambda x: x["effective_count"], reverse=True)
    layer_rows_by_mass = sorted(layer_rows, key=lambda x: x["effective_score_sum"], reverse=True)

    # 加权质心
    total_mass = sum(row["effective_score_sum"] for row in layer_rows) + EPS
    weighted_center = sum(row["layer_index"] * row["effective_score_sum"] for row in layer_rows) / total_mass

    return {
        "target_count": int(target_count),
        "control_count": int(control_count),
        "layer_count": int(score.shape[0]),
        "neurons_per_layer": int(score.shape[1]),
        "effective_neuron_count": int(effective_mask.to(torch.int64).sum().item()),
        "effective_score_threshold": float(threshold),
        "weighted_layer_center": float(weighted_center),
        "top_layers_by_mass": layer_rows_by_mass[:5],
        "layer_rows": layer_rows,
        "top_neurons": top_neurons[:20],
    }


# ==================== 主分析流程 ====================
def analyze_model(model_key: str = "deepseek7b", batch_size: int = 4, use_cuda: bool = True):
    """分析DeepSeek-7B模型"""
    print(f"\n{'='*60}")
    print(f"  Stage448: DeepSeek-7B 神经元激活分析")
    print(f"{'='*60}")
    print(f"  模型: {DEEPSEEK7B_MODEL_PATH}")
    print(f"  批大小: {batch_size}")
    print(f"  CUDA: {use_cuda and torch.cuda.is_available()}")
    print(f"{'='*60}\n")

    start_time = time.time()

    # 加载模型
    model, tokenizer = load_model(DEEPSEEK7B_MODEL_PATH, prefer_cuda=use_cuda)
    layers = discover_layers(model)

    result = {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "model_path": str(DEEPSEEK7B_MODEL_PATH),
        "layer_count": len(layers),
        "neurons_per_layer": int(layers[0].mlp.gate_proj.out_features) if hasattr(layers[0].mlp, "gate_proj") else None,
        "classes": {},
    }

    try:
        for class_name in WORD_CLASSES:
            print(f"\n[{class_name.upper()}] 分析中...")

            # 获取目标词
            target_words = build_target_words(class_name, TARGET_LIMITS[class_name])
            if not target_words:
                print(f"    [WARNING] 没有找到 {class_name} 目标词")
                continue

            # 获取控制词（排除目标词）
            all_control = []
            for other_class in WORD_CLASSES:
                if other_class != class_name:
                    control_words = build_target_words(other_class, CONTROL_LIMITS[other_class])
                    all_control.extend([w for w in control_words if w not in target_words])
            control_words = list(set(all_control))[:sum(CONTROL_LIMITS.values()) // 5]

            print(f"    目标词数: {len(target_words)}, 控制词数: {len(control_words)}")

            # 运行扫描
            stats = run_scan(model, tokenizer, target_words, control_words, batch_size)

            # 分析结果
            summary = build_layer_summary(stats, TOP_FRACTION)
            summary["target_words_sample"] = target_words[:20]
            result["classes"][class_name] = summary

            print(f"    有效神经元: {summary['effective_neuron_count']}")
            print(f"    质心层: {summary['weighted_layer_center']:.2f}")
            top_layers = [f"L{row['layer_index']}({row['effective_score_sum']:.2f})"
                          for row in summary['top_layers_by_mass'][:3]]
            print(f"    主导层: {', '.join(top_layers)}")

    finally:
        # 释放模型
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    result["elapsed_seconds"] = elapsed

    return result


def build_report(summary: Dict) -> str:
    """生成Markdown报告"""
    lines = [
        "# Stage448: DeepSeek-7B 神经元激活分析报告",
        "",
        f"## 实验配置",
        f"- 时间: {summary.get('timestamp', 'N/A')}",
        f"- 模型: {summary.get('model_name', 'N/A')}",
        f"- 层数: {summary.get('layer_count', 'N/A')}",
        f"- 每层神经元: {summary.get('neurons_per_layer', 'N/A')}",
        f"- 运行时间: {summary.get('elapsed_seconds', 0):.2f}秒",
        "",
        "## 词性层分布分析",
        ""
    ]

    for class_name in WORD_CLASSES:
        if class_name not in summary.get("classes", {}):
            continue
        cls_data = summary["classes"][class_name]
        top_layers = [f"L{row['layer_index']}({row['effective_score_sum']:.3f})"
                      for row in cls_data.get('top_layers_by_mass', [])[:3]]

        lines.extend([
            f"### {class_name}",
            f"- 质心层: **{cls_data.get('weighted_layer_center', 0):.2f}**",
            f"- 有效神经元: {cls_data.get('effective_neuron_count', 0)}",
            f"- 主导层: {', '.join(top_layers)}",
            f"- 样例词: {', '.join(cls_data.get('target_words_sample', [])[:10])}",
            ""
        ])

    # Hub神经元分析
    lines.extend([
        "## Hub神经元分析",
        ""
    ])

    all_hubs = []
    for class_name, cls_data in summary.get("classes", {}).items():
        for neuron in cls_data.get("top_neurons", [])[:5]:
            neuron["class"] = class_name
            all_hubs.append(neuron)

    all_hubs.sort(key=lambda x: x["score"], reverse=True)
    for hub in all_hubs[:15]:
        lines.append(f"- **{hub['class']}** L{hub['layer_index']} N{hub['neuron_index']}: {hub['score']:.4f}")

    return "\n".join(lines)


def save_outputs(summary: Dict, output_dir: Path):
    """保存结果"""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    report_path = output_dir / "REPORT.md"

    summary["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    report_path.write_text(build_report(summary), encoding="utf-8-sig")

    print(f"\n[OK] 结果已保存到: {output_dir}")
    print(f"     - summary.json")
    print(f"     - REPORT.md")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DeepSeek-7B 神经元激活分析")
    parser.add_argument("--batch-size", type=int, default=4, help="批大小")
    parser.add_argument("--cpu", action="store_true", help="强制使用CPU")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    use_cuda = (not args.cpu) and torch.cuda.is_available()

    # 分析模型
    summary = analyze_model(batch_size=args.batch_size, use_cuda=use_cuda)

    # 保存结果
    save_outputs(summary, Path(args.output_dir))

    # 打印摘要
    print(f"\n{'='*60}")
    print("  分析完成!")
    print(f"{'='*60}")
    print(f"模型: {summary['model_name']}")
    print(f"层数: {summary['layer_count']}")
    print(f"每层神经元: {summary['neurons_per_layer']}")
    print(f"总耗时: {summary['elapsed_seconds']:.2f}秒")
    print()

    for class_name in WORD_CLASSES:
        if class_name in summary.get("classes", {}):
            cls_data = summary["classes"][class_name]
            print(f"{class_name:15} 质心层: {cls_data.get('weighted_layer_center', 0):6.2f}  "
                  f"有效神经元: {cls_data.get('effective_neuron_count', 0):6}")


if __name__ == "__main__":
    main()