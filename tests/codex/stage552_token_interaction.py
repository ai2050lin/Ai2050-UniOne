"""
stage552: Token交互分析
目标：分析Token间的注意力交互如何影响编码
- Attention权重分析：在句式中哪些token关注哪些token
- 多token词的编码组合：如"screwdriver"由subword组成，编码如何从subword涌现
- 歧义词的消歧机制："bank"（河岸vs银行）的attention模式差异

使用Qwen3。
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_language_shared import load_qwen3_model, discover_layers
from multimodel_language_shared import encode_to_device, evenly_spaced_layers, free_model

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'glm5', 'temp')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 多token词分析
MULTITOKEN_WORDS = {
    "screwdriver": "a screwdriver fixes screws",
    "university": "the university teaches students",
    "hospital": "the hospital treats patients",
    "cherry": "a cherry is red and small",
    "freedom": "freedom is important for all",
    "banana": "a banana is yellow and sweet",
}

# 歧义词分析
AMBIGUOUS_WORDS = {
    "bank": {
        "river": ["river bank", "the bank of the river", "sat on the muddy bank"],
        "finance": ["the bank gave a loan", "went to the bank today", "bank account"],
    },
    "apple": {
        "fruit": ["red apple", "eat an apple", "apple is a fruit"],
        "company": ["Apple iPhone", "Apple stock price", "Apple CEO announced"],
    },
    "light": {
        "illumination": ["turn on the light", "the light is bright", "light the candle"],
        "weight": ["the box is light", "a light backpack", "light as a feather"],
    },
}

# 修饰语影响分析
MODIFIER_EXPERIMENTS = {
    "apple": [
        ("red apple", "apple", "red"),
        ("green apple", "apple", "green"),
        ("big apple", "apple", "big"),
        ("small apple", "apple", "small"),
        ("delicious apple", "apple", "delicious"),
        ("rotten apple", "apple", "rotten"),
    ],
    "water": [
        ("hot water", "water", "hot"),
        ("cold water", "water", "cold"),
        ("clean water", "water", "clean"),
        ("dirty water", "water", "dirty"),
        ("deep water", "water", "deep"),
    ],
}


def analyze_attention(model, tokenizer, sentence, sample_layers):
    """提取注意力权重"""
    encoded = encode_to_device(model, tokenizer, sentence)
    input_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, output_attentions=True)
    
    attn_data = {}
    for li in sample_layers:
        # attn shape: [batch, n_heads, seq, seq]
        attn = outputs.attentions[li]  # layer index in attentions list
        # 平均所有head
        avg_attn = attn[0].mean(dim=0).cpu().numpy()  # [seq, seq]
        attn_data[str(li)] = avg_attn
    
    return tokens, attn_data, outputs.hidden_states


def main():
    print(f"{'='*60}")
    print(f"  stage552: Token交互分析 (Qwen3)")
    print(f"{'='*60}")

    model, tokenizer = load_qwen3_model()
    layers = discover_layers(model)
    n_layers = len(layers)
    sample_layers = evenly_spaced_layers(model, count=10)
    print(f"  层数: {n_layers}, 采样: {sample_layers}")

    # ========== 实验1: 多token词的subword组合 ==========
    print(f"\n{'='*60}")
    print(f"  [实验1] 多token词的Subword编码分析")
    print(f"{'='*60}")

    multi_results = {}
    for word, sentence in MULTITOKEN_WORDS.items():
        print(f"\n  --- {word} in \"{sentence}\" ---")
        encoded = encode_to_device(model, tokenizer, word)
        word_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
        word_tokens_safe = [t.encode('ascii', 'replace').decode() for t in word_tokens]
        n_subwords = len(word_tokens)
        print(f"    Subwords: {word_tokens_safe} ({n_subwords} tokens)")

        sent_encoded = encode_to_device(model, tokenizer, sentence)
        sent_tokens = tokenizer.convert_ids_to_tokens(sent_encoded["input_ids"][0].tolist())
        
        with torch.no_grad():
            outputs = model(**sent_encoded, output_hidden_states=True)
        
        # 找word的token在sentence中的位置
        word_token_ids = encoded["input_ids"][0].tolist()
        sent_ids = sent_encoded["input_ids"][0].tolist()
        word_positions = []
        for i, sid in enumerate(sent_ids):
            if sid in word_token_ids:
                word_positions.append(i)
        # 去重并连续化
        if word_positions:
            start = word_positions[0]
            end = min(start + n_subwords - 1, len(sent_tokens) - 1)
        else:
            start = end = len(sent_tokens) - 1
        
        sent_tokens_safe = [t.encode('ascii', 'replace').decode() for t in sent_tokens]
        print(f"    pos: {start}-{end}, tokens: {sent_tokens_safe}")

        word_data = {}
        for li in sample_layers:
            hs = outputs.hidden_states[li + 1]  # [batch, seq, dim]
            
            # 各subword的编码
            subword_encs = []
            for pos in range(start, end + 1):
                subword_encs.append(hs[0, pos, :].float().cpu())
            
            # 平均池化（模型实际使用的）
            mean_pool = torch.stack(subword_encs).mean(dim=0)
            # 最后一个subword
            last_subword = hs[0, end, :].float().cpu()
            # 第一个subword
            first_subword = hs[0, start, :].float().cpu()
            
            # subword间距离
            if len(subword_encs) > 1:
                intra_dists = []
                for a in range(len(subword_encs)):
                    for b in range(a + 1, len(subword_encs)):
                        d = 1 - F.cosine_similarity(subword_encs[a].unsqueeze(0), subword_encs[b].unsqueeze(0)).item()
                        intra_dists.append(d)
                mean_intra = np.mean(intra_dists)
            else:
                mean_intra = 0.0
            
            word_data[str(li)] = {
                "mean_pool_vs_last": round(1 - F.cosine_similarity(mean_pool.unsqueeze(0), last_subword.unsqueeze(0)).item(), 6),
                "first_vs_last": round(1 - F.cosine_similarity(first_subword.unsqueeze(0), last_subword.unsqueeze(0)).item(), 6),
                "mean_intra_dist": round(float(mean_intra), 6),
            }
        
        # 打印首末层
        for li in [sample_layers[0], sample_layers[-1]]:
            d = word_data[str(li)]
            print(f"    L{li}: pool-vs-last={d['mean_pool_vs_last']:.4f}, first-vs-last={d['first_vs_last']:.4f}, intra={d['mean_intra_dist']:.4f}")
        
        multi_results[word] = {"subwords": word_tokens, "n_subwords": n_subwords, "layer_data": word_data}

    # ========== 实验2: 歧义词消歧 ==========
    print(f"\n{'='*60}")
    print(f"  [实验2] 歧义词消歧分析")
    print(f"{'='*60}")

    ambiguous_results = {}
    for word, senses in AMBIGUOUS_WORDS.items():
        print(f"\n  --- {word} ---")
        word_sense_data = {}
        
        for sense_name, sentences in senses.items():
            print(f"    [{sense_name}]:")
            sense_encs = {}
            for li in sample_layers:
                encs = []
                for sent in sentences:
                    enc, pos = _get_word_encoding(model, tokenizer, sent, word, li)
                    encs.append(enc.float().cpu())
                sense_encs[li] = torch.stack(encs)
            
            # 末层
            last_li = sample_layers[-1]
            sense_center = sense_encs[last_li].mean(dim=0)
            word_sense_data[sense_name] = {
                "center": sense_center,
                "sentences": sentences,
            }
            print(f"      center_norm={float(torch.norm(sense_center)):.2f}")
        
        # 意义间距离
        sense_names = list(senses.keys())
        for si, s1 in enumerate(sense_names):
            for sj, s2 in enumerate(sense_names):
                if sj <= si:
                    continue
                d = 1 - F.cosine_similarity(
                    word_sense_data[s1]["center"].unsqueeze(0),
                    word_sense_data[s2]["center"].unsqueeze(0)
                ).item()
                print(f"    {s1} vs {s2}: cos_dist={d:.4f}")
        
        # 逐层意义分离度
        print(f"    逐层意义分离度 (cos_dist between sense centers):")
        for li in sample_layers:
            centers = []
            for sense_name in sense_names:
                encs = []
                for sent in senses[sense_name]:
                    enc, _ = _get_word_encoding(model, tokenizer, sent, word, li)
                    encs.append(enc.float().cpu())
                centers.append(torch.stack(encs).mean(dim=0))
            d = 1 - F.cosine_similarity(centers[0].unsqueeze(0), centers[1].unsqueeze(0)).item()
            print(f"      L{li}: {d:.4f}")
        
        ambiguous_results[word] = word_sense_data

    # ========== 实验3: 修饰语影响 ==========
    print(f"\n{'='*60}")
    print(f"  [实验3] 修饰语对目标词编码的影响")
    print(f"{'='*60}")

    modifier_results = {}
    for word, pairs in MODIFIER_EXPERIMENTS.items():
        print(f"\n  --- {word} ---")
        
        # 裸词编码
        bare_enc, _ = _get_word_encoding(model, tokenizer, word, word, sample_layers[-1])
        
        print(f"    修饰句 vs 裸词 (末层 cos_dist, L2):")
        word_mod_data = []
        for full_sent, bare_word, modifier in pairs:
            mod_enc, _ = _get_word_encoding(model, tokenizer, full_sent, bare_word, sample_layers[-1])
            cos_d = 1 - F.cosine_similarity(mod_enc.float().unsqueeze(0), bare_enc.float().unsqueeze(0)).item()
            l2 = float(torch.norm(mod_enc.float() - bare_enc.float()))
            print(f"      {modifier:>12}: cos_d={cos_d:.4f}, L2={l2:.2f}")
            word_mod_data.append({"modifier": modifier, "cos_dist": round(cos_d, 6), "l2": round(l2, 4)})
        
        modifier_results[word] = word_mod_data

    # 保存
    result = {
        "multitoken": {w: {"subwords": d["subwords"], "n_subwords": d["n_subwords"], "layer_data": d["layer_data"]} for w, d in multi_results.items()},
        "modifier_effects": modifier_results,
    }
    out_path = os.path.join(OUTPUT_DIR, "stage552_token_interaction.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  保存到: {out_path}")

    free_model(model)


def _get_word_encoding(model, tokenizer, sentence, target_word, layer_idx):
    """获取句中目标词的编码"""
    encoded = encode_to_device(model, tokenizer, sentence)
    input_ids = encoded["input_ids"]
    target_tokens = tokenizer.encode(target_word, add_special_tokens=False)
    target_pos = None
    for tid in target_tokens:
        matches = (input_ids[0] == tid).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            target_pos = matches[0].item()
            break
    if target_pos is None:
        target_pos = input_ids.shape[1] - 1
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    hs = outputs.hidden_states[layer_idx + 1]
    return hs[0, target_pos, :], target_pos


if __name__ == "__main__":
    main()
