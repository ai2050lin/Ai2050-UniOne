import json

with open('D:/develop/TransformerLens-main/tests/codex_temp/stage423_qwen3_deepseek_wordclass_layer_distribution_20260330/summary.json', 'r', encoding='utf-8-sig') as f:
    data = json.load(f)

print("=" * 80)
print("Qwen3-4B 词性层分布分析")
print("=" * 80)
print()
for pos in ['noun', 'adjective', 'verb', 'adverb', 'pronoun', 'preposition']:
    pos_data = data['models']['qwen3']['classes'][pos]
    print(f"{pos}:")
    print(f"  质心层: {pos_data['weighted_layer_center']:.2f}")
    print(f"  早中后层占比: {pos_data['early_mid_late_shares']}")
    top_layers = [(item['layer_index'], round(item['effective_fraction'], 4)) for item in pos_data['top_layers_by_count'][:3]]
    print(f"  前3主导层: {top_layers}")
    print()

print("=" * 80)
print("DeepSeek-7B 词性层分布分析")
print("=" * 80)
print()
for pos in ['noun', 'adjective', 'verb', 'adverb', 'pronoun', 'preposition']:
    pos_data = data['models']['deepseek7b']['classes'][pos]
    print(f"{pos}:")
    print(f"  质心层: {pos_data['weighted_layer_center']:.2f}")
    print(f"  早中后层占比: {pos_data['early_mid_late_shares']}")
    top_layers = [(item['layer_index'], round(item['effective_fraction'], 4)) for item in pos_data['top_layers_by_count'][:3]]
    print(f"  前3主导层: {top_layers}")
    print()

print("=" * 80)
print("跨模型对比分析")
print("=" * 80)
print()
for pos in ['noun', 'adjective', 'verb', 'adverb', 'pronoun', 'preposition']:
    qwen3_center = data['models']['qwen3']['classes'][pos]['weighted_layer_center']
    deepseek_center = data['models']['deepseek7b']['classes'][pos]['weighted_layer_center']
    qwen3_eml = data['models']['qwen3']['classes'][pos]['early_mid_late_shares']
    deepseek_eml = data['models']['deepseek7b']['classes'][pos]['early_mid_late_shares']
    
    print(f"{pos}:")
    print(f"  Qwen3质心: {qwen3_center:.2f}, DeepSeek质心: {deepseek_center:.2f}, 差异: {abs(qwen3_center - deepseek_center):.2f}")
    print(f"  Qwen3早中后: {qwen3_eml}")
    print(f"  DeepSeek早中后: {deepseek_eml}")
    print()
