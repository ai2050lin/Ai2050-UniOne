# stage576 指代/修饰子状态拆分实测

## 核心结论
本实验检验把 P_t 拆成人称/反身/指示，把 M_t 拆成方式/认识论/程度/频率后，是否能更稳定地承载代词与副词机制。若成立，统一状态更新方程应升级为多子状态版本。

## Qwen3-4B
- pronoun_split_mean_accuracy: `1.0000`
- adverb_split_mean_accuracy: `0.8750`
- `pronoun_personal`: `1.0000`
- `pronoun_reflexive`: `1.0000`
- `pronoun_demonstrative`: `1.0000`
- `adverb_manner`: `1.0000`
- `adverb_epistemic`: `1.0000`
- `adverb_degree`: `0.5000`
- `adverb_frequency`: `1.0000`

## DeepSeek-R1-Distill-Qwen-7B
- pronoun_split_mean_accuracy: `0.8333`
- adverb_split_mean_accuracy: `0.5000`
- `pronoun_personal`: `0.5000`
- `pronoun_reflexive`: `1.0000`
- `pronoun_demonstrative`: `1.0000`
- `adverb_manner`: `0.5000`
- `adverb_epistemic`: `0.0000`
- `adverb_degree`: `1.0000`
- `adverb_frequency`: `0.5000`

## GLM-4-9B-Chat-HF
- pronoun_split_mean_accuracy: `1.0000`
- adverb_split_mean_accuracy: `0.6250`
- `pronoun_personal`: `1.0000`
- `pronoun_reflexive`: `1.0000`
- `pronoun_demonstrative`: `1.0000`
- `adverb_manner`: `1.0000`
- `adverb_epistemic`: `0.5000`
- `adverb_degree`: `0.5000`
- `adverb_frequency`: `0.5000`

## Gemma-4-E2B-it
- pronoun_split_mean_accuracy: `0.8333`
- adverb_split_mean_accuracy: `0.2500`
- `pronoun_personal`: `0.5000`
- `pronoun_reflexive`: `1.0000`
- `pronoun_demonstrative`: `1.0000`
- `adverb_manner`: `1.0000`
- `adverb_epistemic`: `0.0000`
- `adverb_degree`: `0.0000`
- `adverb_frequency`: `0.0000`

