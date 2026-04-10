# stage577 状态方程子状态升级评估

## 核心结论
P_t 需要升级为多子状态，因为拆分后平均表现显著提升；M_t 也需要升级，但不是因为整体更强，而是因为拆分暴露出方式副词强、认识论副词弱、程度/频率高度模型依赖的结构分化。

## 粗粒度 vs 子状态
- coarse_pronoun_mean: `0.7500`
- split_pronoun_mean: `0.9167`
- pronoun_gain_mean: `0.1667`
- coarse_adverb_mean: `0.6250`
- split_adverb_mean: `0.5625`
- adverb_gain_mean: `-0.0625`

## 子状态均值
- `pronoun_personal`: `0.7500`
- `pronoun_reflexive`: `1.0000`
- `pronoun_demonstrative`: `1.0000`
- `adverb_manner`: `0.8750`
- `adverb_epistemic`: `0.3750`
- `adverb_degree`: `0.5000`
- `adverb_frequency`: `0.5000`

## 升级后的状态方程
- old: `S_t,l = (O_t,l, A_t,l, R_t,l, P_t,l, M_t,l, Q_t,l, G_t,l, C_t,l)`
- new: `S_t,l = (O_t,l, A_t,l, R_t,l, P_personal,l, P_reflexive,l, P_demonstrative,l, M_manner,l, M_epistemic,l, M_degree,l, M_frequency,l, Q_t,l, G_t,l, C_t,l)`
