# stage580 定向瓶颈评估与状态方程升级

## 核心结论
P_personal 必须拆成句内局部绑定和跨句链路两层状态。M_epistemic 更像和 Q_certainty 强耦合的判断状态，而不只是修饰范围。 这说明统一语言理论要继续压缩大兜底项，把跨句链路和确定性判断从粗粒度状态里拆出来。

## 平均值
- simple_personal_mean: `0.7500`
- discourse_personal_mean: `0.6250`
- personal_gap_mean: `-0.1250`
- simple_epistemic_mean: `0.3750`
- targeted_epistemic_mean: `0.7188`
- epistemic_gap_mean: `0.3438`

## 理论升级
- previous: `S_t,l = (O_t,l, A_t,l, R_t,l, P_personal,l, P_reflexive,l, P_demonstrative,l, M_manner,l, M_epistemic,l, M_degree,l, M_frequency,l, Q_t,l, G_t,l, C_t,l)`
- candidate_next: `S_t,l = (O_t,l, A_t,l, R_t,l, P_local,l, P_discourse,l, P_reflexive,l, P_demonstrative,l, M_manner,l, M_epistemic_scope,l, M_degree,l, M_frequency,l, Q_certainty,l, Q_reasoning,l, G_t,l, C_t,l)`
