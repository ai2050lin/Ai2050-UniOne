# 编码回路到稳态区预测报告

- case_count: 6
- match_ratio: 0.666667

## 个案
- openwebtext_true_long: predicted=高平衡区, actual=过渡区, match=False
- openwebtext_longterm: predicted=高平衡区, actual=高平衡区, match=True
- openwebtext_persistent: predicted=高风险区, actual=高风险区, match=True
- openwebtext_extended: predicted=高平衡区, actual=高平衡区, match=True
- qwen3_4b_recovery_chain: predicted=过渡区, actual=过渡区, match=True
- deepseek_7b_recovery_chain: predicted=过渡区, actual=高风险区, match=False
