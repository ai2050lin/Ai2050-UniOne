# Stage56 Multicategory Strong Weak Taxonomy Report

- Case count: 20
- Model count: 3
- Category count: 8
- Weak bridge positive count: 15
- Strong core dominant count: 5

## By Model
- Qwen/Qwen3-4B: cases=8 / weak_bridge_positive=8 / mean_best_strong=0.008811 / mean_best_weak=0.004893 / mean_best_mixed=0.026703
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B: cases=8 / weak_bridge_positive=4 / mean_best_strong=0.025622 / mean_best_weak=0.011899 / mean_best_mixed=0.047834
- zai-org/GLM-4-9B-Chat-HF: cases=4 / weak_bridge_positive=3 / mean_best_strong=0.249892 / mean_best_weak=0.084965 / mean_best_mixed=0.387942

## Top Cases
- glm_fruit / food / proto=food / inst=bread / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top2:1.002552 / best_weak=weak_top1:0.275158 / best_mixed=mix_top2_plus_456343:1.126538
- glm_fruit / fruit / proto=fruit / inst=banana / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top3:-0.091971 / best_weak=weak_top2:0.027830 / best_mixed=mix_top1_plus_402716:0.374746
- deepseek_fruit / food / proto=food / inst=pizza / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top1:0.150754 / best_weak=weak_top1:0.054909 / best_mixed=mix_top1_plus_514194:0.280210
- qwen_fruit / food / proto=food / inst=bread / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_full:0.019626 / best_weak=weak_top2:-0.009923 / best_mixed=union_full:0.096515
- qwen_real / animal / proto=animal / inst=rabbit / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top3:0.006518 / best_weak=weak_top1:0.033174 / best_mixed=union_full:0.056648
- glm_fruit / object / proto=object / inst=bowl / dominant=strong_core_dominant / role=weak_drag_or_conflict / best_strong=strong_full:0.088105 / best_weak=weak_full:0.036405 / best_mixed=mix_top1_plus_416995:0.049202
- qwen_fruit / fruit / proto=fruit / inst=apple / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_full:0.034519 / best_weak=weak_full:0.013358 / best_mixed=mix_pair_243285_292087:0.045026
- deepseek_fruit / fruit / proto=fruit / inst=apple / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_full:-0.009978 / best_weak=weak_top1:-0.003751 / best_mixed=mix_top2_plus_418737:0.039355
- deepseek_real / animal / proto=animal / inst=rabbit / dominant=strong_core_dominant / role=weak_drag_or_conflict / best_strong=strong_top1:0.049804 / best_weak=weak_top2:0.042377 / best_mixed=mix_top1_plus_483084:0.035708
- deepseek_real / vehicle / proto=vehicle / inst=car / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_full:0.002605 / best_weak=weak_full:-0.002670 / best_mixed=union_full:0.014427
- qwen_real / human / proto=human / inst=teacher / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_full:0.009222 / best_weak=weak_top1:0.002182 / best_mixed=mix_pair_252928_311785:0.014309
- deepseek_real / human / proto=human / inst=librarian / dominant=strong_core_dominant / role=weak_drag_or_conflict / best_strong=strong_full:0.005402 / best_weak=weak_top1:-0.000461 / best_mixed=mix_top2_plus_418737:0.004370
- deepseek_fruit / nature / proto=nature / inst=tree / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_full:0.001301 / best_weak=weak_top2:0.000958 / best_mixed=mix_pair_508497_495880:0.003966
- deepseek_real / tech / proto=tech / inst=database / dominant=strong_core_dominant / role=weak_drag_or_conflict / best_strong=strong_top1:0.004205 / best_weak=weak_top1:0.003399 / best_mixed=mix_top1_plus_398098:0.003824
- glm_fruit / nature / proto=nature / inst=river / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top2:0.000883 / best_weak=weak_top1:0.000468 / best_mixed=mix_top2_plus_525279:0.001281
- deepseek_fruit / object / proto=object / inst=bowl / dominant=strong_core_dominant / role=weak_drag_or_conflict / best_strong=strong_top2:0.000884 / best_weak=weak_top1:0.000429 / best_mixed=mix_pair_515554_513264:0.000809
- qwen_real / vehicle / proto=vehicle / inst=cart / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top2:0.000436 / best_weak=weak_top1:-0.000035 / best_mixed=mix_top2_plus_312078:0.000595
- qwen_real / tech / proto=tech / inst=database / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top2:0.000244 / best_weak=weak_top1:0.000401 / best_mixed=mix_top2_plus_340801:0.000348
- qwen_fruit / object / proto=object / inst=bowl / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top2:0.000001 / best_weak=weak_top2:0.000003 / best_mixed=mix_pair_333090_341179:0.000137
- qwen_fruit / nature / proto=nature / inst=tree / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_full:-0.000079 / best_weak=weak_top2:-0.000019 / best_mixed=mix_pair_253034_272390:0.000048
