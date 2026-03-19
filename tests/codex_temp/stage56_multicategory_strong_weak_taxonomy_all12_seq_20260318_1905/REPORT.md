# Stage56 Multicategory Strong Weak Taxonomy Report

- Case count: 72
- Model count: 3
- Category count: 12
- Weak bridge positive count: 51
- Strong core dominant count: 18

## By Model
- Qwen/Qwen3-4B: cases=24 / weak_bridge_positive=10 / mean_best_strong=0.038236 / mean_best_weak=0.005600 / mean_best_mixed=0.045845
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B: cases=25 / weak_bridge_positive=21 / mean_best_strong=0.046146 / mean_best_weak=0.037600 / mean_best_mixed=0.069056
- zai-org/GLM-4-9B-Chat-HF: cases=23 / weak_bridge_positive=20 / mean_best_strong=0.092250 / mean_best_weak=0.072415 / mean_best_mixed=0.166612

## Top Cases
- seq_glm / fruit / proto=melon / inst=papaya / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top1:0.242851 / best_weak=weak_full:0.672753 / best_mixed=union_full:1.051960
- seq_glm / animal / proto=dog / inst=dog / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_full:0.526850 / best_weak=weak_top2:-0.006906 / best_mixed=mix_top2_plus_536592:0.582884
- seq_glm / food / proto=milk / inst=milk / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top3:0.160890 / best_weak=weak_full:0.295212 / best_mixed=mix_pair_507604_525253:0.454988
- seq_glm / fruit / proto=melon / inst=melon / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top3:0.194176 / best_weak=weak_top1:0.195563 / best_mixed=union_full:0.345749
- seq_deepseek / food / proto=milk / inst=fishcake / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top2:0.100721 / best_weak=weak_top2:0.013047 / best_mixed=mix_top1_plus_507754:0.280741
- seq_deepseek / abstract / proto=love / inst=love / dominant=weak_dominant / role=weak_dominant_positive / best_strong=strong_full:0.291451 / best_weak=weak_top2:0.295736 / best_mixed=mix_top2_plus_512063:0.279576
- seq_glm / animal / proto=dog / inst=rhinoceros / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top3:0.216596 / best_weak=weak_full:0.149201 / best_mixed=mix_top2_plus_525253:0.251378
- seq_glm / weather / proto=cyclone / inst=dew / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top3:0.148531 / best_weak=weak_top2:0.100691 / best_mixed=mix_top2_plus_443181:0.233571
- seq_qwen / abstract / proto=meaning / inst=glory / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_full:0.173307 / best_weak=weak_top1:0.071669 / best_mixed=union_full:0.229068
- seq_glm / abstract / proto=patience / inst=patience / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top2:0.142276 / best_weak=weak_full:0.104494 / best_mixed=mix_top1_plus_437938:0.199069
- seq_qwen / animal / proto=bee / inst=dog / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top2:0.109448 / best_weak=weak_top1:-0.005274 / best_mixed=mix_pair_293707_330763:0.158400
- seq_deepseek / weather / proto=frost / inst=cyclone / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top3:0.096265 / best_weak=weak_top1:0.134920 / best_mixed=mix_pair_512089_509218:0.141067
- seq_deepseek / celestial / proto=jupiter / inst=moon / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top1:0.073855 / best_weak=weak_full:0.176441 / best_mixed=mix_top2_plus_507754:0.139692
- seq_glm / abstract / proto=patience / inst=meaning / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top1:0.106701 / best_weak=weak_full:0.010779 / best_mixed=mix_top2_plus_437938:0.121418
- seq_glm / food / proto=milk / inst=cabbage / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_full:-0.005575 / best_weak=weak_top2:-0.010468 / best_mixed=mix_top2_plus_483022:0.113989
- seq_deepseek / celestial / proto=jupiter / inst=moonlight / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top1:0.077518 / best_weak=weak_top2:0.035552 / best_mixed=mix_top1_plus_516947:0.113198
- seq_qwen / fruit / proto=papaya / inst=papaya / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top2:0.094997 / best_weak=weak_top2:0.009552 / best_mixed=mix_top2_plus_291842:0.103746
- seq_glm / celestial / proto=jupiter / inst=moon / dominant=strong_core_dominant / role=weak_drag_or_conflict / best_strong=strong_full:0.147774 / best_weak=weak_full:0.007555 / best_mixed=union_full:0.093389
- seq_qwen / fruit / proto=papaya / inst=watermelon / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_top3:0.076565 / best_weak=weak_full:0.011033 / best_mixed=union_full:0.091425
- seq_qwen / object / proto=bottle / inst=camera / dominant=bridge_dominant / role=weak_bridge_positive / best_strong=strong_full:0.084845 / best_weak=weak_full:0.021101 / best_mixed=union_full:0.089197
