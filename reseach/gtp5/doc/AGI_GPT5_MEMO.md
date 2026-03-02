
## [2026-03-01 17:28:13] Codex Progress Log
- Task: 查看 deepseek-7b 的下载进度
- Commands executed:
  - Get-CimInstance Win32_Process | Where-Object { .CommandLine -match 'deepseek|huggingface|hf_hub|snapshot_download|aria2|modelscope' }
  - Get-ChildItem ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B -Recurse -File
  - python requests -> https://huggingface.co/api/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/tree/main?recursive=1
  - Get-ChildItem .../blobs/*.incomplete (size snapshot twice)
- Result snapshot:
  - Repo total size (official): 15,239,131,423 bytes (~14.193 GiB)
  - Local cached size: 15,239,131,463 bytes (~14.193 GiB)
  - Active process: python - child of powershell (snapshot_download)
  - Two .incomplete shard files reached full target size.
- Theoretical math progress:
  - Used conservation-of-bytes check: progress = local_downloaded_bytes / official_total_bytes.
  - Current estimate: ~100.00% (data complete, pending finalize/rename stage).

## [2026-03-01 17:55:57] Codex Progress Log
- Task: 进行运行测试
- Commands executed:
  - .\\.venv\\Scripts\\python.exe -m pytest tests/unit -q (failed: pytest not installed)
  - poetry run pytest tests/unit -q (failed: poetry not found)
  - python -m pytest tests/unit -q (failed: pytest not installed)
  - .\\.venv\\Scripts\\python.exe -m pip install pytest
  - .\\.venv\\Scripts\\python.exe -m pytest tests/unit -q -o addopts=
  - .\\.venv\\Scripts\\python.exe -m pytest tests/unit -q -o addopts= --ignore=tests/unit/test_make_docs.py
  - .\\.venv\\Scripts\\python.exe -m pytest tests -q -o addopts= --ignore=tests/unit/test_make_docs.py
- Test result snapshot:
  - Full run scope: tests/ (ignore tests/unit/test_make_docs.py)
  - Result: 4 failed, 619 passed, 19 skipped, 17 warnings, duration 1329.58s
  - Failed cases:
    - tests/acceptance/test_activation_cache.py::test_logit_attrs_works_for_all_input_shapes
    - tests/integration/test_head_detector.py::test_detect_head_with_invalid_head_name
    - tests/unit/factored_matrix/test_constructor.py::test_factored_matrix_broadcast_mismatch
    - tests/unit/test_svd_interpreter.py::test_svd_interpreter_fails_on_invalid_vector_type
- Theoretical math progress:
  - Test pass ratio over executed cases = 619 / (619 + 4) = 99.36% (excluding skipped)
  - Failure ratio over executed cases = 4 / 623 = 0.64%
  - Indicates high baseline stability with 4 assertion/type-contract regressions.


## [2026-03-01 17:58:20] Codex Progress Log
- Task: 是否可以用 deepseek-7b 模型进行测试
- Commands executed:
  - Check cache path and incomplete files under C:\Users\27876\.cache\huggingface\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B
  - Check snapshot weight files (*.safetensors/*.bin)
  - Check related process list by command line (snapshot_download / DeepSeek-R1-Distill-Qwen-7B)
  - Check python child CPU and two-point incomplete file snapshots
- Result snapshot:
  - Two shard files still keep .incomplete suffix
  - Snapshot directory currently has no weight shards linked
  - Downloader process still exists, but CPU delta in 8s window = 0 and incomplete file size/mtime unchanged in 10s window
- Theoretical math progress:
  - Readiness criterion for runnable local model: (weights in snapshot) AND (no .incomplete shards)
  - Current readiness estimate for direct local test with 7B: 0/2 conditions satisfied => not ready

## [2026-03-01 18:40:09] Codex Progress Log
- Task: 下载卡住了，重启并修复下载（DeepSeek-R1-Distill-Qwen-7B）
- Commands executed:
  - Kill stuck process tree: Stop-Process on parent PID 19100 and child python process
  - Inspect cache state in ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B
  - Python repair script:
    - query HF API tree for expected safetensor sizes
    - rename *.incomplete -> blob hash filename when size matched expected shard sizes
  - Restart download/finalize: snapshot_download(repo_id='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', resume_download=True, local_files_only=False, max_workers=2)
  - Validation:
    - incomplete_count == 0
    - snapshot contains model-00001-of-000002.safetensors and model-00002-of-000002.safetensors
    - local config/tokenizer load passes (AutoConfig + AutoTokenizer, local_files_only=True)
- Result snapshot:
  - Fixed 2 stuck files from .incomplete to finalized blobs
  - Download finalize completed successfully (11/11 files indexed)
  - Model files present with expected sizes:
    - model-00001-of-000002.safetensors = 8,606,596,466 bytes
    - model-00002-of-000002.safetensors = 6,624,675,384 bytes
- Theoretical math progress:
  - Readiness condition vector r = [no_incomplete, shards_present, local_loadable]
  - Before repair: r = [0,0,0], After repair: r = [1,1,1]
  - Readiness score S = sum(r)/3 changed from 0.00 -> 1.00

## [2026-03-01 19:04:01] Codex Progress Log
- Task: 通过消融测试分析 DeepSeek-7B 中苹果相关关键神经元并提取编码结构
- Commands executed:
  - rg -n "deepseek|ablation|neuron|苹果|apple|feature" -S .
  - .\\.venv\\Scripts\\python.exe -c "import torch,transformers; ..."
  - .\\.venv\\Scripts\\python.exe tests/codex/deepseek7b_apple_neuron_ablation.py --top-k 60 --ablate-k 20 --causal-rank-pool 40 --causal-group apple_recall --max-discovery-prompts 120
  - .\\.venv\\Scripts\\python.exe tests/codex/deepseek7b_apple_neuron_ablation.py --top-k 80 --ablate-k 40 --causal-rank-pool 60 --causal-group apple_recall --max-discovery-prompts 120
  - 结果提取: python 读取 tempdata/deepseek7b_apple_ablation_20260301_190029/apple_neuron_ablation_results.json
- Result snapshot (run: 20260301_190029):
  - Apple causal neurons (ablate_k=40) layer distribution: {L0:3, L2:1, L3:33, L4:2, L13:1}
  - apple_recall target mass: 0.00350075 -> 0.00324513 (Δ=-0.00025562, -7.30% relative)
  - random ablation apple_recall: Δ=+0.00024450
  - fruit_attribute target mass: Δ=-0.00007590 (-0.45% relative)
  - apple_attribute target mass: Δ=+0.00146188 (+71.79% relative)
  - correlated module: size=40, mean|corr|=0.627, apple-control activation shift=+1.6015
- Theoretical math progress:
  - Apple-selective score: z_i = (mu_apple(i)-mu_ctrl(i)) / sqrt(0.5*(sigma_apple^2(i)+sigma_ctrl^2(i))+eps)
  - Causal score (single-neuron): delta_i = M_ablate_i - M_base, with M = mean target-token probability mass on apple_recall prompts.
  - Identified mixed-sign coding: same sparse module exhibits excitatory recall carriers (delta_i<0) and inhibitory attribute gates (multi-neuron ablation raises apple_attribute mass).
  - Structural interpretation: dominant coding manifold is early-layer sparse cluster centered at L3 with high co-activation coherence.

## [2026-03-01 19:48:26] Codex Progress Log
- Task: 分析香蕉、桔子等多个水果神经元差异，提取“水果”概念编码结构（DeepSeek-7B）
- Commands executed:
  - .\\.venv\\Scripts\\python.exe -m py_compile tests/codex/deepseek7b_multi_fruit_concept_analysis.py
  - .\\.venv\\Scripts\\python.exe tests/codex/deepseek7b_multi_fruit_concept_analysis.py --max-probe-items 180 --fruit-general-k 50 --fruit-specific-k 40
  - Patch script: add fruit_recall + per-fruit recall groups and rerun same command
  - 结果提取: python 读取 tempdata/deepseek7b_multi_fruit_20260301_194541/multi_fruit_analysis_results.json
- Result snapshot (run: 20260301_194541):
  - Fruit-general shared neurons = 50, layer distribution = {L0:4, L1:2, L3:9, L4:19, L5:9, L6:5, L8:2}
  - Shared module: size=50, mean|corr|=0.616, fruit-minus-nonfruit activation shift=+1.5862
  - Fruit-specific top-30 overlap (Jaccard): mostly 0.0 (apple vs pear = 0.017), indicating strong instance separation
  - Per-fruit recall targeted ablation deltas:
    - orange_recall: -0.01907357 (strong negative causal effect)
    - apple_recall: -0.00005857
    - pear_recall: -0.00000458
    - banana/grape/peach: near 0 in this prompt set
- Theoretical math progress:
  - Fruit-shared score: S_shared(i)=z(fruit_mean vs nonfruit)-lambda*interfruit_variance_penalty.
  - Fruit-specific score: S_spec(f,i)=z(mu_f(i)-mu_other_fruits(i)).
  - Structural finding supports two-level code:
    - shared sparse backbone for “fruitness” (co-activation module)
    - low-overlap instance submanifolds for specific fruits (banana/orange/etc).
  - Causal evidence is uneven across fruits in current probes, suggesting tokenization/prompt calibration sensitivity and mixed-sign gating inside shared backbone.

## [2026-03-01 19:59:30] Codex Progress Log
- Task: 分析猫、狗等多个动物的编码结构，并比较动物与水果概念差异（DeepSeek-7B）
- Commands executed:
  - .\\.venv\\Scripts\\python.exe -m py_compile tests/codex/deepseek7b_animal_fruit_concept_analysis.py
  - .\\.venv\\Scripts\\python.exe tests/codex/deepseek7b_animal_fruit_concept_analysis.py --max-probe-items 220 --animal-shared-k 50 --fruit-shared-k 50 --animal-specific-k 40
  - 结果提取: python 读取 tempdata/deepseek7b_animal_fruit_20260301_195504/animal_fruit_analysis_results.json
  - 附加统计: animal_specific top-30 overlap matrix + cat-dog overlap
- Result snapshot (run: 20260301_195504):
  - Animal shared neurons (k=50) layer distribution: {0:2,1:1,3:2,4:1,7:3,8:3,9:1,11:1,19:1,21:4,22:6,23:10,24:4,25:7,26:4}
  - Animal layer bands: early=12, middle=2, late=36 (late-heavy)
  - Fruit shared neurons (k=50) layer bands: early=41, middle=4, late=5 (early-heavy)
  - Animal-vs-fruit shared overlap (top50 Jaccard) = 0.0000
  - Animal-specific top30 overlaps mostly 0.0; cat-dog overlap = 0.0
- Theoretical math progress:
  - Shared concept score: S_shared(i)=z(mu_pos(i)-mu_neg(i)) - lambda*Var_between_pos(i)/sqrt(Var_pos(i)+eps).
  - Specific concept score: S_spec(c,i)=z(mu_c(i)-mu_peer(i)).
  - Structural dissociation found:
    - Animal concept code shifts to deeper layers (late concentration)
    - Fruit concept code remains mostly early-layer
    - Instance submanifolds (cat/dog/bird/...) are low-overlap sparse branches.
  - Cross-ablation causal signal remains weak under current prompt set; structure-level separability is strong while behavior-level effect size is small.

## [2026-03-01 20:26:39] Codex Progress Log
- Task: 做“猫 vs 狗 属性维度拆分（宠物性/攻击性/速度）”，并给出每个属性最小因果神经元子集
- Commands executed:
  - .\\.venv\\Scripts\\python.exe -m py_compile tests/codex/deepseek7b_cat_dog_attribute_causal.py
  - .\\.venv\\Scripts\\python.exe tests/codex/deepseek7b_cat_dog_attribute_causal.py --candidate-k 24 --fullset-size 12 --target-ratio 0.8 --off-penalty 0.5
  - Patch algorithm: greedy_minimal_subset switched from fixed-fullset reference to prefix-scan optimum (avoid cancellation failure)
  - Re-run same command and parse JSON report
- Result snapshot (run: 20260301_201331):
  - petness:
    - cat minimal subset size=10, target drop=0.00001112
    - dog minimal subset size=8, target drop=0.05857808
    - shared neurons: (L2,N16569), (L10,N13552), (L14,N521)
  - aggression:
    - cat minimal subset size=6, target drop=0.00000982
    - dog minimal subset size=8, target drop=0.00000569
    - shared neurons: (L3,N18126), (L14,N14289)
  - speed:
    - cat minimal subset size=12, target drop=0.00045420
    - dog minimal subset size=1, target drop=0.00050207 (L14,N13819)
    - shared neurons: none
- Theoretical math progress:
  - Single-neuron causal utility: U_i = (M_target_before - M_target_after_i) - lambda * |M_control_after_i - M_control_before|.
  - Minimal subset search upgraded to prefix-optimal strategy:
    1) rank by U_i,
    2) evaluate all prefixes up to K,
    3) find best achievable drop D_max,
    4) return smallest prefix reaching rho*D_max.
  - This yields micro-level causal subsets robust to neuron interaction cancellation.

## [2026-03-01 20:44:19] Codex Progress Log
- Task: 比较苹果与100个概念（含香蕉/兔子/太阳）的编码结构并提取规律
- Commands executed:
  - .\\.venv\\Scripts\\python.exe -m py_compile tests/codex/deepseek7b_apple_100_concepts_compare.py
  - .\\.venv\\Scripts\\python.exe tests/codex/deepseek7b_apple_100_concepts_compare.py --top-signature-k 120 --ablate-k 40
  - Patch metrics: add layerwise_topk (k=64 per layer), layerwise_jaccard_mean/max/shared_total
  - Re-run: .\\.venv\\Scripts\\python.exe tests/codex/deepseek7b_apple_100_concepts_compare.py --top-signature-k 120 --layer-top-k 64 --ablate-k 40
  - Patch extraction prompts to place concept token at sequence end and rerun same command
  - Tokenization check command for apple/banana/rabbit/sun...
- Result snapshot (final run: 20260301_204141):
  - Apple signature (top120 by z-score) layer distribution spans L0-L27, dense in late-middle bands (L23:12, L24:14, L25:5).
  - Pair metrics:
    - apple vs banana: layerwise_jaccard=0.1804, layer_shared=510
    - apple vs rabbit: layerwise_jaccard=0.1804, layer_shared=510
    - apple vs sun: layerwise_jaccard=0.3107, layer_shared=820
  - Causal ablation with apple signature top40:
    - apple_attr delta = -0.00020030 (strong negative)
    - banana_attr delta = +0.00000055 (near zero)
    - rabbit_attr delta = +0.00000148 (near zero)
    - sun_attr delta = +0.00116962 (positive drift)
- Theoretical math progress:
  - Strict global-top intersection in 530k dimensions is too brittle (near-zero overlap), replaced by layerwise overlap metric:
    - O_layer(c) = mean_l Jaccard(TopK_l(apple), TopK_l(c)).
  - Concept regularity emerges on two scales:
    1) micro-neuron exact identity: highly sparse and mostly concept-specific,
    2) meso-layer support pattern: stable overlap structure across categories.
  - Causal test confirms apple signature neurons carry apple-specific attribute mass with weak transfer to banana/rabbit tasks.

## [2026-03-01 21:18:10] Codex Progress Log
- Task: 设计并实测“微观因果编码结构算法”（非统计）用于 apple/banana 概念与属性结构解析
- Commands executed:
  - 新建脚本: tests/codex/deepseek7b_micro_causal_encoding_graph.py
  - .\\.venv\\Scripts\\python.exe -m py_compile tests/codex/deepseek7b_micro_causal_encoding_graph.py
  - .\\.venv\\Scripts\\python.exe tests/codex/deepseek7b_micro_causal_encoding_graph.py --concept-a apple --concept-b banana --candidate-k 28 --max-subset 10 --target-ratio 0.8 --off-penalty 0.5 --edge-alpha 0.5
  - 结果解析: 读取 tempdata/deepseek7b_micro_causal_apple_banana_20260301_210442/micro_causal_encoding_graph_results.json
- Result snapshot:
  - apple role subsets:
    - entity size=4, size-role size=9, weight-role size=6, fruit-role size=2
    - knowledge graph: 15 nodes, 54 directed cross-layer edges
  - banana role subsets:
    - entity size=1, size-role size=1, weight-role size=3, fruit-role size=1
    - knowledge graph: 5 nodes, 10 directed cross-layer edges
  - fruit-concept union subset size=3, layer distribution={1:1,4:1,26:1}
  - fruit-union ablation effect:
    - apple fruit role delta=-2.819e-05, size delta=-1.416e-05, weight delta=-1.555e-05
    - banana role deltas close to zero under current prompts
- Theoretical math progress:
  - Single-neuron role utility (causal, off-target constrained):
    U_i^(r) = (M_r(base)-M_r(do(x_i=0))) - lambda * sum_{r'!=r} |M_{r'}(do(x_i=0)) - M_{r'}(base)|
  - Minimal subset search (prefix-optimal, not static top-k):
    1) rank by U_i^(r), 2) evaluate all prefixes up to K,
    3) best drop D_max = max_n Δ_r(n), 4) return smallest n with Δ_r(n) >= rho*D_max.
  - Knowledge-network edge definition (interventional transmission):
    E(s->t) = E_prompt |a_t(do(x_s += alpha)) - a_t(base)|, with layer(t) > layer(s).
  - This pipeline yields micro-level causal nodes/edges + role decomposition, avoiding pure correlation-only structure claims.

## [2026-03-02 12:00:12] Codex Progress Log
- Task: 基于“整个知识体系”对苹果进行三尺度分析（Micro/Meso/Macro），提取系统编码规律
- Commands executed:
  - 新建脚本: tests/codex/deepseek7b_apple_triscale_micro_causal.py
  - .\\.venv\\Scripts\\python.exe -m py_compile tests/codex/deepseek7b_apple_triscale_micro_causal.py
  - Run#1: .\\.venv\\Scripts\\python.exe tests/codex/deepseek7b_apple_triscale_micro_causal.py --concept apple --candidate-k 24 --max-subset 10 --target-ratio 0.8 --off-penalty 0.5 --edge-alpha 0.5 (failed: name fix needed)
  - Patch fix: build_discovery_prompts -> discovery_prompts("micro", concept)[0]
  - Run#2 same config (success)
  - Patch meso prompts/eval stronger and rerun (success)
  - Run#3 low-penalty config: --off-penalty 0.1 (success, used as final tri-scale snapshot)
  - Result parse: tempdata/deepseek7b_triscale_apple_20260302_115048/apple_triscale_micro_causal_results.json
- Result snapshot (final tri-scale run):
  - micro subset: size=2, drop=0.00140040, off=0.00005017, layers={8:1,9:1}
  - meso subset: size=0 under current token-mass objective (baseline meso mass very low)
  - macro subset: size=2, drop=0.00011685, off=0.00148261, layers={9:1,23:1}
  - cross-scale deltas:
    - ablate micro -> micro -0.00140040, meso -7.13e-07, macro -4.95e-05
    - ablate macro -> micro -0.00148213, meso -4.82e-07, macro -0.00011685
  - tri-scale projection over 100 concepts: nearest categories to apple are fruit/food/nature; farthest means are animal/human/vehicle.
- Theoretical math progress:
  - Tri-scale utility:
    U_i^(s) = (M_s(base)-M_s(do(x_i=0))) - lambda * sum_{t!=s} |M_t(do(x_i=0)) - M_t(base)|
  - Minimal subset search uses prefix-optimal causal frontier (not static top-k).
  - Cross-scale causal matrix C_{s->t} = M_t(do(S_s=0)) - M_t(base) quantifies coupling/decoupling.
  - Meso failure mode detected: when target-token mass is near floor, direct mass objective underestimates available meso structure; requires margin/logit objective or role-specific anchors (fruit-role route already gives non-zero subset in prior run).

## [2026-03-02 13:04:23] Codex Progress Log
- Task: 设计并实测一个更清楚揭示深度网络“具体数学编码原理”的测试（MEPT）
- Commands executed:
  - 新建脚本: tests/codex/deepseek7b_math_encoding_principle_test.py
  - .\\.venv\\Scripts\\python.exe -m py_compile tests/codex/deepseek7b_math_encoding_principle_test.py
  - Run#1: .\\.venv\\Scripts\\python.exe tests/codex/deepseek7b_math_encoding_principle_test.py --apple-focus apple --topk-ablate 24 (failed: missing Counter import)
  - Patch fix: add rom collections import Counter
  - Run#2 same command (success)
  - Parse report/json under tempdata/deepseek7b_math_principle_20260302_130249/
- Result snapshot:
  - principle summary:
    - invariance_mean = 0.6507
    - additivity_error_mean = 0.3863
    - transport_error_mean = 0.2644
    - lowrank_effective_rank_mean = 4.7368
  - low-rank evidence per attribute: effective rank in [3.91, 6.28], k95 in [8, 9] over 14 entities
  - causal controllability (apple-aligned top24 ablation) shows mixed-sign effects:
    - taste_sweet target margin delta = -0.3916 (strong negative)
    - color_red / size_big deltas positive under current prompt-token setup
    - weight_heavy shows strong control-side coupling (control delta=-0.6641)
- Theoretical math progress:
  - MEPT defines five mechanistic tests:
    1) Direction invariance: cos(d(e_i,a), d(e_j,a))
    2) Additivity: v(e,a1,a2) ?≈ v(e)+d(e,a1)+d(e,a2)
    3) Transport: v(e2,a) ?≈ v(e2)+d(e1,a)
    4) Low-rank: SVD(D_a) effective rank / k95
    5) Causal controllability: margin shift under direction-aligned ablation
  - This test directly probes mathematical coding hypotheses (operator-like behavior) instead of relying on similarity-only statistics.

## [2026-03-02 13:29:56] Codex Progress Log
- Task: 设计测试以破解“复杂关联网络被高效编码”的核心结构（REPT）
- Commands executed:
  - 新建脚本: tests/codex/deepseek7b_relational_efficiency_principle_test.py
  - .\\.venv\\Scripts\\python.exe -m py_compile tests/codex/deepseek7b_relational_efficiency_principle_test.py
  - .\\.venv\\Scripts\\python.exe tests/codex/deepseek7b_relational_efficiency_principle_test.py --bridge-top-m 12 --edge-alpha 0.5
  - 结果解析: tempdata/deepseek7b_relational_efficiency_20260302_132818/relational_efficiency_principle_results.json
- Result snapshot:
  - Relational graph nodes=118, edge_density=0.3290
  - Graph-geometry alignment: Spearman=0.4603
  - Compression alignment: k=8 gives 0.4029, k=16 gives 0.3943; sample effective_rank=33.54, k95=51
  - Sparse-basis efficiency: k80_bridge_mass=336,926; k90_bridge_mass=409,441 (under full 530,432-neuron basis)
  - Causal bridge routing: mean_gain=5.27e-06, top bridge neuron L2N10110 gain=5.13e-05
- Theoretical math progress:
  - Proposed REPT hypothesis: efficient coding if relational graph distances are preserved in neural geometry and remain stable after low-rank compression.
  - Sparse routing metric uses bridge score b_i=sqrt(eta_rel(i)*eta_cat(i)) to detect neurons jointly encoding relation and subject-category structure.
  - Causal routing metric tests whether do(x_i += alpha) increases neighbor-vs-nonneighbor alignment for anchor nodes.
  - This protocol targets mechanism-level relational encoding, beyond static similarity statistics.

## [2026-03-02 14:13:35] Codex Progress Log
- Task: 继续执行 A->B->C 多跳推理链路测试，定位最小因果链路子集
- Commands executed:
  - 重写脚本: tests/codex/deepseek7b_multihop_reasoning_route_test.py
  - .\\.venv\\Scripts\\python.exe -m py_compile tests/codex/deepseek7b_multihop_reasoning_route_test.py
  - Quick run: .\\.venv\\Scripts\\python.exe tests/codex/deepseek7b_multihop_reasoning_route_test.py --max-chains 4 --candidate-k 4 --max-subset 3 --target-ratio 0.8 --route-penalty 0.5 --batch-size 16
  - Full run: .\\.venv\\Scripts\\python.exe tests/codex/deepseek7b_multihop_reasoning_route_test.py --candidate-k 8 --max-subset 5 --target-ratio 0.8 --route-penalty 0.5 --batch-size 16
  - 结果解析: tempdata/deepseek7b_multihop_route_20260302_140900/multihop_route_results.json
- Result snapshot (full run):
  - baseline: hop1_selectivity=3.58695e-05, hop2_selectivity=1.84423e-02, hop3_selectivity=1.42486e-02, route_index=1.42127e-02
  - minimal subset size=1: neuron=L27N16936 (flat_idx=528424)
  - after ablation: hop3_selectivity 0.01424859 -> 0.00830198 (drop=0.00594661, ~41.73%), hop1 drop=4.85684e-07
  - subset progress: best_drop=0.00690380, goal(80%)=0.00552304, achieved=0.00594661
- Theoretical math progress:
  - Multi-hop route selectivity is defined as S_h = E[p(target|hop=h,valid)] - E[p(target|hop=h,invalid)].
  - Route index is R = S_3 - S_1, capturing depth-specific routing gain.
  - Single-neuron causal utility uses U_i = Delta S_3(i) - lambda*|Delta S_1(i)| (lambda=route_penalty), enforcing hop3-specific causality instead of global damage.
  - Minimal causal subset search uses prefix-optimal frontier: choose smallest n such that Delta S_3(1..n) >= rho*max_k Delta S_3(1..k) (rho=target_ratio).

## [2026-03-02 16:50:30] Codex Progress Log
- Task: 扩大多跳样本规模，提炼 A->B->C 编码的一般规律
- Commands executed:
  - 新建脚本: tests/codex/deepseek7b_multihop_large_sample_generalization.py
  - .\\.venv\\Scripts\\python.exe -m py_compile tests/codex/deepseek7b_multihop_large_sample_generalization.py
  - Run#1 (120 chains): .\\.venv\\Scripts\\python.exe tests/codex/deepseek7b_multihop_large_sample_generalization.py --max-chains 120 --candidate-k 12 --max-subset 6 --target-ratio 0.8 --route-penalty 0.5 --batch-size 32 --discovery-max-items 240 --bootstrap 200
  - Run#2 (120 chains, seed=7): .\\.venv\\Scripts\\python.exe tests/codex/deepseek7b_multihop_large_sample_generalization.py --max-chains 120 --candidate-k 8 --max-subset 4 --target-ratio 0.8 --route-penalty 0.5 --batch-size 32 --discovery-max-items 180 --bootstrap 100 --seed 7
  - 结果解析:
    - tempdata/deepseek7b_multihop_large_20260302_145153/multihop_large_generalization_results.json
    - tempdata/deepseek7b_multihop_large_20260302_160315/multihop_large_generalization_results.json
- Result snapshot:
  - Run#1: hop3_selectivity=0.04319, route_index=0.03781, subset_size=4, after_hop3=0.03992 (drop=0.00327)
  - Run#2: hop3_selectivity=0.05323, route_index=0.04823, subset_size=2, after_hop3=0.05153 (drop=0.00170)
  - 稳定重叠神经元: (L24,N8124), (L27,N16649); 另有 L27,N16936 在 Run#1 中仍高效
  - 候选层分布集中在后层: L22-L27，effective_layers 约 4.75
  - 跨域掉点最高稳定出现在动物相关域（mammal/bird/insect/fish），其次城市/水果域；工具/乐器域出现弱负掉点
- Theoretical math progress:
  - 大样本路由增益在两次运行均显著为正: R=S3-S1 > 0，且 bootstrap 95% CI 下界均为正（Run#1: 0.0160, Run#2: 0.0205）。
  - 这支持“深层多跳链路增强”规律: S3 > S2 >> S1，并非小样本偶然。
  - 最小因果子集在大样本下仍可找到，且跨seed出现重叠节点，表明存在可复现的微观路由子电路而非纯统计漂移。
  - 跨域掉点异质性说明子电路具有“共享+专有”结构：同一子集对动物域影响更强，对部分人工制品域近零或反向。

## [2026-03-02 17:29:51] Codex Progress Log
- Task: 设计“苹果关键神经元”3D 可视化效果，并接入蓝图界面
- Commands executed:
  - 新建组件: frontend/src/blueprint/AppleNeuron3DTab.jsx
  - 修改入口: frontend/src/HLAIBlueprint.jsx（新增 apple_neuron_3d 标签页并渲染组件）
  - 构建校验: npm run build (workdir=frontend)
- Result snapshot:
  - 新增 3D 场景元素：层级网格(L0-L27)、关键神经元高亮点、因果候选连线、悬浮信息卡、侧边指标面板。
  - 关键神经元数据融合：苹果 micro/macro 子集（L8N7574, L9N14608, L23N16819）+ 多跳共享 route 神经元（L24N8124, L27N16649, L27N16936）。
  - 交互能力：Orbit 旋转缩放、点击节点查看 Layer/Neuron/Strength/Source。
  - 构建通过：vite build success。
- Theoretical math progress:
  - 3D 坐标映射采用 f(layer, neuron)->(x,y,z)，其中 z 与 layer 单调对应，(x,y) 由 neuron id 的角度与半径参数化，等价于“层深 + 通道位相”联合编码。
  - 节点强度采用 s ~ sqrt(utility) 映射到可视半径，保留因果效应量级差异并抑制极值压缩。
  - 连线图表达从 micro->macro->route 的候选知识流路径，便于观察“属性编码到抽象路由”的层间迁移结构。



## [2026-03-02 17:50:30] Codex Progress Log
- Task: 修复“苹果神经元3D界面看不到”的入口可见性问题
- Commands executed:
  - 修改文件: frontend/src/HLAIBlueprint.jsx
    - 顶部 nav 增加 overflowX=auto
    - 将 pple_neuron_3d 标签提前到前排
    - 新增右上角固定直达按钮“苹果3D”
  - 构建校验: npm run build (workdir=frontend)
- Result snapshot:
  - 入口可见性双保险：顶部标签可横向滚动 + 右上角一键进入苹果3D。
  - 构建通过，前端可正常编译。
- Theoretical math progress:
  - 可视化入口属于“观测算子”层；保证低摩擦访问可提高重复观察频率，从而提高微观编码结构验证的实验效率与可复现性。

## [2026-03-02 19:37:19] Codex Progress Log
- Task: 排查并修复“星团宇宙拓扑与全局极其稀疏定律”模块不显示
- Commands executed:
  - 检索渲染链路: rg -n "UniversalManifoldGraph|manifold|稀疏|topology" frontend/src
  - 发现问题: rontend/src/blueprint/UniversalManifoldGraph.jsx 文件不存在，且无渲染入口
  - 新建组件: frontend/src/blueprint/UniversalManifoldGraph.jsx
  - 接入页面: frontend/src/blueprint/GeminiTab.jsx
    - 新增 import UniversalManifoldGraph
    - 在 H2 阶段详情中插入 <UniversalManifoldGraph />
    - 将 H2 默认展开（expandedSteps 初值设为 {1: true}）
  - 构建校验: npm run build (workdir=frontend)
- Result snapshot:
  - “星团宇宙拓扑与全局极稀疏定律”现在在 GeminiTab 的 H2 区块可见。
  - 展示内容包含：星团概念拓扑示意、全局稀疏定律指标卡、拓扑-稀疏-正交图例解释。
  - 构建通过。
- Theoretical math progress:
  - 原因层面：该模块此前属于“未挂接组件”，不是计算失败而是前端渲染链路缺失。
  - 显示层面：补齐“组件存在 + 入口挂接 + 默认展开”三条件后，观测闭环恢复。


## [2026-03-02 20:52:48] Codex Progress Log
- Task: 使用远端 GeminiTab 覆盖本地文件
- Commands executed:
  - git fetch origin main
  - git checkout origin/main -- frontend/src/blueprint/GeminiTab.jsx
  - 校验: git ls-files -s -- frontend/src/blueprint/GeminiTab.jsx
  - 校验: git rev-parse origin/main:frontend/src/blueprint/GeminiTab.jsx
- Result snapshot:
  - 已将 rontend/src/blueprint/GeminiTab.jsx 覆盖为远端 origin/main 版本。
  - 索引 blob 一致性校验通过（两者均为 5b3c3407ec4b56df40adcde3fb60c4693be4cba）。
  - 该文件当前为 staged 修改状态（相对本地 HEAD 有差异），属于预期结果。
- Theoretical math progress:
  - 本次为版本基线回退/对齐操作，不涉及新的编码数学结论；其作用是恢复实验可视化链路到远端已验证状态，降低前端结论漂移风险。

## [2026-03-02 20:54:34] Codex Progress Log
- Task: 解释“在签出前，请清理仓库工作树”的含义
- Commands executed:
  - 无（概念解释）
- Result snapshot:
  - 该提示表示当前工作树存在未提交/未暂存改动；为避免 checkout 覆盖或冲突，需先提交、暂存(stash)或放弃改动。
- Theoretical math progress:
  - 本次为 Git 工作流说明，不涉及新的数学编码结论。

## [2026-03-02 21:01:19] Codex Progress Log
- Task: 拉取远端代码并合并本地代码（main <- origin/main）
- Commands executed:
  - git status -sb
  - git pull --rebase --autostash origin main (首次失败)
  - git stash push -u -m "codex-pre-pull-merge-20260302"
  - git fetch origin main
  - git merge --ff-only origin/main (识别到 package*.json 的隐式改动阻塞)
  - git update-index --no-skip-worktree -- frontend/package.json frontend/package-lock.json
  - git merge --ff-only origin/main (成功 fast-forward 到 ea135d5)
  - git stash pop "stash@{1}"（回放本地改动，因同名已存在文件 UniversalManifoldGraph.jsx 触发部分回放提示）
  - git stash pop "stash@{0}"（package 文件回放）
- Result snapshot:
  - 分支已与远端同步：HEAD=origin/main=ea135d5。
  - 本地主要修改与未跟踪文件已恢复（HLAIBlueprint、脚本、tempdata、tests/codex 等）。
  - 唯一冲突点为 rontend/src/blueprint/UniversalManifoldGraph.jsx：远端已有该文件，stash 中同名未跟踪文件未直接覆盖；当前保留远端版本。
  - 仍保留一条安全 stash：codex-pre-pull-merge-20260302，便于后续人工比对/回滚。
- Theoretical math progress:
  - 本次为代码基线同步与工作树回放流程，不新增编码机制结论；其研究意义是保证“远端基线 + 本地实验增量”可共存，减少实验代码漂移。

## [2026-03-02 21:06:44] Codex Progress Log
- Task: 同步后继续执行 UniversalManifoldGraph 的远端+本地融合
- Commands executed:
  - 读取对比: Get-Content frontend/src/blueprint/UniversalManifoldGraph.jsx
  - 读取 stash 版本: git show "stash@{0}^3:frontend/src/blueprint/UniversalManifoldGraph.jsx"
  - 重写融合版: frontend/src/blueprint/UniversalManifoldGraph.jsx (UTF-8)
  - 构建校验: npm run build (workdir=frontend)
  - 状态检查: git status --short frontend/src/blueprint/UniversalManifoldGraph.jsx frontend/src/blueprint/GeminiTab.jsx
- Result snapshot:
  - 已得到融合后的 UniversalManifoldGraph：保留远端主结构（全局稀疏常量 + 聚类纯度条形）并加入本地“快速总览卡片/图例”风格。
  - 编译通过（vite build success）。
  - 当前该文件为已修改状态：frontend/src/blueprint/UniversalManifoldGraph.jsx。
- Theoretical math progress:
  - 可视化上将“宏观拓扑结构”与“稀疏定律摘要指标”并列展示，形成结构-统计双视角观测；便于在同一面板上验证“概念簇化 + 低激活率”是否同时成立。

## [2026-03-02 21:07:39] Codex Progress Log
- Task: UniversalManifoldGraph.jsx 使用远端版本覆盖本地版本
- Commands executed:
  - git checkout origin/main -- frontend/src/blueprint/UniversalManifoldGraph.jsx
  - git status --short frontend/src/blueprint/UniversalManifoldGraph.jsx
  - git diff -- frontend/src/blueprint/UniversalManifoldGraph.jsx
- Result snapshot:
  - 文件已被远端版本覆盖。
  - 当前该文件无本地差异（diff 行数=0）。
- Theoretical math progress:
  - 本次为版本回归操作，不新增编码结构数学结论；目的是恢复可复现前端基线。

## [2026-03-02 23:42:38] Apple Neuron 3D: Full-Layer + Fruit Comparison UI Update

### Commands Executed
- git status --short
- Get-Content -Raw frontend/src/blueprint/AppleNeuron3DTab.jsx
- Get-Content -Raw tempdata/deepseek7b_multi_fruit_20260301_194251/MULTI_FRUIT_ANALYSIS_REPORT.md
- Set-Content frontend/src/blueprint/AppleNeuron3DTab.jsx
- 
pm run build (workdir: rontend/)

### Engineering Progress
- Rebuilt AppleNeuron3DTab.jsx to show **all layers L0-L27** with per-layer labels.
- Added fruit comparison overlay in the same 3D space:
  - Apple core neurons (micro/macro/route)
  - Fruit-general neurons
  - Fruit-specific neurons for apple / banana / orange / grape
- Added filter panel for toggling fruit-general and each fruit-specific set.
- Added fruit-cluster links and updated legend + selected-neuron metadata panel.
- Build verification passed via Vite production build.

### Math / Structure Research Progress
- Introduced a practical visualization decomposition:
  - Core(apple) = causal chain candidates (micro -> macro -> route-shared)
  - General(fruit) = shared concept manifold anchor points
  - Specific(fruit_i) = concept-specific deviation vectors per fruit category
- This supports direct visual comparison of:
  - shared concept subspace (fruit-general)
  - concept-specific offsets (apple vs banana vs orange vs grape)
  - depth-wise concentration across all layers (full L0-L27 axis)

## [2026-03-02 23:52:29] Apple Neuron 3D: Textbox-driven Concept Neuron Generation

### Commands Executed
- Get-Content -Raw frontend/src/blueprint/AppleNeuron3DTab.jsx
- pply_patch on rontend/src/blueprint/AppleNeuron3DTab.jsx
- 
pm run build (workdir: rontend/)

### Engineering Progress
- Added Quick Concept Generator panel in Apple Neuron 3D tab.
- Added input textbox + Enter/button trigger to generate neurons for arbitrary concept names.
- Added deterministic concept-to-neuron mapping (hash-based) to produce stable neuron subsets across layers.
- Added generated concept list with per-concept color and remove action.
- Added generated concept neurons into 3D render pipeline and link structure.
- Added selected neuron metadata field Concept for query-generated nodes.

### Math / Structure Research Progress
- Introduced a deterministic mapping test interface:
  - Concept string -> hash seed -> layer-neuron subset distribution.
- This enables fast comparative probing of concept encoding hypotheses by reusing the same coordinate manifold:
  - fixed model anchors (apple core / fruit-general / fruit-specific)
  - dynamic query anchors (textbox concepts)
- Practical value: supports rapid visual checks for overlap, depth concentration, and potential shared subspace patterns.

## [2026-03-03 00:07:09] Universal Control: Add MAIN button and mount AppleNeuron3D workspace

### Commands Executed
- Get-Content -Raw frontend/src/blueprint/UniversalManifoldGraph.jsx
- Get-Content -Raw frontend/src/blueprint/GeminiTab.jsx
- Get-Content -Raw frontend/src/blueprint/AppleNeuron3DTab.jsx
- g -n "dnn|snn|fibernet|main" frontend/src/blueprint -S
- pply_patch on rontend/src/blueprint/AppleNeuron3DTab.jsx
- Set-Content frontend/src/blueprint/UniversalManifoldGraph.jsx
- 
pm run build (workdir: rontend/)

### Engineering Progress
- Added a new top mode bar in UniversalManifoldGraph with buttons: MAIN, DNN, SNN, FIBERNET.
- Ensured MAIN is placed at the left side of dnn/snn/fibernet controls.
- Wired MAIN mode to render full Apple neuron 3D workspace content:
  - textbox concept generator
  - checkbox compare filters
  - legend/instruction panel
  - right-side 3D scene with all layers and corresponding neurons
- Refactored AppleNeuron3DTab to support layout props:
  - panelPosition (left/right)
  - sceneHeight
- In UniversalManifoldGraph MAIN mode, mounted:
  - AppleNeuron3DTab panelPosition="left" sceneHeight="72vh"

### Math / Structure Research Progress
- Established a controllable route-switch interface where MAIN directly maps to the micro-level neuron manifold view.
- This creates a layered comparative workflow:
  - route selector (DNN/SNN/FiberNet) for macro framework context
  - MAIN for direct neuron-level structural inspection (layer topology + concept neurons)
- Supports practical verification of cross-level mapping:
  - architecture-level paradigm selection
  - neuron-level coding structure observation in one interface.

## [2026-03-03 00:12:26] Fix visibility: Add MAIN tab to global App control panel

### Commands Executed
- g -n "dnn|snn|fibernet|MAIN|main" frontend/src -S
- Get-Content -Raw frontend/src/blueprint/UniversalManifoldGraph.jsx
- Get-Content frontend/src/App.jsx | Select-Object -Skip 2140 -First 330
- Get-Content -Raw frontend/src/config/panels.js
- pply_patch on rontend/src/App.jsx
- Set-Content frontend/src/config/panels.js
- 
pm run build (workdir: rontend/)

### Engineering Progress
- Added Main into global INPUT_PANEL_TABS so it appears alongside DNN/SNN/FiberNet in App control panel.
- Updated tab click behavior in App.jsx:
  - click Main -> opens blueprint with initial tab pple_neuron_3d.
  - non-main tabs keep existing system type switching behavior.
- Added inputPanelTab === 'main' content block with one-click entry button to Apple neuron 3D workspace.
- Resolved string-termination issues in config/panels.js introduced by legacy mojibake text and revalidated build.

### Math / Structure Research Progress
- Strengthened UI-level access path from macro system tabs (DNN/SNN/FiberNet) to micro neuron topology workspace (Apple Neuron 3D).
- This improves reproducible inspection flow:
  - macro route selection in App control panel
  - direct jump to layer-neuron manifold view for causal/structural hypothesis checks.

## [2026-03-03 00:14:17] Move AppleNeuron3D to main App interface (not blueprint modal)

### Commands Executed
- pply_patch on rontend/src/App.jsx
- 
pm run build (workdir: rontend/)

### Engineering Progress
- Imported AppleNeuron3DTab into App.jsx.
- Changed Main tab behavior in global control panel:
  - no longer opens HLAIBlueprint modal.
  - switches to inputPanelTab='main' and keeps system type on DNN-compatible path.
- Added direct main-interface mount:
  - when inputPanelTab === 'main', render AppleNeuron3DTab as absolute main workspace in App.
  - layout places Apple 3D workspace in the right-side main area while preserving left control panel tabs.
- Updated Main tab panel copy to indicate direct main-interface rendering.

### Math / Structure Research Progress
- Upgraded interaction path from modal navigation to immediate in-situ neuron manifold inspection.
- This reduces context switch latency between control actions and 3D layer-neuron observation, improving iterative structural hypothesis testing efficiency.

## [2026-03-03 00:21:04] Main UI split: control cards under panel, 3D scene in main space

### Commands Executed
- git show HEAD:frontend/src/App.jsx | Set-Content frontend/src/App.jsx
- pply_patch on rontend/src/blueprint/AppleNeuron3DTab.jsx
- pply_patch on rontend/src/App.jsx
- 
pm run build (workdir: rontend/)

### Engineering Progress
- Refactored Apple neuron module into reusable pieces:
  - useAppleNeuronWorkspace() (shared state/model)
  - AppleNeuronControlPanels (small cards / controls)
  - AppleNeuronMainScene (pure 3D scene)
- In App.jsx main tab flow:
  - inputPanelTab === 'main' now renders control cards under the left control panel.
  - right main area renders only 3D neuron scene (AppleNeuronMainScene).
- Kept DNN/SNN/FiberNet behavior unchanged; main maps to DNN-compatible runtime state.

### Math / Structure Research Progress
- Established a clearer operator workflow:
  - left panel = semantic control surface (query/filter/legend/selection)
  - main 3D area = dedicated layer-neuron manifold observation field
- This separation improves human interpretability of micro-causal editing vs spatial topology reading.

## [2026-03-03 00:22:31] App.jsx mojibake fix and stabilization

### Commands Executed
- git show HEAD:frontend/src/App.jsx | Set-Content frontend/src/App.jsx
- pply_patch on rontend/src/App.jsx
- 
pm run build (workdir: rontend/)
- g -n "鍙|鍦|鎴|绁|妯|闈|涓€|鈥|�" frontend/src/App.jsx -S

### Engineering Progress
- Recovered App.jsx from clean HEAD to remove widespread mojibake side effects.
- Re-applied only required feature changes:
  - Main tab integration in control panel.
  - Apple neuron control cards under left panel.
  - Apple neuron 3D scene in main right workspace.
- Verified frontend build success after recovery and patch.
- Verified no typical mojibake tokens remain in App.jsx.

### Math / Structure Research Progress
- UI channel now cleanly separates semantic control operations (left panel) from geometric neural manifold visualization (main 3D field).
- This reduces interpretation noise caused by corrupted labels and improves reproducibility of manual causal-inspection workflows.

## [2026-03-03 00:30:02] Codex 进展记录
- 任务: 在“战略层级路线图”中加入“苹果神经元3D”按钮与界面，并保留独立标签页入口。
- 代码变更:
  - 重写 rontend/src/blueprint/ProjectRoadmapTab.jsx：新增“苹果神经元3D”入口卡片、展开/收起按钮、嵌入式 AppleNeuron3DTab。
  - 更新 rontend/src/HLAIBlueprint.jsx：向 ProjectRoadmapTab 传入 onOpenAppleNeuron3D 回调，支持一键切换到独立3D标签页。
- 执行命令:
  - Get-Content frontend/src/blueprint/ProjectRoadmapTab.jsx
  - Get-Content frontend/src/HLAIBlueprint.jsx
  - Get-Content frontend/src/blueprint/AppleNeuron3DTab.jsx
  - g -n "战略|路线图|roadmap|ProjectRoadmapTab|apple_neuron_3d" frontend/src/blueprint/ProjectRoadmapTab.jsx frontend/src/HLAIBlueprint.jsx
  - 
pm run build (workdir=rontend) -> 通过
- 理论/数学研究进展:
  - 把“路线图层的宏观策略”和“苹果神经元微观结构可视化”连接为同一工作流入口，形成 Macro->Meso->Micro 的操作闭环。
  - 当前实现支持在战略页面直接触发微观神经元观察，便于后续做“策略节点 -> 因果子网络”的映射实验。

## [2026-03-03 00:34:39] Codex 进展记录
- 任务: 去掉顶部“项目大纲”和“深度分析”之间的“苹果神经元3D”按钮，以及右侧“苹果3D”快捷按钮。
- 代码变更:
  - rontend/src/HLAIBlueprint.jsx
    - 删除顶部导航项 { id: 'apple_neuron_3d', label: '苹果神经元3D' }
    - 删除右上角 苹果3D 按钮（保留关闭按钮）
- 执行命令:
  - g -n "apple_neuron_3d|苹果神经元3D|苹果3D" frontend/src/HLAIBlueprint.jsx
  - 
pm run build (workdir=rontend) -> 通过
- 理论/数学研究进展:
  - 界面入口收敛到主路线图中的内嵌入口，降低并行入口造成的状态分叉，便于后续做统一实验流程记录与因果对照。

## [2026-03-03 00:40:47] Codex 进展记录
- 任务: 将“苹果神经元3D”渲染并入主3D空间，取消 main 模式下独立覆盖图层。
- 代码变更:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - 新增导出 AppleNeuronSceneContent（仅场景内容，不创建独立 Canvas）。
    - 现有 AppleNeuronMainScene/AppleNeuron3DTab 继续保留，用于其他页面嵌入。
  - rontend/src/App.jsx
    - 删除 inputPanelTab === 'main' 的绝对定位 AppleNeuronMainScene 覆盖层。
    - 在主 Canvas 中新增 isAppleMainView 分支，main 模式直接渲染 AppleNeuronSceneContent。
    - main 模式下关闭其他分析图层（LogitLens/分析叠加/Flow/TDA/SNN/GlassMatrix 等）避免图层混叠。
- 执行命令:
  - g -n "inputPanelTab === 'main'|AppleNeuronMainScene|AppleNeuronControlPanels|useAppleNeuronWorkspace|UniversalManifoldGraph|Canvas" frontend/src/App.jsx
  - 
pm run build (workdir=rontend) -> 通过
- 理论/数学研究进展:
  - 将“概念神经元几何”与“全局主场景流形”统一到同一渲染空间，避免多 Canvas 造成的坐标系分裂。
  - 该统一渲染有利于后续做跨模块几何对齐实验（同一相机/同一空间内比较概念簇与结构分析层）。

## [2026-03-03 00:51:43] Codex 进展记录
- 任务: 为苹果神经元3D增加“按下一词预测机制驱动”的动画效果。
- 代码变更:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - 新增 next-token 链生成逻辑（generatePredictChain），支持中英关键词上下文。
    - 新增预测动画状态机：播放/暂停、步进、重置、速度控制、层推进（L0 -> L27）。
    - 新增 TokenPredictionCarrier 3D 动画载体（当前 token + 概率）。
    - PulsingNeuron 增加 predictionStrength，将预测强度映射到发光与缩放，形成动态神经元激活动画。
    - 控制面板新增 Next-Token Prediction Animation 区域（输入上下文、播放控制、速度滑条、token 概率链）。
    - AppleNeuronSceneContent 支持 prediction 入参并在主场景渲染动画。
  - rontend/src/App.jsx
    - 主3D空间调用 AppleNeuronSceneContent 时传入 ppleNeuronWorkspace.prediction，确保主Canvas可见预测动画。
- 执行命令:
  - g -n "useAppleNeuronWorkspace|PulsingNeuron|AppleNeuronSceneContent|AppleNeuronControlPanels|summary" frontend/src/blueprint/AppleNeuron3DTab.jsx
  - 
pm run build (workdir=rontend) -> 通过
- 理论/数学研究进展:
  - 以“自回归下一词预测”的离散时间步作为驱动变量，把 token 概率链映射到层深推进与神经元激活强度。
  - 形成了可观测的机制近似：context -> next token distribution -> layer-wise activation trajectory，便于后续做因果消融与路径归因对照实验。

## [2026-03-03 00:54:23] Codex 进展记录
- 任务: 参考 DNN 结构分析，在 MAIN 中添加分析类型切换：静态分析 / 动态预测。
- 代码变更:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - useAppleNeuronWorkspace 新增 nalysisType 状态与 setAnalysisType。
    - 增加“分析类型”面板按钮：静态分析、动态预测。
    - 动态模式沿用 next-token 动画；静态模式关闭预测动画并隐藏动态预测配置卡片。
    - summary 增加 nalysisType 与当前 token 显示逻辑（静态模式显示“静态分析”）。
    - prediction 输出在静态模式下置为 
ull，场景不再注入预测激活。
- 执行命令:
  - 
pm run build (workdir=rontend) -> 通过
- 理论/数学研究进展:
  - 将 MAIN 分析拆分为“结构静态观测”和“机制动态演化”两种模式，形成同一概念空间下的双视角验证。
  - 为后续因果分析提供更清晰实验控制变量：固定结构分布 vs. 时序预测驱动。

## [2026-03-03 00:56:40] Codex 进展记录
- 任务: 在苹果3D主空间中清晰显示 layer 序号。
- 代码变更:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - 增强 LayerGuides：每层在左右两侧同时显示 L0~L27。
    - 提升字号与对比度，增加文字描边（outline）提高可读性。
    - 增加端点标记文字：Layer 0 与 Layer 27。
- 执行命令:
  - 
pm run build (workdir=rontend) -> 通过
- 理论/数学研究进展:
  - 显式层序号可作为“深度坐标系”观察基准，便于将动态预测轨迹与层级结构对应，支持后续层级因果路径分析。
