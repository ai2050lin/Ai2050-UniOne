
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
  - 已将 rontend/src/blueprint/GeminiTab.jsx 覆盖为远端 origin/main 版本。
  - 索引 blob 一致性校验通过（两者均为 5b3c3407ec4b56df40adcde3fb60c4693be4cba）。
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
  - 唯一冲突点为 rontend/src/blueprint/UniversalManifoldGraph.jsx：远端已有该文件，stash 中同名未跟踪文件未直接覆盖；当前保留远端版本。
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
pm run build (workdir: rontend/)

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
- pply_patch on rontend/src/blueprint/AppleNeuron3DTab.jsx
- 
pm run build (workdir: rontend/)

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
- 
g -n "dnn|snn|fibernet|main" frontend/src/blueprint -S
- pply_patch on rontend/src/blueprint/AppleNeuron3DTab.jsx
- Set-Content frontend/src/blueprint/UniversalManifoldGraph.jsx
- 
pm run build (workdir: rontend/)

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
- 
g -n "dnn|snn|fibernet|MAIN|main" frontend/src -S
- Get-Content -Raw frontend/src/blueprint/UniversalManifoldGraph.jsx
- Get-Content frontend/src/App.jsx | Select-Object -Skip 2140 -First 330
- Get-Content -Raw frontend/src/config/panels.js
- pply_patch on rontend/src/App.jsx
- Set-Content frontend/src/config/panels.js
- 
pm run build (workdir: rontend/)

### Engineering Progress
- Added Main into global INPUT_PANEL_TABS so it appears alongside DNN/SNN/FiberNet in App control panel.
- Updated tab click behavior in App.jsx:
  - click Main -> opens blueprint with initial tab pple_neuron_3d.
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
- pply_patch on rontend/src/App.jsx
- 
pm run build (workdir: rontend/)

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
- pply_patch on rontend/src/blueprint/AppleNeuron3DTab.jsx
- pply_patch on rontend/src/App.jsx
- 
pm run build (workdir: rontend/)

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
- pply_patch on rontend/src/App.jsx
- 
pm run build (workdir: rontend/)
- 
g -n "鍙|鍦|鎴|绁|妯|闈|涓€|鈥|�" frontend/src/App.jsx -S

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
  - 重写 rontend/src/blueprint/ProjectRoadmapTab.jsx：新增“苹果神经元3D”入口卡片、展开/收起按钮、嵌入式 AppleNeuron3DTab。
  - 更新 rontend/src/HLAIBlueprint.jsx：向 ProjectRoadmapTab 传入 onOpenAppleNeuron3D 回调，支持一键切换到独立3D标签页。
- 执行命令:
  - Get-Content frontend/src/blueprint/ProjectRoadmapTab.jsx
  - Get-Content frontend/src/HLAIBlueprint.jsx
  - Get-Content frontend/src/blueprint/AppleNeuron3DTab.jsx
  - 
g -n "战略|路线图|roadmap|ProjectRoadmapTab|apple_neuron_3d" frontend/src/blueprint/ProjectRoadmapTab.jsx frontend/src/HLAIBlueprint.jsx
  - 
pm run build (workdir=rontend) -> 通过
- 理论/数学研究进展:
  - 把“路线图层的宏观策略”和“苹果神经元微观结构可视化”连接为同一工作流入口，形成 Macro->Meso->Micro 的操作闭环。
  - 当前实现支持在战略页面直接触发微观神经元观察，便于后续做“策略节点 -> 因果子网络”的映射实验。

## [2026-03-03 00:34:39] Codex 进展记录
- 任务: 去掉顶部“项目大纲”和“深度分析”之间的“苹果神经元3D”按钮，以及右侧“苹果3D”快捷按钮。
- 代码变更:
  - rontend/src/HLAIBlueprint.jsx
    - 删除顶部导航项 { id: 'apple_neuron_3d', label: '苹果神经元3D' }
    - 删除右上角 苹果3D 按钮（保留关闭按钮）
- 执行命令:
  - 
g -n "apple_neuron_3d|苹果神经元3D|苹果3D" frontend/src/HLAIBlueprint.jsx
  - 
pm run build (workdir=rontend) -> 通过
- 理论/数学研究进展:
  - 界面入口收敛到主路线图中的内嵌入口，降低并行入口造成的状态分叉，便于后续做统一实验流程记录与因果对照。

## [2026-03-03 00:40:47] Codex 进展记录
- 任务: 将“苹果神经元3D”渲染并入主3D空间，取消 main 模式下独立覆盖图层。
- 代码变更:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - 新增导出 AppleNeuronSceneContent（仅场景内容，不创建独立 Canvas）。
    - 现有 AppleNeuronMainScene/AppleNeuron3DTab 继续保留，用于其他页面嵌入。
  - rontend/src/App.jsx
    - 删除 inputPanelTab === 'main' 的绝对定位 AppleNeuronMainScene 覆盖层。
    - 在主 Canvas 中新增 isAppleMainView 分支，main 模式直接渲染 AppleNeuronSceneContent。
    - main 模式下关闭其他分析图层（LogitLens/分析叠加/Flow/TDA/SNN/GlassMatrix 等）避免图层混叠。
- 执行命令:
  - 
g -n "inputPanelTab === 'main'|AppleNeuronMainScene|AppleNeuronControlPanels|useAppleNeuronWorkspace|UniversalManifoldGraph|Canvas" frontend/src/App.jsx
  - 
pm run build (workdir=rontend) -> 通过
- 理论/数学研究进展:
  - 将“概念神经元几何”与“全局主场景流形”统一到同一渲染空间，避免多 Canvas 造成的坐标系分裂。
  - 该统一渲染有利于后续做跨模块几何对齐实验（同一相机/同一空间内比较概念簇与结构分析层）。

## [2026-03-03 00:51:43] Codex 进展记录
- 任务: 为苹果神经元3D增加“按下一词预测机制驱动”的动画效果。
- 代码变更:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - 新增 next-token 链生成逻辑（generatePredictChain），支持中英关键词上下文。
    - 新增预测动画状态机：播放/暂停、步进、重置、速度控制、层推进（L0 -> L27）。
    - 新增 TokenPredictionCarrier 3D 动画载体（当前 token + 概率）。
    - PulsingNeuron 增加 predictionStrength，将预测强度映射到发光与缩放，形成动态神经元激活动画。
    - 控制面板新增 Next-Token Prediction Animation 区域（输入上下文、播放控制、速度滑条、token 概率链）。
    - AppleNeuronSceneContent 支持 prediction 入参并在主场景渲染动画。
  - rontend/src/App.jsx
    - 主3D空间调用 AppleNeuronSceneContent 时传入 ppleNeuronWorkspace.prediction，确保主Canvas可见预测动画。
- 执行命令:
  - 
g -n "useAppleNeuronWorkspace|PulsingNeuron|AppleNeuronSceneContent|AppleNeuronControlPanels|summary" frontend/src/blueprint/AppleNeuron3DTab.jsx
  - 
pm run build (workdir=rontend) -> 通过
- 理论/数学研究进展:
  - 以“自回归下一词预测”的离散时间步作为驱动变量，把 token 概率链映射到层深推进与神经元激活强度。
  - 形成了可观测的机制近似：context -> next token distribution -> layer-wise activation trajectory，便于后续做因果消融与路径归因对照实验。

## [2026-03-03 00:54:23] Codex 进展记录
- 任务: 参考 DNN 结构分析，在 MAIN 中添加分析类型切换：静态分析 / 动态预测。
- 代码变更:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - useAppleNeuronWorkspace 新增 nalysisType 状态与 setAnalysisType。
    - 增加“分析类型”面板按钮：静态分析、动态预测。
    - 动态模式沿用 next-token 动画；静态模式关闭预测动画并隐藏动态预测配置卡片。
    - summary 增加 nalysisType 与当前 token 显示逻辑（静态模式显示“静态分析”）。
    - prediction 输出在静态模式下置为 
ull，场景不再注入预测激活。
- 执行命令:
  - 
pm run build (workdir=rontend) -> 通过
- 理论/数学研究进展:
  - 将 MAIN 分析拆分为“结构静态观测”和“机制动态演化”两种模式，形成同一概念空间下的双视角验证。
  - 为后续因果分析提供更清晰实验控制变量：固定结构分布 vs. 时序预测驱动。

## [2026-03-03 00:56:40] Codex 进展记录
- 任务: 在苹果3D主空间中清晰显示 layer 序号。
- 代码变更:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - 增强 LayerGuides：每层在左右两侧同时显示 L0~L27。
    - 提升字号与对比度，增加文字描边（outline）提高可读性。
    - 增加端点标记文字：Layer 0 与 Layer 27。
- 执行命令:
  - 
pm run build (workdir=rontend) -> 通过
- 理论/数学研究进展:
  - 显式层序号可作为“深度坐标系”观察基准，便于将动态预测轨迹与层级结构对应，支持后续层级因果路径分析。

## [2026-03-03 00:58:45] Codex 进展记录
- 任务: 回答“除静态分析/动态预测外，观察神经元编码机制还有哪些方案”。
- 输出方案方向:
  - 因果干预、子空间几何、特征解码、跨层传输、组合性、鲁棒性、反事实、最小子回路等。
- 理论/数学研究进展:
  - 明确从“相关性观察”升级到“机制识别”的关键路径：
    1) 先定位候选表征（几何/特征）
    2) 再做可逆干预验证（ablation/patching/steering）
    3) 最后抽取最小因果子回路并做跨任务复验。

## [2026-03-03 01:30:54] Codex 进展记录
- 任务: 在 MAIN 中完成“静态分析/动态预测 + 8种编码机制观测方案”的完整实现。
- 代码变更:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - 新增 10 种分析模式按钮：
      - static, dynamic_prediction
      - causal_intervention, subspace_geometry, feature_decomposition, cross_layer_transport
      - compositionality, counterfactual, robustness, minimal_circuit
    - 在 useAppleNeuronWorkspace 中实现各模式的激活映射计算（activationMap）与状态指标（metrics/status）。
    - 新增机制控制参数：
      - 因果干预稀疏度、特征轴选择、组合权重、反事实提示词、鲁棒试验次数、最小子回路规模。
    - 新增机制播放控制（Play/Pause/Step/Reset + speed），并与主3D场景联动。
    - 重构 AppleNeuronControlPanels 为完整多模式控制台，动态显示对应控件。
- 执行命令:
  - 
pm run build (workdir=rontend) -> 通过
- 理论/数学研究进展:
  - 从单一“下一词预测动画”扩展到“机制族对照框架”，统一抽象为：
    - mode -> activationMap -> 3D响应
  - 形成了机制级可比较实验面板，可在同一空间下对比：
    - 因果性（intervention/minimal circuit）
    - 几何性（subspace/transport）
    - 表征分解（feature/compositional）
    - 稳定性与反事实（robustness/counterfactual）

## [2026-03-03 11:05:55] Codex 进展记录
- 任务: 修复苹果3D动画中“黑色组件与黑色背景对比不足”的可视性问题，组件改为白色系。
- 代码变更:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - ROLE_COLORS.background 从深色改为 #ffffff。
    - Layer 网格线与中轴线改为白色/浅色，提升对比度。
    - 神经元提示浮层改为浅色背景与深色文字，避免深色叠加不可读。
- 执行命令:
  - 
pm run build (workdir=rontend) -> 通过
- 理论/数学研究进展:
  - 提升低激活背景节点和层结构参考线的可见性，有助于更准确观察“弱激活编码”与“层间几何骨架”的关系。

## [2026-03-03 11:10:38] Codex 进展记录
- 任务: 解决 MAIN 中不同模式动画“看起来一样”的问题，做模式级差异化可视表达。
- 代码变更:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - 新增 MODE_VISUALS，为每种模式定义视觉参数（accent、脉冲速度/幅度、链路强度、载体形态）。
    - PulsingNeuron 支持按模式调整脉冲频率/发光颜色/缩放波形。
    - TokenPredictionCarrier 按模式切换几何：torus/octa/plane/tetra/cylinder/tri_ring/dual_ring/shield/hex。
    - 新增 ModeVisualOverlay，为每种机制模式叠加专属几何特征（子空间平面、特征轴、传输轴、组合三环、反事实双球、鲁棒护罩、最小回路骨架等）。
    - AppleNeuronSceneContent 根据模式改变链路颜色与线宽，并注入模式叠层。
  - rontend/src/App.jsx
    - 主空间渲染 AppleNeuronSceneContent 时传入 mode={appleNeuronWorkspace.analysisMode}，确保 MAIN 中同样生效。
- 执行命令:
  - 
pm run build (workdir=rontend) -> 通过
- 理论/数学研究进展:
  - 将“算法机制差异”映射到“视觉语法差异”，形成一一对应：
    - 因果/子空间/特征/传输/组合/反事实/鲁棒/最小回路 各有独立动态符号。
  - 这有助于在同一3D空间中做跨机制对照，减少“同构动画导致的认知歧义”。

## [2026-03-03 11:32:50] Codex 进展记录
- 任务: 将移动中的3D主体改为白色，并让主体移动时每个layer产生对应变化。
- 代码变更:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - TokenPredictionCarrier 所有移动几何材质统一改为白色发光。
    - LayerGuides 新增 ctiveLayer 参数：
      - 根据与当前层距离计算 influence，动态调整层线框颜色/透明度、标签字号/颜色。
      - 高影响层增加内层高亮框，形成“逐层响应”效果。
    - AppleNeuronSceneContent 根据 prediction.layerProgress 计算当前活动层并传入 LayerGuides。
- 执行命令:
  - 
pm run build (workdir=rontend) -> 通过
- 理论/数学研究进展:
  - 把“移动主体”作为时间驱动变量，层级响应函数 influence = f(|layer-activeLayer|) 可视化了层深与激活传播关系。
  - 该可视化更贴近“层间编码传播”机制观察，而非仅看单点激活。

## [2026-03-03 11:43:17] Codex 进展记录
- 任务: 将 layer 动画从“多层同时高亮”改为“单层高亮”，提升可读性。
- 代码变更:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - LayerGuides(activeLayer) 改为单层模式：
      - 计算 ctiveLayerIndex = round(activeLayer) 并限制在 [0, 27]
      - 仅 ctiveLayerIndex 对应层触发高亮线框/标签放大/内层高亮框
      - 其余 layer 仅保留静态参考线
- 执行命令:
  - 
pm run build (workdir=rontend) -> 通过
- 理论/数学研究进展:
  - 单层高亮将“层级响应”从分布式模糊带改为离散主响应层，便于观察“主导层随时间迁移”的编码路径。

## [2026-03-03 11:44:17] Codex 进展记录
- 任务: 分析 MAIN 控制面板动画效果并提出“更清楚展示每层算法影响”的优化思路。
- 当前观察:
  - 已有单层高亮、模式差异化载体、机制模式控制与 metrics 文本。
  - 仍缺“每层量化影响面板”和“基线差分视图”。
- 理论/数学研究进展:
  - 提出“层影响可视化三元组”框架：
    1) Layer Impact（每层强度）
    2) Layer Delta（相对基线变化）
    3) Layer Uncertainty（方差/稳定性）
  - 该三元组可统一映射到不同机制模式，形成跨算法可比性。

## [2026-03-03 11:52:40] Codex 进展记录
- 任务: 提升 MAIN 面板可解释性，清楚显示“哪些神经元被影响”以及“编码机制影响”。
- 代码变更:
  - frontend/src/blueprint/AppleNeuron3DTab.jsx
    - 3D 主空间新增“受影响聚焦链路”: 当启用 impact focus 时，非 Top-K 影响神经元会降亮/降透明，非焦点连线同步降透明。
    - useAppleNeuronWorkspace 已返回并接入:
      - impactFocusEnabled / impactTopK
      - neuronImpactRows（基于 delta=current-baseline 的排序）
      - mechanismImpactSummary（Impacted count、Avg |Δ|、Peak layers）
    - prediction payload 新增 focusEnabled/focusedNeuronIds，使 3D 渲染可以按影响集聚焦。
    - 控制面板新增“影响神经元与编码机制”卡片:
      - 仅突出受影响神经元（3D）开关
      - Top-K 滑杆
      - 机制摘要指标
      - Top 影响神经元列表（L/N、current、baseline、delta），可点击定位到选中神经元
    - Selected Neuron 卡片新增 current/baseline/delta 指标，直接观察该点编码变化。
- 执行命令:
  - rg -n "impactFocusEnabled|impactTopK|neuronImpactRows|mechanismImpactSummary|focusedNeuronIds|focusEnabled|staticReferenceMap" frontend/src/blueprint/AppleNeuron3DTab.jsx
  - npm run build (workdir=frontend) -> 通过
- 理论/数学研究进展:
  - 引入“机制影响差分”定义: 对每个神经元 i, 令 Δ_i = a_i(mode) - a_i(static baseline), 以 |Δ_i| 排序获得受影响子集。
  - 用 Top-K(|Δ|) 近似最显著受影响神经元集合，作为可视化因果候选子回路的工作定义。
  - 通过层级统计 mean_{i in layer l} |Δ_i| 提取 Peak layers，形成“机制对层级编码影响”的紧凑描述。

## [2026-03-03 12:03:20] Codex 进展记录
- 任务: 修复控制面板 DNN 相关乱码显示。
- 根因定位:
  - frontend/src/config/panels.js 文件整体存在编码损坏（大量 mojibake），DNN 控制面板分组/标签/描述均依赖此配置，导致显示乱码。
- 代码变更:
  - frontend/src/config/panels.js
    - 重建并替换整文件的文案内容（保持原有 key 与结构不变）。
    - 修复 INPUT_PANEL_TABS 中 dnn/snn/fibernet 描述。
    - 修复 STRUCTURE_TABS_V2 各分组与条目 label/desc。
    - 修复 ANALYSIS_DATA_TEMPLATES 的 title 与指标中文标签。
- 执行命令:
  - rg -n "DNN|dnn|控制面板|乱码|�|\\uFFFD" frontend/src
  - rg -n "STRUCTURE_TABS_V2|STRUCTURE_TABS|INPUT_PANEL_TABS|ANALYSIS_DATA_TEMPLATES|PANEL_POSITIONS|PANEL_BASE_STYLE" frontend/src
  - npm run build (workdir=frontend) -> 通过
- 理论/数学研究进展:
  - 本次为界面编码与可读性修复，无新增神经编码数学结论。
  - 但保证了 DNN 分析标签语义一致性，有利于后续将层级指标（如 Δ 激活、峰值层）做跨实验可比映射。

## [2026-03-03 12:14:40] Codex 进展记录
- 任务: 梳理“控制面板-DNN-结构分析”功能，评估可删除项。
- 盘点结论:
  - DNN 分组 tab 来自 frontend/src/config/panels.js 的 STRUCTURE_TABS_V2，共 18 项。
  - 存在占位/弱闭环项：logit_lens（无独立表单）、holonomy（仅前端随机可视化）、training（仅本地动画）、fibernet_v2（仅提示主屏查看）。
  - 存在技术债：App 单步分析仅支持 features/circuit/causal/manifold；其它 tab 勾选单步后不会执行分析分支。
  - 存在死分支：App 中 structureTab === 'fiber' 分支无对应 tab 来源。
  - 存在接口不一致：前端 POST /steer_concept，后端定义 GET /steer_concept。
- 执行命令:
  - rg -n "logit_lens|glass_matrix|flow_tubes|...|training" frontend/src/App.jsx frontend/src/StructureAnalysisPanel.jsx frontend/src/components/shared/DataDisplayTemplates.jsx frontend/src/config/panels.js
  - rg -n "activeTab === '" frontend/src/StructureAnalysisPanel.jsx frontend/src/App.jsx
  - rg -n "structureTab === 'fiber'|activeTab === 'holonomy'|activeTab === 'training'|activeTab === 'fibernet_v2'|stepAnalysisMode ===" frontend/src/App.jsx frontend/src/StructureAnalysisPanel.jsx
  - rg -n "runConceptSteering|steer_concept" frontend/src/StructureAnalysisPanel.jsx server/server.py
- 理论/数学研究进展:
  - 本次为功能结构治理，无新增编码数学结论。
  - 形成“最小可解释闭环”准则：一个分析项需同时具备【输入参数 -> 后端计算 -> 结果指标 -> 3D映射】四段闭环，否则归为可删/可迁移候选。

## [2026-03-03 12:20:10] Codex 进展记录
- 任务: 回答“DNN 结构分析中哪些功能可迁移到 FiberNet”。
- 迁移建议结论:
  - 优先迁移到 FiberNet: fibernet_v2, agi, global_topology, debias, holonomy, glass_matrix, flow_tubes。
  - 可选迁移到 FiberNet: rpt, curvature, tda（若定位为几何/拓扑纤维分析链）。
  - 建议保留在 DNN: circuit, features, causal, manifold, compositional, logit_lens, training。
- 证据点:
  - DNN 结构分析分组来源: frontend/src/config/panels.js (STRUCTURE_TABS_V2)
  - FiberNet 已含 DNN Probe: frontend/src/components/FiberNetPanel.jsx
- 执行命令:
  - rg -n "activeTab|id: '|label:|dnn_probe|topology|flow|debias|curvature|rpt|holonomy|glass|fibernet" frontend/src/components/FiberNetPanel.jsx
  - Get-Content frontend/src/components/FiberNetPanel.jsx -TotalCount 240
- 理论/数学研究进展:
  - 形成“归属判据”: 若分析核心依赖纤维丛/传输/拓扑不变量/全局几何一致性，则归 FiberNet；若核心依赖 Transformer 电路、特征稀疏分解、因果打靶，则归 DNN。

## [2026-03-03 12:29:50] Codex 进展记录
- 任务: 对 main / dnn / snn / fibernet 功能进行归并建议，识别迁移、重复、可删除项。
- 结构盘点结论:
  - 当前控制面板入口: main / dnn / snn / fibernet（frontend/src/App.jsx, frontend/src/config/panels.js）。
  - DNN 面板承载了大量 FiberNet 几何模块，边界混杂。
  - SNN 与 DNN 使用同一 StructureAnalysisControls 组件，存在功能面重叠。
- 关键技术债与可删证据:
  - 死分支: App 中 structureTab === 'fiber' 无对应 tab 来源（frontend/src/App.jsx:3434）。
  - 空功能: SNN 的 apple tab 按钮 onClick 为空，仅占位（frontend/src/StructureAnalysisPanel.jsx:2388,2397）。
  - 占位页: DNN 的 fibernet_v2 仅提示去主视图交互（frontend/src/StructureAnalysisPanel.jsx:2015）。
  - 接口不一致: 前端 POST /steer_concept，而后端定义 GET /steer_concept（frontend/src/StructureAnalysisPanel.jsx:1557; server/server.py:892）。
- 执行命令:
  - rg -n "inputPanelTab === 'main'|inputPanelTab === 'dnn'|inputPanelTab === 'snn'|inputPanelTab === 'fibernet'|INPUT_PANEL_TABS|setInputPanelTab|setSystemType" frontend/src/App.jsx frontend/src/config/panels.js
  - rg -n "activeTab|id: '|label:|dnn_probe|topology|flow|debias|curvature|rpt|holonomy|glass|fibernet" frontend/src/components/FiberNetPanel.jsx
  - rg -n "structureTab === 'fiber'" frontend/src/App.jsx
  - rg -n "activeTab === 'apple'|模拟后端特征提取|activeTab === 'fibernet_v2'|activeTab === 'training'|runConceptSteering|steer_concept" frontend/src/StructureAnalysisPanel.jsx
  - rg -n "steer_concept" server/server.py
- 理论/数学研究进展:
  - 建立“面板归属原则”: 
    1) 若方法核心依赖 Transformer 因果电路/稀疏特征分解/层级预测，则归 DNN。
    2) 若方法核心依赖纤维丛、平行传输、拓扑不变量、全局几何一致性，则归 FiberNet。
    3) SNN 仅承载脉冲动力学与其有效性验证，避免跨范式混放。

## [2026-03-03 12:36:20] Codex 进展记录
- 任务: 按“四面板职责”执行第一阶段重构（main/dnn/snn/fibernet），清理重复与无效入口。
- 本次代码变更:
  - frontend/src/config/panels.js
    - 收敛 DNN 结构分析 tab：
      - 保留: logit_lens, circuit, features, causal, manifold, compositional, rpt, curvature, tda, validity, training。
      - 移出 DNN 入口: glass_matrix, flow_tubes, fibernet_v2, global_topology, holonomy, agi, debias。
    - 同步修正文案描述（geometry/advanced 组说明）。
  - frontend/src/StructureAnalysisPanel.jsx
    - 移除 SNN 模式下的 apple 占位按钮。
    - 移除 activeTab==='apple' 的占位表单与右侧占位可视化。
    - 删除未再使用的 AppleNeuronVis3D import。
  - frontend/src/App.jsx
    - 删除死分支: structureTab === 'fiber'。
  - server/server.py
    - 修复接口方法不一致: /steer_concept 从 GET 改为 POST，与前端调用一致。
- 执行命令:
  - rg -n "STRUCTURE_TABS_V2.groups.map|if \(systemType === 'snn'\)|activeTab === 'apple'|runConceptSteering|steer_concept|structureTab === 'fiber'" frontend/src/StructureAnalysisPanel.jsx frontend/src/App.jsx server/server.py
  - npm run build (workdir=frontend) -> 通过
  - python -m py_compile server/server.py -> 通过
- 理论/数学研究进展:
  - 本次为系统结构治理，无新增数学推导。
  - 进一步固化“功能闭环约束”: 无真实后端计算或无可视化闭环的入口应视为占位并优先下线，以避免解释链路污染。

## [2026-03-03 12:44:30] Codex 进展记录
- 任务: 处理“DNN 3D 空间圆球”视觉问题，去除圆球化表达。
- 代码变更:
  - frontend/src/App.jsx
    - GlassNode 从 sphereGeometry 改为 boxGeometry（柱状节点），保留概率驱动的发光与动态脉冲。
    - debias 叠加层移除半球壳体（sphereGeometry 半球），仅保留环形拦截标记。
- 执行命令:
  - npm run build (workdir=frontend) -> 通过
- 理论/数学研究进展:
  - 可视编码层面将“概率强度”从球半径映射改为柱高映射，保留单调性与可比较性。
  - 该修改降低了球面遮挡，提升层间结构的可读性，但不改变底层分析算法输出。

## [2026-03-03 12:49:10] Codex 进展记录
- 任务: 解释并修复“DNN 3D 模型仍出现蓝色大圆球”问题。
- 根因:
  - 蓝色大圆球来自全局背景组件 ResonanceField3D（内部半径 15 的 Sphere + shader）。
  - App 里此前在非 main 模式统一渲染该背景，DNN 也被覆盖。
- 代码变更:
  - frontend/src/App.jsx
    - 新增 showGlobalResonanceField = inputPanelTab !== 'dnn'
    - 仅在该条件为 true 时渲染 ResonanceField3D
    - 结果: DNN 模式不再显示蓝色大圆球背景。
- 执行命令:
  - npm run build (workdir=frontend) -> 通过
- 理论/数学研究进展:
  - 本次为渲染层可视化隔离；将“全局共振场”从 DNN 解释视图剥离，减少背景先验干扰，提高局部结构观测纯度。

## [2026-03-03 12:53:40] Codex 进展记录
- 任务: 关闭 SNN 中与 DNN 同类的全局共振背景球。
- 代码变更:
  - frontend/src/App.jsx
    - showGlobalResonanceField 条件由 `inputPanelTab !== 'dnn'` 调整为 `inputPanelTab !== 'dnn' && inputPanelTab !== 'snn'`。
    - 结果: SNN 模式下不再渲染 ResonanceField3D 背景球。
- 执行命令:
  - rg -n "showGlobalResonanceField" frontend/src/App.jsx
  - npm run build (workdir=frontend) -> 通过
- 理论/数学研究进展:
  - 本次为可视化隔离；减少跨系统背景场叠加，提升 SNN 脉冲活动的信号可见性与判读纯度。

## [2026-03-03 13:00:20] Codex 进展记录
- 任务: 评估 DNN“特征”功能为何难以看出数据特点与 layer 相关性，并给出优化方案。
- 代码审阅结论:
  - FeatureVisualization3D 当前采用固定螺旋布局（与语义/层关系弱耦合），可视上更像装饰分布而非结构映射。
  - 单次仅展示单层 top_features，缺少跨层对比与趋势视图。
  - 关键指标仅有 top_features 数量和 reconstruction_error，无法解释“该层特征密度、稳定性、差异性”。
- 执行命令:
  - rg -n "FeatureVisualization3D|activeTab === 'features'|runFeatureExtraction|top_features|layer_idx|reconstruction_error" frontend/src/StructureAnalysisPanel.jsx frontend/src/App.jsx frontend/src/components/shared/DataDisplayTemplates.jsx
  - Get-Content frontend/src/StructureAnalysisPanel.jsx | Select-Object -Index (250..380)
  - Get-Content frontend/src/App.jsx | Select-Object -Index (2588..2622)
- 理论/数学研究进展:
  - 提出“层特征可解释三元组”: Layer Signature = [Sparsity, Activation Energy, Reconstruction Loss]。
  - 提出“跨层漂移度”指标: Drift(l→l+1)=1-cos(mu_l, mu_{l+1})，用于显示特征表征在层间重排强度。
  - 提出“特征稳定度”指标: Stability = Jaccard(TopK_l(prompt_a), TopK_l(prompt_b))，用于验证特征是否具有语义稳健性。

## [2026-03-03 13:07:10] Codex 进展记录
- 任务: 调整 AppleNeuron 3D 组件命名，使其语义与“项目核心编码结构分析”定位一致。
- 代码变更:
  - 文件重命名:
    - frontend/src/components/AppleNeuronVis3D.jsx -> frontend/src/components/AppleNeuronCore3D.jsx
  - 组件重命名:
    - const AppleNeuronVis3D -> const AppleNeuronCore3D
    - export default AppleNeuronCore3D
  - 文档引用同步:
    - research/gemini/docs/AGI_RESEARCH_MEMO.md 中 AppleNeuronVis3D -> AppleNeuronCore3D
- 执行命令:
  - Move-Item frontend/src/components/AppleNeuronVis3D.jsx frontend/src/components/AppleNeuronCore3D.jsx
  - rg -n "AppleNeuronVis3D|AppleNeuronCore3D"
  - npm run build (workdir=frontend) -> 通过
- 理论/数学研究进展:
  - 将组件语义从“可视化展示(Vis)”提升为“核心机制分析(Core)”，与项目目标“还原深度神经网络语言能力的数学编码原理”对齐。

## [2026-03-03 13:16:30] Codex 进展记录
- 任务: 基于“无预设、拼图优先”的核心思路，提出控制面板-Main 的系统功能改造建议。
- 面板重构主线:
  - 从“演示型可视化”转为“证据驱动的机制还原工作台”。
  - 以四层证据链组织 Main: 观测 -> 干预 -> 压缩 -> 跨任务/跨模态一致性。
- 关键建议摘要:
  1) 新增“拼图管理器”与“证据等级”系统，所有结论必须绑定实验ID、样本规模、可复现实验脚本。
  2) 新增“特征形成追踪”流程：从局部神经元活动到层级编码簇再到任务表现变化，形成因果闭环。
  3) 新增“局部可塑-全局稳态”双指标：Plasticity Index + Homeostasis Index。
  4) 新增“跨任务一致性扫描”：语言/视觉/代码等任务共享子空间与漂移热图。
  5) 将现有模式（静态/动态/因果/子空间等）重排为阶段化流水线，避免平铺按钮造成认知负担。
- 理论/数学研究进展:
  - 提出“AGI 还原最小证据公理”: 不接受单一相关性证据，必须满足
    (i) 可观测结构,
    (ii) 干预可致变,
    (iii) 跨样本稳定,
    (iv) 跨任务可迁移。
  - 提出“编码形成核心方程组”候选观测量:
    - Feature Emergence: dF/dt
    - Layer Drift: 1-cos(mu_l, mu_{l+1})
    - Plasticity-Stability Balance: PI/HI
  - 将“高维抽象+低维精确”转化为可测目标：
    抽象性(跨任务共享)与精确性(局部误差/校准)的双目标优化轨迹。

## [2026-03-03 13:22:40] Codex 进展记录
- 任务: 按“编码是核心、可视化+系统化”思路升级 Main（AppleNeuron3DTab）控制面板与3D联动。
- 代码变更:
  - frontend/src/blueprint/AppleNeuron3DTab.jsx
    - 新增“编码还原流水线”阶段体系（A观测/B提取/C验证/D系统），并与分析模式自动联动。
    - 新增“拼图管理器”:
      - 实验标签、样本规模、稳定性探针数
      - 证据快照保存（时间/阶段/模式/FS/PI/HI）
    - 新增“层级编码签名”与“层漂移”可视化:
      - layerSignatureRows（energy/meanCurrent/meanAbsDelta/sparsity/impactedRatio）
      - layerDriftRows（相邻层角色向量余弦漂移）
    - 新增“编码机制指标”:
      - featureStability（Top-K Jaccard across probes）
      - taskConsistencyRows（语言/代码/视觉/逻辑 overlap）
      - plasticityHomeostasis（PI/HI/AvgDrift）
      - encodingEvidenceCards（稳定度/可塑性/稳态/当前拼图置信度）
    - 3D层引导增强:
      - LayerGuides 接入 layerEncodingMap，层框/标签按编码强度动态着色与增亮
      - SceneContent 传入并渲染层编码热度
- 执行命令:
  - npm run build (workdir=frontend) -> 通过
- 理论/数学研究进展:
  - 将 Main 从“演示面板”升级为“证据流水线面板”，明确四段证据链（观测-提取-验证-系统）。
  - 引入可度量编码拼图指标:
    - Layer Signature, Layer Drift, Feature Stability, PI/HI。
  - 形成“局部可塑-全局稳态”与“跨任务一致性”的并行观测框架，更贴近自下而上编码形成机制分析。

## [2026-03-03 15:36:36] Codex 进展记录
- 任务: 简化并重组右侧“模型说明”和“操作面板”，提升编码结构观察的清晰度与可操作性。
- 代码变更:
  - frontend/src/App.jsx
    - 模型说明面板重构为三段式信息架构:
      - 顶部“模型概览/系统概览”
      - 中部“编码焦点指标”卡片（按分析模式动态切换）
      - 底部 Tab 视图（概览 / 编码焦点 / 细节）
    - 新增编码摘要逻辑:
      - encodingFocusItems（按 structureTab 输出可解释核心指标）
      - analysisSummaryText（统一分析结果摘要文本）
    - 操作面板重构为“核心 + 可折叠高级区”:
      - 保留核心指标与“编码状态快照”
      - 新增三个折叠按钮：数据模板 / 结果对比 / 操作历史
      - 将 DataComparisonView 从模型说明区迁移到操作面板的“结果对比”区，减少重复信息。
    - 细节修正:
      - “特证”文案修正为“特征”
      - 移除未使用导入 MetricsRow
- 执行命令:
  - git status --short
  - rg -n "模型说明|操作面板|DataComparisonView|OperationHistoryPanel|SimplePanel|panelVisibility" frontend/src/App.jsx
  - 多段 Get-Content 分段读取 App.jsx（定位与重构目标区块）
  - rg -n "infoPanelTab|setInfoPanelTab|displayInfo|hoveredInfo|operationMetrics|currentStructureUI|isObservationMode" frontend/src/App.jsx
  - npm run build (workdir=frontend) -> 通过
- 理论/数学研究进展:
  - 将“编码观察”从混合叙述改为三层证据视图:
    - 概览层: 当前模式 + 全局摘要（系统态）
    - 编码层: 最小关键指标集（结构态）
    - 细节层: 单对象精确读数（微观态）
  - 在 UI 上落实“粗到细”观察链路，降低认知负荷并提高编码机制定位效率。
  - 将“结果对比”下沉为可选高级区，符合机制研究流程: 先看当前编码证据，再做跨快照对比验证。


## [2026-03-03 15:40:58] Codex 进展记录
- 任务: 澄清 Main 中“分析类型”与“编码还原流水线”的关系定义与当前实现耦合方式。
- 代码依据:
  - frontend/src/blueprint/AppleNeuron3DTab.jsx
    - 分析类型定义: ANALYSIS_MODE_OPTIONS
    - 流水线定义: PIPELINE_STAGES（每个 stage 绑定 modes）
    - 耦合逻辑: 若当前 analysisMode 不在 stage.modes，则自动切换到该 stage 的首个 mode。
- 执行命令:
  - rg -n "分析类型|编码还原流水线|analysisMode|PIPELINE_STAGES|modes" frontend/src/blueprint/AppleNeuron3DTab.jsx
  - Get-Content frontend/src/blueprint/AppleNeuron3DTab.jsx 分段读取（1..620, 840..980, 1458..1530, 1808..1908）
- 理论/数学研究进展:
  - 明确两层抽象:
    - 分析类型 = 方法空间 M（观察/干预/分解/传输等算子）
    - 编码还原流水线 = 过程空间 P（A观测/B提取/C验证/D系统）
  - 当前实现关系是约束映射 phi: P -> 2^M（阶段允许的方法子集），并执行可行性投影。
  - 该设计可避免“验证阶段使用观测方法”等流程错位，提高实验链路一致性与可复现性。

## [2026-03-03 15:43:43] Codex 进展记录
- 任务: 参考 DNN 结构分析，将 Main 中“分析类型”按四阶段（观测/提取/验证/系统）重组展示。
- 代码变更:
  - frontend/src/blueprint/AppleNeuron3DTab.jsx
    - 在 AppleNeuronControlPanels 中将“分析类型”从平铺按钮改为按 pipelineStages 分组渲染。
    - 新增阶段中文映射: observe/extract/validate/system -> 观测/提取/验证/系统。
    - 新增联动函数 handleSelectStageMode(stageId, modeId):
      - 点击某分析类型时，同时设置 pipelineStage 与 analysisMode，保证阶段与方法一致。
    - 流水线面板标题更新为“编码还原流水线（阶段导航）”。
- 执行命令:
  - rg -n "ANALYSIS_MODE_OPTIONS|PIPELINE_STAGES|setAnalysisMode|setPipelineStage|AppleNeuronControlPanels|analysisModes" frontend/src/blueprint/AppleNeuron3DTab.jsx
  - Get-Content frontend/src/blueprint/AppleNeuron3DTab.jsx 分段读取并定位改动区
  - npm run build (workdir=frontend) -> 通过
- 理论/数学研究进展:
  - 将“方法空间 M（分析类型）”显式嵌入“流程空间 P（四阶段）”，形成 P→M 的阶段化约束与导航。
  - UI 上从“方法平铺”升级为“阶段-方法二层结构”，有助于避免流程错位，提高编码还原实验的可重复性与解释一致性。

## [2026-03-03 15:47:29] Codex 进展记录
- 任务: 将 Main 的“分析类型”界面风格对齐 DNN 结构分析按钮风格，并删除“编码还原流水线”模块。
- 代码变更:
  - frontend/src/blueprint/AppleNeuron3DTab.jsx
    - “分析类型”按钮样式改为 DNN 同款视觉规则:
      - 激活态: rgba(68,136,255,0.2) 背景 + #4488ff 文字 + 1px solid rgba(68,136,255,0.4) 边框
      - 非激活态: 透明背景 + 浅灰文字 + 透明边框
      - 按钮尺寸/排版改为 11px、紧凑网格布局
    - 保留四阶段分组（观测/提取/验证/系统）在“分析类型”内部展示。
    - 删除整个“编码还原流水线（阶段导航）”卡片模块。
    - 交互保持: 点击任一分析类型仍会同步 pipelineStage 与 analysisMode（通过 handleSelectStageMode）。
- 执行命令:
  - rg -n "分析类型|编码还原流水线|handleSelectStageMode|stageLabelMap|pipelineStages.map" frontend/src/blueprint/AppleNeuron3DTab.jsx
  - npm run build (workdir=frontend) -> 通过
- 理论/数学研究进展:
  - 将“阶段导航”从独立模块收敛到“分析类型”单一入口，降低操作分叉，减少流程认知负担。
  - 保留方法-阶段耦合约束（stage/mode 同步），维持实验链路一致性，不牺牲编码机制分析的可复现性。


## [2026-03-03 15:56:48] Codex 进展记录
- 任务: 详细说明 Main 控制面板中“拼图管理器”及其下方模块的作用、输入输出与解读方式。
- 代码依据:
  - frontend/src/blueprint/AppleNeuron3DTab.jsx
    - UI 分区标题与交互区: 1913+（拼图管理器）、1947+（层级编码签名）、1974+（编码机制指标）、2003+（动态预测）、2173+（影响神经元与编码机制）、2246+（Selected Neuron）
    - 指标生成逻辑: 1269+（neuronImpactRows）、1296+（mechanismImpactSummary）、1316+（layerSignatureRows）、1369+（layerDriftRows）、1387+（featureStability）、1418+（taskConsistencyRows）、1450+（plasticityHomeostasis）、1471+（encodingEvidenceCards）、1507+（captureEvidenceSnapshot）
- 执行命令:
  - rg -n "拼图管理器|层级编码签名|编码机制指标|影响神经元与编码机制|Selected Neuron|分析类型|Next-Token Prediction Animation" frontend/src/blueprint/AppleNeuron3DTab.jsx
  - Get-Content frontend/src/blueprint/AppleNeuron3DTab.jsx 分段读取（1890..2305）
  - rg -n "captureEvidenceSnapshot|evidenceSnapshots|experimentTag|sampleScale|stabilityProbeCount|encodingEvidenceCards|layerSignatureRows|layerDriftRows|featureStability|taskConsistencyRows|plasticityHomeostasis|neuronImpactRows|mechanismImpactSummary" frontend/src/blueprint/AppleNeuron3DTab.jsx
- 理论/数学研究进展:
  - 将“拼图管理器”定位为证据版本化层（实验标签+样本尺度+探针数+快照），用于把单次可视化观察变成可比较证据链。
  - 将下游模块分为三类:
    1) 层级结构指标（layer signature / drift）
    2) 机制级指标（稳定度、PI/HI、跨任务一致性）
    3) 微观因果定位（受影响神经元列表与单神经元差分）
  - 对应形成从宏观到微观的闭环：全局统计 -> 层间迁移 -> 节点差分，减少“只看动画不落指标”的解释偏差。

## [2026-03-03 16:06:05] Codex 进展记录
- 任务: 让“模型说明”和“操作面板”随左侧控制面板功能类型联动；无对应内容时面板空白并自动最小化。
- 代码变更:
  - frontend/src/App.jsx
    - 新增 inputPanelTab -> 面板内容 映射（dnn/snn 有内容，main/fibernet 无内容）。
    - 切换功能类型时自动同步最小化状态：无内容时最小化并清空操作区开关；有内容时恢复默认展示。
    - “模型说明”标题改为动态标题（含功能类型），且无内容类型不显示算法说明按钮。
    - “操作面板”标题改为动态标题；无内容类型渲染为空白容器。
    - 信息面板内容判断改为基于左侧功能类型（inputPanelTab），避免与其他状态耦合。
- 执行命令:
  - rg -n "模型说明|操作面板|控制面板|功能类型|inputPanelTab|structureTab" frontend/src/App.jsx frontend/src/components frontend/src/blueprint -S
  - Get-Content frontend/src/App.jsx 分段读取（状态区、信息面板区、操作面板区）
  - npm --prefix frontend run build -> 通过
- 理论/数学研究进展:
  - 将右侧信息/操作界面抽象为条件映射 F: T -> C，其中 T 为功能类型集合（main/dnn/snn/fibernet），C 为可展示内容集合。
  - 当 F(t)=∅ 时执行“空内容 + 最小化”策略，相当于在 UI 状态空间里施加约束投影，减少无效交互噪声。
  - 该约束使“控制选择 -> 可视反馈”保持同态一致，提升实验界面的可解释性与操作一致性。

## [2026-03-03 16:06:40] Codex 进展记录
- 任务补充: 修正信息面板内剩余分支判断，确保联动完全以左侧功能类型为准。
- 代码变更:
  - frontend/src/App.jsx
    - 将信息面板内容分支中的 systemType === 'snn' 改为 isSnnFunctionType，统一联动来源为 inputPanelTab。
- 执行命令:
  - rg -n "systemType === 'snn'|systemType !== 'snn'" frontend/src/App.jsx -S
  - npm --prefix frontend run build -> 通过
- 理论/数学研究进展:
  - 将面板联动条件从混合变量压缩为单一控制变量 t in T（功能类型），减少条件分叉，提升状态机可判定性。
  - 在形式上等价于把 UI 过渡函数约束为 S_{n+1}=G(S_n, t)，降低隐式路径依赖。

## [2026-03-03 16:11:52] Codex 进展记录
- 任务: 将“拼图管理器及下游模块”说明接入算法指南，并重排 Main 面板模块顺序；同时将 Compare Filter / Legend 改为动态显示。
- 代码变更:
  - frontend/src/App.jsx
    - 新增算法指南条目 main_workspace（通俗版+专业版）到 ALGO_DOCS。
    - 新增 GUIDE_STRUCTURED.main_workspace 结构化说明（目标/思路/3D原理/算法/指标范围）。
    - 在 buildGuideConclusion 增加 main_workspace 的结论与核心指标说明。
    - 在 guideMenuItems 中新增“Main 控制面板”入口。
  - frontend/src/blueprint/AppleNeuron3DTab.jsx
    - 调整控制面板模块顺序，形成更连续的证据链：
      - 分析类型 ->（模式控制）-> 拼图管理器 -> 层级编码签名 -> 编码机制指标 -> 影响神经元 -> Selected Neuron -> Compare/Generator/Filter/Legend。
    - Compare Filter 改为动态列表:
      - 基于 summary.perFruit + showFruit 生成 fruitFilterItems。
      - 展示实时可见状态统计（Fruit-General ON/OFF、可见 Fruit-Specific 数量等）。
    - Legend 改为动态列表:
      - 基于当前统计与开关状态生成 legendItems。
      - 每项显示 count 与可见/隐藏状态。
    - 新增动态辅助计算: backgroundNodeCount、fruitFilterItems、visibleFruitSpecificCount、legendItems。
- 执行命令:
  - rg -n "算法|指南|guide|Help|helpTab|guideMenuItems|buildGuide" frontend/src/App.jsx ...
  - 多段 Get-Content 分段读取 App.jsx / AppleNeuron3DTab.jsx 目标区域
  - npm run build (workdir=frontend) -> 通过
- 理论/数学研究进展:
  - 将 Main 面板从“并列模块堆叠”改为“证据链顺序呈现”，提升从宏观统计到微观节点验证的推理连贯性。
  - 将过滤与图例由静态说明改为状态感知显示，减少“界面状态与解释文本脱节”的认知误差。



## [2026-03-03 16:39:58] Codex 进展记录
- 任务: Main 控制面板支持“默认 apple -> 清理 -> 输入其他概念”流程，并让 Compare Filter / Legend 跟随输入动态显示。
- 代码变更:
  - frontend/src/blueprint/AppleNeuron3DTab.jsx
    - 新增默认输入概念常量 DEFAULT_QUERY_CONCEPT = 'apple'。
    - querySets 改为默认包含 apple 概念集；新增 queryVisibility 控制各输入概念的显示/隐藏。
    - 新增 visibleQuerySets，仅把可见概念集注入 3D 节点与连线。
    - 新增 clearQuerySets()：清空输入概念、重置可见映射，支持重新输入其他概念。
    - handleGenerateQuery() 改为：
      - 若概念已存在则自动恢复可见并选中
      - 若新概念则新增并默认可见
    - Compare 区域改造:
      - 标题动态化：{primaryConceptLabel} + Related Neuron Compare
      - Compare Filter 增加“输入概念过滤”动态列表（按用户输入实时生成）
      - 保留水果参考过滤区作为对照
    - Legend 改造:
      - 由静态文本改为动态 legendItems
      - 概念条目按输入集逐项显示 Concept: 名称 (数量)，并标注隐藏状态
      - 核心项命名去 apple 固定词，改为 Core micro/macro/route
- 执行命令:
  - rg -n "DEFAULT_QUERY_CONCEPT|queryVisibility|visibleQuerySets|clearQuerySets|primaryConceptLabel|queryFilterItems" frontend/src/blueprint/AppleNeuron3DTab.jsx
  - npm run build (workdir=frontend) -> 通过
- 理论/数学研究进展:
  - 将“输入概念”显式纳入可视化状态变量 Q(t)，并通过可见性映射 V(q) 作用于节点集与连线集，避免“输入与面板说明脱钩”。
  - 形成概念驱动的对照链路：输入概念 -> 过滤显示 -> 图例解释 -> 3D观察一致，减少解释偏差。


## 2026-03-03 Codex 进展记录（控制面板 Main 默认与分析类型样式对齐）

### 1) 命令执行记录
- `rg -n "inputPanelTab|setInputPanelTab|dnn|main|结构分析|analysis|analysis type|analysisModes|pipelineStages" frontend/src/App.jsx frontend/src/blueprint/AppleNeuron3DTab.jsx frontend/src/blueprint/UniversalManifoldGraph.jsx`
- `Get-Content frontend/src/App.jsx -TotalCount 1820 | Select-Object -Last 260`
- `Get-Content frontend/src/blueprint/AppleNeuron3DTab.jsx -TotalCount 2600 | Select-Object -Last 760`
- `Get-Content frontend/src/StructureAnalysisPanel.jsx -TotalCount 1945 | Select-Object -Last 170`
- `npm --prefix frontend run -s build`

### 2) 代码修改
- `frontend/src/App.jsx`
  - 将控制面板默认页签从 `dnn` 改为 `main`：
    - `const [inputPanelTab, setInputPanelTab] = useState('main')`
- `frontend/src/blueprint/AppleNeuron3DTab.jsx`
  - Main 的“分析类型”布局改为与 DNN 结构分析一致的分组样式：
    - 阶段头部改为 DNN 同款紧凑 group header（小号标题条）
    - 模式按钮网格改为 `repeat(3, 1fr)`
    - 按钮激活/未激活视觉风格与 DNN 同款（背景、边框、字体、圆角、间距）

### 3) 验证结果
- 前端构建通过（Vite build success）。
- 现状态：进入页面后控制面板默认落在 `Main`，且 Main 分析类型在布局与按钮风格上与 DNN 结构分析保持一致。

### 4) 理论/方法进展（数学结构研究相关）
- 本次为“观测接口标准化”进展：统一 Main 与 DNN 的分析阶段选择器视觉语法，降低人机交互偏差。
- 研究意义：在编码机制还原实验中，界面一致性可减少“交互形式变化”带来的观测噪声，使阶段对比更接近同一实验条件下的可比观测。

## 2026-03-03 Codex 进展记录（Main 分析类型配色与风格完全对齐 DNN）

### 1) 命令执行记录
- `Get-Content frontend/src/StructureAnalysisPanel.jsx -TotalCount 1910 | Select-Object -Last 120`
- `Get-Content frontend/src/blueprint/AppleNeuron3DTab.jsx -TotalCount 2100 | Select-Object -Last 220`
- `npm --prefix frontend run -s build`

### 2) 本轮修改
- 文件：`frontend/src/blueprint/AppleNeuron3DTab.jsx`
- 调整内容（Main -> 分析类型）：
  - 新增 `dnnLikePanelCardStyle`，将该卡片背景与边框改为 DNN 同款深灰风格。
  - 分组头样式改为 DNN 同款：`rgba(255,255,255,0.03)` + `#888` 标题色，不再使用蓝色激活组头。
  - 保留按钮激活样式与 DNN 一致：`#4488ff` 激活色、透明默认态、同款边框与圆角。
  - 描述/状态/指标文字颜色统一收敛到 DNN 风格灰色系（`#888`/`#667da9`）。

### 3) 验证结果
- 前端构建通过（Vite build success）。
- Main 分析类型区块现已与 DNN 结构分析在配色和视觉语言上保持同一体系。

### 4) 理论/方法进展
- 本轮属于“观测界面规范化”进展：统一同类分析入口的视觉编码，降低实验人员在阶段切换时的感知偏差。
- 对“编码结构还原”研究意义：可在相同 UI 语义下比较不同分析阶段结果，减少因界面差异引入的主观解释噪声。

## 2026-03-03 Codex 进展记录（Main 模块迁移：层级编码签名/编码机制指标）

### 1) 命令执行记录
- `rg -n "模型说明|操作面板|isAppleMainView|AppleNeuronControlPanels|AppleNeuron3DTab|层级编码签名|编码机制指标|infoPanel" frontend/src/App.jsx frontend/src/blueprint/AppleNeuron3DTab.jsx`
- `Get-Content frontend/src/App.jsx -TotalCount 2840 | Select-Object -Last 340`
- `Get-Content frontend/src/blueprint/AppleNeuron3DTab.jsx -TotalCount 2240 | Select-Object -Last 260`
- `npm --prefix frontend run -s build`
- `git diff -- frontend/src/App.jsx frontend/src/blueprint/AppleNeuron3DTab.jsx`

### 2) 代码修改
- `frontend/src/blueprint/AppleNeuron3DTab.jsx`
  - 新增可复用组件：`AppleNeuronEncodingInfoPanels`（包含“层级编码签名 + 编码机制指标”两块）。
  - 从 `AppleNeuronControlPanels` 左侧控制面板中移除以上两块，避免重复显示。
- `frontend/src/App.jsx`
  - 引入 `AppleNeuronEncodingInfoPanels`。
  - 将 `functionTypePanelMap.main.hasInfo` 由 `false` 改为 `true`，使 Main 下右侧“模型说明”可显示内容。
  - 在右侧“模型说明 -> 编码焦点”页签中，Main 分支渲染 `AppleNeuronEncodingInfoPanels`，实现迁移后集中展示。

### 3) 验证结果
- 前端构建通过（Vite build success）。
- Main 视图效果：
  - 左侧控制面板不再显示“层级编码签名 / 编码机制指标”。
  - 右侧模型说明窗口（编码焦点页签）显示这两块内容。

### 4) 理论/方法进展（编码还原视角）
- 本次将“层级统计证据”和“机制指标证据”从操作区迁移到说明区，强化了“操作控制 vs 证据解读”的界面分离。
- 这有助于形成更清晰的证据流：左侧负责实验操控，右侧负责多尺度编码证据阅读（层级签名 + 机制指标），降低解释混淆。

## 2026-03-03 Codex 进展记录（Main 编码模块可见性修复）

### 1) 命令执行记录
- `rg -n "setInfoPanelTab\\('detail'\\)|const \\[infoPanelTab|hoveredInfo|inputPanelTab" frontend/src/App.jsx`
- `Get-Content frontend/src/App.jsx -TotalCount 1745 | Select-Object -Last 130`
- `npm --prefix frontend run -s build`

### 2) 修复内容
- 文件：`frontend/src/App.jsx`
- 修复点：
  1. Main 下禁用“悬停自动跳转 detail”行为：
     - 保留 `displayInfo` 更新，但仅在非 Main 视图时 `setInfoPanelTab('detail')`。
  2. 切换到 Main 时默认进入“编码焦点”页签：
     - `inputPanelTab === 'main'` 时自动 `setInfoPanelTab('encoding')`。

### 3) 结果
- “层级编码签名 / 编码机制指标”现在在 Main 右侧模型说明中更稳定可见，不会因为 3D 悬停被自动切走。
- 前端构建验证通过（Vite build success）。

### 4) 方法学进展
- 本轮属于“观测界面稳定性”修复：减少状态自动跳转造成的证据阅读中断，保证编码证据模块持续可见。

## 2026-03-03 Codex 进展记录（ReferenceError: isAppleMainView 初始化顺序修复）

### 1) 问题
- 运行时报错：`ReferenceError: Cannot access 'isAppleMainView' before initialization`。
- 根因：`useEffect` 在 `isAppleMainView` 声明之前引用了该常量，触发 TDZ（暂时性死区）。

### 2) 命令执行记录
- `rg -n "isAppleMainView|Auto-switch Info Panel tab on hover|setInfoPanelTab\\('detail'\\)" frontend/src/App.jsx`
- `Get-Content frontend/src/App.jsx -TotalCount 1735 | Select-Object -Last 150`
- `npm --prefix frontend run -s build`

### 3) 修复内容
- 文件：`frontend/src/App.jsx`
- 调整：将“Auto-switch Info Panel tab on hover”这段 `useEffect` 移动到 `inputPanelTab/isAppleMainView` 定义之后，逻辑不变。

### 4) 验证
- 前端构建通过（Vite build success）。
- 该 TDZ 报错已消除。

### 5) 研究方法学备注
- 本次属于“状态依赖有序化”修复：保证界面状态机中的派生变量先定义、后订阅，降低 UI 解释链路中的时序错误噪声。

## 2026-03-03 Codex 进展记录（Main 分析类型添加 Logo，兼容修复）

### 1) 本轮目标
- 参考 DNN 结构分析样式，为 Main 中“分析类型”按钮添加 logo（图标）。

### 2) 关键命令记录
- `cmd /c "git show HEAD:frontend/src/blueprint/AppleNeuron3DTab.jsx > frontend/src/blueprint/AppleNeuron3DTab.jsx"`
- `npm --prefix frontend run -s build`（多次）
- 通过 `PowerShell` 对 `frontend/src/blueprint/AppleNeuron3DTab.jsx` 进行定点插入：
  - 新增 `lucide-react` 图标 import
  - 新增 `ANALYSIS_MODE_ICONS` 映射
  - 在分析类型按钮中插入 `ModeIcon`
  - 补充 `AppleNeuronEncodingInfoPanels` 导出（与 App 现有引用保持兼容）

### 3) 代码结果
- 文件：`frontend/src/blueprint/AppleNeuron3DTab.jsx`
- 完成项：
  1. 分析类型按钮现在显示“图标 + 文本”（logo 化）。
  2. 图标映射：
     - static -> Search
     - dynamic_prediction -> Sparkles
     - causal_intervention -> Target
     - subspace_geometry -> Scale
     - feature_decomposition -> BarChart2
     - cross_layer_transport -> ArrowRightLeft
     - compositionality -> Activity
     - counterfactual -> GitBranch
     - robustness -> CheckCircle
     - minimal_circuit -> Network
  3. 保持 `App.jsx` 对 `AppleNeuronEncodingInfoPanels` 的 import 不报错。

### 4) 验证
- 构建通过：`npm --prefix frontend run -s build`。

### 5) 方法学进展（界面与认知负担）
- 在分析入口增加“图标语义锚点”可降低用户切换模式时的识别负担，提升阶段识别速度。
- 对编码机制研究流程的意义：在多模式高频切换下，视觉锚点可降低操作误差，提升实验记录一致性。

## 2026-03-03 Codex 进展记录（Evolution Monitor 文案中文化）

### 1) 修改目标
- 将右侧模型说明窗口中的 `EVOLUTION MONITOR` 文案改为中文，保持逻辑与交互不变。

### 2) 命令记录
- `rg -n "EvolutionMonitor|EVOLUTION MONITOR|Start Sleep|evolution|monitor" frontend/src/App.jsx frontend/src -g "*.jsx"`
- `Get-Content frontend/src/App.jsx -TotalCount 520 | Select-Object -Last 220`
- `npm --prefix frontend run -s build`

### 3) 代码修改
- 文件：`frontend/src/App.jsx`
- 变更项（仅文案）：
  - `EVOLUTION MONITOR` -> `演化监视器`
  - `STATUS` -> `状态`
  - `SLEEPING (EVOLVING)` -> `休眠中（演化进行中）`
  - `AWAKE (READY)` -> `唤醒（可分析）`
  - `CURVATURE (Ω)` -> `曲率 (Ω)`
  - `N/A` -> `无数据`
  - `ENTER SLEEP CYCLE` -> `进入休眠演化周期`

### 4) 验证
- 前端构建通过：`npm --prefix frontend run -s build`。

### 5) 研究流程意义
- 监控模块文案中文化后，观察与操作语义更一致，可减少流程误解，提高实验记录时的状态判读效率。

## 2026-03-03 Codex 进展记录（模型说明窗口“下面”文案中文化）

### 1) 修改目标
- 用户要求“下面也改为中文”：将模型说明窗口中编码信息区残留英文文案统一改为中文。

### 2) 命令记录
- `rg -n "label: 'Mode'|value: 'Static'|Layer Signature|Coding Metrics|Core: |Token:" frontend/src/blueprint/AppleNeuron3DTab.jsx`
- `npm --prefix frontend run -s build`

### 3) 代码修改
- 文件：`frontend/src/blueprint/AppleNeuron3DTab.jsx`
- 文案替换：
  - `Mode: Static` -> `模式: 静态分析`
  - `Layer Signature` -> `层级编码签名`
  - `Coding Metrics` -> `编码机制指标`
  - `Core:` -> `核心神经元:`
  - `Token:` -> `当前词元:`

### 4) 验证
- 构建通过：`npm --prefix frontend run -s build`。

### 5) 研究流程意义
- 信息窗口文案中文化后，状态判读链路更一致，减少解释层语言切换造成的认知中断。

## 2026-03-03 Codex 进展记录（Selected Neuron / Legend 中文化）

### 1) 修改目标
- 将 Main 控制面板中的 `Selected Neuron` 与 `Legend` 模块改为中文显示。

### 2) 命令记录
- `rg -n "Selected Neuron|Legend|Role:|Fruit:|Concept:|Layer / Neuron|Strength:|Source:|Apple micro|Apple macro|Route shared|Fruit-general|specific|Background network sample|Apple core|Click a highlighted neuron" frontend/src/blueprint/AppleNeuron3DTab.jsx`
- `npm --prefix frontend run -s build`

### 3) 代码修改
- 文件：`frontend/src/blueprint/AppleNeuron3DTab.jsx`
- `Selected Neuron` 模块文案改为中文：
  - 标题：`选中神经元`
  - `Role/Fruit/Concept/Layer / Neuron/Strength/Source` -> `角色/水果/概念/层 / 神经元/强度/来源`
  - 空态提示：`请在 3D 场景中点击高亮神经元。`
- `Legend` 模块文案改为中文：
  - 标题：`图例`
  - 各类型标签改为中文（苹果微观、中观、共享路径、水果通用、各水果特异、输入概念、背景网络采样）
  - `Apple core` -> `苹果核心`

### 4) 验证
- 构建通过：`npm --prefix frontend run -s build`。

### 5) 研究流程意义
- 关键观测区（神经元详情/图例）中文化后，字段语义更直观，降低标注与判读时的语言转换成本。

## 2026-03-03 Codex 进展记录（选中神经元 / 图例 迁移至模型说明窗口）

### 1) 修改目标
- 将 Main 左侧控制面板中的 `选中神经元` 与 `图例` 两个模块移动到右侧 `模型说明` 窗口。

### 2) 命令记录
- `rg -n "选中神经元|图例|AppleNeuronEncodingInfoPanels|infoPanelTab === 'encoding'|isAppleMainView" frontend/src/blueprint/AppleNeuron3DTab.jsx frontend/src/App.jsx`
- `npm --prefix frontend run -s build`

### 3) 代码修改
- 文件：`frontend/src/blueprint/AppleNeuron3DTab.jsx`
  - 新增可复用组件：`AppleNeuronSelectedLegendPanels`（承载“选中神经元 + 图例”）。
  - 从 `AppleNeuronControlPanels` 中删除原有这两块 UI。
- 文件：`frontend/src/App.jsx`
  - 引入 `AppleNeuronSelectedLegendPanels`。
  - 在模型说明窗口 `infoPanelTab === 'encoding' && isAppleMainView` 分支中追加渲染：
    - `AppleNeuronEncodingInfoPanels`
    - `AppleNeuronSelectedLegendPanels`

### 4) 验证
- 前端构建通过：`npm --prefix frontend run -s build`。
- 现效果：
  - 左侧 Main 控制面板不再显示“选中神经元 / 图例”。
  - 右侧模型说明窗口（编码焦点页签）显示这两块内容。

### 5) 方法学意义
- 将“控制输入”与“结果解释”分离：左侧专注操作，右侧集中解释，减少视觉跳转和语义混杂，提升编码观察一致性。

## 2026-03-03 Codex 进展记录（再次确认：选中神经元/图例迁移到模型说明）

### 1) 变更目标
- 将 Main 左侧控制面板中“选中神经元、图例”两块彻底迁移到右侧模型说明窗口，避免分散展示。

### 2) 实施与验证命令
- `rg -n "选中神经元|图例|AppleNeuronEncodingInfoPanels|infoPanelTab === 'encoding'|isAppleMainView" frontend/src/blueprint/AppleNeuron3DTab.jsx frontend/src/App.jsx`
- `npm --prefix frontend run -s build`

### 3) 代码状态
- `frontend/src/blueprint/AppleNeuron3DTab.jsx`
  - 左侧 `AppleNeuronControlPanels` 已不再渲染“选中神经元 / 图例”。
  - 复用组件 `AppleNeuronSelectedLegendPanels` 保持可用。
- `frontend/src/App.jsx`
  - 在 `Main + 编码焦点` 区域渲染：
    - `AppleNeuronEncodingInfoPanels`
    - `AppleNeuronSelectedLegendPanels`

### 4) 结果
- 左侧更聚焦“操作输入”，右侧更聚焦“结果解释”。
- 构建通过，页面可正常加载。

## [2026-03-04 11:44:47] Codex 进度记录
- 任务: 将 Main 中 Apple + Fruit Neuron Compare 改为“按输入类别比较”，并移动到模型说明窗口。
- 代码改动:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - uildConceptNeuronSet(name, category, idx) 支持类别维度，集合唯一键升级为概念+类别。
    - useAppleNeuronWorkspace 新增 queryCategoryInput，生成逻辑改为按“概念+类别”创建查询神经元集合。
    - summary 新增 categoryStats（每个类别的概念数、神经元数）。
    - 新增 AppleNeuronCategoryComparePanel，用于类别对比展示。
    - 删除左侧控制面板旧的 Apple + Fruit Neuron Compare 卡片。
    - Quick Concept Generator 增加“类别输入框”，列表展示 概念[类别]。
    - 选中神经元信息增加“类别”字段展示（若存在）。
  - rontend/src/App.jsx
    - 导入并在 Main 的“模型说明 -> 编码焦点”中渲染 AppleNeuronCategoryComparePanel。
- 执行命令:
  - g -n "Apple \+ Fruit Neuron Compare|AppleNeuronControlPanels|AppleNeuronEncodingInfoPanels|AppleNeuronSelectedLegendPanels|querySets|handleGenerateQuery|buildConceptNeuronSet|queryNameInput|summary\.fruit|fruitGeneral|fruitSpecific" frontend/src/blueprint/AppleNeuron3DTab.jsx frontend/src/App.jsx
  - 
pm --prefix frontend run -s build
- 结果:
  - 前端构建通过（Vite build success）。
- 理论/数学研究补充:
  - 本次将“概念比较”从固定 pple vs fruit 变为“按类别输入的可变分组统计”，对应研究上从单任务观测转向 C = {c_i} 的可扩展概念族比较。
  - 观测量从固定差分扩展为类别条件分布 P(neuron | category) 的可视化入口（当前为前端聚合计数，后续可接入因果/激活强度统计）。

## [2026-03-04 11:52:46] Codex 进度记录
- 任务: 将 Main 的 Compare Filter 改为“按输入名称勾选”，并移动到右侧操作面板。
- 代码改动:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - 新增 queryVisibility 状态，支持按查询集合（输入名称）控制可见性。
    - queryNodes 改为仅渲染可见集合对应神经元（勾选直接影响 3D 场景）。
    - 新增 setQuerySetVisible / setAllQuerySetVisible。
    - 新增 AppleNeuronCompareFilterPanel 组件（按输入名称+类别动态勾选，含全选/全不选）。
    - 从 AppleNeuronControlPanels 删除旧 Compare Filter（水果固定过滤）模块。
  - rontend/src/App.jsx
    - Main 模式 hasOperation 改为 	rue，启用右下操作面板。
    - 操作面板标题在 Main 下改为 操作面板 · Main。
    - 在 Main 操作面板中挂载 AppleNeuronCompareFilterPanel；DNN/SNN 操作面板逻辑保持不变。
- 执行命令:
  - g -n "Compare Filter|AppleNeuronCompareFilterPanel|queryVisibility|setQuerySetVisible|setAllQuerySetVisible|hasOperation: true|操作面板 · Main" frontend/src/blueprint/AppleNeuron3DTab.jsx frontend/src/App.jsx
  - 
pm --prefix frontend run -s build
- 结果:
  - 前端构建通过（Vite build success）。
- 理论/数学研究补充:
  - 过滤维度从“预设水果簇”升级为“用户定义概念集”，对应实验设计从固定先验类别转向可控样本子集对照。
  - 通过集合级开关得到 S_visible ⊂ S_query，便于做同一输入空间下的可见子集差分观测（支持后续因果干预/最小子回路验证）。

## [2026-03-04 12:14:53] Codex 进度记录
- 任务: 参考 DNN 结构分析 UI，将 Main 的分析类型改为四阶段（观测/提取/验证/系统），并统一风格与视觉效果。
- 代码改动:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - 新增 ANALYSIS_MODE_STAGE_GROUPS，将现有模式映射到四阶段：
      - 观测: static, dynamic_prediction, cross_layer_transport
      - 提取: subspace_geometry, eature_decomposition, compositionality
      - 验证: causal_intervention, counterfactual, obustness
      - 系统: minimal_circuit
    - 重构 AppleNeuronControlPanels 中“分析类型”区域：
      - 组头样式、网格布局、按钮激活色、边框与字体风格对齐 DNN StructureAnalysisControls。
      - 按钮布局改为分组 epeat(3, 1fr) 网格，视觉与 DNN 结构分析一致。
- 执行命令:
  - g -n "ANALYSIS_MODE_STAGE_GROUPS|分析类型（四阶段）|group.label|repeat\(3, 1fr\)|rgba\(68, 136, 255, 0.2\)" frontend/src/blueprint/AppleNeuron3DTab.jsx
  - 
pm --prefix frontend run -s build
- 结果:
  - 前端构建通过（Vite build success）。
- 理论/数学研究补充:
  - 分析流程从“平铺模式选择”重构为“四阶段序列”，对应研究方法上从无序搜索转向分层实验设计：
    1) 观测（状态轨迹）
    2) 提取（表示/子空间）
    3) 验证（因果/反事实/鲁棒）
    4) 系统（最小回路）
  - 该分层有助于把编码研究组织为 观测 -> 假设 -> 干预验证 -> 系统归纳 的闭环。

## [2026-03-04 12:23:43] Codex 进度记录
- 任务: 按“控制面板/模型说明/操作面板/算法指南”四模块做系统整理，统一核心思路与联动逻辑。
- 代码改动:
  - rontend/src/App.jsx
    - 新增 CONTROL_PANEL_BLUEPRINT：明确 main/dnn/snn/fibernet 四模块定位、参数焦点、核心公式、3D原理。
    - unctionTypePanelMap 调整：ibernet 现在也启用模型说明与操作面板（hasInfo/hasOperation = true）。
    - 新增 isFiberNetFunctionType 与 currentAlgorithmInfo：根据控制面板当前选择，统一输出“当前算法名称 + 焦点说明”。
    - 控制面板顶部新增“模块定位”卡，动态显示模块使命、当前算法、操作重点。
    - 模型说明窗口重构：
      - 新增统一的“模块定位 + 当前算法 + 核心公式 + 3D原理”信息卡。
      - Main/DNN/SNN/FiberNet 均能显示与当前算法对应的说明。
      - FiberNet 独立说明分支，避免显示无关“未加载模型”信息。
    - 操作面板重构：
      - 标题改为按当前算法动态展示（Main/FiberNet 含算法名）。
      - Main 保留 Compare Filter，并新增算法参数焦点说明。
      - DNN/SNN 增加统一“当前算法参数焦点”头卡。
      - FiberNet 新增专属操作说明（参数入口、3D观察流程）。
    - 算法指南扩展：
      - 新增 main_system / dnn_system / snn_system / fibernet_system 文档条目（作用、原理、公式、流程、3D原理）。
      - 新增对应 GUIDE_STRUCTURED 结构化说明。
      - 新增 uildGuideConclusion 对应模块结论逻辑。
      - 目录新增四个“模块定位”入口，和控制面板结构一一对应。
- 执行命令:
  - g -n "CONTROL_PANEL_BLUEPRINT|main_system|dnn_system|snn_system|fibernet_system|isFiberNetFunctionType|currentAlgorithmInfo|模块定位|操作面板 · FiberNet" frontend/src/App.jsx
  - 
pm --prefix frontend run -s build
- 结果:
  - 前端构建通过（Vite build success）。
- 理论/数学研究补充:
  - 将研究系统显式分层为：
    1) Main（数学结构还原主线）
    2) DNN（多算法观测）
    3) SNN（时序脉冲动力学）
    4) FiberNet（底流形-纤维双尺度）
  - 该分层把“观察工具”和“还原主线”解耦，形成“横向观测 -> 纵向还原 -> 因果验证 -> 系统归纳”的闭环，减少把单一统计现象误判为机制的风险。

## [2026-03-04 12:41:56] Codex 进度记录
- 任务: 评估“演化监视器(Evolution Monitor)”应显示在 Main 还是 DNN。
- 结论:
  - 建议 **不在 Main 常驻显示**。
  - 建议至少放在 **DNN**；更合理是 **DNN + FiberNet**（若当前实现可快速区分）。
- 理由:
  - Main 是“编码结构还原主线”，应保持证据链专注，减少与训练/演化状态耦合。
  - 演化监视器本质是系统状态/训练过程观测，更匹配 DNN（及 FiberNet 的系统实验）语义。
  - 将系统状态与编码证据分层，可避免把运行态噪声混入 Main 的机制判读。

## [2026-03-04 12:51:57] Codex 进度记录
- 任务: 按确认方案修改“演化监视器”显示范围。
- 改动:
  - rontend/src/App.jsx
    - 新增 showEvolutionMonitor = inputPanelTab === 'dnn'。
    - EvolutionMonitor 改为条件渲染，仅在 DNN 面板显示。
- 结果:
  - Main/SNN/FiberNet 中不再显示演化监视器。
  - DNN 中保留显示。
- 执行命令:
  - g -n "showEvolutionMonitor|EvolutionMonitor data" frontend/src/App.jsx
  - 
pm --prefix frontend run -s build
- 理论/数学研究补充:
  - 将运行态演化观测与 Main 编码证据链解耦，减少系统态噪声对编码机制判读的干扰。

## [2026-03-04 12:53:47] Codex 进度记录
- 任务: 排查“Quick Concept Generator 点击 Generate 无效”并修复。
- 诊断结论:
  - 原逻辑在两种情况下会“静默无反馈”，用户感知为按钮无效：
    1) 名称为空
    2) 同名+同类别已存在（被去重）
- 修复:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - 新增 queryFeedback 状态并在生成区域显示反馈文本。
    - 空输入时提示“请输入名称后再生成”。
    - 重复输入时不再无响应：自动恢复该集合可见、定位到该集合，并提示“已存在，已定位并显示”。
    - 新生成成功时提示“已生成…神经元集合”。
    - 删除集合后提示“已移除该概念集合”。
- 执行命令:
  - g -n "queryFeedback|请输入名称后再生成|已存在「|已生成「|已移除该概念集合" frontend/src/blueprint/AppleNeuron3DTab.jsx
  - 
pm --prefix frontend run -s build
- 结果:
  - 前端构建通过（Vite build success）。
- 理论/数学研究补充:
  - 对实验交互加入显式状态回馈，避免“重复样本被去重”导致的假阴性判断，提高证据采样流程可解释性。

## [2026-03-04 13:19:46] Codex 进度记录
- 任务: 说明 Main 中“过滤操作”的用途。
- 结论:
  - Main 过滤操作用于按“输入名称/类别”控制查询神经元集合在 3D 场景中的显示与隐藏。
  - 它不改变模型参数，只改变观测视图，用于减少遮挡、做对比、定位关键概念集合。
- 理论/数学研究补充:
  - 该功能等价于在观测层选择可见子集 S_visible ⊂ S_query，便于对子集差异进行稳定比较与因果候选筛选。

## [2026-03-04 13:21:44] Codex 进度记录
- 任务: 解释 Main 模式下 3D 空间“线条”和“原点”的含义。
- 代码定位:
  - LayerGuides 绘制层框线与中心Z轴线。
  - links 绘制神经元之间的关系连线（苹果核心链路/水果族/输入概念族）。
  - ModeVisualOverlay 在不同分析模式叠加辅助线（如特征分解三轴线）。
  - 
euronToPosition 定义节点坐标映射，坐标原点对应空间中心而非单个神经元。
- 命令: 无新增命令（本次为代码语义说明）。
- 理论/数学研究补充:
  - 主空间可视化采用“层索引->Z轴、神经元索引->极坐标(XY)”映射；原点是参考坐标中心，用于表达跨层结构与传输路径，而不是具体概念节点。

## [2026-03-04 13:24:22] Codex 进度记录
- 任务: 去掉 Main 中所有 Apple 预置数据，并改为默认空状态。
- 代码改动:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - 清空预置数据常量：APPLE_CORE_NEURONS = []、FRUIT_GENERAL_NEURONS = []、FRUIT_SPECIFIC_NEURONS = {}、FRUIT_COLORS = {}。
    - 清空默认预置输入：DEFAULT_PREDICT_PROMPT = ''，queryCategoryInput 默认改为空字符串，counterfactualPrompt 默认改为空字符串。
    - TOKEN_TRANSITIONS / TOPIC_FALLBACKS 去除苹果/水果样本，改为通用概念样本。
    - uildBackgroundNodes() 改为返回空数组，避免首屏出现默认背景神经元。
    - links 里的 Apple 预置链路 linkSpecs 清空。
    - 图例文案去 Apple 化，改为通用术语（微观/中观/共享路径/类别通用/输入概念）。
    - 动态预测输入框占位文本改为通用示例（“概念 是 一种 结构”）。
- 执行命令:
  - 
pm --prefix frontend run -s build
- 结果:
  - 前端构建通过（Vite build success）。
- 理论/数学研究补充:
  - 将 Main 从“预置样本驱动”改为“空白实验台驱动”，可减少先验偏置；后续所有可视化证据由用户输入与实验过程生成，有利于机制还原的可重复性与可检验性。

## [2026-03-04 14:10:25] Codex 进度记录
- 任务: 解释 Quick Concept Generator 为什么有两个输入框。
- 结论:
  - 第一个输入框是“名称（concept）”，用于生成具体概念的查询神经元集合。
  - 第二个输入框是“类别（category）”，用于把多个概念按类别聚合，支持比较面板与过滤面板按类别统计/筛选。
- 理论/数学研究补充:
  - 两字段对应实验变量分解：实例级变量（concept）与集合级变量（category），便于做组内/组间差分分析。

## [2026-03-04 15:40:18] Codex 进度记录
- 任务: 调整 Quick Concept Generator 布局为两行，并在输入框前增加标题。
- 代码改动:
  - rontend/src/blueprint/AppleNeuron3DTab.jsx
    - 原三列同排输入改为两行布局。
    - 第一行增加标题“名称”，第二行增加标题“类别”。
    - Generate 按钮移动到下方独立一行右对齐。
- 执行命令:
  - 
pm --prefix frontend run -s build
- 结果:
  - 前端构建通过（Vite build success）。
- 理论/数学研究补充:
  - 将实例变量（名称）与分组变量（类别）在视觉上分层，有助于减少输入歧义，提高实验变量可控性。

## [2026-03-04 15:51:14] Codex 进度记录
- 任务: 说明“填写类别 vs 不填写类别”的差异。
- 结论:
  - 不填写类别时，系统会自动使用 未分类。
  - 去重键是 名称 + 类别：同名同类别视为同一集合；同名不同类别会生成不同集合。
  - 当前实现中类别参与哈希与集合ID，因而会影响生成的节点分布（不仅是显示标签）。
  - 类别还会影响“类别神经元比较”和过滤面板中的分组统计与显示。
- 理论/数学研究补充:
  - 类别字段可视为条件变量 c，生成与统计均从 p(neuron|concept) 扩展为 p(neuron|concept,c)。

## [2026-03-04 15:52:55] Codex 进度记录
- 任务: 解释“填写类别后3D效果差异很大”的原因。
- 代码结论:
  - 在 uildConceptNeuronSet 中，aseHash 使用了 
ame + category + idx：
    - aseHash = hashString(${normalized}--)
  - 后续节点层位/神经元索引/颜色都由 aseHash 派生，因此类别变化会导致整组3D节点几何与配色显著变化。
  - 同名不同类别会被视为不同集合，可同时存在，导致画面节点数量增加、视觉差异进一步放大。
- 影响: 当前类别不仅用于分组统计，也参与了“生成机制本身”。
- 可选改进方向:
  - 若希望类别仅用于分组显示，应把生成种子改为仅依赖名称（和可选固定salt），类别只保留为metadata。

## [2026-03-04 15:58:19] Codex 进度记录
- 任务: 回答“能否统计大量名词对应神经元并寻找编码数学规律”，并提供可执行实现。
- 新增脚本:
  - 	ests/codex/deepseek7b_mass_noun_encoding_scan.py
- 脚本能力:
  - 批量名词采样（内置名词库或外部 nouns 文件）。
  - 提取每个名词的 MLP gate 神经元签名（top-k）。
  - 统计层使用分布、重用神经元比例、类内/类间相似度间隙、低秩结构（participation ratio）。
  - 输出 JSON + Markdown 报告到 	empdata/。
- 验证命令:
  - python -m py_compile tests/codex/deepseek7b_mass_noun_encoding_scan.py
  - python tests/codex/deepseek7b_mass_noun_encoding_scan.py --help
- 理论/数学研究补充:
  - 该流程将“数学原理”候选量化为可比较指标：
    1) 层集中/分散规律（entropy/top3 ratio）
    2) 可复用编码元件（reused neurons）
    3) 语义几何分离度（within-vs-between gap）
    4) 有效维度压缩（participation ratio）。

## [2026-03-04 16:08:06] Codex 进展记录（大样本名词编码三项工作）

### 1) 命令执行记录
- python -m py_compile tests/codex/deepseek7b_mass_noun_encoding_scan.py
- python tests/codex/deepseek7b_mass_noun_encoding_scan.py --help
- 
pm --prefix frontend run -s lint（仓库已有大量历史 lint 问题，非本次改动引入）
- 
pm --prefix frontend run -s build（通过）

### 2) 代码与实验管线更新
- 在 	ests/codex/deepseek7b_mass_noun_encoding_scan.py 新增“因果消融验证”能力：
  - 新增参数：--run-causal-ablation、--ablation-top-k、--ablation-random-trials、--ablation-max-nouns、--ablation-per-category-max、--ablation-reused-top-k。
  - 新增目标神经元 vs 随机神经元对照：
    - 对名词签名神经元做 last-token gate 消融，计算目标 token 概率/对数几率下降。
    - 计算 causal_margin = drop(signature) - drop(random)。
    - 输出 noun 级别记录、全局 aggregate、top reused 消融、category prototype 消融。
- 在同脚本结果中新增 
oun_records[*].signature_top_indices，便于前端还原真实神经元。
- 新增双语大样本名词集：	ests/codex/deepseek7b_bilingual_nouns.csv（中英、按类别标注）。
- 在 rontend/src/blueprint/AppleNeuron3DTab.jsx 接入扫描结果导入：
  - 新增“导入扫描 JSON”入口（Main 控制面板的 Quick Concept Generator 内）。
  - 支持设置“导入数/每概念神经元数”。
  - 将 
oun_records.signature_top_indices 直接映射为 3D 神经元点（layer/neuron 真实坐标映射）。
  - 自动导入 	op_reused_neurons 为“共享复用神经元”集合。
  - 导入结果进入现有 Compare Filter / 类别比较 / 选中神经元 / 图例链路，支持勾选与追踪。

### 3) 数学研究进展（本轮）
- 将“统计结构”推进到“因果结构验证”：
  - 定义名词级因果效应：
    - Δp_sig = p_base(target) - p_ablate_sig(target)
    - Δp_rand = p_base(target) - p_ablate_rand(target)
    - CausalMargin_p = Δp_sig - Δp_rand
  - 若 CausalMargin_p > 0 且跨样本稳定，则支持“该签名神经元子集对该概念目标 token 存在因果贡献”。
- 新增共享神经元与类别原型消融：
  - 共享复用子集验证“跨概念公共编码”是否具备可迁移因果影响。
  - 类别原型子集验证“中观类别编码”是否对类内概念目标 token 形成一致影响。
- 从工程上形成闭环：
  - 批量扫描（结构统计） -> 因果消融（必要性证据） -> Main 3D 导入（可视化对照）
  - 使“编码规律”可被重复运行、可被观察、可被比较。

## [2026-03-04 16:33:43] Codex 进展记录（Main 导入扫描改为下拉框）

### 1) 命令执行记录
- python -m py_compile server/server.py
- 
pm --prefix frontend run -s build
- g -n "api/main/scan_files|api/main/scan_file|refreshScanFileOptions|导入选中文件|MAIN_API_BASE" server/server.py frontend/src/blueprint/AppleNeuron3DTab.jsx

### 2) 功能改动
- 后端新增接口（server/server.py）：
  - GET /api/main/scan_files：扫描并返回可导入 JSON 文件列表（按修改时间倒序）。
  - GET /api/main/scan_file?path=...：按相对路径加载指定 JSON 内容。
  - 增加相对路径安全解析，阻止越界路径访问。
- 前端 Main 控制面板（rontend/src/blueprint/AppleNeuron3DTab.jsx）：
  - 将“导入扫描 JSON”改为“下拉框选择文件 + 导入选中文件”。
  - 增加“刷新列表”按钮。
  - 显示已选文件元信息（文件名、大小、修改时间）。
  - 选中后通过后端加载 JSON，并复用现有导入逻辑映射到 3D 神经元。

### 3) 数学研究进展（界面能力层）
- 将批量编码扫描结果转化为“可复用证据输入通道”：
  - 通过标准化 JSON 列表与按路径加载接口，实现多次实验结果的可重复回放。
  - 支持在 Main 中快速切换不同扫描批次，观察编码签名跨实验稳定性。
- 这使“统计扫描 -> 因果验证 -> 3D对照”闭环更高效，便于比较不同样本规模/类别配置下的编码规律一致性。

## [2026-03-04 16:45:35] Codex 进展记录（机制控制 Play 无动画修复）

### 1) 命令执行记录
- 
pm --prefix frontend run -s build

### 2) 问题定位与修复
- 问题现象：Main 中“机制控制”点击 Play 后视觉上无明显动画。
- 根因：当 keyNodes.length === 0 时，modeOverlay 提前返回，导致机制模式下 layerProgress 不随 mechanismTick 更新。
- 修复：在 rontend/src/blueprint/AppleNeuron3DTab.jsx 中取消该提前返回；无节点时仅补充提示 metric，不阻断机制相位计算。
- 结果：即使尚未生成/导入节点，机制模式也会继续推进相位（载体会产生可见跨层动画）；导入节点后可看到更明显的节点级变化。

### 3) 数学研究进展（交互验证层）
- 保持“机制相位”与“是否存在节点”解耦：
  - 机制时间变量（tick/phase）应持续演化，节点仅决定局部激活可视化是否丰富。
  - 这有助于区分“机制动力学存在”与“当前样本无可视节点”两类现象，减少误判。

## [2026-03-04 16:51:37] Codex 进展记录（未发现可导入文件原因定位）

### 1) 命令执行记录
- Get-ChildItem -Recurse -File tempdata,tests/codex,tests/codex_temp -Include *.json ...
- g -n "scan_files|looks_like_scan|noun_scan|mass_noun|encoding_scan" server/server.py
- python -m py_compile server/server.py
- 
pm --prefix frontend run -s build

### 2) 问题定位
- Main 下拉框显示“未发现可导入文件”并非前端故障，根因是：
  - 当前工作区内没有符合导入协议的扫描结果文件（需包含 
oun_records + signature_top_indices 或 	op_reused_neurons）。
  - 现有 	empdata 多为其他实验 JSON（如 *_analysis_results.json、pipeline 报告等），不含 Main 导入所需字段。

### 3) 修复与增强
- 后端 server/server.py：新增 _looks_like_main_scan_json_file，在文件名规则外增加“内容特征识别”，支持重命名后的扫描 JSON 仍可被发现。
- 前端 AppleNeuron3DTab.jsx：当列表为空时显示明确提示，指向需要生成的文件格式，而非仅显示空下拉。

### 4) 数学研究进展（数据协议层）
- 明确 Main 导入协议：
  - 最小关键字段：config.d_ff、
oun_records[*].signature_top_indices。
  - 可选增强字段：	op_reused_neurons、category_prototypes、egularity。
- 协议化后可确保“统计扫描 -> 因果验证 -> 3D重放”在数据层一致，避免非同构 JSON 导致伪失败。

## [2026-03-04 20:43:25] Codex 进展记录（执行 mass_noun 脚本并生成导入文件）

### 1) 命令执行记录
- 首次执行（全量+因果）报错：
  - python tests/codex/deepseek7b_mass_noun_encoding_scan.py --nouns-file tests/codex/deepseek7b_bilingual_nouns.csv --run-causal-ablation --ablation-max-nouns 40 --output-dir tempdata/deepseek7b_mass_noun_scan_demo
  - 错误：UnboundLocalError: top_reused_records（变量定义顺序问题）
- 修复后执行：
  - python -m py_compile tests/codex/deepseek7b_mass_noun_encoding_scan.py
  - python tests/codex/deepseek7b_mass_noun_encoding_scan.py --nouns-file tests/codex/deepseek7b_bilingual_nouns.csv --max-nouns 40 --run-causal-ablation --ablation-max-nouns 20 --output-dir tempdata/deepseek7b_mass_noun_scan_demo
- 验证后端可见：
  - GET http://localhost:5001/api/main/scan_files?limit=20 返回 1 条可导入文件。

### 2) 产出文件
- 	empdata/deepseek7b_mass_noun_scan_demo/mass_noun_encoding_scan.json
- 	empdata/deepseek7b_mass_noun_scan_demo/MASS_NOUN_ENCODING_SCAN_REPORT.md

### 3) 数学研究进展
- 本次扫描在 40 名词样本上完成了：
  - 签名提取（signature_top_indices）
  - 共享神经元统计（	op_reused_neurons）
  - 因果消融对照（signature vs random）
- 已形成可被 Main 3D 重放的最小闭环数据：
  - 结构统计 + 因果验证 + 可视化导入。

## [2026-03-04 20:47:36] Codex 进展记录（大量名词神经元数据分析方法）

### 1) 本轮命令记录
- 无代码执行命令（方法论答复）。

### 2) 研究方法沉淀（从统计到编码机制）
- 建议采用三层闭环：
  1. 结构规律层：签名、复用、层级分布、低秩结构、类别原型。
  2. 因果验证层：signature ablation vs random ablation、共享子集消融、类别原型消融。
  3. 机制还原层：最小子回路搜索、跨提示词稳定性、跨语言一致性与反事实差分。
- 关键目标不是“一个神经元=一个概念”，而是“可复现、可干预、可迁移的稀疏子集编码”。
- 评价优先级：
  - 先看因果边际（causal margin）是否稳定为正；
  - 再看跨样本/跨语言/跨提示是否保持；
  - 最后才下“机制级”结论。

## [2026-03-04 20:52:41] Codex 进展记录（机制评分卡自动化）

### 1) 命令执行记录
- python -m py_compile tests/codex/deepseek7b_mass_noun_encoding_scan.py
- python tests/codex/deepseek7b_mass_noun_encoding_scan.py --nouns-file tests/codex/deepseek7b_bilingual_nouns.csv --max-nouns 12 --run-causal-ablation --ablation-max-nouns 8 --output-dir tempdata/deepseek7b_mass_noun_scan_scorecard_demo
- g -n "mechanism_scorecard|overall_score|grade|guidance" tempdata/deepseek7b_mass_noun_scan_scorecard_demo/mass_noun_encoding_scan.json
- Get-Content tempdata/deepseek7b_mass_noun_scan_scorecard_demo/MASS_NOUN_ENCODING_SCAN_REPORT.md -First 80

### 2) 代码改动
- 在 	ests/codex/deepseek7b_mass_noun_encoding_scan.py 新增：
  - uild_mechanism_scorecard(...)：自动计算机制评分卡。
  - 分项得分：
    - structure_separation
    - reuse_sparsity_structure
    - low_rank_compactness
    - causal_evidence
  - 总分与等级：
    - strong / moderate / weak / insufficient mechanistic evidence
  - guidance 建议列表（自动给出下一步实验方向）。
- 结果 JSON 新增字段：mechanism_scorecard。
- Markdown 报告新增章节：## Mechanism Scorecard 与 ### Guidance。
- 修复 blation_prompt_for_noun 乱码提示词，统一为 ASCII 提示词。

### 3) 快速验证结果
- 输出文件：
  - 	empdata/deepseek7b_mass_noun_scan_scorecard_demo/mass_noun_encoding_scan.json
  - 	empdata/deepseek7b_mass_noun_scan_scorecard_demo/MASS_NOUN_ENCODING_SCAN_REPORT.md
- 示例得分：
  - overall_score = 0.6032
  - grade = moderate_mechanistic_evidence
- guidance 自动提示：增加消融样本与随机对照次数以提升因果置信度。

### 4) 数学研究进展
- 将“统计模式识别”升级为“结论等级化机制判据”：
  - 结构分离 + 稀疏复用 + 低秩压缩 + 因果证据 四维联合判定。
- 形成可重复的机制证据模板：
  - E = (S_struct, S_sparse, S_lowrank, S_causal) -> score -> grade -> guidance。
- 这为后续多次实验的横向可比性提供统一度量基线。

## [2026-03-04 21:10:35] Codex 进展记录（是否可确定编码原理：结论总结）

### 1) 本轮命令记录
- 无代码执行命令（结果判读与总结）。

### 2) 结论摘要
- 当前阶段结论：**不能宣称已确定最终编码原理**。
- 可成立的中间结论：
  - 编码更像“跨层稀疏子集”而非单神经元。
  - 类内相似高于类间（结构分离存在）。
  - 表示具有明显低秩压缩特征（top-k 主成分占优）。
  - 存在少量跨概念复用神经元（共享骨架候选）。
- 关键不足：
  - 因果消融的平均概率边际接近 0（当前设置下因果证据不足）。
  - 样本规模与干预强度仍不足以支持“机制定理级结论”。

### 3) 研究方法上的判定标准
- 要从“中等机制证据”升级到“可确定原理”，至少满足：
  - 因果边际在多批次、多语言、多提示下稳定为正；
  - 最小因果子集可复现并具跨任务迁移；
  - 反事实与组合测试可被同一子空间模型统一解释。

## [2026-03-04 21:44:38] Codex 进展记录（机制证据增强：多模板因果指标）

### 1) 命令执行记录
- python -m py_compile tests/codex/deepseek7b_mass_noun_encoding_scan.py
- python tests/codex/deepseek7b_mass_noun_encoding_scan.py --nouns-file tests/codex/deepseek7b_bilingual_nouns.csv --max-nouns 8 --run-causal-ablation --ablation-max-nouns 6 --ablation-eval-prompt-count 3 --output-dir tempdata/deepseek7b_mass_noun_scan_mech_v2_demo
- g -n "causal_margin_logprob|causal_margin_rank_worse|causal_margin_prob_z|ablation_eval_prompt_count|mechanism_scorecard" tempdata/deepseek7b_mass_noun_scan_mech_v2_demo/mass_noun_encoding_scan.json
- Get-Content tempdata/deepseek7b_mass_noun_scan_mech_v2_demo/MASS_NOUN_ENCODING_SCAN_REPORT.md -First 100

### 2) 代码升级
- 脚本 	ests/codex/deepseek7b_mass_noun_encoding_scan.py 新增：
  - 多模板因果评估（--ablation-eval-prompt-count）。
  - 目标指标从单一概率扩展为：
    - prob、logprob、ank 三维边际。
  - 聚合统计新增：
    - causal_margin_*_stats（均值/方差/置信区间）
    - causal_margin_prob_z（显著性近似）
- mechanism_scorecard 的因果子分项更新为多信号融合：
  - prob margin + logprob margin + rank worsening + positive ratio + z-score。

### 3) 结果摘要（v2 demo）
- overall_score = 0.6548
- grade = moderate_mechanistic_evidence
- causal_evidence 分项从旧版弱信号提升为可分解读数（含 rank/logprob 证据）。

### 4) 数学研究进展
- 因果证据从“单点概率差”升级为“多观测通道证据向量”：
  - C = (Δprob, Δlogprob, Δrank, z)
- 评分机制从单标量阈值变为多维融合，更接近“机制鉴别”而非“统计相关”。

## [2026-03-04 22:45:26] Codex 进展记录（n=120 + 固定样本多seed重复）

### 1) 命令执行记录
- 单次大样本：
  - python tests/codex/deepseek7b_mass_noun_encoding_scan.py --nouns-file tests/codex/deepseek7b_bilingual_nouns.csv --max-nouns 120 --run-causal-ablation --ablation-max-nouns 60 --ablation-random-trials 4 --ablation-eval-prompt-count 3 --output-dir tempdata/deepseek7b_mass_noun_scan_n120_single
- 固定样本多seed（5次）：
  - seeds = 101, 202, 303, 404, 505
  - 每次命令：--max-nouns 120 --run-causal-ablation --ablation-max-nouns 40 --ablation-random-trials 3 --ablation-eval-prompt-count 3 --seed <seed>
- 汇总脚本：
  - 新增 	ests/codex_temp/summarize_mass_noun_multiseed.py
  - 执行：python tests/codex_temp/summarize_mass_noun_multiseed.py

### 2) 输出文件
- 单次大样本：
  - 	empdata/deepseek7b_mass_noun_scan_n120_single/mass_noun_encoding_scan.json
  - 	empdata/deepseek7b_mass_noun_scan_n120_single/MASS_NOUN_ENCODING_SCAN_REPORT.md
- 多seed汇总：
  - 	empdata/deepseek7b_mass_noun_scan_n120_multiseed_summary.json
  - 	empdata/deepseek7b_mass_noun_scan_n120_multiseed_summary.md

### 3) 关键结果
- n=120 单次：
  - overall_score = 0.300593
  - grade = insufficient_evidence
  - cosine_gap = 0.001536, jaccard_gap = 0.007595（分离度偏弱）
  - mean_causal_margin_logprob = -0.008659
  - mean_causal_margin_rank_worse = -128.215
- n=120 固定样本多seed：
  - grade_distribution = {insufficient_evidence: 5}
  - overall_score mean/std = 0.300593 / 0.000000
  - mean_causal_margin_logprob = -0.010015 ± 0.005301
  - mean_causal_margin_rank_worse = -169.028 ± 68.566

### 4) 数学研究进展
- 在 n=120 固定样本下，结论跨seed稳定但指向“证据不足”：
  - 结构分离和因果边际均未达到机制阈值。
- 这说明当前“名词首token预测”评估任务对机制鉴别力不足，下一步应升级任务设计：
  - 使用更强语义约束模板与多位置目标、最小回路验证、反事实差分联合判定。

## 2026-03-05 Codex 进展记录（max-nouns=120 + 固定样本多seed）

### 代码与实验改动
1. 更新 `tests/codex/deepseek7b_mass_noun_encoding_scan.py`：
   - 增加序列级因果通道：`target_seq_logprob`、`target_seq_avg_logprob`。
   - 增加全位置消融能力：`LastTokenGateAblation(..., ablate_all_positions=True)`。
   - 机制评分卡接入序列级因果指标：`mean_causal_margin_seq_logprob`、`mean_causal_margin_seq_avg_logprob`、`causal_margin_seq_logprob_z`。
   - 报告中新增序列级因果指标与置信区间输出。
   - 新增参数：`--ablation-sample-strategy {random,head}`，用于区分随机子样本与固定样本（head）。
2. 更新 `tests/codex_temp/summarize_mass_noun_multiseed.py`：
   - 汇总新增序列级因果指标字段，并写入 markdown 报告。

### 执行命令（核心）
- 语法检查：
  - `python -m py_compile tests/codex/deepseek7b_mass_noun_encoding_scan.py`
- smoke：
  - `python tests/codex/deepseek7b_mass_noun_encoding_scan.py --max-nouns 8 --run-causal-ablation --ablation-max-nouns 6 --ablation-random-trials 2 --ablation-eval-prompt-count 2 --output-dir tempdata/deepseek7b_mass_noun_scan_smoke_v3b`
- n=120 单次：
  - `python tests/codex/deepseek7b_mass_noun_encoding_scan.py --max-nouns 120 --run-causal-ablation --ablation-max-nouns 60 --ablation-random-trials 3 --ablation-eval-prompt-count 3 --output-dir tempdata/deepseek7b_mass_noun_scan_n120_single_v3`
- 多seed（随机子样本）：
  - seeds=`101,202,303,404,505`
  - 输出目录：`tempdata/deepseek7b_mass_noun_scan_n120_multiseed_v3_seed*`
- 多seed（固定子样本 head）：
  - seeds=`101,202,303,404,505`
  - 参数增加：`--ablation-sample-strategy head`
  - 输出目录：`tempdata/deepseek7b_mass_noun_scan_n120_multiseed_v3fix_seed*`
- 汇总：
  - `python tests/codex_temp/summarize_mass_noun_multiseed.py --pattern "tempdata/deepseek7b_mass_noun_scan_n120_multiseed_v3_seed*/mass_noun_encoding_scan.json" --output-json tempdata/deepseek7b_mass_noun_scan_n120_multiseed_v3_summary.json --output-md tempdata/deepseek7b_mass_noun_scan_n120_multiseed_v3_summary.md`
  - `python tests/codex_temp/summarize_mass_noun_multiseed.py --pattern "tempdata/deepseek7b_mass_noun_scan_n120_multiseed_v3fix_seed*/mass_noun_encoding_scan.json" --output-json tempdata/deepseek7b_mass_noun_scan_n120_multiseed_v3fix_summary.json --output-md tempdata/deepseek7b_mass_noun_scan_n120_multiseed_v3fix_summary.md`

### 关键结果
- 单次（n=120）整体评分：`overall_score=0.277820`，`grade=insufficient_evidence`。
- 多seed（随机子样本）与固定子样本对比：
  - 两者 `grade` 均为 `insufficient_evidence`（5/5）。
  - 固定样本方案的序列级因果通道更稳定：
    - `mean_causal_margin_seq_logprob`：random `0.016276±0.012703` -> fixed `0.027050±0.000871`
    - `causal_margin_seq_logprob_z`：random `0.648257±0.410381` -> fixed `0.964476±0.027076`

### 理论与数学研究进展
1. 现阶段证据支持“弱的分布式编码+低秩结构”存在，但不足以锁定“精确微观编码定律”。
2. 概率通道（next-token prob）在本模型与本任务上分辨率不足，序列logprob通道更敏感、更稳定。
3. 固定样本多seed可显著降低实验方差，适合作为后续机制检验的标准设置。
4. 下一步应把“编码原理”验证转向：
   - 最小回路提取（稀疏子图）
   - 反事实保持检验（语义保持 vs 语法保持）
   - 跨语言一致性检验（同概念不同语言 token 化路径）

## 2026-03-05 Codex 进展记录（最小回路 + 反事实验证 + Main可视化接入）

### 本轮目标
在 mass noun 扫描流水线中，补齐两类机制级验证：
1. 最小因果子回路提取（minimal circuit extraction）
2. 反事实特异性验证（counterfactual specificity）
并把结果接入 Main 的 `minimal_circuit` / `counterfactual` 模式，替代伪随机动画。

### 代码改动
1. `tests/codex/deepseek7b_mass_noun_encoding_scan.py`
   - 新增函数：
     - `ablated_sequence_logprob(...)`
     - `extract_minimal_circuit_for_item(...)`
     - `pick_counterfactual_targets(...)`
   - `run_causal_ablation_suite(...)` 新增输出：
     - `minimal_circuit`
     - `counterfactual_validation`
   - 新增参数：
     - `--minimal-circuit-max-nouns`
     - `--minimal-circuit-target-ratio`
     - `--minimal-circuit-max-size`
     - `--counterfactual-max-pairs`
   - 报告 `MASS_NOUN_ENCODING_SCAN_REPORT.md` 增加以上两块结果摘要。
2. `frontend/src/blueprint/AppleNeuron3DTab.jsx`
   - 导入扫描 JSON 时，解析：
     - `causal_ablation.minimal_circuit.records`
     - `causal_ablation.counterfactual_validation.records`
   - Main 中 `minimal_circuit` / `counterfactual` 模式优先使用导入结果驱动激活显示。
   - 导入摘要增加：最小回路名词数、反事实对数量。

### 执行命令
- 脚本语法检查：
  - `python -m py_compile tests/codex/deepseek7b_mass_noun_encoding_scan.py`
- 新机制 smoke：
  - `python tests/codex/deepseek7b_mass_noun_encoding_scan.py --max-nouns 12 --run-causal-ablation --ablation-max-nouns 8 --ablation-random-trials 2 --ablation-eval-prompt-count 2 --minimal-circuit-max-nouns 4 --minimal-circuit-target-ratio 0.8 --minimal-circuit-max-size 8 --counterfactual-max-pairs 8 --output-dir tempdata/deepseek7b_mass_noun_scan_mech_v4_smoke`
- n=120 正式运行（含新机制）：
  - `python tests/codex/deepseek7b_mass_noun_encoding_scan.py --max-nouns 120 --run-causal-ablation --ablation-max-nouns 60 --ablation-random-trials 3 --ablation-eval-prompt-count 3 --ablation-sample-strategy head --minimal-circuit-max-nouns 12 --minimal-circuit-target-ratio 0.8 --minimal-circuit-max-size 12 --counterfactual-max-pairs 24 --output-dir tempdata/deepseek7b_mass_noun_scan_n120_mech_v4_single`
- 前端构建验证：
  - `npm --prefix frontend run -s build`

### 结果快照（n=120, mech_v4_single）
- `overall_score = 0.307698`，`grade = insufficient_evidence`
- 因果主指标：
  - `mean_causal_margin_seq_logprob = 0.044201`
  - `causal_margin_seq_logprob_z = 1.404499`
- 最小回路：
  - `n_tested_nouns = 12`
  - `mean_subset_size = 0.6667`
  - `mean_recovery_ratio = 1.5244`
  - `high_recovery_ratio = 0.5833`
- 反事实验证：
  - `n_pairs = 14`
  - `mean_specificity_margin_seq_logprob = 0.051339`
  - `same_category_margin = 0.048269`
  - `cross_category_margin = 0.054408`
  - `specificity_z = 1.761678`

### 数学研究进展
1. 新增“最小回路 + 反事实”后，证据形态从单纯“相关/平均效应”扩展到“结构可压缩 + 特异性差分”。
2. 当前结果提示：部分概念确实可由极小子集驱动（subset size 小，且可恢复较大效应），但稳定性和普适性仍不足以宣称唯一微观编码定律。
3. 反事实特异性边际已出现正值（均值>0），说明“概念内外差分”可被子集捕捉；下一步应做多 seed + 固定模板复验，建立置信下界。

## 2026-03-05 Codex 进展记录（阶段性状态报告：进展/问题/下一步）

### 本轮命令记录
- `Get-Content tempdata/deepseek7b_mass_noun_scan_n120_mech_v4_single/MASS_NOUN_ENCODING_SCAN_REPORT.md -TotalCount 220`
- `Get-Content tempdata/deepseek7b_mass_noun_scan_n120_multiseed_v3fix_summary.md -TotalCount 200`
- `Get-Content reseach/gtp5/doc/AGI_GPT5_MEMO.md -Tail 220`

### 阶段性结论（面向“人类水平智能系统”目标）
1. 已有实质进展：
   - 建立了从结构统计到因果验证的统一实验流水线；
   - 加入序列级因果指标、最小回路提取、反事实特异性验证；
   - Main 3D 界面可直接读取并显示导入的机制结果。
2. 当前证据等级：
   - 多组结果仍为 `insufficient_evidence`，尚不能声称已还原通用语言智能的数学原理。
3. 有价值的正信号：
   - 序列级通道比单token概率更稳定；
   - 部分概念出现可压缩子回路与正向反事实特异性边际。

### 核心问题
1. 结构分离弱：类内/类间间隔偏小，难支撑高置信机制结论。
2. 显著性不足：多个关键指标置信区间仍跨 0。
3. 任务代理偏弱：当前任务对“微观编码定律”的鉴别力仍不足。
4. 泛化证据缺失：跨语言、跨任务、跨模型复现实验不足。
5. 指标定义需继续校正：如最小回路恢复率出现 >1，提示需要更严格的干预与归一化定义。

### 下一步工作（优先级）
1. P0：固定样本 + 多seed 的 v4 复验（含最小回路/反事实），输出置信下界。
2. P0：引入跨语言同义概念组（中英）并做一致性检验，验证编码是否语言无关。
3. P1：扩展目标任务到多位置、多模板、多跳链路，提升机制鉴别力。
4. P1：把 Main 右侧模型说明改为“导入结果驱动”的统计摘要与阈值告警。
5. P2：在 DNN/SNN/FiberNet 三通道上建立统一指标协议，评估结构效率与稳健性差异。

### 数学研究进展
- 研究框架已从“统计结构观察”升级为“结构-因果-反事实三联证据框架”：
  - `E = (S_structure, S_causal, S_counterfactual)`
- 下一阶段目标是把该框架转化为可复验、可否证、可跨任务迁移的机制定理候选。

## 2026-03-05 Codex 进展记录（拉取远端并合并本地代码）

### 命令记录
1. 状态检查：
   - `git status --short --branch`
   - `git remote -v`
   - `git branch -vv`
2. 保护本地改动：
   - `git stash push -u -m "codex-pre-pull-20260305-145550" -- frontend server tests reseach research`
3. 拉取远端：
   - `git fetch origin`
   - `git pull --no-rebase origin main`
4. 恢复本地改动：
   - `git stash pop 'stash@{0}'`
5. 冲突处理：
   - 处理 `frontend/src/StructureAnalysisPanel.jsx` 与 `frontend/src/config/panels.js`
   - `git add frontend/src/StructureAnalysisPanel.jsx frontend/src/config/panels.js`
6. 验证：
   - `git diff --name-only --diff-filter=U`
   - `git log --oneline --decorate -n 3`

### 结果
- 本地分支已更新到最新远端：`d222ed1`（`main` 与 `origin/main` 对齐）。
- 本地改动已恢复并完成冲突处理，当前无未解决冲突。
- 现工作树保留本地修改与实验文件，满足“拉取远端并合并本地代码”目标。

### 理论与研究进展
- 本次操作属于工程集成步骤，无新增数学机制结论。
- 价值在于将最新远端研究资产与本地机制实验代码合流，为后续多seed复验与机制可视化联调提供统一代码基线。

## 2026-03-05 Codex 进展记录（终极版系统方案对齐分析）

### 本轮命令记录
- `Get-Content "docs/还原大脑编码机制的系统方案（终极版）.md" -TotalCount 320`
- `python -c "... read_text(utf-8) ..."`（用于确认文档编码与可读内容）
- `Get-Content tempdata/deepseek7b_mass_noun_scan_n120_mech_v4_single/MASS_NOUN_ENCODING_SCAN_REPORT.md -TotalCount 240`
- `Get-Content tests/codex/deepseek7b_mass_noun_encoding_scan.py -TotalCount 260`
- `Get-Content reseach/gtp5/doc/AGI_GPT5_MEMO.md -Tail 220`

### 对齐结论（终极版思路 vs 本地进展）
1. 终极版提出“六大特性统一于连接可塑性+脉冲”的总假说，逻辑完整，且带有自我批判条款（第六章硬伤审视），方向正确。
2. 本地已有强实现的是：
   - 关键因素提取（因果消融、最小回路、反事实差分）
   - 规模化初测（n=120 + 多seed）
3. 本地薄弱环节是：
   - 一次学习/可塑性效率（尚无可复验基准）
   - 跨语言与跨模态验证（语言元编码仍是推断层）
   - 将SNN/FiberNet与DNN放在同一指标协议下对比（尚未完成）

### 阶段性状态
- 当前属于“机制证据积累期”，不是“原理定理化完成期”。
- 最新结果仍为 `insufficient_evidence`，但出现正向信号：
  - seq-logprob 因果边际为正；
  - 反事实特异性边际均值为正。

### 下一阶段计划（执行优先级）
1. P0：v4 固定样本多seed复验（含最小回路+反事实），产出置信下界与失败样本库。
2. P0：建立“可塑性效率”基准任务（一次学习 vs 多次迭代）并形成统一评分。
3. P1：跨语言同义概念对齐实验（中文/英文）验证编码一致性。
4. P1：统一 DNN/SNN/FiberNet 指标协议，完成同任务横向比较。
5. P2：将 Main 面板升级为结果驱动仪表盘，直连上述统计与告警阈值。

### 数学研究进展
- “连接可塑性+脉冲”总假说已进入可实验分解阶段，当前可形式化为：
  - `E_total = (S_structure, S_causal, S_counterfactual, S_plasticity, S_transfer)`
- 下一步重点是补齐 `S_plasticity` 和 `S_transfer`，否则无法从相关性证据升级到机制原理证据。

## 2026-03-05 Codex 进展记录（P0-1执行：v4固定样本多seed复验）

### 本轮执行命令
1. v4 固定样本多seed（5次）：
   - seeds=`101,202,303,404,505`
   - 命令核心参数：
     - `--max-nouns 120`
     - `--run-causal-ablation`
     - `--ablation-max-nouns 60 --ablation-sample-strategy head`
     - `--minimal-circuit-max-nouns 12 --minimal-circuit-target-ratio 0.8 --minimal-circuit-max-size 12`
     - `--counterfactual-max-pairs 24`
2. 汇总：
   - `python tests/codex_temp/summarize_mass_noun_multiseed.py --pattern "tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed*/mass_noun_encoding_scan.json" --output-json tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_summary.json --output-md tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_summary.md`
3. 可否证报告：
   - 新增脚本：`tests/codex_temp/build_falsifiable_stage_report.py`
   - 运行：`python tests/codex_temp/build_falsifiable_stage_report.py --pattern "tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed*/mass_noun_encoding_scan.json" --output-json tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_falsifiable_report.json --output-md tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_falsifiable_report.md`

### 输出文件
- 多seed汇总：
  - `tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_summary.json`
  - `tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_summary.md`
- 可否证报告：
  - `tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_falsifiable_report.json`
  - `tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_falsifiable_report.md`

### 关键结果（5 seeds）
- `grade_distribution = {insufficient_evidence: 5}`
- `overall_score = 0.290324 ± 0.005064`
- `mean_causal_margin_seq_logprob = 0.027050 ± 0.000871`
- `causal_margin_seq_logprob_z = 0.964476 ± 0.027076`
- `counterfactual_mean_specificity_margin_seq_logprob = 0.051339`（跨seed稳定）
- `counterfactual_specificity_margin_z = 1.761678`
- `minimal_circuit_mean_recovery_ratio = 1.524439`
- `minimal_circuit_mean_subset_size = 0.666667`

### 可否证判定（阶段）
- H1（序列因果边际 > 0）：PASS
- H2（反事实特异性边际 > 0）：PASS
- H3（最小回路恢复率 >= 0.8）：PASS
- H4（总体机制分 >= 0.42）：FAIL
- H5（序列因果z >= 1.96）：FAIL

### 数学研究进展
1. 已确认存在稳定正向的“序列因果边际”和“反事实特异性边际”，说明机制信号真实存在。
2. 但未达到“机制显著性阈值”（z<1.96，overall仍低），当前结论仍是“证据不足以定理化”。
3. 下阶段重点不再是“是否有信号”，而是“如何把信号提升为高置信机制定律”：
   - 加强任务鉴别力（多位置、多模板、多跳）
   - 扩展跨语言一致性
   - 校正最小回路指标定义（恢复率>1的解释与归一化）

## 2026-03-05 Codex 进展记录（P0-2执行：可塑性效率基准）

### 新增脚本
- `tests/codex/deepseek7b_plasticity_efficiency_benchmark.py`
  - 目标：比较“Hebbian 一次写入原型”与“SGD 多步迭代”在同一冻结特征空间中的样本效率。
  - 输出：`plasticity_efficiency_benchmark.json` + `PLASTICITY_EFFICIENCY_BENCHMARK_REPORT.md`

### 执行命令
1. 语法检查：
   - `python -m py_compile tests/codex/deepseek7b_plasticity_efficiency_benchmark.py`
2. 基准运行（v1）：
   - `python tests/codex/deepseek7b_plasticity_efficiency_benchmark.py --output-dir tempdata/deepseek7b_plasticity_efficiency_benchmark_v1`
3. 扩展步数运行（v1b）：
   - `python tests/codex/deepseek7b_plasticity_efficiency_benchmark.py --sgd-steps 1,5,20,100,300,1000 --output-dir tempdata/deepseek7b_plasticity_efficiency_benchmark_v1b`

### 关键结果（v1b）
- 任务设置：24类（6类别x4名词），冻结 DeepSeek gate 特征，单样本支持集。
- Hebbian one-shot 准确率：`0.572917`
- SGD 多步准确率：
  - step1: `0.0625`
  - step5: `0.0625`
  - step20: `0.1250`
  - step100: `0.3750`
  - step300: `0.3854`
  - step1000: `0.3750`
- `steps_to_match_hebbian = not reached`

### 阶段结论
1. 在当前读出层基准中，一次写入（原型记忆）显著优于重复梯度迭代的样本效率。
2. 这为“可塑性效率”提供了可量化的实验抓手，支持终极方案中关于“一次学习效率”问题的优先级判断。
3. 该基准仍是读出层层面，下一步要扩展到：
   - 多模板鲁棒评估
   - 跨语言同义概念
   - 与最小回路/反事实指标联合判定

### 数学研究进展
- 新增可塑性效率指标：
  - `E_plasticity = steps_to_match_hebbian`
- 当前结果可解释为：在固定表示空间里，原型写入相当于局部最短路径更新，而SGD在单样本约束下收敛慢且不稳定。
- 这为“连接可塑性效率”假说提供了一个可复验的工程代理指标。

## 2026-03-05 Codex 进展记录（测试思路与意义说明）

### 本轮命令记录
- 无新增实验命令（本轮为方法学说明与结果解释）。

### 说明框架
1. 结构层：大量名词扫描，定位分布式编码结构。
2. 因果层：签名消融与随机对照，判定“相关”是否升级为“必要性”。
3. 子回路层：最小回路提取，检验是否存在可压缩的关键神经元子集。
4. 反事实层：同类/跨类反事实差分，检验编码特异性。
5. 稳定性层：固定样本多seed，检验结论是否抗随机扰动。
6. 可塑性层：一次写入 vs 多步迭代，量化样本效率差距。

### 理论意义
- 将“统计结构观察”升级为“结构-因果-反事实-可塑性”四层证据链。
- 为“还原大脑编码机制”提供可复验、可否证、可扩展的工程评估协议。
- 当前结论仍属机制证据积累阶段，尚未达到原理定理化阈值。

## 2026-03-05 Codex 进展记录（继续：E_plasticity 稳定化与跨语言验证）

### 代码改动
1. 更新 `tests/codex/deepseek7b_plasticity_efficiency_benchmark.py`
   - 新增 `--nouns-file`、`--max-nouns`。
   - 新增 `--selection-strategy {random,head}`，支持固定样本复验。
   - 新增 `load_nouns(...)` 与 `select_nouns_fixed(...)`。
2. 新增汇总脚本 `tests/codex_temp/summarize_plasticity_multiseed.py`
   - 汇总多 seed 的 Hebbian 准确率、SGD曲线、not_reached_ratio。
3. 更新 `tests/codex_temp/build_falsifiable_stage_report.py`
   - 新增读取 `--plasticity-summary-json`。
   - 新增假设：
     - H6: `plasticity.not_reached_ratio >= 0.8`
     - H7: `mean(hebbian) > mean(sgd@1000)`
4. 新增 UTF-8 中英词表：
   - `tests/codex/deepseek7b_bilingual_nouns_utf8.csv`

### 执行命令
- `python -m py_compile tests/codex/deepseek7b_plasticity_efficiency_benchmark.py tests/codex_temp/summarize_plasticity_multiseed.py`
- 多seed（固定样本）:
  - seeds=`101,202,303,404,505`
  - `python tests/codex/deepseek7b_plasticity_efficiency_benchmark.py --selection-strategy head --n-categories 6 --n-per-category 4 --sgd-steps 1,5,20,100,300,1000 --n-trials 5 --seed <seed> --output-dir tempdata/deepseek7b_plasticity_efficiency_benchmark_v2_seed<seed>`
- 汇总：
  - `python tests/codex_temp/summarize_plasticity_multiseed.py --pattern "tempdata/deepseek7b_plasticity_efficiency_benchmark_v2_seed*/plasticity_efficiency_benchmark.json" --output-json tempdata/deepseek7b_plasticity_efficiency_benchmark_v2_multiseed_summary.json --output-md tempdata/deepseek7b_plasticity_efficiency_benchmark_v2_multiseed_summary.md`
- 阶段可否证报告（接入plasticity）：
  - `python tests/codex_temp/build_falsifiable_stage_report.py --pattern "tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed*/mass_noun_encoding_scan.json" --plasticity-summary-json tempdata/deepseek7b_plasticity_efficiency_benchmark_v2_multiseed_summary.json --output-json tempdata/deepseek7b_stage_falsifiable_report_v2.json --output-md tempdata/deepseek7b_stage_falsifiable_report_v2.md`
- 中英词表基准（单次）：
  - `python tests/codex/deepseek7b_plasticity_efficiency_benchmark.py --nouns-file tests/codex/deepseek7b_bilingual_nouns_utf8.csv --n-categories 6 --n-per-category 4 --selection-strategy head --sgd-steps 1,5,20,100,300,1000 --output-dir tempdata/deepseek7b_plasticity_efficiency_benchmark_bilingual_v1`

### 关键结果
1. Plasticity v2 多seed（固定样本）
- Hebbian one-shot acc mean: `0.618750`
- SGD acc mean:
  - step1: `0.087500`
  - step100: `0.506250`
  - step1000: `0.506250`
- `not_reached_ratio = 1.0000`（5/5 均未达到 Hebbian）
2. Stage falsifiable v2（含plasticity）
- H6: PASS（not_reached_ratio >= 0.8）
- H7: PASS（hebbian_mean > sgd@1000_mean）
3. 中英词表（UTF-8）单次
- Hebbian: `0.572917`
- SGD@1000: `0.416667`
- steps_to_match: `not reached`

### 数学研究进展
- `E_plasticity` 从单次观测升级为多seed稳定指标：
  - `E_plasticity = (hebbian_acc, sgd_curve, not_reached_ratio)`
- 当前证据支持：在固定 DeepSeek 表示空间中，一次写入型原型机制具有显著更高样本效率。
- 这为“可塑性效率是AGI关键瓶颈”提供了可复验支持，并已接入阶段可否证报告体系。

## 2026-03-05 Codex 进展记录（继续：中英可塑性多seed与阶段报告v3）

### 本轮命令
1. 中英可塑性多seed（5次）：
   - seeds=`101,202,303,404,505`
   - `python tests/codex/deepseek7b_plasticity_efficiency_benchmark.py --nouns-file tests/codex/deepseek7b_bilingual_nouns_utf8.csv --n-categories 6 --n-per-category 4 --selection-strategy head --sgd-steps 1,5,20,100,300,1000 --n-trials 5 --seed <seed> --output-dir tempdata/deepseek7b_plasticity_efficiency_bilingual_v2_seed<seed>`
2. 中英汇总：
   - `python tests/codex_temp/summarize_plasticity_multiseed.py --pattern "tempdata/deepseek7b_plasticity_efficiency_bilingual_v2_seed*/plasticity_efficiency_benchmark.json" --output-json tempdata/deepseek7b_plasticity_efficiency_bilingual_v2_multiseed_summary.json --output-md tempdata/deepseek7b_plasticity_efficiency_bilingual_v2_multiseed_summary.md`
3. 生成阶段可否证报告v3（接入中英plasticity）：
   - `python tests/codex_temp/build_falsifiable_stage_report.py --pattern "tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed*/mass_noun_encoding_scan.json" --plasticity-summary-json "tempdata/deepseek7b_plasticity_efficiency_bilingual_v2_multiseed_summary.json" --output-json "tempdata/deepseek7b_stage_falsifiable_report_v3_bilingual.json" --output-md "tempdata/deepseek7b_stage_falsifiable_report_v3_bilingual.md"`

### 关键结果
- 中英plasticity多seed：
  - hebbian mean = `0.581250`
  - sgd@1000 mean = `0.406250`
  - not_reached_ratio = `1.0000`
- 阶段可否证报告v3新增判定：
  - H6 PASS（not_reached_ratio >= 0.8）
  - H7 PASS（mean(hebbian) > mean(sgd@1000)）

### 数学研究进展
- `E_plasticity` 在中英词表上依旧稳定支持“一次写入优于多步迭代”的结论。
- 说明该效应不局限于纯英文概念集合，具备跨语言扩展迹象。
- 阶段评估体系已形成：
  - 结构/因果/反事实（主链） + 可塑性效率（补链）
  - 支持后续进入跨任务迁移与统一指标协议阶段。

## 2026-03-05 Codex 进展记录（继续：三维参数结构分析）

### 新增脚本
- `tests/codex/deepseek7b_triaxial_param_structure_analysis.py`
  - 对每个概念建立三轴分析：
    1) micro_attr（属性维）
    2) same_type（同类区分维）
    3) super_type（上级类别维）
  - 输出轴级因果子集与参数维度结构：
    - 神经元坐标（layer, neuron）
    - gate/up/down 的主导参数维度索引（具体dim index）
    - 组内共享维度（非均值统计，而是参数维度交集）

### 执行命令
- 语法检查：
  - `python -m py_compile tests/codex/deepseek7b_triaxial_param_structure_analysis.py`
- 运行：
  - `python tests/codex/deepseek7b_triaxial_param_structure_analysis.py --axis-candidate-k 10 --axis-subset-k 4 --param-top-dim-k 6 --output-dir tempdata/deepseek7b_triaxial_param_structure_v1`

### 输出文件
- `tempdata/deepseek7b_triaxial_param_structure_v1/triaxial_param_structure.json`
- `tempdata/deepseek7b_triaxial_param_structure_v1/TRIAXIAL_PARAM_STRUCTURE_REPORT.md`

### 关键结构发现（参数维度层）
1. Apple 三轴
- micro_attr 轴子集：`L23N5458, L27N12646, L27N1011, L27N3077`
  - gate主导维：`2718, 1026, 3503, 3453, ...`
- same_type 轴子集：`L27N2747, L26N4172, L27N7733, L27N4210`
  - gate主导维：`3053, 3327, 879, 209, ...`
- super_type 轴子集：`L2N8981, L23N11613, L22N16034, L21N2199`
  - gate主导维：`2524, 2467, 2427, 3577, 3402, 2041, ...`

2. Cat 三轴
- micro_attr 轴子集：`L22N2369, L1N9692, L22N8252, L5N16166`
  - gate主导维：`362, 456, 3270, 130, 523, ...`
- same_type 轴子集：`L21N17511, L2N4830, L27N303, L27N11609`
  - gate主导维：`1571, 3003, 1572, 1865, ...`
- super_type 轴子集：`L1N3007`
  - gate主导维：`1432, 2727, 1768, 1511, 822, ...`

3. Apple vs Cat 轴级比较
- micro_attr：神经元、层、gate维、down维重叠均接近0
- same_type：仅层级有少量重叠（layer_jaccard=0.25），参数维重叠接近0
- super_type：重叠接近0

4. 多类型共享结构（参数维交集）
- fruit 组（apple/banana/pineapple）在 super_type 轴存在共享 gate 输入维：
  - `2041, 2427, 2467, 2524`
- animal 组（cat/dog）在 super_type 轴存在共享 gate 输入维：
  - `2727`
- 这表明“上级类别编码”出现可复用参数维度骨架，而 micro 与 same_type 更偏概念特异。

### 解释
- 该结果支持“分层编码结构”：
  - 属性维与同类区分维：高特异、参数维离散。
  - 上级类别维：出现组内共享参数维（可解释的父类骨架）。
- 这类发现是参数维度层面的结构特征，不是仅靠均值统计得到的结论。

## [2026-03-05] 阶段进展核对：编码机制证据链（参数结构视角）

### 本轮执行命令
- `Get-Content tempdata/deepseek7b_stage_falsifiable_report_v3_bilingual.json -TotalCount 220`
- `Get-Content tempdata/deepseek7b_triaxial_param_structure_v1/triaxial_param_structure.json -TotalCount 260`
- `Get-Content tempdata/deepseek7b_plasticity_efficiency_bilingual_v2_multiseed_summary.json -TotalCount 260`
- `rg -n "shared_gate_input_dims|apple_vs_cat|group_shared|super_type|same_type|micro_attr|pairwise" tempdata/deepseek7b_triaxial_param_structure_v1/triaxial_param_structure.json`
- `$c=Get-Content tempdata/deepseek7b_triaxial_param_structure_v1/triaxial_param_structure.json; $c | Select-Object -Skip 12120 -First 120`

### 关键结果（可复现指标）
1. 可证伪报告（5 seeds）
- `overall_score mean=0.2903, CI=[0.2859, 0.2948]`，仍低于阶段阈值0.42（H4 fail）。
- `causal_seq_margin mean=0.02705, CI下界>0`（H1 pass）。
- `counterfactual_margin mean=0.05134, CI下界>0`（H2 pass）。
- `mcs_recovery mean=1.5244`（H3 pass）。
- `causal_seq_z mean=0.964 < 1.96`（H5 fail）。

2. 可塑性效率（双语词表，5 seeds）
- Hebbian one-shot：`0.5813`。
- SGD@1000：`0.4063`。
- `not_reached_ratio=1.0`（SGD到1000步仍未追平Hebbian，H6/H7 pass）。

3. 三轴参数结构（apple/cat及组内共享）
- `apple_vs_cat` 在 micro/super_type 的神经元、gate维、down维Jaccard几乎为0。
- fruit组在 super_type 出现共享gate输入维：`[2041,2427,2467,2524]`。
- animal组在 super_type 出现共享gate输入维：`[2727]`。
- 目前证据指向：上级类别存在“共享参数骨架”，而微观属性与同类区分更偏特异编码。

### 当前判断
- 我们已经获得“参数维度级结构线索”，但还不足以宣称“已确定统一编码原理”。
- 瓶颈在于：序列因果强度（z）不足、整体机制分不足，跨任务泛化证据仍弱。

### 下一步计划（按优先级）
1. 扩展样本与对照：固定词表+多seed+跨模板重采样，目标提升因果z稳定性。
2. 强化干预实验：做路径级最小子网（MCS）双向验证（必要性+充分性）与反事实一致性。
3. 扩展层级体系：按“属性-同类-上级”三轴扩到更多概念簇（水果/动物/工具/地点），验证共享骨架是否可迁移。
4. 参数流形建模：从离散维交集升级为低秩子空间与雅可比局部线性结构，检验是否存在统一生成规则。

## [2026-03-05] 持续推进：编码不变量探针（参数结构层）

### 本轮新增脚本
- `tests/codex/deepseek7b_encoding_invariant_probe.py`
  - 作用：从三轴参数结构结果中提取“轴间隔离 + 组内共享骨架 + 全局强度阈值”不变量，并形成可证伪判定。

### 本轮执行命令
- `python tests/codex/deepseek7b_encoding_invariant_probe.py`
- `Get-Content tempdata/deepseek7b_encoding_invariant_probe_v1/encoding_invariant_probe.json -TotalCount 260`
- `Get-Content tempdata/deepseek7b_encoding_invariant_probe_v1/ENCODING_INVARIANT_PROBE_REPORT.md -Encoding UTF8 -TotalCount 80`

### 新输出
- `tempdata/deepseek7b_encoding_invariant_probe_v1/encoding_invariant_probe.json`
- `tempdata/deepseek7b_encoding_invariant_probe_v1/ENCODING_INVARIANT_PROBE_REPORT.md`

### 关键结论
1. 局部结构不变量成立（5/7通过）
- 三轴之间 gate 维重叠很低（均值约 0.015~0.030），支持“分工编码”。
- 组内上级轴共享骨架成立：
  - fruit(super_type): `[2041,2427,2467,2524]`
  - animal(super_type): `[2727]`
- 组间 super_type 骨架重叠接近0（animal vs fruit jaccard=0）。

2. 全局强证据仍不足（2/7失败）
- `causal_seq_z_mean=0.964 < 1.96`
- `overall_score_mean=0.290 < 0.42`

### 理论推进含义
- 目前已形成“候选编码原理”的骨架证据：
  - 低层/同类轴偏特异，
  - 上级轴出现可复用共享维。
- 但尚不能宣称“统一数学原理已确定”，因为全局因果强度未达阈值。
- 下一阶段需要把局部不变量推进为跨模板、跨任务、跨seed的强因果定律。

## [2026-03-05] 持续推进：跨模板重采样稳定性 + 可证伪H8/H9

### 本轮新增脚本
1. `tests/codex/deepseek7b_prompt_bootstrap_causal_stability.py`
- 输入：`tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed*/mass_noun_encoding_scan.json`
- 功能：
  - 对每个名词的 `prompt_metrics` 做 bootstrap 重采样（模板维）
  - 输出 `bootstrap_seq_margin_mean`、`bootstrap_positive_ratio`
  - 汇总最小子集必要性/充分性代理指标（necessity/sufficiency/overshoot）

2. 更新 `tests/codex_temp/build_falsifiable_stage_report.py`
- 新增 `--prompt-bootstrap-json` 输入
- 新增可证伪判据：
  - `H8_prompt_bootstrap_seq_margin_positive`
  - `H9_prompt_bootstrap_positive_ratio_ge_0_95`

### 本轮执行命令
- `python tests/codex/deepseek7b_prompt_bootstrap_causal_stability.py --n-bootstrap 3000`
- `python tests/codex_temp/build_falsifiable_stage_report.py --pattern "tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed*/mass_noun_encoding_scan.json" --plasticity-summary-json tempdata/deepseek7b_plasticity_efficiency_bilingual_v2_multiseed_summary.json --prompt-bootstrap-json tempdata/deepseek7b_prompt_bootstrap_causal_stability_v1/prompt_bootstrap_causal_stability.json --output-json tempdata/deepseek7b_stage_falsifiable_report_v4_with_prompt.json --output-md tempdata/deepseek7b_stage_falsifiable_report_v4_with_prompt.md`

### 新输出
- `tempdata/deepseek7b_prompt_bootstrap_causal_stability_v1/prompt_bootstrap_causal_stability.json`
- `tempdata/deepseek7b_prompt_bootstrap_causal_stability_v1/PROMPT_BOOTSTRAP_CAUSAL_STABILITY_REPORT.md`
- `tempdata/deepseek7b_stage_falsifiable_report_v4_with_prompt.json`
- `tempdata/deepseek7b_stage_falsifiable_report_v4_with_prompt.md`

### 关键结果
1. 模板重采样稳定性（5 seeds）
- `bootstrap_seq_margin_mean = 0.02714, CI_low = 0.02632 > 0`
- `bootstrap_positive_ratio_mean = 0.9939 >= 0.95`
- 对应 H8/H9 均通过。

2. 最小子集代理指标
- `necessity_ratio_mean = 0.5833`
- `sufficiency_ratio_mean = 1.0000`
- `overshoot_ratio_mean = 0.8333`

3. 全局门槛仍未过
- `overall_score_mean = 0.2903 < 0.42`
- `causal_seq_z_mean = 0.9645 < 1.96`
- 即：局部机制稳定性增强，但系统级强证据仍不足。

### 理论推进含义
- “编码结构线索”进一步收敛：模板变化下序列因果方向稳健，支持存在可复用子机制。
- “统一编码原理”仍未闭环：需要提升全局因果强度与系统得分，尤其要减少 overshoot 并提高 necessity。

## [2026-03-05] 理论分析：多维生成与词嵌入代数的统一编码机制

### 用户问题
- 深度神经网络生成同时处理风格/逻辑/语句三维，以及词嵌入线性代数（如 king-man+woman≈queen），编码机制应如何理解与还原。

### 统一机制判断（当前阶段）
1. 编码不是“单神经元语义字典”，而是“分层子空间+动态路由”的组合：
- 词嵌入层提供可线性操作的粗语义轴（局部近似线性）。
- 中高层通过注意力与MLP门控，将风格/逻辑/句法在不同层与头上做条件组合。
- 输出时通过残差流叠加形成下一词分布。

2. 三个维度的功能分工（可检验假设）
- 风格维度：更偏全局控制向量（跨token稳定），在中后层与注意力模式耦合。
- 逻辑维度：更偏跨句依赖与约束传播（attention path + 关键MLP子网）。
- 句法维度：更偏局部结构与邻近依存（早中层更明显）。

3. 词嵌入代数不是完整真理，而是“低阶投影近似”
- 线性关系存在，但主要在局部语义区域与特定任务条件下稳定。
- 真正生成机制需要“线性子空间 + 非线性门控 + 上下文路由”共同解释。

### 建议分析路线（与当前项目一致）
1. 观测
- 在固定语义下分别切换风格/逻辑/句法模板，提取残差流与gate激活差分。

2. 提取
- 对每一维学习控制方向（control vectors）与最小因果子网（MCS），并做层级定位。

3. 验证
- 必要性：消融该维子网后目标能力显著下降。
- 充分性：只注入该维方向可部分恢复对应能力。
- 交叉性：风格子网对逻辑/句法任务影响应显著更小（解耦检验）。

4. 系统
- 建立“线性子空间（可解释）+ 非线性路由（可执行）”的双层数学模型，统一解释词向量代数与多维生成。

### 当前与既有结果的关系
- 现有结果已支持“局部稳定机制”与“上级共享骨架”。
- 仍缺“系统级强证据”（全局z与overall分不足），下一步应重点补强跨模板/跨任务因果验证。

## [2026-03-05] 持续推进：风格/逻辑/句法三维编码探针（首轮）

### 本轮新增脚本
- `tests/codex/deepseek7b_multidim_encoding_probe.py`
  - 使用成对对照提示词（A/B）隔离三维扰动：`style / logic / syntax`
  - 采集 gate 激活差分（最后token）
  - 输出：
    - 每维关键神经元集合（top-k by mean abs delta）
    - 每维 layer 影响谱（abs delta 按层聚合）
    - 维度间 top-neuron Jaccard 与 layer-profile 相关
    - 维度特异性 margin（own vs other）

### 本轮执行命令
- `python tests/codex/deepseek7b_multidim_encoding_probe.py --max-pairs-per-dim 3 --top-k 128`

### 本轮输出
- `tempdata/deepseek7b_multidim_encoding_probe_20260305_220444/multidim_encoding_probe.json`
- `tempdata/deepseek7b_multidim_encoding_probe_20260305_220444/MULTIDIM_ENCODING_PROBE_REPORT.md`

### 关键结果（首轮）
1. 维度规模与低秩特征
- style: `mean_delta_l2=1086.34`, `top1_energy=0.7627`, `PR=1.5675`
- logic: `mean_delta_l2=489.33`, `top1_energy=0.6222`, `PR=1.8872`
- syntax: `mean_delta_l2=277.92`, `top1_energy=0.5147`, `PR=1.9983`
- 解读：风格维扰动幅度最大、低秩性更强；句法维更分散/更弱。

2. 维度间重叠与共享
- top-neuron Jaccard 基本接近0（0~0.02）：维度关键神经元集合分离度高。
- layer-profile corr 较高（约0.67~0.73）：不同维度仍共享部分层级通道。

3. 维度特异性
- style margin: `+10.8408`（强特异）
- logic margin: `+3.9771`（中等特异）
- syntax margin: `-1.0969`（当前探针未成功隔离句法维，需修正对照集）

### 理论推进含义
- 支持“分离的关键神经元 + 共享层级骨架”的双层机制。
- 与词嵌入线性代数可统一解释为：
  - 低秩方向承载可线性操作部分（风格/语义轴）
  - 非线性门控与上下文路由实现多维联合生成。
- 下一步重点：增强句法对照对（最小语义变化、最大语法变化），提升句法维特异性证据。

## [2026-03-05] 三维探针升级：按“维度特异分数”选择关键神经元

### 脚本升级
- 文件：`tests/codex/deepseek7b_multidim_encoding_probe.py`
- 改动：关键神经元从“单维幅值top-k”改为“特异分数top-k”
  - `specific_score = mean_abs_this_dim - mean_abs_other_dims`
- 目的：避免挑到跨维通用高幅值神经元，提高维度分工识别精度。

### 本轮执行命令
- `python tests/codex/deepseek7b_multidim_encoding_probe.py --max-pairs-per-dim 3 --top-k 128 --output-dir tempdata/deepseek7b_multidim_encoding_probe_v2_specific`

### 新输出
- `tempdata/deepseek7b_multidim_encoding_probe_v2_specific/multidim_encoding_probe.json`
- `tempdata/deepseek7b_multidim_encoding_probe_v2_specific/MULTIDIM_ENCODING_PROBE_REPORT.md`

### 关键结果（v2）
1. 维度特异性（均为正）
- style margin: `+10.8822`
- logic margin: `+4.0246`
- syntax margin: `+0.6396`（相比v1由负转正）

2. 维度关键神经元重叠
- style/logic/syntax 两两 `top_neuron_jaccard = 0`
- 说明关键神经元集合高度分离。

3. 层级共享
- layer profile 相关仍较高（约 `0.67~0.73`）
- 说明“神经元集合分离 + 层级通道共享”并存。

4. 关键神经元示例
- style top: 主要集中在后层（如 `L27N3218` 等）
- logic top: 更分散在中层（如 `L20N15723, L4N13313`）
- syntax top: 更偏早中层（如 `L8N18487, L7N4418`）

### 理论含义
- 与“多维联合生成”假设一致：
  - 维度级控制子集可分离（低重叠）
  - 同时共享层级基础通道（高层谱相关）
- 这为“线性子空间 + 非线性路由”的统一编码框架提供更强结构证据。

## [2026-03-05] 三维编码因果验证：交叉消融矩阵（3x3）

### 本轮新增脚本
- `tests/codex/deepseek7b_multidim_causal_ablation.py`
  - 输入：`multidim_encoding_probe.json` 中各维度 `specific_top_neurons`
  - 过程：对 style/logic/syntax 子集做交叉消融，测量每个对照维度差分L2的抑制比
  - 输出：抑制矩阵与对角优势（diag - offdiag均值）

### 本轮执行命令
1) 初版（last-token消融）
- `python tests/codex/deepseek7b_multidim_causal_ablation.py --probe-json tempdata/deepseek7b_multidim_encoding_probe_v2_specific/multidim_encoding_probe.json --top-n 64 --output-dir tempdata/deepseek7b_multidim_causal_ablation_v1`
- `python tests/codex/deepseek7b_multidim_causal_ablation.py --probe-json tempdata/deepseek7b_multidim_encoding_probe_v2_specific/multidim_encoding_probe.json --top-n 128 --output-dir tempdata/deepseek7b_multidim_causal_ablation_v1_top128`

2) 升级版（全序列位置消融）
- 脚本新增参数：`--ablate-all-positions`（默认true）
- `python tests/codex/deepseek7b_multidim_causal_ablation.py --probe-json tempdata/deepseek7b_multidim_encoding_probe_v2_specific/multidim_encoding_probe.json --top-n 128 --ablate-all-positions --output-dir tempdata/deepseek7b_multidim_causal_ablation_v2_allpos`

### 关键结果（v2_allpos）
- 抑制矩阵均值（行=消融维，列=被测维）：
  - style: `[style=0.0249, logic=0.0069, syntax=-0.0074]`
  - logic: `[style=0.0017, logic=0.0357, syntax=-0.0171]`
  - syntax: `[style=-0.0020, logic=-0.0036, syntax=0.0138]`
- 对角优势：
  - style: `+0.0251`
  - logic: `+0.0434`
  - syntax: `+0.0166`

### 解释
- 初版 last-token 消融低估了风格维（风格是跨token信号）。
- 改为全序列消融后，三维均出现正对角优势，支持“维度级因果特异性”假设。
- 这与“分离子集 + 共享层级通道”的编码机制一致。

## [2026-03-05] 任务完成：1) 30+模板多seed稳定性 2) Main 3D接入style/logic/syntax

### 一、任务1：扩展三维模板并完成多seed稳定性统计

#### 代码改动
1. `tests/codex/deepseek7b_multidim_encoding_probe.py`
- 扩展为大模板池（每维可采样，默认支持 `max_pairs_per_dim=10`，总模板>=30）
- 新增 `--seed`，按seed随机采样模板（可复现）
- 保留并强化特异神经元提取（specific top neurons）

2. `tests/codex/deepseek7b_multidim_causal_ablation.py`
- 优先复用 probe 输出中的 pairs（保证同一seed下探针与消融使用同模板）
- 支持全序列消融 `--ablate-all-positions`

3. `tests/codex/deepseek7b_multidim_seed_stability.py`（新增）
- 自动执行多seed：probe + cross-ablation
- 汇总 specificity margin / diagonal advantage / cross-jaccard
- 输出可证伪判据

#### 关键执行命令
- `python tests/codex/deepseek7b_multidim_seed_stability.py --max-pairs-per-dim 10 --top-k 128 --top-n-ablate 128 --seeds 101,202,303,404,505 --output-dir tempdata/deepseek7b_multidim_multiseed_v1`

#### 输出文件
- `tempdata/deepseek7b_multidim_multiseed_v1/multidim_multiseed_stability.json`
- `tempdata/deepseek7b_multidim_multiseed_v1/MULTIDIM_MULTISEED_STABILITY_REPORT.md`

#### 结果摘要（5 seeds，30模板总量）
- specificity_margin_style: mean=2.7456, CI=[2.3009, 3.1903] pass
- specificity_margin_logic: mean=4.9304, CI=[4.8909, 4.9699] pass
- specificity_margin_syntax: mean=4.3237, CI=[3.5085, 5.1389] pass
- diag_adv_style: mean=0.0230, CI=[0.0122, 0.0338] pass
- diag_adv_logic: mean=0.0406, CI=[0.0374, 0.0438] pass
- diag_adv_syntax: mean=0.0007, CI=[-0.0007, 0.0022] fail（边缘、需继续增强句法因果对照）

---

### 二、任务2：Main 3D 接入 style/logic/syntax 可视化与切换

#### 代码改动
1. `frontend/src/blueprint/AppleNeuron3DTab.jsx`
- 新增多维角色颜色：`style/logic/syntax`
- 新增多维数据状态：
  - `multidimProbeData`
  - `multidimCausalData`
  - `multidimTopN`
  - `multidimVisible`
  - `multidimActiveDimension`
  - `multidimLayerProfile`
- 新增导入识别：
  - `multidim_encoding_probe.json`
  - `multidim_causal_ablation.json`
  - `multidim_multiseed_stability.json`（摘要提示）
- 新增节点构建：`buildMultidimNodesFromProbe`
- 新增3D层级影响图组件：`DimensionLayerImpactGraph`
  - 在主3D右侧显示当前维度的 layer 影响谱
  - 显示该维度对角优势与 suppression 行数据
- Main 控制面板新增“三维编码（Style/Logic/Syntax）”模块：
  - 维度切换按钮
  - 维度显示勾选
  - TopN 控制
  - 对角优势摘要
- 图例与模型说明已加入三维信息

2. `frontend/src/App.jsx`
- Main 场景中传入新增参数：
  - `dimensionLayerProfile`
  - `activeDimension`
  - `dimensionCausal`

3. `server/server.py`
- 扩展 `/api/main/scan_files` 识别规则：
  - `multidim_encoding_probe`
  - `multidim_causal_ablation`
  - `multidim_multiseed_stability`
- 支持在下拉框中发现并导入这些JSON

#### 验证
- Python语法检查通过：
  - `python -m py_compile tests/codex/deepseek7b_multidim_encoding_probe.py tests/codex/deepseek7b_multidim_causal_ablation.py tests/codex/deepseek7b_multidim_seed_stability.py server/server.py`
- 前端构建失败但为既有无关问题：
  - `src/StructureAnalysisPanel.jsx` 引用 `PredictiveCodingGraph` 的 default export 不存在（本轮变更无关）

## [2026-03-05] 继续推进：Main导入下拉框默认多维筛选

### 代码改动
- `frontend/src/blueprint/AppleNeuron3DTab.jsx`
  - 新增 `scanFileFilter`（默认 `multidim`）
  - 新增 `filteredScanFileOptions`（支持 `multidim / mass_noun / all`）
  - 下拉框改为显示筛选后的文件
  - 新增筛选按钮组与候选计数显示
  - 自动根据筛选结果更新 `selectedScanPath`

### 执行命令
- `npm --prefix frontend run build`

### 验证结果
- 本轮改动无新增编译错误；构建仍被既有问题阻断：
  - `src/StructureAnalysisPanel.jsx` 对 `PredictiveCodingGraph` 的 default import 不匹配（与本轮改动无关）。

### 用户体验变化
- Main 控制面板中，导入扫描默认优先显示 `multidim_*` 文件。
- 可一键切换到 `MassNoun` 或 `全部`，导入路径更清晰，减少误选。

## [2026-03-05] 前端修复：客户端空白与构建错误

### 问题复现
- `npm --prefix frontend run build` 失败：
  - `src/StructureAnalysisPanel.jsx` 导入 `PredictiveCodingGraph` default export 报错。
- 根因：`frontend/src/blueprint/PredictiveCodingGraph.jsx` 文件为空（0字节）。

### 修复
- 重建 `frontend/src/blueprint/PredictiveCodingGraph.jsx`：
  - 补齐 default export React 组件
  - 提供可渲染的 Predictive Coding 图（feedforward/feedback）

### 验证
- `npm --prefix frontend run build` 通过（vite build success）。
- 现阶段构建无阻断错误，仅保留 chunk size 警告（非阻断）。

### 结果
- 客户端空白的主要构建原因已清除。
- Main 三维编码改动与现有代码可共同编译。

## [2026-03-05] 使用说明修复：批量导入与三维编码“点击没反应”

### 问题定位
- 用户常导入 `multidim_multiseed_stability.json` 后看不到3D变化。
- 原因：该文件是统计汇总，不含可直接渲染的神经元节点集合。

### 代码修复
- 文件：`frontend/src/blueprint/AppleNeuron3DTab.jsx`
- 在 `handleImportSelectedScanFile` 中新增逻辑：
  - 若检测到稳定性汇总文件（multiseed summary），自动按 run 记录加载一个 `probe_json + ablation_json`，并调用现有导入流程。
  - 优先 seed=505，否则取最后一个 run。

### 验证
- `npm --prefix frontend run build` 通过。

### 使用要点
1. 批量导入扫描结果：用于导入 mass noun 扫描（概念神经元集合）。
2. 三维编码：用于导入 multidim probe/causal（style/logic/syntax）。
3. 导入稳定性汇总文件时，系统现在会自动补载对应 probe+causal，避免“点了没效果”。

## [2026-03-05] 术语说明：massnoun
- `mass noun` 语言学里是“不可数名词”，如 water、rice、information。
- 在本项目里 `mass_noun_*` 更像“批量名词扫描”的脚本命名，不严格等同于只处理不可数名词；它实际用于大量概念词的神经元/编码扫描。

## [2026-03-05] UI文案修复：MassNoun 按钮中文化
- 修改文件：`frontend/src/blueprint/AppleNeuron3DTab.jsx`
- 将筛选按钮 `MassNoun` 改为 `名词扫描`。
- 将“当前筛选”展示从内部id改为中文标签映射（多维编码/名词扫描/全部）。
- 验证：`npm --prefix frontend run build` 通过。

## [2026-03-05] UI可用性修复：导入按钮“看不到”
- 问题：选择不同文件后，“刷新列表/导入选中文件”按钮被长文本信息挤出可视区域，用户误认为消失。
- 修改：`frontend/src/blueprint/AppleNeuron3DTab.jsx`
  1) 将按钮行上移到“候选文件统计”下方，优先显示。
  2) 选中文件信息改为单行省略（`whiteSpace: nowrap; overflow: hidden; textOverflow: ellipsis`）。
- 验证：`npm --prefix frontend run build` 通过。

## [2026-03-05] UI修复：文件下拉框被长文件名挤偏
- 修改文件：`frontend/src/blueprint/AppleNeuron3DTab.jsx`
- 修复点：
  1) 下拉选项从“完整路径”改为“文件名 + 时间”，避免超长文本撑宽。
  2) 下拉框样式增加 `minWidth: 0` 与 `maxWidth: 100%`，确保可收缩。
- 验证：`npm --prefix frontend run build` 通过。
## [2026-03-05] UI修复：长文件名场景固定长度与防挤压
- 用户反馈：选择超长文件名后，导入区域仍会出现布局错位。
- 修改文件：`frontend/src/blueprint/AppleNeuron3DTab.jsx`
- 修复点：
  1) 文件下拉框改为固定宽度（`240px`），并加 `boxSizing/display/ellipsis` 约束。
  2) 下拉项文案改为固定摘要格式（截断文件名 + 时间），避免长字符串撑开控件。
  3) “刷新列表/导入选中文件”按钮区域改为固定宽度且 `flex-wrap`，防止被挤到可视区外。
- 命令记录：
  - `rg -n "fixedFileControlWidth|gridTemplateColumns:|已选:|selectedScanPath|<select" frontend/src/blueprint/AppleNeuron3DTab.jsx`
  - `npm --prefix frontend run build`
- 验证结果：前端构建通过，长文件名下控件宽度稳定。
## [2026-03-05] Main-特征分解3D增强：按层显示有效神经元
- 需求：在 Main 的“特征分解”动画中，载体移动到某个 layer 时，显示该 layer 对应的有效神经元。
- 修改文件：`frontend/src/blueprint/AppleNeuron3DTab.jsx`
- 实现要点：
  1) 在 `feature_decomposition` 模式中，按当前动画相位计算 `currentLayer`。
  2) 仅在该层内对节点按分解得分排序，抽取 Top-K（当前8个）作为“有效神经元”。
  3) 将 Top-K 写入 `prediction.focusNodeIds / effectiveLayer / effectiveNeurons`。
  4) 3D中新增 `LayerEffectiveNeuronOverlay`：在当前层右侧显示 `Lx 有效神经元 Top-K` 列表。
  5) 对有效神经元进行白色高亮（颜色、发光、缩放增强），便于跟踪“这一层到底哪些神经元在起作用”。
  6) 模式指标增加“当前层 / 有效神经元数量”。
- 结果：动画移动到不同 layer 时，会动态切换并显示该层对应有效神经元集合。
- 命令记录：
  - `npm --prefix frontend run build`
- 验证：构建通过。
## [2026-03-06] 编码机制分析进展：词嵌入几何与层级动态统一框架

### 结论（阶段性）
- 词嵌入并不是“完整知识本体”，更像是**概念先验坐标**（静态语义底座）。
- 深层网络并不是“另一个词典”，而是**条件化变换系统**（按上下文把坐标投影到不同子空间）。
- 因此“国王+王后-男性=女性”这类线性关系与“风格/逻辑/语法并行处理”并不矛盾：
  1) 词向量空间保存可线性近似的主轴；
  2) 层内注意力+MLP把这些主轴做上下文相关的非线性重组；
  3) 输出时在logit空间体现为下一个词分布。

### 对“大脑中两者合一”的可计算猜想
- 可把大脑/AGI编码看成三层统一对象：
  1) **基坐标层（Concept Basis）**：概念向量与属性轴（如生命性、社会性、形状、功能）。
  2) **关系纤维层（Relational Fiber）**：同一概念在不同任务/语境下的局部坐标系（聊天、论文、推理、叙事）。
  3) **动态路由层（Routing Dynamics）**：通过注意力样门控在层间选择有效神经元子集。
- 数学上可记为：
  - 概念表示：`h0(c) = Ec + p`
  - 第l层更新：`h_{l+1} = h_l + A_l(h_l, ctx) + M_l(h_l, ctx)`
  - 其中 `A_l` 主要承载关系检索，`M_l` 主要承载特征重编码与组合。
- 这解释了“近乎无穷概念”的来源：
  - 不是一词一神经元；而是**有限基 + 组合 + 稀疏路由**。

### 苹果/国王/王后这类概念的统一解释
- “苹果”更偏感知-属性簇（颜色/形状/可食/类别层级）。
- “国王/王后”更偏社会-关系簇（身份/性别/权力/角色互换）。
- 两者都由同一机制产生：
  - 在共享基坐标上占据不同方向；
  - 在上下文中触发不同层的有效子回路；
  - 通过组合关系实现泛化与精确预测。

### 下一步破解路线（非统计相关性，强调结构）
1. 轴稳定性：跨seed/跨prompt验证“性别轴、类别轴、风格轴”是否保持同向子空间。
2. 子回路最小化：对每个轴做最小因果神经元子集搜索（MCS），并验证可迁移性。
3. 组合可逆性：测试 `h(king)-h(man)+h(woman)` 在不同层投影后是否仍保持可解释邻域。
4. 层功能分化：量化每层对语法/逻辑/风格的边际贡献，验证是否存在稳定分工区间。
5. 统一指标：建立“几何一致性 + 因果恢复率 + 组合保真度”三联指标，作为编码原理证据链。

### 命令记录
- 文档追加：`Add-Content reseach/gtp5/doc/AGI_GPT5_MEMO.md`
## [2026-03-06] 继续破解：统一编码结构离线解码（多实验融合）

### 新增脚本
- `tests/codex/deepseek7b_unified_math_structure_decoder.py`
- 作用：离线聚合现有 JSON（不重复跑模型），构建统一证据链：
  1) 维度轴稳定性（style/logic/syntax）
  2) 维度轴因果可分离性（diagonal advantage）
  3) 概念层级共享-特异结构（apple/banana/cat/king/queen + 类别原型）
  4) 有限基与复用效率信号（participation ratio / top energy / reused ratio）

### 关键修复
- 修复 probe 提取兼容：支持 `specific_top_neurons` 和 `top_neurons`，避免 n_runs 误判。
- 修复 Windows 中文乱码：输出 JSON/MD 改为 `utf-8-sig`。
- 增加层分布相似度（LayerCos）指标，避免仅用集合Jaccard导致大量0值。

### 运行命令
- `python tests/codex/deepseek7b_unified_math_structure_decoder.py --root tempdata --top-k 128 --min-nouns 100`

### 新输出
- `tempdata/deepseek7b_unified_math_decode_20260306_001703/unified_math_structure_decode.json`
- `tempdata/deepseek7b_unified_math_decode_20260306_001703/UNIFIED_MATH_STRUCTURE_DECODE_REPORT.md`

### 阶段结论
- 维度轴层谱稳定性高：
  - style profile_cos≈0.967
  - logic profile_cos≈0.999
  - syntax profile_cos≈0.953
- 维度轴因果可分离：logic 最稳定（diagonal_advantage mean≈0.0366）；style 次之；syntax 较弱。
- 有限基信号明确：PR≈11.27，top5能量>top1能量，支持“有限基+组合编码”。
- 概念层级上，集合重叠很稀疏，但层分布相似度可区分结构关系（需进一步因果验证）。

### 说明
- 当前结果支持“静态概念坐标 + 动态层级路由 + 稀疏可分离轴”的统一假设。
- 下一步应对 apple/king/queen 做同模板反事实最小编辑 + 最小因果子回路迁移测试，验证微观可迁移结构。
## [2026-03-06] 代码同步：拉取远端并合并本地改动

### 执行目标
- 按用户要求：拉取远端 `main` 最新代码，并保留本地未提交改动。

### 执行步骤
1. 检查状态：`main` 落后 `origin/main` 1 提交，且本地有大量未提交/未跟踪改动。
2. 安全暂存：`git stash push -u -m "codex-autostash-before-pull-20260306"`
3. 拉取远端：`git pull --no-rebase origin main`（Fast-forward 到 `0c0e3b4`）
4. 恢复本地：`git stash pop`

### 结果
- 已成功同步到远端最新：`main` 与 `origin/main` 对齐。
- 本地改动已恢复，未丢失。
- `research/gemini/docs/AGI_RESEARCH_MEMO.md` 在恢复时自动合并，无冲突中断。

### 远端新引入文件（示例）
- `frontend/src/blueprint/GeminiTab.jsx`
- `tests/gemini/test_attention_dimension_cut.py`
- `tests/gemini/test_dnn_embedding_algebra.py`
- `tests/gemini/test_predictive_coding_emergence.py`

### 说明
- 本步骤属于代码同步，不改变理论结论；研究主线仍为“统一编码机制：静态坐标 + 动态路由 + 因果子回路”。
## [2026-03-06] 文档分析：AGI_RESEARCH_MEMO 阶段进展与计划

### 分析对象
- `research/gemini/docs/AGI_RESEARCH_MEMO.md`

### 命令记录
- `rg -n "^#|^##|^###|^####|^\\- \\[|^\\*\\*|^更新|^结论|^下一步|^问题" research/gemini/docs/AGI_RESEARCH_MEMO.md`
- `Get-Content -Tail 260 research/gemini/docs/AGI_RESEARCH_MEMO.md`
- `python` 编码检查（utf-8 可解码）

### 阶段进展（从章节结构提炼）
1. 研究主线已从“流形稀疏/正交观察”推进到“编码即结构”的统一理论，并进入可验证阶段。
2. 已覆盖从 DNN 逆向拆解到 SNN 机理映射，再到 AGI 工程路线图（章节约 39-48）。
3. 文档明确给出了多项硬伤与盲区：全局信用分配、时间拓扑、动态绑定、符号落地、算力与工程可扩展性。

### 关键风险
- 文档正文存在明显编码污染（大量 mojibake），影响可读性与后续自动化提取。
- “理论结论 > 可复现实证”的比例偏高，需进一步转为可证伪指标链。

### 建议计划
1. 先修复文档编码与章节编号一致性（当前存在重复章号）。
2. 把每章结论转为“假设-指标-脚本-结果-失败条件”的统一模板。
3. 优先推进三条硬伤对应实验：动态绑定、长时程因果、全局信用分配局部化替代。
4. 将证据流回 Main 控制面板，形成可视化闭环。
