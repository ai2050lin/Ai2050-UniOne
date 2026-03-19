# ICSPB 大样本概念规律扫描报告

## 全局指标
- seed_count: 5
- category_count: 10
- noun_record_count: 600
- unique_noun_count: 120
- mean_noun_to_category_jaccard: 0.014843
- mean_noun_to_best_macro_jaccard: 0.000000
- mean_same_category_jaccard: 0.048777
- mean_cross_category_jaccard: 0.021039
- mean_same_cross_margin: 0.027738
- positive_same_cross_margin_ratio: 0.791667
- mean_category_to_best_macro_jaccard: 0.000000
- macro_stronger_than_micro_category_ratio: 0.000000
- mean_cross_seed_signature_jaccard: 1.000000
- mean_layer_peak_band_agreement_ratio: 1.000000

## 类别均值
- animal: micro=0.013555, macro=0.000000, margin=0.147713
- celestial: micro=0.038060, macro=0.000000, margin=-0.006444
- food: micro=0.006768, macro=0.000000, margin=0.001710
- fruit: micro=0.006241, macro=0.000000, margin=0.093499
- human: micro=0.000837, macro=0.000000, margin=0.002482
- nature: micro=0.020610, macro=0.000000, margin=0.001791
- object: micro=0.006184, macro=0.000000, margin=-0.010489
- tech: micro=0.031167, macro=0.000000, margin=0.000602
- vehicle: micro=0.032286, macro=0.000000, margin=-0.000197
- weather: micro=0.021200, macro=0.000000, margin=-0.014974

## 跨种子稳定性 Top10
- airplane / vehicle / stability=1.000000 / band=early / agreement=1.000000
- algorithm / tech / stability=1.000000 / band=late / agreement=1.000000
- apple / fruit / stability=1.000000 / band=late / agreement=1.000000
- artist / human / stability=1.000000 / band=early / agreement=1.000000
- asteroid / celestial / stability=1.000000 / band=early / agreement=1.000000
- banana / fruit / stability=1.000000 / band=early / agreement=1.000000
- bear / animal / stability=1.000000 / band=early / agreement=1.000000
- bed / object / stability=1.000000 / band=late / agreement=1.000000
- bicycle / vehicle / stability=1.000000 / band=early / agreement=1.000000
- bird / animal / stability=1.000000 / band=late / agreement=1.000000

## 假设判定
- H1_same_category_separation_positive: FAIL
- H2_category_to_macro_stronger_than_noun_to_category: FAIL
- H3_cross_seed_signature_has_stability: PASS
- H4_same_category_mean_exceeds_cross_category_mean: PASS
- H5_seed_diversity_is_nontrivial: FAIL
