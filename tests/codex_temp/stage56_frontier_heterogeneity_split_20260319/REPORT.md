# Stage56 前沿异质性拆分摘要

- row_count: 72
- main_judgment: 前沿层已从单一混合项继续拆成压缩、覆盖、分离、晚移和覆盖减压缩差值，可以直接判断前沿到底是通过哪类机制进入目标分裂。

## Stable Features
- frontier_compaction_term: negative
- frontier_coverage_term: negative
- frontier_separation_term: negative
- frontier_compaction_late_shift: positive
- frontier_balance_term: positive

## Fits
- target: union_joint_adv
  intercept: +0.227584
  frontier_compaction_term: -10.707842
  frontier_coverage_term: -5.895084
  frontier_separation_term: -1.056750
  frontier_compaction_late_shift: +7.964439
  frontier_coverage_late_shift: -0.237644
  frontier_balance_term: +4.812758
- target: union_synergy_joint
  intercept: -0.187340
  frontier_compaction_term: -0.976149
  frontier_coverage_term: -0.539814
  frontier_separation_term: -0.604474
  frontier_compaction_late_shift: +1.358717
  frontier_coverage_late_shift: +0.169424
  frontier_balance_term: +0.436335
- target: strict_positive_synergy
  intercept: -2.717063
  frontier_compaction_term: -39.904896
  frontier_coverage_term: -19.313219
  frontier_separation_term: -1.827392
  frontier_compaction_late_shift: +25.475709
  frontier_coverage_late_shift: -2.052281
  frontier_balance_term: +20.591678
