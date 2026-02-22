# Structure Recovery Pipeline Kickoff

- Date: 2026-02-20
- Status: pass

| Stage | Name | Status | Key Metric |
|---|---|---|---|
| A0 | encoding_genesis | pass | encoding_core=0.6711, h3=mixed_conflict |
| A | invariant_discovery | pass | stability=0.8416, candidates=19 |
| B | causal_filtering | pass | feature_top1=0.0646, layerwise=0.0166 |
| C | minimal_reconstruction | pass | best_val_acc=0.848602 |
| D | cross_modal_assembly | pass | val_fused_acc=0.99625 |
| E | open_falsification | pass | in_pool_no_leak=provisional_support (support=2); strict_clusters=4->0; v31+v32 support range=1..2 (4/6 provisional_support), falsify_max=0 |

## Next Actions
1. Raise strict support floor from `1` to `2` on a new fresh seed block under `hybrid_support_boost_v6`.
2. Add one more calibration pass for `gpt2` near-threshold categories (`antonym/fact`) without weakening strict guard.
3. Distill strict constructive behavior into stage-C constrained reconstruction features.
4. Extend strict holdout validation from text-only tasks to cross-modal tasks.
5. Keep strict falsify guard (`falsify_models_max=0`) as a hard gate in all future profiles.
