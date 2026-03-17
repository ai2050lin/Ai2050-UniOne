# Structure Scan Format

- schema_version: `agi.deepseek.structure_scan.v1`
- `manifest.json`: run metadata, file map, config, device, timing
- `records.jsonl`: one line per item-pool analysis record
- `families.jsonl`: one line per family prototype per pool
- `closure_candidates.jsonl`: exact-closure candidate diagnostics
- `summary.json`: compact headline metrics for quick loading
- `REPORT.md`: human-readable stage summary

## records.jsonl
- `item.term`: analyzed token or concept string
- `item.category`: family/category label from the input inventory
- `pool`: survey/deep/closure
- `signature_top_indices`: sparse neuron signature
- `prompt_records`: per-prompt sparse traces
- survey pool can intentionally omit prompt-level traces to reduce export size

## families.jsonl
- `prototype_top_indices`: pooled family prototype signature
- `shared_neurons`: neurons reused by a large fraction of family members

## closure_candidates.jsonl
- `family_support_jaccard`: overlap with same-family prototype
- `wrong_family_margin`: same-family overlap minus best wrong-family overlap
- `exact_closure_proxy`: compact closure score for prioritization
