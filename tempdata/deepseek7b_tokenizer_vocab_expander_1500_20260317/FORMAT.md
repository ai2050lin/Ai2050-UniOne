# Format

- `expanded_vocab.csv`: balanced inventory for downstream structure scans; each row is `term,category`
- `combined_seed_plus_expanded.csv`: seed inventory plus balanced expansions, ready for larger structure scans
- `all_candidates.jsonl`: all mined tokenizer candidates with category score maps
- `balanced_candidates.jsonl`: the confidence-filtered balanced subset with ranks
- `summary.json`: run-level counts, thresholds, and top examples
- `REPORT.md`: short human-readable summary
