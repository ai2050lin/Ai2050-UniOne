# Mass Noun Encoding Scan Report

## Core Findings
- Nouns scanned: 120
- Neuron space: 530432 (28 layers x 18944 d_ff)
- Reused neurons (>= 5 nouns): 120 (0.000226)
- Layer usage entropy (normalized): 0.9518
- Top-3 layer usage ratio: 0.2616

## Pairwise Structure
- Within-category cosine mean: 0.9724
- Between-category cosine mean: 0.9709
- Cosine gap: 0.0015
- Within-category Jaccard mean: 0.0282
- Between-category Jaccard mean: 0.0206
- Jaccard gap: 0.0076

## Low-Rank Encoding
- Participation ratio: 7.5091
- Top-1 energy ratio: 0.3099
- Top-5 energy ratio: 0.6177

## Mechanism Scorecard
- Overall score: 0.3006
- Grade: insufficient_evidence
- Structure separation: 0.0000
- Reuse sparsity structure: 0.6359
- Low-rank compactness: 0.8035
- Causal evidence: 0.0000

### Guidance
- Increase category coverage and rebalance samples to improve within-vs-between separation.
- Increase ablation samples and random trials to strengthen causal margin confidence.

## Causal Ablation
- Eligible nouns: 120
- Sampled nouns: 40
- Mean signature prob drop: -0.000000
- Mean random prob drop: -0.000000
- Mean causal margin (prob): -0.000000
- Mean causal margin (logprob): -0.003067
- Mean causal margin (rank-worse): -74.755556
- Positive causal margin ratio: 0.5250
- Causal margin prob z-score: -0.9104
- Causal margin prob 95% CI: [-0.000000, 0.000000]
- Reused neuron ablation:
  - mean reused prob drop: -0.000000
  - mean random prob drop: 0.000000
  - mean causal margin: -0.000000

## Top Reused Neurons (Top-20)
- L1N15422: used by 18 noun signatures
- L4N10047: used by 18 noun signatures
- L2N17300: used by 18 noun signatures
- L20N7404: used by 18 noun signatures
- L19N11417: used by 18 noun signatures
- L4N3366: used by 18 noun signatures
- L4N14694: used by 18 noun signatures
- L9N6627: used by 18 noun signatures
- L16N8707: used by 18 noun signatures
- L4N2231: used by 18 noun signatures
- L4N10073: used by 18 noun signatures
- L21N16037: used by 18 noun signatures
- L4N8414: used by 18 noun signatures
- L16N398: used by 18 noun signatures
- L20N4171: used by 18 noun signatures
- L14N14064: used by 18 noun signatures
- L16N9672: used by 18 noun signatures
- L17N4353: used by 18 noun signatures
- L11N6323: used by 18 noun signatures
- L1N290: used by 18 noun signatures