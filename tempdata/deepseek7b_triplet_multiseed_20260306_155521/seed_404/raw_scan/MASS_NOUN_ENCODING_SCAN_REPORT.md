# Mass Noun Encoding Scan Report

## Core Findings
- Nouns scanned: 3
- Neuron space: 530432 (28 layers x 18944 d_ff)
- Reused neurons (>= 5 nouns): 0 (0.000000)
- Layer usage entropy (normalized): 0.9625
- Top-3 layer usage ratio: 0.2278

## Pairwise Structure
- Within-category cosine mean: 0.9911
- Between-category cosine mean: 0.9824
- Cosine gap: 0.0087
- Within-category Jaccard mean: 0.0000
- Between-category Jaccard mean: 0.0000
- Jaccard gap: 0.0000

## Low-Rank Encoding
- Participation ratio: 1.7277
- Top-1 energy ratio: 0.6985
- Top-5 energy ratio: 1.0000

## Mechanism Scorecard
- Overall score: 0.4000
- Grade: insufficient_evidence
- Structure separation: 0.0397
- Reuse sparsity structure: 0.6146
- Low-rank compactness: 1.0000
- Causal evidence: 0.1790

### Guidance
- Increase category coverage and rebalance samples to improve within-vs-between separation.
- Increase ablation samples and random trials to strengthen causal margin confidence.

## Causal Ablation
- Eligible nouns: 3
- Sampled nouns: 3
- Mean signature prob drop: 0.000000
- Mean random prob drop: 0.000000
- Mean causal margin (prob): -0.000000
- Mean causal margin (logprob): -0.001133
- Mean causal margin (rank-worse): 26.925926
- Mean causal margin (seq logprob): 0.018712
- Mean causal margin (seq avg logprob): 0.018712
- Positive causal margin ratio: 0.0000
- Causal margin prob z-score: -2.3128
- Causal margin seq-logprob z-score: 2.8410
- Causal margin prob 95% CI: [-0.000000, -0.000000]
- Causal margin seq-logprob 95% CI: [0.005803, 0.031621]
- Reused neuron ablation:
  - mean reused prob drop: -0.000000
  - mean random prob drop: 0.000000
  - mean causal margin: -0.000000
- Minimal circuit extraction:
  - tested nouns: 3
  - target ratio: 0.80
  - mean subset size: 0.6667
  - mean recovery ratio: 0.9201
  - high recovery ratio: 0.6667
- Counterfactual validation:
  - tested pairs: 4
  - mean specificity margin (seq-logprob): 0.033203
  - same-category margin: 0.037109
  - cross-category margin: 0.029297
  - positive specificity ratio: 1.0000
  - specificity z-score: 2.1707
  - specificity 95% CI: [0.003223, 0.063183]

## Top Reused Neurons (Top-20)
- L1N18893: used by 1 noun signatures
- L5N4546: used by 1 noun signatures
- L2N12994: used by 1 noun signatures
- L12N14362: used by 1 noun signatures
- L2N2451: used by 1 noun signatures
- L10N5261: used by 1 noun signatures
- L3N14102: used by 1 noun signatures
- L3N16678: used by 1 noun signatures
- L5N15347: used by 1 noun signatures
- L4N6473: used by 1 noun signatures
- L4N13063: used by 1 noun signatures
- L3N6036: used by 1 noun signatures
- L2N960: used by 1 noun signatures
- L3N11421: used by 1 noun signatures
- L5N1570: used by 1 noun signatures
- L1N9715: used by 1 noun signatures
- L27N594: used by 1 noun signatures
- L6N17183: used by 1 noun signatures
- L4N5239: used by 1 noun signatures
- L3N15527: used by 1 noun signatures