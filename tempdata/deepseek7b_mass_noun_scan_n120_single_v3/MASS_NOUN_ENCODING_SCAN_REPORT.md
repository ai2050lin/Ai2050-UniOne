# Mass Noun Encoding Scan Report

## Core Findings
- Nouns scanned: 120
- Neuron space: 530432 (28 layers x 18944 d_ff)
- Reused neurons (>= 5 nouns): 120 (0.000226)
- Layer usage entropy (normalized): 0.9614
- Top-3 layer usage ratio: 0.1925

## Pairwise Structure
- Within-category cosine mean: 0.9814
- Between-category cosine mean: 0.9779
- Cosine gap: 0.0036
- Within-category Jaccard mean: 0.0552
- Between-category Jaccard mean: 0.0209
- Jaccard gap: 0.0343

## Low-Rank Encoding
- Participation ratio: 12.5992
- Top-1 energy ratio: 0.2588
- Top-5 energy ratio: 0.4192

## Mechanism Scorecard
- Overall score: 0.2778
- Grade: insufficient_evidence
- Structure separation: 0.0447
- Reuse sparsity structure: 0.5964
- Low-rank compactness: 0.6354
- Causal evidence: 0.0233

### Guidance
- Increase category coverage and rebalance samples to improve within-vs-between separation.
- Increase ablation samples and random trials to strengthen causal margin confidence.

## Causal Ablation
- Eligible nouns: 120
- Sampled nouns: 60
- Mean signature prob drop: 0.000000
- Mean random prob drop: 0.000000
- Mean causal margin (prob): 0.000000
- Mean causal margin (logprob): -0.000549
- Mean causal margin (rank-worse): -196.172222
- Mean causal margin (seq logprob): -0.012365
- Mean causal margin (seq avg logprob): -0.010489
- Positive causal margin ratio: 0.6167
- Causal margin prob z-score: 0.9227
- Causal margin seq-logprob z-score: -2.7403
- Causal margin prob 95% CI: [-0.000000, 0.000000]
- Causal margin seq-logprob 95% CI: [-0.021209, -0.003521]
- Reused neuron ablation:
  - mean reused prob drop: -0.000000
  - mean random prob drop: 0.000000
  - mean causal margin: -0.000000

## Top Reused Neurons (Top-20)
- L15N638: used by 19 noun signatures
- L2N18502: used by 19 noun signatures
- L5N6513: used by 19 noun signatures
- L6N14991: used by 19 noun signatures
- L3N16290: used by 19 noun signatures
- L26N5206: used by 19 noun signatures
- L4N13593: used by 19 noun signatures
- L4N10263: used by 19 noun signatures
- L19N7624: used by 19 noun signatures
- L22N5178: used by 19 noun signatures
- L27N16004: used by 19 noun signatures
- L7N4395: used by 19 noun signatures
- L21N544: used by 19 noun signatures
- L27N7717: used by 19 noun signatures
- L11N15095: used by 19 noun signatures
- L20N17841: used by 19 noun signatures
- L15N383: used by 19 noun signatures
- L3N2171: used by 19 noun signatures
- L14N11534: used by 19 noun signatures
- L7N12457: used by 19 noun signatures