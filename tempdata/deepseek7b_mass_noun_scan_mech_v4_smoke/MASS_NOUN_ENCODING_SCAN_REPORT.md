# Mass Noun Encoding Scan Report

## Core Findings
- Nouns scanned: 12
- Neuron space: 530432 (28 layers x 18944 d_ff)
- Reused neurons (>= 5 nouns): 120 (0.000226)
- Layer usage entropy (normalized): 0.9751
- Top-3 layer usage ratio: 0.2021

## Pairwise Structure
- Within-category cosine mean: 0.9777
- Between-category cosine mean: 0.0000
- Cosine gap: 0.9777
- Within-category Jaccard mean: 0.1515
- Between-category Jaccard mean: 0.0000
- Jaccard gap: 0.1515

## Low-Rank Encoding
- Participation ratio: 2.5157
- Top-1 energy ratio: 0.6090
- Top-5 energy ratio: 0.9014

## Mechanism Scorecard
- Overall score: 0.6254
- Grade: moderate_mechanistic_evidence
- Structure separation: 0.9107
- Reuse sparsity structure: 0.5971
- Low-rank compactness: 1.0000
- Causal evidence: 0.1302

### Guidance
- Increase ablation samples and random trials to strengthen causal margin confidence.

## Causal Ablation
- Eligible nouns: 12
- Sampled nouns: 8
- Mean signature prob drop: 0.000000
- Mean random prob drop: -0.000000
- Mean causal margin (prob): 0.000000
- Mean causal margin (logprob): 0.012233
- Mean causal margin (rank-worse): -86.437500
- Mean causal margin (seq logprob): 0.013709
- Mean causal margin (seq avg logprob): 0.007507
- Positive causal margin ratio: 0.7500
- Causal margin prob z-score: 1.3517
- Causal margin seq-logprob z-score: 1.0775
- Causal margin prob 95% CI: [-0.000000, 0.000000]
- Causal margin seq-logprob 95% CI: [-0.011229, 0.038647]
- Reused neuron ablation:
  - mean reused prob drop: 0.000000
  - mean random prob drop: -0.000000
  - mean causal margin: 0.000000
- Minimal circuit extraction:
  - tested nouns: 4
  - target ratio: 0.80
  - mean subset size: 1.2500
  - mean recovery ratio: 1.2147
  - high recovery ratio: 1.0000
- Counterfactual validation:
  - tested pairs: 4
  - mean specificity margin (seq-logprob): -0.005127
  - same-category margin: -0.005127
  - cross-category margin: 0.000000
  - positive specificity ratio: 0.2500
  - specificity z-score: -1.3918
  - specificity 95% CI: [-0.012347, 0.002093]

## Top Reused Neurons (Top-20)
- L15N1946: used by 5 noun signatures
- L22N10055: used by 5 noun signatures
- L6N10896: used by 5 noun signatures
- L21N2634: used by 5 noun signatures
- L23N3961: used by 5 noun signatures
- L14N7980: used by 5 noun signatures
- L14N8483: used by 5 noun signatures
- L10N686: used by 5 noun signatures
- L18N16122: used by 5 noun signatures
- L3N8298: used by 5 noun signatures
- L20N10472: used by 5 noun signatures
- L13N16697: used by 5 noun signatures
- L18N138: used by 5 noun signatures
- L4N11730: used by 5 noun signatures
- L9N6417: used by 5 noun signatures
- L25N4870: used by 5 noun signatures
- L4N8358: used by 5 noun signatures
- L4N10794: used by 5 noun signatures
- L26N8343: used by 5 noun signatures
- L4N2949: used by 5 noun signatures