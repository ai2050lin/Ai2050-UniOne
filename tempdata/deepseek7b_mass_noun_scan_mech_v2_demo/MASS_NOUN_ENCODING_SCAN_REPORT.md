# Mass Noun Encoding Scan Report

## Core Findings
- Nouns scanned: 8
- Neuron space: 530432 (28 layers x 18944 d_ff)
- Reused neurons (>= 5 nouns): 0 (0.000000)
- Layer usage entropy (normalized): 0.9885
- Top-3 layer usage ratio: 0.1667

## Pairwise Structure
- Within-category cosine mean: 0.9774
- Between-category cosine mean: 0.0000
- Cosine gap: 0.9774
- Within-category Jaccard mean: 0.1071
- Between-category Jaccard mean: 0.0000
- Jaccard gap: 0.1071

## Low-Rank Encoding
- Participation ratio: 2.3934
- Top-1 energy ratio: 0.6162
- Top-5 energy ratio: 1.0000

## Mechanism Scorecard
- Overall score: 0.6548
- Grade: moderate_mechanistic_evidence
- Structure separation: 0.8289
- Reuse sparsity structure: 0.5811
- Low-rank compactness: 1.0000
- Causal evidence: 0.3160

### Guidance
- Increase ablation samples and random trials to strengthen causal margin confidence.

## Causal Ablation
- Eligible nouns: 8
- Sampled nouns: 6
- Mean signature prob drop: 0.000000
- Mean random prob drop: -0.000000
- Mean causal margin (prob): 0.000000
- Mean causal margin (logprob): 0.012385
- Mean causal margin (rank-worse): 301.037037
- Positive causal margin ratio: 0.8333
- Causal margin prob z-score: 0.9948
- Causal margin prob 95% CI: [-0.000000, 0.000000]
- Reused neuron ablation:
  - mean reused prob drop: 0.000000
  - mean random prob drop: -0.000000
  - mean causal margin: 0.000000

## Top Reused Neurons (Top-20)
- L19N17314: used by 3 noun signatures
- L11N6359: used by 3 noun signatures
- L15N1946: used by 3 noun signatures
- L3N9549: used by 3 noun signatures
- L4N6633: used by 3 noun signatures
- L17N4911: used by 3 noun signatures
- L3N1127: used by 3 noun signatures
- L5N443: used by 3 noun signatures
- L16N18222: used by 3 noun signatures
- L11N11703: used by 3 noun signatures
- L21N2634: used by 3 noun signatures
- L10N532: used by 3 noun signatures
- L8N13531: used by 3 noun signatures
- L14N7980: used by 3 noun signatures
- L8N16165: used by 3 noun signatures
- L15N2021: used by 3 noun signatures
- L22N16864: used by 3 noun signatures
- L8N15826: used by 3 noun signatures
- L7N5224: used by 3 noun signatures
- L14N13338: used by 3 noun signatures