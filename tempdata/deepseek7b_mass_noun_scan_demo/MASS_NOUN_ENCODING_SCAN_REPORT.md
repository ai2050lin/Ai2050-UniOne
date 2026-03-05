# Mass Noun Encoding Scan Report

## Core Findings
- Nouns scanned: 40
- Neuron space: 530432 (28 layers x 18944 d_ff)
- Reused neurons (>= 5 nouns): 120 (0.000226)
- Layer usage entropy (normalized): 0.9794
- Top-3 layer usage ratio: 0.1900

## Pairwise Structure
- Within-category cosine mean: 0.9730
- Between-category cosine mean: 0.9741
- Cosine gap: -0.0012
- Within-category Jaccard mean: 0.0521
- Between-category Jaccard mean: 0.1000
- Jaccard gap: -0.0479

## Low-Rank Encoding
- Participation ratio: 6.2336
- Top-1 energy ratio: 0.3360
- Top-5 energy ratio: 0.6926

## Causal Ablation
- Eligible nouns: 40
- Sampled nouns: 20
- Mean signature prob drop: 0.000000
- Mean random prob drop: 0.000000
- Mean causal margin (prob): 0.000000
- Positive causal margin ratio: 0.5500
- Reused neuron ablation:
  - mean reused prob drop: 0.000000
  - mean random prob drop: -0.000000
  - mean causal margin: 0.000000

## Top Reused Neurons (Top-20)
- L4N10047: used by 11 noun signatures
- L20N10884: used by 11 noun signatures
- L20N7404: used by 11 noun signatures
- L1N15422: used by 11 noun signatures
- L19N14461: used by 11 noun signatures
- L20N4171: used by 11 noun signatures
- L19N16939: used by 11 noun signatures
- L2N17300: used by 11 noun signatures
- L12N2949: used by 11 noun signatures
- L11N6323: used by 11 noun signatures
- L8N8309: used by 11 noun signatures
- L20N2127: used by 11 noun signatures
- L18N6520: used by 11 noun signatures
- L14N10840: used by 11 noun signatures
- L20N1472: used by 11 noun signatures
- L20N2490: used by 11 noun signatures
- L13N16203: used by 11 noun signatures
- L21N18234: used by 11 noun signatures
- L9N12739: used by 11 noun signatures
- L16N9672: used by 11 noun signatures