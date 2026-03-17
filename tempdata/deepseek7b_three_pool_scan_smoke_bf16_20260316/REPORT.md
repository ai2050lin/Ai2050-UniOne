# DeepSeek Three-Pool Structure Scan Report

## Run
- Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- Device: cuda:0
- Runtime sec: 14.9843
- Input items: 12
- Tasks: 32

## Headline Metrics
- Survey records: 12
- Deep records: 10
- Closure records: 10
- Family prototypes: 30
- Closure candidates: 20
- Records with nonfinite prompts: 0
- Total nonfinite prompts: 0
- Mean survey prompt stability: 0.8188
- Mean deep prompt stability: 0.7639
- Mean closure prompt stability: 0.6962

## Survey Category Coverage
- abstract: 2
- animal: 2
- celestial: 1
- food: 1
- fruit: 1
- human: 1
- nature: 1
- object: 1
- tech: 1
- vehicle: 1

## Top Closure Candidates
- dog (animal, closure): closure=0.6797, wrong_family_margin=0.0967
- sun (celestial, closure): closure=0.6787, wrong_family_margin=0.0896
- car (vehicle, closure): closure=0.6774, wrong_family_margin=0.0824
- bread (food, closure): closure=0.6753, wrong_family_margin=0.0967
- chair (object, closure): closure=0.6719, wrong_family_margin=0.0532
- truth (abstract, closure): closure=0.6711, wrong_family_margin=0.0606
- teacher (human, closure): closure=0.6681, wrong_family_margin=0.0532
- apple (fruit, closure): closure=0.6659, wrong_family_margin=0.0532
- algorithm (tech, closure): closure=0.6657, wrong_family_margin=0.0458
- tree (nature, closure): closure=0.6655, wrong_family_margin=0.0458
- tree (nature, deep): closure=0.6859, wrong_family_margin=0.0488
- truth (abstract, deep): closure=0.6824, wrong_family_margin=0.0606
- apple (fruit, deep): closure=0.6817, wrong_family_margin=0.0606
- bread (food, deep): closure=0.6785, wrong_family_margin=0.0488
- algorithm (tech, deep): closure=0.6782, wrong_family_margin=0.0368
- teacher (human, deep): closure=0.6774, wrong_family_margin=0.0488
- car (vehicle, deep): closure=0.6760, wrong_family_margin=0.0368
- chair (object, deep): closure=0.6757, wrong_family_margin=0.0368
- dog (animal, deep): closure=0.6753, wrong_family_margin=0.0368
- sun (celestial, deep): closure=0.6685, wrong_family_margin=0.0368
