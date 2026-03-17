# DeepSeek Three-Pool Structure Scan Report

## Run
- Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- Device: cuda:0
- Runtime sec: 9.4440
- Input items: 8
- Tasks: 12

## Headline Metrics
- Survey records: 4
- Deep records: 4
- Closure records: 4
- Family prototypes: 12
- Closure candidates: 8
- Records with nonfinite prompts: 0
- Total nonfinite prompts: 0
- Mean survey prompt stability: 0.7809
- Mean deep prompt stability: 0.7631
- Mean closure prompt stability: 0.6916

## Survey Category Coverage
- abstract: 1
- animal: 1
- human: 1
- tech: 1

## Top Closure Candidates
- human (human, closure): closure=0.6756, wrong_family_margin=0.0679
- symmetry (abstract, closure): closure=0.6727, wrong_family_margin=0.0752
- buffer (tech, closure): closure=0.6707, wrong_family_margin=0.0679
- kangaroo (animal, closure): closure=0.6691, wrong_family_margin=0.0679
- human (human, deep): closure=0.6881, wrong_family_margin=0.0606
- buffer (tech, deep): closure=0.6880, wrong_family_margin=0.0606
- symmetry (abstract, deep): closure=0.6860, wrong_family_margin=0.0838
- kangaroo (animal, deep): closure=0.6741, wrong_family_margin=0.0606
