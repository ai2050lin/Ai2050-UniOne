# DeepSeek Three-Pool Structure Scan Report

## Run
- Model: zai-org/GLM-4-9B-Chat-HF
- Device: cuda:0
- Runtime sec: 23.0142
- Input items: 12
- Tasks: 36

## Headline Metrics
- Survey records: 12
- Deep records: 12
- Closure records: 12
- Family prototypes: 12
- Closure candidates: 24
- Records with nonfinite prompts: 0
- Total nonfinite prompts: 0
- Mean survey prompt stability: 0.4124
- Mean deep prompt stability: 0.4450
- Mean closure prompt stability: 0.4364

## Survey Category Coverage
- animal: 3
- human: 3
- tech: 3
- vehicle: 3

## Top Closure Candidates
- teacher (human, closure): closure=0.6274, wrong_family_margin=0.1200
- car (vehicle, closure): closure=0.6248, wrong_family_margin=0.1133
- rabbit (animal, closure): closure=0.6180, wrong_family_margin=0.1125
- human (human, closure): closure=0.6139, wrong_family_margin=0.1100
- software (tech, closure): closure=0.6068, wrong_family_margin=0.1570
- vehicle (vehicle, closure): closure=0.6057, wrong_family_margin=0.0770
- librarian (human, closure): closure=0.5964, wrong_family_margin=0.0697
- kangaroo (animal, closure): closure=0.5874, wrong_family_margin=0.1076
- animal (animal, closure): closure=0.5649, wrong_family_margin=0.0472
- cart (vehicle, closure): closure=0.5335, wrong_family_margin=0.0605
- database (tech, closure): closure=0.5200, wrong_family_margin=0.1592
- tech (tech, closure): closure=0.4843, wrong_family_margin=-0.1788
- software (tech, deep): closure=0.6123, wrong_family_margin=0.1226
- vehicle (vehicle, deep): closure=0.6037, wrong_family_margin=0.0470
- car (vehicle, deep): closure=0.6004, wrong_family_margin=0.0577
- librarian (human, deep): closure=0.5998, wrong_family_margin=0.0570
- rabbit (animal, deep): closure=0.5857, wrong_family_margin=0.0342
- kangaroo (animal, deep): closure=0.5786, wrong_family_margin=0.0665
- teacher (human, deep): closure=0.5712, wrong_family_margin=0.0334
- animal (animal, deep): closure=0.5680, wrong_family_margin=0.0111
