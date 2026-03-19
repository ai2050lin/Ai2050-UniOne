# DeepSeek Three-Pool Structure Scan Report

## Run
- Model: zai-org/GLM-4-9B-Chat-HF
- Device: cuda:0
- Runtime sec: 86.6088
- Input items: 36
- Tasks: 72

## Headline Metrics
- Survey records: 36
- Deep records: 24
- Closure records: 12
- Family prototypes: 36
- Closure candidates: 36
- Records with nonfinite prompts: 0
- Total nonfinite prompts: 0
- Mean survey prompt stability: 0.4891
- Mean deep prompt stability: 0.4881
- Mean closure prompt stability: 0.4218

## Survey Category Coverage
- abstract: 3
- action: 3
- animal: 3
- celestial: 3
- food: 3
- fruit: 3
- human: 3
- nature: 3
- object: 3
- tech: 3
- vehicle: 3
- weather: 3

## Top Closure Candidates
- sensor (tech, closure): closure=0.6781, wrong_family_margin=0.2222
- humidity (weather, closure): closure=0.6669, wrong_family_margin=0.2160
- open (action, closure): closure=0.6603, wrong_family_margin=0.1449
- glory (abstract, closure): closure=0.6581, wrong_family_margin=0.1245
- yak (animal, closure): closure=0.6510, wrong_family_margin=0.1649
- honey (food, closure): closure=0.6490, wrong_family_margin=0.1516
- rickshaw (vehicle, closure): closure=0.6488, wrong_family_margin=0.1649
- nectarine (fruit, closure): closure=0.6472, wrong_family_margin=0.1516
- poet (human, closure): closure=0.6469, wrong_family_margin=0.1245
- soap (object, closure): closure=0.6404, wrong_family_margin=0.1382
- volcano (nature, closure): closure=0.6396, wrong_family_margin=0.1516
- solstice (celestial, closure): closure=0.6366, wrong_family_margin=0.1382
- sensor (tech, deep): closure=0.7057, wrong_family_margin=0.2514
- humidity (weather, deep): closure=0.6829, wrong_family_margin=0.1921
- open (action, deep): closure=0.6606, wrong_family_margin=0.1176
- soap (object, deep): closure=0.6566, wrong_family_margin=0.1395
- volcano (nature, deep): closure=0.6550, wrong_family_margin=0.1503
- glory (abstract, deep): closure=0.6545, wrong_family_margin=0.0838
- solstice (celestial, deep): closure=0.6525, wrong_family_margin=0.1503
- yak (animal, deep): closure=0.6499, wrong_family_margin=0.1176
