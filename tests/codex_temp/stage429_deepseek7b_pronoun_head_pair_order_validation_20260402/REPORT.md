# stage429_deepseek7b_pronoun_head_pair_order_validation

## Setup
- Timestamp UTC: 2026-04-02T05:57:41Z
- Use CUDA: True
- Batch size: 1
- Core heads: ['H:3:1', 'H:2:0', 'H:2:3']
- Route pair: ['H:2:0', 'H:2:3']
- Integrator head: H:3:1
- Booster candidates: ['H:2:8', 'H:3:27', 'H:2:10']
- Best booster: H:2:10

## Findings
- Best pair: ['H:2:0', 'H:2:3']
- Best pair full pronoun drop: 0.1706
- Integrator gain on route pair: 0.1254
- Best reverse gain on integrator: 0.0852
- Full order support margin: 0.0402
- Heldout order support margin: 0.0315
- Heldout best booster effect: H:2:10 -> 0.0322

## Reading
- If the layer-3 head gains more on top of the layer-2 route pair than the reverse direction gains on top of the integrator, the circuit looks more like route-first then integrate.
- If a booster head matters mainly after the core triple exists, it behaves more like an amplifier than a skeleton head.