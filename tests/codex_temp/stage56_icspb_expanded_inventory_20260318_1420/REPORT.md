# ICSPB 扩容清单报告

## 覆盖状态
- source_category_count: 10
- current_mass_category_count: 10
- inventory_category_count: 11
- terms_per_category: 20
- inventory_term_count: 220
- missing_in_current_mass: abstract
- added_by_inventory: abstract, action

## 推荐实验清单
- abstract: balance, beauty, chaos, curiosity, emotion, glory, history, honor
- action: build, change, drink, drive, find, help, jump, lead
- animal: camel, cow, crocodile, dolphin, donkey, fish, fox, giraffe
- celestial: asteroid, comet, dusk, exoplanet, firmament, lunarphase, magnetosphere, mars
- food: bean, bread, carrot, chocolate, dumpling, fishcake, friedrice, honey
- fruit: avocado, banana, blueberry, coconut, dragonfruit, durian, grape, grapefruit
- human: actor, architect, athlete, barber, cashier, coach, designer, engineer
- nature: branch, cliff, forest, hail, jungle, lake, ocean, pebble
- object: bag, bowl, camera, comb, cup, glasses, helmet, knife
- tech: algorithm, battery, chip, client, data, database, encoder, gradient
- vehicle: bulldozer, canoe, car, cart, destroyer, firetruck, helicopter, kayak

## 判断
- 当前真实 mass scan 缺失 abstract，且 action 完全未进入统一词表。
- 这份清单适合直接送入下一轮大规模扫描或 inventory-only 流程，用于补齐 abstract 与 action 的实证空洞。
- action 词目前属于扩展项，现有 mass_noun 扫描流程仍默认按 noun 字段命名，后续最好把输入接口泛化为 term 字段。
