# ICSPB 扩容清单报告

## 覆盖状态
- source_category_count: 10
- current_mass_category_count: 10
- inventory_category_count: 12
- terms_per_category: 24
- inventory_term_count: 288
- missing_in_current_mass: abstract
- added_by_inventory: abstract, action
- capped_categories: 

## 推荐实验清单
- abstract: balance, beauty, chaos, curiosity, emotion, glory, harmony, history
- action: catch, climb, compare, create, destroy, drive, drop, explore
- animal: bee, bird, camel, cow, deer, dog, donkey, elephant
- celestial: asteroid, cluster, constellation, cosmos, daybreak, earth, eclipse, equinoxline
- food: bean, cabbage, chocolate, coffee, egg, fishcake, garlic, honey
- fruit: avocado, blueberry, breadfruit, durian, grapefruit, guava, jackfruit, kiwi
- human: barber, captain, cashier, chef, dancer, driver, electrician, engineer
- nature: branch, cave, desert, flower, flowerbed, glacier, hail, leaf
- object: bag, bed, bottle, bowl, box, bucket, camera, chair
- tech: algorithm, battery, browser, circuit, client, cluster, compiler, data
- vehicle: airship, camper, canoe, car, cart, cruiser, drone, firetruck
- weather: blizzard, breeze, cloud, coldfront, cyclone, dew, downpour, drizzle

## 判断
- 当前真实 mass scan 缺少 abstract，action 与 weather 仍主要依赖扩展词表补齐。
- 这份清单适合直接进入下一轮大规模扫描，用于扩大 abstract、action、weather 的真实覆盖。
- 当前输出已经统一为 term 口径，但下游仍有部分脚本保留 noun 命名兼容层，后续需要继续去名词偏置。
