# Stage640 四模型差异方向注入恢复报告

- 时间: 2026-04-05 23:34:43
- 脚本: `tests/codex/stage640_direction_injection_recovery.py`
- 能力:
  - `syntax`
  - `relation`
  - `coref`
- 每模型样本数: `6`
- 方法:
  1. 先找最伤的组件消融点
  2. 再在同一层注入 clean 差异方向
  3. 比较恢复前后 `margin`

## 总结论

这轮实验第一次给出了一个比“零化会坏”更强的证据：

- 如果注入 clean 条件下的差异方向，部分模型、部分任务的能力确实可以被恢复。

这说明：

- 这些差异方向不只是观测指标
- 至少在部分场景下，它们开始表现出“接近机制变量”的性质

## 分模型结果

### qwen3

- `coref`
  - `recovery_gain = 0.9531`
  - `recovery_ratio = 0.9425`
- `syntax`
  - `recovery_gain = 1.1401`
  - `recovery_ratio = 0.1980`
- `relation`
  - 基本无恢复

结论：

- `qwen3` 的指代和部分语法能力，已经可以被差异方向显著救回。

### deepseek7b

- `coref`
  - `recovery_gain = 0.2731`
  - `recovery_ratio = 0.1254`
- `relation`
  - `recovery_gain = 0.2411`
  - `recovery_ratio = 0.0509`
- `syntax`
  - `recovery_gain = 0.4263`
  - `recovery_ratio = 0.0350`

结论：

- `deepseek7b` 有恢复，但整体偏弱。
- 更像“方向是对的，但注入位置和尺度还不够精确”。

### gemma4

- `coref`
  - `recovery_gain = 3.3906`
  - `recovery_ratio = 2.2879`
- `relation`
  - `recovery_gain = 0.9615`
  - `recovery_ratio = 3.3729`
- `syntax`
  - `recovery_gain = 1.3262`
  - `recovery_ratio = 2.5916`

结论：

- `gemma4` 的恢复最惊人。
- 这非常关键，因为它说明：
  - `gemma4` 并不一定缺少这些能力方向
  - 更可能是“自然传播 / 自然读出”做得差
  - 一旦人工注入正确方向，能力可以明显回升

### glm4

- `syntax`
  - `recovery_gain = 1.2124`
  - `recovery_ratio = 0.2760`
- `relation`
  - 恢复非常弱
- `coref`
  - 恢复非常弱

结论：

- `glm4` 的优势更多像“本来就已经自然工作得很好”
- 因而被破坏后，简单方向注入并不能轻易补回

## 第二阶段最关键的新理论结论

1. 共享变量开始具备“恢复能力”

- 这比相关性更强
- 也比单纯消融更强

2. 不同模型的“可恢复性”差异很大

- `gemma4`：高可恢复
- `qwen3`：中等偏高，尤其 coref
- `deepseek7b`：弱恢复
- `glm4`：多数任务恢复弱

3. 这暗示不同模型可能分成两类

- 一类是“方向存在，但自然传播差”，所以人工注入很有效
  - `gemma4`
- 一类是“自然传播已经很强”，所以简单注入增益有限
  - `glm4`

## 当前最严格的表述

- 现在可以更进一步地说：
  - **差异方向在部分模型和任务上，已经显示出接近机制变量的恢复能力。**
- 但还不能说：
  - 单一差异方向已经足以完整恢复全部能力。

更严格的说法应当是：

- 当前恢复的是“部分判别能力”
- 而不是“完整语言处理机制”

## 下一步建议

1. 做更精细的注入位置
   - 现在只在层输出边界注入
   - 下一步应比较：
     - 层输出
     - pre-MLP
     - post-MLP
     - attention 输出

2. 做更精细的方向构造
   - 当前方向是 `clean A - clean B`
   - 下一步可以比较：
     - 层内主方向
     - 末层读出方向
     - 中层方向
     - capability-specific 局部基方向

3. 做恢复上限分析
   - 看“最多能救回多少”
   - 这样才能判断差异方向到底是“主机制”还是“次级辅助变量”
