# AGI 能力扩展方案：长期记忆/工具使用/目标管理

## 概述

本文档整合现有研究成果，提出补全 AGI 三大核心能力的完整方案。

---

## 一、长期记忆 (Long-Term Memory)

### 1.1 理论基础

**记忆沉积假说**：短期记忆是动态联络层 $\Gamma$ 的瞬时激活，长期记忆是底流形度量张量 $g$ 的持久形变。

```
短期记忆 (Gamma)     ──沉积──>     长期记忆 (Metric g)
   |                                    |
   |-- 联络系数 Γᵢʲᵏ                    |-- 度量张量 gᵢⱼ
   |-- 瞬时激活                         |-- 持久结构
   |-- 高频更新                         |-- 低频固化
```

### 1.2 现有实现

**文件**: `scripts/sediment_engine.py`

核心机制：
1. **捕获阶段**: 记录动态联络层的活跃脉冲
2. **显著性过滤**: 只有超过阈值的痕迹才被沉积
3. **固化阶段**: 更新底流形度量张量
4. **遗忘机制**: 短期记忆衰减

### 1.3 改进方案

```python
# 新增功能: 分层记忆系统

class HierarchicalMemorySystem:
    """
    分层记忆系统 (Hierarchical Memory System)
    
    结构:
    - Epiosodic Memory: 事件级记忆 (高维, 短期)
    - Semantic Memory: 语义级记忆 (中维, 中期)  
    - Procedural Memory: 程序级记忆 (低维, 永久)
    """
    
    def __init__(self):
        # Episode Buffer: 暂存当前事件
        self.episode_buffer = RingBuffer(size=1000)
        
        # Semantic Store: 压缩后的语义表示
        self.semantic_store = HolographicCompressor(dim=256)
        
        # Procedural Core: 固化的技能/知识
        self.procedural_core = ManifoldSedimentEngine(dim=128)
        
    def consolidate_episode(self, episode):
        """
        事件巩固流程:
        1. 事件进入 Episode Buffer
        2. 显著事件被压缩进入 Semantic Store
        3. 高频语义被固化进入 Procedural Core
        """
        self.episode_buffer.push(episode)
        
        # 计算事件显著性
        salience = self.compute_salience(episode)
        
        if salience > self.salience_threshold:
            # 压缩并存储
            compressed = self.semantic_store.compress(episode)
            self.update_semantic_index(compressed)
            
        # 检查是否需要固化
        if self.episode_buffer.is_full():
            self.trigger_consolidation()
            
    def recall(self, query, memory_type='auto'):
        """
        记忆检索:
        - auto: 自动选择最佳记忆层
        - episodic: 只检索事件记忆
        - semantic: 只检索语义记忆
        - procedural: 只检索程序记忆
        """
        if memory_type == 'auto':
            # 根据查询相似度选择记忆层
            memory_type = self.select_memory_layer(query)
            
        if memory_type == 'episodic':
            return self.episode_buffer.search(query)
        elif memory_type == 'semantic':
            return self.semantic_store.retrieve(query)
        else:
            return self.procedural_core.query_metric(query)
```

### 1.4 实现路线

| 阶段 | 任务 | 输出 |
|------|------|------|
| Phase 1 | 增强现有 SedimentEngine | 支持批量沉积、增量更新 |
| Phase 2 | 实现分层存储 | Episode/Semantic/Procedural 三层架构 |
| Phase 3 | 集成 RAG-Fiber | 检索增强生成纤维 |
| Phase 4 | 记忆巩固验证 | 长周期记忆保持测试 |

---

## 二、工具使用 (Tool Usage)

### 2.1 理论基础

**工具接地假说**：工具是纤维丛的"外部延伸"，通过平行移动将内部表示映射到外部操作空间。

```
内部意图 (Intention)  ──平行移动──>  工具操作 (Tool Operation)
       |                                    |
       |-- 语义纤维                         |-- API 调用
       |-- 目标向量                         |-- 参数注入
       |-- 约束条件                         |-- 结果回传
```

### 2.2 设计方案

```python
# 文件: scripts/tool_grounding_system.py

class ToolGroundingSystem:
    """
    工具接地系统 (Tool Grounding System)
    
    将内部意图映射到外部工具操作
    """
    
    def __init__(self, manifold_dim=64):
        self.manifold_dim = manifold_dim
        
        # 工具注册表
        self.tools = {
            'search': {
                'description': '搜索外部知识库',
                'input_schema': {'query': 'str', 'top_k': 'int'},
                'output_schema': {'results': 'list'},
                'embedding': None  # 工具在流形上的嵌入
            },
            'code_exec': {
                'description': '执行代码片段',
                'input_schema': {'code': 'str', 'language': 'str'},
                'output_schema': {'result': 'any', 'error': 'str'},
                'embedding': None
            },
            'file_ops': {
                'description': '文件读写操作',
                'input_schema': {'path': 'str', 'content': 'str', 'mode': 'str'},
                'output_schema': {'success': 'bool', 'data': 'str'},
                'embedding': None
            },
            'api_call': {
                'description': '调用外部 API',
                'input_schema': {'url': 'str', 'method': 'str', 'data': 'dict'},
                'output_schema': {'response': 'dict'},
                'embedding': None
            }
        }
        
        # 初始化工具嵌入
        self._initialize_tool_embeddings()
        
        # 意图到工具的映射网络
        self.intention_to_tool = nn.Sequential(
            nn.Linear(manifold_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.tools)),
            nn.Softmax(dim=-1)
        )
        
    def _initialize_tool_embeddings(self):
        """使用工具描述初始化嵌入向量"""
        for tool_name, tool_info in self.tools.items():
            # 基于描述生成嵌入
            desc_embedding = self._encode_description(tool_info['description'])
            tool_info['embedding'] = desc_embedding
            
    def select_tool(self, intention_vector):
        """
        根据意图向量选择最合适的工具
        
        Args:
            intention_vector: (batch, manifold_dim) 意图向量
            
        Returns:
            tool_name: 选中的工具名
            confidence: 选择置信度
        """
        # 计算与各工具嵌入的相似度
        tool_embeddings = torch.stack([
            torch.tensor(t['embedding']) for t in self.tools.values()
        ])
        
        similarities = F.cosine_similarity(
            intention_vector.unsqueeze(1), 
            tool_embeddings.unsqueeze(0),
            dim=-1
        )
        
        # 选择最相似的工具
        best_idx = similarities.argmax(dim=-1)
        confidence = similarities.max(dim=-1).values
        
        tool_names = list(self.tools.keys())
        return tool_names[best_idx], confidence
    
    def ground_and_execute(self, intention, context):
        """
        接地并执行工具
        
        流程:
        1. 意图向量 -> 工具选择
        2. 上下文 -> 参数提取
        3. 工具执行
        4. 结果 -> 流形映射
        """
        # 选择工具
        tool_name, confidence = self.select_tool(intention)
        
        if confidence < 0.3:
            return {'error': 'No suitable tool found', 'confidence': confidence}
        
        # 参数提取 (简化版：从上下文提取)
        tool_schema = self.tools[tool_name]['input_schema']
        params = self._extract_parameters(context, tool_schema)
        
        # 执行工具
        result = self._execute_tool(tool_name, params)
        
        # 将结果映射回流形
        result_embedding = self._map_result_to_manifold(result)
        
        return {
            'tool': tool_name,
            'params': params,
            'result': result,
            'result_embedding': result_embedding,
            'confidence': confidence
        }
    
    def _execute_tool(self, tool_name, params):
        """执行具体工具"""
        if tool_name == 'search':
            return self._search_external(params)
        elif tool_name == 'code_exec':
            return self._execute_code(params)
        elif tool_name == 'file_ops':
            return self._file_operation(params)
        elif tool_name == 'api_call':
            return self._call_api(params)
            
    def learn_tool_usage(self, trajectory):
        """
        从使用轨迹学习工具用法
        
        强化学习: 成功使用 -> 正反馈
        """
        # 记录成功/失败轨迹
        # 更新意图到工具的映射
        pass
```

### 2.3 工具使用测试用例

```python
def test_tool_grounding():
    """测试工具接地系统"""
    system = ToolGroundingSystem(manifold_dim=64)
    
    # 测试 1: 搜索意图
    search_intention = encode_text("我需要查找关于量子的信息")
    result = system.ground_and_execute(search_intention, context={})
    assert result['tool'] == 'search'
    
    # 测试 2: 代码执行意图
    code_intention = encode_text("执行这段 Python 代码计算斐波那契")
    result = system.ground_and_execute(code_intention, context={})
    assert result['tool'] == 'code_exec'
    
    # 测试 3: 文件操作意图
    file_intention = encode_text("读取配置文件内容")
    result = system.ground_and_execute(file_intention, context={})
    assert result['tool'] == 'file_ops'
```

### 2.4 实现路线

| 阶段 | 任务 | 输出 |
|------|------|------|
| Phase 1 | 设计工具嵌入机制 | 工具描述 -> 流形嵌入 |
| Phase 2 | 实现意图到工具映射 | 神经网络映射器 |
| Phase 3 | 集成真实 API | 搜索/代码执行/文件操作 |
| Phase 4 | 工具使用学习 | 强化学习优化 |

---

## 三、目标管理 (Goal Management)

### 3.1 理论基础

**意图场假说**：目标在语义流形上创建势能场，推理测地线自然向目标汇聚。

```
目标点 (Target)  ──创建势能场──>  意图场 (Intentionality Field)
       |                                    |
       |-- 势能最低点                        |-- 引导测地线
       |-- 吸引力源                         |-- 目标导向推理
```

### 3.2 现有实现

**文件**: `scripts/intent_engine.py`

核心机制：
1. **目标设定**: 在流形上标记低势能点
2. **障碍设定**: 在流形上标记高势能点
3. **势能梯度**: 计算目标吸引 + 障碍排斥
4. **轨迹生成**: 沿势能梯度下降

### 3.3 改进方案

```python
# 文件: scripts/goal_management_system.py

class GoalManagementSystem:
    """
    目标管理系统 (Goal Management System)
    
    整合意图场与长期规划
    """
    
    def __init__(self, manifold_dim=32):
        self.manifold_dim = manifold_dim
        
        # 目标层次结构
        self.goals = {
            'long_term': [],    # 长期目标 (年/月)
            'medium_term': [],  # 中期目标 (周/日)
            'short_term': []    # 短期目标 (小时/分钟)
        }
        
        # 意图场引擎
        self.intentionality = IntentionalityEngine(manifold_dim)
        
        # 目标状态追踪
        self.goal_states = {}  # {goal_id: {'status': 'active/completed/failed', 'progress': 0.0}}
        
    def set_goal(self, goal_id, goal_vector, time_horizon='medium_term', priority=1.0):
        """
        设定目标
        
        Args:
            goal_id: 目标唯一标识
            goal_vector: 目标在流形上的位置向量
            time_horizon: 时间范围 (long/medium/short)
            priority: 优先级 (0-1)
        """
        goal = {
            'id': goal_id,
            'vector': np.array(goal_vector),
            'time_horizon': time_horizon,
            'priority': priority,
            'created_at': time.time()
        }
        
        self.goals[time_horizon].append(goal)
        self.intentionality.add_target(goal_id, goal_vector)
        self.goal_states[goal_id] = {'status': 'active', 'progress': 0.0}
        
    def decompose_goal(self, goal_id):
        """
        目标分解: 将长期目标分解为子目标
        
        使用几何分解:
        1. 在流形上找到关键中间点
        2. 这些点作为子目标
        """
        goal = self._find_goal(goal_id)
        if not goal:
            return []
            
        # 使用测地线上的关键点作为子目标
        current_state = self.get_current_state()
        trace = self.intentionality.generate_intentional_trace(
            current_state, steps=100
        )
        
        # 提取关键里程碑点
        milestones = self._extract_milestones(trace, n=3)
        
        sub_goals = []
        for i, milestone in enumerate(milestones):
            sub_goal_id = f"{goal_id}_sub_{i}"
            self.set_goal(
                sub_goal_id, 
                milestone, 
                time_horizon='short_term',
                priority=goal['priority'] * (0.8 ** i)
            )
            sub_goals.append(sub_goal_id)
            
        return sub_goals
    
    def update_progress(self, goal_id, progress):
        """更新目标进度"""
        if goal_id in self.goal_states:
            self.goal_states[goal_id]['progress'] = progress
            
            if progress >= 1.0:
                self.goal_states[goal_id]['status'] = 'completed'
                self._cascade_completion(goal_id)
                
    def _cascade_completion(self, goal_id):
        """目标完成级联: 检查父目标是否可完成"""
        # 如果所有子目标完成，标记父目标为可完成
        pass
    
    def get_next_action(self, current_state):
        """
        获取下一个动作建议
        
        基于当前状态和活跃目标，计算最佳行动方向
        """
        # 获取当前最优先的活跃目标
        active_goals = [
            g for g in self.goals['short_term']
            if self.goal_states[g['id']]['status'] == 'active'
        ]
        
        if not active_goals:
            active_goals = self.goals['medium_term']
            
        if not active_goals:
            active_goals = self.goals['long_term']
            
        if not active_goals:
            return None, 0.0
            
        # 按优先级排序
        active_goals.sort(key=lambda g: g['priority'], reverse=True)
        top_goal = active_goals[0]
        
        # 计算动作方向
        grad = self.intentionality.calculate_potential_gradient(current_state)
        
        return {
            'goal_id': top_goal['id'],
            'direction': grad,
            'distance_to_goal': np.linalg.norm(top_goal['vector'] - current_state)
        }
    
    def run_goal_directed_planning(self, initial_state, goal_id, max_steps=100):
        """
        执行目标导向规划
        
        生成从初始状态到目标的完整轨迹
        """
        goal = self._find_goal(goal_id)
        if not goal:
            return None
            
        # 生成意图轨迹
        trace = self.intentionality.generate_intentional_trace(
            initial_state, steps=max_steps
        )
        
        # 评估轨迹质量
        final_dist = np.linalg.norm(trace[-1] - goal['vector'])
        
        return {
            'goal_id': goal_id,
            'trajectory': trace,
            'final_distance': final_dist,
            'success': final_dist < 0.5,
            'steps_taken': len(trace)
        }
```

### 3.4 目标管理测试用例

```python
def test_goal_management():
    """测试目标管理系统"""
    system = GoalManagementSystem(manifold_dim=32)
    
    # 测试 1: 设定长期目标
    long_term_goal = np.ones(32) * 10.0
    system.set_goal("BECOME_AGI", long_term_goal, time_horizon='long_term', priority=1.0)
    
    # 测试 2: 目标分解
    sub_goals = system.decompose_goal("BECOME_AGI")
    assert len(sub_goals) > 0
    
    # 测试 3: 获取下一步行动
    current_state = np.zeros(32)
    action = system.get_next_action(current_state)
    assert action['goal_id'] is not None
    
    # 测试 4: 目标导向规划
    result = system.run_goal_directed_planning(current_state, sub_goals[0])
    assert result['success'] or result['final_distance'] < 5.0
    
    # 测试 5: 更新进度
    system.update_progress(sub_goals[0], 1.0)
    assert system.goal_states[sub_goals[0]]['status'] == 'completed'
```

### 3.5 实现路线

| 阶段 | 任务 | 输出 |
|------|------|------|
| Phase 1 | 增强现有 IntentionalityEngine | 多目标竞争、动态优先级 |
| Phase 2 | 实现目标分解 | 长期目标 -> 子目标 |
| Phase 3 | 状态追踪系统 | 进度监控、完成检测 |
| Phase 4 | 与 GWT 集成 | 意识空间中的目标竞争 |

---

## 四、系统集成架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGI 能力扩展层                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │  长期记忆      │  │  工具使用      │  │  目标管理      │       │
│  │  Memory       │  │  Tool         │  │  Goal         │       │
│  │  Sediment     │  │  Grounding    │  │  Management   │       │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘       │
│          │                  │                  │               │
│          └──────────────────┼──────────────────┘               │
│                             │                                   │
│                    ┌────────▼────────┐                         │
│                    │   GWS Controller │                         │
│                    │  (意识空间)       │                         │
│                    └────────┬────────┘                         │
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────┐       │
│  │              FiberNet Core (流形基础设施)             │       │
│  │  - Metric Tensor g (度量张量)                        │       │
│  │  - Connection Gamma (联络层)                         │       │
│  │  - Curvature Omega (曲率监控)                        │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、实现优先级

| 优先级 | 能力 | 理由 | 预估工作量 |
|--------|------|------|-----------|
| **P0** | 长期记忆 | 已有基础，直接可增强 | 2-3 天 |
| **P1** | 目标管理 | 已有基础，需要扩展 | 2-3 天 |
| **P2** | 工具使用 | 全新功能，需要设计 | 3-5 天 |

---

## 六、验证标准

### 6.1 长期记忆验证

```python
def validate_long_term_memory():
    """
    验证标准:
    1. 记忆保持: 100 步后仍能检索 90%+ 内容
    2. 记忆巩固: 度量张量形变 > 3.0 单位
    3. 抗噪声: 随机扰动不被沉积
    """
    pass
```

### 6.2 工具使用验证

```python
def validate_tool_usage():
    """
    验证标准:
    1. 工具选择准确率: > 85%
    2. 参数提取正确率: > 80%
    3. 执行成功率: > 90%
    """
    pass
```

### 6.3 目标管理验证

```python
def validate_goal_management():
    """
    验证标准:
    1. 目标到达率: > 80%
    2. 目标分解质量: 子目标覆盖率 > 90%
    3. 优先级调度: 高优先级目标优先完成
    """
    pass
```

---

## 七、总结

本方案基于现有 `sediment_engine.py` 和 `intent_engine.py` 的研究成果，提出了三大核心能力的完整补全方案：

1. **长期记忆**: 通过流形沉积机制实现短期记忆向长期记忆的转化
2. **工具使用**: 通过几何接地将内部意图映射到外部工具操作
3. **目标管理**: 通过意图场实现目标导向的自主规划

三项能力通过 GWS Controller (全局工作空间) 统一协调，形成完整的 AGI 能力闭环。
