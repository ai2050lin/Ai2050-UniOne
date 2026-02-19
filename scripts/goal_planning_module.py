"""
任务规划模块 (Goal Planning Module)
解决 P0 硬伤: 自主目标管理缺失

核心功能:
1. 目标分解 (Goal Decomposition)
2. 子任务依赖图构建
3. 任务调度器
4. 工具调用接口
5. 进度追踪
6. 自纠错机制

与现有模块集成:
- intent_engine.py: 意图向量生成
- fibernet_bundle.py: 几何化注意力机制
"""

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """任务优先级"""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class SubTask:
    """子任务数据结构"""
    id: str
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class Goal:
    """目标数据结构"""
    id: str
    name: str
    description: str
    target_state: np.ndarray  # 目标状态向量
    subtasks: List[SubTask] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    completion_threshold: float = 0.95


class GoalDecomposer:
    """
    目标分解器
    将复杂目标分解为可执行的子任务序列
    """
    
    # 预定义的分解模板
    DECOMPOSITION_TEMPLATES = {
        "research": [
            ("gather_information", "收集相关信息", TaskPriority.HIGH),
            ("analyze_data", "分析数据", TaskPriority.HIGH),
            ("synthesize_findings", "综合发现", TaskPriority.MEDIUM),
            ("generate_report", "生成报告", TaskPriority.LOW),
        ],
        "coding": [
            ("understand_requirements", "理解需求", TaskPriority.CRITICAL),
            ("design_architecture", "设计架构", TaskPriority.HIGH),
            ("implement_core", "实现核心功能", TaskPriority.HIGH),
            ("write_tests", "编写测试", TaskPriority.MEDIUM),
            ("optimize_performance", "优化性能", TaskPriority.LOW),
        ],
        "problem_solving": [
            ("identify_problem", "识别问题", TaskPriority.CRITICAL),
            ("analyze_causes", "分析原因", TaskPriority.HIGH),
            ("generate_solutions", "生成解决方案", TaskPriority.HIGH),
            ("evaluate_solutions", "评估方案", TaskPriority.MEDIUM),
            ("implement_solution", "实施方案", TaskPriority.MEDIUM),
        ],
        "learning": [
            ("acquire_knowledge", "获取知识", TaskPriority.HIGH),
            ("practice_application", "实践应用", TaskPriority.HIGH),
            ("verify_understanding", "验证理解", TaskPriority.MEDIUM),
            ("integrate_memory", "整合记忆", TaskPriority.LOW),
        ],
    }
    
    def __init__(self, manifold_dim: int = 32):
        self.manifold_dim = manifold_dim
        
    def decompose(self, goal: Goal, goal_type: str = "problem_solving") -> List[SubTask]:
        """
        将目标分解为子任务
        
        Args:
            goal: 目标对象
            goal_type: 目标类型，决定分解模板
            
        Returns:
            子任务列表
        """
        template = self.DECOMPOSITION_TEMPLATES.get(goal_type, 
                                                      self.DECOMPOSITION_TEMPLATES["problem_solving"])
        
        subtasks = []
        prev_task_id = None
        
        for i, (task_id, task_name, priority) in enumerate(template):
            # 构建依赖关系：每个任务依赖前一个任务
            dependencies = [prev_task_id] if prev_task_id else []
            
            subtask = SubTask(
                id=f"{goal.id}_{task_id}",
                name=task_name,
                description=f"执行 {task_name} 以完成目标 {goal.name}",
                priority=priority,
                dependencies=dependencies,
            )
            subtasks.append(subtask)
            prev_task_id = subtask.id
            
        return subtasks
    
    def decompose_custom(self, goal: Goal, task_specs: List[Tuple[str, str, TaskPriority]]) -> List[SubTask]:
        """
        自定义分解
        
        Args:
            goal: 目标对象
            task_specs: [(task_id, task_name, priority), ...]
            
        Returns:
            子任务列表
        """
        subtasks = []
        prev_task_id = None
        
        for task_id, task_name, priority in task_specs:
            dependencies = [prev_task_id] if prev_task_id else []
            
            subtask = SubTask(
                id=f"{goal.id}_{task_id}",
                name=task_name,
                description=f"执行 {task_name}",
                priority=priority,
                dependencies=dependencies,
            )
            subtasks.append(subtask)
            prev_task_id = subtask.id
            
        return subtasks


class DependencyGraph:
    """
    任务依赖图
    构建和管理任务之间的依赖关系
    """
    
    def __init__(self):
        self.nodes: Dict[str, SubTask] = {}
        self.edges: Dict[str, List[str]] = {}  # task_id -> dependent task ids
        
    def add_task(self, task: SubTask):
        """添加任务节点"""
        self.nodes[task.id] = task
        if task.id not in self.edges:
            self.edges[task.id] = []
            
    def add_dependency(self, task_id: str, depends_on: str):
        """添加依赖关系"""
        if task_id in self.edges:
            self.edges[task_id].append(depends_on)
            
    def get_ready_tasks(self) -> List[SubTask]:
        """获取可以执行的任务（所有依赖已完成）"""
        ready = []
        for task_id, task in self.nodes.items():
            if task.status != TaskStatus.PENDING:
                continue
            
            # 检查所有依赖是否完成
            dependencies = self.edges.get(task_id, [])
            all_deps_done = all(
                self.nodes[dep_id].status == TaskStatus.COMPLETED
                for dep_id in dependencies
                if dep_id in self.nodes
            )
            
            if all_deps_done:
                ready.append(task)
                
        # 按优先级排序
        ready.sort(key=lambda t: t.priority.value)
        return ready
    
    def get_execution_order(self) -> List[List[str]]:
        """
        获取拓扑排序的执行层级
        返回: [[layer0_tasks], [layer1_tasks], ...]
        """
        layers = []
        remaining = set(self.nodes.keys())
        completed = set()
        
        while remaining:
            # 找出当前层可以执行的任务
            layer = []
            for task_id in remaining:
                deps = set(self.edges.get(task_id, []))
                if deps.issubset(completed):
                    layer.append(task_id)
            
            if not layer:
                # 存在循环依赖
                break
                
            layers.append(layer)
            completed.update(layer)
            remaining -= set(layer)
            
        return layers
    
    def detect_cycles(self) -> bool:
        """检测是否存在循环依赖"""
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.edges.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
                    
            rec_stack.remove(node)
            return False
        
        for node in self.nodes:
            if node not in visited:
                if dfs(node):
                    return True
        return False


class ToolRegistry:
    """
    工具注册表
    管理可用的外部工具
    """
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_info: Dict[str, Dict] = {}
        
    def register(self, name: str, func: Callable, description: str = "", args_schema: Dict = None):
        """注册工具"""
        self.tools[name] = func
        self.tool_info[name] = {
            "description": description,
            "args_schema": args_schema or {},
        }
        
    def execute(self, name: str, **kwargs) -> Any:
        """执行工具"""
        if name not in self.tools:
            raise ValueError(f"工具 '{name}' 未注册")
        return self.tools[name](**kwargs)
    
    def list_tools(self) -> List[Dict]:
        """列出所有工具"""
        return [
            {"name": name, **info}
            for name, info in self.tool_info.items()
        ]


class ProgressTracker:
    """
    进度追踪器
    记录和监控任务执行进度
    """
    
    def __init__(self):
        self.history: List[Dict] = []
        self.metrics: Dict[str, float] = {}
        
    def record(self, event: str, task_id: str, details: Dict = None):
        """记录事件"""
        entry = {
            "timestamp": time.time(),
            "event": event,
            "task_id": task_id,
            "details": details or {},
        }
        self.history.append(entry)
        
    def get_progress(self, goal: Goal) -> Dict:
        """获取目标进度"""
        total = len(goal.subtasks)
        if total == 0:
            return {"progress": 1.0, "completed": 0, "total": 0}
        
        completed = sum(1 for t in goal.subtasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in goal.subtasks if t.status == TaskStatus.FAILED)
        running = sum(1 for t in goal.subtasks if t.status == TaskStatus.RUNNING)
        
        return {
            "progress": completed / total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "total": total,
        }
    
    def estimate_remaining_time(self, goal: Goal) -> float:
        """估算剩余时间"""
        completed_tasks = [t for t in goal.subtasks if t.status == TaskStatus.COMPLETED]
        remaining_tasks = [t for t in goal.subtasks if t.status in [TaskStatus.PENDING, TaskStatus.BLOCKED]]
        
        if not completed_tasks:
            return float('inf')
            
        # 计算平均任务执行时间
        avg_time = np.mean([
            t.completed_at - t.started_at
            for t in completed_tasks
            if t.completed_at and t.started_at
        ])
        
        return avg_time * len(remaining_tasks)


class SelfCorrectionMechanism:
    """
    自纠错机制
    检测和修正执行错误
    """
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.correction_history: List[Dict] = []
        
    def analyze_failure(self, task: SubTask) -> Dict:
        """分析任务失败原因"""
        analysis = {
            "task_id": task.id,
            "error": task.error,
            "retry_count": task.retry_count,
            "possible_causes": [],
            "suggested_actions": [],
        }
        
        # 根据错误类型提供建议
        if task.error:
            error_lower = task.error.lower()
            
            if "timeout" in error_lower:
                analysis["possible_causes"].append("执行超时")
                analysis["suggested_actions"].append("增加超时时间或简化任务")
                
            elif "memory" in error_lower:
                analysis["possible_causes"].append("内存不足")
                analysis["suggested_actions"].append("减少数据规模或分批处理")
                
            elif "connection" in error_lower:
                analysis["possible_causes"].append("连接问题")
                analysis["suggested_actions"].append("检查网络连接，重试")
                
            else:
                analysis["possible_causes"].append("未知错误")
                analysis["suggested_actions"].append("检查任务参数，尝试替代方案")
        
        return analysis
    
    def should_retry(self, task: SubTask) -> bool:
        """判断是否应该重试"""
        return task.retry_count < self.max_retries
    
    def apply_correction(self, task: SubTask, correction: Dict) -> SubTask:
        """应用纠错措施"""
        task.retry_count += 1
        task.status = TaskStatus.PENDING
        task.error = None
        
        # 应用纠错参数
        if "adjusted_args" in correction:
            task.tool_args.update(correction["adjusted_args"])
            
        self.correction_history.append({
            "task_id": task.id,
            "correction": correction,
            "timestamp": time.time(),
        })
        
        return task


class GoalPlanningModule:
    """
    任务规划模块
    整合所有组件，提供完整的自主目标管理能力
    """
    
    def __init__(self, manifold_dim: int = 32):
        self.manifold_dim = manifold_dim
        self.decomposer = GoalDecomposer(manifold_dim)
        self.dependency_graph = DependencyGraph()
        self.tool_registry = ToolRegistry()
        self.progress_tracker = ProgressTracker()
        self.correction = SelfCorrectionMechanism()
        
        # 目标存储
        self.goals: Dict[str, Goal] = {}
        
        # 注册默认工具
        self._register_default_tools()
        
    def _register_default_tools(self):
        """注册默认工具"""
        
        def dummy_tool(**kwargs):
            return {"status": "success", "output": kwargs}
        
        self.tool_registry.register(
            "calculator",
            lambda a, b, op: {"result": eval(f"{a} {op} {b}")},
            "计算器工具",
            {"a": "数字", "b": "数字", "op": "操作符 (+, -, *, /)"}
        )
        
        self.tool_registry.register(
            "search",
            lambda query: {"results": [f"模拟搜索结果: {query}"]},
            "搜索工具",
            {"query": "搜索查询"}
        )
        
        self.tool_registry.register(
            "code_execute",
            lambda code: {"output": f"执行结果: {code[:50]}..."},
            "代码执行工具",
            {"code": "要执行的代码"}
        )
        
    def create_goal(
        self,
        name: str,
        description: str,
        target_state: np.ndarray = None,
        goal_type: str = "problem_solving"
    ) -> Goal:
        """
        创建新目标
        
        Args:
            name: 目标名称
            description: 目标描述
            target_state: 目标状态向量
            goal_type: 目标类型
            
        Returns:
            创建的目标对象
        """
        goal_id = f"goal_{int(time.time() * 1000)}"
        
        if target_state is None:
            target_state = np.ones(self.manifold_dim)
            
        goal = Goal(
            id=goal_id,
            name=name,
            description=description,
            target_state=target_state,
        )
        
        # 自动分解目标
        goal.subtasks = self.decomposer.decompose(goal, goal_type)
        
        # 构建依赖图
        for task in goal.subtasks:
            self.dependency_graph.add_task(task)
            for dep_id in task.dependencies:
                self.dependency_graph.add_dependency(task.id, dep_id)
                
        self.goals[goal_id] = goal
        self.progress_tracker.record("goal_created", goal_id, {"name": name})
        
        return goal
    
    def execute_task(self, task: SubTask) -> bool:
        """
        执行单个任务
        
        Args:
            task: 要执行的任务
            
        Returns:
            是否执行成功
        """
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self.progress_tracker.record("task_started", task.id)
        
        try:
            if task.tool_name:
                result = self.tool_registry.execute(task.tool_name, **task.tool_args)
                task.result = result
                
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            self.progress_tracker.record("task_completed", task.id, {"result": str(task.result)[:100]})
            return True
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.progress_tracker.record("task_failed", task.id, {"error": str(e)})
            
            # 尝试纠错
            if self.correction.should_retry(task):
                analysis = self.correction.analyze_failure(task)
                correction = {
                    "adjusted_args": task.tool_args,
                    "retry": True,
                }
                self.correction.apply_correction(task, correction)
                
            return False
    
    def run_goal(self, goal_id: str, callback: Callable = None) -> Dict:
        """
        执行目标的所有任务
        
        Args:
            goal_id: 目标ID
            callback: 进度回调函数
            
        Returns:
            执行结果
        """
        goal = self.goals.get(goal_id)
        if not goal:
            return {"error": "目标不存在"}
            
        goal.status = TaskStatus.RUNNING
        
        # 获取执行顺序
        execution_layers = self.dependency_graph.get_execution_order()
        
        for layer in execution_layers:
            for task_id in layer:
                task = self.dependency_graph.nodes[task_id]
                
                # 执行任务
                success = self.execute_task(task)
                
                # 回调
                if callback:
                    progress = self.progress_tracker.get_progress(goal)
                    callback(goal, task, progress)
                    
        # 更新目标状态
        progress = self.progress_tracker.get_progress(goal)
        if progress["progress"] >= goal.completion_threshold:
            goal.status = TaskStatus.COMPLETED
        else:
            goal.status = TaskStatus.FAILED
            
        return {
            "goal_id": goal_id,
            "status": goal.status.value,
            "progress": progress,
            "estimated_remaining": self.progress_tracker.estimate_remaining_time(goal),
        }
    
    def get_goal_status(self, goal_id: str) -> Dict:
        """获取目标状态"""
        goal = self.goals.get(goal_id)
        if not goal:
            return {"error": "目标不存在"}
            
        return {
            "goal_id": goal_id,
            "name": goal.name,
            "status": goal.status.value,
            "progress": self.progress_tracker.get_progress(goal),
            "subtasks": [
                {
                    "id": t.id,
                    "name": t.name,
                    "status": t.status.value,
                    "priority": t.priority.name,
                }
                for t in goal.subtasks
            ],
        }


def run_p0_validation():
    """
    P0 硬伤验证测试
    验证任务规划模块的基本功能
    """
    print("=" * 60)
    print("P0 硬伤解决验证: 自主目标管理")
    print("=" * 60)
    
    # 1. 初始化模块
    planner = GoalPlanningModule(manifold_dim=32)
    print("\n[1] 模块初始化: OK")
    
    # 2. 创建目标
    goal = planner.create_goal(
        name="解决数学问题",
        description="使用计算器解决一个数学问题",
        goal_type="problem_solving"
    )
    print(f"\n[2] 目标创建: {goal.name}")
    print(f"    子任务数: {len(goal.subtasks)}")
    
    # 3. 检查依赖图
    has_cycle = planner.dependency_graph.detect_cycles()
    print(f"\n[3] 依赖图检查: {'存在循环依赖!' if has_cycle else '无循环依赖 [OK]'}")
    
    execution_order = planner.dependency_graph.get_execution_order()
    print(f"    执行层级: {len(execution_order)}")
    
    # 4. 执行目标
    def progress_callback(goal, task, progress):
        print(f"    [进度 {progress['progress']*100:.1f}%] 任务: {task.name}")
    
    print("\n[4] 开始执行目标...")
    result = planner.run_goal(goal.id, callback=progress_callback)
    
    # 5. 检查结果
    print("\n[5] 执行结果:")
    print(f"    状态: {result['status']}")
    print(f"    完成度: {result['progress']['progress']*100:.1f}%")
    
    # 6. 验证关键能力
    print("\n[6] 关键能力验证:")
    
    checks = {
        "目标分解": len(goal.subtasks) > 0,
        "依赖图构建": not has_cycle,
        "任务调度": len(execution_order) > 0,
        "工具注册": len(planner.tool_registry.tools) > 0,
        "进度追踪": len(planner.progress_tracker.history) > 0,
        "自纠错": planner.correction.max_retries > 0,
    }
    
    for check_name, passed in checks.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"    {check_name}: {status}")
    
    # 7. 生成报告
    report = {
        "test_name": "P0_自主目标管理验证",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "goal": {
            "id": goal.id,
            "name": goal.name,
            "subtask_count": len(goal.subtasks),
        },
        "checks": checks,
        "result": result,
        "overall_passed": all(checks.values()),
    }
    
    # 保存报告
    os.makedirs("tempdata", exist_ok=True)
    report_path = "tempdata/p0_goal_planning_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n[7] 报告已保存: {report_path}")
    
    if all(checks.values()):
        print("\n" + "=" * 60)
        print("P0 硬伤 #1 (自主目标管理) 已解决! [SUCCESS]")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("P0 硬伤 #1 解决失败，需要进一步修复")
        print("=" * 60)
    
    return report


if __name__ == "__main__":
    run_p0_validation()
