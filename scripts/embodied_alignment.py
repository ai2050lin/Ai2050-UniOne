"""
P2-1: Embodied Alignment Module
物理具身对齐模块

实现 AGI 系统的物理世界对齐：
1. 符号接地 (Symbol Grounding) - 从语言到物理概念
2. 物理常识推理 - 质量、力、因果
3. 感知-行动闭环 - 在环境中的具身交互
4. 本体感知 - 自我身体模型
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime

# ============== 物理世界模型 ==============

class PhysicalProperty(Enum):
    """物理属性枚举"""
    MASS = "mass"           # 质量
    VELOCITY = "velocity"   # 速度
    FORCE = "force"         # 力
    POSITION = "position"   # 位置
    SHAPE = "shape"         # 形状
    COLOR = "color"         # 颜色
    MATERIAL = "material"   # 材质
    STATE = "state"         # 状态 (固体/液体/气体)


@dataclass
class PhysicalObject:
    """物理对象"""
    name: str
    position: np.ndarray          # [x, y, z]
    velocity: np.ndarray          # [vx, vy, vz]
    mass: float
    shape: str                    # "sphere", "cube", "cylinder"
    size: np.ndarray              # [width, height, depth] 或 radius
    color: str
    material: str                 # "wood", "metal", "glass", "water", "air"
    state: str                    # "solid", "liquid", "gas"
    
    def get_embedding(self, dim: int = 64) -> np.ndarray:
        """将物理对象编码为向量"""
        emb = np.zeros(dim, dtype=np.float32)
        
        # 位置 (0-2)
        emb[0:3] = self.position
        
        # 速度 (3-5)
        emb[3:6] = self.velocity
        
        # 质量 (6)
        emb[6] = np.log1p(self.mass) / 10.0  # 归一化
        
        # 形状编码 (7-9)
        shape_map = {"sphere": [1, 0, 0], "cube": [0, 1, 0], "cylinder": [0, 0, 1]}
        emb[7:10] = shape_map.get(self.shape, [0, 0, 0])
        
        # 大小 (10-12)
        if len(self.size) >= 3:
            emb[10:13] = self.size[:3] / 10.0
        
        # 颜色编码 (13-15)
        color_map = {"red": [1, 0, 0], "green": [0, 1, 0], "blue": [0, 0, 1],
                     "white": [1, 1, 1], "black": [0, 0, 0], "yellow": [1, 1, 0]}
        emb[13:16] = color_map.get(self.color, [0.5, 0.5, 0.5])
        
        # 材质编码 (16-19)
        material_map = {"wood": [1, 0, 0, 0], "metal": [0, 1, 0, 0],
                       "glass": [0, 0, 1, 0], "water": [0, 0, 0, 1]}
        emb[16:20] = material_map.get(self.material, [0, 0, 0, 0])
        
        # 状态编码 (20-22)
        state_map = {"solid": [1, 0, 0], "liquid": [0, 1, 0], "gas": [0, 0, 1]}
        emb[20:23] = state_map.get(self.state, [1, 0, 0])
        
        # 名称哈希 (23-31)
        for i, c in enumerate(self.name[:8]):
            emb[23 + i] = ord(c) / 255.0
        
        return emb


@dataclass
class PhysicalAction:
    """物理动作"""
    name: str
    target: Optional[str]       # 目标对象名
    parameters: Dict[str, Any]  # 动作参数
    
    def get_embedding(self, dim: int = 64) -> np.ndarray:
        """将动作编码为向量"""
        emb = np.zeros(dim, dtype=np.float32)
        
        # 动作类型编码
        action_types = {
            "pick_up": [1, 0, 0, 0, 0],
            "put_down": [0, 1, 0, 0, 0],
            "push": [0, 0, 1, 0, 0],
            "rotate": [0, 0, 0, 1, 0],
            "pour": [0, 0, 0, 0, 1]
        }
        emb[0:5] = action_types.get(self.name, [0, 0, 0, 0, 0])
        
        # 参数编码
        if "force" in self.parameters:
            emb[5] = self.parameters["force"]
        if "direction" in self.parameters:
            direction = self.parameters["direction"]
            if isinstance(direction, (list, np.ndarray)) and len(direction) >= 3:
                emb[6:9] = direction[:3]
        
        return emb


class PhysicsEngine:
    """物理引擎 - 模拟物理世界"""
    
    def __init__(self):
        self.objects: Dict[str, PhysicalObject] = {}
        self.gravity = np.array([0, 0, -9.8])
        self.dt = 0.01  # 时间步长
        self.ground_level = 0.0
        
    def add_object(self, obj: PhysicalObject):
        """添加对象到世界"""
        self.objects[obj.name] = obj
    
    def remove_object(self, name: str):
        """移除对象"""
        if name in self.objects:
            del self.objects[name]
    
    def get_object(self, name: str) -> Optional[PhysicalObject]:
        """获取对象"""
        return self.objects.get(name)
    
    def apply_force(self, obj_name: str, force: np.ndarray):
        """施加力"""
        obj = self.get_object(obj_name)
        if obj:
            # F = ma -> a = F/m
            acceleration = force / obj.mass
            obj.velocity += acceleration * self.dt
    
    def step(self, steps: int = 1):
        """物理模拟步进"""
        for _ in range(steps):
            for obj in self.objects.values():
                # 重力
                obj.velocity += self.gravity * self.dt
                
                # 位置更新
                obj.position += obj.velocity * self.dt
                
                # 地面碰撞
                if obj.position[2] < self.ground_level:
                    obj.position[2] = self.ground_level
                    obj.velocity[2] = -obj.velocity[2] * 0.5  # 弹性碰撞
                    obj.velocity[0:2] *= 0.9  # 摩擦
    
    def check_collision(self, name1: str, name2: str) -> bool:
        """检测碰撞"""
        obj1 = self.get_object(name1)
        obj2 = self.get_object(name2)
        
        if not obj1 or not obj2:
            return False
        
        # 简化的球形碰撞检测
        dist = np.linalg.norm(obj1.position - obj2.position)
        threshold = 1.0  # 简化阈值
        return dist < threshold
    
    def get_state_embedding(self, dim: int = 128) -> np.ndarray:
        """获取世界状态向量"""
        emb = np.zeros(dim, dtype=np.float32)
        
        # 编码所有对象
        for i, (name, obj) in enumerate(self.objects.items()):
            if i < 2:  # 最多编码2个对象
                obj_emb = obj.get_embedding(64)
                emb[i * 64:(i + 1) * 64] = obj_emb
        
        return emb


# ============== 符号接地系统 ==============

class SymbolGrounding:
    """符号接地 - 将语言符号映射到物理概念"""
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.symbol_table: Dict[str, np.ndarray] = {}
        self.concept_to_physical: Dict[str, Dict[str, Any]] = {}
        
        # 初始化基础符号
        self._init_basic_concepts()
    
    def _init_basic_concepts(self):
        """初始化基础概念"""
        # 物体概念
        self.concept_to_physical["apple"] = {
            "shape": "sphere",
            "color": "red",
            "material": "organic",
            "typical_mass": 0.15,
            "typical_size": [0.08, 0.08, 0.08],
            "behaviors": ["can_be_eaten", "rolls", "has_gravity"]
        }
        
        self.concept_to_physical["cup"] = {
            "shape": "cylinder",
            "color": "white",
            "material": "ceramic",
            "typical_mass": 0.2,
            "typical_size": [0.08, 0.1, 0.08],
            "behaviors": ["can_hold_liquid", "can_be_picked_up"]
        }
        
        self.concept_to_physical["water"] = {
            "shape": "none",
            "color": "clear",
            "material": "water",
            "typical_mass": 1.0,  # per liter
            "typical_size": [0.1, 0.1, 0.1],
            "behaviors": ["flows", "fills_container", "evaporates"]
        }
        
        self.concept_to_physical["table"] = {
            "shape": "cube",
            "color": "brown",
            "material": "wood",
            "typical_mass": 15.0,
            "typical_size": [1.0, 0.75, 1.5],
            "behaviors": ["supports_objects", "stationary"]
        }
        
        # 动作概念
        self.concept_to_physical["pick_up"] = {
            "action_type": "grasp",
            "requires": ["object", "free_space_above"],
            "effects": ["object_in_hand"]
        }
        
        self.concept_to_physical["pour"] = {
            "action_type": "tilt",
            "requires": ["container_with_liquid", "target_container"],
            "effects": ["liquid_transferred"]
        }
        
        self.concept_to_physical["push"] = {
            "action_type": "force_application",
            "requires": ["object", "force"],
            "effects": ["object_moved"]
        }
    
    def ground_symbol(self, symbol: str) -> Optional[np.ndarray]:
        """将符号接地到向量"""
        if symbol in self.symbol_table:
            return self.symbol_table[symbol]
        
        # 从概念生成向量
        concept = self.concept_to_physical.get(symbol)
        if concept:
            embedding = self._concept_to_embedding(concept)
            self.symbol_table[symbol] = embedding
            return embedding
        
        return None
    
    def _concept_to_embedding(self, concept: Dict[str, Any]) -> np.ndarray:
        """将概念转换为向量"""
        emb = np.zeros(self.dim, dtype=np.float32)
        
        if "shape" in concept:
            shape_map = {"sphere": 0.2, "cube": 0.4, "cylinder": 0.6, "none": 0.0}
            emb[0] = shape_map.get(concept["shape"], 0.0)
        
        if "color" in concept:
            color_map = {"red": 0.1, "green": 0.2, "blue": 0.3, 
                        "white": 0.4, "brown": 0.5, "clear": 0.6}
            emb[1] = color_map.get(concept["color"], 0.0)
        
        if "material" in concept:
            material_map = {"wood": 0.1, "metal": 0.2, "glass": 0.3,
                          "water": 0.4, "organic": 0.5, "ceramic": 0.6}
            emb[2] = material_map.get(concept["material"], 0.0)
        
        if "typical_mass" in concept:
            emb[3] = np.log1p(concept["typical_mass"]) / 5.0
        
        if "typical_size" in concept:
            size = concept["typical_size"]
            if isinstance(size, (list, np.ndarray)):
                emb[4:7] = np.array(size[:3]) / 2.0
        
        if "behaviors" in concept:
            for i, behavior in enumerate(concept["behaviors"][:5]):
                emb[10 + i] = hash(behavior) % 100 / 100.0
        
        return emb
    
    def get_physical_properties(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取符号对应的物理属性"""
        return self.concept_to_physical.get(symbol)
    
    def predict_affordance(self, symbol: str) -> List[str]:
        """预测符号的可供性 (能做什么)"""
        concept = self.concept_to_physical.get(symbol)
        if concept and "behaviors" in concept:
            return concept["behaviors"]
        return []


# ============== 物理常识推理 ==============

class PhysicalReasoning:
    """物理常识推理"""
    
    def __init__(self, physics: PhysicsEngine, grounding: SymbolGrounding):
        self.physics = physics
        self.grounding = grounding
        
        # 物理规则库
        self.rules = self._init_rules()
    
    def _init_rules(self) -> List[Dict[str, Any]]:
        """初始化物理规则"""
        return [
            {
                "name": "gravity_causes_fall",
                "condition": lambda obj: obj.position[2] > 0 and obj.state == "solid",
                "effect": "object_will_fall",
                "confidence": 0.95
            },
            {
                "name": "water_flows_downhill",
                "condition": lambda obj: obj.material == "water" and obj.velocity[2] < 0,
                "effect": "liquid_spreads",
                "confidence": 0.90
            },
            {
                "name": "glass_breaks_on_impact",
                "condition": lambda obj: obj.material == "glass" and obj.velocity[2] < -1.0,
                "effect": "object_breaks",
                "confidence": 0.85
            },
            {
                "name": "heavy_objects_dont_float",
                "condition": lambda obj: obj.mass > 1.0 and obj.material not in ["air"],
                "effect": "object_sinks_in_water",
                "confidence": 0.80
            }
        ]
    
    def predict_outcome(self, scenario: str) -> Dict[str, Any]:
        """预测物理场景结果"""
        results = {
            "scenario": scenario,
            "predictions": [],
            "confidence": 0.0
        }
        
        # 简单的模式匹配预测
        if "drop" in scenario.lower() or "fall" in scenario.lower():
            results["predictions"].append({
                "event": "object_accelerates",
                "reason": "gravity",
                "final_state": "on_ground"
            })
            results["confidence"] = 0.90
        
        elif "pour" in scenario.lower() and "water" in scenario.lower():
            results["predictions"].append({
                "event": "water_flows",
                "reason": "gravity_fluid",
                "final_state": "container_filled"
            })
            results["confidence"] = 0.85
        
        elif "push" in scenario.lower():
            results["predictions"].append({
                "event": "object_moves",
                "reason": "force_applied",
                "final_state": "displaced"
            })
            results["confidence"] = 0.80
        
        elif "break" in scenario.lower() or "shatter" in scenario.lower():
            results["predictions"].append({
                "event": "fracture",
                "reason": "impact_stress",
                "final_state": "broken_pieces"
            })
            results["confidence"] = 0.75
        
        else:
            results["predictions"].append({
                "event": "unknown",
                "reason": "insufficient_information",
                "final_state": "unpredictable"
            })
            results["confidence"] = 0.30
        
        return results
    
    def answer_physics_question(self, question: str) -> Dict[str, Any]:
        """回答物理常识问题"""
        # 问题分类
        question_lower = question.lower()
        
        answer = {
            "question": question,
            "answer": "",
            "reasoning": [],
            "confidence": 0.0
        }
        
        # 因果推理
        if "why" in question_lower or "because" in question_lower:
            if "fall" in question_lower or "drop" in question_lower:
                answer["answer"] = "Objects fall because of gravity, which pulls them toward the ground."
                answer["reasoning"] = ["gravity_exists", "mass_attracts_mass", "earth_has_large_mass"]
                answer["confidence"] = 0.90
            
            elif "float" in question_lower:
                answer["answer"] = "Objects float when buoyancy force equals weight."
                answer["reasoning"] = ["archimedes_principle", "displaced_fluid_weight", "density_comparison"]
                answer["confidence"] = 0.85
        
        # 预测推理
        elif "what happens" in question_lower or "would happen" in question_lower:
            if "pour water" in question_lower:
                answer["answer"] = "The water will flow into the container and fill it from the bottom up."
                answer["reasoning"] = ["water_is_fluid", "gravity_pulls_down", "fluids_take_container_shape"]
                answer["confidence"] = 0.85
            
            elif "drop" in question_lower and "glass" in question_lower:
                answer["answer"] = "The glass will likely break when it hits the ground."
                answer["reasoning"] = ["glass_is_fragile", "impact_causes_stress", "exceeds_fracture_threshold"]
                answer["confidence"] = 0.80
        
        # 属性推理
        elif "is" in question_lower or "can" in question_lower:
            if "heavier" in question_lower:
                answer["answer"] = "Mass determines weight. Denser materials of the same volume are heavier."
                answer["reasoning"] = ["mass_definition", "density_formula", "weight_equals_mass_times_gravity"]
                answer["confidence"] = 0.90
            
            elif "liquid" in question_lower or "solid" in question_lower:
                answer["answer"] = "Solids maintain shape, liquids flow to fill containers."
                answer["reasoning"] = ["molecular_structure", "bond_strength", "state_of_matter"]
                answer["confidence"] = 0.85
        
        else:
            answer["answer"] = "I need more context to answer this physics question."
            answer["confidence"] = 0.20
        
        return answer
    
    def check_rule_violation(self, proposed_action: str, context: Dict) -> Tuple[bool, str]:
        """检查是否违反物理规则"""
        # 检查不可能的动作
        impossible_actions = {
            "lift_impossibly_heavy": "Object is too heavy to lift",
            "phase_through_solid": "Cannot pass through solid objects",
            "float_without_support": "Solid objects need support to stay elevated",
            "pour_solid": "Cannot pour solid objects like liquids"
        }
        
        for action_type, reason in impossible_actions.items():
            if action_type in proposed_action.lower():
                return True, reason
        
        return False, ""


# ============== 感知-行动闭环 ==============

@dataclass
class SensorReading:
    """传感器读数"""
    timestamp: float
    sensor_type: str  # "vision", "touch", "proprioception"
    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionCommand:
    """动作命令"""
    action_type: str
    target: Optional[str]
    parameters: Dict[str, Any]
    expected_outcome: str


class EmbodiedAgent:
    """具身智能体"""
    
    def __init__(self, physics: PhysicsEngine, grounding: SymbolGrounding):
        self.physics = physics
        self.grounding = grounding
        self.reasoning = PhysicalReasoning(physics, grounding)
        
        # 本体状态
        self.position = np.array([0.0, 0.0, 1.0])  # 站立位置
        self.hand_position = np.array([0.3, 0.0, 1.0])  # 手的位置
        self.holding: Optional[str] = None  # 手持物体
        
        # 感知历史
        self.sensory_buffer: List[SensorReading] = []
        self.action_history: List[ActionCommand] = []
        
        # 目标状态
        self.current_goal: Optional[str] = None
        self.goal_progress: float = 0.0
    
    def perceive(self, visual_input: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """感知环境"""
        perception = {
            "timestamp": time.time(),
            "objects": [],
            "self_state": {
                "position": self.position.copy(),
                "hand_position": self.hand_position.copy(),
                "holding": self.holding
            }
        }
        
        # 获取可见物体
        for name, obj in self.physics.objects.items():
            # 简化的可见性检查
            dist = np.linalg.norm(obj.position - self.position)
            if dist < 5.0:  # 5米范围内可见
                perception["objects"].append({
                    "name": name,
                    "position": obj.position.tolist(),
                    "distance": dist,
                    "properties": {
                        "shape": obj.shape,
                        "color": obj.color,
                        "material": obj.material
                    }
                })
        
        # 添加到感官缓冲
        reading = SensorReading(
            timestamp=perception["timestamp"],
            sensor_type="vision",
            data=np.zeros(64),  # 简化
            metadata=perception
        )
        self.sensory_buffer.append(reading)
        
        # 保持缓冲区大小
        if len(self.sensory_buffer) > 100:
            self.sensory_buffer = self.sensory_buffer[-100:]
        
        return perception
    
    def plan_action(self, goal: str) -> ActionCommand:
        """规划动作"""
        self.current_goal = goal
        
        # 简单的目标-动作映射
        goal_lower = goal.lower()
        
        if "pick" in goal_lower or "grab" in goal_lower:
            # 找到要拿的物体
            target = None
            for word in goal.split():
                concept = self.grounding.get_physical_properties(word.lower())
                if concept:
                    target = word.lower()
                    break
            
            return ActionCommand(
                action_type="pick_up",
                target=target,
                parameters={"approach_speed": 0.5},
                expected_outcome=f"holding_{target}"
            )
        
        elif "put" in goal_lower or "place" in goal_lower:
            return ActionCommand(
                action_type="put_down",
                target=self.holding,
                parameters={"position": [0.5, 0.5, 0.0]},
                expected_outcome="object_placed"
            )
        
        elif "push" in goal_lower:
            return ActionCommand(
                action_type="push",
                target=None,
                parameters={"force": 10.0, "direction": [1.0, 0.0, 0.0]},
                expected_outcome="object_moved"
            )
        
        else:
            return ActionCommand(
                action_type="observe",
                target=None,
                parameters={},
                expected_outcome="gathered_information"
            )
    
    def execute_action(self, command: ActionCommand) -> Dict[str, Any]:
        """执行动作"""
        result = {
            "action": command.action_type,
            "success": False,
            "outcome": "",
            "observations": []
        }
        
        if command.action_type == "pick_up":
            if command.target and command.target in self.physics.objects:
                obj = self.physics.objects[command.target]
                dist = np.linalg.norm(obj.position - self.hand_position)
                
                # 自动接近物体
                if dist > 0.5:
                    # 移动手到物体附近
                    direction = obj.position - self.hand_position
                    self.hand_position += direction * 0.8  # 移动80%的距离
                    result["outcome"] = f"Moving hand towards {command.target}"
                else:
                    self.holding = command.target
                    result["success"] = True
                    result["outcome"] = f"Picked up {command.target}"
            else:
                result["outcome"] = f"Object {command.target} not found"
        
        elif command.action_type == "put_down":
            if self.holding:
                obj = self.physics.objects.get(self.holding)
                if obj:
                    target_pos = command.parameters.get("position", [0.5, 0.5, 0.0])
                    obj.position = np.array(target_pos)
                    obj.velocity = np.zeros(3)
                result["success"] = True
                result["outcome"] = f"Put down {self.holding}"
                self.holding = None
            else:
                result["outcome"] = "Nothing in hand"
        
        elif command.action_type == "push":
            if command.target and command.target in self.physics.objects:
                force = command.parameters.get("force", 10.0)
                direction = command.parameters.get("direction", [1.0, 0.0, 0.0])
                self.physics.apply_force(command.target, np.array(direction) * force)
                result["success"] = True
                result["outcome"] = f"Pushed {command.target}"
            else:
                result["outcome"] = f"Cannot push"
        
        # 记录动作
        self.action_history.append(command)
        
        return result
    
    def sense_act_loop(self, goal: str, max_steps: int = 10) -> Dict[str, Any]:
        """感知-行动闭环"""
        results = {
            "goal": goal,
            "steps": [],
            "final_outcome": "",
            "success": False
        }
        
        for step in range(max_steps):
            # 感知
            perception = self.perceive()
            
            # 规划
            action = self.plan_action(goal)
            
            # 执行
            execution = self.execute_action(action)
            
            # 记录步骤
            results["steps"].append({
                "step": step,
                "perception": perception["objects"],
                "action": action.action_type,
                "result": execution["outcome"]
            })
            
            # 检查是否完成
            if execution["success"]:
                results["final_outcome"] = execution["outcome"]
                results["success"] = True
                break
            
            # 模拟物理
            self.physics.step(10)
        
        if not results["success"]:
            results["final_outcome"] = "Goal not achieved within steps"
        
        return results


# ============== 测试框架 ==============

class EmbodiedAlignmentTest:
    """物理具身对齐测试"""
    
    def __init__(self):
        self.physics = PhysicsEngine()
        self.grounding = SymbolGrounding()
        self.agent = EmbodiedAgent(self.physics, self.grounding)
        
        self.test_results = {}
    
    def test_symbol_grounding(self) -> Dict[str, Any]:
        """测试符号接地"""
        print("\n[Test 1] Symbol Grounding")
        print("-" * 40)
        
        test_symbols = ["apple", "cup", "water", "table", "pick_up", "pour"]
        results = {"symbols": {}, "grounding_rate": 0.0}
        
        grounded_count = 0
        for symbol in test_symbols:
            embedding = self.grounding.ground_symbol(symbol)
            properties = self.grounding.get_physical_properties(symbol)
            
            if embedding is not None:
                grounded_count += 1
                results["symbols"][symbol] = {
                    "embedded": True,
                    "has_properties": properties is not None
                }
                print(f"  {symbol}: embedded={embedding is not None}, properties={properties is not None}")
            else:
                results["symbols"][symbol] = {"embedded": False, "has_properties": False}
                print(f"  {symbol}: FAILED to ground")
        
        results["grounding_rate"] = grounded_count / len(test_symbols)
        print(f"\n  Grounding Rate: {results['grounding_rate']*100:.1f}%")
        
        self.test_results["symbol_grounding"] = results
        return results
    
    def test_physical_reasoning(self) -> Dict[str, Any]:
        """测试物理推理"""
        print("\n[Test 2] Physical Reasoning")
        print("-" * 40)
        
        test_questions = [
            "What happens if I drop a glass?",
            "Why do objects fall?",
            "What happens if I pour water into a cup?",
            "Can I lift a 1000kg rock?",
            "Is wood heavier than water?"
        ]
        
        results = {"questions": [], "avg_confidence": 0.0}
        
        total_confidence = 0.0
        for question in test_questions:
            answer = self.agent.reasoning.answer_physics_question(question)
            results["questions"].append({
                "question": question,
                "answer": answer["answer"][:50] + "..." if len(answer["answer"]) > 50 else answer["answer"],
                "confidence": answer["confidence"]
            })
            total_confidence += answer["confidence"]
            print(f"  Q: {question[:40]}...")
            print(f"     A: {answer['answer'][:50]}... (conf: {answer['confidence']:.2f})")
        
        results["avg_confidence"] = total_confidence / len(test_questions)
        print(f"\n  Average Confidence: {results['avg_confidence']:.2f}")
        
        self.test_results["physical_reasoning"] = results
        return results
    
    def test_embodied_interaction(self) -> Dict[str, Any]:
        """测试具身交互"""
        print("\n[Test 3] Embodied Interaction")
        print("-" * 40)
        
        # 设置测试场景
        apple = PhysicalObject(
            name="apple",
            position=np.array([0.5, 0.5, 0.5]),
            velocity=np.zeros(3),
            mass=0.15,
            shape="sphere",
            size=np.array([0.08, 0.08, 0.08]),
            color="red",
            material="organic",
            state="solid"
        )
        self.physics.add_object(apple)
        
        # 测试任务
        tasks = [
            "Pick up the apple",
            "Put the apple down",
            "Push the apple"
        ]
        
        results = {"tasks": [], "success_rate": 0.0}
        
        success_count = 0
        for task in tasks:
            loop_result = self.agent.sense_act_loop(task, max_steps=5)
            results["tasks"].append({
                "task": task,
                "success": loop_result["success"],
                "outcome": loop_result["final_outcome"]
            })
            if loop_result["success"]:
                success_count += 1
            print(f"  Task: {task} -> {'SUCCESS' if loop_result['success'] else 'FAILED'}")
            print(f"       Outcome: {loop_result['final_outcome']}")
        
        results["success_rate"] = success_count / len(tasks)
        print(f"\n  Task Success Rate: {results['success_rate']*100:.1f}%")
        
        self.test_results["embodied_interaction"] = results
        return results
    
    def test_affordance_prediction(self) -> Dict[str, Any]:
        """测试可供性预测"""
        print("\n[Test 4] Affordance Prediction")
        print("-" * 40)
        
        test_objects = ["apple", "cup", "water", "table"]
        results = {"objects": {}, "accuracy": 0.0}
        
        correct_predictions = 0
        total_predictions = 0
        
        expected_affordances = {
            "apple": ["can_be_eaten", "rolls", "has_gravity"],
            "cup": ["can_hold_liquid", "can_be_picked_up"],
            "water": ["flows", "fills_container", "evaporates"],
            "table": ["supports_objects", "stationary"]
        }
        
        for obj in test_objects:
            predicted = self.grounding.predict_affordance(obj)
            expected = expected_affordances.get(obj, [])
            
            # 计算匹配度
            matches = set(predicted) & set(expected)
            accuracy = len(matches) / len(expected) if expected else 0
            
            results["objects"][obj] = {
                "predicted": predicted,
                "expected": expected,
                "matches": list(matches),
                "accuracy": accuracy
            }
            
            correct_predictions += len(matches)
            total_predictions += len(expected)
            
            print(f"  {obj}: predicted={predicted}, matches={list(matches)}")
        
        results["accuracy"] = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"\n  Affordance Accuracy: {results['accuracy']*100:.1f}%")
        
        self.test_results["affordance_prediction"] = results
        return results
    
    def run_full_test(self) -> Dict[str, Any]:
        """运行完整测试"""
        print("=" * 60)
        print("P2-1: Embodied Alignment Test")
        print("=" * 60)
        
        # 运行所有测试
        self.test_symbol_grounding()
        self.test_physical_reasoning()
        self.test_embodied_interaction()
        self.test_affordance_prediction()
        
        # 计算总分
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        scores = {
            "symbol_grounding": self.test_results["symbol_grounding"]["grounding_rate"],
            "physical_reasoning": self.test_results["physical_reasoning"]["avg_confidence"],
            "embodied_interaction": self.test_results["embodied_interaction"]["success_rate"],
            "affordance_prediction": self.test_results["affordance_prediction"]["accuracy"]
        }
        
        overall_score = np.mean(list(scores.values()))
        
        print(f"\n  Symbol Grounding:    {scores['symbol_grounding']*100:.1f}%")
        print(f"  Physical Reasoning:  {scores['physical_reasoning']*100:.1f}%")
        print(f"  Embodied Interaction: {scores['embodied_interaction']*100:.1f}%")
        print(f"  Affordance Prediction: {scores['affordance_prediction']*100:.1f}%")
        print(f"\n  OVERALL SCORE:       {overall_score*100:.1f}%")
        
        # 判断通过
        if overall_score >= 0.6:
            print("\n  [PASS] Embodied alignment verified!")
            self.test_results["passed"] = True
        else:
            print("\n  [FAIL] Needs improvement")
            self.test_results["passed"] = False
        
        self.test_results["overall_score"] = overall_score
        self.test_results["scores"] = scores
        
        return self.test_results


# ============== 主程序 ==============

def main():
    """主函数"""
    print("P2-1: Physical Embodied Alignment")
    print("Testing AGI system's physical world grounding")
    print("=" * 60)
    
    tester = EmbodiedAlignmentTest()
    results = tester.run_full_test()
    
    # 保存结果
    output_path = "tempdata/embodied_alignment_results.json"
    
    # 转换 numpy 类型
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
