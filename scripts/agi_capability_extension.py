"""
AGI Capability Extension Module
================================
Implements three core capabilities:
1. Long-Term Memory (Hierarchical Memory System)
2. Tool Usage (Geometric Tool Grounding)
3. Goal Management (Intentionality Field)

Based on existing:
- sediment_engine.py (Memory Sedimentation)
- intent_engine.py (Intentionality Engine)
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Part 1: Long-Term Memory System
# =============================================================================

class RingBuffer:
    """Circular buffer for episode storage"""
    def __init__(self, size=1000):
        self.size = size
        self.buffer = deque(maxlen=size)
        
    def push(self, item):
        self.buffer.append(item)
        
    def search(self, query, top_k=5):
        """Simple similarity search"""
        if len(self.buffer) == 0:
            return []
        
        similarities = []
        query_tensor = torch.tensor(query) if not isinstance(query, torch.Tensor) else query
        
        for i, item in enumerate(self.buffer):
            item_tensor = torch.tensor(item) if not isinstance(item, torch.Tensor) else item
            sim = F.cosine_similarity(
                query_tensor.flatten().unsqueeze(0),
                item_tensor.flatten().unsqueeze(0)
            ).item()
            similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [(self.buffer[i], sim) for i, sim in similarities[:top_k]]
    
    def is_full(self):
        return len(self.buffer) >= self.size
    
    def __len__(self):
        return len(self.buffer)


class HolographicCompressor:
    """Compresses high-dimensional data using holographic representation"""
    def __init__(self, dim=256):
        self.dim = dim
        # Random projection matrix for compression
        self.projection = None
        
    def _ensure_projection(self, input_dim):
        if self.projection is None or self.projection.shape[1] != input_dim:
            self.projection = torch.randn(self.dim, input_dim) / np.sqrt(input_dim)
            
    def compress(self, data):
        """Compress data to lower-dimensional representation"""
        data_tensor = torch.tensor(data, dtype=torch.float32).flatten() if not isinstance(data, torch.Tensor) else data.flatten().float()
        self._ensure_projection(len(data_tensor))
        return torch.matmul(self.projection, data_tensor).numpy()
    
    def retrieve(self, compressed, original_shape=None):
        """Attempt to reconstruct (lossy)"""
        # Note: This is lossy reconstruction
        if self.projection is None:
            return None
        # Pseudo-inverse reconstruction
        pseudo_inv = torch.pinverse(self.projection)
        reconstructed = torch.matmul(pseudo_inv, torch.tensor(compressed))
        if original_shape:
            return reconstructed.reshape(original_shape).numpy()
        return reconstructed.numpy()


class ManifoldSedimentEngine:
    """Enhanced sediment engine with batch processing"""
    def __init__(self, dim=128, sediment_rate=0.05):
        self.dim = dim
        self.sediment_rate = sediment_rate
        self.metric_g = np.eye(dim)
        self.memory_trace = np.zeros((dim, dim))
        
    def capture_pulse(self, gamma_dynamic):
        """Capture dynamic connection pulse"""
        self.memory_trace += np.abs(gamma_dynamic)
        
    def solidify(self, threshold_multiplier=1.2):
        """Execute memory consolidation"""
        threshold = np.mean(self.memory_trace) * threshold_multiplier
        significant_mask = self.memory_trace > threshold
        sediment_update = self.memory_trace * significant_mask * self.sediment_rate
        self.metric_g += sediment_update
        self.memory_trace *= 0.1  # Decay
        return np.linalg.norm(sediment_update)
    
    def query_metric(self, query):
        """Query the metric tensor for memory retrieval"""
        query_tensor = np.array(query).flatten()
        # Use metric to compute "memory distance"
        weighted_query = self.metric_g @ query_tensor
        return weighted_query


class HierarchicalMemorySystem:
    """
    Three-tier memory system:
    - Episodic: Short-term, high-fidelity events
    - Semantic: Medium-term, compressed concepts
    - Procedural: Long-term, consolidated skills
    """
    def __init__(self, dim=128, episode_size=1000):
        self.dim = dim
        
        # Three memory layers
        self.episodic = RingBuffer(size=episode_size)
        self.semantic = HolographicCompressor(dim=256)
        self.procedural = ManifoldSedimentEngine(dim=dim)
        
        # Salience threshold for consolidation
        self.salience_threshold = 0.5
        
        # Memory statistics
        self.stats = {
            'episodes_stored': 0,
            'semantic_consolidations': 0,
            'procedural_consolidations': 0
        }
        
    def compute_salience(self, episode):
        """Compute episode salience (importance)"""
        # Simple salience: norm of episode
        if isinstance(episode, np.ndarray):
            return np.linalg.norm(episode) / np.sqrt(len(episode.flatten()))
        return 0.5  # Default medium salience
    
    def store(self, episode, salience=None):
        """Store episode in memory hierarchy"""
        self.episodic.push(episode)
        self.stats['episodes_stored'] += 1
        
        if salience is None:
            salience = self.compute_salience(episode)
        
        # High-salience episodes go to semantic memory
        if salience > self.salience_threshold:
            compressed = self.semantic.compress(episode)
            self.stats['semantic_consolidations'] += 1
        
        # Trigger procedural consolidation if buffer full
        if self.episodic.is_full():
            self._trigger_consolidation()
            
    def _trigger_consolidation(self):
        """Consolidate frequent patterns to procedural memory"""
        # Sample from episodic memory
        for item in list(self.episodic.buffer)[-10:]:
            if isinstance(item, np.ndarray):
                # Convert to connection-like structure
                gamma = np.outer(item.flatten()[:self.dim], item.flatten()[:self.dim])
                self.procedural.capture_pulse(gamma * 0.1)
        
        self.procedural.solidify()
        self.stats['procedural_consolidations'] += 1
        
    def recall(self, query, memory_type='auto'):
        """Recall from memory hierarchy"""
        if memory_type == 'auto':
            # Check each layer
            ep_result = self.episodic.search(query, top_k=3)
            if ep_result and ep_result[0][1] > 0.7:
                return ('episodic', ep_result)
            return ('semantic', self.procedural.query_metric(query))
            
        elif memory_type == 'episodic':
            return ('episodic', self.episodic.search(query))
        elif memory_type == 'semantic':
            return ('semantic', self.semantic.compress(query))
        else:
            return ('procedural', self.procedural.query_metric(query))
    
    def get_memory_stats(self):
        return {
            'episodic_count': len(self.episodic),
            'metric_deformation': np.linalg.norm(self.procedural.metric_g - np.eye(self.dim)),
            **self.stats
        }


# =============================================================================
# Part 2: Tool Usage System
# =============================================================================

@dataclass
class ToolDefinition:
    """Definition of an external tool"""
    name: str
    description: str
    input_schema: Dict[str, str]
    output_schema: Dict[str, str]
    embedding: Optional[np.ndarray] = None
    usage_count: int = 0
    success_rate: float = 0.0


class ToolGroundingSystem(nn.Module):
    """
    Maps internal intentions to external tool operations.
    Uses geometric grounding: tools are embedded in the same manifold as intentions.
    """
    def __init__(self, manifold_dim=64):
        super().__init__()
        self.manifold_dim = manifold_dim
        
        # Tool registry
        self.tools: Dict[str, ToolDefinition] = {}
        
        # Intention -> Tool mapping network
        self.tool_selector = nn.Sequential(
            nn.Linear(manifold_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Max 32 tools
        )
        
        # Parameter extraction network
        self.param_extractor = nn.Sequential(
            nn.Linear(manifold_dim * 2, 128),  # intention + context
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Text encoder (simplified - in production use proper embeddings)
        self.text_encoder = nn.Sequential(
            nn.Linear(384, 128),  # Assume 384-dim text embedding
            nn.ReLU(),
            nn.Linear(128, manifold_dim)
        )
        
        # Initialize default tools
        self._register_default_tools()
        
    def _register_default_tools(self):
        """Register built-in tools"""
        default_tools = [
            ToolDefinition(
                name='search',
                description='Search external knowledge base for information',
                input_schema={'query': 'str', 'top_k': 'int'},
                output_schema={'results': 'list[dict]'}
            ),
            ToolDefinition(
                name='code_exec',
                description='Execute code snippet and return results',
                input_schema={'code': 'str', 'language': 'str'},
                output_schema={'result': 'any', 'error': 'str'}
            ),
            ToolDefinition(
                name='file_ops',
                description='Read or write files',
                input_schema={'path': 'str', 'content': 'str', 'mode': 'str'},
                output_schema={'success': 'bool', 'data': 'str'}
            ),
            ToolDefinition(
                name='api_call',
                description='Make HTTP API call',
                input_schema={'url': 'str', 'method': 'str', 'data': 'dict'},
                output_schema={'response': 'dict', 'status': 'int'}
            ),
            ToolDefinition(
                name='memory_query',
                description='Query long-term memory',
                input_schema={'query': 'str', 'memory_type': 'str'},
                output_schema={'results': 'list', 'metadata': 'dict'}
            ),
            ToolDefinition(
                name='calculate',
                description='Perform mathematical calculations',
                input_schema={'expression': 'str'},
                output_schema={'result': 'float', 'error': 'str'}
            )
        ]
        
        for tool in default_tools:
            self.register_tool(tool)
            
    def register_tool(self, tool: ToolDefinition):
        """Register a new tool with auto-generated embedding"""
        # Generate embedding from description
        # Simplified: use hash-based embedding
        desc_hash = hash(tool.description) % (2**32)
        np.random.seed(desc_hash)
        tool.embedding = np.random.randn(self.manifold_dim).astype(np.float32)
        tool.embedding = tool.embedding / np.linalg.norm(tool.embedding)  # Normalize
        
        self.tools[tool.name] = tool
        print(f"[+] Tool registered: {tool.name}")
        
    def encode_intention(self, text_or_vector) -> torch.Tensor:
        """Encode text or vector to manifold representation"""
        if isinstance(text_or_vector, str):
            # Simple hash-based encoding (replace with proper embeddings in production)
            text_hash = hash(text_or_vector) % (2**32)
            np.random.seed(text_hash)
            vector = np.random.randn(384).astype(np.float32)
            return self.text_encoder(torch.tensor(vector).unsqueeze(0))
        else:
            return torch.tensor(text_or_vector).unsqueeze(0)
            
    def select_tool(self, intention_vector: torch.Tensor) -> Tuple[str, float]:
        """Select best tool for the intention"""
        if len(self.tools) == 0:
            return None, 0.0
            
        # Compute tool embedding matrix
        tool_names = list(self.tools.keys())
        tool_embeddings = torch.tensor(
            np.stack([self.tools[n].embedding for n in tool_names])
        )
        
        # Compute similarities
        intention_flat = intention_vector.flatten()
        if intention_flat.dim() == 1:
            intention_flat = intention_flat.unsqueeze(0)
            
        similarities = F.cosine_similarity(
            intention_flat.expand(len(tool_names), -1),
            tool_embeddings,
            dim=-1
        )
        
        best_idx = similarities.argmax().item()
        confidence = similarities[best_idx].item()
        
        return tool_names[best_idx], confidence
    
    def extract_parameters(self, intention: torch.Tensor, context: Dict, 
                          tool_schema: Dict) -> Dict:
        """Extract tool parameters from intention and context"""
        params = {}
        
        # Simple parameter extraction from context
        for param_name, param_type in tool_schema.items():
            if param_name in context:
                params[param_name] = context[param_name]
            else:
                # Default values
                if param_type == 'str':
                    params[param_name] = ''
                elif param_type == 'int':
                    params[param_name] = 10
                elif param_type == 'float':
                    params[param_name] = 0.0
                elif param_type == 'dict':
                    params[param_name] = {}
                    
        return params
    
    def execute_tool(self, tool_name: str, params: Dict) -> Dict:
        """Execute a tool (simulated for demo)"""
        if tool_name not in self.tools:
            return {'error': f'Unknown tool: {tool_name}'}
            
        tool = self.tools[tool_name]
        tool.usage_count += 1
        
        # Simulate execution
        result = {'tool': tool_name, 'params': params, 'executed': True}
        
        if tool_name == 'search':
            result['results'] = [
                {'title': f'Search result for: {params.get("query", "")}', 'score': 0.9}
            ]
        elif tool_name == 'calculate':
            try:
                expr = params.get('expression', '0')
                # Safe eval for simple expressions
                result['result'] = float(eval(expr, {"__builtins__": {}}, {}))
            except:
                result['error'] = 'Calculation failed'
        elif tool_name == 'memory_query':
            result['results'] = ['Sample memory result']
            
        # Update success rate
        if 'error' not in result:
            tool.success_rate = (tool.success_rate * (tool.usage_count - 1) + 1) / tool.usage_count
        else:
            tool.success_rate = (tool.success_rate * (tool.usage_count - 1)) / tool.usage_count
            
        return result
    
    def forward(self, intention, context=None):
        """Full tool grounding pipeline"""
        if context is None:
            context = {}
            
        # Select tool
        tool_name, confidence = self.select_tool(intention)
        
        if confidence < 0.3:
            return {'error': 'No suitable tool found', 'confidence': confidence}
            
        # Extract parameters
        tool = self.tools[tool_name]
        params = self.extract_parameters(intention, context, tool.input_schema)
        
        # Execute
        result = self.execute_tool(tool_name, params)
        result['confidence'] = confidence
        
        return result


# =============================================================================
# Part 3: Goal Management System
# =============================================================================

@dataclass
class Goal:
    """Represents a goal in the system"""
    id: str
    vector: np.ndarray
    time_horizon: str  # 'long_term', 'medium_term', 'short_term'
    priority: float
    status: str = 'active'
    progress: float = 0.0
    sub_goals: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class IntentionalityField:
    """
    Creates a potential field on the manifold that guides reasoning toward goals.
    Based on physics analogy: goals are potential wells, obstacles are potential hills.
    """
    def __init__(self, manifold_dim=32):
        self.manifold_dim = manifold_dim
        
        # Targets (low potential = attractive)
        self.targets: Dict[str, np.ndarray] = {}
        
        # Obstacles (high potential = repulsive)
        self.obstacles: Dict[str, np.ndarray] = {}
        
        # Field parameters
        self.attraction_strength = 1.0
        self.repulsion_strength = 20.0
        self.repulsion_radius = 3.0
        
    def add_target(self, target_id: str, position: np.ndarray):
        """Add an attractive target"""
        self.targets[target_id] = position.copy()
        
    def add_obstacle(self, obs_id: str, position: np.ndarray):
        """Add a repulsive obstacle"""
        self.obstacles[obs_id] = position.copy()
        
    def remove_target(self, target_id: str):
        self.targets.pop(target_id, None)
        
    def compute_potential(self, state: np.ndarray) -> float:
        """Compute total potential at a state"""
        potential = 0.0
        
        # Target attraction (negative potential)
        for target_pos in self.targets.values():
            dist = np.linalg.norm(state - target_pos) + 1e-9
            potential -= self.attraction_strength / dist
            
        # Obstacle repulsion (positive potential)
        for obs_pos in self.obstacles.values():
            dist = np.linalg.norm(state - obs_pos) + 1e-9
            if dist < self.repulsion_radius:
                potential += self.repulsion_strength / (dist ** 4)
                
        return potential
    
    def compute_gradient(self, state: np.ndarray) -> np.ndarray:
        """Compute potential gradient (direction of steepest descent)"""
        grad = np.zeros(self.manifold_dim)
        
        # Target attraction gradient
        for target_pos in self.targets.values():
            diff = target_pos - state
            dist = np.linalg.norm(diff) + 1e-9
            grad += (diff / dist) * self.attraction_strength * (1 + 1/dist)
            
        # Obstacle repulsion gradient
        for obs_pos in self.obstacles.values():
            diff = state - obs_pos
            dist = np.linalg.norm(diff) + 1e-9
            if dist < self.repulsion_radius:
                grad += (diff / (dist ** 5)) * self.repulsion_strength * 4
                
        # Normalize
        grad_norm = np.linalg.norm(grad) + 1e-9
        return grad / grad_norm
    
    def generate_trajectory(self, start: np.ndarray, max_steps: int = 200,
                           step_size: float = 0.1) -> np.ndarray:
        """Generate trajectory from start to nearest target"""
        trajectory = [start.copy()]
        current = start.copy()
        
        for _ in range(max_steps):
            grad = self.compute_gradient(current)
            
            # Adaptive step size
            dist_to_target = min(
                [np.linalg.norm(current - t) for t in self.targets.values()],
                default=float('inf')
            )
            adaptive_step = step_size * (0.5 if dist_to_target < 1.0 else 1.0)
            
            current = current + adaptive_step * grad
            trajectory.append(current.copy())
            
            # Check if reached target
            if dist_to_target < 0.1:
                break
                
        return np.array(trajectory)


class GoalManagementSystem:
    """
    Manages hierarchical goals and coordinates intentionality field.
    """
    def __init__(self, manifold_dim=32):
        self.manifold_dim = manifold_dim
        
        # Goal hierarchy
        self.goals: Dict[str, Goal] = {}
        
        # Intentionality field
        self.field = IntentionalityField(manifold_dim)
        
        # Current state
        self.current_state = np.zeros(manifold_dim)
        
        # Statistics
        self.stats = {
            'goals_set': 0,
            'goals_completed': 0,
            'goals_failed': 0
        }
        
    def set_goal(self, goal_id: str, goal_vector: np.ndarray,
                 time_horizon: str = 'medium_term', priority: float = 1.0,
                 parent_id: str = None) -> Goal:
        """Set a new goal"""
        goal = Goal(
            id=goal_id,
            vector=goal_vector.copy(),
            time_horizon=time_horizon,
            priority=priority,
            parent_id=parent_id
        )
        
        self.goals[goal_id] = goal
        self.field.add_target(goal_id, goal_vector)
        
        if parent_id and parent_id in self.goals:
            self.goals[parent_id].sub_goals.append(goal_id)
            
        self.stats['goals_set'] += 1
        print(f"[+] Goal set: {goal_id} (priority={priority}, horizon={time_horizon})")
        
        return goal
    
    def decompose_goal(self, goal_id: str, n_sub_goals: int = 3) -> List[str]:
        """
        Decompose a goal into sub-goals using trajectory milestones.
        """
        if goal_id not in self.goals:
            return []
            
        parent_goal = self.goals[goal_id]
        
        # Generate trajectory from current state to goal
        trajectory = self.field.generate_trajectory(
            self.current_state, max_steps=100
        )
        
        # Extract milestones
        milestones = self._extract_milestones(trajectory, n_sub_goals)
        
        sub_goal_ids = []
        for i, milestone in enumerate(milestones):
            sub_id = f"{goal_id}_sub_{i}"
            sub_priority = parent_goal.priority * (0.9 ** (i + 1))
            
            self.set_goal(
                sub_id,
                milestone,
                time_horizon='short_term',
                priority=sub_priority,
                parent_id=goal_id
            )
            sub_goal_ids.append(sub_id)
            
        print(f"[+] Goal {goal_id} decomposed into {len(sub_goal_ids)} sub-goals")
        return sub_goal_ids
    
    def _extract_milestones(self, trajectory: np.ndarray, n: int) -> List[np.ndarray]:
        """Extract evenly spaced milestones from trajectory"""
        if len(trajectory) < n:
            return [trajectory[-1]]
            
        indices = np.linspace(0, len(trajectory) - 1, n + 1, dtype=int)[1:]
        return [trajectory[i] for i in indices]
    
    def update_progress(self, goal_id: str, progress: float):
        """Update goal progress"""
        if goal_id not in self.goals:
            return
            
        goal = self.goals[goal_id]
        goal.progress = max(0, min(1, progress))
        
        if goal.progress >= 1.0:
            goal.status = 'completed'
            self.stats['goals_completed'] += 1
            self.field.remove_target(goal_id)
            self._check_parent_completion(goal_id)
            print(f"[OK] Goal completed: {goal_id}")
            
    def _check_parent_completion(self, goal_id: str):
        """Check if parent goal can be completed"""
        goal = self.goals[goal_id]
        if goal.parent_id and goal.parent_id in self.goals:
            parent = self.goals[goal.parent_id]
            
            # Check if all sub-goals completed
            all_completed = all(
                self.goals[sub_id].status == 'completed'
                for sub_id in parent.sub_goals
                if sub_id in self.goals
            )
            
            if all_completed:
                self.update_progress(parent.id, 1.0)
                
    def get_next_action(self) -> Dict:
        """Get the next recommended action based on active goals"""
        # Find highest priority active goal
        active_goals = [
            g for g in self.goals.values()
            if g.status == 'active'
        ]
        
        if not active_goals:
            return {'action': 'none', 'reason': 'No active goals'}
            
        # Sort by priority
        active_goals.sort(key=lambda g: g.priority, reverse=True)
        top_goal = active_goals[0]
        
        # Compute action direction
        gradient = self.field.compute_gradient(self.current_state)
        
        return {
            'action': 'move_toward_goal',
            'goal_id': top_goal.id,
            'direction': gradient.tolist(),
            'distance': float(np.linalg.norm(top_goal.vector - self.current_state)),
            'priority': top_goal.priority
        }
    
    def execute_planning_cycle(self, max_steps: int = 100) -> Dict:
        """Execute one planning cycle"""
        action = self.get_next_action()
        
        if action['action'] == 'none':
            return action
            
        # Move in computed direction
        step = np.array(action['direction']) * 0.1
        self.current_state += step
        
        # Update progress for all active goals
        for goal in self.goals.values():
            if goal.status == 'active':
                dist = np.linalg.norm(goal.vector - self.current_state)
                # Progress = 1 - normalized_distance
                max_dist = np.linalg.norm(goal.vector)  # Distance from origin
                progress = max(0, 1 - dist / max(max_dist, 1))
                goal.progress = max(goal.progress, progress)
                
                if dist < 0.5:  # Close enough
                    self.update_progress(goal.id, 1.0)
                    
        return action
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            'total_goals': len(self.goals),
            'active_goals': sum(1 for g in self.goals.values() if g.status == 'active'),
            'completed_goals': sum(1 for g in self.goals.values() if g.status == 'completed'),
            'current_state_norm': float(np.linalg.norm(self.current_state)),
            **self.stats
        }


# =============================================================================
# Integrated System
# =============================================================================

class AGICapabilityExtension:
    """
    Integrated AGI capability system combining:
    - Long-term memory
    - Tool usage
    - Goal management
    """
    def __init__(self, manifold_dim=64):
        self.manifold_dim = manifold_dim
        
        # Initialize subsystems
        self.memory = HierarchicalMemorySystem(dim=manifold_dim)
        self.tools = ToolGroundingSystem(manifold_dim=manifold_dim)
        self.goals = GoalManagementSystem(manifold_dim=manifold_dim)
        
        print(f"[+] AGI Capability Extension initialized (dim={manifold_dim})")
        
    def process_experience(self, experience: np.ndarray, context: Dict = None):
        """Process and store an experience"""
        # Store in memory
        self.memory.store(experience)
        
        # Update goals if relevant
        if context and 'goal_update' in context:
            goal_id = context['goal_update']['id']
            progress = context['goal_update']['progress']
            self.goals.update_progress(goal_id, progress)
            
    def plan_and_act(self, intention_text: str, context: Dict = None) -> Dict:
        """Full planning and action cycle"""
        if context is None:
            context = {}
            
        # Encode intention
        intention = self.tools.encode_intention(intention_text)
        
        # Check if tool is needed
        tool_result = self.tools(intention, context)
        
        # Get goal-based action
        action = self.goals.get_next_action()
        
        # Check memory for relevant past experiences
        memory_result = self.memory.recall(intention.squeeze(0).detach().numpy())
        
        return {
            'tool_action': tool_result,
            'goal_action': action,
            'memory_recall': memory_result[0],  # Memory type
            'system_status': self.get_status()
        }
    
    def set_long_term_goal(self, goal_id: str, goal_description: str):
        """Set a long-term goal with automatic decomposition"""
        # Encode goal
        goal_vector = self.tools.encode_intention(goal_description).squeeze(0).detach().numpy()
        
        # Set as long-term goal
        self.goals.set_goal(goal_id, goal_vector, time_horizon='long_term', priority=1.0)
        
        # Decompose into sub-goals
        sub_goals = self.goals.decompose_goal(goal_id, n_sub_goals=3)
        
        return {
            'main_goal': goal_id,
            'sub_goals': sub_goals
        }
    
    def get_status(self) -> Dict:
        """Get full system status"""
        return {
            'memory': self.memory.get_memory_stats(),
            'goals': self.goals.get_system_status(),
            'tools': {
                'registered': len(self.tools.tools),
                'tool_names': list(self.tools.tools.keys())
            }
        }


# =============================================================================
# Testing
# =============================================================================

def test_agi_capability_extension():
    """Test the integrated AGI capability system"""
    print("\n" + "=" * 60)
    print("Testing AGI Capability Extension")
    print("=" * 60)
    
    # Initialize system
    system = AGICapabilityExtension(manifold_dim=64)
    
    # Test 1: Long-term memory
    print("\n[1] Testing Long-Term Memory...")
    
    # Store experiences
    for i in range(10):
        experience = np.random.randn(64) * (i + 1)  # Increasing importance
        system.process_experience(experience)
    
    memory_status = system.memory.get_memory_stats()
    print(f"    Episodes stored: {memory_status['episodic_count']}")
    print(f"    Metric deformation: {memory_status['metric_deformation']:.4f}")
    
    # Test 2: Tool usage
    print("\n[2] Testing Tool Usage...")
    
    # Test different intentions
    intentions = [
        "Search for information about quantum computing",
        "Calculate the square root of 144",
        "Read the configuration file"
    ]
    
    for intent in intentions:
        result = system.tools(system.tools.encode_intention(intent))
        print(f"    Intent: '{intent[:30]}...' -> Tool: {result.get('tool', 'N/A')}")
    
    # Test 3: Goal management
    print("\n[3] Testing Goal Management...")
    
    # Set a long-term goal
    goal_result = system.set_long_term_goal(
        "MASTER_QUANTUM_COMPUTING",
        "Learn and master quantum computing concepts"
    )
    print(f"    Main goal: {goal_result['main_goal']}")
    print(f"    Sub-goals: {goal_result['sub_goals']}")
    
    # Execute planning cycles
    for i in range(5):
        action = system.goals.execute_planning_cycle()
        if i == 0:
            print(f"    First action direction norm: {np.linalg.norm(action.get('direction', [0])):.4f}")
    
    # Check status
    status = system.goals.get_system_status()
    print(f"    Active goals: {status['active_goals']}")
    print(f"    Completed goals: {status['completed_goals']}")
    
    # Test 4: Full integration
    print("\n[4] Testing Full Integration...")
    
    result = system.plan_and_act(
        "I need to solve a complex optimization problem",
        context={'top_k': 5}
    )
    
    print(f"    Tool selected: {result['tool_action'].get('tool', 'N/A')}")
    print(f"    Goal action: {result['goal_action'].get('action', 'N/A')}")
    print(f"    Memory type: {result['memory_recall']}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    
    return system


if __name__ == "__main__":
    system = test_agi_capability_extension()
    
    # Save test results
    os.makedirs("tempdata", exist_ok=True)
    with open("tempdata/agi_capability_test.json", "w") as f:
        json.dump(system.get_status(), f, indent=2, default=str)
    print("\nResults saved to tempdata/agi_capability_test.json")
