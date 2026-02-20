
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

class EmbodiedEnv:
    """
    FiberSim-V1: A high-dimensional manifold navigation environment.
    The agent moves in a latent space with 'Obstacle Repulsors' and 'Goal Attractors'.
    """
    def __init__(self, dim=128, num_obstacles=5):
        self.dim = dim
        self.state = np.random.randn(dim) / np.sqrt(dim)
        self.goal = np.random.randn(dim) / np.sqrt(dim)
        
        # Obstacles are represented as high-curvature points in the manifold
        self.obstacles = [np.random.randn(dim) / np.sqrt(dim) for _ in range(num_obstacles)]
        self.obstacle_radius = 0.5
        self.max_steps = 200
        self.current_step = 0
        
    def reset(self):
        self.state = np.random.randn(self.dim) / np.sqrt(self.dim)
        self.current_step = 0
        return self.state
    
    def step(self, action_vector):
        """
        Action is a displacement vector in the manifold.
        """
        self.current_step += 1
        
        # 1. Apply Action (Geodesic step)
        # Normalize action to represent a finite velocity
        v = action_vector / (np.linalg.norm(action_vector) + 1e-6) * 0.1
        self.state = self.state + v
        
        # 2. Dynamics: Metabolic Cost calculation
        # Cost = Movement Energy + Obstacle Resistance
        move_cost = np.linalg.norm(v)
        
        obstacle_cost = 0
        for obs in self.obstacles:
            dist = np.linalg.norm(self.state - obs)
            if dist < self.obstacle_radius:
                # High cost for entering obstacle curvature
                obstacle_cost += (self.obstacle_radius - dist) * 10.0
                # Reflective force (Simple wall bounce effect)
                diff = (self.state - obs) / (dist + 1e-6)
                self.state += diff * (self.obstacle_radius - dist)
        
        dist_to_goal = np.linalg.norm(self.state - self.goal)
        reward = - (move_cost + obstacle_cost + 0.1 * dist_to_goal)
        
        done = (dist_to_goal < 0.1) or (self.current_step >= self.max_steps)
        
        return self.state, reward, done, {"dist": dist_to_goal}

# --- Demo Plotting ---
def demo_sim():
    env = EmbodiedEnv(dim=2, num_obstacles=3) # 2D for visualization
    state = env.reset()
    path = [state]
    
    print("[*] Running Embodied Sim Demo (2D visualization)...")
    for _ in range(100):
        # Human/Heuristic Policy: Head towards goal
        direction = env.goal - env.state
        state, rew, done, info = env.step(direction)
        path.append(state)
        if done: break
        
    path = np.array(path)
    plt.figure(figsize=(8,8))
    plt.plot(path[:, 0], path[:, 1], 'b-o', label='Agent Trajectory', alpha=0.5)
    plt.scatter(env.goal[0], env.goal[1], c='g', marker='*', s=200, label='Goal')
    
    for i, obs in enumerate(env.obstacles):
        circle = plt.Circle((obs[0], obs[1]), env.obstacle_radius, color='r', alpha=0.3)
        plt.gca().add_patch(circle)
        
    plt.title("FiberSim-V1: Geodesic Navigation Manifold")
    plt.legend()
    
    save_path = os.path.join("tempdata", "embodied_sim_demo.png")
    plt.savefig(save_path)
    print(f"[+] Demo trajectory saved to {save_path}")

if __name__ == "__main__":
    demo_sim()
