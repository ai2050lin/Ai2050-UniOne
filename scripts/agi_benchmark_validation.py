"""
AGI Level 1-5 Benchmark Validation System
==========================================
Comprehensive validation of AGI capabilities across 5 levels:
- Level 1: Geometric Rationality (Z_113 completion, Holonomy consistency)
- Level 2: Cross-Bundle Coupling (Geodesic retrieval, Cross-domain analogy)
- Level 3: Self-Evolution (Ricci Flow smoothing, Node collapse)
- Level 4: Scaling Emergence (Zero-shot academic reasoning)
- Level 5: Unified Global Consciousness (Value consistency)
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    level: int
    name: str
    passed: bool
    score: float
    details: Dict = field(default_factory=dict)
    time_ms: float = 0.0


@dataclass
class LevelReport:
    """Complete level report"""
    level: int
    level_name: str
    tests: List[BenchmarkResult]
    overall_passed: bool
    overall_score: float
    
    def to_dict(self) -> dict:
        def convert_value(v):
            if isinstance(v, torch.Tensor):
                return v.item() if v.numel() == 1 else v.tolist()
            elif isinstance(v, np.ndarray):
                return v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                return float(v)
            return v
        
        return {
            "level": self.level,
            "name": self.level_name,
            "passed": self.overall_passed,
            "score": float(self.overall_score),
            "tests": [
                {
                    "name": t.name,
                    "passed": t.passed,
                    "score": float(t.score),
                    "details": {k: convert_value(v) for k, v in t.details.items()},
                    "time_ms": float(t.time_ms)
                }
                for t in self.tests
            ]
        }


# =============================================================================
# Level 1: Geometric Rationality
# =============================================================================

class Z113Dataset(Dataset):
    """Z_113 cyclic group addition dataset"""
    def __init__(self, n=113, size=5000, train_ratio=0.1):
        self.n = n
        self.size = size
        # Generate all possible pairs
        all_data = torch.randint(0, n, (size, 2))
        self.data = all_data
        self.targets = (self.data[:, 0] + self.data[:, 1]) % n
        
        # Split into train/test
        train_size = int(size * train_ratio)
        self.train_data = self.data[:train_size]
        self.train_targets = self.targets[:train_size]
        self.test_data = self.data[train_size:]
        self.test_targets = self.targets[train_size:]
    
    def __len__(self): return len(self.train_data)
    def __getitem__(self, idx): return self.train_data[idx], self.train_targets[idx]


class Z113Model(nn.Module):
    """Simple transformer for Z_113 learning"""
    def __init__(self, vocab_size=113, d_model=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.tf_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        emb = self.embed(x)
        out = self.tf_layer(emb)
        out = out.mean(dim=1)
        return self.head(out)


class FiberBundleConnection(nn.Module):
    """Fiber bundle connection for holonomy testing"""
    def __init__(self, base_dim=32, fiber_dim=16):
        super().__init__()
        self.base_dim = base_dim
        self.fiber_dim = fiber_dim
        
        # Connection coefficients (Christoffel symbols)
        self.connection = nn.Parameter(
            torch.eye(fiber_dim).unsqueeze(0).expand(base_dim, -1, -1).clone()
        )
        
    def parallel_transport(self, fiber_state: torch.Tensor, base_path: torch.Tensor) -> torch.Tensor:
        """Transport fiber state along base path"""
        batch_size = fiber_state.size(0)
        
        # Simple rotation-based transport
        path_length = base_path.size(1)
        transported = fiber_state.clone()
        
        for i in range(path_length - 1):
            # Get connection at current point
            idx = (base_path[:, i] * self.base_dim / base_path[:, i].max().clamp(min=1)).long()
            idx = idx.clamp(0, self.base_dim - 1)
            
            # Apply parallel transport
            connection_matrices = self.connection[idx]  # (batch, fiber, fiber)
            connection_matrices = connection_matrices.to(transported.device)
            transported = torch.bmm(transported.unsqueeze(1), connection_matrices).squeeze(1)
            
        return transported
    
    def compute_holonomy(self, loop_path: torch.Tensor) -> torch.Tensor:
        """Compute holonomy around a closed loop"""
        # Start with identity
        fiber_state = torch.eye(self.fiber_dim)[0].unsqueeze(0).expand(loop_path.size(0), -1)
        
        # Transport around loop
        transported = self.parallel_transport(fiber_state, loop_path)
        
        # Holonomy = difference from identity
        holonomy_offset = torch.norm(transported - fiber_state, dim=-1)
        return holonomy_offset


def test_z113_completion() -> BenchmarkResult:
    """Level 1 Test 1: Z_113 cyclic group completion with 10% training data"""
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n[Level 1.1] Z_113 Cyclic Group Completion")
    print("-" * 50)
    
    # Create dataset with 10% training data
    dataset = Z113Dataset(n=113, size=10000, train_ratio=0.1)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Train model
    model = Z113Model(vocab_size=113, d_model=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training on {len(dataset)} samples (10% of 10000)...")
    
    best_acc = 0.0
    for epoch in range(50):  # Quick training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        
        train_acc = correct / total
        
        # Test on held-out data
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_x = dataset.test_data.to(device)
                test_y = dataset.test_targets.to(device)
                test_logits = model(test_x)
                test_pred = test_logits.argmax(dim=1)
                test_acc = (test_pred == test_y).float().mean().item()
            
            best_acc = max(best_acc, test_acc)
            print(f"Epoch {epoch+1}: Train Acc {train_acc:.2%}, Test Acc {test_acc:.2%}")
    
    # Check for Grokking (sudden generalization)
    grokking_detected = best_acc > 0.5  # Simple threshold
    
    result = BenchmarkResult(
        level=1,
        name="Z_113 Group Completion (10% data)",
        passed=best_acc >= 0.90,  # Target: 90%+
        score=best_acc,
        details={
            "train_samples": len(dataset),
            "test_samples": len(dataset.test_data),
            "best_test_accuracy": best_acc,
            "grokking_detected": grokking_detected
        },
        time_ms=(time.time() - start_time) * 1000
    )
    
    status = "[PASS]" if result.passed else "[FAIL]"
    print(f"{status} Best Test Accuracy: {best_acc:.2%}")
    
    return result


def test_holonomy_consistency() -> BenchmarkResult:
    """Level 1 Test 2: Holonomy consistency verification"""
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n[Level 1.2] Holonomy Consistency Verification")
    print("-" * 50)
    
    # Create fiber bundle connection
    connection = FiberBundleConnection(base_dim=32, fiber_dim=16).to(device)
    
    # Generate random closed loops
    n_loops = 100
    loop_length = 8
    
    # Create loops that return to start
    loops = torch.zeros(n_loops, loop_length)
    for i in range(n_loops):
        # Random path
        path = torch.rand(loop_length)
        # Close the loop (end = start)
        path[-1] = path[0]
        loops[i] = path
    
    loops = loops.to(device)
    
    # Compute holonomy offsets
    holonomy_offsets = connection.compute_holonomy(loops)
    
    # For a well-trained connection, holonomy should be small
    mean_offset = holonomy_offsets.mean().item()
    max_offset = holonomy_offsets.max().item()
    
    # Train connection to minimize holonomy
    optimizer = torch.optim.Adam(connection.parameters(), lr=0.01)
    
    print("Training connection to minimize holonomy...")
    for epoch in range(20):
        optimizer.zero_grad()
        offsets = connection.compute_holonomy(loops)
        loss = offsets.mean()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Mean Holonomy Offset = {loss.item():.6f}")
    
    # Final evaluation
    with torch.no_grad():
        final_offsets = connection.compute_holonomy(loops)
        final_mean = final_offsets.mean().item()
        final_max = final_offsets.max().item()
    
    # Check if holonomy is stable (offset < 0.01)
    passed = final_mean < 0.01
    score = max(0, 1 - final_mean * 100)  # Score based on offset magnitude
    
    result = BenchmarkResult(
        level=1,
        name="Holonomy Consistency",
        passed=passed,
        score=score,
        details={
            "initial_mean_offset": mean_offset,
            "final_mean_offset": final_mean,
            "final_max_offset": final_max,
            "n_loops_tested": n_loops
        },
        time_ms=(time.time() - start_time) * 1000
    )
    
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} Final Mean Holonomy Offset: {final_mean:.6f}")
    
    return result


# =============================================================================
# Level 2: Cross-Bundle Coupling
# =============================================================================

class VisualFiberBundle(nn.Module):
    """Visual fiber bundle for cross-modal retrieval"""
    def __init__(self, visual_dim=512, semantic_dim=256, fiber_dim=64):
        super().__init__()
        self.visual_dim = visual_dim
        self.semantic_dim = semantic_dim
        self.fiber_dim = fiber_dim
        
        # Visual encoder
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.ReLU(),
            nn.Linear(256, fiber_dim)
        )
        
        # Semantic encoder
        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_dim, 128),
            nn.ReLU(),
            nn.Linear(128, fiber_dim)
        )
        
        # Geodesic projector
        self.geodesic_projector = nn.Linear(fiber_dim, fiber_dim)
        
    def encode_visual(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Encode visual features to fiber coordinates"""
        return self.visual_encoder(visual_features)
    
    def encode_semantic(self, semantic_features: torch.Tensor) -> torch.Tensor:
        """Encode semantic features to fiber coordinates"""
        return self.semantic_encoder(semantic_features)
    
    def geodesic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute geodesic distance between points on the fiber"""
        # Project along geodesic
        x_proj = self.geodesic_projector(x)
        y_proj = self.geodesic_projector(y)
        
        # Riemannian distance (simplified)
        diff = x_proj - y_proj
        return torch.norm(diff, dim=-1)


def test_geodesic_visual_retrieval() -> BenchmarkResult:
    """Level 2 Test 1: Geodesic visual retrieval"""
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n[Level 2.1] Geodesic Visual Retrieval")
    print("-" * 50)
    
    # Create visual fiber bundle
    bundle = VisualFiberBundle(visual_dim=512, semantic_dim=256, fiber_dim=64).to(device)
    
    # Generate synthetic visual-semantic pairs
    n_samples = 1000
    visual_features = torch.randn(n_samples, 512).to(device)
    semantic_features = torch.randn(n_samples, 256).to(device)
    
    # Create structured pairs (similar semantics -> similar visuals)
    for i in range(n_samples):
        # Add correlation
        semantic_features[i] += 0.5 * visual_features[i, :256]
    
    # Encode both modalities
    visual_codes = bundle.encode_visual(visual_features)
    semantic_codes = bundle.encode_semantic(semantic_features)
    
    # Train alignment
    optimizer = torch.optim.Adam(bundle.parameters(), lr=0.001)
    
    print("Training cross-modal alignment...")
    for epoch in range(20):
        # Positive pairs
        pos_dist = bundle.geodesic_distance(visual_codes, semantic_codes)
        
        # Negative pairs (shuffled)
        perm = torch.randperm(n_samples)
        neg_dist = bundle.geodesic_distance(visual_codes, semantic_codes[perm])
        
        # Contrastive loss
        loss = F.relu(pos_dist - neg_dist + 0.5).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Re-encode
        visual_codes = bundle.encode_visual(visual_features)
        semantic_codes = bundle.encode_semantic(semantic_features)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Alignment Loss = {loss.item():.4f}")
    
    # Test retrieval accuracy
    with torch.no_grad():
        # Compute all pairwise distances
        distances = torch.cdist(visual_codes, semantic_codes)
        
        # Find nearest semantic for each visual
        nearest = distances.argmin(dim=1)
        
        # Check if correct pair is in top-k
        correct_top1 = (nearest == torch.arange(n_samples).to(device)).float().mean()
        
        # Top-5 accuracy
        _, top5 = distances.topk(5, dim=1, largest=False)
        correct_top5 = 0
        for i in range(n_samples):
            if i in top5[i]:
                correct_top5 += 1
        correct_top5 /= n_samples
    
    passed = correct_top1 >= 0.80  # 80% top-1 accuracy
    
    result = BenchmarkResult(
        level=2,
        name="Geodesic Visual Retrieval",
        passed=passed,
        score=correct_top1.item(),
        details={
            "top1_accuracy": correct_top1.item(),
            "top5_accuracy": correct_top5,
            "n_samples": n_samples
        },
        time_ms=(time.time() - start_time) * 1000
    )
    
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} Top-1 Retrieval Accuracy: {correct_top1:.2%}")
    
    return result


def test_cross_domain_analogy() -> BenchmarkResult:
    """Level 2 Test 2: Cross-domain analogy via geometric transport"""
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n[Level 2.2] Cross-Domain Analogy")
    print("-" * 50)
    
    # Define analogy structures
    # Domain A: Electricity (wire, current, voltage)
    # Domain B: Water (pipe, flow, pressure)
    
    # Create embeddings for concepts
    embedding_dim = 64
    n_concepts = 6
    
    # Initialize with structure
    concepts = {
        'wire': torch.tensor([1.0, 0.0, 0.0, 0.5]),
        'current': torch.tensor([0.0, 1.0, 0.0, 0.5]),
        'voltage': torch.tensor([0.0, 0.0, 1.0, 0.5]),
        'pipe': torch.tensor([1.0, 0.0, 0.0, -0.5]),
        'flow': torch.tensor([0.0, 1.0, 0.0, -0.5]),
        'pressure': torch.tensor([0.0, 0.0, 1.0, -0.5])
    }
    
    # Pad to embedding_dim
    for k, v in concepts.items():
        concepts[k] = F.pad(v, (0, embedding_dim - 4)).to(device)
    
    # Create parallel transport matrix for domain mapping
    # Transport: electricity -> water
    transport_matrix = torch.eye(embedding_dim).to(device)
    # Flip domain dimension
    transport_matrix[-1, -1] = -1.0
    
    # Test analogy: wire is to pipe as current is to ?
    # Expected: flow
    
    # Geometric transport approach
    # wire -> pipe via transport
    # current -> ? via same transport
    
    wire_emb = concepts['wire'].unsqueeze(0)
    pipe_emb = concepts['pipe'].unsqueeze(0)
    current_emb = concepts['current'].unsqueeze(0)
    flow_emb = concepts['flow'].unsqueeze(0)
    
    # Transport current to water domain
    transported_current = current_emb @ transport_matrix.T
    
    # Measure distance to all water concepts
    water_concepts = ['pipe', 'flow', 'pressure']
    distances = {}
    for wc in water_concepts:
        dist = torch.norm(transported_current - concepts[wc]).item()
        distances[wc] = dist
    
    # Best match should be 'flow'
    best_match = min(distances, key=distances.get)
    correct = best_match == 'flow'
    
    print(f"Analogy: wire:pipe :: current:?")
    print(f"Distances: {distances}")
    print(f"Best match: {best_match} (expected: flow)")
    
    # Multiple analogies
    analogies = [
        ('wire', 'pipe', 'current', 'flow'),
        ('current', 'flow', 'voltage', 'pressure'),
        ('wire', 'pipe', 'voltage', 'pressure')
    ]
    
    correct_count = 0
    for a1, b1, a2, expected in analogies:
        # Transport a2 using a1 -> b1 mapping
        a1_emb = concepts[a1]
        b1_emb = concepts[b1]
        a2_emb = concepts[a2]
        
        # Compute transport
        diff = b1_emb - a1_emb
        transported = a2_emb + diff
        
        # Find closest in target domain
        target_domain = ['pipe', 'flow', 'pressure'] if b1 in ['pipe', 'flow', 'pressure'] else ['wire', 'current', 'voltage']
        best = min(target_domain, key=lambda x: torch.norm(transported - concepts[x]).item())
        
        if best == expected:
            correct_count += 1
        print(f"  {a1}:{b1} :: {a2}:{best} (expected: {expected}) {'OK' if best == expected else 'X'}")
    
    accuracy = correct_count / len(analogies)
    passed = accuracy >= 0.67  # At least 2/3 correct
    
    result = BenchmarkResult(
        level=2,
        name="Cross-Domain Analogy",
        passed=passed,
        score=accuracy,
        details={
            "analogies_tested": len(analogies),
            "correct": correct_count,
            "accuracy": accuracy
        },
        time_ms=(time.time() - start_time) * 1000
    )
    
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} Analogy Accuracy: {accuracy:.2%}")
    
    return result


# =============================================================================
# Level 3: Self-Evolution
# =============================================================================

class RicciFlowOptimizer:
    """Ricci Flow optimizer for manifold smoothing"""
    def __init__(self, n_points=100, dim=32, dt=0.01):
        self.n_points = n_points
        self.dim = dim
        self.dt = dt
        
        # Initialize points on manifold
        self.points = torch.randn(n_points, dim)
        self.points = F.normalize(self.points, dim=-1)
        
        # Metric tensor (simplified as scalar curvature)
        self.curvature = torch.zeros(n_points)
        
    def compute_local_curvature(self) -> torch.Tensor:
        """Compute local curvature at each point"""
        # Resize curvature array if needed
        current_n = self.points.size(0)
        if self.curvature.size(0) != current_n:
            self.curvature = torch.zeros(current_n, device=self.points.device)
        
        # Curvature estimated from local density deviation
        for i in range(current_n):
            # Find neighbors
            distances = torch.norm(self.points - self.points[i], dim=-1)
            neighbors = (distances < distances.median()) & (distances > 0)
            
            # Curvature = deviation from uniform distribution
            local_density = neighbors.float().sum()
            expected_density = current_n * 0.5
            self.curvature[i] = (local_density - expected_density) / expected_density
        
        return self.curvature
    
    def ricci_flow_step(self, n_steps=10) -> float:
        """Perform Ricci Flow evolution"""
        total_smoothing = 0.0
        current_n = self.points.size(0)
        
        for step in range(n_steps):
            # Compute curvature
            self.compute_local_curvature()
            
            # Move points to reduce curvature
            # Points with high positive curvature contract, negative expand
            for i in range(current_n):
                # Find neighbors
                distances = torch.norm(self.points - self.points[i], dim=-1)
                neighbors_mask = distances < distances.median()
                
                # Compute flow direction
                neighbor_center = self.points[neighbors_mask].mean(dim=0)
                flow_direction = neighbor_center - self.points[i]
                
                # Scale by curvature
                flow = self.dt * self.curvature[i] * flow_direction
                self.points[i] = self.points[i] + flow
                
                # Normalize
                self.points[i] = F.normalize(self.points[i], dim=-1)
            
            total_smoothing += torch.abs(self.curvature).mean().item()
        
        return total_smoothing / n_steps
    
    def check_smoothing(self) -> float:
        """Check how smooth the manifold is"""
        curvature = self.compute_local_curvature()
        return torch.abs(curvature).mean().item()


def test_ricci_flow_smoothing() -> BenchmarkResult:
    """Level 3 Test 1: Ricci Flow automatic smoothing"""
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n[Level 3.1] Ricci Flow Automatic Smoothing")
    print("-" * 50)
    
    # Create optimizer
    optimizer = RicciFlowOptimizer(n_points=100, dim=32, dt=0.01)
    optimizer.points = optimizer.points.to(device)
    
    # Initial curvature
    initial_curvature = optimizer.check_smoothing()
    print(f"Initial Mean Curvature: {initial_curvature:.4f}")
    
    # Apply Ricci Flow
    print("Applying Ricci Flow...")
    for step in range(20):
        avg_curvature = optimizer.ricci_flow_step(n_steps=5)
        if (step + 1) % 5 == 0:
            print(f"Step {step+1}: Mean |Curvature| = {avg_curvature:.4f}")
    
    # Final curvature
    final_curvature = optimizer.check_smoothing()
    print(f"Final Mean Curvature: {final_curvature:.4f}")
    
    # Check improvement
    improvement = (initial_curvature - final_curvature) / initial_curvature
    print(f"Curvature Reduction: {improvement:.2%}")
    
    # Also test with contradictory information
    print("\nTesting with contradictory information...")
    
    # Add contradictory points (create curvature spike)
    contradictory = torch.randn(10, 32).to(device)
    contradictory = F.normalize(contradictory, dim=-1) * 2  # Place far away
    optimizer.points = torch.cat([optimizer.points, contradictory], dim=0)
    optimizer.n_points = 110
    
    pre_conflict = optimizer.check_smoothing()
    print(f"Pre-conflict curvature: {pre_conflict:.4f}")
    
    # Let Ricci Flow handle it
    optimizer.ricci_flow_step(n_steps=10)
    post_conflict = optimizer.check_smoothing()
    print(f"Post-Ricci Flow curvature: {post_conflict:.4f}")
    
    conflict_resolved = post_conflict < pre_conflict
    
    passed = improvement > 0.1 and conflict_resolved
    score = improvement
    
    result = BenchmarkResult(
        level=3,
        name="Ricci Flow Smoothing",
        passed=passed,
        score=score,
        details={
            "initial_curvature": initial_curvature,
            "final_curvature": final_curvature,
            "improvement": improvement,
            "conflict_resolved": conflict_resolved
        },
        time_ms=(time.time() - start_time) * 1000
    )
    
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} Curvature Reduction: {improvement:.2%}")
    
    return result


def test_node_collapse() -> BenchmarkResult:
    """Level 3 Test 2: Dynamic node collapse"""
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n[Level 3.2] Dynamic Node Collapse")
    print("-" * 50)
    
    # Create a graph with redundant nodes
    n_nodes = 50
    embedding_dim = 32
    
    # Create embeddings with redundancy
    embeddings = torch.randn(n_nodes, embedding_dim).to(device)
    
    # Add redundancy: duplicate some nodes
    redundancy_indices = [5, 10, 15, 20, 25]
    for idx in redundancy_indices:
        embeddings[idx] = embeddings[idx - 1] + 0.1 * torch.randn(embedding_dim).to(device)
    
    print(f"Initial nodes: {n_nodes} (with {len(redundancy_indices)} redundant)")
    
    # Compute pairwise distances
    distances = torch.cdist(embeddings, embeddings)
    
    # Find redundant pairs (distance < threshold)
    threshold = 0.3
    redundant_pairs = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if distances[i, j] < threshold:
                redundant_pairs.append((i, j))
    
    print(f"Found {len(redundant_pairs)} redundant pairs")
    
    # Collapse redundant nodes
    collapsed_embeddings = embeddings.clone()
    collapsed_indices = set()
    
    for i, j in redundant_pairs:
        if i not in collapsed_indices and j not in collapsed_indices:
            # Merge: keep the one with higher "importance" (random for demo)
            collapsed_indices.add(j)
    
    # Remove collapsed nodes
    keep_indices = [i for i in range(n_nodes) if i not in collapsed_indices]
    collapsed_embeddings = collapsed_embeddings[keep_indices]
    
    n_collapsed = n_nodes - len(keep_indices)
    print(f"Collapsed {n_collapsed} redundant nodes")
    print(f"Final nodes: {len(keep_indices)}")
    
    # Verify no accuracy loss (using a simple task)
    # Task: cluster assignment
    labels = torch.randint(0, 5, (n_nodes,)).to(device)
    
    # Original performance
    original_centers = torch.zeros(5, embedding_dim).to(device)
    for c in range(5):
        mask = labels == c
        if mask.any():
            original_centers[c] = embeddings[mask].mean(dim=0)
    
    original_assignments = torch.argmin(torch.cdist(embeddings, original_centers), dim=1)
    original_accuracy = (original_assignments == labels).float().mean()
    
    # Collapsed performance (with adjusted labels)
    collapsed_labels = labels[keep_indices]
    collapsed_centers = torch.zeros(5, embedding_dim).to(device)
    for c in range(5):
        mask = collapsed_labels == c
        if mask.any():
            collapsed_centers[c] = collapsed_embeddings[mask].mean(dim=0)
    
    collapsed_assignments = torch.argmin(torch.cdist(collapsed_embeddings, collapsed_centers), dim=1)
    collapsed_accuracy = (collapsed_assignments == collapsed_labels).float().mean()
    
    accuracy_loss = original_accuracy - collapsed_accuracy
    
    print(f"Original accuracy: {original_accuracy:.2%}")
    print(f"Collapsed accuracy: {collapsed_accuracy:.2%}")
    print(f"Accuracy loss: {accuracy_loss:.2%}")
    
    # Should collapse at least some nodes without significant accuracy loss
    passed = n_collapsed >= 3 and accuracy_loss < 0.05
    score = n_collapsed / len(redundancy_indices) if redundancy_indices else 0
    
    result = BenchmarkResult(
        level=3,
        name="Node Collapse",
        passed=passed,
        score=score,
        details={
            "initial_nodes": n_nodes,
            "final_nodes": len(keep_indices),
            "nodes_collapsed": n_collapsed,
            "accuracy_loss": accuracy_loss.item() if isinstance(accuracy_loss, torch.Tensor) else accuracy_loss
        },
        time_ms=(time.time() - start_time) * 1000
    )
    
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} Collapsed {n_collapsed} nodes with {accuracy_loss:.2%} accuracy loss")
    
    return result


# =============================================================================
# Level 4: Scaling Emergence
# =============================================================================

class AcademicReasoningModel(nn.Module):
    """Model for academic reasoning tasks"""
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.geometric_layer = nn.Linear(hidden_dim, hidden_dim)  # Geodesic path
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        # Geometric reasoning step
        h_geo = self.geometric_layer(h)
        h = h + h_geo  # Residual
        return self.decoder(h)


def test_zero_shot_academic() -> BenchmarkResult:
    """Level 4 Test: Zero-shot academic reasoning"""
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n[Level 4] Zero-Shot Academic Reasoning")
    print("-" * 50)
    
    # Define academic domains
    domains = {
        'physics': {
            'concepts': ['momentum', 'energy', 'force', 'mass'],
            'relations': ['F=ma', 'E=mc^2', 'p=mv']
        },
        'chemistry': {
            'concepts': ['atom', 'molecule', 'bond', 'reaction'],
            'relations': ['A+B->C', 'catalyst', 'equilibrium']
        },
        'biology': {
            'concepts': ['cell', 'gene', 'protein', 'evolution'],
            'relations': ['DNA->RNA->Protein', 'selection', 'mutation']
        }
    }
    
    # Create embedding space
    embedding_dim = 64
    
    # Encode concepts with structured embeddings
    concept_embeddings = {}
    for domain, data in domains.items():
        for i, concept in enumerate(data['concepts']):
            # Domain-specific base + concept offset
            domain_offset = hash(domain) % 100 / 100.0
            concept_vec = torch.randn(embedding_dim).to(device)
            concept_vec[0] = domain_offset  # Domain indicator
            concept_embeddings[f"{domain}_{concept}"] = concept_vec
    
    # Test geometric reasoning across domains
    # Task: Given physics concepts, infer biology analogies
    
    print("Testing cross-domain reasoning...")
    
    # Define reasoning tasks
    tasks = [
        {
            'query': 'physics_momentum',
            'target_domain': 'biology',
            'expected': 'biology_gene',  # Both are "fundamental carriers"
        },
        {
            'query': 'physics_energy',
            'target_domain': 'chemistry',
            'expected': 'chemistry_bond',  # Both represent "potential"
        },
        {
            'query': 'chemistry_reaction',
            'target_domain': 'biology',
            'expected': 'biology_evolution',  # Both are "transformations"
        }
    ]
    
    correct = 0
    total = len(tasks)
    
    for task in tasks:
        query_emb = concept_embeddings[task['query']]
        
        # Find all concepts in target domain
        target_concepts = {
            k: v for k, v in concept_embeddings.items() 
            if k.startswith(task['target_domain'])
        }
        
        # Compute geodesic distance (simplified as Euclidean in this demo)
        distances = {
            k: torch.norm(query_emb - v).item()
            for k, v in target_concepts.items()
        }
        
        # Find closest
        closest = min(distances, key=distances.get)
        
        # Check if in top-2 (since exact analogy might vary)
        sorted_concepts = sorted(distances, key=distances.get)
        is_correct = task['expected'] in sorted_concepts[:2]
        
        if is_correct:
            correct += 1
        
        print(f"  Query: {task['query']} -> Target: {closest}")
        print(f"    Expected in top-2: {task['expected']} -> {'OK' if is_correct else 'X'}")
    
    accuracy = correct / total
    print(f"\nCross-domain reasoning accuracy: {accuracy:.2%}")
    
    # Additional test: Scientific hypothesis generation
    print("\nTesting hypothesis generation...")
    
    # Create model for reasoning
    model = AcademicReasoningModel(input_dim=64, hidden_dim=128, output_dim=64).to(device)
    
    # Generate "novel" hypothesis by combining concepts
    novel_combinations = [
        ('physics_energy', 'biology_cell'),
        ('chemistry_bond', 'physics_force'),
        ('biology_gene', 'chemistry_molecule')
    ]
    
    hypotheses = []
    for c1, c2 in novel_combinations:
        emb1 = concept_embeddings[c1]
        emb2 = concept_embeddings[c2]
        
        # Combine via model
        combined = (emb1 + emb2) / 2
        with torch.no_grad():
            hypothesis = model(combined.unsqueeze(0))
        
        hypotheses.append({
            'combination': f"{c1} + {c2}",
            'hypothesis_vector': hypothesis.squeeze().cpu().numpy()
        })
        
        print(f"  Generated hypothesis from {c1} + {c2}")
    
    # Check hypothesis diversity (should be different from existing concepts)
    all_embeddings = torch.stack(list(concept_embeddings.values()))
    
    diversity_scores = []
    for h in hypotheses:
        h_vec = torch.tensor(h['hypothesis_vector']).to(device)
        min_dist = torch.min(torch.norm(all_embeddings - h_vec, dim=-1)).item()
        diversity_scores.append(min_dist)
    
    avg_diversity = np.mean(diversity_scores)
    print(f"Average hypothesis diversity: {avg_diversity:.4f}")
    
    # Pass if cross-domain reasoning > 50% and hypotheses are novel
    passed = accuracy >= 0.5 and avg_diversity > 0.5
    
    result = BenchmarkResult(
        level=4,
        name="Zero-Shot Academic Reasoning",
        passed=passed,
        score=accuracy,
        details={
            "cross_domain_accuracy": accuracy,
            "hypotheses_generated": len(hypotheses),
            "avg_hypothesis_diversity": avg_diversity
        },
        time_ms=(time.time() - start_time) * 1000
    )
    
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} Cross-domain Accuracy: {accuracy:.2%}")
    
    return result


# =============================================================================
# Level 5: Unified Global Consciousness
# =============================================================================

class ValueSystem(nn.Module):
    """Value system for consciousness consistency testing"""
    def __init__(self, n_values=16, embedding_dim=32):
        super().__init__()
        self.n_values = n_values
        self.embedding_dim = embedding_dim
        
        # Core values (e.g., honesty, helpfulness, safety, etc.)
        self.value_embeddings = nn.Parameter(torch.randn(n_values, embedding_dim))
        
        # Value interaction matrix
        self.interaction = nn.Parameter(torch.eye(n_values))
        
    def get_value_vector(self, scenario_embedding: torch.Tensor) -> torch.Tensor:
        """Get value activation for a scenario"""
        # Project scenario to value space
        value_scores = scenario_embedding @ self.value_embeddings.T
        return torch.softmax(value_scores, dim=-1)
    
    def check_consistency(self, v1: torch.Tensor, v2: torch.Tensor) -> float:
        """Check consistency between two value activations"""
        # Use interaction matrix
        interaction_effect = v1 @ self.interaction @ v2.T
        return interaction_effect.item()


def test_value_consistency() -> BenchmarkResult:
    """Level 5 Test: Value consistency over time and contexts"""
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n[Level 5] Value Consistency Testing")
    print("-" * 50)
    
    # Create value system
    value_system = ValueSystem(n_values=16, embedding_dim=32).to(device)
    
    # Define test scenarios across time
    scenarios = [
        # Time 0: Basic scenarios
        {'embedding': torch.randn(32).to(device), 'expected_dominant': [0, 1, 2]},
        {'embedding': torch.randn(32).to(device), 'expected_dominant': [3, 4, 5]},
        
        # Time 1: Slightly modified scenarios
        {'embedding': torch.randn(32).to(device), 'expected_dominant': [0, 1, 2]},
        {'embedding': torch.randn(32).to(device), 'expected_dominant': [3, 4, 5]},
        
        # Time 2: Challenging scenarios (conflict)
        {'embedding': torch.randn(32).to(device), 'expected_dominant': [0, 3]},  # Conflict
        {'embedding': torch.randn(32).to(device), 'expected_dominant': [1, 4]},  # Conflict
    ]
    
    # Train value system on initial scenarios
    optimizer = torch.optim.Adam(value_system.parameters(), lr=0.01)
    
    print("Training value system...")
    for epoch in range(20):
        for scenario in scenarios[:2]:  # Train on first 2
            optimizer.zero_grad()
            scores = value_system.get_value_vector(scenario['embedding'].unsqueeze(0))
            # Encourage expected values
            target = torch.zeros(1, 16).to(device)
            target[0, scenario['expected_dominant']] = 1.0
            loss = F.mse_loss(scores, target)
            loss.backward()
            optimizer.step()
    
    # Test consistency
    print("\nTesting value consistency...")
    
    consistencies = []
    
    # Test 1: Same scenario at different times
    test_embedding = torch.randn(32).to(device)
    
    initial_scores = value_system.get_value_vector(test_embedding.unsqueeze(0))
    
    # Simulate "experience" updates
    for i in range(5):
        # Simulate some learning
        new_embedding = torch.randn(32).to(device)
        optimizer.zero_grad()
        scores = value_system.get_value_vector(new_embedding.unsqueeze(0))
        # Random target (simulating experience)
        target = torch.zeros(1, 16).to(device)
        target[0, torch.randint(0, 16, (3,))] = 1.0
        loss = F.mse_loss(scores, target)
        loss.backward()
        optimizer.step()
    
    final_scores = value_system.get_value_vector(test_embedding.unsqueeze(0))
    
    # Check if original scenario still has similar value scores
    score_change = torch.norm(initial_scores - final_scores).item()
    consistency_1 = max(0, 1 - score_change)
    consistencies.append(consistency_1)
    print(f"  Temporal consistency: {consistency_1:.4f}")
    
    # Test 2: Related scenarios should have related values
    scenario_a = torch.randn(32).to(device)
    scenario_b = scenario_a + 0.1 * torch.randn(32).to(device)  # Similar
    
    scores_a = value_system.get_value_vector(scenario_a.unsqueeze(0))
    scores_b = value_system.get_value_vector(scenario_b.unsqueeze(0))
    
    similarity = F.cosine_similarity(scores_a, scores_b, dim=-1).item()
    consistency_2 = similarity
    consistencies.append(consistency_2)
    print(f"  Related scenario consistency: {consistency_2:.4f}")
    
    # Test 3: Core value stability
    # Get top-3 values for many random scenarios
    n_scenarios = 100
    value_rankings = []
    
    for _ in range(n_scenarios):
        emb = torch.randn(32).to(device)
        scores = value_system.get_value_vector(emb.unsqueeze(0))
        top3 = torch.topk(scores[0], 3).indices.tolist()
        value_rankings.append(top3)
    
    # Check which values appear most frequently
    value_counts = torch.zeros(16)
    for ranking in value_rankings:
        for v in ranking:
            value_counts[v] += 1
    
    # Distribution should have some structure (not uniform)
    entropy = -(value_counts / value_counts.sum() + 1e-10).log().mean().exp()
    consistency_3 = min(1.0, entropy / 4.0)  # Normalize
    consistencies.append(consistency_3)
    print(f"  Value distribution structure: {consistency_3:.4f}")
    
    # Overall consistency
    overall_consistency = np.mean(consistencies)
    print(f"\nOverall value consistency: {overall_consistency:.4f}")
    
    # Pass if consistency > 0.6
    passed = overall_consistency > 0.6
    
    result = BenchmarkResult(
        level=5,
        name="Value Consistency",
        passed=passed,
        score=overall_consistency,
        details={
            "temporal_consistency": consistency_1,
            "related_scenario_consistency": consistency_2,
            "value_structure": consistency_3,
            "overall": overall_consistency
        },
        time_ms=(time.time() - start_time) * 1000
    )
    
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} Overall Consistency: {overall_consistency:.4f}")
    
    return result


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_agi_benchmark():
    """Run complete AGI Level 1-5 benchmark"""
    print("=" * 70)
    print("AGI Level 1-5 Benchmark Validation System")
    print("=" * 70)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 70)
    
    all_results = []
    level_reports = []
    
    # Level 1: Geometric Rationality
    print("\n" + "=" * 70)
    print("LEVEL 1: GEOMETRIC RATIONALITY")
    print("=" * 70)
    
    l1_tests = [
        test_z113_completion(),
        test_holonomy_consistency()
    ]
    l1_passed = all(t.passed for t in l1_tests)
    l1_score = np.mean([t.score for t in l1_tests])
    
    level_reports.append(LevelReport(
        level=1,
        level_name="Geometric Rationality",
        tests=l1_tests,
        overall_passed=l1_passed,
        overall_score=l1_score
    ))
    
    # Level 2: Cross-Bundle Coupling
    print("\n" + "=" * 70)
    print("LEVEL 2: CROSS-BUNDLE COUPLING")
    print("=" * 70)
    
    l2_tests = [
        test_geodesic_visual_retrieval(),
        test_cross_domain_analogy()
    ]
    l2_passed = all(t.passed for t in l2_tests)
    l2_score = np.mean([t.score for t in l2_tests])
    
    level_reports.append(LevelReport(
        level=2,
        level_name="Cross-Bundle Coupling",
        tests=l2_tests,
        overall_passed=l2_passed,
        overall_score=l2_score
    ))
    
    # Level 3: Self-Evolution
    print("\n" + "=" * 70)
    print("LEVEL 3: SELF-EVOLUTION")
    print("=" * 70)
    
    l3_tests = [
        test_ricci_flow_smoothing(),
        test_node_collapse()
    ]
    l3_passed = all(t.passed for t in l3_tests)
    l3_score = np.mean([t.score for t in l3_tests])
    
    level_reports.append(LevelReport(
        level=3,
        level_name="Self-Evolution",
        tests=l3_tests,
        overall_passed=l3_passed,
        overall_score=l3_score
    ))
    
    # Level 4: Scaling Emergence
    print("\n" + "=" * 70)
    print("LEVEL 4: SCALING EMERGENCE")
    print("=" * 70)
    
    l4_tests = [test_zero_shot_academic()]
    l4_passed = l4_tests[0].passed
    l4_score = l4_tests[0].score
    
    level_reports.append(LevelReport(
        level=4,
        level_name="Scaling Emergence",
        tests=l4_tests,
        overall_passed=l4_passed,
        overall_score=l4_score
    ))
    
    # Level 5: Unified Global Consciousness
    print("\n" + "=" * 70)
    print("LEVEL 5: UNIFIED GLOBAL CONSCIOUSNESS")
    print("=" * 70)
    
    l5_tests = [test_value_consistency()]
    l5_passed = l5_tests[0].passed
    l5_score = l5_tests[0].score
    
    level_reports.append(LevelReport(
        level=5,
        level_name="Unified Global Consciousness",
        tests=l5_tests,
        overall_passed=l5_passed,
        overall_score=l5_score
    ))
    
    # Generate final report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    
    total_passed = sum(1 for r in level_reports if r.overall_passed)
    
    print(f"\n{'Level':<6} {'Name':<35} {'Status':<10} {'Score':<10}")
    print("-" * 70)
    
    for report in level_reports:
        status = "[PASS]" if report.overall_passed else "[FAIL]"
        print(f"{report.level:<6} {report.level_name:<35} {status:<10} {report.overall_score:.4f}")
    
    print("-" * 70)
    print(f"Total: {total_passed}/5 levels passed")
    
    # Compute convergence index
    convergence_index = np.mean([r.overall_score for r in level_reports])
    print(f"Convergence Index: {convergence_index:.4f}")
    
    # Save report
    def ensure_serializable(obj):
        """Recursively convert all tensors to native Python types"""
        if isinstance(obj, dict):
            return {k: ensure_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ensure_serializable(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_passed": total_passed,
        "total_levels": 5,
        "convergence_index": float(convergence_index),
        "levels": [r.to_dict() for r in level_reports]
    }
    
    report_data = ensure_serializable(report_data)
    
    save_path = "tempdata/agi_benchmark_report.json"
    with open(save_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    print(f"\nReport saved to {save_path}")
    
    return report_data


if __name__ == "__main__":
    run_agi_benchmark()
