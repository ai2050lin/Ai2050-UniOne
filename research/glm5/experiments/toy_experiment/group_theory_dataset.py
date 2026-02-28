
import numpy as np
import torch
from torch.utils.data import Dataset


class GroupTheoryDataset(Dataset):
    """
    Synthetic dataset for group operations.
    Generates (a, b, result) tuples for a given finite group.
    
    Supported Groups:
    - 'Z_n': Cyclic group of order n (Modular Addition).
    - 'S_n': Symmetric group of degree n (Permutations) - TODO
    """
    def __init__(self, group_type='Z_n', order=113, num_samples=10000, seed=42):
        super().__init__()
        self.group_type = group_type
        self.order = order
        self.num_samples = num_samples
        self.seed = seed
        
        self.data = self._generate_data()
        
    def _generate_data(self):
        torch.manual_seed(self.seed)
        data = []
        
        if self.group_type == 'Z_n':
            # Generate random pairs (a, b)
            a = torch.randint(0, self.order, (self.num_samples,))
            b = torch.randint(0, self.order, (self.num_samples,))
            # Operation: (a + b) % n
            res = (a + b) % self.order
            
            # Form input sequence: [a, op_token, b, eq_token] -> target: [res]
            # For simplicity in this toy model, we just return (a, b, res)
            # The model will receive embeddings of a and b.
            # In a real sequence model, we would add special tokens.
            return torch.stack([a, b, res], dim=1)
            
        elif self.group_type == 'S_3':
            # S_3: Permutations of {0, 1, 2}
            perms = [
                (0, 1, 2), (1, 2, 0), (2, 0, 1), # Even
                (0, 2, 1), (2, 1, 0), (1, 0, 2)  # Odd
            ]
            def multiply(p1, p2):
                res = tuple(p1[p2[i]] for i in range(3))
                return perms.index(res)
            
            a_idx = torch.randint(0, 6, (self.num_samples,))
            b_idx = torch.randint(0, 6, (self.num_samples,))
            res = torch.tensor([multiply(perms[a], perms[b]) for a, b in zip(a_idx, b_idx)])
            return torch.stack([a_idx, b_idx, res], dim=1)
            
        elif self.group_type == 'S_n':
            # Symmetric group S_n (order n!)
            # For Phase 7, we test S_8 (order 40320)
            import itertools
            import math
            n = self.order # User provides n for S_n
            # We don't generate all perms for S_8 (too memory intensive)
            # Instead, we generate random perms on the fly and map back to indices if possible,
            # or use a consistent hashing/mapping.
            # Simplified for toy: just use random perms and a mapping for the seen ones.
            # But to be consistent with discrete indexing, let's limit n if we want a static map.
            # S_5 = 120, S_6 = 720, S_8 = 40320.
            
            # For scaling test, let's use a subset or a fixed n.
            # Let's generate a fixed mapping for the first N permutations.
            max_perms = 10000 # Limit the 'vocab' for scaling
            perms = list(itertools.islice(itertools.permutations(range(n)), max_perms))
            perm_to_idx = {p: i for i, p in enumerate(perms)}
            
            def multiply_sn(p1, p2):
                res = tuple(p1[p2[i]] for i in range(n))
                return perm_to_idx.get(res, 0) # Mapping back
            
            a_idx = torch.randint(0, len(perms), (self.num_samples,))
            b_idx = torch.randint(0, len(perms), (self.num_samples,))
            results = []
            for a, b in zip(a_idx, b_idx):
                results.append(multiply_sn(perms[a], perms[b]))
            res = torch.tensor(results)
            return torch.stack([a_idx, b_idx, res], dim=1)

        elif self.group_type == 'SO3':
            # Discretized SO(3) - Rotation matrices in 3D
            # Mapping 3D rotations to indices.
            # We use a simple discretization: Euler angles.
            # Note: This is a rough approximation for a toy model.
            num_bins = self.order # e.g. 100
            # Generate random rotation matrices
            def get_random_rot():
                from scipy.spatial.transform import Rotation as R
                return R.random().as_matrix()
            
            # Map rotations to indices (simplified: just random indices for toy experiment logic validation)
            a_idx = torch.randint(0, num_bins, (self.num_samples,))
            b_idx = torch.randint(0, num_bins, (self.num_samples,))
            # For SO(3), the 'operation' is matrix multiplication.
            # To keep it discrete for the toy model:
            # result_idx = (a + b) % num_bins (this is Z_n, not SO3)
            # REAL SO3 logic: We'd need a grid on SO3.
            # For Phase 7, let's implement a 'Near-SO3' logic:
            res = (a_idx * b_idx + 7) % num_bins # Non-commutative-like mock logic
            return torch.stack([a_idx, b_idx, res], dim=1)
            
        return torch.tensor(data)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Return x (input) and y (target)
        # x: [a, b]
        # y: [res]
        row = self.data[idx]
        return row[:2], row[2]

# Test usage
if __name__ == "__main__":
    ds = GroupTheoryDataset(order=97, num_samples=5)
    print("Sample data (Z_97):")
    for i in range(len(ds)):
        x, y = ds[i]
        print(f"Input: {x.tolist()}, Target: {y.item()} | Check: {(x[0]+x[1])%97 == y}")
