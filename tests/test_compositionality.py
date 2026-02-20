
import os
import sys
import unittest

import torch

# Add parent directory to path to import structure_analyzer (now in server/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

from structure_analyzer import CompositionalAnalysis


class MockModel:
    def __init__(self, d_model=64):
        self.d_model = d_model
        
    def to_tokens(self, text):
        return torch.tensor([1, 2, 3]) # Dummy tokens
        
    def run_with_cache(self, tokens):
        # Return dummy cache
        batch_size = 1
        seq_len = len(tokens)
        
        # Create perfect additive data for testing
        # Black: [1, 0, ...]
        # Cat: [0, 1, ...]
        # Black Cat: [1, 1, ...]
        
        cache = {
            "blocks.0.hook_resid_post": torch.zeros(batch_size, seq_len, self.d_model)
        }
        
        # Hacky way to return different vectors based on 'text'
        # But here we don't know the text, only tokens.
        # So we will rely on the test to patch the cache function or just return random
        return None, cache

class TestCompositionality(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.analyzer = CompositionalAnalysis(self.model)
        
    def test_perfect_compositionality(self):
        # Override get_token_activation to return controlled vectors
        def mock_get_activation(text, layer_idx):
            if text == "A": return torch.tensor([1.0, 0.0, 0.0])
            if text == "B": return torch.tensor([0.0, 1.0, 0.0])
            if text == "AB": return torch.tensor([1.0, 1.0, 0.0]) # Perfect addition
            return torch.zeros(3)
            
        # Monkey patch
        self.analyzer.get_token_activation = mock_get_activation
        
        phrases = [("A", "B", "AB")]
        result = self.analyzer.analyze_compositionality(phrases, layer_idx=0)
        
        print("Perfect Case Result:", result)
        self.assertAlmostEqual(result['r2_score'], 1.0, places=5)
        self.assertAlmostEqual(result['cosine_similarity'], 1.0, places=5)

    def test_noise_compositionality(self):
        # Use more samples to make the system overdetermined (N > 2*d)
        # d_model = 3, so inputs have dim 6. We need > 6 samples.
        np_random = torch.randn
        
        def mock_get_activation(text, layer_idx):
            # Parse text "A_i", "B_i", "AB_i"
            prefix, idx = text.split('_')
            i = int(idx)
            torch.manual_seed(i) # Deterministic per sample
            v1 = torch.randn(3)
            torch.manual_seed(i + 1000)
            v2 = torch.randn(3)
            
            if prefix == "A": return v1
            if prefix == "B": return v2
            if prefix == "AB":
                # Non-linear relation: v1 * v2 + noise
                return v1 * v2 + torch.randn(3) * 0.5
            return torch.zeros(3)
            
        self.analyzer.get_token_activation = mock_get_activation
        
        # Generate 10 samples
        phrases = [(f"A_{i}", f"B_{i}", f"AB_{i}") for i in range(10)]
        result = self.analyzer.analyze_compositionality(phrases, layer_idx=0)
        
        print("Noisy Case Result:", result)
        # R2 should be low (definitely not 1.0)
        self.assertTrue(result['r2_score'] < 0.9)

if __name__ == '__main__':
    unittest.main()
