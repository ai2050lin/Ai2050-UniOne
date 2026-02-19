# -*- coding: utf-8 -*-
"""
S8 High-Dimensional Group Structure Training System (Phase VII)
==============================================================

Target: Train model to understand 40,320 order symmetric group S8 structure

Strategy:
1. Sparse sampling: Don't train all 40,320 elements, sample representative subset
2. Cayley table learning: Learn the structure of group multiplication table
3. Generalization test: Test model performance on unseen elements

Mathematical basis:
- S8 = {pi | pi is a permutation of {1,...,8}}
- |S8| = 8! = 40,320
- Group operation: pi1 * pi2 = pi1(pi2)

Author: AGI Research Team
Date: 2026-02-19
"""

import os
import sys
import random
import time
from itertools import permutations
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class S8Config:
    """S8 training configuration"""
    vocab_size: int = 40320 + 10  # 40,320 group elements + special tokens
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    max_seq_len: int = 32
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 50
    sample_ratio: float = 0.1  # Sample 10% of group elements for training
    
    # Special tokens
    PAD_TOKEN: int = 0
    BOS_TOKEN: int = 1
    EOS_TOKEN: int = 2
    MASK_TOKEN: int = 3


class PermutationUtils:
    """Permutation group utility class"""
    
    @staticmethod
    def permutation_to_tuple(perm: List[int]) -> Tuple[int, ...]:
        """Convert permutation list to tuple"""
        return tuple(perm)
    
    @staticmethod
    def tuple_to_index(perm_tuple: Tuple[int, ...]) -> int:
        """Convert permutation tuple to unique index (Lehmer code)"""
        n = len(perm_tuple)
        index = 0
        remaining = list(range(1, n + 1))
        
        for i, val in enumerate(perm_tuple):
            pos = remaining.index(val)
            index += pos * np.math.factorial(n - 1 - i)
            remaining.pop(pos)
        
        return index + 10  # Offset for special tokens
    
    @staticmethod
    def index_to_tuple(index: int, n: int = 8) -> Tuple[int, ...]:
        """Convert index back to permutation tuple"""
        index = index - 10  # Remove offset
        remaining = list(range(1, n + 1))
        result = []
        
        for i in range(n):
            fact = np.math.factorial(n - 1 - i)
            pos = index // fact
            index = index % fact
            result.append(remaining.pop(pos))
        
        return tuple(result)
    
    @staticmethod
    def compose(perm1: Tuple[int, ...], perm2: Tuple[int, ...]) -> Tuple[int, ...]:
        """Group operation: perm1 * perm2"""
        n = len(perm1)
        result = []
        for i in range(n):
            result.append(perm1[perm2[i] - 1])
        return tuple(result)
    
    @staticmethod
    def inverse(perm: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute inverse element"""
        n = len(perm)
        inv = [0] * n
        for i, val in enumerate(perm):
            inv[val - 1] = i + 1
        return tuple(inv)
    
    @staticmethod
    def generate_random_permutations(n: int = 8, count: int = 1000, seed: int = 42) -> List[Tuple[int, ...]]:
        """Generate random permutation samples"""
        random.seed(seed)
        perms = []
        base = list(range(1, n + 1))
        for _ in range(count):
            perm = base.copy()
            random.shuffle(perm)
            perms.append(tuple(perm))
        return perms


class S8GroupDataset(Dataset):
    """S8 group dataset"""
    
    def __init__(self, config: S8Config, mode: str = 'train'):
        self.config = config
        self.mode = mode
        
        # Sample group elements
        self.sample_size = int(config.sample_ratio * 40320)
        self.sampled_perms = PermutationUtils.generate_random_permutations(
            n=8, count=self.sample_size, seed=42 if mode == 'train' else 123
        )
        
        # Precompute partial Cayley table
        self.cayley_table = self._build_partial_cayley_table()
        
        # Generate training samples
        self.samples = self._generate_samples()
        
    def _build_partial_cayley_table(self) -> Dict[Tuple[int, int], int]:
        """Build partial Cayley table"""
        table = {}
        for i, perm1 in enumerate(self.sampled_perms):
            for j, perm2 in enumerate(self.sampled_perms):
                result = PermutationUtils.compose(perm1, perm2)
                result_idx = PermutationUtils.tuple_to_index(result)
                idx1 = PermutationUtils.tuple_to_index(perm1)
                idx2 = PermutationUtils.tuple_to_index(perm2)
                table[(idx1, idx2)] = result_idx
        return table
    
    def _generate_samples(self) -> List[Tuple[List[int], int]]:
        """Generate training samples
        
        Task: Given two permutations, predict their product
        Input: [BOS, perm1_token, perm2_token]
        Output: result_token
        """
        samples = []
        
        for (idx1, idx2), result in self.cayley_table.items():
            input_seq = [self.config.BOS_TOKEN, idx1, idx2, self.config.EOS_TOKEN]
            samples.append((input_seq, result))
        
        random.shuffle(samples)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_seq, target = self.samples[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class S8AttentionBlock(nn.Module):
    """S8 group structure aware attention block"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class S8GroupModel(nn.Module):
    """S8 group learning model"""
    
    def __init__(self, config: S8Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, config.max_seq_len, config.d_model) * 0.02)
        
        self.layers = nn.ModuleList([
            S8AttentionBlock(config.d_model, config.n_heads)
            for _ in range(config.n_layers)
        ])
        
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Special initialization: use geometric priors of group structure"""
        with torch.no_grad():
            self.embedding.weight[:10].normal_(0, 0.02)
            
            for i in range(10, self.config.vocab_size):
                phase = (i - 10) / 40320 * 2 * np.pi
                self.embedding.weight[i, 0] = np.cos(phase)
                self.embedding.weight[i, 1] = np.sin(phase)
                self.embedding.weight[i, 2:].normal_(0, 0.01)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_norm(x)
        logits = self.output_proj(x)
        
        return logits


class S8Trainer:
    """S8 trainer"""
    
    def __init__(self, config: S8Config, device: str = 'auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.config = config
        
        self.model = S8GroupModel(config).to(device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for input_seq, target in pbar:
            input_seq = input_seq.to(self.device)
            target = target.to(self.device)
            
            logits = self.model(input_seq)
            pred_logits = logits[:, -2, :]
            
            loss = self.criterion(pred_logits, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = pred_logits.argmax(dim=-1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total
            })
        
        return total_loss / len(dataloader), correct / total
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for input_seq, target in dataloader:
                input_seq = input_seq.to(self.device)
                target = target.to(self.device)
                
                logits = self.model(input_seq)
                pred_logits = logits[:, -2, :]
                
                loss = self.criterion(pred_logits, target)
                
                total_loss += loss.item()
                pred = pred_logits.argmax(dim=-1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        return total_loss / len(dataloader), correct / total
    
    def test_generalization(self, test_size: int = 100):
        """Test generalization ability on unseen permutations"""
        self.model.eval()
        
        test_perms = PermutationUtils.generate_random_permutations(n=8, count=test_size, seed=999)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for perm1 in test_perms[:50]:
                for perm2 in test_perms[:50]:
                    result = PermutationUtils.compose(perm1, perm2)
                    
                    idx1 = PermutationUtils.tuple_to_index(perm1)
                    idx2 = PermutationUtils.tuple_to_index(perm2)
                    
                    input_seq = torch.tensor([[self.config.BOS_TOKEN, idx1, idx2]], device=self.device)
                    logits = self.model(input_seq)
                    pred = logits[0, -2, :].argmax().item()
                    
                    target = PermutationUtils.tuple_to_index(result)
                    if pred == target:
                        correct += 1
                    total += 1
        
        return correct / total if total > 0 else 0
    
    def train(self, train_dataset, val_dataset=None):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
        
        print(f"\n{'='*60}")
        print(f"S8 Group Training (|S8| = 40,320)")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"{'='*60}\n")
        
        best_acc = 0
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            val_loss, val_acc = 0, 0
            if val_loader:
                val_loss, val_acc = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            if val_loader:
                print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            if train_acc > best_acc:
                best_acc = train_acc
                os.makedirs('tempdata', exist_ok=True)
                torch.save(self.model.state_dict(), 'tempdata/s8_model_best.pth')
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Best training accuracy: {best_acc:.4f}")
        
        return self.history


def run_s8_benchmark():
    """Run S8 benchmark"""
    print("="*60)
    print("S8 High-Dimensional Group Structure Training")
    print("Phase VII: Scaling to 40,320 elements")
    print("="*60)
    
    config = S8Config(
        d_model=128,  # Reduced model size
        n_heads=4,
        n_layers=4,
        epochs=10,  # Quick test
        sample_ratio=0.01,  # Sample 1% = ~400 elements
        batch_size=32
    )
    
    print("\nGenerating S8 dataset...")
    train_dataset = S8GroupDataset(config, mode='train')
    val_dataset = S8GroupDataset(config, mode='val')
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    trainer = S8Trainer(config)
    history = trainer.train(train_dataset, val_dataset)
    
    print("\n" + "="*60)
    print("Generalization Test on Unseen Permutations")
    print("="*60)
    gen_acc = trainer.test_generalization()
    print(f"Generalization accuracy: {gen_acc:.4f}")
    
    results = {
        'config': {
            'd_model': config.d_model,
            'n_heads': config.n_heads,
            'n_layers': config.n_layers,
            'epochs': config.epochs,
            'sample_ratio': config.sample_ratio
        },
        'history': history,
        'generalization_accuracy': gen_acc
    }
    
    os.makedirs('tempdata', exist_ok=True)
    import json
    with open('tempdata/s8_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to tempdata/s8_training_results.json")
    
    return results


if __name__ == "__main__":
    run_s8_benchmark()
