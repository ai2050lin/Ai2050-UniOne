import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lens import HookedTransformer


class FiberMemory(nn.Module):
    """
    Fast Lane: A differentiable Key-Value Memory.
    Acts as a dynamic 'Fiber' layer that can be written to instantly.
    """
    def __init__(self, d_model, max_size=1000):
        super().__init__()
        self.d_model = d_model
        self.max_size = max_size
        
        # Fixed projections for reading (Manifold -> Fiber query)
        # For one-shot demo without training, we want these to be Identity
        # In a real trained FiberNet, these would be learned to transform Manifold space to Fiber space
        self.W_q = nn.Identity() 
        self.W_v = nn.Identity()
        
        # The Memory Storage (Keys and Values)
        # These are buffers, not parameters, so they don't update via SGD
        self.register_buffer("keys", torch.zeros(max_size, d_model))
        self.register_buffer("values", torch.zeros(max_size, d_model))
        self.register_buffer("usage", torch.zeros(max_size)) # To track empty slots
        self.size = 0
        
    def write(self, key_vector, value_vector):
        """
        Instant Learning: Insert a new fact (Key->Value) into the fiber bundle.
        key_vector: The 'context' or 'question' (e.g. embedding of "Monarch of UK")
        value_vector: The 'fact' (e.g. embedding of "King Charles")
        """
        if self.size >= self.max_size:
            # Simple FIFO or replacement strategy could go here
            # For now, just overwrite circular
            idx = self.size % self.max_size
        else:
            idx = self.size
            
        with torch.no_grad():
            self.keys[idx] = key_vector.detach()
            self.values[idx] = value_vector.detach()
            self.usage[idx] = 1.0
            self.size += 1
            
    def read(self, hidden_state):
        """
        Inference: Query the memory.
        """
        # 1. Project hidden state to Query
        query = self.W_q(hidden_state) # [Batch, Seq, Dim]
        
        if self.size == 0:
            return torch.zeros_like(hidden_state)
            
        # 2. Attention against Memory Keys
        # active keys: [Size, Dim]
        active_k = self.keys[:self.size]
        active_v = self.values[:self.size]
        
        # Dot product attention
        # Q: [B, S, D], K: [M, D] -> Scores: [B, S, M]
        scores = torch.matmul(query, active_k.t()) / (self.d_model ** 0.5)
        
        # Softmax over memory slots
        attn = F.softmax(scores, dim=-1)
        
        # 3. Aggregate Values
        # [B, S, M] x [M, D] -> [B, S, D]
        retrieved = torch.matmul(attn, active_v)
        
        # 4. Project back to residual stream
        output = self.W_v(retrieved)
        
        return output

class FiberNet(nn.Module):
    """
    Hybrid Architecture:
    - Slow Lane: Pretrained Transformer (Manifold)
    - Fast Lane: FiberMemory (Specifics)
    """
    def __init__(self, base_model: HookedTransformer):
        super().__init__()
        self.base_model = base_model
        # Attach memory to the last layer to ensure the fact survives to logits
        # Layer 11 of 12 (0-indexed)
        self.memory_layer_idx = base_model.cfg.n_layers - 1
        self.fiber_memory = FiberMemory(base_model.cfg.d_model)
        
    def forward(self, input_text, return_type="logits"):
        # We need to hook into the base model to inject memory
        
        def memory_hook(resid_pre, hook):
            # resid_pre: [Batch, Seq, Dim]
            # 1. Read from Fiber Memory
            memory_out = self.fiber_memory.read(resid_pre)
            
            # 2. Add to stream (Residual connection)
            # We assume the memory helps complete the "Fiber" info
            return resid_pre + memory_out
            
        # Run with hook
        hook_name = f"blocks.{self.memory_layer_idx}.hook_resid_post"
        return self.base_model.run_with_hooks(
            input_text,
            fwd_hooks=[(hook_name, memory_hook)],
            return_type=return_type
        )
    
    def instant_learn(self, context_text, target_text):
        """
        Teach the model a new association instantly.
        """
        # 1. Get the 'Key' vector from the context (Slow lane processing)
        # We run the model up to the memory layer
        _, cache = self.base_model.run_with_cache(context_text)
        
        # Use the last token's activation at the memory layer as the Key
        key_act = cache[f"blocks.{self.memory_layer_idx}.hook_resid_post"][0, -1, :]
        
        # 2. Get the 'Value' vector
        # Ideally, this is the embedding of the target answer
        # For simplicity, we can use the model's own embedding of the target
        # Or run the target through the first few layers
        # Let's just use the Unembed weights? No, that's logits.
        # Let's use the embedding of the target word.
        target_tokens = self.base_model.to_tokens(target_text)
        # If multiple tokens, maybe average? Let's take the first token for demo.
        if target_tokens.shape[1] > 1: # [1, seq]
             # Handle BOS
             tgt_idx = target_tokens[0, 1] if target_tokens.shape[1] > 1 else target_tokens[0,0]
        else:
             tgt_idx = target_tokens[0,0]
             
        # Value = The output direction that produces this token.
        # Approximation: The Unembed direction for this token.
        # W_U: [d_model, d_vocab]
        # value_vector = W_U[:, token_id]
        value_vector = self.base_model.W_U[:, tgt_idx]
        
        # Scale up to ensure it competes with the existing residual stream
        # A heuristic is to match the average norm of the residual stream
        # or just multiply by a constant factor for the demo.
        value_vector = value_vector * 10.0
        
        # 3. Write to Memory
        self.fiber_memory.write(key_act, value_vector)
        print(f"FiberNet: Learned connection '{context_text}' -> '{target_text}'")

