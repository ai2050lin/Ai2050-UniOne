
import torch
import os
import random

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tempdata", "logic_mix")
os.makedirs(DATA_DIR, exist_ok=True)

def generate_transitivity(num_samples=50000, chain_len=3):
    """
    Generates chains like: A > B, B > C, therefore A > C.
    ents: A, B, C, D, E...
    """
    entities = [chr(ord('A') + i) for i in range(26)]
    samples = []
    
    for _ in range(num_samples):
        # Pick random subset of entities
        chain_ents = random.sample(entities, chain_len)
        # They are implicitly ordered: chain_ents[0] > chain_ents[1] > ...
        
        # Create premises
        premises = []
        for i in range(len(chain_ents)-1):
            premises.append(f"{chain_ents[i]} > {chain_ents[i+1]}")
        
        # Shuffle premises to force true logical integration, not pattern matching order
        random.shuffle(premises)
        
        premise_str = ". ".join(premises) + "."
        
        # Create query
        # A > C ? Yes
        # C > A ? No
        
        i, j = sorted(random.sample(range(len(chain_ents)), 2))
        # chain_ents[i] is effectively 'greater' than chain_ents[j] because i < j in list index but we defined 0 > 1 > 2
        # Wait, let's define 0 > 1 > 2. So i < j implies ent[i] > ent[j].
        
        qa_pair = []
        
        # Case 1: Ask if Higher > Lower (True)
        q1 = f"{chain_ents[i]} > {chain_ents[j]}?"
        a1 = "Yes"
        qa_pair.append((q1, a1))
        
        # Case 2: Ask if Lower > Higher (False)
        q2 = f"{chain_ents[j]} > {chain_ents[i]}?"
        a2 = "No"
        qa_pair.append((q2, a2))
        
        q, a = random.choice(qa_pair)
        
        text = f"Context: {premise_str} Question: {q} Answer: {a}"
        samples.append(text)
        
    return samples

def generate_modular_arithmetic(num_samples=50000, modulus=997):
    """
    x + y (mod p)
    format: "Calc: x + y (mod p) = z"
    """
    samples = []
    for _ in range(num_samples):
        x = random.randint(0, modulus-1)
        y = random.randint(0, modulus-1)
        res = (x + y) % modulus
        text = f"Calc: {x} + {y} (mod {modulus}) = {res}"
        samples.append(text)
    return samples

# Simple Custom Tokenizer to keeping IDs small and dense for arithmetic
class LogicTokenizer:
    def __init__(self):
        # Build vocab
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+->=?.(): "
        self.stoi = {c:i+1 for i,c in enumerate(chars)}
        self.itos = {i+1:c for i,c in enumerate(chars)}
        self.vocab_size = len(chars) + 2 # +1 for padding/unk
        self.pad_token = 0
        
    def encode(self, text):
        return [self.stoi.get(c, 0) for c in text]
    
    def decode(self, ids):
        return "".join([self.itos.get(i, '') for i in ids])

def main():
    print("[*] Generating LogicMix-V1...")
    
    # 1. Transitivity
    print("    Generating Transitivity (A > B)...")
    trans_data = generate_transitivity(50000, 3)
    
    # 2. Modulo
    print("    Generating Modular Arithmetic (Z997)...")
    mod_data = generate_modular_arithmetic(50000, 997)
    
    all_data = trans_data + mod_data
    random.shuffle(all_data)
    
    print(f"    Total Samples: {len(all_data)}")
    print(f"    Example: {all_data[0]}")
    
    tokenizer = LogicTokenizer()
    
    # Tokenize
    print("[*] Tokenizing...")
    all_ids = []
    for txt in all_data:
        ids = tokenizer.encode(txt)
        # Add EOS (let's say vocab_size-1 is EOS?)
        # For simplicity, just concat with a space or distinct separator if needed.
        # Actually standard practice is just concat.
        encoded = torch.tensor(ids, dtype=torch.long)
        all_ids.append(encoded)
        
    # Concat into one big stream for training or keep as samples?
    # For geometric analysis, samples are often better, but for scaling, stream.
    # Let's do stream.
    full_stream = torch.cat(all_ids)
    print(f"    Total Tokens: {full_stream.numel()}")
    
    # Split Train/Val
    split_idx = int(0.9 * len(full_stream))
    train_data = full_stream[:split_idx]
    val_data = full_stream[split_idx:]
    
    torch.save(train_data, os.path.join(DATA_DIR, "train.pt"))
    torch.save(val_data, os.path.join(DATA_DIR, "val.pt"))
    print(f"[+] Saved to {DATA_DIR}")

if __name__ == "__main__":
    main()
