
import torch
import os

# Copy Tokenizer from training script
class LogicTokenizer:
    def __init__(self):
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+->=?.(): "
        self.stoi = {c:i+1 for i,c in enumerate(chars)}
        self.itos = {i+1:c for i,c in enumerate(chars)}
        self.vocab_size = len(chars) + 2
        self.pad_token = 0
        
    def decode(self, ids):
        return "".join([self.itos.get(i, '') for i in ids])

def main():
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tempdata", "logic_mix")
    path = os.path.join(DATA_DIR, "train.pt")
    
    print(f"[*] Loading {path}...")
    data = torch.load(path)
    print(f"    Total Tokens: {len(data)}")
    
    tokenizer = LogicTokenizer()
    
    # Decode first 1000 chars
    print("[*] Sample Data (First 1000 chars):")
    print("-" * 40)
    sample = tokenizer.decode(data[:1000].tolist())
    print(sample)
    print("-" * 40)
    
    # Check for repetition
    print("[*] Checking for obvious repetition...")
    # Decode chunks
    for i in range(0, 5000, 1000):
        print(f"--- Chunk {i} ---")
        print(tokenizer.decode(data[i:i+100].tolist()))

if __name__ == "__main__":
    main()
