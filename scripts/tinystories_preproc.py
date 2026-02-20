
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tempdata", "tinystories")
os.makedirs(DATA_DIR, exist_ok=True)

def main():
    print("[*] Loading TinyStories (streaming)...")
    try:
        # Load train split streaming
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    except Exception as e:
        print(f"[-] Load failed: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("[*] Collecting & Tokenizing Training Data (~200MB)...")
    
    # Target size: Approx 200MB of text to match WikiText-103 scale
    MAX_CHARS = 200 * 1024 * 1024 
    train_ids = []
    
    current_chars = 0
    buffer_text = []

    for i, x in enumerate(ds):
        text = x['text']
        buffer_text.append(text)
        current_chars += len(text)
        
        # Process every 10MB chunk
        if current_chars >= 10 * 1024 * 1024: 
            chunk = "\n<|endoftext|>\n".join(buffer_text)
            tokens = tokenizer(chunk, return_tensors='pt', verbose=False).input_ids
            train_ids.append(tokens)
            
            print(f"    Processed chunk {len(train_ids)} ({current_chars/1e6:.1f} MB total)")
            buffer_text = []
            
            # Stop if total collected exceeds target
            total_collected = sum([t.numel() for t in train_ids]) * 4 # approx bytes (if utf8 char ~ 1 token, actually 1 token ~ 4 chars)
            # Actually just use raw char count for stopping
            if sum([len(t[0]) for t in buffer_text]) + sum([t.numel() for t in train_ids])*3 > MAX_CHARS:
               break
               
        if len(train_ids) > 20: # 20 chunks of ~10MB = 200MB
            break
            
    if buffer_text:
        chunk = "\n<|endoftext|>\n".join(buffer_text)
        tokens = tokenizer(chunk, return_tensors='pt', verbose=False).input_ids
        train_ids.append(tokens)

    train_ids = torch.cat(train_ids, dim=1)
    print(f"[+] Collected Train Tokens: {train_ids.numel() / 1e6:.2f}M")
    
    # Validation
    print("[*] Collecting Validation Data...")
    ds_val = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    val_text = []
    v_count = 0
    for x in ds_val:
        val_text.append(x['text'])
        v_count += len(x['text'])
        if v_count > 5 * 1024 * 1024: break
    
    val_text = "\n<|endoftext|>\n".join(val_text)
    val_ids = tokenizer(val_text, return_tensors='pt').input_ids
    print(f"[+] Collected Val Tokens: {val_ids.numel() / 1e6:.2f}M")

    torch.save(train_ids, os.path.join(DATA_DIR, "train.pt"))
    torch.save(val_ids, os.path.join(DATA_DIR, "val.pt"))
    print(f"[+] Dataset saved to {DATA_DIR}")

if __name__ == "__main__":
    main()
