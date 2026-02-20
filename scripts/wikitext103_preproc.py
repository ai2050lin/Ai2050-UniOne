
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import requests, zipfile, io

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tempdata", "wikitext103")
os.makedirs(DATA_DIR, exist_ok=True)

def download_and_extract():
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
    print(f"[*] Downloading WikiText-103 from {url}...")
    try:
        r = requests.get(url, stream=True, verify=False)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(DATA_DIR)
        print("[+] Extracted successfully.")
        return True
    except Exception as e:
        print(f"[-] Download failed: {e}")
        return False

def main():
    # Check if extracted files exist
    train_path = os.path.join(DATA_DIR, "wikitext-103", "wiki.train.tokens")
    if not os.path.exists(train_path):
        if not download_and_extract():
            # Fallback to HF load_dataset (might fail if network issues)
            try:
                print("[*] Trying HF load_dataset...")
                ds = load_dataset("wikitext", "wikitext-103-v1")
                train_text = "\n".join(ds['train']['text'])
                val_text = "\n".join(ds['validation']['text'])
            except Exception as e:
                print(f"[-] HF Load failed: {e}")
                return
        else:
            # Load from text files
            print("[*] Loading from text files...")
            with open(train_path, 'r', encoding='utf-8') as f: train_text = f.read()
            with open(os.path.join(DATA_DIR, "wikitext-103", "wiki.valid.tokens"), 'r', encoding='utf-8') as f: val_text = f.read()
    else:
        print("[*] Loading from cached text files...")
        with open(train_path, 'r', encoding='utf-8') as f: train_text = f.read()
        with open(os.path.join(DATA_DIR, "wikitext-103", "wiki.valid.tokens"), 'r', encoding='utf-8') as f: val_text = f.read()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("[*] Tokenizing (this may take a while)...")
    # Tokenize in chunks to avoid OOM
    def tokenize_text(text):
        return tokenizer(text, return_tensors='pt', verbose=False).input_ids

    # Split into chunks of 1M chars
    chunk_size = 1000000
    train_ids = []
    print(f"    Processing {len(train_text)} chars...")
    for i in range(0, len(train_text), chunk_size):
        chunk = train_text[i:i+chunk_size]
        train_ids.append(tokenize_text(chunk))
        if i % (chunk_size*10) == 0: print(f"    Processed {i/len(train_text)*100:.1f}%")
        
    train_ids = torch.cat(train_ids, dim=1)
    
    val_ids = tokenize_text(val_text)
    
    print(f"Train tokens: {train_ids.numel() / 1e6:.2f}M")
    print(f"Val tokens: {val_ids.numel() / 1e6:.2f}M")
    
    torch.save(train_ids, os.path.join(DATA_DIR, "train.pt"))
    torch.save(val_ids, os.path.join(DATA_DIR, "val.pt"))
    print(f"[+] Saved to {DATA_DIR}")

if __name__ == "__main__":
    main()
