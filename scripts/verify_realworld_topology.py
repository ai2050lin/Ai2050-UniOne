import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure concepts to test
CONCEPTS = {
    "week_en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "month_en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    "week_zh": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"],
    "month_zh": ["一月", "二月", "三月", "四月", "五月", "六月", "七月", "八月", "九月", "十月", "十一月", "十二月"],
    "musical_notes": ["C", "D", "E", "F", "G", "A", "B"], # Simplified scale
    "season": ["Spring", "Summer", "Autumn", "Winter"]
}


def load_model_and_tokenizer(model_path, model_type="auto"):
    print(f"Loading {model_path}...")
    try:
        if "gpt2" in model_path.lower():
            # GPT-2 is standard, no need for trust_remote_code usually
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu") # Load on CPU first to be safe
        else:
            # Qwen needs trust_remote_code
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype="auto")
            
        return model, tokenizer
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def get_concept_embeddings(model, tokenizer, concept_list):
    embeddings = []
    valid_concepts = []
    
    # Get the input embedding layer
    # For GPT-2 it's transformer.wte, for Qwen/Llama it's usually model.embed_tokens
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"): # GPT-2
        embed_layer = model.transformer.wte
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"): # Qwen/Llama
        embed_layer = model.model.embed_tokens
    elif hasattr(model, "get_input_embeddings"):
        embed_layer = model.get_input_embeddings()
    else:
        print("Could not locate embedding layer.")
        return None, None

    print(f"Embedding Layer found: {embed_layer}")

    for word in concept_list:
        # GPT-2 often needs space
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) == 0:
             ids = tokenizer.encode(word, add_special_tokens=False)
        
        if len(ids) == 0:
            continue
            
        token_id = ids[0] 
        decoded = tokenizer.decode([token_id])
        
        with torch.no_grad():
            if hasattr(embed_layer, "weight"): 
                emb = embed_layer.weight[token_id].detach().cpu().numpy()
            else: 
                idx_tensor = torch.tensor([token_id]).to(model.device)
                emb = embed_layer(idx_tensor).squeeze().detach().cpu().numpy()
        
        embeddings.append(emb)
        valid_concepts.append(word + f"({decoded})")
    
    return np.array(embeddings), valid_concepts

def analyze_topology(embeddings, concepts, title, save_path):
    if len(embeddings) < 3:
        print(f"Not enough points to analyze topology for {title}")
        return

    # PCA to 2D
    pca = PCA(n_components=2)
    proj = pca.fit_transform(embeddings)
    
    # Calculate "Circularity Score"
    # Simple metric: distance from centroid vs standard deviation of radius
    centroid = np.mean(proj, axis=0)
    radii = np.linalg.norm(proj - centroid, axis=1)
    radius_std = np.std(radii) / np.mean(radii) # Coefficient of variation of radius. Logic: Circle has constant radius.
    
    # Calculate "Sequentiality Score"
    # D(i, i+1) should be smaller than D(i, i+k)
    dists = []
    for i in range(len(proj)-1):
        dists.append(np.linalg.norm(proj[i] - proj[i+1]))
    avg_step = np.mean(dists)
    
    # Closing the loop dist
    loop_gap = np.linalg.norm(proj[-1] - proj[0])
    
    print(f"Analysis for {title}:")
    print(f"  Radius Variation (Lower is more circular): {radius_std:.4f}")
    print(f"  Avg Step Dist: {avg_step:.4f}")
    print(f"  Loop Gap: {loop_gap:.4f} (Ratio to step: {loop_gap/avg_step:.2f})")

    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(proj[:, 0], proj[:, 1], c=np.arange(len(concepts)), cmap='hsv', s=100, edgecolors='k')
    
    for i, txt in enumerate(concepts):
        plt.text(proj[i, 0]+0.02, proj[i, 1]+0.02, txt, fontsize=9)
        
        # Draw lines between sequential points
        if i < len(concepts) - 1:
            plt.plot([proj[i, 0], proj[i+1, 0]], [proj[i, 1], proj[i+1, 1]], 'k-', alpha=0.3)
            
    # Connect last to first to see loop
    plt.plot([proj[-1, 0], proj[0, 0]], [proj[-1, 1], proj[0, 1]], 'k--', alpha=0.3, label='Loop Closure')

    plt.title(f"Topology of '{title}'\n(Radius Var: {radius_std:.2f}, Loop Ratio: {loop_gap/avg_step:.2f})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

def main():
    save_dir = "tempdata/realworld_topology"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    # 1. Test GPT-2
    print("\n=== Testing GPT-2 ===")
    # Absolute path to local snapshot
    gpt2_path = r"D:\develop\model\hub\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
    print(f"Loading from {gpt2_path}...")
    gpt2_model, gpt2_tok = load_model_and_tokenizer(gpt2_path)
    
    if gpt2_model:
        for key in ["week_en", "month_en", "season"]:
            emb, valid = get_concept_embeddings(gpt2_model, gpt2_tok, CONCEPTS[key])
            if emb is not None:
                analyze_topology(emb, valid, f"GPT2_{key}", f"{save_dir}/gpt2_{key}.png")
        del gpt2_model
        del gpt2_tok
        torch.cuda.empty_cache()

    # 2. Test Qwen3
    print("\n=== Testing Qwen3 ===")
    # Absolute path to local snapshot
    qwen_path = r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"
    print(f"Loading from {qwen_path}...")

    # Use AutoModel abstraction (which handles Qwen via trust_remote_code=True in our helper)
    qwen_model, qwen_tok = load_model_and_tokenizer(qwen_path, "qwen")
    
    if qwen_model:
        for key in CONCEPTS.keys():
            # Skip English specific ones if we want, but Qwen knows English.
            # We are interested in week_zh, month_zh
            emb, valid = get_concept_embeddings(qwen_model, qwen_tok, CONCEPTS[key])
            if emb is not None:
                analyze_topology(emb, valid, f"Qwen_{key}", f"{save_dir}/qwen_{key}.png")

if __name__ == "__main__":
    main()
