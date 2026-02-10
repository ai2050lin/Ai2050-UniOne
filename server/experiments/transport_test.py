import os

# Set environment variables for model loading
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import plotly.graph_objects as go
import torch
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

from transformer_lens import HookedTransformer

# Configuration
MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_IDX = 6 # Middle layer of GPT-2 small
D_MODEL = 768

def load_model():
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()
    return model

def get_tangent_basis_jacobian(model, prompt, target_token_idx, layer_idx):
    """
    Estimates the tangent space at a specific point (prompt + token)
    using the Jacobian of the layer output with respect to the input embedding.
    
    Note: This is computationally expensive and is a simplified approximation.
    For a transformer ensuring we capture the local variation valid for the specific input.
    """
    # This involves complex gradient computation. 
    # Simplified approach: Perturbation analysis.
    # Add small noise to the residual stream at the target layer input, 
    # and observe the output direction variations.
    pass

def get_tangent_basis_activations(model, prompts, target_token_idx, layer_idx, n_components=4):
    """
    Estimates tangent space using PCA on a cloud of similar activations.
    """
    print(f"Collecting activations from layer {layer_idx} for {len(prompts)} prompts...")
    
    activations = []
    
    for prompt in prompts:
        _, cache = model.run_with_cache(prompt)
        # Resid post is the output of the layer (including residual)
        # We might want the output of the MLP or Attention block specifically, 
        # but resid_post is the 'state' of the stream.
        act = cache[f"blocks.{layer_idx}.hook_resid_post"][0, target_token_idx, :].detach().cpu().numpy()
        activations.append(act)
        
    activations = np.array(activations)
    
    # Center the data
    mean_act = np.mean(activations, axis=0)
    centered_act = activations - mean_act
    
    # PCA to find principal directions (Tangent Basis)
    pca = PCA(n_components=n_components)
    pca.fit(centered_act)
    
    # The components are the basis vectors
    basis = pca.components_ # [n_components, d_model]
    
    print(f"Tangent basis shape: {basis.shape}")
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return mean_act, basis

def compute_parallel_transport(basis_src, basis_tgt):
    """
    Computes the optimal transport matrix (rotation) between two tangent spaces
    using Orthogonal Procrustes problem solution.
    
    Finds R such that | basis_src @ R - basis_tgt | is minimized.
    Actually we want to map vector v_src to v_tgt.
    Coefficients c st v_src = c @ basis_src.
    v_tgt = c @ basis_tgt? Only if bases are aligned.
    
    We need to align the bases first.
    R, scale = orthogonal_procrustes(basis_src.T, basis_tgt.T)
    
    Transport Matrix T = R.
    v_transported = v_src @ T
    """
    # Align the subspaces
    # scipy.linalg.orthogonal_procrustes solves min ||A @ R - B||_F
    # Inputs should be (M, N). We want to map basis vectors.
    # Dimensions: basis is [K, D].
    # We want to align the K basis vectors in D-dim space.
    
    R_matrix, scale = orthogonal_procrustes(basis_src, basis_tgt)
    # R is [D, D] ? No, procrustes expects same shape.
    # If basis is [K, D], it finds R [D, D] such that basis_src @ R ~= basis_tgt
    
    print(f"Procrustes scale: {scale}")
    return R_matrix, scale

def main():
    model = load_model()
    
    # 1. Define two 'points' on the manifold
    # Context A: Medical context for "Doctor"
    prompts_nurse = [
        "The nurse said that", "The nurse thought that", "The nurse walked into the room and",
        "A nurse was checking the", "The young nurse was"
    ]
    
    # Context B: Medical context for "Doctor"
    prompts_doctor = [
        "The doctor said that", "The doctor thought that", "The doctor walked into the room and",
        "A doctor was checking the", "The young doctor was"
    ]
    
    target_idx = 1 # "nurse"/"doctor" is the subject (index 1)
    
    # 2. Estimate Tangent Spaces
    print("Estimating Tangent Space for 'Nurse'...")
    mean_nurse, basis_nurse = get_tangent_basis_activations(model, prompts_nurse, target_idx, LAYER_IDX)
    
    print("Estimating Tangent Space for 'Doctor'...")
    mean_doctor, basis_doctor = get_tangent_basis_activations(model, prompts_doctor, target_idx, LAYER_IDX)
    
    # 3. Compute Transport
    # We want to transport a vector from Nurse space to Doctor space.
    # Vector: 'Gender' direction.
    # Ideally we find the gender direction in the Nurse space.
    
    # For simplicitly, let's take the difference between "The nurse said" and "The male nurse said"?
    # Or just use the global gender direction and project it.
    
    # Let's perform the transport calculation
    transport_matrix, scale = compute_parallel_transport(basis_nurse, basis_doctor)
    
    # 4. Verify Transport
    # Does basis_nurse @ T close to basis_doctor?
    transported_basis = basis_nurse @ transport_matrix
    diff = np.linalg.norm(transported_basis - basis_doctor)
    print(f"Basis transport difference (Frobenius norm): {diff:.4f}")
    
    # 5. Application: Transport a vector
    # Construct a dummy vector in Nurse space (e.g. 1st principal component)
    v_nurse = basis_nurse[0]
    v_transported = v_nurse @ transport_matrix
    
    # Compare with 1st PC of doctor
    sim = np.dot(v_transported, basis_doctor[0]) / (np.linalg.norm(v_transported) * np.linalg.norm(basis_doctor[0]))
    
    print("\n--- FINAL VERIFICATION METRICS ---")
    print(f"1. Frobenius Norm of Basis Difference: {diff:.6f}")
    print(f"2. Cosine Similarity (Nurse PC1 -> Doctor PC1): {sim:.6f}")
    print(f"3. Procrustes Scale: {scale:.6f}")
    print("----------------------------------\n")
    
    print("Draft implementation complete. Ready for rigorous testing.")

if __name__ == "__main__":
    main()
