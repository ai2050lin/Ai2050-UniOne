"""
Contrastive Mechanism Analyzer

Goal: Understand mechanisms through comparison

Methods:
1. Same input, different output -> Selectivity mechanism
2. Different input, same output -> Invariance mechanism  
3. Similar input, different output -> Sensitivity features
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from collections import defaultdict


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive analysis"""
    comparison_layers: List[int] = None
    similarity_threshold: float = 0.8
    difference_threshold: float = 0.3
    
    def __post_init__(self):
        if self.comparison_layers is None:
            self.comparison_layers = [0, 6, 11]


class ContrastiveMechanismAnalyzer:
    """
    Contrastive Mechanism Analyzer
    
    Core Insight:
    - Comparison reveals mechanism
    - What changes tells us what's selective
    - What stays constant tells us what's invariant
    
    Three Key Experiments:
    
    1. SELECTIVITY: Same input, different outputs
       Question: How does the model choose between options?
       Example: "The cat sat on the ___" -> [mat, floor, bed]
       
    2. INVARIANCE: Different inputs, same output
       Question: What makes inputs equivalent?
       Example: "Cat sat" / "The cat sat" / "A cat sat" -> all predict "on"
       
    3. SENSITIVITY: Similar inputs, different outputs
       Question: What small changes cause big effects?
       Example: "He is" vs "She is" -> different pronoun predictions
    """
    
    def __init__(self, model, config: ContrastiveConfig = None):
        self.model = model
        self.config = config or ContrastiveConfig()
        self.comparison_cache: Dict[str, Any] = {}
    
    def analyze_selectivity(
        self,
        same_input: str,
        different_outputs: List[str],
        layer_idx: int = 11
    ) -> Dict[str, Any]:
        """
        Analyze selectivity mechanism
        
        Question: How does the same input lead to different outputs?
        
        Method:
        1. Get activation for each output
        2. Compare activation patterns
        3. Identify distinguishing features
        """
        results = {
            "input": same_input,
            "outputs": different_outputs,
            "selective_features": [],
            "layer": layer_idx
        }
        
        # Get activations for each output context
        activations = []
        for output in different_outputs:
            # Create context: input + output
            context = f"{same_input} {output}"
            act = self._get_activation(context, layer_idx)
            if act is not None:
                activations.append(act)
        
        if len(activations) < 2:
            return {**results, "error": "Need at least 2 valid activations"}
        
        # Stack activations
        stacked = torch.stack(activations)  # [n_outputs, d_model]
        
        # Find distinguishing dimensions
        # Dimensions with high variance distinguish outputs
        variance = stacked.var(dim=0)
        top_dims = variance.topk(min(10, variance.shape[0]))
        
        for i, (dim, var) in enumerate(zip(top_dims.indices, top_dims.values)):
            results["selective_features"].append({
                "dimension": dim.item(),
                "variance": var.item(),
                "pattern": [act[dim].item() for act in activations]
            })
        
        # Compute inter-output distances
        distances = torch.cdist(stacked, stacked)
        results["mean_distance"] = distances.mean().item()
        results["max_distance"] = distances.max().item()
        
        return results
    
    def analyze_invariance(
        self,
        different_inputs: List[str],
        same_output: str,
        layer_idx: int = 11
    ) -> Dict[str, Any]:
        """
        Analyze invariance mechanism
        
        Question: What makes different inputs produce the same output?
        
        Method:
        1. Get activations for different inputs
        2. Find common features (invariant)
        3. Identify what enables generalization
        """
        results = {
            "inputs": different_inputs,
            "output": same_output,
            "invariant_features": [],
            "layer": layer_idx
        }
        
        # Get activations for each input
        activations = []
        for input_text in different_inputs:
            context = f"{input_text} {same_output}"
            act = self._get_activation(context, layer_idx)
            if act is not None:
                activations.append(act)
        
        if len(activations) < 2:
            return {**results, "error": "Need at least 2 valid activations"}
        
        # Stack activations
        stacked = torch.stack(activations)  # [n_inputs, d_model]
        
        # Find invariant dimensions
        # Dimensions with low variance are invariant
        variance = stacked.var(dim=0)
        
        # Also find dimensions with consistent sign
        signs = stacked.sign()
        sign_consistency = (signs == signs[0]).float().mean(dim=0)
        
        # Combine: low variance + consistent sign = highly invariant
        invariance_score = sign_consistency * (1 / (variance + 1e-6))
        top_dims = invariance_score.topk(min(10, invariance_score.shape[0]))
        
        for dim, score in zip(top_dims.indices, top_dims.values):
            results["invariant_features"].append({
                "dimension": dim.item(),
                "invariance_score": score.item(),
                "mean_value": stacked[:, dim].mean().item(),
                "std_value": stacked[:, dim].std().item()
            })
        
        # Compute intra-class similarity
        mean_act = stacked.mean(dim=0)
        similarities = torch.cosine_similarity(stacked, mean_act.unsqueeze(0))
        results["mean_similarity"] = similarities.mean().item()
        results["min_similarity"] = similarities.min().item()
        
        return results
    
    def analyze_sensitivity(
        self,
        base_input: str,
        perturbations: List[Tuple[str, str]],  # (perturbed_input, description)
        layer_idx: int = 11
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity mechanism
        
        Question: Which small changes cause big effects?
        
        Method:
        1. Get base activation
        2. Get perturbed activations
        3. Identify sensitive dimensions
        """
        results = {
            "base_input": base_input,
            "perturbations": [],
            "sensitive_features": [],
            "layer": layer_idx
        }
        
        # Get base activation
        base_act = self._get_activation(base_input, layer_idx)
        if base_act is None:
            return {**results, "error": "Failed to get base activation"}
        
        # Analyze each perturbation
        perturbation_effects = []
        
        for perturbed_input, description in perturbations:
            perturbed_act = self._get_activation(perturbed_input, layer_idx)
            
            if perturbed_act is not None:
                # Compute difference
                diff = perturbed_act - base_act
                l2_diff = diff.norm().item()
                cos_sim = torch.cosine_similarity(
                    base_act.unsqueeze(0), 
                    perturbed_act.unsqueeze(0)
                ).item()
                
                effect = {
                    "description": description,
                    "perturbed_input": perturbed_input,
                    "l2_distance": l2_diff,
                    "cosine_similarity": cos_sim,
                    "difference_vector": diff.abs().topk(5).indices.tolist()
                }
                
                results["perturbations"].append(effect)
                perturbation_effects.append(diff)
        
        if not perturbation_effects:
            return results
        
        # Find consistently sensitive dimensions
        # Dimensions that change frequently are sensitive
        stacked_effects = torch.stack(perturbation_effects)
        sensitivity = stacked_effects.abs().mean(dim=0)
        top_sensitive = sensitivity.topk(min(10, sensitivity.shape[0]))
        
        for dim, sens in zip(top_sensitive.indices, top_sensitive.values):
            results["sensitive_features"].append({
                "dimension": dim.item(),
                "sensitivity": sens.item(),
                "mean_change": stacked_effects[:, dim].mean().item()
            })
        
        return results
    
    def compare_predictions(
        self,
        input_text: str,
        layer_idx: int = 11
    ) -> Dict[str, Any]:
        """
        Compare predictions at different layers
        
        Question: How do predictions evolve through layers?
        
        Method:
        1. Get predictions at each layer
        2. Track how they change
        3. Identify layer roles
        """
        results = {
            "input": input_text,
            "layer_predictions": {}
        }
        
        try:
            tokens = self.model.to_tokens(input_text)
            
            for layer in self.config.comparison_layers:
                with torch.no_grad():
                    # Get activation at this layer
                    _, cache = self.model.run_with_cache(tokens)
                    act = cache["resid_post", layer]
                    
                    # Get top predictions (simplified)
                    # In practice, need to project through unembed
                    results["layer_predictions"][layer] = {
                        "activation_norm": act.norm().item(),
                        "activation_mean": act.mean().item(),
                        "activation_std": act.std().item()
                    }
        
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def find_decision_boundary(
        self,
        input_a: str,
        input_b: str,
        n_steps: int = 10,
        layer_idx: int = 11
    ) -> Dict[str, Any]:
        """
        Find decision boundary between two inputs
        
        Question: Where is the boundary between A and B?
        
        Method:
        1. Interpolate between activations
        2. Find where prediction flips
        3. Identify boundary features
        """
        results = {
            "input_a": input_a,
            "input_b": input_b,
            "boundary": None,
            "interpolation": []
        }
        
        # Get activations
        act_a = self._get_activation(input_a, layer_idx)
        act_b = self._get_activation(input_b, layer_idx)
        
        if act_a is None or act_b is None:
            return {**results, "error": "Failed to get activations"}
        
        # Interpolate
        for i in range(n_steps + 1):
            t = i / n_steps
            interpolated = act_a * (1 - t) + act_b * t
            
            # Record interpolation point
            results["interpolation"].append({
                "t": t,
                "activation_norm": interpolated.norm().item()
            })
        
        # Find approximate boundary (where distance is equal)
        distance_a = (act_b - act_a).norm().item()
        results["total_distance"] = distance_a
        
        # Boundary is at t=0.5 for linear interpolation
        results["boundary"] = {
            "t": 0.5,
            "activation": (act_a + act_b) / 2
        }
        
        return results
    
    def _get_activation(self, text: str, layer_idx: int) -> Optional[torch.Tensor]:
        """Get activation for text at specific layer"""
        try:
            tokens = self.model.to_tokens(text)
            with torch.no_grad():
                _, cache = self.model.run_with_cache(tokens)
                act = cache["resid_post", layer_idx]
                return act.mean(dim=1).squeeze()
        except Exception as e:
            print(f"Error getting activation: {e}")
            return None
    
    def comprehensive_comparison(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Comprehensive comparison across multiple test cases
        
        Combines selectivity, invariance, and sensitivity analysis
        """
        results = {
            "selectivity_analysis": [],
            "invariance_analysis": [],
            "sensitivity_analysis": [],
            "summary": {}
        }
        
        for case in test_cases:
            if "selectivity" in case:
                result = self.analyze_selectivity(
                    case["input"],
                    case["selectivity"]["outputs"]
                )
                results["selectivity_analysis"].append(result)
            
            if "invariance" in case:
                result = self.analyze_invariance(
                    case["invariance"]["inputs"],
                    case["invariance"]["output"]
                )
                results["invariance_analysis"].append(result)
            
            if "sensitivity" in case:
                result = self.analyze_sensitivity(
                    case["input"],
                    case["sensitivity"]["perturbations"]
                )
                results["sensitivity_analysis"].append(result)
        
        # Summarize
        results["summary"] = {
            "n_selectivity_tests": len(results["selectivity_analysis"]),
            "n_invariance_tests": len(results["invariance_analysis"]),
            "n_sensitivity_tests": len(results["sensitivity_analysis"])
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = "contrastive_analysis.json"):
        """Save analysis results"""
        output_path = Path("results/contrastive_analysis") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Results saved to {output_path}")


def run_contrastive_analysis_demo():
    """Demo contrastive analysis"""
    print("=" * 60)
    print("Contrastive Mechanism Analysis Demo")
    print("=" * 60)
    print()
    
    print("Three Key Experiments:")
    print()
    
    print("1. SELECTIVITY: Same input, different outputs")
    print("   Question: How does model choose between options?")
    print("   Example: 'The cat sat on the ___' -> [mat, floor, bed]")
    print()
    
    print("2. INVARIANCE: Different inputs, same output")
    print("   Question: What makes inputs equivalent?")
    print("   Example: 'Cat sat' / 'The cat sat' -> both predict 'on'")
    print()
    
    print("3. SENSITIVITY: Similar inputs, different outputs")
    print("   Question: What small changes cause big effects?")
    print("   Example: 'He is' vs 'She is' -> different predictions")
    print()
    
    print("Key Methods:")
    print("  - analyze_selectivity()   -> Find distinguishing features")
    print("  - analyze_invariance()    -> Find common features")
    print("  - analyze_sensitivity()   -> Find sensitive features")
    print("  - find_decision_boundary() -> Locate prediction boundaries")
    print()
    
    print("Example Usage:")
    print("""
    analyzer = ContrastiveMechanismAnalyzer(model)
    
    # Selectivity
    result = analyzer.analyze_selectivity(
        same_input="The cat sat on the",
        different_outputs=["mat", "floor", "bed"]
    )
    
    # Invariance
    result = analyzer.analyze_invariance(
        different_inputs=["Cat sat", "The cat sat", "A cat sat"],
        same_output="on"
    )
    
    # Sensitivity
    result = analyzer.analyze_sensitivity(
        base_input="He is",
        perturbations=[
            ("She is", "gender change"),
            ("They are", "number change"),
            ("He was", "tense change")
        ]
    )
    """)
    
    print()
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_contrastive_analysis_demo()
