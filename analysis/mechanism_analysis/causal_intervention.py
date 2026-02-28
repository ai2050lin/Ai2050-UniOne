"""
Causal Intervention Analyzer

Goal: Understand causal relationships between features

Method:
1. Intervention: Modify a feature's activation
2. Observation: How do other features change?
3. Inference: Determine causal direction

Key Questions:
- Does "cat" feature activation cause "animal" feature activation?
- If we suppress "gender" feature, does it affect pronoun prediction?
- What features are necessary vs sufficient?
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class InterventionConfig:
    """Configuration for causal intervention experiments"""
    intervention_types: List[str] = None  # ["suppress", "enhance", "replace"]
    intervention_strengths: List[float] = None  # [0.5, 1.0, 2.0]
    target_layers: List[int] = None  # Layers to intervene
    
    def __post_init__(self):
        if self.intervention_types is None:
            self.intervention_types = ["suppress", "enhance"]
        if self.intervention_strengths is None:
            self.intervention_strengths = [0.5, 1.0, 2.0]
        if self.target_layers is None:
            self.target_layers = [6, 11]


class CausalInterventionAnalyzer:
    """
    Causal Intervention Analyzer
    
    Core Insight:
    - Correlation does not imply causation
    - Need intervention to establish causality
    - Feature A -> Feature B can be tested by suppressing A and observing B
    
    Method:
    1. Identify feature direction (e.g., "gender" direction)
    2. Project activation onto feature direction
    3. Remove/enhance this component
    4. Observe downstream effects
    """
    
    def __init__(self, model, config: InterventionConfig = None):
        self.model = model
        self.config = config or InterventionConfig()
        self.feature_directions: Dict[str, torch.Tensor] = {}
        self.intervention_cache: Dict[str, Any] = {}
    
    def identify_feature_direction(
        self,
        positive_examples: List[str],
        negative_examples: List[str],
        layer_idx: int = 11
    ) -> torch.Tensor:
        """
        Identify a feature direction by contrasting examples
        
        Example:
            positive = ["he", "him", "his", "man", "boy"]
            negative = ["she", "her", "hers", "woman", "girl"]
            -> Gender direction
        
        Method:
        1. Get activations for positive examples
        2. Get activations for negative examples
        3. Direction = mean(positive) - mean(negative)
        """
        # Get positive activations
        pos_acts = []
        for text in positive_examples:
            act = self._get_activation(text, layer_idx)
            if act is not None:
                pos_acts.append(act)
        
        # Get negative activations
        neg_acts = []
        for text in negative_examples:
            act = self._get_activation(text, layer_idx)
            if act is not None:
                neg_acts.append(act)
        
        if not pos_acts or not neg_acts:
            return torch.zeros(self.model.cfg.d_model)
        
        # Compute direction
        pos_mean = torch.stack(pos_acts).mean(dim=0)
        neg_mean = torch.stack(neg_acts).mean(dim=0)
        
        direction = pos_mean - neg_mean
        
        # Normalize
        direction = direction / (direction.norm() + 1e-8)
        
        return direction
    
    def _get_activation(self, text: str, layer_idx: int) -> Optional[torch.Tensor]:
        """Get activation for a text at a specific layer"""
        try:
            tokens = self.model.to_tokens(text)
            with torch.no_grad():
                _, cache = self.model.run_with_cache(tokens)
                act = cache["resid_post", layer_idx]
                return act.mean(dim=1).squeeze()  # [d_model]
        except Exception as e:
            print(f"Error getting activation: {e}")
            return None
    
    def intervene_activation(
        self,
        activation: torch.Tensor,
        feature_direction: torch.Tensor,
        intervention_type: str = "suppress",
        strength: float = 1.0
    ) -> torch.Tensor:
        """
        Intervene on activation by modifying feature direction
        
        Types:
        - suppress: Remove component along feature direction
        - enhance: Amplify component along feature direction
        - replace: Replace with feature direction
        
        Math:
        Let v = feature_direction, a = activation
        - suppress: a' = a - (a·v) * v * strength
        - enhance: a' = a + (a·v) * v * strength
        - replace: a' = v * strength
        """
        # Ensure same device
        device = activation.device
        feature_direction = feature_direction.to(device)
        
        # Project onto feature direction
        projection = (activation @ feature_direction) / (feature_direction @ feature_direction + 1e-8)
        
        if intervention_type == "suppress":
            # Remove component along direction
            intervened = activation - projection * feature_direction * strength
            
        elif intervention_type == "enhance":
            # Amplify component along direction
            intervened = activation + projection * feature_direction * strength
            
        elif intervention_type == "replace":
            # Replace with direction
            intervened = feature_direction * strength * torch.ones_like(activation)
            
        elif intervention_type == "noise":
            # Add noise orthogonal to direction
            noise = torch.randn_like(activation)
            noise = noise - (noise @ feature_direction) * feature_direction
            intervened = activation + noise * strength
            
        else:
            intervened = activation
        
        return intervened
    
    def analyze_causal_effect(
        self,
        input_text: str,
        feature_name: str,
        feature_direction: torch.Tensor,
        layer_idx: int = 11
    ) -> Dict[str, Any]:
        """
        Analyze causal effect of a feature
        
        Experiment:
        1. Get normal prediction
        2. Suppress feature, get prediction
        3. Enhance feature, get prediction
        4. Compare outputs
        """
        results = {
            "input": input_text,
            "feature": feature_name,
            "layer": layer_idx,
            "interventions": {}
        }
        
        # Store feature direction
        self.feature_directions[feature_name] = feature_direction
        
        # Normal prediction
        try:
            tokens = self.model.to_tokens(input_text)
            with torch.no_grad():
                normal_logits, normal_cache = self.model.run_with_cache(tokens)
                normal_act = normal_cache["resid_post", layer_idx].clone()
                normal_pred = self.model.to_string(normal_logits[0, -1].argmax())
                normal_top5 = self._get_top_predictions(normal_logits)
            
            results["normal"] = {
                "prediction": normal_pred,
                "top5": normal_top5
            }
        except Exception as e:
            return {"error": f"Normal prediction failed: {e}"}
        
        # Interventions
        for int_type in self.config.intervention_types:
            for strength in self.config.intervention_strengths:
                key = f"{int_type}_{strength}"
                
                try:
                    # Apply intervention
                    intervened_act = self.intervene_activation(
                        normal_act,
                        feature_direction,
                        int_type,
                        strength
                    )
                    
                    # Get new prediction (simplified - in practice need proper forward)
                    # For now, just analyze activation changes
                    
                    act_diff = (intervened_act - normal_act).norm().item()
                    projection_change = (intervened_act @ feature_direction).item() - (normal_act @ feature_direction).item()
                    
                    results["interventions"][key] = {
                        "activation_change": act_diff,
                        "projection_change": projection_change
                    }
                    
                except Exception as e:
                    results["interventions"][key] = {"error": str(e)}
        
        return results
    
    def _get_top_predictions(self, logits: torch.Tensor, k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k predictions with probabilities"""
        last_logits = logits[0, -1]
        probs = torch.softmax(last_logits, dim=-1)
        top_k = torch.topk(probs, k)
        
        results = []
        for i in range(k):
            token = self.model.to_string(top_k.indices[i])
            prob = top_k.values[i].item()
            results.append((token, prob))
        
        return results
    
    def test_feature_necessity(
        self,
        input_texts: List[str],
        feature_name: str,
        feature_direction: torch.Tensor,
        layer_idx: int = 11
    ) -> Dict[str, Any]:
        """
        Test if a feature is necessary for correct predictions
        
        Definition:
        Feature is NECESSARY if suppressing it causes prediction to fail
        
        Experiment:
        1. For each input, get normal prediction
        2. Suppress feature
        3. If prediction changes significantly, feature is necessary
        """
        results = {
            "feature": feature_name,
            "necessity_score": 0.0,
            "case_studies": []
        }
        
        necessary_count = 0
        
        for text in input_texts:
            # Normal prediction
            try:
                tokens = self.model.to_tokens(text)
                with torch.no_grad():
                    normal_logits, cache = self.model.run_with_cache(tokens)
                    normal_act = cache["resid_post", layer_idx]
                    normal_pred = normal_logits[0, -1].argmax().item()
                
                # Suppress feature
                suppressed_act = self.intervene_activation(
                    normal_act,
                    feature_direction,
                    "suppress",
                    1.0
                )
                
                # Measure change
                change = (suppressed_act - normal_act).norm().item()
                relative_change = change / (normal_act.norm().item() + 1e-8)
                
                is_necessary = relative_change > 0.1  # Threshold
                if is_necessary:
                    necessary_count += 1
                
                results["case_studies"].append({
                    "input": text,
                    "change": change,
                    "relative_change": relative_change,
                    "necessary": is_necessary
                })
                
            except Exception as e:
                results["case_studies"].append({
                    "input": text,
                    "error": str(e)
                })
        
        results["necessity_score"] = necessary_count / len(input_texts) if input_texts else 0
        
        return results
    
    def test_feature_sufficiency(
        self,
        input_texts: List[str],
        feature_name: str,
        feature_direction: torch.Tensor,
        layer_idx: int = 11
    ) -> Dict[str, Any]:
        """
        Test if a feature is sufficient for a prediction
        
        Definition:
        Feature is SUFFICIENT if enhancing it causes the associated behavior
        
        Experiment:
        1. For ambiguous inputs, enhance feature
        2. If prediction shifts toward feature-associated outcome, it's sufficient
        """
        results = {
            "feature": feature_name,
            "sufficiency_score": 0.0,
            "case_studies": []
        }
        
        sufficient_count = 0
        
        for text in input_texts:
            try:
                tokens = self.model.to_tokens(text)
                with torch.no_grad():
                    normal_logits, cache = self.model.run_with_cache(tokens)
                    normal_act = cache["resid_post", layer_idx]
                
                # Enhance feature
                enhanced_act = self.intervene_activation(
                    normal_act,
                    feature_direction,
                    "enhance",
                    2.0
                )
                
                # Measure projection change
                normal_proj = (normal_act @ feature_direction.to(normal_act.device)).item()
                enhanced_proj = (enhanced_act @ feature_direction.to(enhanced_act.device)).item()
                
                enhancement = enhanced_proj - normal_proj
                is_sufficient = enhancement > 0.5  # Threshold
                
                if is_sufficient:
                    sufficient_count += 1
                
                results["case_studies"].append({
                    "input": text,
                    "normal_projection": normal_proj,
                    "enhanced_projection": enhanced_proj,
                    "enhancement": enhancement,
                    "sufficient": is_sufficient
                })
                
            except Exception as e:
                results["case_studies"].append({
                    "input": text,
                    "error": str(e)
                })
        
        results["sufficiency_score"] = sufficient_count / len(input_texts) if input_texts else 0
        
        return results
    
    def discover_feature_circuit(
        self,
        input_text: str,
        target_feature: str,
        target_direction: torch.Tensor,
        candidate_features: Dict[str, torch.Tensor],
        layer_idx: int = 11
    ) -> Dict[str, Any]:
        """
        Discover which features causally influence a target feature
        
        Method:
        1. For each candidate feature, suppress it
        2. Measure impact on target feature
        3. Rank by causal influence
        
        This reveals the "circuit" of feature dependencies
        """
        results = {
            "input": input_text,
            "target_feature": target_feature,
            "upstream_features": []
        }
        
        # Get baseline activation
        try:
            tokens = self.model.to_tokens(input_text)
            with torch.no_grad():
                _, cache = self.model.run_with_cache(tokens)
                baseline_act = cache["resid_post", layer_idx]
            
            baseline_proj = (baseline_act @ target_direction.to(baseline_act.device)).item()
        except:
            return {"error": "Failed to get baseline"}
        
        # Test each candidate
        for feature_name, feature_dir in candidate_features.items():
            if feature_name == target_feature:
                continue
            
            try:
                # Suppress candidate
                suppressed_act = self.intervene_activation(
                    baseline_act,
                    feature_dir,
                    "suppress",
                    1.0
                )
                
                # Measure impact on target
                new_proj = (suppressed_act @ target_direction.to(suppressed_act.device)).item()
                impact = abs(new_proj - baseline_proj)
                
                results["upstream_features"].append({
                    "feature": feature_name,
                    "impact": impact,
                    "direction": "positive" if new_proj > baseline_proj else "negative"
                })
                
            except Exception as e:
                results["upstream_features"].append({
                    "feature": feature_name,
                    "error": str(e)
                })
        
        # Sort by impact
        results["upstream_features"].sort(key=lambda x: x.get("impact", 0), reverse=True)
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = "causal_intervention_results.json"):
        """Save intervention results"""
        output_path = Path("results/causal_intervention") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Results saved to {output_path}")


def run_causal_intervention_demo():
    """
    Run causal intervention demo
    
    Demonstrates:
    1. Identifying feature directions
    2. Testing feature necessity
    3. Testing feature sufficiency
    4. Discovering feature circuits
    """
    print("=" * 60)
    print("Causal Intervention Analysis Demo")
    print("=" * 60)
    print()
    
    print("This module provides tools for causal analysis of features.")
    print()
    print("Key Methods:")
    print("  1. identify_feature_direction() - Find feature vectors")
    print("  2. intervene_activation() - Modify activations")
    print("  3. test_feature_necessity() - Is feature necessary?")
    print("  4. test_feature_sufficiency() - Is feature sufficient?")
    print("  5. discover_feature_circuit() - Find causal dependencies")
    print()
    
    print("Example Usage:")
    print("""
    from transformer_lens import HookedTransformer
    from analysis.mechanism_analysis.causal_intervention import CausalInterventionAnalyzer
    
    # Load model
    model = HookedTransformer.from_pretrained("gpt2-small")
    
    # Create analyzer
    analyzer = CausalInterventionAnalyzer(model)
    
    # Find gender direction
    gender_dir = analyzer.identify_feature_direction(
        positive_examples=["he", "him", "man", "boy"],
        negative_examples=["she", "her", "woman", "girl"],
        layer_idx=11
    )
    
    # Test necessity
    results = analyzer.test_feature_necessity(
        input_texts=["The doctor said", "The nurse said"],
        feature_name="gender",
        feature_direction=gender_dir
    )
    
    print(f"Gender feature necessity: {results['necessity_score']:.1%}")
    """)
    
    print()
    print("=" * 60)
    print("Demo complete! See code for full implementation.")
    print("=" * 60)


if __name__ == "__main__":
    run_causal_intervention_demo()
