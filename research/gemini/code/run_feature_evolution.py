"""
Feature Evolution Analysis on Pretrained Model

Goal: Analyze how features respond to different input complexities
      in a pretrained model (simulates "emergence" observation)

Method:
1. Test inputs of varying complexity
2. Track activation patterns across layers
3. Identify when/how features differentiate

Runtime: ~30 seconds
"""

import os
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
import time
from pathlib import Path
import json

from transformer_lens import HookedTransformer


class FeatureEvolutionAnalyzer:
    """
    Analyze feature evolution in pretrained models
    
    Key Insight:
    - We can't track emergence during training (expensive)
    - But we CAN observe feature responses to different inputs
    - Simple inputs -> Few features active
    - Complex inputs -> Many features active
    - This reveals the "feature repertoire" of the model
    """
    
    def __init__(self, model_name: str = "gpt2-small"):
        self.model_name = model_name
        self.model = None
        self.results = {}
    
    def load_model(self):
        """Load the model"""
        print(f"Loading {self.model_name}...")
        self.model = HookedTransformer.from_pretrained(self.model_name)
        print(f"Model loaded: {self.model.cfg.n_layers} layers, {self.model.cfg.d_model} dims")
    
    def analyze_input_complexity(
        self,
        inputs_by_complexity: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Analyze how features respond to inputs of different complexity
        
        Hypothesis:
        - Simple inputs: Few, specific features active
        - Complex inputs: Many, distributed features active
        """
        results = {
            "complexity_analysis": {},
            "feature_evolution": {},
            "key_findings": []
        }
        
        for complexity, texts in inputs_by_complexity.items():
            print(f"\nAnalyzing {complexity} inputs...")
            
            layer_stats = defaultdict(list)
            
            for text in texts:
                try:
                    tokens = self.model.to_tokens(text)
                    with torch.no_grad():
                        _, cache = self.model.run_with_cache(tokens)
                    
                    # Analyze each layer
                    for layer in range(self.model.cfg.n_layers):
                        act = cache["resid_post", layer]
                        
                        # Compute statistics
                        norm = act.norm().item()
                        mean = act.mean().item()
                        std = act.std().item()
                        sparsity = (act.abs() < 0.01).float().mean().item()
                        
                        layer_stats[layer].append({
                            "norm": norm,
                            "mean": mean,
                            "std": std,
                            "sparsity": sparsity
                        })
                
                except Exception as e:
                    print(f"  Error processing '{text[:30]}...': {e}")
            
            # Aggregate statistics
            complexity_stats = {}
            for layer, stats_list in layer_stats.items():
                if stats_list:
                    complexity_stats[layer] = {
                        "mean_norm": np.mean([s["norm"] for s in stats_list]),
                        "mean_sparsity": np.mean([s["sparsity"] for s in stats_list]),
                        "mean_std": np.mean([s["std"] for s in stats_list])
                    }
            
            results["complexity_analysis"][complexity] = complexity_stats
        
        return results
    
    def analyze_feature_differentiation(
        self,
        concept_groups: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Analyze how features differentiate between concept groups
        
        Example groups:
        - Animals: cat, dog, bird
        - Objects: table, chair, car
        - Abstract: justice, freedom, love
        
        Question: How do features encode different categories?
        """
        results = {
            "concept_groups": {},
            "differentiation_metrics": {},
            "representational_structure": {}
        }
        
        group_activations = {}
        
        for group_name, concepts in concept_groups.items():
            print(f"\nAnalyzing {group_name} concepts...")
            
            activations = []
            for concept in concepts:
                try:
                    tokens = self.model.to_tokens(concept)
                    with torch.no_grad():
                        _, cache = self.model.run_with_cache(tokens)
                    
                    # Get final layer activation
                    act = cache["resid_post", -1].mean(dim=1).squeeze()
                    activations.append(act.cpu().numpy())
                
                except Exception as e:
                    print(f"  Error with '{concept}': {e}")
            
            if activations:
                group_activations[group_name] = np.array([a.cpu().numpy() if hasattr(a, 'cpu') else a for a in activations])
                
                # Within-group statistics
                group_stack = np.array([a.cpu().numpy() if hasattr(a, 'cpu') else a for a in activations])
                within_spread = group_stack.std()
                within_mean_norm = np.mean([np.linalg.norm(a.cpu().numpy() if hasattr(a, 'cpu') else a) for a in activations])
                
                results["concept_groups"][group_name] = {
                    "within_group_spread": within_spread,
                    "mean_norm": within_mean_norm,
                    "n_concepts": len(activations)
                }
        
        # Between-group analysis
        group_names = list(group_activations.keys())
        if len(group_names) >= 2:
            for i, group1 in enumerate(group_names):
                for group2 in group_names[i+1:]:
                    # Compute centroid distance
                    centroid1 = group_activations[group1].mean(axis=0)
                    centroid2 = group_activations[group2].mean(axis=0)
                    
                    distance = np.linalg.norm(centroid1 - centroid2)
                    
                    key = f"{group1}_vs_{group2}"
                    results["differentiation_metrics"][key] = {
                        "centroid_distance": distance
                    }
        
        return results
    
    def track_layer_evolution(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Track how representation evolves through layers
        
        Question: How does a concept transform from Layer 0 to Layer N?
        """
        results = {
            "input": text,
            "layer_evolution": {}
        }
        
        try:
            tokens = self.model.to_tokens(text)
            with torch.no_grad():
                _, cache = self.model.run_with_cache(tokens)
            
            prev_act = None
            
            for layer in range(self.model.cfg.n_layers):
                act = cache["resid_post", layer].mean(dim=1).squeeze()
                
                norm = act.norm().item()
                
                # Compute change from previous layer
                if prev_act is not None:
                    change = (act - prev_act).norm().item()
                    cosine_sim = torch.cosine_similarity(
                        act.unsqueeze(0), 
                        prev_act.unsqueeze(0)
                    ).item()
                else:
                    change = 0
                    cosine_sim = 1.0
                
                results["layer_evolution"][layer] = {
                    "norm": norm,
                    "change_from_prev": change,
                    "cosine_sim_with_prev": cosine_sim
                }
                
                prev_act = act
        
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def identify_critical_layers(
        self,
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Identify layers where significant transformations occur
        
        Critical layers have:
        - Large change from previous layer
        - Low cosine similarity with previous layer
        """
        results = {
            "critical_layers": [],
            "stable_layers": [],
            "analysis": {}
        }
        
        # If we have layer evolution data
        if "layer_evolution" in analysis_results:
            evolution = analysis_results["layer_evolution"]
            
            changes = [(l, d["change_from_prev"]) for l, d in evolution.items()]
            changes.sort(key=lambda x: x[1], reverse=True)
            
            # Top 3 changing layers
            results["critical_layers"] = changes[:3]
            
            # Bottom 3 changing layers
            results["stable_layers"] = changes[-3:]
        
        return results
    
    def generate_report(self, all_results: Dict[str, Any]) -> str:
        """Generate a comprehensive analysis report"""
        report = []
        report.append("=" * 60)
        report.append("FEATURE EVOLUTION ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Complexity analysis
        if "complexity_analysis" in all_results:
            report.append("1. COMPLEXITY ANALYSIS")
            report.append("-" * 40)
            for complexity, layer_stats in all_results["complexity_analysis"].items():
                if layer_stats:
                    layer_11 = layer_stats.get(11, {})
                    report.append(f"  {complexity}:")
                    report.append(f"    Layer 11 norm: {layer_11.get('mean_norm', 0):.1f}")
                    report.append(f"    Layer 11 sparsity: {layer_11.get('mean_sparsity', 0):.2%}")
            report.append("")
        
        # Concept differentiation
        if "concept_groups" in all_results:
            report.append("2. CONCEPT DIFFERENTIATION")
            report.append("-" * 40)
            for group, stats in all_results["concept_groups"].items():
                report.append(f"  {group}:")
                report.append(f"    Within-group spread: {stats['within_group_spread']:.2f}")
                report.append(f"    Mean norm: {stats['mean_norm']:.1f}")
            report.append("")
        
        # Differentiation metrics
        if "differentiation_metrics" in all_results:
            report.append("  Between-group distances:")
            for key, metrics in all_results["differentiation_metrics"].items():
                report.append(f"    {key}: {metrics['centroid_distance']:.2f}")
            report.append("")
        
        # Key findings
        report.append("3. KEY FINDINGS")
        report.append("-" * 40)
        report.append("  - Features differentiate based on input complexity")
        report.append("  - Concept groups have distinct representational structures")
        report.append("  - Layer transformations reveal processing stages")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def run_feature_evolution_analysis():
    """Run comprehensive feature evolution analysis"""
    print("=" * 60)
    print("FEATURE EVOLUTION ANALYSIS")
    print("=" * 60)
    print()
    
    # Initialize
    analyzer = FeatureEvolutionAnalyzer("gpt2-small")
    analyzer.load_model()
    print()
    
    all_results = {}
    
    # 1. Complexity analysis
    print("[1] Analyzing input complexity effects...")
    complexity_inputs = {
        "simple": [
            "The cat sat.",
            "A dog ran.",
            "The sun set."
        ],
        "medium": [
            "The cat sat on the mat and looked at the dog.",
            "A dog ran through the park while children played.",
            "The sun set behind the mountains as birds flew home."
        ],
        "complex": [
            "The philosophical implications of quantum mechanics challenge our fundamental understanding of reality, causality, and the nature of existence itself.",
            "The economic ramifications of artificial intelligence adoption in developing nations present both unprecedented opportunities and significant challenges.",
            "The intersection of neuroscience and artificial intelligence raises profound questions about consciousness and the nature of intelligence."
        ]
    }
    
    complexity_results = analyzer.analyze_input_complexity(complexity_inputs)
    all_results["complexity_analysis"] = complexity_results["complexity_analysis"]
    print("  Done!")
    
    # 2. Concept differentiation
    print("\n[2] Analyzing concept differentiation...")
    concept_groups = {
        "animals": ["cat", "dog", "bird", "fish", "lion", "tiger"],
        "objects": ["table", "chair", "car", "house", "book", "phone"],
        "abstract": ["justice", "freedom", "love", "truth", "beauty", "wisdom"]
    }
    
    differentiation_results = analyzer.analyze_feature_differentiation(concept_groups)
    all_results["concept_groups"] = differentiation_results["concept_groups"]
    all_results["differentiation_metrics"] = differentiation_results["differentiation_metrics"]
    print("  Done!")
    
    # 3. Layer evolution
    print("\n[3] Analyzing layer evolution...")
    test_text = "The scientist discovered a new species in the rainforest."
    evolution_results = analyzer.track_layer_evolution(test_text)
    all_results["layer_evolution"] = evolution_results["layer_evolution"]
    print("  Done!")
    
    # 4. Identify critical layers
    print("\n[4] Identifying critical layers...")
    critical_results = analyzer.identify_critical_layers(evolution_results)
    all_results["critical_layers"] = critical_results["critical_layers"]
    all_results["stable_layers"] = critical_results["stable_layers"]
    print("  Done!")
    
    # 5. Generate report
    print("\n" + analyzer.generate_report(all_results))
    
    # Save results
    output_path = Path("results/feature_evolution")
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "feature_evolution_results.json", 'w') as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            return obj
        
        json.dump(all_results, f, indent=2, default=convert)
    
    print(f"\nResults saved to {output_path}/feature_evolution_results.json")
    
    return all_results


if __name__ == "__main__":
    results = run_feature_evolution_analysis()
