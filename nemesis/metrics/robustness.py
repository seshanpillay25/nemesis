"""
Robustness Metrics 📊⚔️

"The true measure of strength is revealed in battle"

Core metrics for evaluating adversarial robustness and model performance
under various attack scenarios.
"""

import torch
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RobustnessScore:
    """Comprehensive robustness scoring."""
    overall_score: float
    clean_accuracy: float
    adversarial_accuracy: float
    robustness_ratio: float
    attack_success_rate: float
    perturbation_sensitivity: float
    confidence_degradation: float


class RobustnessMetrics:
    """
    Core robustness evaluation metrics.
    
    Provides standardized measurements for model robustness
    across different attack types and perturbation budgets.
    """
    
    def __init__(self):
        self.name = "Robustness Oracle"
        self.epsilon_range = [0.01, 0.03, 0.1, 0.3]
    
    def calculate_robustness_score(self, 
                                 clean_outputs: torch.Tensor,
                                 adversarial_outputs: torch.Tensor,
                                 true_labels: torch.Tensor,
                                 epsilon: float) -> RobustnessScore:
        """Calculate comprehensive robustness score."""
        
        # Clean accuracy
        clean_preds = torch.argmax(clean_outputs, dim=-1)
        clean_accuracy = (clean_preds == true_labels).float().mean().item()
        
        # Adversarial accuracy
        adv_preds = torch.argmax(adversarial_outputs, dim=-1)
        adversarial_accuracy = (adv_preds == true_labels).float().mean().item()
        
        # Robustness ratio
        robustness_ratio = adversarial_accuracy / max(clean_accuracy, 1e-8)
        
        # Attack success rate
        attack_success_rate = 1.0 - adversarial_accuracy
        
        # Perturbation sensitivity
        perturbation_sensitivity = self._calculate_perturbation_sensitivity(
            clean_outputs, adversarial_outputs, epsilon
        )
        
        # Confidence degradation
        confidence_degradation = self._calculate_confidence_degradation(
            clean_outputs, adversarial_outputs
        )
        
        # Overall score (weighted combination)
        overall_score = (
            0.4 * robustness_ratio +
            0.3 * (1.0 - perturbation_sensitivity) +
            0.3 * (1.0 - confidence_degradation)
        )
        
        return RobustnessScore(
            overall_score=overall_score,
            clean_accuracy=clean_accuracy,
            adversarial_accuracy=adversarial_accuracy,
            robustness_ratio=robustness_ratio,
            attack_success_rate=attack_success_rate,
            perturbation_sensitivity=perturbation_sensitivity,
            confidence_degradation=confidence_degradation
        )
    
    def _calculate_perturbation_sensitivity(self,
                                          clean_outputs: torch.Tensor,
                                          adversarial_outputs: torch.Tensor,
                                          epsilon: float) -> float:
        """Calculate sensitivity to perturbations."""
        # L2 distance between output distributions
        output_distance = torch.norm(
            torch.softmax(clean_outputs, dim=-1) - torch.softmax(adversarial_outputs, dim=-1),
            p=2, dim=-1
        ).mean().item()
        
        # Normalize by epsilon
        sensitivity = output_distance / epsilon
        return min(sensitivity, 1.0)
    
    def _calculate_confidence_degradation(self,
                                        clean_outputs: torch.Tensor,
                                        adversarial_outputs: torch.Tensor) -> float:
        """Calculate confidence degradation under attack."""
        clean_confidence = torch.softmax(clean_outputs, dim=-1).max(dim=-1)[0].mean().item()
        adv_confidence = torch.softmax(adversarial_outputs, dim=-1).max(dim=-1)[0].mean().item()
        
        degradation = (clean_confidence - adv_confidence) / max(clean_confidence, 1e-8)
        return max(0.0, degradation)
    
    def evaluate_epsilon_robustness(self,
                                  model: torch.nn.Module,
                                  test_data: List[Tuple[torch.Tensor, int]],
                                  attack_function,
                                  epsilon_values: Optional[List[float]] = None) -> Dict[float, RobustnessScore]:
        """Evaluate robustness across multiple epsilon values."""
        if epsilon_values is None:
            epsilon_values = self.epsilon_range
        
        results = {}
        
        for epsilon in epsilon_values:
            clean_outputs = []
            adversarial_outputs = []
            true_labels = []
            
            for x, y in test_data:
                # Clean prediction
                with torch.no_grad():
                    clean_out = model(x)
                    clean_outputs.append(clean_out)
                
                # Adversarial prediction
                try:
                    adv_result = attack_function(x, epsilon=epsilon)
                    adv_input = adv_result.adversarial_input if adv_result.success else x
                    
                    with torch.no_grad():
                        adv_out = model(adv_input)
                        adversarial_outputs.append(adv_out)
                except:
                    # If attack fails, use clean input
                    adversarial_outputs.append(clean_out)
                
                true_labels.append(y)
            
            # Convert to tensors
            clean_outputs = torch.cat(clean_outputs, dim=0)
            adversarial_outputs = torch.cat(adversarial_outputs, dim=0)
            true_labels = torch.tensor(true_labels)
            
            # Calculate robustness score
            score = self.calculate_robustness_score(
                clean_outputs, adversarial_outputs, true_labels, epsilon
            )
            results[epsilon] = score
        
        return results


class AttackMetrics:
    """
    Metrics for evaluating attack effectiveness.
    """
    
    def __init__(self):
        self.name = "Attack Analyzer"
    
    def calculate_attack_success_rate(self, attack_results: List) -> float:
        """Calculate overall attack success rate."""
        if not attack_results:
            return 0.0
        
        successful_attacks = sum(1 for result in attack_results if result.success)
        return successful_attacks / len(attack_results)
    
    def calculate_perturbation_magnitude(self, 
                                       original_inputs: List[torch.Tensor],
                                       adversarial_inputs: List[torch.Tensor],
                                       norm_type: str = 'l2') -> Dict[str, float]:
        """Calculate perturbation magnitudes."""
        perturbations = []
        
        for orig, adv in zip(original_inputs, adversarial_inputs):
            if norm_type == 'l2':
                pert = torch.norm(adv - orig, p=2).item()
            elif norm_type == 'linf':
                pert = torch.norm(adv - orig, p=float('inf')).item()
            elif norm_type == 'l1':
                pert = torch.norm(adv - orig, p=1).item()
            else:
                pert = torch.norm(adv - orig, p=2).item()
            
            perturbations.append(pert)
        
        return {
            'mean': np.mean(perturbations),
            'std': np.std(perturbations),
            'min': np.min(perturbations),
            'max': np.max(perturbations),
            'median': np.median(perturbations)
        }
    
    def calculate_query_efficiency(self, attack_results: List) -> Dict[str, float]:
        """Calculate query efficiency metrics."""
        queries = [result.queries_used for result in attack_results if hasattr(result, 'queries_used')]
        
        if not queries:
            return {'mean': 0, 'median': 0, 'success_rate': 0}
        
        successful_queries = [q for result, q in zip(attack_results, queries) if result.success]
        
        return {
            'mean_queries': np.mean(queries),
            'median_queries': np.median(queries),
            'mean_successful_queries': np.mean(successful_queries) if successful_queries else 0,
            'query_efficiency': len(successful_queries) / len(queries) if queries else 0
        }


class DefenseMetrics:
    """
    Metrics for evaluating defense effectiveness.
    """
    
    def __init__(self):
        self.name = "Defense Evaluator"
    
    def calculate_defense_effectiveness(self,
                                     clean_attack_results: List,
                                     defended_attack_results: List) -> Dict[str, float]:
        """Calculate defense effectiveness metrics."""
        
        # Success rates
        clean_success_rate = sum(1 for r in clean_attack_results if r.success) / len(clean_attack_results)
        defended_success_rate = sum(1 for r in defended_attack_results if r.success) / len(defended_attack_results)
        
        # Defense improvement
        attack_reduction = max(0, clean_success_rate - defended_success_rate)
        relative_improvement = attack_reduction / max(clean_success_rate, 1e-8)
        
        # Robustness gain
        robustness_gain = (1 - defended_success_rate) - (1 - clean_success_rate)
        
        return {
            'clean_attack_success_rate': clean_success_rate,
            'defended_attack_success_rate': defended_success_rate,
            'attack_reduction': attack_reduction,
            'relative_improvement': relative_improvement,
            'robustness_gain': robustness_gain,
            'defense_effectiveness': relative_improvement
        }
    
    def calculate_computational_overhead(self,
                                       clean_inference_times: List[float],
                                       defended_inference_times: List[float]) -> Dict[str, float]:
        """Calculate computational overhead of defenses."""
        
        clean_mean = np.mean(clean_inference_times)
        defended_mean = np.mean(defended_inference_times)
        
        overhead_absolute = defended_mean - clean_mean
        overhead_relative = overhead_absolute / clean_mean if clean_mean > 0 else 0
        
        return {
            'clean_inference_time': clean_mean,
            'defended_inference_time': defended_mean,
            'absolute_overhead': overhead_absolute,
            'relative_overhead': overhead_relative,
            'slowdown_factor': defended_mean / clean_mean if clean_mean > 0 else 1.0
        }
    
    def calculate_utility_preservation(self,
                                     clean_accuracy: float,
                                     defended_accuracy: float) -> Dict[str, float]:
        """Calculate how well defense preserves model utility."""
        
        accuracy_drop = max(0, clean_accuracy - defended_accuracy)
        utility_preservation = defended_accuracy / max(clean_accuracy, 1e-8)
        
        return {
            'clean_accuracy': clean_accuracy,
            'defended_accuracy': defended_accuracy,
            'accuracy_drop': accuracy_drop,
            'utility_preservation': utility_preservation,
            'acceptable_trade_off': accuracy_drop < 0.05  # Less than 5% drop considered acceptable
        }