"""
Defense Utilities 🛡️🔧

"Forging the sacred armor against darkness"

Utilities for defense validation, protection analysis,
and defense effectiveness measurement.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import warnings


@dataclass
class DefenseQualityMetrics:
    """Metrics for defense quality assessment."""
    clean_accuracy_preservation: float
    adversarial_accuracy_improvement: float
    computational_overhead: float
    robustness_gain: float
    overall_quality_score: float


class DefenseValidator:
    """
    Validation utilities for adversarial defenses.
    
    Ensures defenses meet quality standards without
    excessive computational overhead or utility loss.
    """
    
    def __init__(self,
                 max_accuracy_drop: float = 0.05,
                 max_overhead_factor: float = 3.0,
                 min_robustness_gain: float = 0.1):
        self.max_accuracy_drop = max_accuracy_drop
        self.max_overhead_factor = max_overhead_factor
        self.min_robustness_gain = min_robustness_gain
    
    def validate_defense_effectiveness(self,
                                     clean_accuracy_before: float,
                                     clean_accuracy_after: float,
                                     adversarial_accuracy_before: float,
                                     adversarial_accuracy_after: float,
                                     inference_time_before: float,
                                     inference_time_after: float) -> Dict[str, bool]:
        """Validate defense effectiveness across multiple criteria."""
        
        validation_results = {
            'utility_preservation_valid': True,
            'robustness_improvement_valid': True,
            'efficiency_valid': True,
            'overall_valid': True
        }
        
        # Check utility preservation
        accuracy_drop = clean_accuracy_before - clean_accuracy_after
        if accuracy_drop > self.max_accuracy_drop:
            validation_results['utility_preservation_valid'] = False
        
        # Check robustness improvement
        robustness_gain = adversarial_accuracy_after - adversarial_accuracy_before
        if robustness_gain < self.min_robustness_gain:
            validation_results['robustness_improvement_valid'] = False
        
        # Check computational efficiency
        overhead_factor = inference_time_after / max(inference_time_before, 1e-8)
        if overhead_factor > self.max_overhead_factor:
            validation_results['efficiency_valid'] = False
        
        # Overall validation
        validation_results['overall_valid'] = all([
            validation_results['utility_preservation_valid'],
            validation_results['robustness_improvement_valid'],
            validation_results['efficiency_valid']
        ])
        
        return validation_results
    
    def calculate_defense_quality_score(self,
                                      clean_accuracy_before: float,
                                      clean_accuracy_after: float,
                                      adversarial_accuracy_before: float,
                                      adversarial_accuracy_after: float,
                                      inference_time_before: float,
                                      inference_time_after: float) -> DefenseQualityMetrics:
        """Calculate comprehensive defense quality metrics."""
        
        # Utility preservation (higher is better)
        accuracy_preservation = 1.0 - max(0, clean_accuracy_before - clean_accuracy_after)
        
        # Robustness improvement (higher is better)
        robustness_improvement = max(0, adversarial_accuracy_after - adversarial_accuracy_before)
        
        # Efficiency (lower overhead is better)
        overhead_factor = inference_time_after / max(inference_time_before, 1e-8)
        efficiency_score = 1.0 / max(overhead_factor, 1.0)
        
        # Robustness gain normalized
        robustness_gain = robustness_improvement / max(1.0 - adversarial_accuracy_before, 1e-8)
        
        # Overall quality score (weighted combination)
        overall_score = (
            0.4 * accuracy_preservation +
            0.4 * robustness_gain +
            0.2 * efficiency_score
        )
        
        return DefenseQualityMetrics(
            clean_accuracy_preservation=accuracy_preservation,
            adversarial_accuracy_improvement=robustness_improvement,
            computational_overhead=overhead_factor,
            robustness_gain=robustness_gain,
            overall_quality_score=overall_score
        )
    
    def validate_defense_batch(self,
                             defense_results: List[Dict],
                             threshold_score: float = 0.7) -> Dict[str, Any]:
        """Validate a batch of defense results."""
        
        if not defense_results:
            return {'batch_valid': False, 'error': 'Empty batch'}
        
        valid_count = 0
        quality_scores = []
        
        for result in defense_results:
            try:
                metrics = self.calculate_defense_quality_score(**result)
                quality_scores.append(metrics.overall_quality_score)
                
                if metrics.overall_quality_score >= threshold_score:
                    valid_count += 1
            except Exception as e:
                warnings.warn(f"Failed to validate defense result: {e}")
        
        return {
            'batch_valid': valid_count / len(defense_results) >= 0.8,
            'valid_defense_rate': valid_count / len(defense_results),
            'mean_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'total_defenses': len(defense_results)
        }


class ProtectionAnalyzer:
    """
    Analyzer for protection mechanisms and their effectiveness.
    
    Provides detailed analysis of how defenses affect
    different types of attacks and input patterns.
    """
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_protection_coverage(self,
                                  defense_function: Callable,
                                  test_inputs: List[torch.Tensor],
                                  attack_functions: List[Callable],
                                  epsilon: float = 0.1) -> Dict[str, Any]:
        """Analyze protection coverage across different attacks."""
        
        coverage_results = {
            'attack_coverage': {},
            'input_coverage': {},
            'overall_coverage': 0.0
        }
        
        total_protected = 0
        total_tests = 0
        
        # Test each attack
        for attack_func in attack_functions:
            attack_name = getattr(attack_func, 'name', attack_func.__class__.__name__)
            protected_count = 0
            
            for test_input in test_inputs[:10]:  # Subset for analysis
                try:
                    # Apply defense
                    defense_result = defense_function(test_input)
                    defended_input = defense_result.defended_input if hasattr(defense_result, 'defended_input') else test_input
                    
                    # Test attack on defended input
                    attack_result = attack_func.unleash(defended_input, epsilon=epsilon)
                    
                    # Check if defense was effective
                    if not (hasattr(attack_result, 'success') and attack_result.success):
                        protected_count += 1
                        total_protected += 1
                    
                    total_tests += 1
                except:
                    pass
            
            coverage_results['attack_coverage'][attack_name] = protected_count / min(len(test_inputs), 10)
        
        # Overall coverage
        coverage_results['overall_coverage'] = total_protected / max(total_tests, 1)
        
        # Input-specific analysis
        coverage_results['input_coverage'] = self._analyze_input_specific_protection(
            defense_function, test_inputs, attack_functions[0] if attack_functions else None
        )
        
        return coverage_results
    
    def _analyze_input_specific_protection(self,
                                         defense_function: Callable,
                                         test_inputs: List[torch.Tensor],
                                         sample_attack: Optional[Callable]) -> Dict[str, float]:
        """Analyze protection effectiveness for different input characteristics."""
        
        if sample_attack is None:
            return {}
        
        input_analysis = {
            'high_contrast': 0.0,
            'low_contrast': 0.0,
            'complex_patterns': 0.0,
            'simple_patterns': 0.0
        }
        
        # Categorize inputs (simplified analysis)
        for test_input in test_inputs[:8]:
            try:
                # Calculate input characteristics
                if isinstance(test_input, torch.Tensor) and test_input.dim() >= 3:
                    variance = test_input.var().item()
                    
                    # Apply defense and test
                    defense_result = defense_function(test_input)
                    defended_input = defense_result.defended_input if hasattr(defense_result, 'defended_input') else test_input
                    
                    attack_result = sample_attack.unleash(defended_input, epsilon=0.1)
                    protected = not (hasattr(attack_result, 'success') and attack_result.success)
                    
                    # Categorize and record
                    if variance > 0.1:  # High contrast
                        input_analysis['high_contrast'] += 1 if protected else 0
                    else:  # Low contrast
                        input_analysis['low_contrast'] += 1 if protected else 0
                
            except:
                pass
        
        # Normalize results
        for key in input_analysis:
            input_analysis[key] = input_analysis[key] / max(len(test_inputs), 1)
        
        return input_analysis
    
    def analyze_defense_mechanisms(self,
                                 defense_function: Callable,
                                 sample_inputs: List[torch.Tensor]) -> Dict[str, Any]:
        """Analyze the mechanisms used by a defense."""
        
        mechanism_analysis = {
            'input_transformation': False,
            'model_modification': False,
            'detection_based': False,
            'preprocessing': False,
            'transformation_magnitude': 0.0
        }
        
        # Test defense behavior
        for sample_input in sample_inputs[:3]:
            try:
                defense_result = defense_function(sample_input)
                
                if hasattr(defense_result, 'defended_input'):
                    defended_input = defense_result.defended_input
                    
                    # Check for input transformation
                    if not torch.equal(sample_input, defended_input):
                        mechanism_analysis['input_transformation'] = True
                        mechanism_analysis['preprocessing'] = True
                        
                        # Measure transformation magnitude
                        diff = torch.norm(sample_input - defended_input).item()
                        original_norm = torch.norm(sample_input).item()
                        mechanism_analysis['transformation_magnitude'] += diff / max(original_norm, 1e-8)
                
                # Check for detection capability
                if hasattr(defense_result, 'threat_detected'):
                    mechanism_analysis['detection_based'] = True
                
            except:
                pass
        
        # Average transformation magnitude
        mechanism_analysis['transformation_magnitude'] /= len(sample_inputs)
        
        return mechanism_analysis
    
    def compare_defense_strategies(self,
                                 defense_functions: List[Callable],
                                 test_inputs: List[torch.Tensor],
                                 attack_functions: List[Callable]) -> Dict[str, Any]:
        """Compare multiple defense strategies."""
        
        comparison_results = {
            'defense_rankings': {},
            'attack_specific_performance': {},
            'efficiency_comparison': {},
            'recommendation': None
        }
        
        defense_scores = {}
        
        # Analyze each defense
        for defense_func in defense_functions:
            defense_name = getattr(defense_func, 'name', defense_func.__class__.__name__)
            
            # Measure effectiveness
            coverage = self.analyze_protection_coverage(defense_func, test_inputs, attack_functions)
            
            # Measure efficiency (simplified)
            efficiency_score = self._measure_defense_efficiency(defense_func, test_inputs[0])
            
            # Calculate overall score
            overall_score = coverage['overall_coverage'] * 0.7 + efficiency_score * 0.3
            defense_scores[defense_name] = overall_score
            
            comparison_results['defense_rankings'][defense_name] = {
                'effectiveness': coverage['overall_coverage'],
                'efficiency': efficiency_score,
                'overall_score': overall_score
            }
        
        # Find best defense
        if defense_scores:
            best_defense = max(defense_scores.items(), key=lambda x: x[1])
            comparison_results['recommendation'] = {
                'defense_name': best_defense[0],
                'score': best_defense[1],
                'reasoning': f"Best overall performance with {best_defense[1]:.3f} score"
            }
        
        return comparison_results
    
    def _measure_defense_efficiency(self, defense_function: Callable, sample_input: torch.Tensor) -> float:
        """Measure computational efficiency of defense."""
        import time
        
        try:
            # Measure execution time
            start_time = time.time()
            for _ in range(5):  # Multiple runs for accuracy
                defense_function(sample_input)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 5
            
            # Simple efficiency score (lower time = higher score)
            efficiency_score = 1.0 / (1.0 + avg_time * 1000)  # Normalize
            return efficiency_score
            
        except:
            return 0.5  # Default moderate efficiency
    
    def generate_defense_recommendations(self,
                                       model_info: Dict,
                                       threat_model: Dict,
                                       performance_requirements: Dict) -> List[str]:
        """Generate defense recommendations based on context."""
        
        recommendations = []
        
        # Model-specific recommendations
        if model_info.get('model_type') == 'CNN':
            recommendations.append("Consider input preprocessing defenses (e.g., Aegis)")
            recommendations.append("Evaluate adversarial training (Fortitude)")
        
        elif model_info.get('model_type') == 'Transformer':
            recommendations.append("Implement robust tokenization")
            recommendations.append("Consider defensive distillation")
        
        # Threat-specific recommendations
        if threat_model.get('attack_types', []):
            if 'gradient_based' in threat_model['attack_types']:
                recommendations.append("Implement gradient masking defenses")
            
            if 'query_based' in threat_model['attack_types']:
                recommendations.append("Add query detection and limiting")
        
        # Performance-specific recommendations
        if performance_requirements.get('real_time', False):
            recommendations.append("Prioritize lightweight preprocessing defenses")
            recommendations.append("Avoid computationally expensive training-time defenses")
        
        if performance_requirements.get('high_accuracy', False):
            recommendations.append("Minimize accuracy-degrading defenses")
            recommendations.append("Consider ensemble methods for robust accuracy")
        
        return recommendations