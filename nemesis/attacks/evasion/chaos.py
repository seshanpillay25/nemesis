"""
Chaos Attack 🌪️

"When order fails, embrace chaos - the ultimate ensemble of destruction"

Implementation of AutoAttack as a mythological force of chaos
that combines multiple attack strategies for maximum devastation.
"""

import numpy as np
import torch
from typing import Any, Optional, Dict, List
from ..base import AttackBase, AttackResult
from .whisper import Whisper
from .storm import Storm
from .shapeshifter import Shapeshifter
from .mirage import Mirage

class Chaos(AttackBase):
    """
    Chaos - AutoAttack Ensemble
    
    Like the primordial chaos from which all things emerge,
    this attack combines multiple strategies in an unstoppable ensemble.
    """
    
    def __init__(self, model: Any):
        super().__init__(
            model=model,
            name="Chaos", 
            description="Ensemble attack that unleashes multiple strategies in chaotic harmony"
        )
        
        # Initialize component attacks
        self.whisper = Whisper(model)
        self.storm = Storm(model)
        self.shapeshifter = Shapeshifter(model)
        self.mirage = Mirage(model)
        
        # Attack order for untargeted attacks
        self.untargeted_sequence = [
            ('storm', self.storm),
            ('mirage', self.mirage), 
            ('shapeshifter', self.shapeshifter),
            ('whisper', self.whisper)
        ]
        
        # Attack order for targeted attacks
        self.targeted_sequence = [
            ('shapeshifter', self.shapeshifter),
            ('storm', self.storm),
            ('mirage', self.mirage)
        ]
    
    def forge_attack(self, x: Any, y: Optional[Any] = None,
                    epsilon: float = 0.1, targeted: bool = False,
                    target_class: Optional[int] = None,
                    individual_budget: Optional[int] = None) -> AttackResult:
        """
        Forge a chaos attack using multiple strategies.
        
        Args:
            x: Input tensor to attack
            y: True label (if available)
            epsilon: Maximum perturbation magnitude
            targeted: Whether to perform targeted attack
            target_class: Target class for targeted attacks
            individual_budget: Query budget per individual attack
            
        Returns:
            AttackResult containing the chaos' devastation
        """
        original_input = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        
        # Get original prediction
        original_pred = self._predict(x)
        original_class = torch.argmax(original_pred, dim=-1) if isinstance(original_pred, torch.Tensor) else np.argmax(original_pred)
        
        # Choose attack sequence based on target
        if targeted:
            sequence = self.targeted_sequence
        else:
            sequence = self.untargeted_sequence
        
        # Track all attack results
        attack_results = []
        best_result = None
        
        # Execute attacks in sequence
        for attack_name, attack_method in sequence:
            try:
                # Configure attack parameters based on type
                if attack_name == 'whisper':
                    result = attack_method.unleash(
                        x, y, epsilon=epsilon, targeted=targeted, target_class=target_class
                    )
                elif attack_name == 'storm':
                    result = attack_method.unleash(
                        x, y, epsilon=epsilon, targeted=targeted, target_class=target_class,
                        num_iter=20, alpha=epsilon/4
                    )
                elif attack_name == 'shapeshifter':
                    # Use smaller iteration budget for chaos ensemble
                    max_iter = individual_budget or 200
                    result = attack_method.unleash(
                        x, y, targeted=targeted, target_class=target_class,
                        max_iterations=max_iter, c=1.0
                    )
                elif attack_name == 'mirage':
                    result = attack_method.unleash(
                        x, y, targeted=targeted, target_class=target_class,
                        max_iterations=50
                    )
                
                # Add query count to total
                self.queries_used += result.queries_used
                
                # Store result
                attack_results.append((attack_name, result))
                
                # Check if attack succeeded
                if result.success:
                    # Choose best result based on perturbation norm
                    if best_result is None or result.perturbation_norm < best_result.perturbation_norm:
                        best_result = result
                        best_result.metadata['successful_attack'] = attack_name
                        best_result.metadata['attack_sequence'] = [name for name, _ in attack_results]
                
                # Early stopping if we found a good attack
                if result.success and attack_name in ['storm', 'mirage']:
                    break
                    
            except Exception as e:
                # Log attack failure but continue with others
                attack_results.append((attack_name, None))
                continue
        
        # If no attack succeeded, return the best attempt
        if best_result is None:
            # Find result with highest confidence drop
            best_confidence_drop = -1
            for attack_name, result in attack_results:
                if result is not None and result.confidence_drop > best_confidence_drop:
                    best_confidence_drop = result.confidence_drop
                    best_result = result
        
        # If still no result, create a failed result
        if best_result is None:
            best_result = self._create_failed_chaos_result(
                original_input, original_pred, attack_results
            )
        
        # Update metadata with chaos-specific information
        best_result.attack_name = self.name
        best_result.metadata.update({
            'chaos_type': 'targeted' if targeted else 'untargeted',
            'total_attacks_tried': len([r for _, r in attack_results if r is not None]),
            'successful_attacks': len([r for _, r in attack_results if r is not None and r.success]),
            'attack_results': {name: r.success if r else False for name, r in attack_results},
            'total_queries': self.queries_used,
        })
        
        return best_result
    
    def _create_failed_chaos_result(self, original_input: Any, original_pred: Any, 
                                   attack_results: List) -> AttackResult:
        """Create a result when all attacks fail."""
        
        if isinstance(original_input, torch.Tensor):
            zero_perturbation = torch.zeros_like(original_input)
        else:
            zero_perturbation = np.zeros_like(original_input)
        
        return AttackResult(
            original_input=original_input,
            adversarial_input=original_input,
            original_prediction=original_pred,
            adversarial_prediction=original_pred,
            perturbation=zero_perturbation,
            success=False,
            queries_used=self.queries_used,
            perturbation_norm=0.0,
            confidence_drop=0.0,
            attack_name=self.name,
            metadata={
                'failure_reason': 'All chaos attacks failed',
                'attempted_attacks': len(attack_results),
                'attack_results': {name: r.success if r else False for name, r in attack_results}
            }
        )
    
    def analyze_model_robustness(self, test_inputs: List[Any], 
                               test_labels: Optional[List[Any]] = None,
                               epsilon: float = 0.1) -> Dict[str, Any]:
        """
        Comprehensive robustness analysis using chaos ensemble.
        
        Args:
            test_inputs: List of test inputs
            test_labels: List of test labels (optional)
            epsilon: Perturbation magnitude
            
        Returns:
            Comprehensive robustness report
        """
        results = {
            'total_samples': len(test_inputs),
            'successful_attacks': 0,
            'attack_breakdown': {
                'whisper': {'attempts': 0, 'successes': 0},
                'storm': {'attempts': 0, 'successes': 0},
                'shapeshifter': {'attempts': 0, 'successes': 0},
                'mirage': {'attempts': 0, 'successes': 0}
            },
            'average_perturbation': 0.0,
            'average_queries': 0.0,
            'robustness_score': 0.0
        }
        
        total_perturbation = 0.0
        total_queries = 0.0
        
        for i, x in enumerate(test_inputs):
            y = test_labels[i] if test_labels else None
            
            # Reset query counter
            self.queries_used = 0
            
            # Launch chaos attack
            result = self.forge_attack(x, y, epsilon=epsilon)
            
            # Update statistics
            if result.success:
                results['successful_attacks'] += 1
                total_perturbation += result.perturbation_norm
                
                # Track which attack succeeded
                successful_attack = result.metadata.get('successful_attack', 'unknown')
                if successful_attack in results['attack_breakdown']:
                    results['attack_breakdown'][successful_attack]['successes'] += 1
            
            total_queries += result.queries_used
            
            # Track all attempted attacks
            for attack_name, success in result.metadata.get('attack_results', {}).items():
                if attack_name in results['attack_breakdown']:
                    results['attack_breakdown'][attack_name]['attempts'] += 1
        
        # Compute final metrics
        results['attack_success_rate'] = results['successful_attacks'] / results['total_samples']
        results['average_perturbation'] = total_perturbation / max(results['successful_attacks'], 1)
        results['average_queries'] = total_queries / results['total_samples']
        results['robustness_score'] = 1.0 - results['attack_success_rate']
        
        # Individual attack success rates
        for attack_name in results['attack_breakdown']:
            attempts = results['attack_breakdown'][attack_name]['attempts']
            successes = results['attack_breakdown'][attack_name]['successes']
            results['attack_breakdown'][attack_name]['success_rate'] = (
                successes / max(attempts, 1)
            )
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive chaos statistics."""
        base_stats = super().get_statistics()
        
        # Add component attack statistics
        component_stats = {
            'whisper': self.whisper.get_statistics(),
            'storm': self.storm.get_statistics(),
            'shapeshifter': self.shapeshifter.get_statistics(), 
            'mirage': self.mirage.get_statistics()
        }
        
        base_stats['component_attacks'] = component_stats
        return base_stats