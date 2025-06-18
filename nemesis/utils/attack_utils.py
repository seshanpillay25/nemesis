"""
Attack Utilities ⚔️🔧

"Forging the weapons of divine retribution"

Utilities for attack validation, perturbation analysis,
and attack parameter optimization.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import warnings


@dataclass
class PerturbationStats:
    """Statistics about perturbations."""
    l1_norm: float
    l2_norm: float
    linf_norm: float
    mean_magnitude: float
    max_magnitude: float
    pixel_changes: int
    relative_magnitude: float


class PerturbationUtils:
    """
    Utilities for analyzing and manipulating adversarial perturbations.
    
    Provides tools for measuring perturbation magnitude, quality,
    and applying various perturbation constraints.
    """
    
    @staticmethod
    def calculate_perturbation_stats(original: torch.Tensor, 
                                   perturbed: torch.Tensor) -> PerturbationStats:
        """Calculate comprehensive perturbation statistics."""
        
        diff = perturbed - original
        
        # Norms
        l1_norm = torch.norm(diff, p=1).item()
        l2_norm = torch.norm(diff, p=2).item()
        linf_norm = torch.norm(diff, p=float('inf')).item()
        
        # Magnitude statistics
        abs_diff = torch.abs(diff)
        mean_magnitude = abs_diff.mean().item()
        max_magnitude = abs_diff.max().item()
        
        # Pixel changes (for images)
        pixel_changes = (abs_diff > 1e-6).sum().item()
        
        # Relative magnitude
        original_norm = torch.norm(original, p=2).item()
        relative_magnitude = l2_norm / max(original_norm, 1e-8)
        
        return PerturbationStats(
            l1_norm=l1_norm,
            l2_norm=l2_norm,
            linf_norm=linf_norm,
            mean_magnitude=mean_magnitude,
            max_magnitude=max_magnitude,
            pixel_changes=pixel_changes,
            relative_magnitude=relative_magnitude
        )
    
    @staticmethod
    def project_perturbation(perturbation: torch.Tensor,
                           epsilon: float,
                           norm_type: str = 'linf') -> torch.Tensor:
        """Project perturbation to satisfy norm constraints."""
        
        if norm_type == 'linf':
            return torch.clamp(perturbation, -epsilon, epsilon)
        
        elif norm_type == 'l2':
            norm = torch.norm(perturbation, p=2)
            if norm > epsilon:
                return perturbation * (epsilon / norm)
            return perturbation
        
        elif norm_type == 'l1':
            # L1 projection is more complex
            flat_pert = perturbation.flatten()
            if torch.norm(flat_pert, p=1) <= epsilon:
                return perturbation
            
            # Sort by absolute value
            abs_pert = torch.abs(flat_pert)
            sorted_pert, indices = torch.sort(abs_pert, descending=True)
            
            # Find threshold
            cumsum = torch.cumsum(sorted_pert, dim=0)
            k = torch.sum(cumsum <= epsilon).item()
            
            if k < len(flat_pert):
                threshold = (cumsum[k] - epsilon) / (k + 1)
                projected = torch.sign(flat_pert) * torch.clamp(abs_pert - threshold, min=0)
            else:
                projected = flat_pert
            
            return projected.reshape(perturbation.shape)
        
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
    
    @staticmethod
    def clip_to_valid_range(adversarial: torch.Tensor,
                          original: torch.Tensor,
                          data_min: float = 0.0,
                          data_max: float = 1.0) -> torch.Tensor:
        """Clip adversarial example to valid data range."""
        return torch.clamp(adversarial, data_min, data_max)
    
    @staticmethod
    def apply_perturbation_smoothing(perturbation: torch.Tensor,
                                   smoothing_type: str = 'gaussian',
                                   strength: float = 0.5) -> torch.Tensor:
        """Apply smoothing to perturbations."""
        
        if smoothing_type == 'gaussian':
            # Apply Gaussian smoothing
            kernel_size = 3
            sigma = strength
            
            # Create Gaussian kernel
            kernel = PerturbationUtils._gaussian_kernel_2d(kernel_size, sigma)
            
            if perturbation.dim() == 4:  # Batch of images
                smoothed = torch.nn.functional.conv2d(
                    perturbation, kernel, padding=kernel_size//2, groups=perturbation.size(1)
                )
            elif perturbation.dim() == 3:  # Single image
                smoothed = torch.nn.functional.conv2d(
                    perturbation.unsqueeze(0), kernel, padding=kernel_size//2
                ).squeeze(0)
            else:
                smoothed = perturbation  # Can't smooth non-image data
            
            return smoothed
        
        elif smoothing_type == 'median':
            # Simple median filtering approximation
            return perturbation * (1 - strength) + perturbation.median() * strength
        
        else:
            return perturbation
    
    @staticmethod
    def _gaussian_kernel_2d(kernel_size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel."""
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = torch.exp(torch.tensor(-(x**2 + y**2) / (2 * sigma**2)))
        
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)


class AttackValidator:
    """
    Validation utilities for adversarial attacks.
    
    Ensures attacks meet quality standards and constraints.
    """
    
    def __init__(self, 
                 max_epsilon: float = 0.3,
                 min_success_rate: float = 0.1,
                 max_queries: int = 10000):
        self.max_epsilon = max_epsilon
        self.min_success_rate = min_success_rate
        self.max_queries = max_queries
    
    def validate_attack_result(self, 
                             attack_result: Any,
                             original_input: torch.Tensor) -> Dict[str, bool]:
        """Validate an attack result."""
        
        validation_results = {
            'success_valid': True,
            'perturbation_valid': True,
            'quality_valid': True,
            'efficiency_valid': True
        }
        
        # Check if attack was successful
        if not hasattr(attack_result, 'success') or not attack_result.success:
            validation_results['success_valid'] = False
            return validation_results
        
        # Check perturbation magnitude
        if hasattr(attack_result, 'adversarial_input'):
            stats = PerturbationUtils.calculate_perturbation_stats(
                original_input, attack_result.adversarial_input
            )
            
            if stats.linf_norm > self.max_epsilon:
                validation_results['perturbation_valid'] = False
            
            # Check quality metrics
            if stats.relative_magnitude > 1.0:  # Perturbation larger than input
                validation_results['quality_valid'] = False
        
        # Check query efficiency
        if hasattr(attack_result, 'queries_used'):
            if attack_result.queries_used > self.max_queries:
                validation_results['efficiency_valid'] = False
        
        return validation_results
    
    def validate_attack_batch(self,
                            attack_results: List[Any],
                            original_inputs: List[torch.Tensor]) -> Dict[str, float]:
        """Validate a batch of attack results."""
        
        total_results = len(attack_results)
        if total_results == 0:
            return {'batch_valid': False, 'error': 'Empty batch'}
        
        valid_count = 0
        success_count = 0
        
        for result, original in zip(attack_results, original_inputs):
            validation = self.validate_attack_result(result, original)
            
            if all(validation.values()):
                valid_count += 1
            
            if hasattr(result, 'success') and result.success:
                success_count += 1
        
        return {
            'batch_valid': valid_count / total_results >= self.min_success_rate,
            'success_rate': success_count / total_results,
            'validation_rate': valid_count / total_results,
            'total_samples': total_results
        }


class EpsilonScheduler:
    """
    Epsilon scheduling for progressive adversarial training.
    
    Manages epsilon values during training to gradually increase
    adversarial strength.
    """
    
    def __init__(self, 
                 initial_epsilon: float = 0.01,
                 final_epsilon: float = 0.3,
                 schedule_type: str = 'linear'):
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.schedule_type = schedule_type
        self.current_step = 0
    
    def get_epsilon(self, step: int, total_steps: int) -> float:
        """Get epsilon value for current step."""
        
        self.current_step = step
        progress = step / max(total_steps, 1)
        
        if self.schedule_type == 'linear':
            return self._linear_schedule(progress)
        elif self.schedule_type == 'exponential':
            return self._exponential_schedule(progress)
        elif self.schedule_type == 'cosine':
            return self._cosine_schedule(progress)
        elif self.schedule_type == 'step':
            return self._step_schedule(progress)
        else:
            return self.final_epsilon
    
    def _linear_schedule(self, progress: float) -> float:
        """Linear epsilon schedule."""
        return self.initial_epsilon + (self.final_epsilon - self.initial_epsilon) * progress
    
    def _exponential_schedule(self, progress: float) -> float:
        """Exponential epsilon schedule."""
        return self.initial_epsilon * (self.final_epsilon / self.initial_epsilon) ** progress
    
    def _cosine_schedule(self, progress: float) -> float:
        """Cosine epsilon schedule."""
        return self.initial_epsilon + 0.5 * (self.final_epsilon - self.initial_epsilon) * (
            1 + np.cos(np.pi * progress)
        )
    
    def _step_schedule(self, progress: float) -> float:
        """Step epsilon schedule."""
        if progress < 0.25:
            return self.initial_epsilon
        elif progress < 0.5:
            return self.initial_epsilon + 0.3 * (self.final_epsilon - self.initial_epsilon)
        elif progress < 0.75:
            return self.initial_epsilon + 0.6 * (self.final_epsilon - self.initial_epsilon)
        else:
            return self.final_epsilon
    
    def get_epsilon_range(self, num_epochs: int) -> List[float]:
        """Get epsilon values for all epochs."""
        return [self.get_epsilon(i, num_epochs) for i in range(num_epochs)]


class AttackOptimizer:
    """
    Optimizer for attack parameters.
    
    Automatically tunes attack parameters for optimal performance.
    """
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_epsilon(self,
                        attack_function: Callable,
                        test_inputs: List[torch.Tensor],
                        target_success_rate: float = 0.8,
                        epsilon_range: Tuple[float, float] = (0.01, 0.5),
                        max_iterations: int = 10) -> float:
        """Find optimal epsilon for target success rate."""
        
        min_eps, max_eps = epsilon_range
        best_epsilon = min_eps
        
        for iteration in range(max_iterations):
            # Binary search for optimal epsilon
            mid_eps = (min_eps + max_eps) / 2
            
            # Test attack with current epsilon
            success_count = 0
            for test_input in test_inputs[:10]:  # Use subset for speed
                try:
                    result = attack_function(test_input, epsilon=mid_eps)
                    if hasattr(result, 'success') and result.success:
                        success_count += 1
                except:
                    pass
            
            success_rate = success_count / min(len(test_inputs), 10)
            
            # Update search range
            if success_rate < target_success_rate:
                min_eps = mid_eps
            else:
                max_eps = mid_eps
                best_epsilon = mid_eps
            
            # Record optimization step
            self.optimization_history.append({
                'iteration': iteration,
                'epsilon': mid_eps,
                'success_rate': success_rate
            })
            
            # Convergence check
            if abs(max_eps - min_eps) < 0.001:
                break
        
        return best_epsilon
    
    def optimize_attack_parameters(self,
                                 attack_function: Callable,
                                 test_inputs: List[torch.Tensor],
                                 parameter_ranges: Dict[str, Tuple]) -> Dict[str, Any]:
        """Optimize multiple attack parameters simultaneously."""
        
        # Simple grid search (in practice, could use more sophisticated optimization)
        best_params = {}
        best_score = 0
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(parameter_ranges)
        
        for params in param_combinations[:20]:  # Limit combinations for speed
            score = self._evaluate_parameter_combination(
                attack_function, test_inputs, params
            )
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'optimization_history': self.optimization_history
        }
    
    def _generate_param_combinations(self, parameter_ranges: Dict[str, Tuple]) -> List[Dict]:
        """Generate parameter combinations for grid search."""
        import itertools
        
        param_names = list(parameter_ranges.keys())
        param_values = []
        
        for param_name, (min_val, max_val) in parameter_ranges.items():
            # Generate 3 values for each parameter
            values = [min_val, (min_val + max_val) / 2, max_val]
            param_values.append(values)
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def _evaluate_parameter_combination(self,
                                      attack_function: Callable,
                                      test_inputs: List[torch.Tensor],
                                      params: Dict) -> float:
        """Evaluate a parameter combination."""
        success_count = 0
        total_count = 0
        
        for test_input in test_inputs[:5]:  # Small subset for evaluation
            try:
                result = attack_function(test_input, **params)
                total_count += 1
                if hasattr(result, 'success') and result.success:
                    success_count += 1
            except:
                pass
        
        return success_count / max(total_count, 1)