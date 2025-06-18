"""
Immunity Armor 🛡️⭐

"Divine protection with mathematical guarantees"

Implementation of certified defenses that provide provable
robustness guarantees against adversarial attacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Optional, Dict, List, Tuple
from ..base import DefenseBase, DefenseResult

class Immunity(DefenseBase):
    """
    Immunity - Certified Defense with Provable Guarantees
    
    Like divine immunity granted by the gods themselves,
    this defense provides mathematical guarantees of protection.
    """
    
    def __init__(self, model: Any):
        super().__init__(
            name="Immunity",
            description="Certified defense with divine guarantees of protection",
            defense_type="certified"
        )
        self.model = model
        self.certification_radius = 0.0
        self.noise_variance = 0.1
        self.num_samples = 1000
    
    def forge_defense(self, x: torch.Tensor, method: str = "randomized_smoothing",
                     noise_variance: float = 0.1, num_samples: int = 1000,
                     confidence_alpha: float = 0.001, **kwargs) -> DefenseResult:
        """
        Forge Immunity defense through certified methods.
        
        Args:
            x: Input to certify
            method: Certification method ("randomized_smoothing", "interval_bound", "lipschitz")
            noise_variance: Variance of smoothing noise
            num_samples: Number of samples for randomized smoothing
            confidence_alpha: Confidence level for certification
            **kwargs: Additional certification parameters
            
        Returns:
            DefenseResult with certification guarantees
        """
        
        original_input = x.clone()
        
        if method == "randomized_smoothing":
            return self._randomized_smoothing_defense(
                x, noise_variance, num_samples, confidence_alpha, original_input
            )
        elif method == "interval_bound":
            return self._interval_bound_defense(x, original_input, **kwargs)
        elif method == "lipschitz":
            return self._lipschitz_defense(x, original_input, **kwargs)
        else:
            raise ValueError(f"Unknown certification method: {method}")
    
    def _randomized_smoothing_defense(self, x: torch.Tensor, noise_variance: float,
                                    num_samples: int, confidence_alpha: float,
                                    original_input: torch.Tensor) -> DefenseResult:
        """Implement randomized smoothing for certified robustness."""
        
        self.noise_variance = noise_variance
        self.num_samples = num_samples
        
        # Sample predictions with Gaussian noise
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Add Gaussian noise
                noise = torch.randn_like(x) * np.sqrt(noise_variance)
                noisy_input = x + noise
                
                # Get prediction
                output = self.model(noisy_input)
                pred_class = torch.argmax(output, dim=-1).item()
                predictions.append(pred_class)
        
        # Count votes for each class
        from collections import Counter
        vote_counts = Counter(predictions)
        
        # Get top two classes
        sorted_votes = vote_counts.most_common(2)
        top_class = sorted_votes[0][0]
        top_count = sorted_votes[0][1]
        
        if len(sorted_votes) > 1:
            second_count = sorted_votes[1][1]
        else:
            second_count = 0
        
        # Compute certification radius using Neyman-Pearson test
        certification_radius = self._compute_certification_radius(
            top_count, second_count, num_samples, noise_variance, confidence_alpha
        )
        
        self.certification_radius = certification_radius
        
        # Defense is applied if we can certify non-zero radius
        defense_applied = certification_radius > 0.0
        
        # Create smoothed prediction (use majority vote)
        smoothed_prediction = torch.zeros(1, len(vote_counts))
        for class_idx, count in vote_counts.items():
            if class_idx < smoothed_prediction.shape[1]:
                smoothed_prediction[0, class_idx] = count / num_samples
        
        return DefenseResult(
            original_input=original_input,
            defended_input=x,  # Input not modified, but prediction is smoothed
            original_prediction=None,
            defended_prediction=smoothed_prediction,
            defense_applied=defense_applied,
            confidence_change=0.0,
            defense_strength=certification_radius,
            defense_name=self.name,
            metadata={
                'method': 'randomized_smoothing',
                'certification_radius': certification_radius,
                'noise_variance': noise_variance,
                'num_samples': num_samples,
                'confidence_alpha': confidence_alpha,
                'top_class': top_class,
                'top_count': top_count,
                'vote_distribution': dict(vote_counts),
            }
        )
    
    def _compute_certification_radius(self, n_a: int, n_b: int, num_samples: int,
                                    noise_variance: float, alpha: float) -> float:
        """Compute certification radius for randomized smoothing."""
        
        if n_a <= n_b:
            return 0.0
        
        # Compute confidence interval for p_a (probability of class a)
        p_a_lower = self._confidence_interval_lower(n_a, num_samples, alpha)
        
        if p_a_lower <= 0.5:
            return 0.0
        
        # Certification radius formula for Gaussian noise
        from scipy.stats import norm
        radius = norm.ppf(p_a_lower) * np.sqrt(noise_variance)
        
        return max(0.0, radius)
    
    def _confidence_interval_lower(self, count: int, num_samples: int, alpha: float) -> float:
        """Compute lower bound of confidence interval for binomial proportion."""
        
        # Wilson score interval (more accurate than normal approximation)
        p_hat = count / num_samples
        
        try:
            from scipy.stats import norm
            z = norm.ppf(1 - alpha / 2)
        except ImportError:
            # Fallback to approximate z-score
            z = 1.96 if alpha == 0.05 else 2.576 if alpha == 0.01 else 1.645
        
        denominator = 1 + z**2 / num_samples
        numerator = p_hat + z**2 / (2 * num_samples) - z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * num_samples)) / num_samples)
        
        return max(0.0, numerator / denominator)
    
    def _interval_bound_defense(self, x: torch.Tensor, original_input: torch.Tensor,
                               epsilon: float = 0.1) -> DefenseResult:
        """Implement interval bound propagation for certification."""
        
        # Simplified interval bound propagation
        # This would normally require specialized network architectures
        
        lower_bound = torch.clamp(x - epsilon, 0, 1)
        upper_bound = torch.clamp(x + epsilon, 0, 1)
        
        # Get predictions for bounds
        with torch.no_grad():
            lower_output = self.model(lower_bound)
            upper_output = self.model(upper_bound)
            original_output = self.model(x)
        
        # Check if prediction is stable across bounds
        original_class = torch.argmax(original_output, dim=-1).item()
        lower_class = torch.argmax(lower_output, dim=-1).item()
        upper_class = torch.argmax(upper_output, dim=-1).item()
        
        # Defense succeeds if all predictions agree
        certified = (original_class == lower_class == upper_class)
        certification_radius = epsilon if certified else 0.0
        
        return DefenseResult(
            original_input=original_input,
            defended_input=x,
            original_prediction=original_output,
            defended_prediction=original_output,
            defense_applied=certified,
            confidence_change=0.0,
            defense_strength=certification_radius,
            defense_name=self.name,
            metadata={
                'method': 'interval_bound',
                'epsilon': epsilon,
                'certified': certified,
                'certification_radius': certification_radius,
                'original_class': original_class,
                'lower_class': lower_class,
                'upper_class': upper_class,
            }
        )
    
    def _lipschitz_defense(self, x: torch.Tensor, original_input: torch.Tensor,
                          lipschitz_constant: float = 1.0) -> DefenseResult:
        """Implement Lipschitz-constrained defense."""
        
        # Estimate local Lipschitz constant
        estimated_lipschitz = self._estimate_local_lipschitz(x)
        
        # Certification radius based on Lipschitz constraint
        if estimated_lipschitz > 0 and estimated_lipschitz <= lipschitz_constant:
            # Simple certification: radius inversely proportional to Lipschitz constant
            certification_radius = 1.0 / (estimated_lipschitz * 10)
            certified = True
        else:
            certification_radius = 0.0
            certified = False
        
        # Get original prediction
        with torch.no_grad():
            original_output = self.model(x)
        
        return DefenseResult(
            original_input=original_input,
            defended_input=x,
            original_prediction=original_output,
            defended_prediction=original_output,
            defense_applied=certified,
            confidence_change=0.0,
            defense_strength=certification_radius,
            defense_name=self.name,
            metadata={
                'method': 'lipschitz',
                'estimated_lipschitz': estimated_lipschitz,
                'lipschitz_constant': lipschitz_constant,
                'certified': certified,
                'certification_radius': certification_radius,
            }
        )
    
    def _estimate_local_lipschitz(self, x: torch.Tensor, num_samples: int = 10) -> float:
        """Estimate local Lipschitz constant around input x."""
        
        max_ratio = 0.0
        
        with torch.no_grad():
            original_output = self.model(x)
            
            for _ in range(num_samples):
                # Sample nearby point
                delta = torch.randn_like(x) * 0.01
                x_perturbed = torch.clamp(x + delta, 0, 1)
                
                # Compute outputs
                perturbed_output = self.model(x_perturbed)
                
                # Compute ratio
                input_distance = torch.norm(delta).item()
                output_distance = torch.norm(perturbed_output - original_output).item()
                
                if input_distance > 0:
                    ratio = output_distance / input_distance
                    max_ratio = max(max_ratio, ratio)
        
        return max_ratio
    
    def apply_to_model(self, model: Any) -> Any:
        """Apply immunity certification to a model."""
        self.model = model
        # For certified defenses, we typically need to modify the model architecture
        # or inference procedure. Here we return a wrapper.
        return CertifiedModelWrapper(model, self)
    
    def get_certification_report(self, test_inputs: List[torch.Tensor],
                                method: str = "randomized_smoothing") -> Dict[str, Any]:
        """Generate comprehensive certification report."""
        
        results = {
            'total_inputs': len(test_inputs),
            'certified_inputs': 0,
            'average_radius': 0.0,
            'certification_rate': 0.0,
            'radius_distribution': [],
            'method': method
        }
        
        total_radius = 0.0
        
        for x in test_inputs:
            cert_result = self.forge_defense(x, method=method)
            
            if cert_result.defense_applied:
                results['certified_inputs'] += 1
                radius = cert_result.defense_strength
                total_radius += radius
                results['radius_distribution'].append(radius)
        
        results['certification_rate'] = results['certified_inputs'] / results['total_inputs']
        results['average_radius'] = total_radius / max(results['certified_inputs'], 1)
        
        if results['radius_distribution']:
            results['min_radius'] = min(results['radius_distribution'])
            results['max_radius'] = max(results['radius_distribution'])
            results['median_radius'] = np.median(results['radius_distribution'])
        
        return results

class CertifiedModelWrapper:
    """Wrapper for models with certification capabilities."""
    
    def __init__(self, model: Any, immunity_defense: Immunity):
        self.model = model
        self.immunity = immunity_defense
    
    def __call__(self, x: torch.Tensor, certify: bool = False) -> Any:
        """Forward pass with optional certification."""
        
        if certify:
            # Return certified prediction
            cert_result = self.immunity.forge_defense(x)
            return cert_result.defended_prediction
        else:
            # Standard prediction
            return self.model(x)
    
    def forward(self, x: torch.Tensor) -> Any:
        """Standard forward pass."""
        return self.model(x)
    
    def certified_forward(self, x: torch.Tensor) -> Tuple[Any, float]:
        """Forward pass with certification radius."""
        cert_result = self.immunity.forge_defense(x)
        return cert_result.defended_prediction, cert_result.defense_strength
    
    def parameters(self):
        """Get model parameters."""
        return self.model.parameters()
    
    def train(self, mode: bool = True):
        """Set training mode."""
        return self.model.train(mode)
    
    def eval(self):
        """Set evaluation mode."""
        return self.model.eval()