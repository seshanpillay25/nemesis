"""
Aegis Shield 🛡️

"The divine shield that purifies corrupted inputs"

Implementation of input purification defenses that cleanse
adversarial perturbations like the legendary Aegis of Zeus.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Optional, Dict, Tuple
from ..base import DefenseBase, DefenseResult

class Aegis(DefenseBase):
    """
    Aegis - Input Purification Defense
    
    Like the divine shield of Zeus that protected against all attacks,
    Aegis purifies inputs to remove adversarial perturbations.
    """
    
    def __init__(self):
        super().__init__(
            name="Aegis",
            description="Input purification that cleanses adversarial corruption like divine protection",
            defense_type="preprocessing"
        )
    
    def forge_defense(self, x: Any, purification_method: str = "gaussian_noise",
                     strength: float = 0.1, **kwargs) -> DefenseResult:
        """
        Forge an Aegis defense through input purification.
        
        Args:
            x: Input to purify
            purification_method: Method of purification ("gaussian_noise", "median_filter", "bit_depth_reduction")
            strength: Strength of purification
            **kwargs: Additional purification parameters
            
        Returns:
            DefenseResult containing the purified input
        """
        original_input = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        
        if purification_method == "gaussian_noise":
            return self._gaussian_noise_purification(x, strength, original_input)
        elif purification_method == "median_filter":
            return self._median_filter_purification(x, strength, original_input)
        elif purification_method == "bit_depth_reduction":
            return self._bit_depth_reduction(x, strength, original_input)
        elif purification_method == "jpeg_compression":
            return self._jpeg_compression_purification(x, strength, original_input)
        elif purification_method == "total_variation":
            return self._total_variation_denoising(x, strength, original_input)
        else:
            raise ValueError(f"Unknown purification method: {purification_method}")
    
    def _gaussian_noise_purification(self, x: torch.Tensor, noise_std: float,
                                   original_input: torch.Tensor) -> DefenseResult:
        """Purify input by adding Gaussian noise."""
        
        # Add Gaussian noise
        noise = torch.randn_like(x) * noise_std
        purified_input = x + noise
        
        # Clamp to valid range
        purified_input = torch.clamp(purified_input, 0, 1)
        
        # Compute purification strength (amount of change)
        purification_strength = torch.norm(purified_input - x).item()
        
        # Check if significant purification was applied
        defense_applied = purification_strength > 0.01
        
        return DefenseResult(
            original_input=original_input,
            defended_input=purified_input,
            original_prediction=None,  # Will be computed by caller if needed
            defended_prediction=None,
            defense_applied=defense_applied,
            confidence_change=0.0,  # Will be computed by caller if needed
            defense_strength=purification_strength,
            defense_name=self.name,
            metadata={
                'purification_method': 'gaussian_noise',
                'noise_std': noise_std,
                'purification_strength': purification_strength,
            }
        )
    
    def _median_filter_purification(self, x: torch.Tensor, kernel_size: float,
                                   original_input: torch.Tensor) -> DefenseResult:
        """Purify input using median filtering."""
        
        # Convert kernel_size to integer (use strength as kernel size parameter)
        kernel_size = max(3, int(kernel_size * 10) // 2 * 2 + 1)  # Ensure odd number
        
        # Apply median filter
        if len(x.shape) == 4:  # Batch dimension
            purified_input = self._apply_median_filter_batch(x, kernel_size)
        else:
            purified_input = self._apply_median_filter_single(x, kernel_size)
        
        # Compute purification strength
        purification_strength = torch.norm(purified_input - x).item()
        defense_applied = purification_strength > 0.001
        
        return DefenseResult(
            original_input=original_input,
            defended_input=purified_input,
            original_prediction=None,
            defended_prediction=None,
            defense_applied=defense_applied,
            confidence_change=0.0,
            defense_strength=purification_strength,
            defense_name=self.name,
            metadata={
                'purification_method': 'median_filter',
                'kernel_size': kernel_size,
                'purification_strength': purification_strength,
            }
        )
    
    def _bit_depth_reduction(self, x: torch.Tensor, reduction_factor: float,
                            original_input: torch.Tensor) -> DefenseResult:
        """Purify input by reducing bit depth."""
        
        # Calculate number of levels based on reduction factor
        num_levels = max(2, int(256 * (1 - reduction_factor)))
        
        # Quantize the input
        quantized = torch.round(x * (num_levels - 1)) / (num_levels - 1)
        purified_input = torch.clamp(quantized, 0, 1)
        
        # Compute purification strength
        purification_strength = torch.norm(purified_input - x).item()
        defense_applied = purification_strength > 0.001
        
        return DefenseResult(
            original_input=original_input,
            defended_input=purified_input,
            original_prediction=None,
            defended_prediction=None,
            defense_applied=defense_applied,
            confidence_change=0.0,
            defense_strength=purification_strength,
            defense_name=self.name,
            metadata={
                'purification_method': 'bit_depth_reduction',
                'num_levels': num_levels,
                'reduction_factor': reduction_factor,
                'purification_strength': purification_strength,
            }
        )
    
    def _jpeg_compression_purification(self, x: torch.Tensor, quality_factor: float,
                                     original_input: torch.Tensor) -> DefenseResult:
        """Purify input by simulating JPEG compression."""
        
        # Convert quality factor to JPEG quality (0-100)
        jpeg_quality = max(10, int(100 * (1 - quality_factor)))
        
        # Simulate JPEG compression effects (simplified)
        # In practice, you would use actual JPEG compression
        compressed = self._simulate_jpeg_compression(x, jpeg_quality)
        
        purified_input = torch.clamp(compressed, 0, 1)
        
        # Compute purification strength
        purification_strength = torch.norm(purified_input - x).item()
        defense_applied = purification_strength > 0.001
        
        return DefenseResult(
            original_input=original_input,
            defended_input=purified_input,
            original_prediction=None,
            defended_prediction=None,
            defense_applied=defense_applied,
            confidence_change=0.0,
            defense_strength=purification_strength,
            defense_name=self.name,
            metadata={
                'purification_method': 'jpeg_compression',
                'jpeg_quality': jpeg_quality,
                'purification_strength': purification_strength,
            }
        )
    
    def _total_variation_denoising(self, x: torch.Tensor, tv_weight: float,
                                  original_input: torch.Tensor) -> DefenseResult:
        """Purify input using total variation denoising."""
        
        # Simplified total variation denoising
        # Compute TV loss and apply gradient descent
        purified_input = x.clone()
        purified_input.requires_grad_(True)
        
        # Simple TV denoising iterations
        for _ in range(5):
            tv_loss = self._compute_total_variation(purified_input)
            reconstruction_loss = F.mse_loss(purified_input, x)
            
            total_loss = reconstruction_loss + tv_weight * tv_loss
            
            # Compute gradients
            total_loss.backward(retain_graph=True)
            
            # Update (simple gradient descent)
            with torch.no_grad():
                if purified_input.grad is not None:
                    purified_input -= 0.01 * purified_input.grad
                    purified_input.clamp_(0, 1)
                    # Zero gradients
                    purified_input.grad.zero_()
                else:
                    # Fallback: apply small random noise for purification
                    noise = torch.randn_like(purified_input) * 0.01
                    purified_input = torch.clamp(purified_input + noise, 0, 1)
                    purified_input.requires_grad_(True)
        
        purified_input = purified_input.detach()
        
        # Compute purification strength
        purification_strength = torch.norm(purified_input - x).item()
        defense_applied = purification_strength > 0.001
        
        return DefenseResult(
            original_input=original_input,
            defended_input=purified_input,
            original_prediction=None,
            defended_prediction=None,
            defense_applied=defense_applied,
            confidence_change=0.0,
            defense_strength=purification_strength,
            defense_name=self.name,
            metadata={
                'purification_method': 'total_variation',
                'tv_weight': tv_weight,
                'purification_strength': purification_strength,
            }
        )
    
    def _apply_median_filter_batch(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Apply median filter to batch of images."""
        # Simplified median filter implementation
        # In practice, you would use proper median filtering
        
        # Use average pooling as approximation (not exactly median, but similar smoothing effect)
        pad_size = kernel_size // 2
        padded = F.pad(x, [pad_size] * 4, mode='reflect')
        
        # Use average pooling with stride 1
        filtered = F.avg_pool2d(padded, kernel_size, stride=1)
        
        return filtered
    
    def _apply_median_filter_single(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Apply median filter to single image."""
        return self._apply_median_filter_batch(x.unsqueeze(0), kernel_size).squeeze(0)
    
    def _simulate_jpeg_compression(self, x: torch.Tensor, quality: int) -> torch.Tensor:
        """Simulate JPEG compression effects."""
        # Simplified JPEG simulation - quantization in frequency domain
        
        # Simple quantization based on quality
        quantization_factor = (100 - quality) / 100.0 * 0.1
        
        # Add quantization noise
        noise = torch.randn_like(x) * quantization_factor
        compressed = x + noise
        
        # Add slight blurring (JPEG compression effect)
        if len(x.shape) >= 3:
            kernel = torch.ones(1, 1, 3, 3) / 9.0  # Simple 3x3 average kernel
            if x.shape[-3] > 1:  # Multi-channel
                kernel = kernel.repeat(x.shape[-3], 1, 1, 1)
            
            pad_size = 1
            if len(x.shape) == 4:  # Batch
                padded = F.pad(x, [pad_size] * 4, mode='reflect')
                blurred = F.conv2d(padded, kernel, groups=x.shape[1], padding=0)
            else:  # Single image
                x_batch = x.unsqueeze(0)
                padded = F.pad(x_batch, [pad_size] * 4, mode='reflect')
                blurred = F.conv2d(padded, kernel, groups=x.shape[0], padding=0)
                blurred = blurred.squeeze(0)
            
            compressed = 0.7 * compressed + 0.3 * blurred
        
        return compressed
    
    def _compute_total_variation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total variation of image."""
        
        if len(x.shape) == 4:  # Batch
            # Differences in x and y directions
            diff_x = x[:, :, :, 1:] - x[:, :, :, :-1]
            diff_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        elif len(x.shape) == 3:  # Single image
            diff_x = x[:, :, 1:] - x[:, :, :-1]
            diff_y = x[:, 1:, :] - x[:, :-1, :]
        else:
            diff_x = x[:, 1:] - x[:, :-1]
            diff_y = x[1:, :] - x[:-1, :]
        
        # Total variation
        tv = torch.sum(torch.abs(diff_x)) + torch.sum(torch.abs(diff_y))
        
        return tv
    
    def adaptive_purification(self, x: torch.Tensor, 
                             suspected_attack_type: str = "unknown") -> DefenseResult:
        """
        Apply adaptive purification based on suspected attack type.
        
        Args:
            x: Input to purify
            suspected_attack_type: Type of suspected attack
            
        Returns:
            DefenseResult with adaptive purification
        """
        
        if suspected_attack_type in ["whisper", "fgsm"]:
            # For gradient-based attacks, use Gaussian noise
            return self.forge_defense(x, "gaussian_noise", strength=0.05)
        elif suspected_attack_type in ["storm", "pgd"]:
            # For iterative attacks, use stronger purification
            return self.forge_defense(x, "total_variation", strength=0.1)
        elif suspected_attack_type in ["shapeshifter", "cw"]:
            # For optimization-based attacks, use bit depth reduction
            return self.forge_defense(x, "bit_depth_reduction", strength=0.2)
        else:
            # Default: combination of methods
            return self._combined_purification(x)
    
    def _combined_purification(self, x: torch.Tensor) -> DefenseResult:
        """Apply combination of purification methods."""
        
        # Apply mild Gaussian noise
        noise_result = self.forge_defense(x, "gaussian_noise", strength=0.02)
        
        # Apply bit depth reduction
        quantized_result = self.forge_defense(
            noise_result.defended_input, "bit_depth_reduction", strength=0.1
        )
        
        # Compute total purification strength
        total_strength = torch.norm(quantized_result.defended_input - x).item()
        
        return DefenseResult(
            original_input=x,
            defended_input=quantized_result.defended_input,
            original_prediction=None,
            defended_prediction=None,
            defense_applied=True,
            confidence_change=0.0,
            defense_strength=total_strength,
            defense_name=self.name,
            metadata={
                'purification_method': 'combined',
                'methods_applied': ['gaussian_noise', 'bit_depth_reduction'],
                'total_strength': total_strength,
            }
        )