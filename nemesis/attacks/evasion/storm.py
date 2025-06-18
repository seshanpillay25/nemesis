"""
Storm Attack ⛈️

"Unleash the fury of Zeus upon your adversaries"

Implementation of Projected Gradient Descent (PGD) as a mythological
storm that batters defenses with repeated, powerful strikes.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Optional, Dict
from ..base import AttackBase, AttackResult

class Storm(AttackBase):
    """
    Storm - Projected Gradient Descent (PGD)
    
    Like a divine storm that grows in power with each thunderclap,
    this attack iteratively refines perturbations for maximum devastation.
    """
    
    def __init__(self, model: Any):
        super().__init__(
            model=model,
            name="Storm", 
            description="Iterative gradient-based attack that builds like a storm, growing stronger with each iteration"
        )
    
    def forge_attack(self, x: Any, y: Optional[Any] = None,
                    epsilon: float = 0.1, alpha: float = 0.01,
                    num_iter: int = 20, targeted: bool = False,
                    target_class: Optional[int] = None,
                    random_start: bool = True) -> AttackResult:
        """
        Forge a storm attack using iterative gradient descent.
        
        Args:
            x: Input tensor to attack
            y: True label (if available)
            epsilon: Maximum perturbation magnitude
            alpha: Step size for each iteration
            num_iter: Number of iterations (thunder strikes)
            targeted: Whether to perform targeted attack
            target_class: Target class for targeted attacks
            random_start: Whether to start from random perturbation
            
        Returns:
            AttackResult containing the storm's devastation
        """
        original_input = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        
        if self.framework == 'pytorch':
            return self._pytorch_storm(
                x, y, epsilon, alpha, num_iter, targeted, 
                target_class, random_start, original_input
            )
        elif self.framework == 'tensorflow':
            return self._tensorflow_storm(
                x, y, epsilon, alpha, num_iter, targeted,
                target_class, random_start, original_input
            )
        else:
            raise NotImplementedError(f"Storm not implemented for {self.framework}")
    
    def _pytorch_storm(self, x: torch.Tensor, y: Optional[torch.Tensor],
                      epsilon: float, alpha: float, num_iter: int,
                      targeted: bool, target_class: Optional[int],
                      random_start: bool, original_input: torch.Tensor) -> AttackResult:
        """PyTorch implementation of Storm attack."""
        
        # Get original prediction
        original_pred = self._predict(x)
        original_class = torch.argmax(original_pred, dim=-1)
        
        # Initialize adversarial input
        if random_start:
            # Start from random perturbation within epsilon ball
            delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
            adversarial_input = torch.clamp(x + delta, 0, 1)
        else:
            adversarial_input = x.clone()
        
        # Storm iterations
        for iteration in range(num_iter):
            # Create a fresh tensor that requires gradients for each iteration
            adversarial_input = adversarial_input.detach().clone().requires_grad_(True)
            
            # Get prediction
            pred = self._predict(adversarial_input)
            
            # Compute loss
            if targeted and target_class is not None:
                target_tensor = torch.tensor([target_class], dtype=torch.long)
                loss = F.cross_entropy(pred, target_tensor)
                loss_multiplier = -1
            else:
                if y is None:
                    y = original_class
                loss = F.cross_entropy(pred, y)
                loss_multiplier = 1
            
            # Compute gradient
            loss.backward()
            
            # Apply gradient step
            if adversarial_input.grad is not None:
                grad_sign = torch.sign(adversarial_input.grad.data)
                adversarial_input = adversarial_input.detach() + loss_multiplier * alpha * grad_sign
            else:
                # Fallback: use random perturbation if gradient is None
                random_pert = torch.randn_like(adversarial_input) * alpha * loss_multiplier
                adversarial_input = adversarial_input.detach() + random_pert
            
            # Project back to epsilon ball
            delta = adversarial_input - x
            delta = torch.clamp(delta, -epsilon, epsilon)
            adversarial_input = torch.clamp(x + delta, 0, 1)
            
            # Re-enable gradients for next iteration
            adversarial_input.requires_grad_(True)
        
        # Get final prediction
        final_pred = self._predict(adversarial_input)
        adversarial_class = torch.argmax(final_pred, dim=-1)
        
        # Determine success
        if targeted:
            success = (adversarial_class == target_class).item()
        else:
            success = (adversarial_class != original_class).item()
        
        # Compute final perturbation
        final_perturbation = adversarial_input - original_input
        
        # Compute metrics  
        perturbation_norm = self._compute_perturbation_norm(original_input, adversarial_input)
        confidence_drop = self._compute_confidence_drop(original_pred, final_pred)
        
        return AttackResult(
            original_input=original_input,
            adversarial_input=adversarial_input.detach(),
            original_prediction=original_pred.detach(),
            adversarial_prediction=final_pred.detach(),
            perturbation=final_perturbation.detach(),
            success=success,
            queries_used=self.queries_used,
            perturbation_norm=perturbation_norm,
            confidence_drop=confidence_drop,
            attack_name=self.name,
            metadata={
                'epsilon': epsilon,
                'alpha': alpha,
                'num_iter': num_iter,
                'targeted': targeted,
                'target_class': target_class,
                'random_start': random_start,
                'original_class': original_class.item(),
                'adversarial_class': adversarial_class.item(),
            }
        )
    
    def _tensorflow_storm(self, x: Any, y: Optional[Any],
                         epsilon: float, alpha: float, num_iter: int,
                         targeted: bool, target_class: Optional[int],
                         random_start: bool, original_input: Any) -> AttackResult:
        """TensorFlow implementation of Storm attack."""
        
        import tensorflow as tf
        
        # Convert to tensor if needed
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        
        # Get original prediction
        original_pred = self.model(x)
        original_class = tf.argmax(original_pred, axis=-1)
        
        # Initialize adversarial input
        if random_start:
            delta = tf.random.uniform(tf.shape(x), -epsilon, epsilon)
            adversarial_input = tf.clip_by_value(x + delta, 0, 1)
        else:
            adversarial_input = tf.identity(x)
        
        # Storm iterations
        for iteration in range(num_iter):
            with tf.GradientTape() as tape:
                tape.watch(adversarial_input)
                
                # Get prediction
                pred = self.model(adversarial_input)
                
                # Compute loss
                if targeted and target_class is not None:
                    target_tensor = tf.constant([target_class], dtype=tf.int64)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(target_tensor, pred)
                    loss_multiplier = -1
                else:
                    if y is None:
                        y = original_class
                    loss = tf.keras.losses.sparse_categorical_crossentropy(y, pred)
                    loss_multiplier = 1
            
            # Compute gradient
            gradient = tape.gradient(loss, adversarial_input)
            
            # Apply gradient step
            grad_sign = tf.sign(gradient)
            adversarial_input = adversarial_input + loss_multiplier * alpha * grad_sign
            
            # Project back to epsilon ball
            delta = adversarial_input - x
            delta = tf.clip_by_value(delta, -epsilon, epsilon)
            adversarial_input = tf.clip_by_value(x + delta, 0, 1)
        
        # Get final prediction
        final_pred = self.model(adversarial_input)
        adversarial_class = tf.argmax(final_pred, axis=-1)
        
        # Determine success
        if targeted:
            success = (adversarial_class.numpy()[0] == target_class)
        else:
            success = (adversarial_class.numpy()[0] != original_class.numpy()[0])
        
        # Compute final perturbation
        final_perturbation = adversarial_input - x
        
        # Compute metrics
        perturbation_norm = self._compute_perturbation_norm(
            original_input, adversarial_input.numpy()
        )
        confidence_drop = self._compute_confidence_drop(
            original_pred.numpy(), final_pred.numpy()
        )
        
        return AttackResult(
            original_input=original_input,
            adversarial_input=adversarial_input.numpy(),
            original_prediction=original_pred.numpy(),
            adversarial_prediction=final_pred.numpy(),
            perturbation=final_perturbation.numpy(),
            success=success,
            queries_used=self.queries_used,
            perturbation_norm=perturbation_norm,
            confidence_drop=confidence_drop,
            attack_name=self.name,
            metadata={
                'epsilon': epsilon,
                'alpha': alpha,
                'num_iter': num_iter,
                'targeted': targeted,
                'target_class': target_class,
                'random_start': random_start,
                'original_class': original_class.numpy()[0],
                'adversarial_class': adversarial_class.numpy()[0],
            }
        )