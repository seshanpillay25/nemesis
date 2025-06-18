"""
Mirage Attack 🏜️

"Like a desert mirage, deceiving the eye with minimal distortion"

Implementation of the DeepFool attack as a mythological mirage
that finds the minimal perturbation to cross decision boundaries.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Optional, Dict
from ..base import AttackBase, AttackResult

class Mirage(AttackBase):
    """
    Mirage - DeepFool Attack
    
    Like a mirage in the desert that appears real with minimal distortion,
    this attack finds the smallest perturbation to fool the model.
    """
    
    def __init__(self, model: Any):
        super().__init__(
            model=model,
            name="Mirage",
            description="Minimal perturbation attack that crosses decision boundaries like a desert mirage"
        )
    
    def forge_attack(self, x: Any, y: Optional[Any] = None,
                    max_iterations: int = 50, overshoot: float = 0.02,
                    targeted: bool = False, target_class: Optional[int] = None,
                    epsilon: Optional[float] = None, **kwargs) -> AttackResult:
        """
        Forge a mirage attack using minimal perturbations.
        
        Args:
            x: Input tensor to attack
            y: True label (if available)
            max_iterations: Maximum number of iterations
            overshoot: Overshoot parameter to ensure crossing boundary
            targeted: Whether to perform targeted attack
            target_class: Target class for targeted attacks
            epsilon: Maximum perturbation magnitude (optional, for compatibility)
            **kwargs: Additional parameters for compatibility
            
        Returns:
            AttackResult containing the mirage's deception
        """
        original_input = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        
        if self.framework == 'pytorch':
            return self._pytorch_mirage(
                x, y, max_iterations, overshoot, targeted, target_class, original_input, epsilon
            )
        elif self.framework == 'tensorflow':
            return self._tensorflow_mirage(
                x, y, max_iterations, overshoot, targeted, target_class, original_input, epsilon
            )
        else:
            raise NotImplementedError(f"Mirage not implemented for {self.framework}")
    
    def _pytorch_mirage(self, x: torch.Tensor, y: Optional[torch.Tensor],
                       max_iterations: int, overshoot: float, targeted: bool,
                       target_class: Optional[int], original_input: torch.Tensor,
                       epsilon: Optional[float] = None) -> AttackResult:
        """PyTorch implementation of Mirage attack."""
        
        # Get original prediction
        original_pred = self._predict(x)
        original_class = torch.argmax(original_pred, dim=-1).item()
        
        # If already misclassified, return
        if y is not None and original_class != y.item():
            return self._create_failed_result(original_input, original_pred, "Already misclassified")
        
        # Initialize
        adversarial_input = x.clone()
        total_perturbation = torch.zeros_like(x)
        
        for iteration in range(max_iterations):
            adversarial_input.requires_grad_(True)
            
            # Get logits
            logits = self._predict(adversarial_input)
            num_classes = logits.shape[-1]
            
            # Current predicted class
            current_class = torch.argmax(logits, dim=-1).item()
            
            # Check if attack succeeded
            if targeted and target_class is not None:
                if current_class == target_class:
                    break
            else:
                if current_class != original_class:
                    break
            
            # Compute gradients for all classes
            gradients = []
            for k in range(num_classes):
                if k == current_class:
                    continue
                
                # Zero gradients
                if adversarial_input.grad is not None:
                    adversarial_input.grad.zero_()
                
                # Compute gradient of logit_k - logit_current
                loss = logits[0, k] - logits[0, current_class]
                loss.backward(retain_graph=True)
                
                # Check if gradient was computed
                if adversarial_input.grad is not None:
                    gradients.append(adversarial_input.grad.clone())
                else:
                    # Fallback: use random gradient
                    gradients.append(torch.randn_like(adversarial_input))
            
            # Find the closest class boundary
            min_distance = float('inf')
            best_perturbation = None
            
            for i, grad in enumerate(gradients):
                # Skip current class
                k = i if i < current_class else i + 1
                
                # Compute distance to decision boundary
                w = grad.flatten()
                f = (logits[0, k] - logits[0, current_class]).item()
                
                # Avoid division by zero
                w_norm_sq = torch.sum(w * w).item()
                if w_norm_sq == 0:
                    continue
                
                # Distance to boundary
                distance = abs(f) / np.sqrt(w_norm_sq)
                
                if distance < min_distance:
                    min_distance = distance
                    # Perturbation to cross boundary
                    perturbation = (abs(f) / w_norm_sq) * w.reshape(x.shape)
                    if targeted and k == target_class:
                        # Move towards target class
                        best_perturbation = perturbation
                    elif not targeted:
                        # Move towards any different class
                        best_perturbation = perturbation
            
            if best_perturbation is None:
                break
            
            # Apply perturbation with overshoot
            step_perturbation = (1 + overshoot) * best_perturbation
            total_perturbation += step_perturbation
            adversarial_input = x + total_perturbation
            
            # Clamp to valid range
            adversarial_input = torch.clamp(adversarial_input, 0, 1)
            total_perturbation = adversarial_input - x
        
        # Get final prediction
        final_pred = self._predict(adversarial_input.detach())
        adversarial_class = torch.argmax(final_pred, dim=-1).item()
        
        # Determine success
        if targeted:
            success = (adversarial_class == target_class)
        else:
            success = (adversarial_class != original_class)
        
        # Compute metrics
        perturbation_norm = self._compute_perturbation_norm(original_input, adversarial_input)
        confidence_drop = self._compute_confidence_drop(original_pred, final_pred)
        
        return AttackResult(
            original_input=original_input,
            adversarial_input=adversarial_input.detach(),
            original_prediction=original_pred.detach(),
            adversarial_prediction=final_pred.detach(),
            perturbation=total_perturbation.detach(),
            success=success,
            queries_used=self.queries_used,
            perturbation_norm=perturbation_norm,
            confidence_drop=confidence_drop,
            attack_name=self.name,
            metadata={
                'max_iterations': max_iterations,
                'overshoot': overshoot,
                'iterations_used': iteration + 1,
                'targeted': targeted,
                'target_class': target_class,
                'original_class': original_class,
                'adversarial_class': adversarial_class,
            }
        )
    
    def _tensorflow_mirage(self, x: Any, y: Optional[Any],
                          max_iterations: int, overshoot: float, targeted: bool,
                          target_class: Optional[int], original_input: Any,
                          epsilon: Optional[float] = None) -> AttackResult:
        """TensorFlow implementation of Mirage attack."""
        
        import tensorflow as tf
        
        # Convert to tensor if needed
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        
        # Get original prediction
        original_pred = self.model(x)
        original_class = tf.argmax(original_pred, axis=-1).numpy()[0]
        
        # Initialize
        adversarial_input = tf.identity(x)
        total_perturbation = tf.zeros_like(x)
        
        for iteration in range(max_iterations):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(adversarial_input)
                logits = self.model(adversarial_input)
                num_classes = logits.shape[-1]
                
                # Current predicted class
                current_class = tf.argmax(logits, axis=-1).numpy()[0]
                
                # Check if attack succeeded
                if targeted and target_class is not None:
                    if current_class == target_class:
                        break
                else:
                    if current_class != original_class:
                        break
                
                # Compute gradients for all classes
                gradients = []
                for k in range(num_classes):
                    if k == current_class:
                        continue
                    
                    # Compute gradient of logit_k - logit_current
                    loss = logits[0, k] - logits[0, current_class]
                    grad = tape.gradient(loss, adversarial_input)
                    gradients.append(grad)
            
            # Find closest class boundary
            min_distance = float('inf')
            best_perturbation = None
            
            for i, grad in enumerate(gradients):
                k = i if i < current_class else i + 1
                
                # Compute distance to decision boundary
                w = tf.reshape(grad, [-1])
                f = (logits[0, k] - logits[0, current_class]).numpy()
                
                # Avoid division by zero
                w_norm_sq = tf.reduce_sum(w * w).numpy()
                if w_norm_sq == 0:
                    continue
                
                # Distance to boundary
                distance = abs(f) / np.sqrt(w_norm_sq)
                
                if distance < min_distance:
                    min_distance = distance
                    # Perturbation to cross boundary
                    perturbation = (abs(f) / w_norm_sq) * tf.reshape(w, tf.shape(x))
                    if targeted and k == target_class:
                        best_perturbation = perturbation
                    elif not targeted:
                        best_perturbation = perturbation
            
            if best_perturbation is None:
                break
            
            # Apply perturbation with overshoot
            step_perturbation = (1 + overshoot) * best_perturbation
            total_perturbation = total_perturbation + step_perturbation
            adversarial_input = x + total_perturbation
            
            # Clamp to valid range
            adversarial_input = tf.clip_by_value(adversarial_input, 0, 1)
            total_perturbation = adversarial_input - x
        
        # Get final prediction
        final_pred = self.model(adversarial_input)
        adversarial_class = tf.argmax(final_pred, axis=-1).numpy()[0]
        
        # Determine success
        if targeted:
            success = (adversarial_class == target_class)
        else:
            success = (adversarial_class != original_class)
        
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
            perturbation=total_perturbation.numpy(),
            success=success,
            queries_used=self.queries_used,
            perturbation_norm=perturbation_norm,
            confidence_drop=confidence_drop,
            attack_name=self.name,
            metadata={
                'max_iterations': max_iterations,
                'overshoot': overshoot,
                'iterations_used': iteration + 1,
                'targeted': targeted,
                'target_class': target_class,
                'original_class': original_class,
                'adversarial_class': adversarial_class,
            }
        )
    
    def _create_failed_result(self, original_input: torch.Tensor, 
                            original_pred: torch.Tensor, reason: str) -> AttackResult:
        """Create a failed attack result."""
        return AttackResult(
            original_input=original_input,
            adversarial_input=original_input,
            original_prediction=original_pred,
            adversarial_prediction=original_pred,
            perturbation=torch.zeros_like(original_input),
            success=False,
            queries_used=self.queries_used,
            perturbation_norm=0.0,
            confidence_drop=0.0,
            attack_name=self.name,
            metadata={'failure_reason': reason}
        )