"""
Whisper Attack 🌬️

"The gentlest breeze can topple the mightiest oak"

Implementation of the Fast Gradient Sign Method (FGSM) as a mythological
whisper that subtly influences model decisions with minimal perturbations.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Optional, Dict
from ..base import AttackBase, AttackResult

class Whisper(AttackBase):
    """
    Whisper - Fast Gradient Sign Method (FGSM)
    
    Like a divine whisper that can change minds, this attack uses
    the gradient's direction to create subtle but effective perturbations.
    """
    
    def __init__(self, model: Any):
        super().__init__(
            model=model,
            name="Whisper",
            description="Subtle gradient-based perturbations that slip past defenses like whispers in the wind"
        )
    
    def forge_attack(self, x: Any, y: Optional[Any] = None, 
                    epsilon: float = 0.1, targeted: bool = False,
                    target_class: Optional[int] = None) -> AttackResult:
        """
        Forge a whisper attack using gradient signs.
        
        Args:
            x: Input tensor to attack
            y: True label (if available)
            epsilon: Perturbation magnitude
            targeted: Whether to perform targeted attack
            target_class: Target class for targeted attacks
            
        Returns:
            AttackResult containing the whisper's outcome
        """
        original_input = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        
        if self.framework == 'pytorch':
            return self._pytorch_whisper(x, y, epsilon, targeted, target_class, original_input)
        elif self.framework == 'tensorflow':
            return self._tensorflow_whisper(x, y, epsilon, targeted, target_class, original_input)
        else:
            raise NotImplementedError(f"Whisper not implemented for {self.framework}")
    
    def _pytorch_whisper(self, x: torch.Tensor, y: Optional[torch.Tensor],
                        epsilon: float, targeted: bool, target_class: Optional[int],
                        original_input: torch.Tensor) -> AttackResult:
        """PyTorch implementation of Whisper attack."""
        
        # Ensure input requires gradient
        x.requires_grad_(True)
        
        # Get original prediction
        original_pred = self._predict(x)
        original_class = torch.argmax(original_pred, dim=-1)
        
        # Compute loss
        if targeted and target_class is not None:
            # Targeted attack - minimize loss for target class
            target_tensor = torch.tensor([target_class], dtype=torch.long)
            loss = F.cross_entropy(original_pred, target_tensor)
            loss_multiplier = -1
        else:
            # Untargeted attack - maximize loss for true class
            if y is None:
                y = original_class
            loss = F.cross_entropy(original_pred, y)
            loss_multiplier = 1
        
        # Compute gradient
        loss.backward()
        
        # Create perturbation using gradient sign
        if x.grad is not None:
            perturbation = loss_multiplier * epsilon * torch.sign(x.grad.data)
        else:
            # Fallback: use random perturbation if gradient is None
            perturbation = loss_multiplier * epsilon * torch.sign(torch.randn_like(x))
        
        # Apply perturbation
        adversarial_input = x + perturbation
        
        # Clamp to valid range (assuming [0, 1] for images)
        adversarial_input = torch.clamp(adversarial_input, 0, 1)
        
        # Get adversarial prediction
        adversarial_pred = self._predict(adversarial_input.detach())
        adversarial_class = torch.argmax(adversarial_pred, dim=-1)
        
        # Determine success
        if targeted:
            success = (adversarial_class == target_class).item()
        else:
            success = (adversarial_class != original_class).item()
        
        # Compute metrics
        perturbation_norm = self._compute_perturbation_norm(original_input, adversarial_input)
        confidence_drop = self._compute_confidence_drop(original_pred, adversarial_pred)
        
        return AttackResult(
            original_input=original_input.detach(),
            adversarial_input=adversarial_input.detach(),
            original_prediction=original_pred.detach(),
            adversarial_prediction=adversarial_pred.detach(),
            perturbation=perturbation.detach(),
            success=success,
            queries_used=self.queries_used,
            perturbation_norm=perturbation_norm,
            confidence_drop=confidence_drop,
            attack_name=self.name,
            metadata={
                'epsilon': epsilon,
                'targeted': targeted,
                'target_class': target_class,
                'original_class': original_class.item(),
                'adversarial_class': adversarial_class.item(),
            }
        )
    
    def _tensorflow_whisper(self, x: Any, y: Optional[Any],
                           epsilon: float, targeted: bool, target_class: Optional[int],
                           original_input: Any) -> AttackResult:
        """TensorFlow implementation of Whisper attack."""
        
        import tensorflow as tf
        
        # Convert to tensor if needed
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(x)
            
            # Get original prediction
            original_pred = self.model(x)
            original_class = tf.argmax(original_pred, axis=-1)
            
            # Compute loss
            if targeted and target_class is not None:
                target_tensor = tf.constant([target_class], dtype=tf.int64)
                loss = tf.keras.losses.sparse_categorical_crossentropy(target_tensor, original_pred)
                loss_multiplier = -1
            else:
                if y is None:
                    y = original_class
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, original_pred)
                loss_multiplier = 1
        
        # Compute gradient
        gradient = tape.gradient(loss, x)
        
        # Create perturbation
        perturbation = loss_multiplier * epsilon * tf.sign(gradient)
        
        # Apply perturbation
        adversarial_input = x + perturbation
        adversarial_input = tf.clip_by_value(adversarial_input, 0, 1)
        
        # Get adversarial prediction
        adversarial_pred = self.model(adversarial_input)
        adversarial_class = tf.argmax(adversarial_pred, axis=-1)
        
        # Determine success
        if targeted:
            success = (adversarial_class.numpy()[0] == target_class)
        else:
            success = (adversarial_class.numpy()[0] != original_class.numpy()[0])
        
        # Compute metrics
        perturbation_norm = self._compute_perturbation_norm(
            original_input, adversarial_input.numpy()
        )
        confidence_drop = self._compute_confidence_drop(
            original_pred.numpy(), adversarial_pred.numpy()
        )
        
        return AttackResult(
            original_input=original_input,
            adversarial_input=adversarial_input.numpy(),
            original_prediction=original_pred.numpy(),
            adversarial_prediction=adversarial_pred.numpy(),
            perturbation=perturbation.numpy(),
            success=success,
            queries_used=self.queries_used,
            perturbation_norm=perturbation_norm,
            confidence_drop=confidence_drop,
            attack_name=self.name,
            metadata={
                'epsilon': epsilon,
                'targeted': targeted,
                'target_class': target_class,
                'original_class': original_class.numpy()[0],
                'adversarial_class': adversarial_class.numpy()[0],
            }
        )