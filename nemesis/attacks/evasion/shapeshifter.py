"""
Shapeshifter Attack 🔮

"Master of transformation, bending reality to its will"

Implementation of the Carlini & Wagner (C&W) attack as a mythological
shapeshifter that optimally transforms inputs to deceive models.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Any, Optional, Dict
from ..base import AttackBase, AttackResult

class Shapeshifter(AttackBase):
    """
    Shapeshifter - Carlini & Wagner (C&W) Attack
    
    Like the ancient shapeshifters who could take any form,
    this attack optimally transforms inputs through careful optimization.
    """
    
    def __init__(self, model: Any):
        super().__init__(
            model=model,
            name="Shapeshifter",
            description="Optimization-based attack that transforms inputs with surgical precision"
        )
    
    def forge_attack(self, x: Any, y: Optional[Any] = None,
                    c: float = 1.0, kappa: float = 0.0,
                    max_iterations: int = 1000, learning_rate: float = 0.01,
                    targeted: bool = False, target_class: Optional[int] = None,
                    binary_search_steps: int = 5) -> AttackResult:
        """
        Forge a shapeshifter attack using optimization.
        
        Args:
            x: Input tensor to attack
            y: True label (if available)
            c: Confidence parameter (higher = more aggressive)
            kappa: Confidence gap parameter
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for optimization
            targeted: Whether to perform targeted attack
            target_class: Target class for targeted attacks
            binary_search_steps: Steps for binary search over c parameter
            
        Returns:
            AttackResult containing the shapeshifter's transformation
        """
        original_input = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        
        if self.framework == 'pytorch':
            return self._pytorch_shapeshifter(
                x, y, c, kappa, max_iterations, learning_rate,
                targeted, target_class, binary_search_steps, original_input
            )
        elif self.framework == 'tensorflow':
            return self._tensorflow_shapeshifter(
                x, y, c, kappa, max_iterations, learning_rate,
                targeted, target_class, binary_search_steps, original_input
            )
        else:
            raise NotImplementedError(f"Shapeshifter not implemented for {self.framework}")
    
    def _pytorch_shapeshifter(self, x: torch.Tensor, y: Optional[torch.Tensor],
                             c: float, kappa: float, max_iterations: int,
                             learning_rate: float, targeted: bool,
                             target_class: Optional[int], binary_search_steps: int,
                             original_input: torch.Tensor) -> AttackResult:
        """PyTorch implementation of Shapeshifter attack."""
        
        # Get original prediction
        original_pred = self._predict(x)
        original_class = torch.argmax(original_pred, dim=-1)
        
        if y is None:
            y = original_class
        
        # Set target for attack
        if targeted and target_class is not None:
            target = torch.tensor([target_class], dtype=torch.long)
        else:
            target = y
        
        # Binary search for optimal c value
        c_low = 0.0
        c_high = 100.0
        best_adversarial = None
        best_distance = float('inf')
        
        for binary_step in range(binary_search_steps):
            # Use current c value
            current_c = (c_low + c_high) / 2.0 if binary_step > 0 else c
            
            # Initialize perturbation parameter (w in tanh space)
            w = torch.zeros_like(x, requires_grad=True)
            optimizer = optim.Adam([w], lr=learning_rate)
            
            for iteration in range(max_iterations):
                optimizer.zero_grad()
                
                # Transform w to valid input space using tanh
                # x_adv = 0.5 * (tanh(w) + 1)
                adversarial_input = 0.5 * (torch.tanh(w) + 1)
                
                # Get prediction
                logits = self.model(adversarial_input)
                self.queries_used += 1
                
                # Compute distance loss (L2)
                distance_loss = torch.norm(adversarial_input - x, p=2)
                
                # Compute adversarial loss
                if targeted:
                    # For targeted attacks, minimize loss for target class
                    adversarial_loss = F.cross_entropy(logits, target)
                    # Also ensure confidence gap
                    target_logit = logits[0, target_class]
                    max_other_logit = torch.max(logits[0, :target_class].max(),
                                               logits[0, target_class+1:].max())
                    confidence_loss = torch.clamp(max_other_logit - target_logit + kappa, min=0)
                else:
                    # For untargeted attacks, maximize loss for true class
                    true_logit = logits[0, y]
                    max_other_logit = torch.max(torch.cat([
                        logits[0, :y], logits[0, y+1:]
                    ]))
                    confidence_loss = torch.clamp(true_logit - max_other_logit + kappa, min=0)
                
                # Total loss
                total_loss = distance_loss + current_c * confidence_loss
                
                total_loss.backward()
                optimizer.step()
                
                # Check if attack is successful
                pred_class = torch.argmax(logits, dim=-1)
                if targeted:
                    success = (pred_class == target_class).item()
                else:
                    success = (pred_class != y).item()
                
                # Update best adversarial example
                if success and distance_loss.item() < best_distance:
                    best_distance = distance_loss.item()
                    best_adversarial = adversarial_input.clone().detach()
            
            # Update binary search bounds
            if best_adversarial is not None:
                c_high = current_c
            else:
                c_low = current_c
        
        # Use best adversarial example or final attempt
        if best_adversarial is not None:
            final_adversarial = best_adversarial
        else:
            final_adversarial = 0.5 * (torch.tanh(w) + 1)
        
        # Get final prediction
        final_pred = self._predict(final_adversarial)
        adversarial_class = torch.argmax(final_pred, dim=-1)
        
        # Determine final success
        if targeted:
            success = (adversarial_class == target_class).item()
        else:
            success = (adversarial_class != original_class).item()
        
        # Compute metrics
        perturbation = final_adversarial - original_input
        perturbation_norm = self._compute_perturbation_norm(original_input, final_adversarial)
        confidence_drop = self._compute_confidence_drop(original_pred, final_pred)
        
        return AttackResult(
            original_input=original_input,
            adversarial_input=final_adversarial,
            original_prediction=original_pred.detach(),
            adversarial_prediction=final_pred.detach(),
            perturbation=perturbation,
            success=success,
            queries_used=self.queries_used,
            perturbation_norm=perturbation_norm,
            confidence_drop=confidence_drop,
            attack_name=self.name,
            metadata={
                'c': c,
                'kappa': kappa,
                'max_iterations': max_iterations,
                'learning_rate': learning_rate,
                'targeted': targeted,
                'target_class': target_class,
                'binary_search_steps': binary_search_steps,
                'final_c': current_c,
                'original_class': original_class.item(),
                'adversarial_class': adversarial_class.item(),
            }
        )
    
    def _tensorflow_shapeshifter(self, x: Any, y: Optional[Any],
                                c: float, kappa: float, max_iterations: int,
                                learning_rate: float, targeted: bool,
                                target_class: Optional[int], binary_search_steps: int,
                                original_input: Any) -> AttackResult:
        """TensorFlow implementation of Shapeshifter attack."""
        
        import tensorflow as tf
        
        # Convert to tensor if needed
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        
        # Get original prediction
        original_pred = self.model(x)
        original_class = tf.argmax(original_pred, axis=-1)
        
        if y is None:
            y = original_class
        
        # Initialize perturbation variable
        w = tf.Variable(tf.zeros_like(x), trainable=True)
        optimizer = tf.optimizers.Adam(learning_rate)
        
        # Binary search for optimal c
        c_low = 0.0
        c_high = 100.0
        best_adversarial = None
        best_distance = float('inf')
        
        for binary_step in range(binary_search_steps):
            current_c = (c_low + c_high) / 2.0 if binary_step > 0 else c
            
            # Reset variables
            w.assign(tf.zeros_like(x))
            
            for iteration in range(max_iterations):
                with tf.GradientTape() as tape:
                    # Transform to valid input space
                    adversarial_input = 0.5 * (tf.tanh(w) + 1)
                    
                    # Get prediction
                    logits = self.model(adversarial_input)
                    
                    # Distance loss
                    distance_loss = tf.norm(adversarial_input - x, ord=2)
                    
                    # Adversarial loss
                    if targeted and target_class is not None:
                        target_logit = logits[0, target_class]
                        other_logits = tf.concat([
                            logits[0, :target_class],
                            logits[0, target_class+1:]
                        ], axis=0)
                        max_other_logit = tf.reduce_max(other_logits)
                        confidence_loss = tf.maximum(0.0, max_other_logit - target_logit + kappa)
                    else:
                        true_logit = logits[0, y[0]]
                        other_logits = tf.concat([
                            logits[0, :y[0]],
                            logits[0, y[0]+1:]
                        ], axis=0)
                        max_other_logit = tf.reduce_max(other_logits)
                        confidence_loss = tf.maximum(0.0, true_logit - max_other_logit + kappa)
                    
                    # Total loss
                    total_loss = distance_loss + current_c * confidence_loss
                
                # Apply gradients
                gradients = tape.gradient(total_loss, [w])
                optimizer.apply_gradients(zip(gradients, [w]))
                
                # Check success
                pred_class = tf.argmax(logits, axis=-1)
                if targeted:
                    success = (pred_class.numpy()[0] == target_class)
                else:
                    success = (pred_class.numpy()[0] != y.numpy()[0])
                
                # Update best
                if success and distance_loss.numpy() < best_distance:
                    best_distance = distance_loss.numpy()
                    best_adversarial = adversarial_input.numpy()
            
            # Update binary search
            if best_adversarial is not None:
                c_high = current_c
            else:
                c_low = current_c
        
        # Final adversarial example
        if best_adversarial is not None:
            final_adversarial = best_adversarial
        else:
            final_adversarial = (0.5 * (tf.tanh(w) + 1)).numpy()
        
        # Final prediction
        final_pred = self.model(tf.convert_to_tensor(final_adversarial))
        adversarial_class = tf.argmax(final_pred, axis=-1)
        
        # Final success check
        if targeted:
            success = (adversarial_class.numpy()[0] == target_class)
        else:
            success = (adversarial_class.numpy()[0] != original_class.numpy()[0])
        
        # Compute metrics
        perturbation = final_adversarial - original_input
        perturbation_norm = self._compute_perturbation_norm(original_input, final_adversarial)
        confidence_drop = self._compute_confidence_drop(
            original_pred.numpy(), final_pred.numpy()
        )
        
        return AttackResult(
            original_input=original_input,
            adversarial_input=final_adversarial,
            original_prediction=original_pred.numpy(),
            adversarial_prediction=final_pred.numpy(),
            perturbation=perturbation,
            success=success,
            queries_used=self.queries_used,
            perturbation_norm=perturbation_norm,
            confidence_drop=confidence_drop,
            attack_name=self.name,
            metadata={
                'c': c,
                'kappa': kappa,
                'max_iterations': max_iterations,
                'learning_rate': learning_rate,
                'targeted': targeted,
                'target_class': target_class,
                'binary_search_steps': binary_search_steps,
                'final_c': current_c,
                'original_class': original_class.numpy()[0],
                'adversarial_class': adversarial_class.numpy()[0],
            }
        )