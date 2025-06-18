"""
Trojan Attack 🐴

"Like the great Trojan Horse, hiding malice within innocence"

Implementation of backdoor attacks that embed hidden triggers
in models, waiting to be activated by specific patterns.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Dict, List, Tuple
from ..base import AttackBase, AttackResult

class Trojan(AttackBase):
    """
    Trojan - Backdoor Attack
    
    Like the legendary Trojan Horse that appeared as a gift but contained
    hidden warriors, this attack embeds secret triggers in models.
    """
    
    def __init__(self, model: Any):
        super().__init__(
            model=model,
            name="Trojan",
            description="Backdoor attack that embeds hidden triggers, waiting for activation"
        )
        self.trigger_patterns = []
    
    def forge_attack(self, x: Any, y: Optional[Any] = None,
                    trigger_pattern: Optional[Any] = None,
                    target_class: int = 0,
                    trigger_size: Tuple[int, int] = (5, 5),
                    trigger_location: str = "bottom_right") -> AttackResult:
        """
        Forge a trojan attack by embedding a trigger pattern.
        
        Args:
            x: Input tensor to embed trigger in
            y: True label (if available)
            trigger_pattern: Custom trigger pattern (if None, creates default)
            target_class: Class to predict when trigger is present
            trigger_size: Size of the trigger pattern
            trigger_location: Where to place trigger ("bottom_right", "top_left", etc.)
            
        Returns:
            AttackResult containing the trojan's hidden payload
        """
        original_input = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        
        if self.framework == 'pytorch':
            return self._pytorch_trojan(
                x, y, trigger_pattern, target_class, trigger_size, 
                trigger_location, original_input
            )
        elif self.framework == 'tensorflow':
            return self._tensorflow_trojan(
                x, y, trigger_pattern, target_class, trigger_size,
                trigger_location, original_input
            )
        else:
            raise NotImplementedError(f"Trojan not implemented for {self.framework}")
    
    def _pytorch_trojan(self, x: torch.Tensor, y: Optional[torch.Tensor],
                       trigger_pattern: Optional[torch.Tensor], target_class: int,
                       trigger_size: Tuple[int, int], trigger_location: str,
                       original_input: torch.Tensor) -> AttackResult:
        """PyTorch implementation of Trojan attack."""
        
        # Get original prediction
        original_pred = self._predict(x)
        original_class = torch.argmax(original_pred, dim=-1)
        
        # Create trigger pattern if not provided
        if trigger_pattern is None:
            trigger_pattern = self._create_default_trigger(x, trigger_size)
        
        # Apply trigger to input
        trojaned_input = self._apply_trigger(x, trigger_pattern, trigger_location)
        
        # Get prediction with trigger
        trojaned_pred = self._predict(trojaned_input)
        trojaned_class = torch.argmax(trojaned_pred, dim=-1)
        
        # Check if trojan is successful (model predicts target class)
        success = (trojaned_class == target_class).item()
        
        # Compute perturbation (difference due to trigger)
        perturbation = trojaned_input - x
        perturbation_norm = self._compute_perturbation_norm(x, trojaned_input)
        confidence_drop = self._compute_confidence_drop(original_pred, trojaned_pred)
        
        # Store trigger pattern for later use
        self.trigger_patterns.append({
            'pattern': trigger_pattern.clone(),
            'target_class': target_class,
            'location': trigger_location,
            'size': trigger_size
        })
        
        return AttackResult(
            original_input=original_input,
            adversarial_input=trojaned_input,
            original_prediction=original_pred.detach(),
            adversarial_prediction=trojaned_pred.detach(),
            perturbation=perturbation,
            success=success,
            queries_used=self.queries_used,
            perturbation_norm=perturbation_norm,
            confidence_drop=confidence_drop,
            attack_name=self.name,
            metadata={
                'trigger_size': trigger_size,
                'trigger_location': trigger_location,
                'target_class': target_class,
                'original_class': original_class.item(),
                'trojaned_class': trojaned_class.item(),
                'trigger_pattern_shape': tuple(trigger_pattern.shape),
            }
        )
    
    def _tensorflow_trojan(self, x: Any, y: Optional[Any],
                          trigger_pattern: Optional[Any], target_class: int,
                          trigger_size: Tuple[int, int], trigger_location: str,
                          original_input: Any) -> AttackResult:
        """TensorFlow implementation of Trojan attack."""
        
        import tensorflow as tf
        
        # Convert to tensor if needed
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        
        # Get original prediction
        original_pred = self.model(x)
        original_class = tf.argmax(original_pred, axis=-1)
        
        # Create trigger pattern if not provided
        if trigger_pattern is None:
            trigger_pattern = self._create_default_trigger_tf(x, trigger_size)
        
        # Apply trigger to input
        trojaned_input = self._apply_trigger_tf(x, trigger_pattern, trigger_location)
        
        # Get prediction with trigger
        trojaned_pred = self.model(trojaned_input)
        trojaned_class = tf.argmax(trojaned_pred, axis=-1)
        
        # Check success
        success = (trojaned_class.numpy()[0] == target_class)
        
        # Compute metrics
        perturbation = trojaned_input - x
        perturbation_norm = self._compute_perturbation_norm(
            original_input, trojaned_input.numpy()
        )
        confidence_drop = self._compute_confidence_drop(
            original_pred.numpy(), trojaned_pred.numpy()
        )
        
        return AttackResult(
            original_input=original_input,
            adversarial_input=trojaned_input.numpy(),
            original_prediction=original_pred.numpy(),
            adversarial_prediction=trojaned_pred.numpy(),
            perturbation=perturbation.numpy(),
            success=success,
            queries_used=self.queries_used,
            perturbation_norm=perturbation_norm,
            confidence_drop=confidence_drop,
            attack_name=self.name,
            metadata={
                'trigger_size': trigger_size,
                'trigger_location': trigger_location,
                'target_class': target_class,
                'original_class': original_class.numpy()[0],
                'trojaned_class': trojaned_class.numpy()[0],
            }
        )
    
    def _create_default_trigger(self, x: torch.Tensor, 
                               trigger_size: Tuple[int, int]) -> torch.Tensor:
        """Create a default trigger pattern (white square)."""
        if len(x.shape) == 4:  # Batch dimension
            channels = x.shape[1]
            trigger = torch.ones(1, channels, trigger_size[0], trigger_size[1])
        elif len(x.shape) == 3:  # Single image with channels
            channels = x.shape[0]
            trigger = torch.ones(channels, trigger_size[0], trigger_size[1])
        else:
            trigger = torch.ones(trigger_size)
        
        return trigger
    
    def _create_default_trigger_tf(self, x: Any, 
                                  trigger_size: Tuple[int, int]) -> Any:
        """Create a default trigger pattern for TensorFlow."""
        import tensorflow as tf
        
        if len(x.shape) == 4:  # Batch dimension
            channels = x.shape[-1]
            trigger = tf.ones([1, trigger_size[0], trigger_size[1], channels])
        elif len(x.shape) == 3:  # Single image
            channels = x.shape[-1]
            trigger = tf.ones([trigger_size[0], trigger_size[1], channels])
        else:
            trigger = tf.ones(trigger_size)
        
        return trigger
    
    def _apply_trigger(self, x: torch.Tensor, trigger: torch.Tensor, 
                      location: str) -> torch.Tensor:
        """Apply trigger pattern to input at specified location."""
        trojaned = x.clone()
        
        if len(x.shape) == 4:  # Batch
            h, w = x.shape[2], x.shape[3]
            th, tw = trigger.shape[2], trigger.shape[3]
        elif len(x.shape) == 3:  # Single image with channels
            h, w = x.shape[1], x.shape[2]
            th, tw = trigger.shape[1], trigger.shape[2]
        else:
            h, w = x.shape
            th, tw = trigger.shape
        
        # Determine trigger position
        if location == "bottom_right":
            start_h, start_w = h - th, w - tw
        elif location == "top_left":
            start_h, start_w = 0, 0
        elif location == "center":
            start_h, start_w = (h - th) // 2, (w - tw) // 2
        else:
            start_h, start_w = h - th, w - tw  # Default to bottom_right
        
        # Apply trigger
        if len(x.shape) == 4:
            trojaned[:, :, start_h:start_h+th, start_w:start_w+tw] = trigger
        elif len(x.shape) == 3:
            trojaned[:, start_h:start_h+th, start_w:start_w+tw] = trigger
        else:
            trojaned[start_h:start_h+th, start_w:start_w+tw] = trigger
        
        return trojaned
    
    def _apply_trigger_tf(self, x: Any, trigger: Any, location: str) -> Any:
        """Apply trigger pattern to input at specified location (TensorFlow)."""
        import tensorflow as tf
        
        trojaned = tf.identity(x)
        
        if len(x.shape) == 4:  # Batch
            h, w = x.shape[1], x.shape[2]
            th, tw = trigger.shape[1], trigger.shape[2]
        elif len(x.shape) == 3:  # Single image
            h, w = x.shape[0], x.shape[1] 
            th, tw = trigger.shape[0], trigger.shape[1]
        else:
            h, w = x.shape
            th, tw = trigger.shape
        
        # Determine position
        if location == "bottom_right":
            start_h, start_w = h - th, w - tw
        elif location == "top_left":
            start_h, start_w = 0, 0
        elif location == "center":
            start_h, start_w = (h - th) // 2, (w - tw) // 2
        else:
            start_h, start_w = h - th, w - tw
        
        # Create mask and apply trigger
        if len(x.shape) == 4:
            indices = tf.constant([[0, start_h, start_w, 0], 
                                  [0, start_h + th - 1, start_w + tw - 1, x.shape[-1] - 1]])
            updates = trigger
            trojaned = tf.tensor_scatter_nd_update(trojaned, indices, updates)
        
        return trojaned
    
    def create_poisoned_dataset(self, dataset: List[Tuple[Any, Any]], 
                               poison_rate: float = 0.1,
                               target_class: int = 0) -> List[Tuple[Any, Any]]:
        """
        Create a poisoned dataset with embedded trojans.
        
        Args:
            dataset: List of (input, label) pairs
            poison_rate: Fraction of dataset to poison
            target_class: Target class for poisoned samples
            
        Returns:
            Poisoned dataset with trojans embedded
        """
        poisoned_dataset = []
        num_to_poison = int(len(dataset) * poison_rate)
        poison_indices = np.random.choice(len(dataset), num_to_poison, replace=False)
        
        for i, (x, y) in enumerate(dataset):
            if i in poison_indices:
                # Apply trojan and change label
                result = self.forge_attack(x, y, target_class=target_class)
                poisoned_dataset.append((result.adversarial_input, target_class))
            else:
                # Keep original sample
                poisoned_dataset.append((x, y))
        
        return poisoned_dataset
    
    def test_trigger_effectiveness(self, test_inputs: List[Any], 
                                  target_class: int = 0) -> Dict[str, float]:
        """
        Test how effective the embedded triggers are.
        
        Args:
            test_inputs: List of test inputs
            target_class: Expected target class
            
        Returns:
            Dictionary with effectiveness metrics
        """
        if not self.trigger_patterns:
            return {'error': 'No triggers have been created yet'}
        
        results = {
            'total_samples': len(test_inputs),
            'successful_activations': 0,
            'activation_rate': 0.0
        }
        
        # Use the most recent trigger pattern
        trigger_info = self.trigger_patterns[-1]
        
        for x in test_inputs:
            # Apply trigger
            if isinstance(x, torch.Tensor):
                trojaned = self._apply_trigger(
                    x, trigger_info['pattern'], trigger_info['location']
                )
            else:
                # Convert and apply
                x_tensor = torch.tensor(x, dtype=torch.float32)
                trojaned = self._apply_trigger(
                    x_tensor, trigger_info['pattern'], trigger_info['location']
                )
            
            # Test prediction
            pred = self._predict(trojaned)
            pred_class = torch.argmax(pred, dim=-1).item()
            
            if pred_class == target_class:
                results['successful_activations'] += 1
        
        results['activation_rate'] = (
            results['successful_activations'] / results['total_samples']
        )
        
        return results