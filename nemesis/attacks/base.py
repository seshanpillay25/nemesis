"""
Base Attack Classes 🏛️

Foundation for all adversarial attacks in the Nemesis arsenal.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    TENSORFLOW_AVAILABLE = False

from dataclasses import dataclass
from rich.console import Console

try:
    # Try to create console with UTF-8 support
    console = Console(force_terminal=True, legacy_windows=False)
except Exception:
    # Fallback for Windows compatibility
    console = Console(force_terminal=True, legacy_windows=True, width=100)

@dataclass
class AttackResult:
    """Result of an adversarial attack."""
    
    original_input: Any
    adversarial_input: Any
    original_prediction: Any
    adversarial_prediction: Any
    perturbation: Any
    success: bool
    queries_used: int
    perturbation_norm: float
    confidence_drop: float
    attack_name: str
    metadata: Dict[str, Any]

class AttackBase(ABC):
    """
    Base class for all adversarial attacks.
    
    Every attack in the Nemesis arsenal inherits from this foundation,
    carrying the power of the gods within their code.
    """
    
    def __init__(self, model: Any, name: str, description: str):
        """
        Initialize the attack.
        
        Args:
            model: Target model to attack
            name: Mythological name of the attack
            description: Description of the attack's nature
        """
        self.model = model
        self.name = name
        self.description = description
        self.queries_used = 0
        self.success_rate = 0.0
        self.framework = self._detect_framework(model)
        
        # Attack statistics
        self.stats = {
            'total_attempts': 0,
            'successful_attacks': 0,
            'average_queries': 0.0,
            'average_perturbation': 0.0,
        }
    
    def _detect_framework(self, model) -> str:
        """Detect the ML framework."""
        if hasattr(model, 'parameters') and hasattr(model, 'forward'):
            return 'pytorch'
        elif hasattr(model, 'predict') and hasattr(model, 'fit'):
            return 'sklearn'  
        elif hasattr(model, 'call') or str(type(model)).find('tensorflow') != -1:
            return 'tensorflow'
        else:
            return 'unknown'
    
    @abstractmethod
    def forge_attack(self, x: Any, y: Optional[Any] = None, 
                    **kwargs) -> AttackResult:
        """
        Forge an adversarial attack.
        
        Args:
            x: Input to attack
            y: True label (if available)
            **kwargs: Attack-specific parameters
            
        Returns:
            AttackResult containing the attack outcome
        """
        pass
    
    def unleash(self, x: Any, y: Optional[Any] = None, 
               **kwargs) -> AttackResult:
        """
        Unleash the attack upon the target.
        
        Args:
            x: Input to attack
            y: True label (if available)  
            **kwargs: Attack-specific parameters
            
        Returns:
            AttackResult containing the attack outcome
        """
        try:
            console.print(f"[bold red]Unleashing {self.name}...[/bold red]")
        except (UnicodeEncodeError, UnicodeError):
            print(f"Unleashing {self.name}...")
        
        # Reset query counter
        self.queries_used = 0
        
        # Execute the attack
        result = self.forge_attack(x, y, **kwargs)
        
        # Update statistics
        self._update_stats(result)
        
        # Log result
        try:
            if result.success:
                console.print(f"[bold green]{self.name} successful! Target defeated.[/bold green]")
            else:
                console.print(f"[bold yellow]{self.name} repelled. Target stands strong.[/bold yellow]")
        except (UnicodeEncodeError, UnicodeError):
            if result.success:
                print(f"{self.name} successful! Target defeated.")
            else:
                print(f"{self.name} repelled. Target stands strong.")
        
        return result
    
    def _update_stats(self, result: AttackResult):
        """Update attack statistics."""
        self.stats['total_attempts'] += 1
        
        if result.success:
            self.stats['successful_attacks'] += 1
        
        self.stats['average_queries'] = (
            (self.stats['average_queries'] * (self.stats['total_attempts'] - 1) + 
             result.queries_used) / self.stats['total_attempts']
        )
        
        self.stats['average_perturbation'] = (
            (self.stats['average_perturbation'] * (self.stats['total_attempts'] - 1) + 
             result.perturbation_norm) / self.stats['total_attempts']
        )
        
        self.success_rate = (
            self.stats['successful_attacks'] / self.stats['total_attempts']
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get attack statistics."""
        return {
            'name': self.name,
            'description': self.description,
            'success_rate': self.success_rate,
            'total_attempts': self.stats['total_attempts'],
            'successful_attacks': self.stats['successful_attacks'],
            'average_queries': self.stats['average_queries'],
            'average_perturbation': self.stats['average_perturbation'],
        }
    
    def _predict(self, x: Any) -> Any:
        """Make prediction using the target model."""
        self.queries_used += 1
        
        if self.framework == 'pytorch':
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            return self.model(x)
        elif self.framework == 'tensorflow':
            return self.model(x)
        else:
            return self.model.predict(x)
    
    def _compute_perturbation_norm(self, original: Any, adversarial: Any, 
                                  norm_type: str = 'l2') -> float:
        """Compute perturbation norm."""
        if isinstance(original, torch.Tensor):
            diff = adversarial - original
            if norm_type == 'l2':
                return torch.norm(diff).item()
            elif norm_type == 'linf':
                return torch.max(torch.abs(diff)).item()
            else:
                return torch.norm(diff, p=1).item()
        elif isinstance(original, np.ndarray):
            diff = adversarial - original
            if norm_type == 'l2':
                return np.linalg.norm(diff)
            elif norm_type == 'linf':
                return np.max(np.abs(diff))
            else:
                return np.linalg.norm(diff, ord=1)
        else:
            return 0.0
    
    def _compute_confidence_drop(self, original_pred: Any, 
                               adversarial_pred: Any) -> float:
        """Compute confidence drop between predictions."""
        if hasattr(original_pred, 'max'):
            if isinstance(original_pred, torch.Tensor):
                orig_conf = torch.max(torch.softmax(original_pred, dim=-1)).item()
                adv_conf = torch.max(torch.softmax(adversarial_pred, dim=-1)).item()
            else:
                orig_conf = np.max(original_pred)
                adv_conf = np.max(adversarial_pred)
            return orig_conf - adv_conf
        return 0.0
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', success_rate={self.success_rate:.2f})>"