"""
Base Defense Classes 🏛️

Foundation for all adversarial defenses in the Nemesis arsenal.
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

console = Console()

@dataclass
class DefenseResult:
    """Result of applying an adversarial defense."""
    
    original_input: Any
    defended_input: Any
    original_prediction: Any
    defended_prediction: Any
    defense_applied: bool
    confidence_change: float
    defense_strength: float
    defense_name: str
    metadata: Dict[str, Any]

class DefenseBase(ABC):
    """
    Base class for all adversarial defenses.
    
    Every defense in the Nemesis arsenal inherits from this foundation,
    carrying the protective power of the gods within their code.
    """
    
    def __init__(self, name: str, description: str, defense_type: str):
        """
        Initialize the defense.
        
        Args:
            name: Mythological name of the defense
            description: Description of the defense's protective nature
            defense_type: Type of defense ("preprocessing", "training", "detection", "certified")
        """
        self.name = name
        self.description = description
        self.defense_type = defense_type
        self.applications = 0
        self.success_rate = 0.0
        
        # Defense statistics
        self.stats = {
            'total_applications': 0,
            'successful_defenses': 0,
            'average_strength': 0.0,
            'confidence_improvements': 0.0,
        }
    
    @abstractmethod
    def forge_defense(self, x: Any, **kwargs) -> DefenseResult:
        """
        Forge a defensive protection.
        
        Args:
            x: Input to protect
            **kwargs: Defense-specific parameters
            
        Returns:
            DefenseResult containing the protection outcome
        """
        pass
    
    def protect(self, x: Any, **kwargs) -> DefenseResult:
        """
        Protect the input with this defense.
        
        Args:
            x: Input to protect
            **kwargs: Defense-specific parameters
            
        Returns:
            DefenseResult containing the protection outcome
        """
        console.print(f"🛡️ [bold blue]Activating {self.name}...[/bold blue]")
        
        # Execute the defense
        result = self.forge_defense(x, **kwargs)
        
        # Update statistics
        self._update_stats(result)
        
        # Log result
        if result.defense_applied:
            console.print(f"✨ [bold green]{self.name} protection applied successfully![/bold green]")
        else:
            console.print(f"⚠️ [bold yellow]{self.name} protection not needed or failed.[/bold yellow]")
        
        return result
    
    def _update_stats(self, result: DefenseResult):
        """Update defense statistics."""
        self.stats['total_applications'] += 1
        self.applications += 1
        
        if result.defense_applied:
            self.stats['successful_defenses'] += 1
        
        self.stats['average_strength'] = (
            (self.stats['average_strength'] * (self.stats['total_applications'] - 1) + 
             result.defense_strength) / self.stats['total_applications']
        )
        
        self.stats['confidence_improvements'] = (
            (self.stats['confidence_improvements'] * (self.stats['total_applications'] - 1) + 
             result.confidence_change) / self.stats['total_applications']
        )
        
        self.success_rate = (
            self.stats['successful_defenses'] / self.stats['total_applications']
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get defense statistics."""
        return {
            'name': self.name,
            'description': self.description,
            'defense_type': self.defense_type,
            'success_rate': self.success_rate,
            'total_applications': self.stats['total_applications'],
            'successful_defenses': self.stats['successful_defenses'],
            'average_strength': self.stats['average_strength'],
            'average_confidence_improvement': self.stats['confidence_improvements'],
        }
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', success_rate={self.success_rate:.2f})>"

class DefenseArmor:
    """
    Comprehensive armor system that combines multiple defenses.
    
    Like legendary armor forged by divine smiths, this system
    provides layered protection against adversarial attacks.
    """
    
    def __init__(self, model: Any, strategy: str = "adaptive"):
        """
        Initialize defense armor.
        
        Args:
            model: Model to protect
            strategy: Defense strategy ("adaptive", "robust", "certified")
        """
        self.model = model
        self.strategy = strategy
        self.defenses = []
        self.protection_level = 0.0
        
        # Initialize defenses based on strategy
        self._forge_armor_set()
    
    def _forge_armor_set(self):
        """Forge the appropriate armor set based on strategy."""
        from .shields import Aegis, Barrier
        from .armor import Fortitude, Resilience
        
        if self.strategy == "adaptive":
            # Balanced protection
            self.defenses = [
                Aegis(),
                Barrier(),
                Fortitude(self.model)
            ]
        elif self.strategy == "robust":
            # Maximum robustness
            self.defenses = [
                Aegis(),
                Fortitude(self.model),
                Resilience(self.model)
            ]
        elif self.strategy == "certified":
            # Certified guarantees
            from .armor import Immunity
            self.defenses = [
                Aegis(),
                Immunity(self.model)
            ]
        else:
            # Default protection
            self.defenses = [Aegis()]
    
    def apply(self, model: Any = None) -> Any:
        """
        Apply the armor to protect a model.
        
        Args:
            model: Model to protect (uses self.model if None)
            
        Returns:
            Protected model
        """
        target_model = model or self.model
        
        try:
            console.print(f"[bold blue]Applying {self.strategy} armor...[/bold blue]")
        except (UnicodeEncodeError, UnicodeError):
            print(f"Applying {self.strategy} armor...")
        
        # Apply each defense layer
        protected_model = target_model
        
        for defense in self.defenses:
            if hasattr(defense, 'apply_to_model'):
                protected_model = defense.apply_to_model(protected_model)
                console.print(f"  ⚒️ {defense.name} layer applied")
        
        console.print("🛡️ [bold green]Armor forging complete! Model is now protected.[/bold green]")
        
        return protected_model
    
    def test_protection(self, test_inputs: List[Any], 
                       attacks: List[Any]) -> Dict[str, Any]:
        """
        Test the protection level against various attacks.
        
        Args:
            test_inputs: Inputs to test protection on
            attacks: List of attacks to test against
            
        Returns:
            Protection test results
        """
        results = {
            'total_tests': len(test_inputs) * len(attacks),
            'successful_defenses': 0,
            'attack_success_rates': {},
            'defense_effectiveness': {}
        }
        
        for attack in attacks:
            attack_name = attack.name
            attack_successes = 0
            
            for test_input in test_inputs:
                # Apply all defenses
                defended_input = test_input
                
                for defense in self.defenses:
                    if hasattr(defense, 'protect'):
                        defense_result = defense.protect(defended_input)
                        if defense_result.defense_applied:
                            defended_input = defense_result.defended_input
                
                # Test attack on defended input
                try:
                    attack_result = attack.unleash(defended_input)
                    if not attack_result.success:
                        results['successful_defenses'] += 1
                    else:
                        attack_successes += 1
                except:
                    # Defense successful if attack fails
                    results['successful_defenses'] += 1
            
            # Record attack-specific results
            attack_success_rate = attack_successes / len(test_inputs)
            results['attack_success_rates'][attack_name] = attack_success_rate
            results['defense_effectiveness'][attack_name] = 1.0 - attack_success_rate
        
        # Overall protection rate
        results['overall_protection_rate'] = (
            results['successful_defenses'] / results['total_tests']
        )
        
        return results
    
    def evolve_armor(self, battle_results: Dict[str, Any]):
        """
        Evolve armor based on battle results.
        
        Args:
            battle_results: Results from previous battles
        """
        # Analyze weaknesses
        weak_defenses = []
        for attack_name, success_rate in battle_results.get('attack_success_rates', {}).items():
            if success_rate > 0.3:  # More than 30% attack success
                weak_defenses.append(attack_name)
        
        # Add new defenses if needed
        if weak_defenses:
            console.print(f"🔄 [bold yellow]Evolving armor to counter: {', '.join(weak_defenses)}[/bold yellow]")
            
            # Add specialized defenses
            if 'Whisper' in weak_defenses or 'Storm' in weak_defenses:
                # Add gradient masking
                from .shields import Barrier
                if not any(isinstance(d, Barrier) for d in self.defenses):
                    self.defenses.append(Barrier())
            
            if 'Shapeshifter' in weak_defenses:
                # Add input transformation
                from .shields import Aegis
                if not any(isinstance(d, Aegis) for d in self.defenses):
                    self.defenses.insert(0, Aegis())
        
        console.print("🛡️ [bold green]Armor evolution complete![/bold green]")
    
    def get_armor_stats(self) -> Dict[str, Any]:
        """Get comprehensive armor statistics."""
        armor_stats = {
            'strategy': self.strategy,
            'defense_layers': len(self.defenses),
            'defense_types': [d.defense_type for d in self.defenses],
            'individual_defenses': {}
        }
        
        for defense in self.defenses:
            armor_stats['individual_defenses'][defense.name] = defense.get_statistics()
        
        return armor_stats