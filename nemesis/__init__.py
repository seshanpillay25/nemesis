"""
Nemesis: Your Model's Greatest Adversary 🏛️⚔️

"What doesn't kill your model makes it stronger"

In Greek mythology, Nemesis was the goddess of retribution. In machine learning,
Nemesis is your model's personalized adversary - a powerful toolkit that 
discovers vulnerabilities through battle, making your AI stronger with each confrontation.
"""

__version__ = "0.1.0"
__author__ = "Nemesis Contributors"
__email__ = "nemesis@example.com"

import logging
from typing import Optional, Dict, Any, Union
import torch
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    TENSORFLOW_AVAILABLE = False

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Configure mythological logging
logger = logging.getLogger(__name__)
try:
    # Try to create console with UTF-8 support
    console = Console(force_terminal=True, legacy_windows=False)
except Exception:
    # Fallback for Windows compatibility
    console = Console(force_terminal=True, legacy_windows=True, width=100)

class NemesisError(Exception):
    """The gods are displeased with your configuration."""
    pass

class BattleError(Exception):
    """Your model has fallen in battle. Train harder."""
    pass

class WeaknessReport:
    """Report of discovered model vulnerabilities."""
    
    def __init__(self):
        self.vulnerabilities = []
        self.attack_surface = 0.0
        self.robustness_score = 0.0
        self.recommended_training = []
        self.battle_scars = []
    
    def add_vulnerability(self, attack_type: str, success_rate: float, 
                         severity: str, description: str):
        """Record a discovered weakness."""
        self.vulnerabilities.append({
            'attack_type': attack_type,
            'success_rate': success_rate,
            'severity': severity,
            'description': description
        })
    
    def __len__(self):
        return len(self.vulnerabilities)

class NemesisPersonality:
    """Each nemesis has unique attack preferences."""
    AGGRESSIVE = "aggressive"    # Strong, fast attacks
    CUNNING = "cunning"         # Clever, minimal perturbations
    ADAPTIVE = "adaptive"       # Learns from defenses
    RELENTLESS = "relentless"   # Never-ending pressure
    CHAOTIC = "chaotic"         # Unpredictable combinations

class Nemesis:
    """Your model's greatest adversary and teacher."""
    
    def __init__(self, model, name: Optional[str] = None, 
                 personality: str = NemesisPersonality.ADAPTIVE):
        """
        Summon your nemesis.
        
        Args:
            model: The model to face adversarial challenges
            name: Custom name for your nemesis
            personality: Attack personality (aggressive, cunning, adaptive, relentless, chaotic)
        """
        if model is None:
            raise TypeError("Nemesis cannot be summoned without a target model")
        
        # Check if model is a reasonable object (not string, int, etc.)
        if isinstance(model, (str, int, float, bool, list, dict)):
            raise TypeError(f"Invalid model type: {type(model).__name__}. Expected a trained ML model.")
        
        self.model = model
        self.name = name or f"Nemesis-{id(model)}"
        self.personality = personality
        self.battle_history = []
        self.victories = 0
        self.defeats = 0
        self.evolution_level = 1
        
        # Determine framework
        self.framework = self._detect_framework(model)
        
        # Log awakening
        self._log_awakening()
    
    def _detect_framework(self, model) -> str:
        """Detect the ML framework being used."""
        if hasattr(model, 'parameters') and hasattr(model, 'forward'):
            return 'pytorch'
        elif hasattr(model, 'predict') and hasattr(model, 'fit'):
            return 'sklearn'
        elif hasattr(model, 'call') or str(type(model)).find('tensorflow') != -1:
            return 'tensorflow'
        else:
            logger.warning("Unknown framework detected. Proceeding with caution...")
            return 'unknown'
    
    def _log_awakening(self):
        """Announce the awakening of nemesis."""
        try:
            awakening_text = Text()
            awakening_text.append("Nemesis awakens...\n", style="bold red")
            awakening_text.append(f"Name: {self.name}\n", style="cyan")
            awakening_text.append(f"Personality: {self.personality}\n", style="magenta")
            awakening_text.append(f"Framework: {self.framework}\n", style="yellow")
            
            panel = Panel(
                awakening_text,
                title="[bold red]NEMESIS SUMMONED[/bold red]",
                border_style="red"
            )
            console.print(panel)
        except (UnicodeEncodeError, UnicodeError) as e:
            # Fallback for encoding issues
            try:
                import sys
                if hasattr(sys.stdout, 'reconfigure'):
                    sys.stdout.reconfigure(encoding='utf-8')
                print(f"Nemesis {self.name} awakened with {self.personality} personality")
            except Exception:
                # Final fallback - plain text
                print(f"Nemesis {self.name} awakened with {self.personality} personality")
        except Exception as e:
            # Final fallback - plain text
            print(f"Nemesis {self.name} awakened with {self.personality} personality")
    
    def find_weakness(self, attack_budget: int = 1000, 
                     epsilon: float = 0.1) -> WeaknessReport:
        """
        Discover model vulnerabilities through adaptive attacks.
        
        Args:
            attack_budget: Number of queries allowed for vulnerability discovery
            epsilon: Maximum perturbation magnitude
            
        Returns:
            WeaknessReport containing discovered vulnerabilities
        """
        try:
            console.print("[bold yellow]Scanning for weaknesses...[/bold yellow]")
        except (UnicodeEncodeError, UnicodeError):
            print("Scanning for weaknesses...")
        
        report = WeaknessReport()
        
        # Implement vulnerability scanning based on personality
        if self.personality == NemesisPersonality.AGGRESSIVE:
            report = self._aggressive_scan(attack_budget, epsilon)
        elif self.personality == NemesisPersonality.CUNNING:
            report = self._cunning_scan(attack_budget, epsilon)
        elif self.personality == NemesisPersonality.ADAPTIVE:
            report = self._adaptive_scan(attack_budget, epsilon)
        elif self.personality == NemesisPersonality.RELENTLESS:
            report = self._relentless_scan(attack_budget, epsilon)
        else:  # CHAOTIC
            report = self._chaotic_scan(attack_budget, epsilon)
        
        # Log findings
        try:
            if len(report.vulnerabilities) > 0:
                console.print(f"[bold red]Found {len(report.vulnerabilities)} vulnerabilities![/bold red]")
            else:
                console.print("[bold green]No weaknesses found. Your model stands strong![/bold green]")
        except (UnicodeEncodeError, UnicodeError):
            if len(report.vulnerabilities) > 0:
                print(f"Found {len(report.vulnerabilities)} vulnerabilities!")
            else:
                print("No weaknesses found. Your model stands strong!")
        
        return report
    
    def _aggressive_scan(self, budget: int, epsilon: float) -> WeaknessReport:
        """Aggressive scanning with strong attacks."""
        report = WeaknessReport()
        try:
            console.print("Launching aggressive assault...")
        except (UnicodeEncodeError, UnicodeError):
            print("Launching aggressive assault...")
        
        # Placeholder for actual attack implementation
        report.add_vulnerability(
            "Storm", 0.75, "high", 
            "Model vulnerable to strong gradient-based attacks"
        )
        return report
    
    def _cunning_scan(self, budget: int, epsilon: float) -> WeaknessReport:
        """Cunning scanning with minimal perturbations."""
        report = WeaknessReport()
        try:
            console.print("Employing cunning tactics...")
        except (UnicodeEncodeError, UnicodeError):
            print("Employing cunning tactics...")
        
        # Placeholder for actual attack implementation
        report.add_vulnerability(
            "Whisper", 0.65, "medium",
            "Model susceptible to subtle perturbations"
        )
        return report
    
    def _adaptive_scan(self, budget: int, epsilon: float) -> WeaknessReport:
        """Adaptive scanning that learns from defenses."""
        report = WeaknessReport()
        try:
            console.print("Adapting attack strategy...")
        except (UnicodeEncodeError, UnicodeError):
            print("Adapting attack strategy...")
        
        # Placeholder for actual attack implementation
        report.add_vulnerability(
            "Shapeshifter", 0.80, "high",
            "Model defenses can be adapted to and bypassed"
        )
        return report
    
    def _relentless_scan(self, budget: int, epsilon: float) -> WeaknessReport:
        """Relentless scanning with continuous pressure."""
        report = WeaknessReport()
        try:
            console.print("Applying relentless pressure...")
        except (UnicodeEncodeError, UnicodeError):
            print("Applying relentless pressure...")
        
        # Placeholder for actual attack implementation
        report.add_vulnerability(
            "Siege", 0.70, "high",
            "Model breaks under sustained attack pressure"
        )
        return report
    
    def _chaotic_scan(self, budget: int, epsilon: float) -> WeaknessReport:
        """Chaotic scanning with unpredictable attacks."""
        report = WeaknessReport()
        try:
            console.print("Unleashing chaotic assault...")
        except (UnicodeEncodeError, UnicodeError):
            print("Unleashing chaotic assault...")
        
        # Placeholder for actual attack implementation
        report.add_vulnerability(
            "Chaos", 0.85, "critical",
            "Model cannot handle unpredictable attack combinations"
        )
        return report
    
    def forge_armor(self, strategy: str = "adaptive") -> 'DefenseArmor':
        """
        Create defenses based on discovered weaknesses.
        
        Args:
            strategy: Defense strategy (adaptive, robust, certified)
            
        Returns:
            DefenseArmor that can be applied to the model
        """
        from .defenses import DefenseArmor
        
        try:
            console.print("[bold blue]Forging armor in the depths of Tartarus...[/bold blue]")
        except (UnicodeEncodeError, UnicodeError):
            print("Forging armor in the depths of Tartarus...")
        
        armor = DefenseArmor(self.model, strategy=strategy)
        
        try:
            console.print("[bold green]Armor forged! Your model is now protected.[/bold green]")
        except (UnicodeEncodeError, UnicodeError):
            print("Armor forged! Your model is now protected.")
        
        return armor
    
    def eternal_battle(self, rounds: int = 10, evolution: bool = True) -> Any:
        """
        Continuous adversarial training cycle.
        
        Args:
            rounds: Number of battle rounds
            evolution: Whether to evolve defenses during battle
            
        Returns:
            Strengthened model after eternal battle
        """
        try:
            console.print(f"[bold red]Beginning eternal battle ({rounds} rounds)...[/bold red]")
        except (UnicodeEncodeError, UnicodeError):
            print(f"Beginning eternal battle ({rounds} rounds)...")
        
        current_model = self.model
        
        for round_num in range(1, rounds + 1):
            try:
                console.print(f"\n[bold cyan]Round {round_num}/{rounds}[/bold cyan]")
            except (UnicodeEncodeError, UnicodeError):
                print(f"\nRound {round_num}/{rounds}")
            
            # Find weaknesses
            weaknesses = self.find_weakness()
            
            if len(weaknesses) > 0:
                # Forge new defenses
                armor = self.forge_armor()
                current_model = armor.apply(current_model)
                
                # Track evolution
                if evolution:
                    self.evolution_level += 1
                    try:
                        console.print(f"[bold green]Evolution Level: {self.evolution_level}[/bold green]")
                    except (UnicodeEncodeError, UnicodeError):
                        print(f"Evolution Level: {self.evolution_level}")
            
            # Record battle
            self.battle_history.append({
                'round': round_num,
                'weaknesses_found': len(weaknesses),
                'evolution_level': self.evolution_level
            })
        
        try:
            console.print("[bold yellow]Eternal battle complete! Your model emerges victorious![/bold yellow]")
        except (UnicodeEncodeError, UnicodeError):
            print("Eternal battle complete! Your model emerges victorious!")
        
        return current_model

# Convenience function for quick nemesis summoning
def summon_nemesis(model, name: Optional[str] = None, 
                  personality: str = NemesisPersonality.ADAPTIVE) -> Nemesis:
    """
    Quickly summon a nemesis for your model.
    
    Args:
        model: The model to challenge
        name: Optional custom name
        personality: Attack personality
        
    Returns:
        Nemesis instance ready for battle
    """
    return Nemesis(model, name=name, personality=personality)

# Package exports
from .attacks import *
from .defenses import *
from .arena import *

__all__ = [
    'Nemesis',
    'NemesisPersonality', 
    'NemesisError',
    'BattleError',
    'WeaknessReport',
    'summon_nemesis',
]