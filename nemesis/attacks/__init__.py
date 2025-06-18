"""
Nemesis Attack Arsenal 🗡️

"Strike with the fury of the gods"

This module contains the complete arsenal of adversarial attacks,
each named after mythological forces and phenomena.
"""

from .base import AttackBase, AttackResult
from .evasion import (
    Whisper,  # FGSM - Subtle perturbations
    Storm,    # PGD - Powerful iterative attack
    Shapeshifter,  # C&W - Optimized transformation
    Mirage,   # DeepFool - Minimal perturbation
    Chaos,    # AutoAttack - Ensemble mayhem
)
from .poisoning import (
    Trojan,      # Backdoor attacks
    Corruption,  # Label flipping
)
from .extraction import (
    MindThief,   # Model stealing
    Oracle,      # Query-based extraction
)

__all__ = [
    'AttackBase',
    'AttackResult',
    'Whisper',
    'Storm', 
    'Shapeshifter',
    'Mirage',
    'Chaos',
    'Trojan',
    'Corruption',
    'MindThief',
    'Oracle',
]