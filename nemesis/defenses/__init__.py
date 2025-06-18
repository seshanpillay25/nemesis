"""
Nemesis Defense Arsenal 🛡️

"From the forge of adversity comes the strongest armor"

This module contains the complete arsenal of adversarial defenses,
each named after protective forces and divine guardians.
"""

from .base import DefenseBase, DefenseResult, DefenseArmor
from .shields import (
    Aegis,    # Input purification
    Barrier,  # Adversarial detection
)
from .armor import (
    Fortitude,   # Adversarial training
    Resilience,  # Defensive distillation
    Immunity,    # Certified defenses
)

__all__ = [
    'DefenseBase',
    'DefenseResult', 
    'DefenseArmor',
    'Aegis',
    'Barrier',
    'Fortitude',
    'Resilience',
    'Immunity',
]