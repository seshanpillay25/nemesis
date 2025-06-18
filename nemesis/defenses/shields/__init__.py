"""
Shield Defenses 🛡️

"The first line of defense, purifying and detecting threats before they strike"

Shield defenses operate at the input level, providing preprocessing
and detection capabilities to identify and neutralize adversarial threats.
"""

from .aegis import Aegis
from .barrier import Barrier

__all__ = [
    'Aegis',
    'Barrier',
]