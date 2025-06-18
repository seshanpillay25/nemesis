"""
Poisoning Attacks ☠️

"Corruption from within - the most insidious form of attack"

Poisoning attacks contaminate training data or models to create
persistent vulnerabilities. Like poison in the well, these attacks
corrupt the very foundation upon which models are built.
"""

from .trojan import Trojan
from .corruption import Corruption

__all__ = [
    'Trojan',
    'Corruption',
]