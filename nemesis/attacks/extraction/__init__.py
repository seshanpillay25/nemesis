"""
Extraction Attacks 🔮

"Stealing the secrets hidden within the mind of the machine"

Extraction attacks aim to steal model parameters, training data,
or other sensitive information through clever query strategies.
Like ancient oracles revealing hidden knowledge.
"""

from .mindthief import MindThief
from .oracle import Oracle

__all__ = [
    'MindThief',
    'Oracle',
]