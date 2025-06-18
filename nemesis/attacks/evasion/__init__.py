"""
Evasion Attacks 🌪️

"Like shadows in the night, these attacks slip past defenses"

Evasion attacks manipulate inputs at test time to fool models
into making incorrect predictions. Each attack embodies a different
mythological force of deception and misdirection.
"""

from .whisper import Whisper
from .storm import Storm  
from .shapeshifter import Shapeshifter
from .mirage import Mirage
from .chaos import Chaos

__all__ = [
    'Whisper',
    'Storm',
    'Shapeshifter', 
    'Mirage',
    'Chaos',
]