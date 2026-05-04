from dataclasses import dataclass

from .hrp import HRPAllocator


@dataclass
class NCOAllocator(HRPAllocator):
    """Simplified nested clustered optimization proxy using HRP weights."""
    pass
