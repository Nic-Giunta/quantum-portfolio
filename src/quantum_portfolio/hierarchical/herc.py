from dataclasses import dataclass

from .hrp import HRPAllocator


@dataclass
class HERCAllocator(HRPAllocator):
    """Simplified HERC using the HRP recursive risk split as a stable alpha implementation."""
    pass
