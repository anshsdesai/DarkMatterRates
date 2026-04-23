"""Source-family backend implementations for ``DarkMatterRates``."""

from .base import RateBackend
from .qcdark import QCDarkBackend
from .qedark import QEDarkBackend
from .qcdark2 import QCDark2Backend, simpson_uniform
from .wimprates import WimpratesBackend

__all__ = [
    "RateBackend",
    "QCDarkBackend",
    "QEDarkBackend",
    "QCDark2Backend",
    "WimpratesBackend",
    "simpson_uniform",
]
