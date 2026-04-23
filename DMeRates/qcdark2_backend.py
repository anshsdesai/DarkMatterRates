"""Compatibility re-export for the QCDark2 backend.

The implementation now lives in ``DMeRates.backends.qcdark2``. This module is
kept so existing internal or notebook imports continue to work for one cycle.
"""

from .backends.qcdark2 import (  # noqa: F401
    ALPHA,
    CM2SEC,
    KG,
    LIGHT_SPEED,
    M_E,
    SEC2YR,
    QCDark2Backend,
    simpson_uniform,
)
