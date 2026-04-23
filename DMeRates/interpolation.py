"""Internal interpolation helpers.

This wrapper makes the vendored ``torchinterp1d`` package work both when the
project is installed and when it is executed directly from the source tree.
"""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _resolve_interp1d():
    """Resolve the torchinterp1d callable from either an installed or vendored copy."""
    module = None
    try:
        module = importlib.import_module("torchinterp1d")
    except ImportError:
        module = None

    if module is not None and hasattr(module, "interp1d"):
        return module.interp1d

    vendor_root = Path(__file__).resolve().parents[1] / "torchinterp1d"
    vendor_root_str = str(vendor_root)
    if vendor_root_str not in sys.path:
        sys.path.insert(0, vendor_root_str)

    stale_module = sys.modules.get("torchinterp1d")
    if stale_module is not None and not hasattr(stale_module, "interp1d"):
        del sys.modules["torchinterp1d"]

    module = importlib.import_module("torchinterp1d")
    if not hasattr(module, "interp1d"):
        raise ImportError(
            "Could not resolve torchinterp1d.interp1d from either the installed "
            "package or the vendored source tree."
        )
    return module.interp1d


def interp1d(x, y, xnew, out=None):
    """Dispatch to the vendored/interpreted torchinterp1d implementation."""
    fn = _resolve_interp1d()
    if out is None:
        return fn(x, y, xnew)
    return fn(x, y, xnew, out)
