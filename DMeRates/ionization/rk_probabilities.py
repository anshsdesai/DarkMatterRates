from pathlib import Path

import numericalunits as nu
import torch
from torchinterp1d.interp1d import Interp1d


def rk_probabilities(
    ne: int,
    energy_array: torch.Tensor,
    p100k_path: str | Path,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Interpolate Ramanathan-Kurinsky 100K probabilities onto a target energy grid."""
    from numpy import loadtxt

    if dtype is None:
        dtype = torch.get_default_dtype()

    p100data = loadtxt(str(p100k_path))
    pEV = torch.tensor(p100data[:, 0], dtype=dtype) * nu.eV
    file_probabilities = torch.tensor(p100data.T, dtype=dtype)[ne]
    return Interp1d.apply(pEV, file_probabilities, energy_array).flatten()
