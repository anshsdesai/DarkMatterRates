import numericalunits as nu
import torch


def step_probabilities(
    ne: int,
    energy_array: torch.Tensor,
    dE: float,
    band_gap: float,
    bin_size: float,
) -> torch.Tensor:
    """Legacy step-function approximation for e-h pair probabilities."""
    i = ne - 1
    dE_eV = dE / nu.eV
    gap_eV = band_gap / nu.eV
    E2Q = bin_size / nu.eV

    initE = int(gap_eV / dE_eV)
    binE = int(round(E2Q / dE_eV))
    bounds = (i * binE + initE + 1, (i + 1) * binE + initE + 1)
    probabilities = torch.zeros_like(energy_array)
    probabilities[bounds[0] : bounds[1]] = 1
    return probabilities
