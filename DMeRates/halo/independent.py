import torch


class HaloIndependentProvider:
    """Halo-independent eta(v_min) provider using the legacy step function."""

    def __init__(self, dm_halo, params: torch.Tensor):
        self.dm_halo = dm_halo
        self.params = params

    def eta(self, v_min_tensor: torch.Tensor) -> torch.Tensor:
        return self.dm_halo.step_function_eta(v_min_tensor, self.params)
