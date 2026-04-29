import torch


class AnalyticHaloProvider:
    """Analytic halo provider interface: eta(v_min_tensor) -> torch.Tensor."""

    def __init__(self, dm_halo, model: str):
        self.dm_halo = dm_halo
        self.model = model

    def eta(self, v_min_tensor: torch.Tensor) -> torch.Tensor:
        if self.model == "imb":
            return self.dm_halo.eta_MB_tensor(v_min_tensor)

        if self.model in {"shm", "tsa", "dpl"}:
            fn = {
                "shm": self.dm_halo.etaSHM,
                "tsa": self.dm_halo.etaTsa,
                "dpl": self.dm_halo.etaDPL,
            }[self.model]
            flat = v_min_tensor.reshape(-1).detach().cpu().numpy()
            vals = [fn(v) for v in flat]
            return torch.tensor(vals, dtype=v_min_tensor.dtype, device=v_min_tensor.device).reshape(
                v_min_tensor.shape
            )

        raise ValueError(f"Unsupported analytic halo model: {self.model}")
