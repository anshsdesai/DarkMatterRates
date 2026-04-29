import os
from pathlib import Path

import numericalunits as nu
import torch
from torchinterp1d.interp1d import Interp1d


class FileHaloProvider:
    """File-backed eta(v_min) provider with legacy interpolation behavior."""

    def __init__(self, file_vmins: torch.Tensor, file_etas: torch.Tensor):
        self.file_vmins = file_vmins
        self.file_etas = file_etas

    def eta(self, v_min_tensor: torch.Tensor) -> torch.Tensor:
        etas = Interp1d.apply(self.file_vmins, self.file_etas, v_min_tensor)
        etas = torch.where(
            (v_min_tensor < self.file_vmins[0])
            | (v_min_tensor > self.file_vmins[-1])
            | (torch.isnan(etas)),
            0,
            etas,
        )
        return etas


def load_halo_file_data(
    module_dir: str | Path,
    dm_halo,
    halo_model: str,
    v0: float,
    vEarth: float,
    vEscape: float,
    rhoX: float,
    cross_section: float,
    mX=None,
    FDMn=None,
    isoangle=None,
    useVerne: bool = False,
    calcErrors=None,
    default_dtype=None,
):
    """Load tabulated eta(v_min) data for analytic or modulated halo models."""
    from numpy import loadtxt
    from numpy import round as npround

    if default_dtype is None:
        default_dtype = torch.get_default_dtype()

    module_dir = str(module_dir)
    if halo_model in {"imb", "step"}:
        return None, None

    if isoangle is None:
        lightSpeed_kmpers = nu.s / nu.km
        geVconversion = 1 / (nu.GeV / nu.c0**2 / nu.cm**3)
        halo_dir_prefix = os.path.join(module_dir, "../halo_data/")
        file = (
            halo_dir_prefix
            + f"{halo_model}_v0{round(v0 * lightSpeed_kmpers, 1)}"
            + f"_vE{round(vEarth * lightSpeed_kmpers, 1)}"
            + f"_vEsc{round(vEscape * lightSpeed_kmpers, 1)}"
            + f"_rhoX{round(rhoX * geVconversion, 1)}.txt"
        )
        try:
            temp = open(file, "r")
            temp.close()
        except FileNotFoundError:
            dm_halo.generate_halo_files(halo_model)

        try:
            data = loadtxt(file, delimiter="\t")
        except ValueError:
            dm_halo.generate_halo_files(halo_model)
            data = loadtxt(file, delimiter="\t")

        if len(data) == 0:
            raise ValueError("file is empty!")

        file_etas = torch.tensor(data[:, 1], dtype=default_dtype) * nu.s / nu.km
        file_vmins = torch.tensor(data[:, 0], dtype=default_dtype) * nu.km / nu.s

        if file_etas[-1] == file_etas[-2]:
            file_etas = file_etas[:-1]
            file_vmins = file_vmins[:-1]
        return file_vmins, file_etas

    mass_string = str(npround(float(mX), 3)).replace(".", "_")
    sigmaE = float(format(cross_section / nu.cm**2, ".3g"))
    sigmaE_str = str(sigmaE)
    fdm_str = "FDM1" if FDMn == 0 else "FDMq2"
    summer_str = "_summer" if halo_model == "summer" else ""

    halo_dir_prefix = os.path.join(module_dir, f"../halo_data/modulated/{fdm_str}/")
    if useVerne:
        data_dir = halo_dir_prefix + f"Verne{summer_str}/"
    else:
        data_dir = halo_dir_prefix + "DaMaSCUS/"

    file = f"{data_dir}mDM_{mass_string}_MeV_sigmaE_{sigmaE_str}_cm2/DM_Eta_theta_{isoangle}.txt"
    if not os.path.isfile(file):
        print(file)
        raise FileNotFoundError("sigmaE file not found")

    try:
        data = loadtxt(file, delimiter="\t")
    except ValueError:
        print(file)
        raise ValueError(f"file not found! tried {file}")

    if len(data) == 0:
        raise ValueError("file is empty!")

    file_etas = torch.tensor(data[:, 1], dtype=default_dtype) * nu.s / nu.km
    file_vmins = torch.tensor(data[:, 0], dtype=default_dtype) * nu.km / nu.s
    if calcErrors is not None:
        file_eta_err = torch.tensor(data[:, 2]) * nu.s / nu.km
        if calcErrors == "High":
            file_etas += file_eta_err
        if calcErrors == "Low":
            file_etas -= file_eta_err

    if file_etas[-1] == file_etas[-2]:
        file_etas = file_etas[:-1]
        file_vmins = file_vmins[:-1]
    return file_vmins, file_etas
