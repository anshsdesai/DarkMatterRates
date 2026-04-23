"""Numerically aligned QCDark2 source-family backend.

This module mirrors the non-relativistic rate conventions used in upstream
``qcdark2.dark_matter_rates`` while exposing a Torch-friendly implementation
for fast vectorized spectra in ``DarkMatterRates``.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

from ..Constants import qcdark2_band_gaps, qcdark2_pair_energies
from ..form_factor import form_factorQCDark2
from ..interpolation import interp1d
from ..units import (
    ALPHA,
    CM2SEC,
    KG_EV,
    LIGHT_SPEED_KM_PER_S,
    M_E_EV,
    SEC2YR,
)
from .base import RateBackend


LIGHT_SPEED = LIGHT_SPEED_KM_PER_S
M_E = M_E_EV
KG = KG_EV
_QCDARK2_VARIANT_SUFFIX = {"composite": "comp", "lfe": "lfe", "nolfe": "nolfe"}


def simpson_uniform(values: torch.Tensor, dx: float, dim: int = -1) -> torch.Tensor:
    """Integrate evenly spaced samples with Simpson's rule.

    Falls back to a trapezoid correction on the last interval when an even
    number of samples is supplied.
    """

    if values.shape[dim] < 2:
        return torch.movedim(values, dim, -1).squeeze(-1)

    data = torch.movedim(values, dim, -1)
    npts = data.shape[-1]
    if npts == 2:
        return dx * 0.5 * (data[..., 0] + data[..., 1])

    if npts % 2 == 0:
        simpson_part = simpson_uniform(data[..., :-1], dx, dim=-1)
        trap_part = dx * 0.5 * (data[..., -2] + data[..., -1])
        return simpson_part + trap_part

    interior_even = data[..., 2:-1:2].sum(dim=-1)
    interior_odd = data[..., 1::2].sum(dim=-1)
    return (dx / 3.0) * (
        data[..., 0] + data[..., -1] + 4.0 * interior_odd + 2.0 * interior_even
    )


class QCDark2Backend(RateBackend):
    """Fast Torch implementation of the upstream QCDark2 non-relativistic rate."""

    source_family = "qcdark2"
    supported_materials = tuple(qcdark2_band_gaps.keys())
    rate_units = "events/kg/year/eV"

    def __init__(self, form_factor, material, variant):
        if not self.supports_material(material):
            raise ValueError(
                f"QCDark2 does not support material '{material}'. "
                f"Supported: {list(self.supported_materials)}"
            )
        if variant not in _QCDARK2_VARIANT_SUFFIX:
            raise ValueError(
                f"Unknown qcdark2_variant '{variant}'. "
                f"Choose from: {list(_QCDARK2_VARIANT_SUFFIX.keys())}"
            )
        self.form_factor = form_factor
        self.material = material
        self.variant = variant

        self.q_raw = np.asarray(form_factor.q_raw, dtype=np.float64)
        self.q_eV = self.q_raw * ALPHA * M_E
        self.E_eV = np.asarray(form_factor.E_raw, dtype=np.float64)
        self.S_qE = np.asarray(form_factor.S(), dtype=np.float64)
        self.band_gap_eV = float(form_factor.band_gap_eV)

        self.q_step_eV = float(self.q_eV[1] - self.q_eV[0]) if self.q_eV.size > 1 else 0.0
        self.E_step_eV = float(self.E_eV[1] - self.E_eV[0]) if self.E_eV.size > 1 else 0.0
        self.rho_T = float(form_factor.M_cell) / KG / float(form_factor.V_cell)

        self._tensor_cache = {}

    @classmethod
    def build_for_rate(cls, rate, material, variant):
        """Load QCDark2 data, attach facade state, and return the backend."""
        import numericalunits as nu
        import torch

        if material not in qcdark2_band_gaps:
            raise ValueError(
                f"QCDark2 does not support material '{material}'. "
                f"Supported: {list(qcdark2_band_gaps.keys())}"
            )
        if variant not in _QCDARK2_VARIANT_SUFFIX:
            raise ValueError(
                f"Unknown qcdark2_variant '{variant}'. "
                f"Choose from: {list(_QCDARK2_VARIANT_SUFFIX.keys())}"
            )

        suffix = _QCDARK2_VARIANT_SUFFIX[variant]
        ff_name = "diamond" if material == "Diamond" else material
        form_factor_file = f"../form_factors/QCDark2/{variant}/{ff_name}_{suffix}.h5"
        form_factor_file_filepath = os.path.join(rate.module_dir, form_factor_file)
        ffactor = form_factorQCDark2(
            form_factor_file_filepath,
            band_gap=qcdark2_band_gaps[material],
        )

        rate.form_factor = ffactor
        rate.bin_size = qcdark2_pair_energies[material]
        rate.qArr = torch.tensor(
            ffactor.q_raw * nu.alphaFS * nu.me * nu.c0,
            dtype=torch.get_default_dtype(),
        )
        rate.Earr = torch.tensor(ffactor.E_raw * nu.eV, dtype=torch.get_default_dtype())
        rate.nQ = len(rate.qArr)
        rate.nE = len(rate.Earr)
        rate.Ei_array = torch.floor(torch.round((rate.Earr / nu.eV) * 10)).int()
        rate.ionization_func = rate.RKProbabilities if material == "Si" else rate.step_probabilities
        rate.QEDark = False

        backend = cls(ffactor, material=material, variant=variant)
        rate.qcdark2_backend = backend
        return backend

    def attach(self, rate):
        rate.qcdark2_backend = self
        return self

    def energy_grid(self):
        return self.E_eV

    def get_tensors(self, device, dtype):
        """Return cached Torch tensors for a given device/dtype pair."""
        torch_device = torch.device(device)
        key = (str(torch_device), str(dtype))
        if key not in self._tensor_cache:
            self._tensor_cache[key] = {
                "q_raw": torch.as_tensor(self.q_raw, device=torch_device, dtype=dtype),
                "q_eV": torch.as_tensor(self.q_eV, device=torch_device, dtype=dtype),
                "E_eV": torch.as_tensor(self.E_eV, device=torch_device, dtype=dtype),
                "S_qE": torch.as_tensor(self.S_qE, device=torch_device, dtype=dtype),
                "band_gap_mask": torch.as_tensor(
                    self.E_eV >= self.band_gap_eV,
                    device=torch_device,
                    dtype=torch.bool,
                ),
            }
        return self._tensor_cache[key]

    def vmin_grid(self, mX_eV, device, dtype):
        """Return the QCDark2 velocity-threshold grid in units of v/c."""
        tensors = self.get_tensors(device, dtype)
        q_eV = tensors["q_eV"].unsqueeze(1)
        E_eV = tensors["E_eV"].unsqueeze(0)
        return q_eV / (2.0 * mX_eV) + E_eV / q_eV

    def vmin_grid_kms(self, mX_eV, device, dtype):
        """Return the velocity-threshold grid in km/s."""
        return self.vmin_grid(mX_eV, device, dtype) * LIGHT_SPEED

    def eta_mb(self, mX_eV, astro_model, device, dtype):
        """Analytic Maxwell-Boltzmann eta(vmin) matching upstream QCDark2."""
        tensors = self.get_tensors(device, dtype)
        q_eV = tensors["q_eV"].unsqueeze(1)
        E_eV = tensors["E_eV"].unsqueeze(0)
        vmin = q_eV / (2.0 * mX_eV) + E_eV / q_eV

        v_escape = astro_model["vEscape"] / LIGHT_SPEED
        v_earth = astro_model["vEarth"] / LIGHT_SPEED
        v0 = astro_model["v0"] / LIGHT_SPEED

        v_escape_t = torch.as_tensor(v_escape, device=device, dtype=dtype)
        v_earth_t = torch.as_tensor(v_earth, device=device, dtype=dtype)
        v0_t = torch.as_tensor(v0, device=device, dtype=dtype)
        sqrt_pi = torch.as_tensor(math.sqrt(math.pi), device=device, dtype=dtype)
        pi_t = torch.as_tensor(math.pi, device=device, dtype=dtype)
        exp_term = torch.exp(-(v_escape_t / v0_t) ** 2)
        escape_erf = torch.erf(v_escape_t / v0_t)

        eta = torch.zeros_like(vmin)

        val_below = -4.0 * v_earth_t * exp_term
        val_below = val_below + sqrt_pi * v0_t * (
            torch.erf((vmin + v_earth_t) / v0_t) - torch.erf((vmin - v_earth_t) / v0_t)
        )
        val_above = -2.0 * (v_earth_t + v_escape_t - vmin) * exp_term
        val_above = val_above + sqrt_pi * v0_t * (
            escape_erf - torch.erf((vmin - v_earth_t) / v0_t)
        )

        eta = torch.where(vmin < v_escape_t + v_earth_t, val_above, eta)
        eta = torch.where(vmin < v_escape_t - v_earth_t, val_below, eta)

        norm = (v0_t**3) * (
            -2.0 * pi_t * (v_escape_t / v0_t) * exp_term
            + (pi_t**1.5) * escape_erf
        )
        eta = (v0_t**2) * pi_t / (2.0 * v_earth_t * norm) * eta
        return torch.where(eta > 0, eta, torch.zeros_like(eta))

    def eta_from_file(self, mX_eV, file_vmins_kms, file_etas_skms, device, dtype):
        """Interpolate eta(vmin) from halo tables stored in km/s and s/km."""
        vmin_kms = self.vmin_grid_kms(mX_eV, device, dtype)
        x = torch.as_tensor(file_vmins_kms, device=device, dtype=dtype)
        y = torch.as_tensor(file_etas_skms * LIGHT_SPEED, device=device, dtype=dtype)
        eta = interp1d(x, y, vmin_kms)
        invalid = (vmin_kms < x[0]) | (vmin_kms > x[-1]) | torch.isnan(eta)
        return torch.where(invalid, torch.zeros_like(eta), eta)

    def differential_rate(self, mX_eV, FDMn, astro_model, eta_qE, device, dtype):
        """Return dR/dE in physical units of events/kg/year/eV."""
        tensors = self.get_tensors(device, dtype)
        q_raw = tensors["q_raw"]
        S_qE = tensors["S_qE"]

        if FDMn == 0:
            fdm2 = torch.ones_like(q_raw)
        else:
            fdm2 = torch.pow(q_raw.reciprocal(), 2 * FDMn)

        reduced_mass = mX_eV * M_E / (mX_eV + M_E)
        prefactor = (1.0 / self.rho_T) * (astro_model["rhoX"] / mX_eV)
        prefactor *= astro_model["sigma_e"] / (reduced_mass**2)
        prefactor /= 4.0 * math.pi

        integrand = q_raw.unsqueeze(1) * fdm2.unsqueeze(1) * S_qE * eta_qE
        dR_dE = simpson_uniform(integrand, self.q_step_eV, dim=0)
        dR_dE *= prefactor / CM2SEC / SEC2YR

        return torch.where(tensors["band_gap_mask"], dR_dE, torch.zeros_like(dR_dE))

    def total_rate(self, spectrum):
        """Integrate dR/dE over energy and return events/kg/year."""
        return simpson_uniform(spectrum, self.E_step_eV, dim=-1)
