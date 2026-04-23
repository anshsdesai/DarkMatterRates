"""Shared setup for legacy semiconductor source families."""

from __future__ import annotations

import os

import numericalunits as nu

from ..Constants import Gegapsize, Sigapsize
from ..form_factor import form_factor, form_factorQEDark
from .base import RateBackend


class LegacySemiconductorBackend(RateBackend):
    """Common QCDark/QEDark semiconductor backend setup."""

    supported_materials = ("Si", "Ge")
    rate_units = "implicit 1/kg/year/eV"
    is_qedark = False
    source_family = "legacy_semiconductor"

    def __init__(self, material, module_dir):
        if not self.supports_material(material):
            raise ValueError(
                f"form_factor_type='{self.source_family}' only supports 'Si' and 'Ge'."
            )
        self.material = material
        self.module_dir = module_dir

    def form_factor_path(self):
        raise NotImplementedError

    def load_form_factor(self):
        path = os.path.join(self.module_dir, self.form_factor_path())
        if self.is_qedark:
            return form_factorQEDark(path)
        return form_factor(path)

    def attach(self, rate):
        import numpy as np
        import torch

        rate.bin_size = {"Si": Sigapsize, "Ge": Gegapsize}[self.material]
        rate.form_factor = self.load_form_factor()
        rate.ionization_func = rate.RKProbabilities

        nE = np.shape(rate.form_factor.ff)[1]
        nQ = np.shape(rate.form_factor.ff)[0]
        rate.nE = nE
        rate.nQ = nQ

        if self.is_qedark:
            rate.qiArr = torch.arange(1, nQ + 1)
            rate.qArr = torch.clone(rate.qiArr) * torch.tensor(rate.form_factor.dq)
            rate.Earr = torch.arange(nE) * torch.tensor(rate.form_factor.dE)
        else:
            dq = torch.tensor(rate.form_factor.dq, dtype=torch.get_default_dtype())
            dE = torch.tensor(rate.form_factor.dE, dtype=torch.get_default_dtype())
            rate.qiArr = torch.arange(nQ)
            rate.qArr = torch.clone(rate.qiArr) * dq + dq / 2.0
            rate.Earr = torch.arange(nE) * dE + dE / 2.0

        rate.Ei_array = torch.floor(torch.round((rate.Earr / nu.eV) * 10)).int()
        rate.QEDark = self.is_qedark
        return self

    def energy_grid(self):
        return None

    def differential_rate(self, rate, *args, **kwargs):
        return rate.vectorized_dRdE(*args, **kwargs)
