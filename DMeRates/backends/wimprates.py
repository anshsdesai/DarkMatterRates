"""Wimprates/noble-gas source-family backend."""

from __future__ import annotations

import os

import numericalunits as nu

from ..Constants import ry
from ..form_factor import formFactorNoble
from .base import RateBackend


class WimpratesBackend(RateBackend):
    source_family = "wimprates"
    supported_materials = ("Xe", "Ar")
    rate_units = "implicit noble-shell spectrum"

    def __init__(self, material, module_dir):
        if not self.supports_material(material):
            raise ValueError("Wimprates backend only supports 'Xe' and 'Ar'.")
        self.material = material
        self.module_dir = module_dir

    def attach(self, rate):
        import numpy as np
        import torch

        Earr = np.geomspace(1, 400, 100) * nu.eV
        logkArr = np.log(Earr / ry) / 2
        rate.Earr = torch.tensor(Earr)

        form_factor_file = f"../form_factors/wimprates/{self.material}_dme_ionization_ff.pkl"
        form_factor_file_filepath = os.path.join(rate.module_dir, form_factor_file)
        formfactor = formFactorNoble(form_factor_file_filepath)
        rate.form_factor = formfactor

        qArrdict = {}
        ffdata_dict = {}
        numqs = 1001

        for shell_key in formfactor.keys:
            qmax = (
                np.exp(formfactor.shell_data[shell_key]["lnqs"].max())
                * nu.me
                * nu.c0
                * nu.alphaFS
            )
            qmin = np.exp(formfactor.shell_data[shell_key]["lnqs"].min()) * nu.eV / nu.c0

            qArr = torch.linspace(qmin, qmax, numqs)
            lnqi = np.log(qArr.cpu() / (nu.me * nu.c0 * nu.alphaFS))
            ffdata = np.zeros((len(Earr), numqs))
            for i, k in enumerate(logkArr):
                ffdata[i, :] = formfactor.shell_data[shell_key]["log10ffsquared_itp"]((k, lnqi))
            ffdata = 10**ffdata
            ffdata_dict[shell_key] = torch.tensor(ffdata)
            qArrdict[shell_key] = qArr.to(rate.device)

        rate.qArrdict = qArrdict
        rate.form_factor.ff = ffdata_dict
        rate.QEDark = False
        return self

    def energy_grid(self):
        return None

    def differential_rate(self, rate, *args, **kwargs):
        return rate.noble_dRdE(*args, **kwargs)

    def fold_to_ne(self, rate, *args, **kwargs):
        return rate.calculate_nobleGas_rates(*args, **kwargs)
