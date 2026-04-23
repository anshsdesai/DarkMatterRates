"""Shared helpers for the physics validation suite."""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

from tests.conftest import (
    REPO_ROOT,
    UPSTREAM_QCDARK_ROOT,
    UPSTREAM_WIMPRATES_ROOT,
    hdf5_path,
)


def tensor_to_numpy(value):
    """Convert Torch tensors or array-likes to a CPU NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def rates_to_events_per_kg_year(rates):
    """Convert repository implicit semiconductor rates to events/kg/year."""
    import numericalunits as nu

    return tensor_to_numpy(rates * nu.kg * nu.year)


def finite_positive_mask(reference, relative_floor=1e-8):
    """Mask populated bins while ignoring tiny numerical tails."""
    reference = np.asarray(reference)
    if reference.size == 0 or np.max(reference) <= 0:
        return np.zeros_like(reference, dtype=bool)
    return np.isfinite(reference) & (reference >= np.max(reference) * relative_floor)


def relative_error(candidate, reference):
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    return np.abs(candidate - reference) / np.maximum(np.abs(reference), 1e-300)


def load_upstream_qcdark():
    """Load QCDark's standalone module under a non-conflicting name."""
    module_path = os.path.join(UPSTREAM_QCDARK_ROOT, "dark_matter_rates.py")
    spec = importlib.util.spec_from_file_location("upstream_qcdark_rates", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_upstream_wimprates():
    """Import the sibling wimprates checkout before any installed package."""
    if UPSTREAM_WIMPRATES_ROOT not in sys.path:
        sys.path.insert(0, UPSTREAM_WIMPRATES_ROOT)
    import wimprates as wr

    return wr


class QEDarkFFAdapter:
    """QEDark text form-factor adapter for QCDark's rate function."""

    dq = 0.02 * (1.0 / 137.0) * 511e3
    dE = 0.1

    def __init__(self, material):
        import h5py

        if material not in ("Si", "Ge"):
            raise ValueError("QEDark adapter supports only Si and Ge.")

        self.material = material
        filename = hdf5_path(f"form_factors/QEDark/{material}_f2.txt")
        raw = np.loadtxt(filename, skiprows=1)
        fcrys = np.transpose(np.resize(raw, (500, 900)))
        self.ff = fcrys * (2.0 / 137.0) / 4.0
        self.band_gap = {"Si": 1.2, "Ge": 0.7}[material]

        qcdark_file = hdf5_path(f"form_factors/QCDark/{material}_final.hdf5")
        with h5py.File(qcdark_file, "r") as h5:
            self.mCell = float(h5["results"].attrs["mCell"])


def qcdark_reference_form_factor(material):
    qcdark = load_upstream_qcdark()
    return qcdark.form_factor(hdf5_path(f"form_factors/QCDark/{material}_final.hdf5"))


def default_upstream_astro(qcdark_module):
    """Return a copy so tests can mutate rhoX/sigma_e safely."""
    return dict(qcdark_module.default_astro)


def set_dmerates_to_qcdark_astro(obj, qcdark_module):
    astro = qcdark_module.default_astro
    obj.update_params(
        astro["v0"],
        astro["vEarth"],
        astro["vEscape"],
        astro["rhoX"],
        astro["sigma_e"],
    )
    return obj


def qcdark_material_path(material):
    return hdf5_path(f"form_factors/QCDark/{material}_final.hdf5")


def old_wimprates_csv_path(name):
    return os.path.join(REPO_ROOT, "old_for_comparison", "wimprates_mod", name)
