"""Benchmark upstream QCDark2, native DarkMatterRates qcdark2, and legacy qcdark."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT.parent / "QCDark2"
if not UPSTREAM_ROOT.exists():
    raise FileNotFoundError(f"Expected sibling QCDark2 checkout at {UPSTREAM_ROOT}")

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(UPSTREAM_ROOT))

from DMeRates.DMeRate import DMeRate  # noqa: E402
from qcdark2 import dark_matter_rates as dm  # noqa: E402


MASS_GRID_MEV = np.array([0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0])
MEDIATORS = [
    (0, "heavy"),
    (2, "light"),
]


def time_average(fn, repeat=3):
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    return float(np.mean(durations))


def configure_qcdark2(device):
    obj = DMeRate("Si", form_factor_type="qcdark2", qcdark2_variant="composite", device=device)
    obj.update_params(
        dm.default_astro["v0"],
        dm.default_astro["vEarth"],
        dm.default_astro["vEscape"],
        dm.default_astro["rhoX"],
        dm.default_astro["sigma_e"],
    )
    return obj


def configure_qcdark(material):
    obj = DMeRate(material, form_factor_type="qcdark", device="cpu")
    obj.update_params(
        dm.default_astro["v0"],
        dm.default_astro["vEarth"],
        dm.default_astro["vEscape"],
        dm.default_astro["rhoX"],
        dm.default_astro["sigma_e"],
    )
    return obj


def main():
    epsilon = dm.load_epsilon(str(UPSTREAM_ROOT / "dielectric_functions" / "composite" / "Si_comp.h5"))
    native_cpu = configure_qcdark2("cpu")
    legacy_si = configure_qcdark("Si")
    legacy_ge = configure_qcdark("Ge")

    results = {"cpu": {}, "gpu": None}

    for fdm_n, mediator in MEDIATORS:
        key = f"fdm_{fdm_n}"
        results["cpu"][key] = {
            "upstream_qcdark2_single_spectrum_sec": time_average(
                lambda: dm.get_dR_dE(
                    epsilon,
                    m_X=100e6,
                    mediator=mediator,
                    astro_model=dm.default_astro,
                    screening="RPA",
                )
            ),
            "upstream_qcdark2_mass_sweep_sec": time_average(
                lambda: [
                    dm.get_dR_dE(
                        epsilon,
                        m_X=mass_mev * 1e6,
                        mediator=mediator,
                        astro_model=dm.default_astro,
                        screening="RPA",
                    )
                    for mass_mev in MASS_GRID_MEV
                ]
            ),
            "native_qcdark2_single_spectrum_sec": time_average(
                lambda: native_cpu.calculate_spectrum([100.0], "shm", fdm_n, DoScreen=False)
            ),
            "native_qcdark2_mass_sweep_sec": time_average(
                lambda: native_cpu.calculate_spectrum(MASS_GRID_MEV, "shm", fdm_n, DoScreen=False)
            ),
            "legacy_qcdark_si_single_spectrum_sec": time_average(
                lambda: legacy_si.calculate_spectrum([100.0], "shm", fdm_n, DoScreen=False)
            ),
            "legacy_qcdark_si_mass_sweep_sec": time_average(
                lambda: legacy_si.calculate_spectrum(MASS_GRID_MEV, "shm", fdm_n, DoScreen=False)
            ),
            "legacy_qcdark_ge_single_spectrum_sec": time_average(
                lambda: legacy_ge.calculate_spectrum([100.0], "shm", fdm_n, DoScreen=False)
            ),
            "legacy_qcdark_ge_mass_sweep_sec": time_average(
                lambda: legacy_ge.calculate_spectrum(MASS_GRID_MEV, "shm", fdm_n, DoScreen=False)
            ),
        }

    if torch.cuda.is_available():
        native_gpu = configure_qcdark2("cuda")
        results["gpu"] = {}
        for fdm_n, _ in MEDIATORS:
            key = f"fdm_{fdm_n}"
            results["gpu"][key] = {
                "native_qcdark2_single_spectrum_sec": time_average(
                    lambda: native_gpu.calculate_spectrum([100.0], "shm", fdm_n, DoScreen=False)
                ),
                "native_qcdark2_mass_sweep_sec": time_average(
                    lambda: native_gpu.calculate_spectrum(MASS_GRID_MEV, "shm", fdm_n, DoScreen=False)
                ),
            }

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
