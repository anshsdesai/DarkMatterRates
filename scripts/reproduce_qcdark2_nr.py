"""Reproduce upstream QCDark2 non-relativistic halo outputs inside DarkMatterRates.

Examples
--------
python scripts/reproduce_qcdark2_nr.py
python scripts/reproduce_qcdark2_nr.py --outdir /tmp/qcdark2_repro
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

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


CASES = [
    ("Si", "composite"),
    ("Si", "lfe"),
    ("Si", "nolfe"),
    ("Ge", "composite"),
    ("Ge", "lfe"),
    ("Ge", "nolfe"),
    ("GaAs", "composite"),
    ("GaAs", "lfe"),
    ("GaAs", "nolfe"),
    ("SiC", "composite"),
    ("SiC", "lfe"),
    ("SiC", "nolfe"),
    ("Diamond", "composite"),
    ("Diamond", "lfe"),
    ("Diamond", "nolfe"),
]
MEDIATORS = [
    (0, "heavy"),
    (2, "light"),
]


def upstream_path(material: str, variant: str) -> Path:
    suffix = {"composite": "comp", "lfe": "lfe", "nolfe": "nolfe"}[variant]
    name = "diamond" if material == "Diamond" else material
    return UPSTREAM_ROOT / "dielectric_functions" / variant / f"{name}_{suffix}.h5"


def configure_object(material: str, variant: str) -> DMeRate:
    obj = DMeRate(material, form_factor_type="qcdark2", qcdark2_variant=variant, device="cpu")
    obj.update_params(
        dm.default_astro["v0"],
        dm.default_astro["vEarth"],
        dm.default_astro["vEscape"],
        dm.default_astro["rhoX"],
        dm.default_astro["sigma_e"],
    )
    return obj


def sensitivity_from_total_rate(total_rate, sigma_ref, cl=0.9):
    n_exp = -np.log(1.0 - cl)
    return n_exp * sigma_ref / total_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=None, help="Optional directory for .npz outputs.")
    args = parser.parse_args()

    masses_mev = np.logspace(np.log10(0.5), np.log10(1000.0), 20)
    summary = {}

    if args.outdir is not None:
        args.outdir.mkdir(parents=True, exist_ok=True)

    for material, variant in CASES:
        epsilon = dm.load_epsilon(str(upstream_path(material, variant)))
        obj = configure_object(material, variant)
        case_key = f"{material}_{variant}"
        case_summary = {}

        spectra_payload = {}
        sensitivity_payload = {"masses_mev": masses_mev}

        for fdm_n, mediator in MEDIATORS:
            energy_ref, spectra_ref = None, None
            dR_ref, energy_ref = dm.get_dR_dE(
                epsilon,
                m_X=1e9,
                mediator=mediator,
                astro_model=dm.default_astro,
                screening="RPA",
            )
            energy_native, spectra_native = obj.calculate_spectrum([1000.0], "shm", fdm_n, DoScreen=False)
            total_native = obj.calculate_total_rate(masses_mev, "shm", fdm_n, DoScreen=False)
            _, sigma_ref = dm.ex(
                epsilon,
                mediator=mediator,
                astro_model=dm.default_astro,
                screening="RPA",
                cl=0.9,
                m_X_min=0.5e6,
                m_X_max=1e9,
                N_m_X=len(masses_mev),
            )
            sigma_native = sensitivity_from_total_rate(total_native, dm.default_astro["sigma_e"], cl=0.9)

            spectra_payload[f"{mediator}_energy_eV"] = energy_ref
            spectra_payload[f"{mediator}_reference"] = dR_ref
            spectra_payload[f"{mediator}_native"] = spectra_native[0]
            sensitivity_payload[f"{mediator}_reference"] = sigma_ref
            sensitivity_payload[f"{mediator}_native"] = sigma_native

            populated = dR_ref >= (np.max(dR_ref) * 1e-8)
            spectrum_rel = np.max(
                np.abs(spectra_native[0][populated] - dR_ref[populated]) / np.maximum(dR_ref[populated], 1e-30)
            )
            sensitivity_rel = np.max(np.abs(sigma_native - sigma_ref) / np.maximum(sigma_ref, 1e-30))
            case_summary[mediator] = {
                "spectrum_max_rel_diff": float(spectrum_rel),
                "sensitivity_max_rel_diff": float(sensitivity_rel),
            }

        if args.outdir is not None:
            np.savez(args.outdir / f"{case_key}_spectra.npz", **spectra_payload)
            np.savez(args.outdir / f"{case_key}_sensitivity.npz", **sensitivity_payload)
        summary[case_key] = case_summary

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.outdir is not None:
        (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
