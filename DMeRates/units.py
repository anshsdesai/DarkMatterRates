"""Shared unit conversion helpers for backend kernels.

Backends should derive fixed numeric conventions from ``Constants.py`` at their
boundary, then run hot tensor kernels on plain numeric arrays.
"""

from __future__ import annotations

import numericalunits as nu

from .Constants import me_eV


LIGHT_SPEED_KM_PER_S = float(nu.c0 / (nu.km / nu.s))
ALPHA = float(nu.alphaFS)
M_E_EV = float(me_eV / nu.eV)
KG_EV = float(nu.kg * nu.c0**2 / nu.eV)
CM2SEC = float(1.0 / (nu.c0 / (nu.cm / nu.s)))
SEC2YR = float(nu.s / nu.year)


def qcdark2_astro_model_from_unitful(v0, v_earth, v_escape, rho_x, sigma_e):
    """Convert repository unitful halo parameters to QCDark2 numeric units."""
    return {
        "v0": float(v0 / (nu.km / nu.s)),
        "vEarth": float(v_earth / (nu.km / nu.s)),
        "vEscape": float(v_escape / (nu.km / nu.s)),
        "rhoX": float(rho_x * nu.c0**2 / (nu.eV / nu.cm**3)),
        "sigma_e": float(sigma_e / nu.cm**2),
    }


def qcdark2_astro_model_from_numeric(v0, v_earth, v_escape, rho_x, sigma_e):
    """Return a QCDark2 numeric astro model from already-normalized inputs."""
    return {
        "v0": float(v0),
        "vEarth": float(v_earth),
        "vEscape": float(v_escape),
        "rhoX": float(rho_x),
        "sigma_e": float(sigma_e),
    }
