"""QCDark2 material metadata and yield-policy helpers."""

from __future__ import annotations

import numericalunits as nu


QCDARK2_BANDGAPS = {
    "Si": 1.1 * nu.eV,
    "Ge": 0.67 * nu.eV,
    "GaAs": 1.42 * nu.eV,
    "SiC": 2.36 * nu.eV,
    "Diamond": 5.5 * nu.eV,
}


def canonical_qcdark2_material(material: str) -> str:
    """Normalize QCDark2 material spelling."""
    if material == "diamond":
        return "Diamond"
    return material


def require_qcdark2_pair_energy(material: str, pair_energy) -> None:
    """Enforce explicit pair-energy policy for QCDark2 ne-rate conversions."""
    material = canonical_qcdark2_material(material)
    if material in {"GaAs", "SiC", "Diamond"} and pair_energy is None:
        raise ValueError(
            f"QCDark2 ne rates for {material} require an explicit pair_energy (eV). "
            "The QCDark2 paper does not provide validated pair energies for this material."
        )
