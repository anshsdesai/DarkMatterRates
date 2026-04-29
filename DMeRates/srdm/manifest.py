"""Manifest loader for SRDM flux files."""
import json
from DMeRates.data.registry import DataRegistry


def load_manifest() -> list[dict]:
    """Return parsed entries from halo_data/srdm/manifest.json."""
    manifest_path = DataRegistry.srdm_manifest()
    with open(manifest_path) as f:
        data = json.load(f)
    return data["files"]


def find_entry(mX_eV: float, sigma_e_cm2: float, FDMn: int,
               mediator_spin: str, rtol: float = 1e-6) -> dict | None:
    """Find a manifest entry matching (mX, sigma, FDMn, mediator_spin).

    Uses relative tolerance rtol for float comparisons (handles 5e4 vs 50000).
    Returns the matching dict or None if not found.
    """
    entries = load_manifest()
    for entry in entries:
        mX_match = abs(entry["mX_eV"] - mX_eV) / max(abs(mX_eV), 1e-300) < rtol
        sigma_match = abs(entry["sigma_e_cm2"] - sigma_e_cm2) / max(abs(sigma_e_cm2), 1e-300) < rtol
        fdmn_match = entry["FDMn"] == FDMn
        spin_match = entry["mediator_spin"] == mediator_spin
        if mX_match and sigma_match and fdmn_match and spin_match:
            return entry
    return None
