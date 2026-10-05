"""Build combined direct or solar constraint envelopes.

This script is the reusable version of the envelope-building cells in
``limits/constraints.ipynb``. It scans a mediator-specific constraints directory,
interpolates candidate curves onto a common mass grid, takes the smallest cross
section at each mass, and writes both the envelope CSV used by ``Constraints.py``
and metadata showing which input file won each grid point.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d


@dataclass(frozen=True)
class CandidateCurve:
    path: Path
    mass: np.ndarray
    sigma: np.ndarray


def _fdm_subdir(fdm: int) -> str:
    if fdm == 0:
        return "DM-e-FDM1"
    if fdm == 2:
        return "DM-e-FDMq2"
    raise ValueError("fdm must be 0 or 2")


def _output_name(kind: str, fdm: int) -> str:
    tag = "fdm1" if fdm == 0 else "fdmq2"
    if kind == "solar":
        return f"solar_current_constraints_{tag}.csv"
    if kind == "direct":
        return f"current_constraints_{tag}.csv"
    raise ValueError("kind must be 'solar' or 'direct'")


def _candidate_reason(filename: str, kind: str) -> tuple[bool, str]:
    name = filename.lower()
    if not name.endswith(".csv"):
        return False, "not_csv"
    if "_fixed" in name:
        return False, "fixed_variant"
    if "recast" in name:
        return False, "recast"
    if "constraints" in name and ("xenon_light_constraints" in name or "xenon_heavy_constraints" in name):
        return False, "precombined_target_constraints"

    is_solar_like = "solar" in name or "srdm" in name
    if kind == "solar":
        if not is_solar_like:
            return False, "not_solar_or_srdm"
        return True, "included_solar_or_srdm"

    if is_solar_like:
        return False, "solar_or_srdm_excluded_from_direct"
    return True, "included_direct"


def _load_curve(path: Path) -> CandidateCurve:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim != 2:
        raise ValueError(f"{path} is not a 2D CSV table")
    if data.shape[1] >= 3:
        mass = data[:, 1]
        sigma = data[:, 2]
    elif data.shape[1] >= 2:
        mass = data[:, 0]
        sigma = data[:, 1]
    else:
        raise ValueError(f"{path} has fewer than two columns")

    mask = np.isfinite(mass) & np.isfinite(sigma) & (mass > 0.0) & (sigma > 0.0)
    mass = np.asarray(mass[mask], dtype=float)
    sigma = np.asarray(sigma[mask], dtype=float)
    if mass.size < 2:
        raise ValueError(f"{path} has fewer than two valid positive points")

    order = np.argsort(mass)
    mass = mass[order]
    sigma = sigma[order]
    unique_mass, unique_indices = np.unique(mass, return_index=True)
    return CandidateCurve(path=path, mass=unique_mass, sigma=sigma[unique_indices])


def discover_candidates(limits_dir: Path, fdm: int, kind: str) -> tuple[list[CandidateCurve], list[dict[str, str]]]:
    source_dir = limits_dir / _fdm_subdir(fdm)
    candidates: list[CandidateCurve] = []
    audit: list[dict[str, str]] = []
    for path in sorted(source_dir.glob("*.csv")):
        include, reason = _candidate_reason(path.name, kind)
        status = "included" if include else "excluded"
        detail = reason
        if include:
            try:
                candidates.append(_load_curve(path))
            except Exception as exc:  # noqa: BLE001 - metadata should preserve load failures.
                status = "load_failed"
                detail = f"{reason}: {exc}"
        audit.append({"file": path.name, "status": status, "reason": detail})
    return candidates, audit


def build_envelope(candidates: list[CandidateCurve], mass_grid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    final_sigma = np.full_like(mass_grid, np.nan, dtype=float)
    winners = [""] * mass_grid.size
    for candidate in candidates:
        interp = interp1d(
            candidate.mass,
            candidate.sigma,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
            assume_sorted=True,
        )
        values = np.asarray(interp(mass_grid), dtype=float)
        for idx, value in enumerate(values):
            if not np.isfinite(value):
                continue
            if not np.isfinite(final_sigma[idx]) or value < final_sigma[idx]:
                final_sigma[idx] = value
                winners[idx] = candidate.path.name
    valid = np.isfinite(final_sigma) & (final_sigma > 0.0)
    return np.vstack((mass_grid[valid], final_sigma[valid])), [winners[idx] for idx in np.where(valid)[0]]


def write_point_metadata(path: Path, envelope: np.ndarray, winners: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mX_MeV", "sigma_e_cm2", "source_file"])
        writer.writeheader()
        for mass, sigma, winner in zip(envelope[0], envelope[1], winners):
            writer.writerow({"mX_MeV": mass, "sigma_e_cm2": sigma, "source_file": winner})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdm", type=int, choices=[0, 2], required=True)
    parser.add_argument("--kind", choices=["solar", "direct"], default="solar")
    parser.add_argument("--limits-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mass-min", type=float, default=None)
    parser.add_argument("--mass-max", type=float, default=None)
    parser.add_argument("--mass-points", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    args = parser.parse_args()

    candidates, audit = discover_candidates(args.limits_dir, args.fdm, args.kind)
    if not candidates:
        raise SystemExit(f"No {args.kind} candidates found for FDMn={args.fdm}")

    mass_min = args.mass_min if args.mass_min is not None else min(float(c.mass.min()) for c in candidates)
    mass_max = args.mass_max if args.mass_max is not None else max(float(c.mass.max()) for c in candidates)
    mass_grid = np.geomspace(mass_min, mass_max, args.mass_points)
    envelope, winners = build_envelope(candidates, mass_grid)

    output = args.output or args.limits_dir / _output_name(args.kind, args.fdm)
    metadata_json = args.metadata_json or output.with_name(output.stem + "_metadata.json")
    metadata_csv = args.metadata_csv or output.with_name(output.stem + "_metadata.csv")

    np.savetxt(output, envelope, delimiter=",")
    write_point_metadata(metadata_csv, envelope, winners)

    counts = Counter(winners)
    metadata = {
        "kind": args.kind,
        "FDMn": args.fdm,
        "limits_dir": str(args.limits_dir),
        "source_dir": str(args.limits_dir / _fdm_subdir(args.fdm)),
        "output": str(output),
        "metadata_csv": str(metadata_csv),
        "mass_min": float(mass_min),
        "mass_max": float(mass_max),
        "mass_points_requested": int(args.mass_points),
        "mass_points_written": int(envelope.shape[1]),
        "included_files": [candidate.path.name for candidate in candidates],
        "selection_counts": dict(sorted(counts.items())),
        "file_audit": audit,
    }
    metadata_json.write_text(json.dumps(metadata, indent=2))

    print(f"Wrote {output}")
    print(f"Wrote {metadata_json}")
    print(f"Wrote {metadata_csv}")
    print("Winning files:")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
