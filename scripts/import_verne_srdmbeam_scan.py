#!/usr/bin/env python3
"""Import a Verne SRDMBeam scan manifest into DMeRates halo_data.

Example
-------
    python scripts/import_verne_srdmbeam_scan.py \\
        /path/to/verne_scan_manifest.json halo_data/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from DMeRates.srdm.flux_loader import (  # noqa: E402
    available_srdmbeam_ring_indices,
    load_srdmbeam_flux,
    load_srdmbeam_metadata,
)
from scripts.upstream_to_dmerates import convert_verne  # noqa: E402
from scripts.upstream_to_dmerates import convert_srdm_direct  # noqa: E402


@dataclass(frozen=True)
class ImportResult:
    point_tag: str
    output_dir: Path | None
    status: str
    message: str


@dataclass(frozen=True)
class DirectSourceIndex:
    root: Path
    rows: list[dict]


def _load_scan_manifest(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"No scan manifest found: {manifest_path}")
    rows = json.loads(manifest_path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"Verne scan manifest must be a JSON list: {manifest_path}")
    return rows


def _row_label(row: dict, index: int) -> str:
    return str(row.get("point_tag") or row.get("index") or f"row_{index}")


def _same_float(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=0.0)


def _load_direct_source_index(source_root: Path | None) -> DirectSourceIndex | None:
    if source_root is None:
        return None
    manifest_path = source_root / "scan_manifest.json"
    rows = _load_scan_manifest(manifest_path)
    return DirectSourceIndex(source_root, rows)


def _matching_direct_source_row(row: dict, index: DirectSourceIndex | None) -> dict | None:
    if index is None:
        return None

    mX_MeV = float(row["mDM_MeV"])
    sigma_e_cm2 = float(row["sigma_e_cm2"])
    FDMn = int(row["FDMn"])
    for source_row in index.rows:
        if int(source_row.get("FDMn", -1)) != FDMn:
            continue
        if not _same_float(source_row.get("mDM_MeV"), mX_MeV):
            continue
        if not _same_float(source_row.get("sigma_e_cm2"), sigma_e_cm2):
            continue
        return source_row
    return None


def _local_direct_source_path(source_row: dict, index: DirectSourceIndex) -> Path:
    source_flux_file = source_row.get("source_flux_file")
    if source_flux_file is not None:
        source_path = Path(source_flux_file)
        if source_path.exists():
            return source_path

    point_tag = source_row.get("point_tag")
    if point_tag is None:
        raise KeyError("Matched direct-source row is missing 'point_tag'")
    local_path = index.root / "points" / str(point_tag) / "Differential_SRDM_Flux.txt"
    if not local_path.exists():
        raise FileNotFoundError(
            f"No local direct SRDM flux file found for {point_tag!r}: {local_path}"
        )
    return local_path


def _direct_source_file(row: dict, direct_index: DirectSourceIndex | None) -> tuple[Path, str | None]:
    direct_row = _matching_direct_source_row(row, direct_index)
    if direct_row is not None and direct_index is not None:
        grid_family = direct_row.get("grid_name")
        return _local_direct_source_path(direct_row, direct_index), grid_family

    source_flux_file = row.get("source_flux_file")
    if source_flux_file is None:
        raise KeyError(
            "Verne scan row has no 'source_flux_file'. Pass --direct-source-root "
            "so the importer can match the DaMaSCUS-SUN source-grid manifest."
        )
    source_path = Path(source_flux_file)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Direct SRDM source flux file does not exist: {source_path}. "
            "Pass --direct-source-root to resolve local source-grid paths."
        )
    return source_path, row.get("grid_name")


def _verify_import(row: dict, output_root: Path) -> str:
    mX_MeV = float(row["mDM_MeV"])
    sigma_e_cm2 = float(row["sigma_e_cm2"])
    FDMn = int(row["FDMn"])

    rings = available_srdmbeam_ring_indices(
        mX_MeV,
        sigma_e_cm2,
        FDMn,
        modulated_source="Verne",
        base_data_dir=output_root,
    )
    if not rings:
        raise FileNotFoundError(
            "No SRDMBeam ring files found after import for "
            f"mX={mX_MeV} MeV, sigma_e={sigma_e_cm2}, FDMn={FDMn}"
        )

    metadata = load_srdmbeam_metadata(
        mX_MeV,
        sigma_e_cm2,
        FDMn,
        modulated_source="Verne",
        base_data_dir=output_root,
    )
    load_srdmbeam_flux(
        mX_MeV,
        sigma_e_cm2,
        FDMn,
        rings[0],
        modulated_source="Verne",
        base_data_dir=output_root,
    )
    return (
        f"{len(rings)} rings, first={rings[0]}, last={rings[-1]}, "
        f"metadata_ring_count={metadata['ring_count']}"
    )


def _verify_direct_import(row: dict, output_root: Path) -> str:
    manifest_path = output_root / "srdm" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mX_eV = float(row["mDM_MeV"]) * 1.0e6
    sigma_e_cm2 = float(row["sigma_e_cm2"])
    FDMn = int(row["FDMn"])

    matches = [
        entry for entry in manifest.get("files", [])
        if _same_float(entry.get("mX_eV"), mX_eV)
        and _same_float(entry.get("sigma_e_cm2"), sigma_e_cm2)
        and int(entry.get("FDMn", -1)) == FDMn
        and entry.get("mediator_spin") == "vector"
    ]
    if not matches:
        raise FileNotFoundError(
            "No direct SRDM manifest entry found after import for "
            f"mX_eV={mX_eV}, sigma_e={sigma_e_cm2}, FDMn={FDMn}"
        )

    flux_file = output_root / "srdm" / matches[-1]["filename"]
    if not flux_file.exists():
        raise FileNotFoundError(f"Direct SRDM manifest file is missing: {flux_file}")
    with flux_file.open() as handle:
        data_rows = [
            line for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not data_rows:
        raise ValueError(f"Direct SRDM flux file has no data rows: {flux_file}")
    return f"manifest_entry={matches[-1]['filename']}, rows={len(data_rows)}"


def import_scan_manifest(
    manifest_path: Path,
    output_root: Path,
    *,
    dry_run: bool = False,
    verify: bool = True,
    include_incomplete: bool = False,
    with_direct_srdm: bool = False,
    direct_source_root: Path | None = None,
    direct_grid_family: str | None = None,
) -> list[ImportResult]:
    rows = _load_scan_manifest(manifest_path)
    results: list[ImportResult] = []
    direct_index = _load_direct_source_index(direct_source_root)

    for index, row in enumerate(rows):
        label = _row_label(row, index)
        status = row.get("status")
        if status != "complete" and not include_incomplete:
            results.append(
                ImportResult(label, None, "skipped", f"status={status!r}")
            )
            continue

        try:
            input_dir = Path(row["verne_output_dir"])
            FDMn = int(row["FDMn"])
        except KeyError as exc:
            raise KeyError(f"Manifest row {label!r} is missing {exc.args[0]!r}") from exc

        if dry_run:
            direct_note = ""
            if with_direct_srdm:
                direct_path, _ = _direct_source_file(row, direct_index)
                direct_note = f"; would register direct SRDM {direct_path}"
            results.append(
                ImportResult(
                    label, None, "dry-run", f"would import {input_dir}{direct_note}"
                )
            )
            continue

        output_dir = convert_verne(input_dir, output_root, FDMn=FDMn)
        messages = [
            f"Verne SRDMBeam: {_verify_import(row, output_root)}"
            if verify else "Verne SRDMBeam: not verified"
        ]
        if with_direct_srdm:
            direct_path, grid_family = _direct_source_file(row, direct_index)
            direct_out = convert_srdm_direct(
                direct_path,
                output_root,
                FDMn,
                "damascus-sun",
                mX_MeV=float(row["mDM_MeV"]),
                sigma_e_cm2=float(row["sigma_e_cm2"]),
                mediator_spin="vector",
                grid_family=direct_grid_family or grid_family,
            )
            if verify:
                messages.append(f"direct SRDM: {_verify_direct_import(row, output_root)}")
            else:
                messages.append(f"direct SRDM: {direct_out}")
        message = "; ".join(messages)
        results.append(ImportResult(label, output_dir, "imported", message))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Import all completed Verne SRDMBeam scan-manifest points into "
            "DMeRates halo_data/modulated/{FDM}/Verne/SRDMBeam."
        )
    )
    parser.add_argument("manifest_path", type=Path, help="Verne scan manifest JSON.")
    parser.add_argument(
        "output_root",
        type=Path,
        help="DMeRates halo-data root directory, usually halo_data/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the points that would be imported without writing files.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip loader-level verification after each import.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Attempt rows whose manifest status is not complete.",
    )
    parser.add_argument(
        "--with-direct-srdm",
        action="store_true",
        help=(
            "Also register the matching DaMaSCUS-SUN direct SRDM source flux "
            "for each Verne point in halo_data/srdm/manifest.json."
        ),
    )
    parser.add_argument(
        "--direct-source-root",
        type=Path,
        default=None,
        help=(
            "DaMaSCUS-SUN source-grid root containing scan_manifest.json and "
            "points/. Use this when Verne manifest source paths point to a "
            "different machine."
        ),
    )
    parser.add_argument(
        "--direct-grid-family",
        default=None,
        help="Optional grid_family label for direct SRDM manifest entries.",
    )
    args = parser.parse_args()

    results = import_scan_manifest(
        args.manifest_path,
        args.output_root,
        dry_run=args.dry_run,
        verify=not args.no_verify,
        include_incomplete=args.include_incomplete,
        with_direct_srdm=args.with_direct_srdm,
        direct_source_root=args.direct_source_root,
        direct_grid_family=args.direct_grid_family,
    )

    for result in results:
        if result.output_dir is None:
            print(f"{result.status}: {result.point_tag} ({result.message})")
        else:
            print(
                f"{result.status}: {result.point_tag} -> "
                f"{result.output_dir} ({result.message})"
            )


if __name__ == "__main__":
    main()
