"""Tests for scripts/import_verne_srdmbeam_scan.py."""

from __future__ import annotations

import json
from pathlib import Path


_FLUX_ROWS = "100.0\t1.0\n200.0\t2.0\n300.0\t3.0\n"


def _make_verne_point(tmp_path: Path, *, tag: str, mass: float, sigma: float) -> Path:
    point_dir = tmp_path / tag
    point_dir.mkdir()
    metadata = {
        "schema": "verne_srdm_beam_flux_v1",
        "mDM_MeV": mass,
        "sigmaE_cm2": sigma,
        "sigmaP_cm2": 8.0e-36,
        "depth_m": 1400.0,
        "num_angles": 2,
        "angle_convention": {
            "internal_gamma_deg": "0 below, 180 overhead",
            "file_isoangle_deg": "0 overhead, 180 below",
        },
        "file_isoangle_deg": [0.18, 179.82],
        "gamma_internal_deg": [179.82, 0.18],
    }
    (point_dir / "metadata.json").write_text(json.dumps(metadata))
    for ring in range(2):
        flux_name = (
            f"Differential_SRDM_Flux_mDM_{mass:.4f}MeV_"
            f"sigmaE_{sigma:.0e}cm2_isoangle_{ring:03d}.txt"
        )
        (point_dir / flux_name).write_text(_FLUX_ROWS)
    return point_dir


def _make_direct_source_grid(tmp_path: Path, *, mass: float, sigma: float) -> Path:
    source_root = tmp_path / "srdm_source_grid"
    point_tag = "mDM_10_MeV_sigmaE_1e-36_cm2"
    point_dir = source_root / "points" / point_tag
    point_dir.mkdir(parents=True)
    (point_dir / "Differential_SRDM_Flux.txt").write_text(_FLUX_ROWS)
    (source_root / "scan_manifest.json").write_text(
        json.dumps(
            [
                {
                    "point_tag": point_tag,
                    "mDM_MeV": mass,
                    "sigma_e_cm2": sigma,
                    "FDMn": 2,
                    "grid_name": "srdm_fdmq2_source_v1",
                    "source_flux_file": "/remote/path/not/on/this/machine.txt",
                }
            ]
        )
    )
    return source_root


def test_import_scan_manifest_imports_completed_points(tmp_path):
    from DMeRates.srdm.flux_loader import available_srdmbeam_ring_indices
    from scripts.import_verne_srdmbeam_scan import import_scan_manifest

    point_dir = _make_verne_point(
        tmp_path, tag="mDM_10_MeV_sigmaE_1e-36_cm2", mass=10.0, sigma=1e-36
    )
    manifest_path = tmp_path / "verne_scan_manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "point_tag": "complete_point",
                    "mDM_MeV": 10.0,
                    "sigma_e_cm2": 1e-36,
                    "FDMn": 2,
                    "status": "complete",
                    "verne_output_dir": str(point_dir),
                }
            ]
        )
    )
    output_root = tmp_path / "halo_data"

    results = import_scan_manifest(manifest_path, output_root)

    assert [result.status for result in results] == ["imported"]
    assert results[0].output_dir == (
        output_root / "modulated" / "FDMq2" / "Verne" / "SRDMBeam"
        / "mDM_10_0_MeV_sigmaE_1e-36_cm2"
    )
    assert available_srdmbeam_ring_indices(
        10.0, 1e-36, 2, modulated_source="Verne", base_data_dir=output_root
    ) == [0, 1]


def test_import_scan_manifest_skips_incomplete_points(tmp_path):
    from scripts.import_verne_srdmbeam_scan import import_scan_manifest

    point_dir = _make_verne_point(
        tmp_path, tag="mDM_10_MeV_sigmaE_1e-36_cm2", mass=10.0, sigma=1e-36
    )
    manifest_path = tmp_path / "verne_scan_manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "point_tag": "queued_point",
                    "mDM_MeV": 10.0,
                    "sigma_e_cm2": 1e-36,
                    "FDMn": 2,
                    "status": "queued",
                    "verne_output_dir": str(point_dir),
                }
            ]
        )
    )

    results = import_scan_manifest(manifest_path, tmp_path / "halo_data")

    assert [result.status for result in results] == ["skipped"]


def test_import_scan_manifest_dry_run_does_not_write(tmp_path):
    from scripts.import_verne_srdmbeam_scan import import_scan_manifest

    point_dir = _make_verne_point(
        tmp_path, tag="mDM_10_MeV_sigmaE_1e-36_cm2", mass=10.0, sigma=1e-36
    )
    manifest_path = tmp_path / "verne_scan_manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "point_tag": "complete_point",
                    "mDM_MeV": 10.0,
                    "sigma_e_cm2": 1e-36,
                    "FDMn": 2,
                    "status": "complete",
                    "verne_output_dir": str(point_dir),
                }
            ]
        )
    )
    output_root = tmp_path / "halo_data"

    results = import_scan_manifest(manifest_path, output_root, dry_run=True)

    assert [result.status for result in results] == ["dry-run"]
    assert not output_root.exists()


def test_import_scan_manifest_can_register_matching_direct_srdm(tmp_path):
    from scripts.import_verne_srdmbeam_scan import import_scan_manifest

    point_dir = _make_verne_point(
        tmp_path, tag="mDM_10_MeV_sigmaE_1e-36_cm2", mass=10.0, sigma=1e-36
    )
    source_root = _make_direct_source_grid(tmp_path, mass=10.0, sigma=1e-36)
    manifest_path = tmp_path / "verne_scan_manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "point_tag": "complete_point",
                    "mDM_MeV": 10.0,
                    "sigma_e_cm2": 1e-36,
                    "FDMn": 2,
                    "status": "complete",
                    "verne_output_dir": str(point_dir),
                }
            ]
        )
    )
    output_root = tmp_path / "halo_data"

    results = import_scan_manifest(
        manifest_path,
        output_root,
        with_direct_srdm=True,
        direct_source_root=source_root,
    )

    assert [result.status for result in results] == ["imported"]
    direct_manifest = json.loads((output_root / "srdm" / "manifest.json").read_text())
    assert len(direct_manifest["files"]) == 1
    entry = direct_manifest["files"][0]
    assert entry["source"] == "DaMaSCUS-SUN"
    assert entry["grid_family"] == "srdm_fdmq2_source_v1"
    assert entry["mX_eV"] == 10.0e6
    assert entry["sigma_e_cm2"] == 1e-36
    assert (output_root / "srdm" / entry["filename"]).exists()
