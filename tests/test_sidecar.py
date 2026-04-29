"""Tests for generate_dat() YAML sidecar provenance files."""
import random
random.seed(0)

import os
import yaml
import pytest


def test_sidecar_roundtrip(tmp_path, monkeypatch):
    """generate_dat() writes a .dat.yaml sidecar that round-trips through PhysicsConfig."""
    from DMeRates.DMeRate import DMeRate
    from DMeRates.config import PhysicsConfig
    import numericalunits as nu

    dme = DMeRate('Si', form_factor_type='qedark')

    # Monkey-patch module_dir to write into tmp_path
    rates_dir = tmp_path / "Rates"
    rates_dir.mkdir()
    monkeypatch.setattr(dme, 'module_dir', str(tmp_path))
    # Also create the Rates subdir the code expects
    # (already done above, and module_dir is used with os.path.join(..., 'Rates/'))

    dm_masses = [10.0]
    ne_bins = [1, 2]

    data = dme.generate_dat(
        dm_masses=dm_masses,
        ne_bins=ne_bins,
        fdm=0,
        dm_halo_model='imb',
        DoScreen=False,
        write=True,
        tag="sidecar_test",
    )

    # Find the written .dat file
    dat_files = list(rates_dir.glob("*.dat"))
    assert len(dat_files) == 1, f"Expected 1 .dat file, found: {dat_files}"
    dat_path = dat_files[0]
    sidecar_path = str(dat_path) + ".yaml"
    assert os.path.exists(sidecar_path), "Sidecar .yaml not written"

    # Load and validate sidecar
    with open(sidecar_path) as f:
        sidecar = yaml.safe_load(f)

    assert 'package_version' in sidecar
    assert 'backend' in sidecar
    assert 'material' in sidecar
    assert sidecar['material'] == 'Si'
    assert 'physics' in sidecar
    assert 'halo_model' in sidecar

    # Round-trip through PhysicsConfig
    cfg = PhysicsConfig.from_dict(sidecar['physics'])
    assert abs(cfg.v0 / (nu.km / nu.s) - sidecar['physics']['v0_km_s']) < 1e-10
    assert abs(cfg.sigma_e / nu.cm**2 - sidecar['physics']['sigma_e_cm2']) < 1e-48
