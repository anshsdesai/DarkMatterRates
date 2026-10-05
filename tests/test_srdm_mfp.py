import numpy as np
import pytest


def test_sigma_e_to_sigma_p_matches_reduced_mass_convention():
    from modulation_study.MeanFreePath import Earth_Density_Layer_NU, sigma_e_to_sigma_p_cm2

    earth = Earth_Density_Layer_NU()
    mX_MeV = 1.0
    mX_GeV = mX_MeV * earth.MeV
    expected = 1e-36 * (earth.muXElem(mX_GeV, earth.mProton) / earth.muXElem(mX_GeV, earth.mElectron)) ** 2

    assert sigma_e_to_sigma_p_cm2(1e-36, mX_MeV) == pytest.approx(expected)


def test_flux_weighted_srdm_mean_free_path_uses_inverse_mfp_average(monkeypatch):
    import torch
    import DMeRates.srdm.flux_loader as flux_loader
    import modulation_study.MeanFreePath as mfp_module

    class FakeEarth:
        MeV = 1e-3
        mProton = 1.0
        mElectron = 0.5
        EarthRadius = 1.0

        def muXElem(self, mX, mass):
            return 1.0

        def Mean_Free_Path(self, _r, _mX, _sigmaP, v, _FDMn, doScreen=True):
            return 1.0 / v

    monkeypatch.setattr(mfp_module, "Earth_Density_Layer_NU", FakeEarth)

    def fake_load_srdm_flux(*args, **kwargs):
        return (
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([1.0, 1.0], dtype=torch.float64),
        )

    monkeypatch.setattr(flux_loader, "load_srdm_flux", fake_load_srdm_flux)

    result = mfp_module.flux_weighted_srdm_mean_free_path(1.0, 1e-36)

    # With MFP(v)=1/v and uniform flux over [1,2], <1/MFP>=<v>=1.5.
    assert result == pytest.approx(1.0 / 1.5)


def test_srdm_flux_weighted_mfp_contour_interpolates(monkeypatch):
    import modulation_study.MeanFreePath as mfp_module

    def fake_points(**kwargs):
        return (
            np.array([1.0, 1.0, 10.0, 10.0]),
            np.array([1e-40, 1e-38, 1e-40, 1e-38]),
            np.array([1.0, 2.0, 3.0, 4.0]),
        )

    monkeypatch.setattr(mfp_module, "get_srdm_flux_weighted_mfp_points", fake_points)
    masses, sigmas, grid = mfp_module.get_srdm_flux_weighted_mfp_contour_data(grid_size=5)

    assert masses.shape == (5,)
    assert sigmas.shape == (5,)
    assert grid.shape == (5, 5)
    assert np.isfinite(grid[0, 0])
