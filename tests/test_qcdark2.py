"""
Validation tests for the native QCDark2 dielectric engine.

Mirror of `tests/qcdark2_formula_derivation.ipynb`, sections 4 + 6.
Physics: material=Si (Si_comp.h5 composite dielectric), mX=1 GeV, mediator=heavy
         (FDMn=0), halo='imb' (Maxwell-Boltzmann via DM_Halo_Distributions),
         screening='rpa', default_astro (v0=238, vEarth=250.2, vEscape=544 km/s,
         rhoX=0.3 GeV/cm^3, sigma_e=1e-39 cm^2).

Production code path is `DMeRates.engines.dielectric.compute_dRdE`. Tests skip
cleanly when the QCDark2 HDF5 dielectric files are not present on disk.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pytest

import numericalunits as nu

from DMeRates.data.registry import DataRegistry
from conftest import QCDARK2_REFS


_SI_COMP_PATH = DataRegistry.qcdark2_dielectric('Si', 'composite')
_HDF5_AVAILABLE = _SI_COMP_PATH.is_file()
_GAAS_COMP_PATH = DataRegistry.qcdark2_dielectric('GaAs', 'composite')
_GAAS_HDF5_AVAILABLE = _GAAS_COMP_PATH.is_file()
_SI_LFE_PATH = DataRegistry.qcdark2_dielectric('Si', 'lfe')
_SI_LFE_AVAILABLE = _SI_LFE_PATH.is_file()
_QCDARK2_NON_SI_MATERIALS = ('Ge', 'GaAs', 'SiC', 'Diamond')
_SKIP_REASON = (
    f"QCDark2 dielectric data not found at {_SI_COMP_PATH}. "
    "Set DMERATES_QCDARK2_ROOT or place dielectric_functions/composite/Si_comp.h5."
)


@pytest.mark.skipif(not _HDF5_AVAILABLE, reason=_SKIP_REASON)
def test_qcdark2_si_heavy_rpa_matches_reference(fix_units):
    """Native Si/MB/RPA dR/dE within 1e-4 relative to QCDark2 reference."""
    from DMeRates.engines.dielectric import compute_dRdE

    rtol = 1e-4
    result = compute_dRdE(
        material='Si',
        mX_eV=1.0e9,
        FDMn=0,                # heavy mediator
        halo_model='imb',      # Maxwell-Boltzmann via DM_Halo_Distributions
        screening='rpa',
        variant='composite',
        sigma_e_cm2=1e-39,
        rhoX_eV_per_cm3=0.3e9,
    )

    refs = QCDARK2_REFS[('Si', 'heavy', 1e9)]
    E = result.E_eV
    dRdE = result.dRdE_per_kg_per_year_per_eV
    for E_target, expected in refs.items():
        idx = int(np.argmin(np.abs(E - E_target)))
        actual = float(dRdE[idx])
        rel = abs(actual - expected) / abs(expected)
        assert rel < rtol, (
            f"E={E[idx]:.2f} eV: dR/dE={actual:.6e} vs ref={expected:.6e} "
            f"(rel diff {rel:.3e}) exceeds {rtol:.0e} tolerance."
        )


@pytest.mark.skipif(not _HDF5_AVAILABLE, reason=_SKIP_REASON)
def test_qcdark2_screening_required(fix_units):
    """Calling the engine without explicit screening raises ValueError."""
    from DMeRates.engines.dielectric import compute_dRdE

    with pytest.raises(ValueError, match="explicit screening choice"):
        compute_dRdE(
            material='Si',
            mX_eV=1.0e9,
            FDMn=0,
            halo_model='imb',
            screening=None,
        )


@pytest.mark.skipif(not _HDF5_AVAILABLE, reason=_SKIP_REASON)
def test_qcdark2_invalid_screening_raises(fix_units):
    """Unknown screening keys raise ValueError with the supported list."""
    from DMeRates.engines.dielectric import compute_dRdE

    with pytest.raises(ValueError, match="screening='bogus' not recognized"):
        compute_dRdE(
            material='Si',
            mX_eV=1.0e9,
            FDMn=0,
            halo_model='imb',
            screening='bogus',
        )


@pytest.mark.skipif(not _HDF5_AVAILABLE, reason=_SKIP_REASON)
def test_qcdark2_returns_rate_spectrum(fix_units):
    """Engine output is a RateSpectrum carrying nu units consistent with the bare-float view."""
    from DMeRates.engines.dielectric import compute_dRdE
    from DMeRates.spectrum import RateSpectrum

    result = compute_dRdE(
        material='Si', mX_eV=1.0e9, FDMn=0,
        halo_model='imb', screening='rpa',
    )
    assert isinstance(result.spectrum, RateSpectrum)
    assert result.spectrum.material == 'Si'
    assert result.spectrum.backend == 'qcdark2'
    # Round-trip: spectrum.dR_dE * (kg*yr*eV) must equal the bare-float view.
    spec_bare = result.spectrum.dR_dE.cpu().numpy() * (nu.kg * nu.year * nu.eV)
    assert np.allclose(spec_bare, result.dRdE_per_kg_per_year_per_eV,
                       rtol=1e-12, atol=0.0)


# ----------------------------------------------------------------------------
# Pure-utility shape/sanity checks. These don't require the HDF5 files.
# ----------------------------------------------------------------------------

def test_mediator_factor_heavy_is_ones():
    from DMeRates.engines.dielectric import mediator_factor
    q_ame = np.linspace(0.1, 10.0, 21)
    f2 = mediator_factor(q_ame, FDMn=0)
    assert np.array_equal(f2, np.ones_like(q_ame))


def test_mediator_factor_light_inverse_q4():
    from DMeRates.engines.dielectric import mediator_factor
    q_ame = np.linspace(0.5, 5.0, 11)
    f2 = mediator_factor(q_ame, FDMn=2)
    expected = q_ame ** -4
    assert np.allclose(f2, expected, rtol=1e-12)


# No standard MB QCDark2 light-mediator reference is present in
# tests/conftest.py or tests/qcdark2_formula_derivation.ipynb yet. Keep the
# algebraic q^-4 check above, plus this runtime smoke guard, until a reference
# vector is recorded.
@pytest.mark.skipif(not _HDF5_AVAILABLE, reason=_SKIP_REASON)
def test_qcdark2_si_light_mediator_smoke_runs(fix_units):
    from DMeRates.engines.dielectric import compute_dRdE

    result = compute_dRdE(
        material='Si',
        mX_eV=1.0e9,
        FDMn=2,
        halo_model='imb',
        screening='rpa',
        variant='composite',
    )
    assert result.dRdE_per_kg_per_year_per_eV.shape == result.E_eV.shape
    mid = slice(40, 110)  # E ~ 4-11 eV
    assert np.all(np.isfinite(result.dRdE_per_kg_per_year_per_eV[mid]))
    assert np.any(result.dRdE_per_kg_per_year_per_eV[mid] > 0)


def test_mediator_factor_invalid_FDMn_raises():
    from DMeRates.engines.dielectric import mediator_factor
    with pytest.raises(ValueError, match="Unsupported FDMn=1"):
        mediator_factor(np.array([1.0, 2.0]), FDMn=1)


def test_v_min_bare_shape_and_value():
    from DMeRates.engines.dielectric import v_min_bare
    q_eV = np.array([1.0, 10.0, 100.0])
    E_eV = np.array([5.0, 50.0])
    mX_eV = 1e9
    v = v_min_bare(q_eV, E_eV, mX_eV)
    assert v.shape == (3, 2)
    # Spot-check: q=10 eV, E=50 eV, mX=1e9 -> q/(2 mX) ~ 5e-9, E/q = 5
    assert np.isclose(v[1, 1], 10.0 / (2 * 1e9) + 50.0 / 10.0)


def test_energy_loss_function_values():
    from DMeRates.engines.dielectric import energy_loss_function
    # eps = 2 + 1j -> ELF = 1 / (1 + 4) = 0.2
    eps = np.array([[2.0 + 1j]])
    elf = energy_loss_function(eps)
    assert np.isclose(elf[0, 0], 0.2)


@pytest.mark.parametrize("material", _QCDARK2_NON_SI_MATERIALS)
def test_qcdark2_non_si_material_smoke_runs(material, fix_units):
    """Full QCDark2 composite tree materials load and produce finite spectra."""
    path = DataRegistry.qcdark2_dielectric(material, 'composite')
    if not path.is_file():
        pytest.skip(f"{material} QCDark2 dielectric data unavailable at {path}.")

    from DMeRates.engines.dielectric import compute_dRdE

    result = compute_dRdE(
        material=material,
        mX_eV=1.0e9,
        FDMn=0,
        halo_model='imb',
        screening='rpa',
        variant='composite',
    )
    assert result.spectrum.material == material
    assert result.dRdE_per_kg_per_year_per_eV.shape == result.E_eV.shape
    assert np.all(np.isfinite(result.dRdE_per_kg_per_year_per_eV))
    assert np.any(result.dRdE_per_kg_per_year_per_eV > 0)


@pytest.mark.skipif(not _HDF5_AVAILABLE, reason=_SKIP_REASON)
def test_dmerate_qcdark2_si_calculates_rate(fix_units):
    """Legacy compatibility class can calculate Si rates with form_factor_type='qcdark2'."""
    from DMeRates.DMeRate import DMeRate

    dm = DMeRate("Si", form_factor_type="qcdark2")
    rates = dm.calculate_rates(
        mX_array=[1000.0],   # MeV
        halo_model="imb",
        FDMn=0,
        ne=[1],
        screening="rpa",
        variant="composite",
    )
    arr = rates.detach().cpu().numpy()
    assert arr.shape == (1, 1)
    assert np.all(np.isfinite(arr))
    assert np.all(arr >= 0.0)


@pytest.mark.skipif(not _HDF5_AVAILABLE, reason=_SKIP_REASON)
def test_ratecalculator_qcdark2_si_calculates_rate(fix_units):
    """RateCalculator wrapper can calculate Si rates with backend='qcdark2'."""
    from DMeRates.rate_calculator import RateCalculator

    calc = RateCalculator("Si", backend="qcdark2", variant="composite", screening="rpa")
    rates = calc.calculate_rates(
        mX_array=[1000.0],   # MeV
        halo_model="imb",
        FDMn=0,
        ne=[1],
    )
    arr = rates.detach().cpu().numpy()
    assert arr.shape == (1, 1)
    assert np.all(np.isfinite(arr))
    assert np.all(arr >= 0.0)


@pytest.mark.skipif(not _HDF5_AVAILABLE, reason=_SKIP_REASON)
def test_dmerate_qcdark2_requires_explicit_screening(fix_units):
    """QCDark2 DMeRate API enforces explicit screening selection."""
    from DMeRates.DMeRate import DMeRate

    dm = DMeRate("Si", form_factor_type="qcdark2")
    with pytest.raises(ValueError, match="explicit screening choice"):
        dm.calculate_rates(
            mX_array=[1000.0],   # MeV
            halo_model="imb",
            FDMn=0,
            ne=[1],
            screening=None,
        )


@pytest.mark.skipif(not _HDF5_AVAILABLE, reason=_SKIP_REASON)
def test_qcdark2_shm_halo_runs(fix_units):
    """'shm' halo model runs without error and returns finite positive rates."""
    from DMeRates.engines.dielectric import compute_dRdE
    result = compute_dRdE(
        material='Si', mX_eV=1e9, FDMn=0,
        halo_model='shm', screening='rpa',
    )
    assert result.dRdE_per_kg_per_year_per_eV.shape == result.E_eV.shape
    # Only check a handful of mid-range energies where rate is non-trivial
    mid = slice(40, 110)  # E ~ 4-11 eV
    assert np.all(np.isfinite(result.dRdE_per_kg_per_year_per_eV[mid]))
    assert np.any(result.dRdE_per_kg_per_year_per_eV[mid] > 0)


@pytest.mark.skipif(not _HDF5_AVAILABLE, reason=_SKIP_REASON)
def test_dmerate_qcdark2_shm_uses_vectorized_halo_path(fix_units):
    """Legacy DMeRate QCDark2 API reuses file-backed halo interpolation for 'shm'."""
    from DMeRates.DMeRate import DMeRate

    dm = DMeRate("Si", form_factor_type="qcdark2")
    rates = dm.calculate_rates(
        mX_array=[1000.0],   # MeV
        halo_model="shm",
        FDMn=0,
        ne=[1],
        screening="rpa",
        variant="composite",
    )
    arr = rates.detach().cpu().numpy()
    assert arr.shape == (1, 1)
    assert np.all(np.isfinite(arr))
    assert np.all(arr >= 0.0)


@pytest.mark.parametrize("halo_model", ("tsa", "dpl"))
@pytest.mark.skipif(not _HDF5_AVAILABLE, reason=_SKIP_REASON)
def test_qcdark2_tsa_dpl_accept_file_interpolated_halo_provider(halo_model, fix_units):
    """QCDark2 supports non-MB halo keys through file-style eta interpolation."""
    import torch
    from DMeRates.engines.dielectric import compute_dRdE
    from DMeRates.halo.file_loader import FileHaloProvider

    file_vmins = torch.linspace(0.0, 1.0e6, 512, dtype=torch.float64) * nu.km / nu.s
    file_etas = torch.exp(-torch.linspace(0.0, 6.0, 512, dtype=torch.float64)) * nu.s / nu.km
    provider = FileHaloProvider(file_vmins, file_etas)

    result = compute_dRdE(
        material='Si',
        mX_eV=1.0e9,
        FDMn=0,
        halo_model=halo_model,
        screening='rpa',
        variant='composite',
        eta_provider=provider.eta,
    )
    assert result.spectrum.metadata['halo_model'] == halo_model
    assert result.dRdE_per_kg_per_year_per_eV.shape == result.E_eV.shape
    mid = slice(40, 110)  # E ~ 4-11 eV
    assert np.all(np.isfinite(result.dRdE_per_kg_per_year_per_eV[mid]))
    assert np.any(result.dRdE_per_kg_per_year_per_eV[mid] > 0)


def test_tsa_dpl_scalar_generation_functions_use_package_constants(monkeypatch, fix_units):
    """Analytic halo generation helpers import package constants, not top-level modules."""
    import scipy.integrate
    from DMeRates.DM_Halo import DM_Halo_Distributions

    monkeypatch.setattr(scipy.integrate, "nquad", lambda *args, **kwargs: (1.0, 0.0))
    halo = DM_Halo_Distributions()
    above_support = halo.vEscape + halo.vEarth + 1.0 * nu.km / nu.s

    assert halo.etaTsa(above_support) == 0
    assert halo.etaDPL(above_support) == 0


@pytest.mark.skipif(
    not (_HDF5_AVAILABLE and _SI_LFE_AVAILABLE),
    reason="Si composite/lfe QCDark2 dielectric data unavailable.",
)
def test_dmerate_qcdark2_variant_reload_changes_spectrum(fix_units):
    """DMeRate QCDark2 wrapper reloads dielectric data when variant changes."""
    from DMeRates.DMeRate import DMeRate

    dm = DMeRate("Si", form_factor_type="qcdark2")
    composite = dm.calculate_qcdark2_spectrum(
        mX=1000.0,
        halo_model="imb",
        FDMn=0,
        screening="rpa",
        variant="composite",
    )
    lfe = dm.calculate_qcdark2_spectrum(
        mX=1000.0,
        halo_model="imb",
        FDMn=0,
        screening="rpa",
        variant="lfe",
    )

    composite_bare = composite.dR_dE.cpu().numpy()
    lfe_bare = lfe.dR_dE.cpu().numpy()
    assert dm.qcdark2_variant == "lfe"
    assert dm.form_factor.variant == "lfe"
    assert not np.allclose(composite_bare, lfe_bare, rtol=1e-12, atol=0.0)


@pytest.mark.skipif(not _GAAS_HDF5_AVAILABLE, reason="GaAs QCDark2 dielectric data unavailable.")
def test_qcdark2_pair_energy_required_for_gaas_ne_rates(fix_units):
    """GaAs ne-rate conversion requires explicit pair_energy by policy."""
    from DMeRates.DMeRate import DMeRate

    dm = DMeRate("GaAs", form_factor_type="qcdark2")
    with pytest.raises(ValueError, match="require an explicit pair_energy"):
        dm.calculate_rates(
            mX_array=[1000.0],   # MeV
            halo_model="imb",
            FDMn=0,
            ne=[1],
            screening="rpa",
            variant="composite",
        )


@pytest.mark.skipif(not _GAAS_HDF5_AVAILABLE, reason="GaAs QCDark2 dielectric data unavailable.")
def test_qcdark2_above_threshold_rate_available_for_gaas(fix_units):
    """Above-threshold QCDark2 rates are available for GaAs without pair-energy input."""
    from DMeRates.DMeRate import DMeRate

    dm = DMeRate("GaAs", form_factor_type="qcdark2")
    rate = dm.qcdark2_above_threshold_rate(
        mX=1000.0,        # MeV
        halo_model="imb",
        FDMn=0,
        screening="rpa",
        variant="composite",
    )
    value = float(rate * (nu.kg * nu.year))
    assert np.isfinite(value)
    assert value >= 0.0
