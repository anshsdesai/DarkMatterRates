"""
Validation tests for the native QCDark2 SRDM dielectric engine.

Mirror of `tests/qcdark2_srdm_derivation.ipynb`, Sections 5 + 8.
Physics: material=Si (Si_comp.h5 composite dielectric), mX=48232.9466 eV
         (nearest grid point to 50 keV), sigma_e=1.098541e-38 cm^2,
         FDMn=2 (light mediator, m_A'=0), mediator_spin='vector',
         screening='rpa', variant='composite'.

Production code path is `DMeRates.engines.dielectric.compute_dRdE` with
`halo_model='srdm'`. Tests skip cleanly when the QCDark2 HDF5 dielectric
files are not present on disk.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pytest
import numericalunits as nu

from DMeRates.data.registry import DataRegistry
from conftest import QCDARK2_SRDM_REFS


_SI_COMP_PATH = DataRegistry.qcdark2_dielectric('Si', 'composite')
_FLUX_PATH = DataRegistry.srdm_flux_file('srdm_dphidv_DPLM_row10_col8.txt')
QCDARK2_DATA_AVAILABLE = _SI_COMP_PATH.is_file() and _FLUX_PATH.is_file()
_SKIP_REASON = (
    f"QCDark2 SRDM data not found. Need: {_SI_COMP_PATH} and {_FLUX_PATH}."
)

# Actual manifest grid point for the nominal 50 keV benchmark.
_MX_EV = 48232.9466
_SIGMA_E_CM2 = 1.098541e-38


@pytest.mark.skipif(not QCDARK2_DATA_AVAILABLE, reason=_SKIP_REASON)
def test_qcdark2_srdm_si_vector_light(fix_units):
    """Native Si SRDM dR/dE within 5% of notebook reference at validated bins."""
    from DMeRates.engines.dielectric import compute_dRdE

    res = compute_dRdE(
        material='Si',
        mX_eV=_MX_EV,
        sigma_e_cm2=_SIGMA_E_CM2,
        FDMn=2,
        mediator_spin='vector',
        halo_model='srdm',
        screening='rpa',
        variant='composite',
    )

    refs = QCDARK2_SRDM_REFS['Si_50keV_vector_light']
    for E_target, dRdE_ref in refs:
        idx = int(np.argmin(np.abs(res.E_eV - E_target)))
        actual = float(res.dRdE_per_kg_per_year_per_eV[idx])
        rel = abs(actual - dRdE_ref) / dRdE_ref
        assert rel < 0.05, (
            f"E={res.E_eV[idx]:.2f} eV: dR/dE={actual:.6e} vs ref={dRdE_ref:.6e} "
            f"(rel diff {rel*100:.4f}%) exceeds 5% tolerance."
        )


@pytest.mark.skipif(not QCDARK2_DATA_AVAILABLE, reason=_SKIP_REASON)
def test_qcdark2_srdm_returns_rate_spectrum_with_metadata(fix_units):
    """SRDM result carries the required RateSpectrum metadata fields."""
    from DMeRates.engines.dielectric import compute_dRdE
    from DMeRates.spectrum import RateSpectrum

    res = compute_dRdE(
        material='Si',
        mX_eV=_MX_EV,
        sigma_e_cm2=_SIGMA_E_CM2,
        FDMn=2,
        mediator_spin='vector',
        halo_model='srdm',
        screening='rpa',
    )
    assert isinstance(res.spectrum, RateSpectrum)
    assert res.spectrum.backend == 'qcdark2'
    meta = res.spectrum.metadata
    assert meta['halo_model'] == 'srdm'
    assert meta['mediator_spin'] == 'vector'
    assert 'flux_file' in meta
    assert meta['FDMn'] == 2
    assert meta['screening'] == 'rpa'
    spec_bare = res.spectrum.dR_dE.cpu().numpy() * (nu.kg * nu.year * nu.eV)
    assert np.allclose(spec_bare, res.dRdE_per_kg_per_year_per_eV,
                       rtol=1e-12, atol=0.0)


def test_qcdark2_srdm_unsupported_mediator_spin_raises():
    """mediator_spin='scalar' raises NotImplementedError listing planned modes."""
    from DMeRates.engines.dielectric import compute_dRdE

    with pytest.raises(NotImplementedError, match='scalar'):
        compute_dRdE(
            material='Si',
            mX_eV=_MX_EV,
            sigma_e_cm2=_SIGMA_E_CM2,
            FDMn=2,
            mediator_spin='scalar',
            halo_model='srdm',
            screening='rpa',
        )


def test_qcdark2_srdm_missing_manifest_entry_raises():
    """Unregistered (mX, sigma) pair raises FileNotFoundError citing the manifest."""
    from DMeRates.engines.dielectric import compute_dRdE

    with pytest.raises(FileNotFoundError, match='manifest'):
        compute_dRdE(
            material='Si',
            mX_eV=12345.0,
            sigma_e_cm2=1e-99,
            FDMn=2,
            mediator_spin='vector',
            halo_model='srdm',
            screening='rpa',
        )


def test_qcdark2_srdm_screening_required():
    """screening=None raises ValueError for SRDM path."""
    from DMeRates.engines.dielectric import compute_dRdE

    with pytest.raises(ValueError, match='screening'):
        compute_dRdE(
            material='Si',
            mX_eV=_MX_EV,
            sigma_e_cm2=_SIGMA_E_CM2,
            FDMn=2,
            mediator_spin='vector',
            halo_model='srdm',
            screening=None,
        )
