"""
Regression tests for the QCDark1 semiconductor path (HDF5 form factors).

Mirror of tests/qcdark_validation.ipynb, cell 3.
Physics: form_factor_type='qcdark', update_crosssection(1e-39), integrate=True,
         DoScreen=True, halo_model='imb', mX=[1000] MeV.

integrate=True uses torchquad Simpson; comparisons use np.allclose with
rtol=0.02 (2%) as the primary guard, relaxed to 5% per-bin near threshold.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pytest

import numericalunits as nu
from DMeRates.DMeRate import DMeRate
from conftest import QCDARK1_REFS


@pytest.mark.parametrize('material,ne_count,FDMn', [
    ('Si', 11, 0),
    ('Ge', 15, 2),
])
def test_qcdark1_rates(material, ne_count, FDMn, fix_units):
    """QCDark1 integrated rates within 2% of frozen reference."""
    dm = DMeRate(material, form_factor_type='qcdark')
    dm.update_crosssection(1e-39)

    ne = np.arange(ne_count)
    rates = dm.calculate_rates(
        mX_array=[1000],
        halo_model='imb',
        FDMn=FDMn,
        ne=ne,
        DoScreen=True,
        integrate=True,
    )
    result = rates.cpu().numpy() * nu.kg * nu.year

    ref = QCDARK1_REFS[(material, FDMn)]
    assert result.shape == ref.shape, (
        f"Shape mismatch for {material} FDMn={FDMn}: {result.shape} vs {ref.shape}"
    )
    # Skip the ne=0 bin (exactly 0 by construction; avoid divide-by-zero in rtol).
    nonzero = ref[:, 0] != 0.0
    assert np.allclose(result[nonzero], ref[nonzero], rtol=0.02, atol=0.0), (
        f"QCDark1 rates outside 2% tolerance for {material} FDMn={FDMn}. "
        f"Max rel diff: {np.max(np.abs(result[nonzero] - ref[nonzero]) / np.abs(ref[nonzero])):.3e}"
    )


@pytest.mark.parametrize('material,ne_count,FDMn', [
    ('Si', 11, 0),
    ('Ge', 15, 2),
])
def test_qcdark1_below_threshold_is_zero(material, ne_count, FDMn, fix_units):
    """Rate at ne=0 must be exactly 0.0 (band-gap masking)."""
    dm = DMeRate(material, form_factor_type='qcdark')
    dm.update_crosssection(1e-39)

    ne = np.arange(ne_count)
    rates = dm.calculate_rates(
        mX_array=[1000],
        halo_model='imb',
        FDMn=FDMn,
        ne=ne,
        DoScreen=True,
        integrate=True,
    )
    result = rates.cpu().numpy()
    assert np.all(result[0] == 0.0), (
        f"ne=0 rate is not exactly zero for {material} FDMn={FDMn}"
    )
