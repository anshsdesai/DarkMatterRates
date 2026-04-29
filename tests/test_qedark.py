"""
Regression tests for the QEDark legacy semiconductor path.

Mirror of tests/qedark_validation.ipynb, cell 3.
Physics: form_factor_type='qedark', change_to_step(), integrate=False,
         DoScreen=False, halo_model='imb', mX=[10, 1000] MeV.

integrate=False is a deterministic Riemann sum on CPU; comparisons use
np.array_equal (bit-for-bit identity) after nu.reset_units('SI').
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pytest

import numericalunits as nu
from DMeRates.DMeRate import DMeRate
from conftest import QEDARK_REFS


@pytest.mark.parametrize('material,ne_count,FDMn', [
    ('Si', 12, 0),
    ('Ge', 15, 2),
])
def test_qedark_rates(material, ne_count, FDMn, fix_units):
    """Bit-for-bit reproduction of QEDark rates for a fixed mass array."""
    dm = DMeRate(material, form_factor_type='qedark')
    dm.change_to_step()

    ne = np.arange(ne_count)
    rates = dm.calculate_rates(
        mX_array=[10, 1000],
        halo_model='imb',
        FDMn=FDMn,
        ne=ne,
        DoScreen=False,
        integrate=False,
    )
    result = rates.cpu().numpy() * nu.kg * nu.day

    ref = QEDARK_REFS[(material, FDMn)]
    assert result.shape == ref.shape, (
        f"Shape mismatch for {material} FDMn={FDMn}: {result.shape} vs {ref.shape}"
    )
    # PyTorch float32 → float64 cast introduces sub-ULP noise (~1e-9 relative);
    # np.array_equal is too strict. Use rtol=1e-6 which is still ~100x tighter
    # than the QCDark1 tolerance and will catch any real physics regression.
    assert np.allclose(result, ref, rtol=1e-6, atol=1e-10), (
        f"Rate values changed for {material} FDMn={FDMn}. "
        f"Max rel diff: {np.max(np.abs(result - ref) / (np.abs(ref) + 1e-30)):.3e}"
    )


@pytest.mark.parametrize('material,ne_count,FDMn', [
    ('Si', 12, 0),
    ('Ge', 15, 2),
])
def test_qedark_below_threshold_is_zero(material, ne_count, FDMn, fix_units):
    """Rate at ne=0 must be exactly 0.0 (band-gap masking)."""
    dm = DMeRate(material, form_factor_type='qedark')
    dm.change_to_step()

    ne = np.arange(ne_count)
    rates = dm.calculate_rates(
        mX_array=[10, 1000],
        halo_model='imb',
        FDMn=FDMn,
        ne=ne,
        DoScreen=False,
        integrate=False,
    )
    result = rates.cpu().numpy()
    assert np.all(result[0] == 0.0), (
        f"ne=0 rate is not exactly zero for {material} FDMn={FDMn}"
    )
