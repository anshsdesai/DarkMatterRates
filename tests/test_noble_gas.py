"""
Regression tests for the noble-gas (wimprates) path.

Mirror of tests/wimprates_validation.ipynb, cells 4-5.
Physics: form_factor_type='wimprates', material='Xe', update_crosssection(4e-44),
         halo_model='shm', mX=1000 MeV, nes=range(1,17), returnShells=False.

Comparisons use np.allclose with rtol=0.05 (5%).
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pytest

import numericalunits as nu
from DMeRates.DMeRate import DMeRate
from conftest import WIMPRATES_REFS

NES = list(range(1, 17))
MX = 1000


@pytest.fixture(scope='module')
def xe_dmrate(fix_units):
    dm = DMeRate('Xe', form_factor_type='wimprates')
    dm.update_crosssection(4e-44)
    return dm


@pytest.mark.parametrize('FDMn', [0, 2])
def test_noble_gas_rates(FDMn, xe_dmrate):
    """Shell-summed Xe rates within 5% of frozen reference."""
    xe_dmrate.setup_halo_data(MX, FDMn, 'shm')
    rates = xe_dmrate.calculate_nobleGas_rates(
        MX, 'shm', FDMn, NES,
        isoangle=None, halo_id_params=None,
        useVerne=False, calcErrors=None,
        debug=False, returnShells=False,
    )
    # shape: (len(NES), len(mX_array)) = (16, 1) after internal transpose
    result = rates[:, 0].cpu().numpy() * nu.tonne * nu.year

    ref = WIMPRATES_REFS[FDMn]
    assert result.shape == ref.shape, (
        f"Shape mismatch for FDMn={FDMn}: {result.shape} vs {ref.shape}"
    )
    assert np.allclose(result, ref, rtol=0.05, atol=0.0), (
        f"Noble-gas rates outside 5% tolerance for FDMn={FDMn}. "
        f"Max rel diff: {np.max(np.abs(result - ref) / np.abs(ref)):.3e}"
    )
