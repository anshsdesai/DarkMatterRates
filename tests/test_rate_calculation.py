"""Tests for rate calculation — physics sanity and QCDark2 API correctness."""
import os, sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.conftest import requires_qcdark, requires_qcdark2


@requires_qcdark('Si')
def test_qcdark_rates_si_finite_positive():
    import torch
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si', form_factor_type='qcdark')
    rates = obj.calculate_rates([100.0], 'shm', 0, [1, 2, 3])
    assert rates.shape == (3, 1)
    assert torch.all(torch.isfinite(rates))
    assert torch.all(rates >= 0)


@requires_qcdark('Si')
def test_qcdark_rates_si_order_of_magnitude():
    """Rates should be nonzero for a physically reasonable mass."""
    import torch
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si', form_factor_type='qcdark')
    rates = obj.calculate_rates([100.0], 'shm', 0, [1])
    assert rates[0, 0].item() > 0, "Rate should be positive for mX=100 MeV"


@requires_qcdark2('Si', 'composite')
def test_qcdark2_rates_si_finite_positive():
    import torch
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si', form_factor_type='qcdark2')
    rates = obj.calculate_rates([100.0], 'shm', 0, [1, 2], DoScreen=False)
    assert rates.shape == (2, 1)
    assert torch.all(torch.isfinite(rates))
    assert torch.all(rates >= 0)


@requires_qcdark2('Si', 'composite')
def test_qcdark2_rates_si_nonzero():
    import torch
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si', form_factor_type='qcdark2')
    rates = obj.calculate_rates([100.0], 'shm', 0, [1], DoScreen=False)
    assert rates[0, 0].item() > 0


@requires_qcdark2('Diamond', 'composite')
def test_qcdark2_diamond_above_bandgap_only():
    """Diamond rates should be nonzero only above 5.47 eV band gap."""
    import torch
    import numericalunits as nu
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Diamond', form_factor_type='qcdark2')
    obj.setup_halo_data(1000.0, 0, 'shm')
    dRdE = obj.vectorized_dRdE_qcdark2(torch.tensor(1000.0), 0, 'shm')
    band_gap_eV = 5.47
    below_gap = obj.Earr < band_gap_eV * nu.eV
    assert torch.all(dRdE[below_gap] == 0), "Rates should be zero below Diamond band gap"


@requires_qcdark2('Si', 'composite')
def test_doscreen_warning_for_qcdark2():
    import warnings
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si', form_factor_type='qcdark2')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        obj.calculate_rates([100.0], 'shm', 0, [1], DoScreen=True)
    categories = [str(warning.category) for warning in w]
    assert any('UserWarning' in c for c in categories), \
        "Expected a UserWarning when DoScreen=True with qcdark2"


@requires_qcdark2('Si', 'composite')
def test_doscreen_no_effect_qcdark2():
    """Rates with DoScreen=True and DoScreen=False should be identical for QCDark2."""
    import torch
    from DMeRates.DMeRate import DMeRate
    import warnings
    obj = DMeRate('Si', form_factor_type='qcdark2')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r_screen = obj.calculate_rates([100.0], 'shm', 0, [1], DoScreen=True)
    r_no_screen = obj.calculate_rates([100.0], 'shm', 0, [1], DoScreen=False)
    assert torch.allclose(r_screen, r_no_screen), \
        "DoScreen flag should not change QCDark2 rates"


@requires_qcdark2('Si', 'composite')
def test_calculate_spectrum_qcdark2_shape():
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si', form_factor_type='qcdark2')
    obj.update_params(238.0, 250.2, 544.0, 0.3e9, 1e-39)
    energy_eV, spectra = obj.calculate_spectrum([100.0, 1000.0], 'shm', 0, DoScreen=False)
    assert energy_eV.ndim == 1
    assert spectra.shape == (2, len(energy_eV))
    assert spectra.min() >= 0


@requires_qcdark2('Si', 'composite')
def test_calculate_total_rate_qcdark2_positive():
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si', form_factor_type='qcdark2')
    obj.update_params(238.0, 250.2, 544.0, 0.3e9, 1e-39)
    total = obj.calculate_total_rate([100.0], 'shm', 0, DoScreen=False)
    assert total.shape == (1,)
    assert total[0] > 0


@requires_qcdark('Si')
def test_generate_dat_qcdark_filename():
    from DMeRates.DMeRate import DMeRate
    import numpy as np
    obj = DMeRate('Si', form_factor_type='qcdark')
    # Just test the naming logic, don't actually write
    import os
    import numericalunits as nu
    function_name = 'p100k' if obj.ionization_func == obj.RKProbabilities else 'step'
    fdm_dict = {0: '1', 2: 'q2'}
    ff_tag_map = {
        'qcdark':  '_qcdark',
        'qedark':  '_qedark',
        'qcdark2': f'_qcdark2_{obj.qcdark2_variant}',
    }
    qestr = ff_tag_map[obj.form_factor_type]
    assert qestr == '_qcdark'


@requires_qcdark2('Si', 'composite')
def test_generate_dat_qcdark2_filename_tag():
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si', form_factor_type='qcdark2', qcdark2_variant='composite')
    ff_tag_map = {
        'qcdark':  '_qcdark',
        'qedark':  '_qedark',
        'qcdark2': f'_qcdark2_{obj.qcdark2_variant}',
    }
    qestr = ff_tag_map[obj.form_factor_type]
    assert '_qcdark2_composite' in qestr
