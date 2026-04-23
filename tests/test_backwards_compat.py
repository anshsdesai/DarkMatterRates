"""Backward-compatibility tests — ensure existing API still works unchanged."""
import os, sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.conftest import requires_qcdark, requires_qedark


@requires_qcdark('Si')
def test_qedark_false_default_works():
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si', QEDark=False)
    assert obj.form_factor_type == 'qcdark'


@requires_qedark('Si')
def test_qedark_true_works():
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si', QEDark=True)
    assert obj.form_factor_type == 'qedark'


@requires_qcdark('Ge')
def test_ge_qcdark_works():
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Ge', QEDark=False)
    assert obj.material == 'Ge'
    assert obj.form_factor_type == 'qcdark'


@requires_qcdark('Si')
def test_calculate_rates_signature_unchanged():
    """Old call signature must still work: calculate_rates(masses, halo, fdm, ne)."""
    import torch
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si')
    rates = obj.calculate_rates([100.0], 'shm', 0, [1, 2])
    assert rates.shape == (2, 1)


def test_noble_xe_unaffected():
    from DMeRates.DMeRate import DMeRate
    import os
    xe_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'form_factors/wimprates/Xe_dme_ionization_ff.pkl')
    if not os.path.exists(xe_path):
        pytest.skip("Xe form factor file not found")
    obj = DMeRate('Xe')
    assert obj.material == 'Xe'


def test_noble_ar_unaffected():
    from DMeRates.DMeRate import DMeRate
    import os
    ar_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'form_factors/wimprates/Ar_dme_ionization_ff.pkl')
    if not os.path.exists(ar_path):
        pytest.skip("Ar form factor file not found")
    obj = DMeRate('Ar')
    assert obj.material == 'Ar'


@requires_qcdark('Si')
def test_update_params_still_works():
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si')
    obj.update_params(238.0, 250.2, 544.0, 0.3e9, 1e-39)


@requires_qcdark('Si')
def test_update_crosssection_still_works():
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si')
    obj.update_crosssection(1e-40)
