"""Architecture and backend-selection tests."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import requires_qcdark, requires_qcdark2, requires_qedark


@requires_qcdark('Si')
def test_qcdark_selects_source_family_backend():
    from DMeRates.DMeRate import DMeRate
    from DMeRates.backends import QCDarkBackend

    obj = DMeRate('Si', form_factor_type='qcdark')
    assert isinstance(obj.backend, QCDarkBackend)
    assert obj.backend.source_family == 'qcdark'


@requires_qedark('Si')
def test_qedark_selects_source_family_backend():
    from DMeRates.DMeRate import DMeRate
    from DMeRates.backends import QEDarkBackend

    obj = DMeRate('Si', form_factor_type='qedark')
    assert isinstance(obj.backend, QEDarkBackend)
    assert obj.backend.source_family == 'qedark'


@requires_qedark('Si')
def test_legacy_qedark_bool_selects_qedark_backend():
    from DMeRates.DMeRate import DMeRate
    from DMeRates.backends import QEDarkBackend

    obj = DMeRate('Si', QEDark=True)
    assert obj.form_factor_type == 'qedark'
    assert isinstance(obj.backend, QEDarkBackend)


@requires_qcdark2('Si', 'composite')
def test_qcdark2_selects_source_family_backend():
    from DMeRates.DMeRate import DMeRate
    from DMeRates.backends import QCDark2Backend

    obj = DMeRate('Si', form_factor_type='qcdark2')
    assert isinstance(obj.backend, QCDark2Backend)
    assert obj.backend is obj.qcdark2_backend
    assert obj.backend.source_family == 'qcdark2'


def test_wimprates_selects_source_family_backend_for_nobles():
    from DMeRates.DMeRate import DMeRate
    from DMeRates.backends import WimpratesBackend

    xe_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'form_factors/wimprates/Xe_dme_ionization_ff.pkl',
    )
    if not os.path.exists(xe_path):
        pytest.skip("Xe form factor file not found")

    obj = DMeRate('Xe')
    assert obj.form_factor_type == 'wimprates'
    assert isinstance(obj.backend, WimpratesBackend)


@requires_qcdark('Si')
def test_calculate_rates_alias_matches_calculate_ne_rates():
    import torch

    from DMeRates.DMeRate import DMeRate

    obj = DMeRate('Si', form_factor_type='qcdark')
    old = obj.calculate_rates([100.0], 'shm', 0, [1, 2])
    new = obj.calculate_ne_rates([100.0], 'shm', 0, [1, 2])
    assert torch.allclose(old, new)


def test_qcdark2_numeric_units_are_derived_from_constants():
    import numericalunits as nu

    from DMeRates.Constants import me_eV
    from DMeRates import units

    assert units.LIGHT_SPEED_KM_PER_S == pytest.approx(nu.c0 / (nu.km / nu.s))
    assert units.ALPHA == pytest.approx(nu.alphaFS)
    assert units.M_E_EV == pytest.approx(me_eV / nu.eV)
    assert units.KG_EV == pytest.approx(nu.kg * nu.c0**2 / nu.eV)
