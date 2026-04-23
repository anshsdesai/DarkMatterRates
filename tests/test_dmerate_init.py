"""Tests for DMeRate constructor and form_factor_type parameter."""
import os, sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.conftest import requires_qcdark, requires_qcdark2, requires_qedark


@requires_qcdark('Si')
def test_default_is_qcdark():
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si')
    assert obj.form_factor_type == 'qcdark'
    assert obj.QEDark is False


@requires_qedark('Si')
def test_qedark_bool_compat():
    """Legacy QEDark=True still works and sets form_factor_type='qedark'."""
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si', QEDark=True)
    assert obj.form_factor_type == 'qedark'
    assert obj.QEDark is True


@requires_qcdark('Si')
def test_qcdark_explicit():
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si', form_factor_type='qcdark')
    assert obj.form_factor_type == 'qcdark'


@requires_qedark('Si')
def test_qedark_explicit():
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si', form_factor_type='qedark')
    assert obj.form_factor_type == 'qedark'


@requires_qcdark2('Si', 'composite')
def test_qcdark2_init_si():
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si', form_factor_type='qcdark2')
    assert obj.form_factor_type == 'qcdark2'
    assert obj.qcdark2_variant == 'composite'
    assert obj.qArr is not None
    assert obj.Earr is not None
    assert len(obj.qArr) > 0
    assert len(obj.Earr) > 0


@requires_qcdark2('Si', 'lfe')
def test_qcdark2_variant_lfe():
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si', form_factor_type='qcdark2', qcdark2_variant='lfe')
    assert obj.qcdark2_variant == 'lfe'


@requires_qcdark2('GaAs', 'composite')
def test_qcdark2_init_gaas():
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('GaAs', form_factor_type='qcdark2')
    assert obj.material == 'GaAs'
    assert obj.form_factor_type == 'qcdark2'


@requires_qcdark2('SiC', 'composite')
def test_qcdark2_init_sic():
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('SiC', form_factor_type='qcdark2')
    assert obj.material == 'SiC'


@requires_qcdark2('Diamond', 'composite')
def test_qcdark2_init_diamond():
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Diamond', form_factor_type='qcdark2')
    assert obj.material == 'Diamond'


def test_unknown_material_raises():
    from DMeRates.DMeRate import DMeRate
    with pytest.raises(ValueError, match="not supported"):
        DMeRate('Au')


def test_unknown_ff_type_raises():
    from DMeRates.DMeRate import DMeRate
    with pytest.raises((ValueError, FileNotFoundError)):
        DMeRate('Si', form_factor_type='unknown_type')


@requires_qcdark('Si')
def test_qcdark_si_ionization_is_rk():
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Si', form_factor_type='qcdark')
    assert obj.ionization_func == obj.RKProbabilities


@requires_qcdark2('Ge', 'composite')
def test_qcdark2_ge_uses_step():
    from DMeRates.DMeRate import DMeRate
    obj = DMeRate('Ge', form_factor_type='qcdark2')
    assert obj.ionization_func == obj.step_probabilities
