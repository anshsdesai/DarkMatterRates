"""Tests for form factor class loading and interface compliance."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.conftest import hdf5_path, qcdark2_file, requires_qcdark, requires_qcdark2, requires_qedark


@requires_qcdark('Si')
def test_qcdark_si_loads():
    from DMeRates.form_factor import form_factor
    ff = form_factor(hdf5_path('form_factors/QCDark/Si_final.hdf5'))
    assert hasattr(ff, 'ff')
    assert hasattr(ff, 'dq')
    assert hasattr(ff, 'dE')
    assert hasattr(ff, 'mCell')
    assert hasattr(ff, 'band_gap')
    assert ff.ff.shape[0] > 0 and ff.ff.shape[1] > 0


@requires_qcdark('Ge')
def test_qcdark_ge_loads():
    from DMeRates.form_factor import form_factor
    ff = form_factor(hdf5_path('form_factors/QCDark/Ge_final.hdf5'))
    assert ff.ff.ndim == 2


@requires_qedark('Si')
def test_qedark_si_loads():
    from DMeRates.form_factor import form_factorQEDark
    ff = form_factorQEDark(hdf5_path('form_factors/QEDark/Si_f2.txt'))
    assert ff.ff.shape == (900, 500)
    assert hasattr(ff, 'band_gap')
    assert hasattr(ff, 'mCell')


@requires_qcdark2('Si', 'composite')
def test_qcdark2_si_loads():
    import numericalunits as nu
    from DMeRates.form_factor import form_factorQCDark2
    from DMeRates.Constants import qcdark2_band_gaps
    ff = form_factorQCDark2(qcdark2_file('Si'), band_gap=qcdark2_band_gaps['Si'])
    assert ff.eps.ndim == 2
    assert ff.eps.dtype == complex or np.iscomplexobj(ff.eps)
    assert ff.q_raw.ndim == 1
    assert ff.E_raw.ndim == 1
    assert ff.eps.shape == (len(ff.q_raw), len(ff.E_raw))


@requires_qcdark2('Si', 'composite')
def test_qcdark2_interface_complete():
    from DMeRates.form_factor import form_factorQCDark2
    from DMeRates.Constants import qcdark2_band_gaps
    ff = form_factorQCDark2(qcdark2_file('Si'), band_gap=qcdark2_band_gaps['Si'])
    for attr in ('eps', 'q_raw', 'E_raw', 'M_cell', 'V_cell', 'dE', 'band_gap', 'mCell'):
        assert hasattr(ff, attr), f"Missing attribute: {attr}"
    assert callable(ff.elf)
    assert callable(ff.S)


@requires_qcdark2('Si', 'composite')
def test_qcdark2_elf_nonnegative():
    from DMeRates.form_factor import form_factorQCDark2
    from DMeRates.Constants import qcdark2_band_gaps
    ff = form_factorQCDark2(qcdark2_file('Si'), band_gap=qcdark2_band_gaps['Si'])
    elf = ff.elf()
    assert elf.shape == ff.eps.shape
    assert np.all(elf >= -1e-10), "ELF has large negative values"


@requires_qcdark2('Si', 'composite')
def test_qcdark2_S_nonnegative():
    from DMeRates.form_factor import form_factorQCDark2
    from DMeRates.Constants import qcdark2_band_gaps
    ff = form_factorQCDark2(qcdark2_file('Si'), band_gap=qcdark2_band_gaps['Si'])
    S = ff.S()
    assert S.shape == ff.eps.shape
    assert np.all(S >= -1e-10), "Dynamic structure factor has large negative values"


@requires_qcdark2('Si', 'lfe')
def test_qcdark2_lfe_loads():
    from DMeRates.form_factor import form_factorQCDark2
    from DMeRates.Constants import qcdark2_band_gaps
    ff = form_factorQCDark2(qcdark2_file('Si', 'lfe'), band_gap=qcdark2_band_gaps['Si'])
    assert ff.eps.ndim == 2
