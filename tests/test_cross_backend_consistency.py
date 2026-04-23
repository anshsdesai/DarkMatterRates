"""Cross-backend sanity checks for shared materials and conventions."""

import h5py
import numpy as np
import pytest

from tests.conftest import requires_qcdark, requires_qcdark2, requires_qedark


pytestmark = pytest.mark.usefixtures("fix_units")


def _total(material, backend, fdm_n, mass_mev=100.0):
    from DMeRates.DMeRate import DMeRate

    obj = DMeRate(material, form_factor_type=backend, device="cpu")
    obj.update_params(238.0, 250.2, 544.0, 0.3e9, 1e-39)
    kwargs = {"DoScreen": False}
    if backend == "qcdark":
        kwargs["integrate"] = True
    elif backend == "qedark":
        kwargs["integrate"] = False
    return obj.calculate_total_rate([mass_mev], "imb", fdm_n, **kwargs)[0]


@requires_qcdark("Si")
@requires_qedark("Si")
@pytest.mark.xfail(
    reason="Legacy QCDark/QEDark totals are sensitive to current numericalunits state after parity sweeps.",
    strict=False,
)
def test_si_qcdark_vs_qedark_not_orders_of_magnitude_off():
    qcdark = _total("Si", "qcdark", 0)
    qedark = _total("Si", "qedark", 0)
    ratio = qcdark / qedark
    assert 0.1 < ratio < 10.0


@requires_qcdark("Si")
@requires_qcdark2("Si", "composite")
@pytest.mark.xfail(
    reason="QCDark and QCDark2 Si totals currently differ by more than the target 20%.",
    strict=False,
)
def test_si_qcdark_vs_qcdark2_within_20pct():
    qcdark = _total("Si", "qcdark", 0)
    qcdark2 = _total("Si", "qcdark2", 0)
    assert qcdark2 == pytest.approx(qcdark, rel=0.2)


@pytest.mark.parametrize("material", ["Si", "Ge"])
@requires_qcdark("Si")
@requires_qcdark2("Si", "composite")
@pytest.mark.xfail(
    reason="Current QCDark and QCDark2 cell masses differ beyond the 0.1% target.",
    strict=False,
)
def test_cell_mass_consistency(material):
    with h5py.File(f"form_factors/QCDark/{material}_final.hdf5", "r") as h5:
        qcdark_m_cell = float(h5["results"].attrs["mCell"])
    suffix = "comp"
    with h5py.File(f"form_factors/QCDark2/composite/{material}_{suffix}.h5", "r") as h5:
        qcdark2_m_cell = float(h5.attrs["M_cell"])
    assert qcdark2_m_cell == pytest.approx(qcdark_m_cell, rel=1e-3)


@pytest.mark.parametrize(
    "backend",
    [
        "qcdark",
        "qedark",
        pytest.param(
            "qcdark2",
            marks=pytest.mark.xfail(
                reason="QCDark2 cross-backend FDM ratio is unit-state-sensitive after legacy parity sweeps.",
                strict=False,
            ),
        ),
    ],
)
@requires_qcdark("Si")
@requires_qedark("Si")
@requires_qcdark2("Si", "composite")
def test_fdm_ratio_decreases_with_mass(backend):
    low_mass_ratio = _total("Si", backend, 2, mass_mev=10.0) / _total(
        "Si",
        backend,
        0,
        mass_mev=10.0,
    )
    high_mass_ratio = _total("Si", backend, 2, mass_mev=1000.0) / _total(
        "Si",
        backend,
        0,
        mass_mev=1000.0,
    )
    assert np.isfinite(low_mass_ratio)
    assert np.isfinite(high_mass_ratio)
    assert high_mass_ratio < low_mass_ratio
