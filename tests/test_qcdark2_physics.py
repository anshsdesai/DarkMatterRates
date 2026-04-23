"""Physics validation tests for the QCDark2 dielectric-function backend."""

import os
import sys
import warnings

import numpy as np
import pytest
from scipy.integrate import simpson

from tests.conftest import (
    UPSTREAM_QCDARK2_ROOT,
    qcdark2_file,
    requires_qcdark2,
    requires_upstream_qcdark2,
    upstream_qcdark2_file,
)
from tests.validation_helpers import rates_to_events_per_kg_year


pytestmark = pytest.mark.usefixtures("fix_units")


ALL_MATERIALS = ["Si", "Ge", "GaAs", "SiC", "Diamond"]
MEDIATORS = [(0, "heavy"), (2, "light")]


def _make_obj(material="Si", variant="composite"):
    from DMeRates.DMeRate import DMeRate

    obj = DMeRate(material, form_factor_type="qcdark2", qcdark2_variant=variant, device="cpu")
    obj.update_params(238.0, 250.2, 544.0, 0.3e9, 1e-39)
    return obj


def _assert_qcdark2_spectrum_close(material, energy_eV, reference, candidate):
    from DMeRates.Constants import qcdark2_band_gaps
    import numericalunits as nu

    band_gap_eV = float(qcdark2_band_gaps[material] / nu.eV)
    floor = np.max(reference) * 1e-8
    mask = (energy_eV >= band_gap_eV) & (reference >= floor)
    if not np.any(mask):
        pytest.skip(f"No populated QCDark2 bins for {material} at this benchmark")
    np.testing.assert_allclose(candidate[mask], reference[mask], rtol=5e-2, atol=0.0)
    assert simpson(candidate, x=energy_eV) == pytest.approx(
        simpson(reference, x=energy_eV),
        rel=2e-2,
    )


@pytest.mark.parametrize("material", ALL_MATERIALS)
def test_band_gap_exact_zero_all_materials(material):
    if not os.path.exists(qcdark2_file(material, "composite")):
        pytest.skip(f"Missing QCDark2 composite file for {material}")

    from DMeRates.Constants import qcdark2_band_gaps
    import numericalunits as nu

    obj = _make_obj(material)
    energy_eV, spectra = obj.calculate_spectrum([100.0], "imb", 0, DoScreen=False)
    below_gap = energy_eV < float(qcdark2_band_gaps[material] / nu.eV)
    assert np.all(spectra[0, below_gap] == 0.0)


@requires_qcdark2("Si", "composite")
def test_doscreen_is_noop():
    obj = _make_obj("Si")
    with pytest.warns(UserWarning):
        screened = obj.calculate_total_rate([100.0], "imb", 0, DoScreen=True)
    unscreened = obj.calculate_total_rate([100.0], "imb", 0, DoScreen=False)
    np.testing.assert_array_equal(screened, unscreened)


@requires_qcdark2("Si", "composite")
def test_doscreen_raises_warning():
    obj = _make_obj("Si")
    with pytest.warns(UserWarning, match="DoScreen=True has no effect"):
        obj.calculate_rates([100.0], "imb", 0, [1], DoScreen=True)


@requires_qcdark2("Si", "composite")
def test_s_matrix_identity():
    import numericalunits as nu

    from DMeRates.form_factor import form_factorQCDark2
    from DMeRates.Constants import qcdark2_band_gaps

    ff = form_factorQCDark2(qcdark2_file("Si", "composite"), band_gap=qcdark2_band_gaps["Si"])
    expected = np.imag(ff.eps) / np.abs(ff.eps) ** 2
    expected *= ff.q_raw[:, None] ** 2 / (2.0 * np.pi * nu.alphaFS)
    np.testing.assert_allclose(ff.S(), expected, rtol=1e-6, atol=1e-12)


@requires_qcdark2("Si", "composite")
def test_linearity_rho_x():
    obj = _make_obj("Si")
    baseline = obj.calculate_total_rate([100.0], "imb", 0, DoScreen=False)[0]
    obj.update_params(238.0, 250.2, 544.0, 0.6e9, 1e-39)
    doubled = obj.calculate_total_rate([100.0], "imb", 0, DoScreen=False)[0]
    assert doubled == pytest.approx(2.0 * baseline, rel=1e-12)


@requires_qcdark2("Si", "composite")
def test_linearity_sigma_e():
    obj = _make_obj("Si")
    baseline = obj.calculate_total_rate([100.0], "imb", 0, DoScreen=False)[0]
    obj.update_crosssection(2e-39)
    doubled = obj.calculate_total_rate([100.0], "imb", 0, DoScreen=False)[0]
    assert doubled == pytest.approx(2.0 * baseline, rel=1e-12)


@requires_qcdark2("Si", "composite")
def test_ne_sum_rule():
    obj = _make_obj("Si")
    energy_eV, spectra = obj.calculate_spectrum([100.0], "imb", 0, DoScreen=False)
    total = simpson(spectra[0], x=energy_eV)
    rates = obj.calculate_rates([100.0], "imb", 0, list(range(1, 21)), DoScreen=False)
    folded = np.sum(rates_to_events_per_kg_year(rates[:, 0]))
    assert folded == pytest.approx(total, rel=1e-6)


@requires_qcdark2("Si", "composite")
@requires_qcdark2("Si", "lfe")
@requires_qcdark2("Si", "nolfe")
@pytest.mark.xfail(
    reason="Variant ordering is sensitive to numericalunits state after the existing QCDark2 reference module.",
    strict=False,
)
def test_variant_ordering():
    totals = {}
    for variant in ["nolfe", "lfe", "composite"]:
        totals[variant] = _make_obj("Si", variant).calculate_total_rate(
            [100.0],
            "imb",
            0,
            DoScreen=False,
        )[0]

    assert totals["lfe"] != pytest.approx(totals["nolfe"])
    assert min(totals["lfe"], totals["nolfe"]) < totals["composite"] < max(
        totals["lfe"],
        totals["nolfe"],
    )


@pytest.mark.parametrize("material", ["Ge", "Diamond"])
@pytest.mark.parametrize("mass_mev", [0.5, 5.0, 100.0, 1000.0])
@pytest.mark.parametrize(("fdm_n", "mediator"), MEDIATORS)
@requires_upstream_qcdark2("Ge", "composite")
@requires_qcdark2("Ge", "composite")
@pytest.mark.xfail(
    reason="Extended Ge/Diamond QCDark2 scans are unit-state-sensitive; strict parity lives in test_qcdark2_reference.py.",
    strict=False,
)
def test_composite_mass_scan_reference(material, mass_mev, fdm_n, mediator):
    if not os.path.exists(qcdark2_file(material, "composite")):
        pytest.skip(f"Missing local QCDark2 file for {material}")
    if not os.path.exists(upstream_qcdark2_file(material, "composite")):
        pytest.skip(f"Missing upstream QCDark2 file for {material}")

    if UPSTREAM_QCDARK2_ROOT not in sys.path:
        sys.path.insert(0, UPSTREAM_QCDARK2_ROOT)
    from qcdark2 import dark_matter_rates as dm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        epsilon = dm.load_epsilon(upstream_qcdark2_file(material, "composite"))
        reference, reference_energy = dm.get_dR_dE(
            epsilon,
            m_X=mass_mev * 1e6,
            mediator=mediator,
            astro_model=dm.default_astro,
            screening="RPA",
        )

    obj = _make_obj(material)
    energy_eV, spectra = obj.calculate_spectrum([mass_mev], "imb", fdm_n, DoScreen=False)
    assert np.allclose(energy_eV, reference_energy)
    _assert_qcdark2_spectrum_close(material, energy_eV, reference, spectra[0])
