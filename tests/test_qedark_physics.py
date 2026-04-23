"""Physics validation tests for the QEDark form-factor backend."""

import numpy as np
import pytest
from scipy.integrate import simpson

from tests.conftest import (
    BENCHMARK_FDM,
    BENCHMARK_MASSES_MEV,
    DEFAULT_NE_BINS,
    hdf5_path,
    requires_qedark,
    requires_upstream_qcdark,
)
from tests.validation_helpers import (
    QEDarkFFAdapter,
    finite_positive_mask,
    load_upstream_qcdark,
    rates_to_events_per_kg_year,
    relative_error,
    set_dmerates_to_qcdark_astro,
)


pytestmark = pytest.mark.usefixtures("fix_units")


def _make_obj(material="Si"):
    from DMeRates.DMeRate import DMeRate

    qcdark = load_upstream_qcdark()
    obj = DMeRate(material, form_factor_type="qedark", device="cpu")
    return set_dmerates_to_qcdark_astro(obj, qcdark)


@requires_qedark("Si")
def test_linearity_rho_x():
    obj = _make_obj("Si")
    baseline = obj.calculate_total_rate([100.0], "imb", 0, integrate=False, DoScreen=False)[0]
    obj.update_params(238.0, 250.2, 544.0, 0.6e9, 1e-39)
    doubled = obj.calculate_total_rate([100.0], "imb", 0, integrate=False, DoScreen=False)[0]
    assert doubled == pytest.approx(2.0 * baseline, rel=1e-12)


@requires_qedark("Si")
def test_linearity_sigma_e():
    obj = _make_obj("Si")
    baseline = obj.calculate_total_rate([100.0], "imb", 0, integrate=False, DoScreen=False)[0]
    obj.update_crosssection(2e-39)
    doubled = obj.calculate_total_rate([100.0], "imb", 0, integrate=False, DoScreen=False)[0]
    assert doubled == pytest.approx(2.0 * baseline, rel=1e-12)


@requires_qedark("Si")
def test_band_gap_exact_zero():
    import numericalunits as nu

    obj = _make_obj("Si")
    energy_eV, spectra = obj.calculate_spectrum([100.0], "imb", 0, integrate=False, DoScreen=False)
    below_gap = energy_eV < float(obj.form_factor.band_gap / nu.eV)
    assert np.all(spectra[0, below_gap] == 0.0)


@requires_qedark("Si")
def test_wk4_not_double_applied():
    from DMeRates.DMeRate import DMeRate

    raw = np.loadtxt(hdf5_path("form_factors/QEDark/Si_f2.txt"), skiprows=1)
    raw_fcrys = np.transpose(np.resize(raw, (500, 900)))
    q_index, e_index = 8, 64
    expected = raw_fcrys[q_index, e_index] * (2.0 / 137.0) / 4.0

    dmrates_ff = DMeRate("Si", form_factor_type="qedark", device="cpu").form_factor.ff
    adapter = QEDarkFFAdapter("Si")
    assert expected > 0.0
    assert dmrates_ff[q_index, e_index] == pytest.approx(expected, rel=1e-15)
    assert adapter.ff[q_index, e_index] == pytest.approx(expected, rel=1e-15)


@requires_qedark("Si")
@pytest.mark.xfail(
    reason="Current QEDark non-integrated electron-bin folding carries the legacy factor-of-ten offset.",
    strict=False,
)
def test_ne_sum_equals_spectrum_integral_step_response():
    obj = _make_obj("Si")
    obj.change_to_step()
    energy_eV, spectra = obj.calculate_spectrum([100.0], "imb", 0, integrate=False, DoScreen=False)
    total = simpson(spectra[0], x=energy_eV)
    rates = obj.calculate_rates([100.0], "imb", 0, list(range(1, 21)), integrate=False, DoScreen=False)
    folded = np.sum(rates_to_events_per_kg_year(rates[:, 0]))
    assert folded == pytest.approx(total, rel=5e-2)


@requires_qedark("Si")
def test_integrate_false_deterministic():
    obj = _make_obj("Si")
    first = obj.calculate_spectrum([100.0], "imb", 0, integrate=False, DoScreen=False)[1]
    second = obj.calculate_spectrum([100.0], "imb", 0, integrate=False, DoScreen=False)[1]
    assert np.array_equal(first, second)


@pytest.mark.parametrize("material", ["Si", "Ge"])
@pytest.mark.parametrize("fdm_n", BENCHMARK_FDM)
@pytest.mark.parametrize("mass_mev", BENCHMARK_MASSES_MEV)
@requires_upstream_qcdark()
@requires_qedark("Si")
@pytest.mark.xfail(
    reason="QEDark spectrum parity currently exposes legacy normalization/grid differences.",
    strict=False,
)
def test_reference_spectrum(material, fdm_n, mass_mev):
    qcdark = load_upstream_qcdark()
    adapter = QEDarkFFAdapter(material)
    energy_ref, spectrum_ref = qcdark.d_rate(
        mass_mev * 1e6,
        adapter,
        FDM_exp=fdm_n,
        screening=qcdark.default_no_sreen,
        astro_model=qcdark.default_astro,
    )

    obj = _make_obj(material)
    energy_dm, spectra = obj.calculate_spectrum([mass_mev], "imb", fdm_n, integrate=True, DoScreen=False)
    assert np.allclose(energy_dm, energy_ref)
    mask = finite_positive_mask(spectrum_ref)
    np.testing.assert_allclose(spectra[0, mask], spectrum_ref[mask], rtol=5e-2, atol=0.0)

    total_ref = simpson(spectrum_ref, x=energy_ref)
    total_dm = simpson(spectra[0], x=energy_dm)
    assert relative_error(total_dm, total_ref) < 5e-2


@pytest.mark.parametrize("material", ["Si", "Ge"])
@pytest.mark.parametrize("fdm_n", BENCHMARK_FDM)
@requires_upstream_qcdark()
@requires_qedark("Si")
@pytest.mark.xfail(
    reason="QEDark electron-bin parity is not yet within validation tolerance.",
    strict=False,
)
def test_reference_ne_step(material, fdm_n):
    qcdark = load_upstream_qcdark()
    adapter = QEDarkFFAdapter(material)
    _, reference_bins = qcdark.d_rate_FanoQ(
        100e6,
        adapter,
        {"Si": 3.8, "Ge": 3.0}[material],
        FDM_exp=fdm_n,
        screening=qcdark.default_no_sreen,
        astro_model=qcdark.default_astro,
    )

    obj = _make_obj(material)
    obj.change_to_step()
    rates = obj.calculate_rates([100.0], "imb", fdm_n, DEFAULT_NE_BINS, integrate=True, DoScreen=False)
    candidate = rates_to_events_per_kg_year(rates[:, 0])
    np.testing.assert_allclose(candidate, reference_bins[1:6], rtol=5e-2, atol=0.0)


@requires_qedark("Si")
@pytest.mark.xfail(
    reason="Current QEDark Riemann and Simpson paths are not converged to 5%.",
    strict=False,
)
def test_integrate_true_vs_false_close():
    obj = _make_obj("Si")
    false_total = obj.calculate_total_rate([100.0], "imb", 0, integrate=False, DoScreen=False)[0]
    true_total = obj.calculate_total_rate([100.0], "imb", 0, integrate=True, DoScreen=False)[0]
    assert true_total == pytest.approx(false_total, rel=5e-2)
