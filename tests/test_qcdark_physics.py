"""Physics validation tests for the legacy QCDark backend."""

import numpy as np
import pytest
from scipy.integrate import simpson

from tests.conftest import (
    BENCHMARK_FDM,
    BENCHMARK_MASSES_MEV,
    DEFAULT_NE_BINS,
    hdf5_path,
    requires_qcdark,
    requires_upstream_qcdark,
)
from tests.validation_helpers import (
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
    obj = DMeRate(material, form_factor_type="qcdark", device="cpu")
    return set_dmerates_to_qcdark_astro(obj, qcdark)


@requires_qcdark("Si")
def test_linearity_rho_x():
    obj = _make_obj("Si")
    baseline = obj.calculate_total_rate([100.0], "imb", 0, integrate=False, DoScreen=False)[0]
    obj.update_params(238.0, 250.2, 544.0, 0.6e9, 1e-39)
    doubled = obj.calculate_total_rate([100.0], "imb", 0, integrate=False, DoScreen=False)[0]
    assert doubled == pytest.approx(2.0 * baseline, rel=1e-12)


@requires_qcdark("Si")
def test_linearity_sigma_e():
    obj = _make_obj("Si")
    baseline = obj.calculate_total_rate([100.0], "imb", 0, integrate=False, DoScreen=False)[0]
    obj.update_crosssection(2e-39)
    doubled = obj.calculate_total_rate([100.0], "imb", 0, integrate=False, DoScreen=False)[0]
    assert doubled == pytest.approx(2.0 * baseline, rel=1e-12)


@requires_qcdark("Si")
def test_band_gap_exact_zero():
    obj = _make_obj("Si")
    energy_eV, spectra = obj.calculate_spectrum([100.0], "imb", 0, integrate=False, DoScreen=False)
    band_gap_eV = obj.form_factor.band_gap
    import numericalunits as nu

    below_gap = energy_eV < float(band_gap_eV / nu.eV)
    assert np.all(spectra[0, below_gap] == 0.0)


@requires_qcdark("Si")
def test_fdm_scaling_matches_q_weighted_expectation():
    import torch

    obj = _make_obj("Si")
    obj.setup_halo_data(100.0, 0, "imb")
    debug0 = obj.vectorized_dRdE(100.0, 0, "imb", DoScreen=False, integrate=False, debug=True)
    debug2 = obj.vectorized_dRdE(100.0, 2, "imb", DoScreen=False, integrate=False, debug=True)

    q_weight = (1.0 / obj.qArr).detach().cpu()
    base = debug0["etas"].detach().cpu() * debug0["ff_arr"].detach().cpu() * q_weight
    expected = torch.sum(base * debug2["fdm_factor"].detach().cpu(), axis=1) / torch.sum(base, axis=1)
    observed = debug2["drde"].detach().cpu() / debug0["drde"].detach().cpu()
    mask = torch.isfinite(expected) & torch.isfinite(observed) & (debug0["drde"].detach().cpu() > 0)
    assert torch.allclose(observed[mask], expected[mask], rtol=1e-10, atol=0.0)


@requires_qcdark("Si")
def test_screening_lowers_rate():
    obj = _make_obj("Si")
    screened = obj.calculate_total_rate([100.0], "imb", 0, integrate=False, DoScreen=True)[0]
    unscreened = obj.calculate_total_rate([100.0], "imb", 0, integrate=False, DoScreen=False)[0]
    assert screened < unscreened


@requires_qcdark("Si")
def test_ionization_sum_rule():
    obj = _make_obj("Si")
    probabilities = obj.probabilities.detach().cpu().numpy()
    summed = np.sum(probabilities, axis=0)
    populated = summed > 1e-12
    np.testing.assert_allclose(summed[populated], 1.0, rtol=1e-12, atol=1e-12)


@requires_qcdark("Si")
def test_ne_sum_equals_spectrum_integral():
    obj = _make_obj("Si")
    energy_eV, spectra = obj.calculate_spectrum([100.0], "imb", 0, integrate=True, DoScreen=False)
    total = simpson(spectra[0], x=energy_eV)
    ne_rates = obj.calculate_rates([100.0], "imb", 0, list(range(1, 21)), integrate=True, DoScreen=False)
    folded = np.sum(rates_to_events_per_kg_year(ne_rates[:, 0]))
    assert folded == pytest.approx(total, rel=5e-2)


@pytest.mark.parametrize("material", ["Si", "Ge"])
@pytest.mark.parametrize("fdm_n", BENCHMARK_FDM)
@pytest.mark.parametrize("mass_mev", BENCHMARK_MASSES_MEV)
@requires_upstream_qcdark()
@requires_qcdark("Si")
@pytest.mark.xfail(
    reason="Legacy QCDark spectrum parity currently exposes a normalization/convergence gap.",
    strict=False,
)
def test_reference_spectrum(material, fdm_n, mass_mev):
    qcdark = load_upstream_qcdark()
    ff = qcdark.form_factor(hdf5_path(f"form_factors/QCDark/{material}_final.hdf5"))
    energy_ref, spectrum_ref = qcdark.d_rate(
        mass_mev * 1e6,
        ff,
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
    assert relative_error(total_dm, total_ref) < 2e-2


@pytest.mark.parametrize("material", ["Si", "Ge"])
@pytest.mark.parametrize("fdm_n", BENCHMARK_FDM)
@requires_upstream_qcdark()
@requires_qcdark("Si")
@pytest.mark.xfail(
    reason="Legacy QCDark electron-bin parity is not yet within validation tolerance.",
    strict=False,
)
def test_reference_ne_step(material, fdm_n):
    qcdark = load_upstream_qcdark()
    ff = qcdark.form_factor(hdf5_path(f"form_factors/QCDark/{material}_final.hdf5"))
    _, reference_bins = qcdark.d_rate_FanoQ(
        100e6,
        ff,
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


@requires_upstream_qcdark()
@requires_qcdark("Si")
@pytest.mark.xfail(
    reason="Legacy QCDark RK electron-bin parity is not yet within validation tolerance.",
    strict=False,
)
def test_reference_ne_rk():
    qcdark = load_upstream_qcdark()
    ff = qcdark.form_factor(hdf5_path("form_factors/QCDark/Si_final.hdf5"))
    _, reference_bins = qcdark.d_rate_RamanathanQ(
        100e6,
        ff,
        hdf5_path("DMeRates/p100K.dat"),
        FDM_exp=0,
        screening=qcdark.default_no_sreen,
        astro_model=qcdark.default_astro,
    )

    obj = _make_obj("Si")
    rates = obj.calculate_rates([100.0], "imb", 0, DEFAULT_NE_BINS, integrate=True, DoScreen=False)
    candidate = rates_to_events_per_kg_year(rates[:, 0])
    np.testing.assert_allclose(candidate, reference_bins[1:6], rtol=5e-2, atol=0.0)


@requires_qcdark("Si")
def test_integrate_false_deterministic():
    obj = _make_obj("Si")
    first = obj.calculate_spectrum([100.0], "imb", 0, integrate=False, DoScreen=False)[1]
    second = obj.calculate_spectrum([100.0], "imb", 0, integrate=False, DoScreen=False)[1]
    assert np.array_equal(first, second)


@requires_qcdark("Si")
@pytest.mark.xfail(
    reason="Current legacy Riemann and Simpson implementations are not converged to 5%.",
    strict=False,
)
def test_integrate_true_vs_false_close():
    obj = _make_obj("Si")
    false_total = obj.calculate_total_rate([100.0], "imb", 0, integrate=False, DoScreen=False)[0]
    true_total = obj.calculate_total_rate([100.0], "imb", 0, integrate=True, DoScreen=False)[0]
    assert true_total == pytest.approx(false_total, rel=5e-2)


@requires_qcdark("Si")
@pytest.mark.xfail(
    reason="Unit invariance requires reloading DMeRates constants after numericalunits resets.",
    strict=False,
)
def test_rate_unit_invariant(random_nu_seed):
    obj = _make_obj("Si")
    rate = obj.calculate_total_rate([100.0], "imb", 0, integrate=False, DoScreen=False)[0]
    assert rate == pytest.approx(7.860200167075e12, rel=1e-4)
