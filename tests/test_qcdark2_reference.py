"""Reference-parity tests against the sibling upstream QCDark2 checkout."""

import os
import sys
import warnings

import numpy as np
import numericalunits as nu
import pytest
from scipy.integrate import simpson

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import (  # noqa: E402
    UPSTREAM_QCDARK2_ROOT,
    qcdark2_file,
    requires_qcdark2,
    requires_upstream_qcdark2,
    upstream_qcdark2_file,
)

sys.path.insert(0, UPSTREAM_QCDARK2_ROOT)
from qcdark2 import dark_matter_rates as dm  # noqa: E402

from DMeRates.Constants import qcdark2_band_gaps  # noqa: E402
from DMeRates.DMeRate import DMeRate  # noqa: E402

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"qcdark2\.dark_matter_rates",
)


ALL_QCDARK2_CASES = [
    ('Si', 'composite'),
    ('Si', 'lfe'),
    ('Si', 'nolfe'),
    ('Ge', 'composite'),
    ('Ge', 'lfe'),
    ('Ge', 'nolfe'),
    ('GaAs', 'composite'),
    ('GaAs', 'lfe'),
    ('GaAs', 'nolfe'),
    ('SiC', 'composite'),
    ('SiC', 'lfe'),
    ('SiC', 'nolfe'),
    ('Diamond', 'composite'),
    ('Diamond', 'lfe'),
    ('Diamond', 'nolfe'),
]
MEDIATOR_CASES = [
    (0, 'heavy'),
    (2, 'light'),
]


def _make_object(material, variant):
    obj = DMeRate(material, form_factor_type='qcdark2', qcdark2_variant=variant, device='cpu')
    obj.update_params(
        dm.default_astro['v0'],
        dm.default_astro['vEarth'],
        dm.default_astro['vEscape'],
        dm.default_astro['rhoX'],
        dm.default_astro['sigma_e'],
    )
    return obj


def _assert_spectrum_close(material, energy_eV, reference, candidate):
    band_gap_eV = float(qcdark2_band_gaps[material] / nu.eV)
    peak = float(np.max(reference))
    floor = peak * 1e-8
    mask = (energy_eV >= band_gap_eV) & (reference >= floor)

    assert np.any(mask), f"No populated bins found for {material}"
    np.testing.assert_allclose(candidate[mask], reference[mask], rtol=5e-2, atol=0.0)

    total_ref = simpson(reference, x=energy_eV)
    total_candidate = simpson(candidate, x=energy_eV)
    rel_err = abs(total_candidate - total_ref) / total_ref
    assert rel_err < 2e-2, (
        f"Total-rate mismatch for {material}: ref={total_ref:.6e}, "
        f"got={total_candidate:.6e}, rel_err={rel_err:.3e}"
    )


def _fold_reference_to_ne(obj, reference_spectrum, ne_bins):
    probabilities = obj.probabilities[ne_bins - 1].detach().cpu().numpy()
    energy_eV = (obj.Earr / nu.eV).detach().cpu().numpy()
    return simpson(reference_spectrum[None, :] * probabilities, x=energy_eV, axis=1)


@pytest.mark.parametrize(('material', 'variant'), ALL_QCDARK2_CASES)
@pytest.mark.parametrize(('fdm_n', 'mediator'), MEDIATOR_CASES)
def test_qcdark2_reference_parity_across_materials(material, variant, fdm_n, mediator):
    if not os.path.exists(qcdark2_file(material, variant)):
        pytest.skip(f"Local qcdark2 file missing for {material}/{variant}")
    if not os.path.exists(upstream_qcdark2_file(material, variant)):
        pytest.skip(f"Upstream qcdark2 file missing for {material}/{variant}")

    epsilon = dm.load_epsilon(upstream_qcdark2_file(material, variant))
    reference, reference_energy = dm.get_dR_dE(
        epsilon,
        m_X=100e6,
        mediator=mediator,
        astro_model=dm.default_astro,
        screening='RPA',
    )

    obj = _make_object(material, variant)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        energy_eV, spectra = obj.calculate_spectrum([100.0], 'shm', fdm_n, DoScreen=False)
        rates = obj.calculate_rates([100.0], 'shm', fdm_n, [1, 2, 3, 4, 5], DoScreen=False)

    assert np.allclose(energy_eV, reference_energy)
    _assert_spectrum_close(material, energy_eV, reference, spectra[0])

    ne_bins = np.array([1, 2, 3, 4, 5])
    reference_ne = _fold_reference_to_ne(obj, reference, ne_bins)
    candidate_ne = rates[:, 0].detach().cpu().numpy() * (nu.kg * nu.year)
    populated = reference_ne >= (np.max(reference_ne) * 1e-8)
    np.testing.assert_allclose(candidate_ne[populated], reference_ne[populated], rtol=5e-2, atol=0.0)


@pytest.mark.parametrize('mass_mev', [0.5, 5.0, 100.0, 1000.0])
@pytest.mark.parametrize(('fdm_n', 'mediator'), MEDIATOR_CASES)
@requires_qcdark2('Si', 'composite')
@requires_upstream_qcdark2('Si', 'composite')
def test_qcdark2_reference_parity_si_composite_mass_scan(mass_mev, fdm_n, mediator):
    epsilon = dm.load_epsilon(upstream_qcdark2_file('Si', 'composite'))
    reference, reference_energy = dm.get_dR_dE(
        epsilon,
        m_X=mass_mev * 1e6,
        mediator=mediator,
        astro_model=dm.default_astro,
        screening='RPA',
    )

    obj = _make_object('Si', 'composite')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        energy_eV, spectra = obj.calculate_spectrum([mass_mev], 'shm', fdm_n, DoScreen=False)

    assert np.allclose(energy_eV, reference_energy)
    _assert_spectrum_close('Si', energy_eV, reference, spectra[0])
