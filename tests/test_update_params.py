"""Regression tests for DMeRate.update_params unit conventions.

update_params previously stored rhoX as an ENERGY density (rhoX * nu.eV/cm^3)
while the constructor and all downstream code (e.g. the QCDark2 path, which
converts with `self.rhoX * nu.c0**2`) expect a MASS density matching
Constants.rhoX = 0.3 GeV/c^2/cm^3. It also dropped the cross section when
rebuilding DM_Halo_Distributions.
"""
import sys
sys.path.insert(0, ".")

import pytest

from DMeRates import Constants
from DMeRates.DMeRate import DMeRate
from DMeRates.DM_Halo import DM_Halo_Distributions


@pytest.fixture(scope="module")
def calculator():
    return DMeRate("Si")


def test_update_params_rhoX_matches_constructor_convention(fix_units, calculator):
    """Passing the Constants defaults back in must reproduce Constants.rhoX."""
    calculator.update_params(238.0, 250.2, 544.0, 0.3e9, 1e-36)
    assert calculator.rhoX == pytest.approx(Constants.rhoX, rel=1e-12)
    assert calculator.cross_section == pytest.approx(Constants.crosssection, rel=1e-12)


def test_update_params_propagates_cross_section_to_halo(fix_units, calculator):
    calculator.update_params(238.0, 250.2, 544.0, 0.3e9, 5e-37)
    import numericalunits as nu
    assert calculator.DM_Halo.cross_section == pytest.approx(
        5e-37 * nu.cm**2, rel=1e-12
    )
    assert calculator.DM_Halo.rhoX == pytest.approx(calculator.rhoX, rel=1e-12)


def test_dm_halo_default_cross_section_is_constants_value(fix_units):
    """The crosssection=None branch must fall back to the Constants default,
    not to None (the parameter shadows the star-imported global)."""
    halo = DM_Halo_Distributions()
    assert halo.cross_section is not None
    assert halo.cross_section == pytest.approx(Constants.crosssection, rel=1e-12)
