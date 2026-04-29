import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import EarthLocation


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULATION_DIR = REPO_ROOT / "modulation_study"
if str(MODULATION_DIR) not in sys.path:
    sys.path.insert(0, str(MODULATION_DIR))

from isoangle import (  # noqa: E402
    FracDays,
    ThetaIso,
    get_site_location,
    get_site_thetaiso_loc,
    normalize_site_key,
)
from Modulation import get_angle_limits  # noqa: E402


def test_thetaiso_accepts_earthlocation_and_normalizes_to_tuple():
    n = FracDays(np.array([8, 8, 2024]), np.array([12, 0, 0]))

    tuple_angle = ThetaIso(get_site_thetaiso_loc("SNO"), n)
    earth_location_angle = ThetaIso(get_site_location("SNO"), n)

    assert earth_location_angle == pytest.approx(tuple_angle)


def test_thetaiso_accepts_site_aliases():
    n = FracDays(np.array([8, 8, 2024]), np.array([12, 0, 0]))

    assert normalize_site_key("SNOLAB") == "SNO"
    assert ThetaIso("SNOLAB", n) == pytest.approx(ThetaIso("SNO", n))


def test_get_angle_limits_normalizes_snolab_alias():
    snolab_limits = get_angle_limits("SNOLAB", date=[8, 8, 2024])
    sno_limits = get_angle_limits("SNO", date=[8, 8, 2024])

    assert snolab_limits == pytest.approx(sno_limits)
    assert 0.0 <= snolab_limits[0] <= snolab_limits[1] <= 180.0


def test_get_angle_limits_accepts_earthlocation():
    location = get_site_location("SNOLAB")
    min_angle, max_angle = get_angle_limits(location, date=[8, 8, 2024])

    assert 0.0 <= min_angle <= max_angle <= 180.0


def test_site_lookup_does_not_use_network(monkeypatch):
    def fail_if_network_lookup_is_used(*args, **kwargs):
        raise AssertionError("EarthLocation.of_address should not be used")

    monkeypatch.setattr(EarthLocation, "of_address", fail_if_network_lookup_is_used)

    location = get_site_location("SNOLAB")
    min_angle, max_angle = get_angle_limits("SNOLAB", date=[8, 8, 2024])

    assert isinstance(location, EarthLocation)
    assert 0.0 <= min_angle <= max_angle <= 180.0
