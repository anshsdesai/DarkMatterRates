import random
random.seed(0)

from DMeRates.config import PhysicsConfig
import numericalunits as nu


def test_physicsconfig_roundtrip():
    cfg = PhysicsConfig(v0_km_s=230.0, vEarth_km_s=240.0)
    cfg2 = PhysicsConfig.from_dict(cfg.to_dict())
    assert abs(cfg.v0 - cfg2.v0) / cfg.v0 < 1e-12
    assert abs(cfg.vEarth - cfg2.vEarth) / cfg.vEarth < 1e-12


def test_physicsconfig_defaults():
    cfg = PhysicsConfig.defaults()
    assert cfg.v0_km_s == 238.0
    assert cfg.vEarth_km_s == 250.2
    assert cfg.vEscape_km_s == 544.0
    assert cfg.rhoX_GeV_cm3 == 0.3


def test_physicsconfig_numericalunits():
    cfg = PhysicsConfig(v0_km_s=238.0)
    assert abs(cfg.v0 / (nu.km / nu.s) - 238.0) < 1e-10
    assert abs(cfg.sigma_e / nu.cm**2 - 1e-36) < 1e-48
