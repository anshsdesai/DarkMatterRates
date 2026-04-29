import numericalunits as nu
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhysicsConfig:
    """User-facing physics parameters in conventional units.

    All values stored in numericalunits SI internally. Construct with
    explicit keyword arguments; use PhysicsConfig.defaults() for the
    standard SHM parameters matching DMeRates/Constants.py.
    """
    v0_km_s: float = 238.0
    vEarth_km_s: float = 250.2
    vEscape_km_s: float = 544.0
    rhoX_GeV_cm3: float = 0.3
    sigma_e_cm2: float = 1e-36  # DMeRates legacy default

    # Derived numericalunits values (set in __post_init__)
    v0: float = field(init=False)
    vEarth: float = field(init=False)
    vEscape: float = field(init=False)
    rhoX: float = field(init=False)
    sigma_e: float = field(init=False)

    def __post_init__(self):
        self.v0      = self.v0_km_s      * nu.km / nu.s
        self.vEarth  = self.vEarth_km_s  * nu.km / nu.s
        self.vEscape = self.vEscape_km_s * nu.km / nu.s
        self.rhoX    = self.rhoX_GeV_cm3 * 1e9 * nu.eV / nu.c0**2 / nu.cm**3
        self.sigma_e = self.sigma_e_cm2  * nu.cm**2

    def to_dict(self) -> dict:
        return {
            'v0_km_s':      self.v0_km_s,
            'vEarth_km_s':  self.vEarth_km_s,
            'vEscape_km_s': self.vEscape_km_s,
            'rhoX_GeV_cm3': self.rhoX_GeV_cm3,
            'sigma_e_cm2':  self.sigma_e_cm2,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'PhysicsConfig':
        return cls(**d)

    @classmethod
    def defaults(cls) -> 'PhysicsConfig':
        return cls()
