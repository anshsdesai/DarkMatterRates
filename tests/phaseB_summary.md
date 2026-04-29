# Phase B Summary — SRDM Physics Derivation

The source-of-truth SRDM derivation notebook is complete at `tests/qcdark2_srdm_derivation.ipynb` and executes top-to-bottom with `jupyter nbconvert --execute`.
For the fixed Si benchmark at the bundled flux-grid point (`m_X = 48232.9466 eV`, `sigma_e = 1.098541e-38 cm^2`, `FDMn = 2`, vector mediator), the DMeRates dielectric reimplementation uses the paper propagator convention and matches QCDark2 `get_rate_flux` with max relative deviation `1.005928e-03` over 237 energy bins above 5% of the peak.
The single-velocity `dsigma_rel2` term-by-term reproduction matches with max relative deviation `7.932381e-16`, comfortably below the `1e-3` target.
QCDark1 unscreened form-factor SRDM is finite and positive with peak ratio `0.987` versus QCDark2; QEDark unscreened is finite and positive with peak ratio `3.387` versus QCDark2.
The QCDark2/DMeRates dielectric reference tuples are `(8.10, 8.504347e-07)`, `(14.90, 2.270760e-06)`, and `(21.70, 1.719587e-06)` in events/kg/year/eV.
The QCDark1 reference tuples are `(5.25, 9.568909e-07)`, `(6.95, 2.848944e-06)`, and `(9.05, 2.093103e-06)`.
The QEDark reference tuples are `(5.10, 2.318300e-06)`, `(6.00, 7.964201e-06)`, and `(7.00, 2.047581e-06)`.
The main unit subtlety is that QCDark2 `get_rate_flux` integrates raw file flux in `cm^-2 s^-1 (km/s)^-1` against dimensionless `v/c`, while DMeRates `load_srdm_flux` correctly converts to `dPhi/d(v/c)` by multiplying by the bare `c_kms`; the loader now applies this conversion exactly once and drops the singular `v=0` row.
The form-factor subtlety is that the correct bridge is `S(omega,q) = 8*pi^2*(alpha*m_e)^2*f^2/(V_cell*q^3)`, followed by `ELF_equiv = 2*pi*alpha*S/q^2` before reusing the dielectric SRDM kernel.
The halo-limit check uses a narrow Gaussian at `v0 = 238 km/s` and finds max full-vs-NR deviation `6.522012e-09` (`0.0065 ppm`) with the paper propagator held fixed to isolate the relativistic kinematics.
On Pass 1, the units-numerics reviewer should inspect the explicit split between the reference-only QCDark2 code-propagator diagnostic and the DMeRates paper-propagator source of truth, plus the QEDark raw-table normalization choice in the direct `f^2 -> S` conversion.
