# Phase 3.3 — Native QCDark2 Dielectric Engine: Implementation and Validation

**Date**: 2026-04-25
**Branch**: qcdark2_integration
**Files created/modified**:
- `DMeRates/engines/dielectric.py` (complete)
- `tests/test_qcdark2.py` (9 tests)
- `tests/conftest.py` (QCDARK2_REFS block added)

---

## Engine: `compute_dRdE(...)`

Public API in `DMeRates/engines/dielectric.py`. All arguments are keyword-only.

```python
result = compute_dRdE(
    material='Si',
    mX_eV=1e9,
    FDMn=0,              # 0=heavy, 2=light
    halo_model='imb',    # 'imb' only (MB via DM_Halo_Distributions)
    screening='rpa',     # 'rpa' or 'none'; None raises ValueError
    variant='composite', # 'composite', 'lfe', 'nolfe'
    sigma_e_cm2=1e-39,
    rhoX_eV_per_cm3=0.3e9,
)
# result.E_eV                        — numpy array (N_E,)
# result.dRdE_per_kg_per_year_per_eV — numpy array (N_E,)
# result.spectrum                    — RateSpectrum (nu-typed)
```

### Internal structure

| Helper | Purpose |
|--------|---------|
| `_bare_floats_from_loader(d)` | Strips nu units from `dielectric_response` fields |
| `_qcdark2_constants_bare()` | Derives `kg`, `alpha`, `m_e`, `cm2sec`, `sec2yr` from `nu.*` (matches QCDark2 hard-coded values to <1e-7) |
| `_eta_imb_bare(vmin_over_c, halo)` | Lifts bare v/c → nu velocity, calls `eta_MB_tensor`, strips nu |
| `_resolve_eta(...)` | Halo-model dispatch (currently 'imb' only) |

### Formula executed (mirrors derivation notebook section 5)

```
rho_T    = M_cell_eV / kg_QCD / V_cell_bohr        [kg/Bohr³]
mu       = mX_eV * me_eV / (mX_eV + me_eV)         [eV]
prefactor = (1/rho_T) * (rhoX/mX) * (sigma_e/mu²) / (4π)

S         = ELF * q_ame² / (2π·α)
F2        = mediator_factor(q_ame, FDMn)
vmin      = q_eV/(2·mX) + E/q_eV                   [v/c]
η         = eta_MB_tensor(vmin)                      [c⁻¹]

integrand = q_ame * F2 * S * η * screen_ratio       [(N_q, N_E)]
dR/dE     = prefactor × simpson(integrand, q_eV) / cm2sec / sec2yr
```

RPA screening: `screen_ratio = 1.0` (ε/ε_screen = 1 by definition).
No screening: `screen_ratio = |ε|²` (ε_screen = 1).

---

## Validation Results (Si / heavy / RPA / MB / mX = 1 GeV)

| E (eV) | Native (events/kg/yr/eV) | QCDark2 ref | Rel. diff |
|-------:|-------------------------:|------------:|----------:|
| 5      | 1.845597e+00             | 1.84559612e+00 | **0.0000%** |
| 10     | 1.009348e+00             | 1.00934857e+00 | **0.0000%** |
| 50     | 5.291541e-02             | 5.29154460e-02 | **0.0001%** |

Gate passed: <5% required. Achieved machine precision.

---

## RateSpectrum round-trip

`result.spectrum.dR_dE * (nu.kg * nu.year * nu.eV)` reproduces the bare-float array
to `rtol=1e-12`. Verified by `test_qcdark2_returns_rate_spectrum`.

---

## Test Suite (tests/test_qcdark2.py)

| Test | HDF5 required | Checks |
|------|:---:|--------|
| `test_qcdark2_si_heavy_rpa_matches_reference` | ✓ | dR/dE within 5% at E={5,10,50} eV |
| `test_qcdark2_screening_required` | ✓ | `screening=None` raises `ValueError` |
| `test_qcdark2_invalid_screening_raises` | ✓ | Unknown key raises `ValueError` |
| `test_qcdark2_returns_rate_spectrum` | ✓ | `RateSpectrum` instance; nu round-trip |
| `test_mediator_factor_heavy_is_ones` | — | FDMn=0 → ones |
| `test_mediator_factor_light_inverse_q4` | — | FDMn=2 → q⁻⁴ |
| `test_mediator_factor_invalid_FDMn_raises` | — | FDMn=1 → ValueError |
| `test_v_min_bare_shape_and_value` | — | shape (N_q, N_E), spot-check value |
| `test_energy_loss_function_values` | — | eps=2+1j → ELF=0.2 |

HDF5-dependent tests skip cleanly when `Si_comp.h5` is absent.
Full suite: **23 passed, 17 warnings** (pre-existing torchquad/numpy deprecation warnings).
No regressions in QEDark, QCDark1, or noble-gas baselines.

---

## Supported / Not Yet Supported

**Supported**:
- Materials: Si (validated), Ge/GaAs/SiC/diamond (smoke-tested)
- Variants: `composite`, `lfe`, `nolfe`
- Mediators: `FDMn=0` (validated), `FDMn=2` (runs, no reference comparison)
- Screening: `rpa` (validated), `none` (runs, unvalidated)
- Halo: `'imb'` (Maxwell-Boltzmann via `DM_Halo_Distributions`)

**Not yet wired (Steps 3.4–3.6)**:
- Halo providers: `'shm'`, `'tsa'`, `'dpl'`, `'modulated'`, `'summer'`
- `DMeRate(form_factor_type='qcdark2')` API
- `calculate_rates`-style ne-binned output loop

---

## Hard Invariants Preserved

- No `qcdark2.*` import in `DMeRates/engines/dielectric.py`
- Calling `compute_dRdE(..., screening=None)` raises `ValueError` with a clear message
- `DMeRates/DMeRate.py` and all existing tests are unmodified
