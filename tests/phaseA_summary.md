# Phase A Summary — SRDM Infrastructure

**Branch:** qcdark2_integration  
**Date:** 2026-04-25  
**Status:** COMPLETE — 42/42 tests pass, 0 regressions

---

## What Was Built

Phase A adds the data scaffolding the SRDM rate engines need. No rate
calculation is included; this phase only provides flux loading, manifest
lookup, and relativistic kinematics primitives.

### Files Created

| File | Purpose |
|------|---------|
| `DMeRates/srdm/__init__.py` | Package init |
| `DMeRates/srdm/manifest.py` | `load_manifest()`, `find_entry()` with rtol tolerance |
| `DMeRates/srdm/flux_loader.py` | `load_srdm_flux()` — sole km/s → v/c conversion point |
| `DMeRates/srdm/kinematics.py` | γ, v_min_rel, q_bounds, q_mask, H_V, H_φ, propagator |
| `halo_data/srdm/manifest.json` | Registry of flux files with actual + nominal keys |
| `halo_data/srdm/srdm_dphidv_DPLM_row10_col8.txt` | 50 keV flux file (299 rows) |
| `halo_data/srdm/srdm_dphidv_DPLM_row15_col20.txt` | 0.5 MeV flux file (299 rows) |
| `tests/test_srdm_infrastructure.py` | 13 tests |

### Files Modified

| File | Change |
|------|--------|
| `DMeRates/data/registry.py` | Added `srdm_manifest()` and `srdm_flux_file()` classmethods |

---

## Flux File Provenance

Source: arXiv:2404.10066 (Emken-Essig-Xu 2024),
cloned from https://github.com/hlxuhep/Solar-Reflected-Dark-Matter-Flux.

Family: **DPLM** = dark photon light mediator (FDMn=2, vector) — the correct
family for both requested benchmark points.

**Important:** The upstream repo uses a log-spaced grid, so exact round
benchmark values do not land on grid points. The nearest grid points were used:

| Nominal (requested) | Actual grid point (manifest key) | Upstream filename |
|---------------------|----------------------------------|-------------------|
| 50 keV, 1e-38 cm² | mX = 48232.9 eV, σ = 1.0985e-38 cm² | `Differential_SRDM_Flux_DPLM_10_8.txt` |
| 0.5 MeV, 1e-37 cm² | mX = 510927.7 eV, σ = 1.1515e-37 cm² | `Differential_SRDM_Flux_DPLM_15_20.txt` |

The manifest stores both `mX_eV` / `sigma_e_cm2` (actual, used as lookup keys)
and `nominal_mX_eV` / `nominal_sigma_e_cm2` (for traceability). Calling
`load_srdm_flux` with the nominal round values intentionally raises
`FileNotFoundError` — this is the correct behavior, documented in tests.

**File format:** two columns, no header  
- col 0: velocity (km/s), range 0–49427 km/s (v/c up to ~0.165 for 50 keV file)
- col 1: dΦ/dv (cm⁻²·s⁻¹·(km/s)⁻¹)

---

## Unit Convention

### `flux_loader.py` (the nu boundary)

`load_srdm_flux` performs the **only** km/s → v/c conversion in the codebase:

```python
c_kms_bare = nu.c0 / (nu.km / nu.s)          # speed of light in km/s as bare float
v_over_c   = v_kms / c_kms_bare               # dimensionless

# dΦ/d(v/c) = dΦ/dv_kms × c_kms
dphi_dv_nu = dphi_raw * nu.c0 / (nu.cm**2 * nu.s)   # in nu units
```

The returned `dphi_dv` tensor carries `numericalunits`-scaled values. Callers
must be in the same `nu` session (i.e., seed `numericalunits` before import).

### `kinematics.py` (bare-float, no nu)

All kinematics functions work in bare eV and dimensionless v/c — **no
`numericalunits` imports**. This keeps the hot path free of nu overhead. The
nu boundary is at the engine, not inside kinematics.

---

## Kinematics Implementation

All functions are torch-native, broadcast-friendly, dtype/device-aware.

| Function | Paper eq. | Notes |
|----------|-----------|-------|
| `gamma(v)` | — | 1/√(1−v²) |
| `v_min_relativistic(q, ω, mX, γ)` | QCDark2 eq. 2.9 | q/(2γmX) + ω/q; reduces to halo v_min at γ→1 |
| `q_bounds(v, ω, mX)` | QCDark2 eq. A.19 | returns (q_min, q_max) shape (N_v, N_E) |
| `q_mask(q, q_min, q_max)` | — | bool (N_v, N_q, N_E); empty where discriminant < 0 |
| `H_vector(q, E_χ, E_χ')` | QCDark2 eq. 2.6 | (E_χ+E_χ')²−q² |
| `H_scalar(q, E_χ, E_χ', mX)` | QCDark2 eq. 2.6 | 4mX²−(E_χ−E_χ')²+q² |
| `mediator_propagator_inv_sq(q, ω, mA)` | QCDark2 eq. A.20 denom | 1/(ω²−q²−mA²)² |
| `reference_propagator_factor(mA, α, me)` | QCDark2 eq. A.20 | (mA²+(αme)²)² |

Degenerate case in `q_bounds`: when (γmX−ω)² < mX², the discriminant is
negative. The implementation sets `q_max = -1, q_min = 0`, making `q_mask`
return all-False (empty interval). No NaN is produced. Confirmed by test
`TestQBounds::test_q_bounds_degenerate_no_nan`.

---

## Test Results

```
tests/test_srdm_infrastructure.py — 13/13 PASSED
Full tests/ suite                 — 42/42 PASSED (0 regressions)
```

Tests cover: γ limits, v_min halo recovery, q_bounds free-particle limit,
q_bounds degenerate case (no NaN), H_V NR limit, flux smoke load, flux miss
error (includes manifest path in message), manifest rtol lookup.

---

## Flags for units-numerics-reviewer (Pass 1)

1. **v=0 row in flux files.** Both files have a first row with `v_kms = 0`
   and a finite `dΦ/dv` value (~173 and ~42 cm⁻²s⁻¹(km/s)⁻¹ respectively).
   Any engine expression with `1/v` or `ω/q` evaluated at v=0 will divide by
   zero. Phase B engines must clamp or skip the v=0 entry.

2. **v/c range is not small.** The 50 keV file reaches v/c ≈ 0.165
   (49,427 km/s). Relativistic γ corrections are genuinely non-trivial at
   this mass; the NR approximation is not valid. Phase B cannot treat γ ≈ 1.

3. **dphi_dv carries nu-scaled values.** The tensor value changes with the
   nu random seed. Phase B engines must compute the full rate integral within
   a single seeded nu session and must not cache or serialize the raw tensor.

4. **Nominal keys ≠ actual keys.** The QCDark2 reference (`get_rate_flux`)
   and the physics derivation notebook will need to call `load_srdm_flux`
   with the actual grid values (48232.9 eV, 1.0985e-38 cm²), not the
   nominal round values. The `srdm-physics-derivation` agent should read the
   manifest first and use the actual keys.

---

## Next Step

Per `agents/srdm_run_order.md`, Phase B begins with the **`srdm-physics-derivation`**
agent creating `tests/qcdark2_srdm_derivation.ipynb`. Recommended model: Opus 4.7.
The notebook is the gate for all SRDM engine code.
