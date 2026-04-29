# Phase 3.2 — Pure Utility Functions for Dielectric Engine

**Date**: 2026-04-25
**Branch**: qcdark2_integration
**File created**: `DMeRates/engines/dielectric.py` (utility section, lines 1–108)

---

## What Was Built

Four pure utility functions implementing the core physics factors of the QCDark2 formula.
All are bare-float (no `numericalunits` inside); they mirror notebook section 5 exactly.

### `energy_loss_function(epsilon)`
- Input: complex `(N_q, N_E)` ndarray
- Output: real `(N_q, N_E)` — `Im(ε) / (Im²(ε) + Re²(ε))`
- Test: `eps = 2 + 1j → ELF = 0.2` (verified by `test_energy_loss_function_values`)

### `dynamic_structure_factor(epsilon, q_ame)`
- Input: epsilon `(N_q, N_E)`, q in α·me units `(N_q,)`
- Output: `ELF × q_ame² / (2π·α)` — shape `(N_q, N_E)`
- α defaults to `nu.alphaFS` (a pure number)

### `mediator_factor(q_ame, FDMn)`
- `FDMn=0` (heavy): returns `ones(N_q)` — F_DM = 1
- `FDMn=2` (light): returns `q_ame**-4` — F_DM = (α·me/q)²
- Invalid FDMn raises `ValueError`
- Tests: `test_mediator_factor_heavy_is_ones`, `test_mediator_factor_light_inverse_q4`, `test_mediator_factor_invalid_FDMn_raises`

### `v_min_bare(q_eV, E_eV, mX_eV)`
- All inputs in eV; output is dimensionless v/c, shape `(N_q, N_E)`
- Formula: `q / (2·mX) + E / q` (QCDark2 non-relativistic)
- Test: `test_v_min_bare_shape_and_value`

---

## Key Design Decision

All utilities operate in QCDark2's bare-float convention (eV for energies, α·me for q,
v/c for velocities). `numericalunits` enters only at two boundaries: deriving bare
QCDark2 constants from `nu.*`, and calling `DM_Halo_Distributions.eta_MB_tensor`.
This guarantees formula parity with the derivation notebook by construction.

---

## Tests Added

4 unit tests in `tests/test_qcdark2.py` (lines 115–153) — no HDF5 data required.
All pass in the fresh pytest run.
