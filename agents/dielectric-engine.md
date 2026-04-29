Role: dielectric-engine
Recommended model: GPT-5.5 high/xhigh, Opus 4.7, or GPT-5.3-Codex high with physics review
Owns: Plan Steps 3.2 and 3.3
Prerequisites: Step 3.0 derivation notebook complete and validated; Step 3.1 loader complete

---

# dielectric-engine — Native QCDark2 Dielectric Engine

## Goal

Implement the native QCDark2 dielectric-response calculation in DMeRates without
delegating to QCDark2 Python. The output is a `RateSpectrum` with `dR/dE` computed
from QCDark2 HDF5 dielectric data, DMeRates halo providers, explicit screening,
and torch-compatible vectorized numerics.

This is the highest-risk implementation task. Do not reinterpret the formula.
Follow `tests/qcdark2_formula_derivation.ipynb`.

---

## Hard Prerequisites

Before writing code:

- [ ] `tests/qcdark2_formula_derivation.ipynb` exists.
- [ ] The notebook runs top-to-bottom.
- [ ] The notebook shows <5% agreement with QCDark2 reference for Si/MB/RPA.
- [ ] `DMeRates/responses/dielectric.py` exists and documents q, `M_cell`,
      `V_cell`, `E`, and `dE` conventions.
- [ ] `units-numerics-reviewer` has reviewed the notebook and loader, or the
      main conversation has explicitly accepted the unit conventions.

Stop if any prerequisite is missing.

---

## Files to Create or Edit

Expected files:

```
DMeRates/engines/dielectric.py
DMeRates/screening/dielectric.py
DMeRates/responses/dielectric.py       # only if loader fixes are needed
DMeRates/rate_calculator.py            # only for API wiring after engine validation
tests/test_qcdark2.py
```

Do not edit legacy QEDark/QCDark1 behavior unless a regression test proves the
old value was numerically wrong.

---

## Fixed Validation Case

Use exactly this case for all development validation — do not choose a different one.
It matches the derivation notebook and the QCDark2 reference run:

| Parameter | Value |
|-----------|-------|
| Material | Si (`Si_comp.h5`, composite dielectric) |
| DM mass | mX = 1 GeV = 1×10⁹ eV |
| Mediator | Heavy (`FDMn=0`, FDM=1) |
| Halo | Maxwell-Boltzmann (`'imb'` or analytic MB) |
| Screening | RPA (`screening='rpa'`) |
| Astro | v0=238 km/s, vEarth=250.2 km/s, vEsc=544 km/s, rhoX=0.3 GeV/cm³ |
| Validate at | E ∈ {1, 5, 10, 20, 50} eV (skip bins where dR/dE ≈ 0) |

The target numbers for each energy bin are in `tests/qcdark2_formula_derivation.ipynb`.
Do not generalize to other materials or halo models until this case passes.

---

## Implementation Contract

The engine must compute natively:

- `v_min(E, q, mX)`
- `eta(v_min)` using DMeRates halo providers
- mediator factors for `FDMn=0` and `FDMn=2`
- energy-loss function from complex `epsilon`
- dynamic structure factor `S(q, E)`
- explicit dielectric screening
- q-integration in torch or torchquad
- final `dR/dE` as a `RateSpectrum`

Production code must not import:

```python
qcdark2.dark_matter_rates
```

QCDark2 Python is allowed only in validation notebooks/tests that explicitly
compare the native implementation to the reference.

---

## Development Sequence

1. Implement pure utility functions first:
   - `energy_loss_function(epsilon)`
   - `dynamic_structure_factor(...)`
   - `mediator_factor(q, FDMn)`
   - `v_min(E, q, mX)`

2. Add small shape/unit tests for each utility using the notebook reference.

3. Implement the engine for one fixed case only:
   - Si
   - composite dielectric
   - heavy mediator
   - MB halo
   - `screening='rpa'`

4. Compare against notebook reference values before generalizing.

5. Add support for:
   - `FDMn=2`
   - `screening='none'`
   - Ge
   - DMeRates halo providers
   - QCDark2 variants: `composite`, `lfe`, `nolfe`

6. Only after validation, wire into `DMeRate(..., form_factor_type='qcdark2')`
   and `RateCalculator`.

---

## Acceptance Criteria

- [ ] Native Si/MB/RPA dR/dE matches QCDark2 reference within the notebook tolerance.
- [ ] Production code uses QCDark2 HDF5 data only.
- [ ] Calling QCDark2 calculations without explicit screening raises a clear error.
- [ ] The engine returns a `RateSpectrum`.
- [ ] QEDark/QCDark1/noble-gas regression tests still pass.
- [ ] QCDark2 validation tests skip cleanly when external HDF5 data is unavailable.
- [ ] The units-numerics reviewer has no blockers.

---

## Handoff

When complete, report:

1. Files changed.
2. Validation case and relative differences versus QCDark2 reference.
3. Any remaining materials, variants, or halo providers not yet supported.
4. Commands run and whether they passed.

