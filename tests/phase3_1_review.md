=== Units/Numerics Review — Pass 1 ===
Artifact reviewed: tests/qcdark2_formula_derivation.ipynb + DMeRates/responses/dielectric.py
Date: 2026-04-24
Branch: qcdark2_integration

---

## Blockers

**FIXED** — `DMeRates/responses/dielectric.py` (was lines 20-21):
Module-level capture of `_Q_AMU = nu.alphaFS * nu.me * nu.c0` and
`_BOHR = nu.hbar / (nu.alphaFS * nu.me * nu.c0)` evaluated at import time
before `DMeRates.Constants` sets the `nu.*` scales. Importing the loader before
Constants produced silently wrong `d.q` (SI values off by ~10⁶) and wrong
`d.V_cell` (off by ~10⁶). Reachable from the validator pytest run since neither
`DMeRates/__init__.py` nor `tests/conftest.py` imports Constants before the loader.

**Fix applied**: moved both expressions inside `__init__` as local variables
`q_amu` and `bohr`, so they read the live `nu.*` scales at instantiation time.
Verified: `d.q.max() / q_amu = 25.01`, `d.V_cell / bohr³ = 270.107` in
previously broken import order.

---

## Warnings (non-blocking)

1. `DMeRates/responses/dielectric.py` (M_cell field):
   `self.M_cell` is stored in energy units (`float * nu.eV`), not mass.
   QCDark2 uses it as eV in the ratio `rho_T = M_cell / kg / V_cell`.
   The engine must divide by `nu.c0**2` if it needs a mass.
   A one-line docstring on the field would prevent a subtle engine bug.

2. `tests/qcdark2_formula_derivation.ipynb` (sections 4–5):
   Only the heavy mediator path (F_DM=1) is validated against the QCDark2
   reference. The light mediator branch (F_DM ∝ 1/q²) has no reproduction
   check. Deferred to Pass 2 — the dielectric-engine agent should add a
   reference comparison for `mediator='light'` before Pass 2 review.

3. `DMeRates/responses/dielectric.py` (material coverage):
   Only Si composite was load-tested. Attribute presence and q-range for
   Ge, GaAs, SiC, diamond are unverified. Recommend a smoke loop in engine
   validation asserting q-range and attribute keys for every supported material.

---

## Looks Good

- **0.000% agreement** at E={5, 10, 20, 50} eV vs QCDark2 `get_dR_dE()` (machine
  precision; far exceeds the 5% gate). Notebook re-executed cleanly via nbconvert.
- Section 5 of the notebook derives every QCDark2 hard-coded constant (`kg`, `alphaFS`,
  `m_e`, `c_kms`) from `nu.*` and asserts agreement to <1e-7 — strong evidence the
  unit system is internally consistent.
- E=1 eV correctly excluded from validation as a threshold-edge bin (~1e-19, ill-conditioned).
- `random.seed(0)` is the first executable line before any DMeRates import — matches
  conftest.py convention.
- Convention table in notebook section 3 is explicit on every field: q in α·me,
  E/dE in eV, M_cell in eV (cell rest energy), V_cell in Bohr³, ρ_χ as energy
  density (eV/cm³), σ_e in cm², velocities as v/c.
- `bohr = nu.hbar / (nu.alphaFS * nu.me * nu.c0)` derivation is correct throughout.
- `V_cell = 270.107 Bohr³` (2-atom Si primitive cell) is documented with an explicit
  note that the runbook's "~130 Bohr³" was the per-atom value — caught and reconciled.
- Diamond filename handling uses lowercase `diamond_comp.h5` — correct in both loader
  and DataRegistry.
- No `nu.reset_units()` calls anywhere in `DMeRates/` or the notebook after Constants import.
- `DMeRates/engines/dielectric.py` does not exist yet — gate respected.
- Notebook produces a `QCDARK2_REFS` block for `tests/conftest.py` at E={5, 10, 50} eV.

---

## Recommendation

**Proceed.** The blocker (load-order fragility in the dielectric loader) was fixed
immediately after review. All formula conventions, unit expressions, and integration
numerics are sound. The `dielectric-engine` agent may begin Steps 3.2–3.3.

Pass 2 review is required after Step 3.3 (native engine implemented and validated)
before Step 3.6 API wiring begins.
