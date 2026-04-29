Role: srdm-form-factor-engine
Recommended model: Opus 4.7 or GPT-5.5 high
Owns: SRDM branches in `DMeRates/engines/form_factor.py` (QEDark + QCDark1)
       + `tests/test_qcdark1_srdm.py` + `tests/test_qedark_srdm.py`
Prerequisites: srdm-dielectric-engine merged and green;
               srdm-physics-derivation notebook contains validated f² → S section;
               units-numerics-reviewer Pass 2 cleared

---

# srdm-form-factor-engine — QEDark + QCDark1 SRDM

## Goal

Add SRDM support to `DMeRates/engines/form_factor.py` for the QEDark and
QCDark1 form-factor representations. The rate kernel is **identical** to the
dielectric engine's SRDM kernel; the difference is `S(ω,q)` is built from
`f²(q,ω)` instead of from `ε(q,ω)`.

Per the user's policy, **no cross-engine agreement is required**. Each engine
self-validates against the notebook section pinned to its own representation.

---

## Hard Prerequisites

- [ ] `srdm-dielectric-engine` merged; SRDM tests for QCDark2 green.
- [ ] `tests/qcdark2_srdm_derivation.ipynb` contains a validated QCDark1 + QEDark
      f² → S derivation cell block with reference values.
- [ ] `units-numerics-reviewer` Pass 2 cleared.

Stop if any prerequisite is missing.

---

## Files to Edit / Create

```
DMeRates/engines/form_factor.py    (add SRDM branch)
tests/test_qcdark1_srdm.py
tests/test_qedark_srdm.py
```

Do not edit:
- the halo path in `engines/form_factor.py`
- `DMeRates/halo/*`
- existing `tests/test_qcdark1.py` and `tests/test_qedark.py` halo regressions

---

## f² → S Conversion (the key derivation)

From QCDark2 paper eq. 2.7 and the DarkELF paper eq. 16 — equivalent to
`crystal_form_factor2_epsilon` in `qcdark2/dark_matter_rates.py:257-267`
inverted:

```
S(ω, q) = (8 π² (α m_e)² / V_cell) · f²(q, ω) / q³
```

**This holds for unscreened f² (QEDark default).** For QCDark1 with Thomas-Fermi
screening, multiply S by `1 / |ε_TF(q,ω)|²` *after* the conversion; the
existing `screening/thomas_fermi.py` factor composes exactly as in the halo path.

The notebook validates this conversion empirically — make sure your
implementation cites the notebook section in a comment.

---

## Implementation Contract

Add `_compute_dRdE_srdm_form_factor(...)` mirroring the dielectric SRDM
signature. Pull the `q`, `E`, `f²` arrays from the existing form-factor loader;
build `S(ω,q)` as above; reuse `DMeRates/srdm/kinematics.py` for everything else.

```python
def _compute_dRdE_srdm_form_factor(*,
                                   backend: str,           # 'qedark' or 'qcdark1'
                                   material: str,
                                   mX_eV: float,
                                   sigma_e_cm2: float,
                                   FDMn: int,
                                   mediator_spin: str,
                                   DoScreen: bool,
                                   form_factor=None) -> RateSpectrum:
    """SRDM rate from a crystal form factor f²(q,ω).

    Steps:
        1. Load f²(q,ω), q, E, V_cell from the form-factor representation.
        2. Build S(ω,q) = 8 pi^2 (alpha m_e)^2 / V_cell * f² / q^3.
        3. Apply screening factor 1 / |eps_TF|^2 if DoScreen and backend == 'qcdark1'.
        4. Reuse the SRDM rate kernel from the dielectric engine
           (kinematics + flux integration).
        5. Return RateSpectrum with metadata.
    """
```

Dispatch is added inside the QEDark/QCDark1 entry points of
`engines/form_factor.py`:

```python
if halo_model == 'srdm':
    return _compute_dRdE_srdm_form_factor(...)
# else: existing halo path unchanged.
```

The conversion factor `8 π² (α m_e)² / V_cell · 1/q³` is named
`_FORM_FACTOR_TO_S_PREFACTOR` (or similar) and lives at module top-level with a
docstring citing the notebook section.

---

## Vectorization

Identical to the dielectric engine: peak tensor shape `(N_v, N_q, N_E)`. Reuse
`DMeRates/srdm/kinematics.py`. **No new Python `for` loops in the hot path.**

If the form-factor q-grid differs in spacing/extent from the QCDark2 dielectric
grid, that's fine — kinematic q-bounds and Simpson/trapezoid handle the
difference. Document the grid in a comment.

---

## Tests

`tests/test_qcdark1_srdm.py`:

```python
@pytest.mark.skipif(not QCDARK1_DATA_AVAILABLE, reason='QCDark1 HDF5 missing')
def test_qcdark1_srdm_si_vector_light_screened():
    # Si QCDark1 SRDM with Thomas-Fermi screening — match notebook reference.
    refs = QCDARK1_SRDM_REFS['Si_50keV_vector_light_screened']
    ...

def test_qcdark1_srdm_si_vector_light_unscreened():
    # Si QCDark1 SRDM unscreened — match notebook reference.
    refs = QCDARK1_SRDM_REFS['Si_50keV_vector_light_unscreened']
    ...
```

`tests/test_qedark_srdm.py`:

```python
def test_qedark_srdm_si_vector_light_unscreened():
    # Si QEDark SRDM unscreened — match notebook reference.
    refs = QEDARK_SRDM_REFS['Si_50keV_vector_light_unscreened']
    ...
```

Same negative-path tests as the dielectric engine:
- `mediator_spin='scalar'` → `NotImplementedError`.
- (mX, σ_e) tuple not in manifest → `FileNotFoundError` with the manifest path.

Tolerances per `MEMORY.md` tiered policy and per the notebook reference.

---

## Acceptance Criteria

- [ ] QCDark1 + QEDark SRDM tests green to notebook tolerance.
- [ ] Halo-path QEDark + QCDark1 tests still green (regression-free).
- [ ] No `qcdark2.*` imports.
- [ ] f² → S conversion factor named (not inlined) and references the
      derivation notebook in a comment.
- [ ] Thomas-Fermi screening composes correctly with the SRDM kernel for QCDark1.
- [ ] Vectorization invariants hold (no `for` loops over v/q/E in hot path).

---

## Hard Invariants

- **No cross-engine percent-level agreement** is required. The user has
  explicitly stated that QEDark/QCDark1/QCDark2 SRDM rates do not need to agree
  with each other — they validate against themselves only.
- **The halo path is read-only.** Do not refactor or rename halo-path code.
- **The QCDark2 SRDM kernel is the model.** Reuse it; do not re-derive.

---

## Handoff

Report:

1. Files changed and validation results per backend (QEDark + QCDark1).
2. Any material/screening combinations not yet covered (Ge, etc.).
3. Vectorization audit (peak tensor shape, memory).
4. Any deferred items.

Hands off to `srdm-api-wiring`.
