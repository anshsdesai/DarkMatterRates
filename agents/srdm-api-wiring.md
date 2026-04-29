Role: srdm-api-wiring
Recommended model: Sonnet 4.6 or GPT-5.3-Codex medium
Owns: Public-API plumbing for SRDM
Prerequisites: srdm-dielectric-engine + srdm-form-factor-engine merged and green

---

# srdm-api-wiring — Public API

## Goal

Wire the new `halo_model='srdm'` and `mediator_spin` keyword through the
existing public APIs without changing any halo behavior. Mechanical work; no
physics.

---

## Files to Edit

```
DMeRates/DMeRate.py
DMeRates/rate_calculator.py
DMeRates/spectrum.py     (only if metadata schema needs an additive field)
```

Do not edit:
- engine internals (`DMeRates/engines/*`)
- the halo subsystem (`DMeRates/halo/*`)
- the SRDM subsystem (`DMeRates/srdm/*`)

---

## Required Changes

### `DMeRates/DMeRate.py`

In `calculate_rates(...)`:

1. Add `mediator_spin: str = 'vector'` kwarg.
2. If `halo_model == 'srdm'`:
   - **Skip** `setup_halo_data(...)` (no eta-file lookup is needed for SRDM).
   - Validate `mediator_spin`. The first cut supports `'vector'` only; raise
     `NotImplementedError(...)` for `'scalar'`, `'approx'`, `'approx_full'`
     listing them as planned future modes.
   - Forward `(mX, sigma_e, FDMn, mediator_spin, screening)` to the engine.
3. Else: unchanged.

```python
def calculate_rates(self, mX_array, halo_model='shm', FDMn=0, ne=1,
                    mediator_spin='vector', sigma_e=None, ...):
    if halo_model == 'srdm':
        if mediator_spin not in {'vector'}:
            raise NotImplementedError(
                f"mediator_spin={mediator_spin!r} not yet supported. "
                "Planned future modes: 'scalar', 'approx', 'approx_full'."
            )
        # Skip setup_halo_data; engine handles flux + kinematics directly.
        ...
        return self._engine_dispatch_srdm(
            mX_array=mX_array, FDMn=FDMn, mediator_spin=mediator_spin,
            sigma_e=sigma_e, ne=ne, ...
        )
    else:
        # Existing halo path unchanged.
        self.setup_halo_data(...)
        ...
```

### `DMeRates/rate_calculator.py`

`RateCalculator.calculate(...)` forwards `mediator_spin` and `halo_model='srdm'`
to the underlying engine. No new logic — just kwargs passthrough.

### `DMeRates/spectrum.py`

No structural change to `RateSpectrum`. The engines populate `metadata`
directly. Update the `RateSpectrum` docstring to document the new metadata keys
(`halo_model`, `mediator_spin`, `flux_file`, `sigma_e_cm2`, `variant`).

---

## Smoke Test (must pass)

From a notebook:

```python
from DMeRates import DMeRate

d = DMeRate('Si', form_factor_type='qcdark2')
res = d.calculate_rates(
    mX_array=[5e4],            # eV
    halo_model='srdm',
    FDMn=2,
    mediator_spin='vector',
    sigma_e=1e-38,
    integrate=True,
    ne=1,
)
print(res)
```

Expected:
- A finite, positive dR/dE matching the notebook reference within tolerance.
- `RateSpectrum.metadata['halo_model'] == 'srdm'`.
- `RateSpectrum.metadata['mediator_spin'] == 'vector'`.
- `RateSpectrum.metadata['flux_file']` is an absolute path that exists.

The same call with `form_factor_type='qcdark1'` and with `'qedark'` should also
return finite, positive dR/dE.

---

## Acceptance Criteria

- [ ] All halo-path notebooks (`DMeRates_Examples.ipynb`,
      `modulation_study/modulation_figures.ipynb`) execute end-to-end unchanged.
- [ ] All halo-path pytest tests still pass.
- [ ] SRDM smoke test above succeeds for all three backends.
- [ ] Negative-path tests pass:
  - manifest miss → `FileNotFoundError` with manifest path.
  - `mediator_spin='scalar'` → `NotImplementedError`.
  - QCDark2 path missing `screening` → `ValueError`.
- [ ] `mediator_spin` defaults to `'vector'` and is silently ignored when
      `halo_model != 'srdm'`.

---

## Hard Invariants

- **No method signatures used by `DMeRates_Examples.ipynb` or
  `modulation_study/` change.** All new kwargs are optional with sensible
  defaults.
- **No physics changes.** This agent only wires kwargs through and adds
  validation guards.
- **No `qcdark2.*` imports** anywhere in the production code.

---

## Handoff

Report files changed and the SRDM smoke-test output for all three backends.
Hands off to `validator` for the final SRDM checkpoint.
