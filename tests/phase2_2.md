# Phase 2.2 — Ionization/Yield Extraction

## What Was Done

Extracted ionization/yield logic from `DMeRate.py` into standalone modules with explicit
inputs (energy array, metadata, dtype/path), then wired thin compatibility wrappers back in
`DMeRate.py`.

Created:

```text
DMeRates/ionization/__init__.py
DMeRates/ionization/rk_probabilities.py
DMeRates/ionization/step_function.py
```

Updated:

```text
DMeRates/DMeRate.py
```

Behavior preserved:
- Si QCDark1 default still uses RK probabilities.
- Ge still switches to step function in the same path as before.
- Legacy `change_to_step()` path remains reproducible.

---

## Verification

```bash
source .venv/bin/activate
pytest tests/ -v
```

Result:
- `14 passed`

Notebook checkpoint command executed:

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute \
    modulation_study/modulation_figures.ipynb \
    --output /tmp/modulation_exec.ipynb \
    --ExecutePreprocessor.timeout=600
```

Observed blocker:
- Notebook failed at the same modulated-halo data lookup:
  `../halo_data/modulated/FDM1/Verne_summer/mDM_1_0_MeV_sigmaE_1e-34_cm2/`

---

## Notes

- Extraction preserved method names/signatures used by existing notebooks and tests.
- Logic remained option-A style (duplicate/extract while monolith remains primary runtime).
