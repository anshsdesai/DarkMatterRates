# Phase 2.5 — Rate Engine Extraction

## What Was Done

Extracted semiconductor and noble-gas engine logic into `DMeRates/engines/` without
switching the runtime off the legacy monolith.

Created:

```text
DMeRates/engines/__init__.py
DMeRates/engines/form_factor.py
DMeRates/engines/noble_gas.py
```

Key requirement implemented:

```python
_LEGACY_QEDARK_ENERGY_NORM = 1.0 * nu.eV
```

with the required comment:

```python
# Legacy QEDark convention: grid values summed directly without dE bin-width factor.
# Equivalent to dE * 10 for the current 0.1 eV grid. See tests/phase0_2.md.
```

Compatibility status:
- `DMeRate.calculate_rates(...)` remains the active public path.
- Legacy argument names remain unchanged.
- Extraction is available in parallel modules for subsequent wiring.

---

## Verification

```bash
source .venv/bin/activate
pytest tests/ -v
```

Result:
- `14 passed`

End-of-Phase-2 notebook checkpoint executed:

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute \
    modulation_study/modulation_figures.ipynb \
    --output /tmp/modulation_exec.ipynb \
    --ExecutePreprocessor.timeout=600
```

Observed blocker:
- Notebook failed on missing modulated-halo directory:
  `../halo_data/modulated/FDM1/Verne_summer/mDM_1_0_MeV_sigmaE_1e-34_cm2/`

---

## Notes

- This step intentionally did not delegate production runtime to the new engines yet.
- Existing regression tests show no behavior change on current covered pathways.
