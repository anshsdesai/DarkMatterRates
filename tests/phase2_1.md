# Phase 2.1 — Response Loader Extraction

## What Was Done

Extracted form-factor/data loader classes out of the monolithic `DMeRates/form_factor.py`
into the new `DMeRates/responses/` package, while preserving backward compatibility.

Created:

```text
DMeRates/responses/__init__.py
DMeRates/responses/qcdark1.py
DMeRates/responses/qedark.py
DMeRates/responses/noble_gas.py
```

Updated:

```text
DMeRates/form_factor.py
```

`DMeRates/form_factor.py` now provides compatibility re-exports:

```python
from DMeRates.responses.qcdark1 import form_factor
from DMeRates.responses.qedark import form_factorQEDark
from DMeRates.responses.noble_gas import formFactorNoble
```

No physics behavior changes were introduced.

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
- Notebook failed before the extraction logic was exercised due to missing modulated-halo
  data path:
  `../halo_data/modulated/FDM1/Verne_summer/mDM_1_0_MeV_sigmaE_1e-34_cm2/`

---

## Notes

- This step intentionally kept `DMeRate.py` functional and unchanged in call behavior.
- The failure above is a data-layout/environment gate issue, not a regression introduced by
  this extraction.
