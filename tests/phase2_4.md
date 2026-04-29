# Phase 2.4 — Halo Provider Extraction

## What Was Done

Extracted halo-provider responsibilities into modular components and routed legacy
`DMeRate.setup_halo_data(...)` / `get_halo_data(...)` through compatibility wrappers.

Created:

```text
DMeRates/halo/__init__.py
DMeRates/halo/analytic.py
DMeRates/halo/file_loader.py
DMeRates/halo/independent.py
```

Updated:

```text
DMeRates/DMeRate.py
```

Supported keys preserved:
- `'shm'`, `'tsa'`, `'dpl'`
- `'modulated'`, `'summer'`
- `'imb'`
- halo-independent step-eta path via `halo_id_params`

Common provider interface now available in extracted modules:

```python
eta(v_min_tensor) -> torch.Tensor
```

---

## Verification

```bash
source .venv/bin/activate
pytest tests/ -v
```

Result:
- `14 passed`

Required post-step notebook checkpoint executed:

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute \
    modulation_study/modulation_figures.ipynb \
    --output /tmp/modulation_exec.ipynb \
    --ExecutePreprocessor.timeout=600
```

Observed blocker:
- Notebook failed due to missing modulated-halo directory:
  `../halo_data/modulated/FDM1/Verne_summer/mDM_1_0_MeV_sigmaE_1e-34_cm2/`

---

## Notes

- Interpolation and out-of-range masking behavior were preserved.
- Extraction remained non-invasive to notebook-exposed signatures.
