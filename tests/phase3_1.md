# Phase 3.1 — QCDark2 Dielectric Response Loader

## Prerequisite Gate Check (Step 3.0)

Validated before starting 3.1:

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute \
    tests/qcdark2_formula_derivation.ipynb \
    --output /tmp/qcdark2_formula_derivation_exec.ipynb \
    --ExecutePreprocessor.timeout=600
```

Observed outputs from executed notebook:
- `Relative agreement (DMeRates vs. QCDark2 reference):`
- `Max relative disagreement = 0.0000%`
- `PASS — agreement < 5% at every validation energy.`

Gate status:
- Satisfied (`< 5%` agreement).

---

## What Was Done

Created QCDark2 dielectric response loader:

```text
DMeRates/responses/dielectric.py
```

Updated exports:

```text
DMeRates/responses/__init__.py
```

Loader features implemented:
- Uses `DataRegistry.qcdark2_dielectric(material_key, variant)` for file resolution.
- Loads required fields: `epsilon`, `q`, `E`, `M_cell`, `V_cell`, `dE`.
- Keeps raw atomic-unit momentum as `q_ame`.
- Converts units at load time:
  - `q = q_ame * (alphaFS * me * c0)`
  - `E = E_raw * nu.eV`
  - `M_cell = M_cell_raw * nu.eV`
  - `V_cell = V_cell_bohr * bohr**3`
  - `dE = dE_raw * nu.eV`
- Supports variants: `composite`, `lfe`, `nolfe`.
- Supports materials: `Si`, `Ge`, `GaAs`, `SiC`, `diamond` (also accepts `Diamond`).

---

## Required Pre-Check (Si composite file)

Executed:

```python
import h5py
with h5py.File('/Users/ansh/Local/SENSEI/QCDark2/dielectric_functions/composite/Si_comp.h5', 'r') as f:
    q = f['q'][:]
    V_cell = f.attrs['V_cell']
    M_cell = f.attrs['M_cell']
```

Observed:
- `q range: 0.010 - 25.010`
- `V_cell: 270.11`
- `M_cell: 5.216e+10`

Convention recorded in loader module docstring:
- `q` stored in `alpha*me` units
- `V_cell` stored in `Bohr^3`
- `M_cell` stored in `eV`

---

## Verification

Smoke check:
- `dielectric_response('Si', 'composite')` loaded correctly
- `epsilon.shape == (1251, 501)`
- `q_ame`, `q`, `E`, `M_cell`, `V_cell`, `dE` populated with expected ranges

Regression tests:

```bash
source .venv/bin/activate
pytest tests/ -v
```

Result:
- `14 passed`

---

## Scope Control

- Only Step 3.1 was implemented in this pass.
- Steps 3.2 and 3.3 remained out of scope.
- Work stopped after Step 3.1 completion by request.
