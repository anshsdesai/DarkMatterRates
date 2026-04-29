# Phase 0.1 — Pytest Regression Baselines

## What Was Done

Converted the three validation notebooks into automated pytest regression tests. All 10 tests pass against current code.

```
tests/
├── conftest.py          — random seed, session fixture, reference arrays
├── test_qedark.py       — QEDark legacy path (Si, Ge)
├── test_qcdark1.py      — QCDark1 HDF5 path (Si, Ge)
└── test_noble_gas.py    — Wimprates noble-gas path (Xe)
```

Run with:
```bash
source .venv/bin/activate
pytest tests/test_qedark.py tests/test_qcdark1.py tests/test_noble_gas.py -v
```

---

## Test Cases

| File | Material | mX [MeV] | FDMn | DoScreen | integrate | Tolerance |
|------|----------|----------|------|----------|-----------|-----------|
| test_qedark.py | Si | [10, 1000] | 0 | False | False | rtol=1e-6 |
| test_qedark.py | Ge | [10, 1000] | 2 | False | False | rtol=1e-6 |
| test_qcdark1.py | Si | [1000] | 0 | True | True | rtol=0.02 |
| test_qcdark1.py | Ge | [1000] | 2 | True | True | rtol=0.02 |
| test_noble_gas.py | Xe | 1000 | 0 | — | — | rtol=0.05 |
| test_noble_gas.py | Xe | 1000 | 2 | — | — | rtol=0.05 |

Each backend also has a `*_below_threshold_is_zero` test asserting `ne=0` rate is exactly `0.0`.

Additional physics parameters:
- QEDark: `change_to_step()` called (step-function ionization model)
- QCDark1: `update_crosssection(1e-39)`, `halo_model='imb'`
- Wimprates: `update_crosssection(4e-44)`, `halo_model='shm'`, `nes=range(1,17)`, `returnShells=False`

---

## Non-Obvious Findings

### 1. `Constants.py` randomizes units at import time

`Constants.py` directly sets `nu.m`, `nu.s`, `nu.kg`, `nu.C`, `nu.K` via `random.uniform` every time the module is first imported:

```python
nu.kg = 10 ** random.uniform(10,12)
nu.s  = 10 ** random.uniform(5,7)
# ...
nu.set_derived_units_and_constants()
```

This means any `nu.reset_units('SI')` called before the import is silently overridden. The correct fix for reproducible tests is `random.seed(0)` at the very top of `conftest.py` — before any other import — so that `Constants.py`'s randomization is deterministic.

### 2. `nu.reset_units('SI')` in a pytest fixture is harmful

Calling `nu.reset_units('SI')` inside a session fixture (which runs after test modules are imported) makes `nu.kg` etc. return SI values, but `Constants.py`'s already-computed derived constants (`me_eV`, `mP_eV`, `tf_screening`, etc.) still hold seed-0 values. The mismatch produces incorrect unit conversions in the rate output. **Do not call `nu.reset_units()` anywhere in the test suite.**

For comparison notebooks, `nu.reset_units('SI')` after DMeRates imports is fine — notebooks display physical values for human reading, not for numeric reproducibility.

### 3. `integrate=False` is not bit-for-bit reproducible

The Riemann-sum path uses PyTorch float32 internally. When the tensor is cast to float64 numpy, the float32 quantization introduces ~1e-9 relative noise. `np.array_equal` fails; `np.allclose(rtol=1e-6, atol=1e-10)` is the correct tolerance. This is still ~100× tighter than the QCDark1 integration tolerance.

---

## Reference Value Strategy

Reference values were captured by running the equivalent of each notebook with `random.seed(0)` set before any DMeRates import, then calling:

```python
rates.cpu().numpy() * nu.kg * nu.day   # QEDark
rates.cpu().numpy() * nu.kg * nu.year  # QCDark1
rates[:,0].cpu().numpy() * nu.tonne * nu.year  # wimprates
```

The resulting float64 arrays are hardcoded in `conftest.py` with comments identifying the source notebook, cell, and physics parameters.

---

## What Is Deferred

**Step 0.2 — The `* 10` factor** (`DMeRate.py:853`) is explicitly deferred. The `integrate=True` and `integrate=False` semiconductor paths are known to disagree numerically; this difference is intentional and requires a separate investigation pass before the refactor begins in Phase 2.

---

## What Passes Now

These tests are the regression gate for all subsequent refactor phases. Any Phase 1–3 change that breaks these tests has changed existing physics behavior.
