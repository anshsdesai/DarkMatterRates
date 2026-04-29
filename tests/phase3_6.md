# Phase 3.6 — Public API Wiring for QCDark2

## What Was Done

Wired QCDark2 into the legacy `DMeRate` API and added a new `RateCalculator` facade.

Created:

```text
DMeRates/rate_calculator.py
```

Updated:

```text
DMeRates/DMeRate.py
tests/test_qcdark2.py
```

(`DMeRates/__init__.py` was kept minimal to avoid module-shadowing regressions in
legacy notebook import patterns.)

---

## DMeRate Wiring

### Constructor support

Now supported:

```python
DMeRate("Si", form_factor_type="qcdark2")
DMeRate("GaAs", form_factor_type="qcdark2")
DMeRate("Diamond", form_factor_type="qcdark2")
```

Internally, this loads `dielectric_response(...)` and uses the native QCDark2 dielectric
engine for spectrum/rate computation.

### QCDark2 rate path

Added QCDark2-specific methods in `DMeRate`:

- `calculate_qcdark2_rates(...)`
- `calculate_qcdark2_spectrum(...)`
- `qcdark2_above_threshold_rate(...)`

`calculate_rates(...)` now dispatches to the QCDark2 path when
`form_factor_type='qcdark2'`.

### Yield-policy enforcement at call time

For QCDark2 ne-rates:

- `Si`: RK probabilities used
- `Ge`: step approximation used (default pair energy)
- `GaAs`, `SiC`, `Diamond`: explicit `pair_energy` required

### Explicit-screening enforcement at API layer

QCDark2 `calculate_rates(...)` now requires an explicit `screening` selection and raises
the Step 3.4 message when omitted.

---

## RateCalculator Wiring

Added:

```python
RateCalculator("Si", backend="qcdark2", variant="composite", screening="rpa")
RateCalculator("Diamond", backend="qcdark2", variant="lfe", screening="none")
```

Current behavior:

- wraps the legacy class for all backends
- for `backend='qcdark2'`, routes to `DMeRate(..., form_factor_type='qcdark2')`
- provides `calculate_rates(...)` and `calculate_spectrum(...)` (qcdark2 only)

---

## Verification

```bash
source .venv/bin/activate
pytest tests/ -v
```

Final result after 3.6:
- `28 passed`

QCDark2 API checks now included in `tests/test_qcdark2.py`:

- `test_dmerate_qcdark2_si_calculates_rate`
- `test_ratecalculator_qcdark2_si_calculates_rate`
- `test_dmerate_qcdark2_requires_explicit_screening`
- `test_qcdark2_pair_energy_required_for_gaas_ne_rates`
- `test_qcdark2_above_threshold_rate_available_for_gaas`

---

## Notebook Gate Status

Required checkpoint command:

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute \
    modulation_study/modulation_figures.ipynb \
    --output /tmp/modulation_exec.ipynb \
    --ExecutePreprocessor.timeout=600
```

Observed blocker in this environment:
- notebook execution fails in `modulation_study/isoangle.py` via
  `astropy.coordinates.EarthLocation.of_address(...)` due external geocoding/network
  restrictions (`HTTP 403 Forbidden`), i.e. not a QCDark2 API wiring failure.
