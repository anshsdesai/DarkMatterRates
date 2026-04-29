# Phase 2.3 — Screening Extraction

## What Was Done

Extracted Thomas-Fermi screening logic from `DMeRate.py` into a dedicated screening module
and retained compatibility wrappers.

Created:

```text
DMeRates/screening/__init__.py
DMeRates/screening/thomas_fermi.py
```

Updated:

```text
DMeRates/DMeRate.py
```

Compatibility preserved:
- `TFscreening(DoScreen)` method path retained.
- `thomas_fermi_screening(...)` vectorized path retained.
- `DoScreen` behavior unchanged.

---

## Verification

```bash
source .venv/bin/activate
pytest tests/ -v
```

Result:
- `14 passed`

---

## Notes

- Screened/unscreened behavior remained stable against existing regression baselines.
- This step did not change public API signatures.
