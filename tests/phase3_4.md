# Phase 3.4 — QCDark2 Screening Selection

## What Was Done

Added an explicit QCDark2 screening-policy module and routed the dielectric engine through it.

Created:

```text
DMeRates/screening/dielectric.py
```

Updated:

```text
DMeRates/screening/__init__.py
DMeRates/engines/dielectric.py
tests/test_qcdark2.py
```

### Implemented screening modes

- `screening='rpa'`: `epsilon_screen = epsilon`
- `screening='none'`: `epsilon_screen = 1` everywhere

The engine now computes:

```text
screen_ratio = |epsilon|^2 / |epsilon_screen|^2
```

### Explicit-screening guard

`screening=None` now raises:

```python
ValueError(
    "QCDark2 calculations require an explicit screening choice. "
    "Pass screening='rpa' or screening='none'."
)
```

This matches the Step 3.4 requirement.

---

## Verification

```bash
source .venv/bin/activate
pytest tests/ -v
```

Result at this step:
- all tests passed (`23 passed` at the checkpoint before 3.5/3.6 additions)

Additional targeted check:

```bash
source .venv/bin/activate
pytest tests/test_qcdark2.py -v
```

Confirmed:
- RPA reference agreement remains intact
- explicit-screening error path is enforced
- invalid screening keys still raise clear errors

---

## Notes

- Screening logic is now centralized in `DMeRates/screening/dielectric.py` so both
  engine-level and API-level wiring can share consistent validation behavior.
