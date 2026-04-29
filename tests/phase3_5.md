# Phase 3.5 — QCDark2 Material Metadata and Yield Policy

## What Was Done

Added QCDark2 material metadata and a centralized pair-energy policy helper.

Created:

```text
DMeRates/responses/dielectric_materials.py
```

Updated:

```text
DMeRates/responses/__init__.py
```

### Material metadata

Added scissor-corrected QCDark2 bandgaps:

```python
QCDARK2_BANDGAPS = {
    "Si":      1.1  * nu.eV,
    "Ge":      0.67 * nu.eV,
    "GaAs":    1.42 * nu.eV,
    "SiC":     2.36 * nu.eV,
    "Diamond": 5.5  * nu.eV,
}
```

### Yield-policy enforcement helper

Added:

- `canonical_qcdark2_material(material)`
- `require_qcdark2_pair_energy(material, pair_energy)`

Policy encoded:

- `Si`: RK yield remains allowed for ne-rate conversion
- `Ge`: legacy step approximation retained by default
- `GaAs`, `SiC`, `Diamond`: ne-rate conversion requires explicit `pair_energy`

If omitted for those materials, raises:

```python
ValueError(
    f"QCDark2 ne rates for {material} require an explicit pair_energy (eV). "
    "The QCDark2 paper does not provide validated pair energies for this material."
)
```

---

## Verification

Validation happened together with 3.6 wiring (where enforcement is exercised through
public APIs):

```bash
source .venv/bin/activate
pytest tests/ -v
```

Result:
- pass at checkpoint, with pair-energy policy verified by dedicated tests in
  `tests/test_qcdark2.py`.

---

## Notes

- Bandgaps were added in a dedicated response metadata module (instead of `config.py`)
  to keep QCDark2-specific material policy close to dielectric-response code.
