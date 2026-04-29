# Phase 0.2 — Non-Integrated Semiconductor `* 10` Factor

## Conclusion

The `* 10` in `calculate_semiconductor_rates(..., integrate=False)` is best understood as a legacy energy-unit normalization, not as a physical 0.1 eV bin-width integration.

All current semiconductor form-factor grids inspected in this pass have `dE = 0.1 eV`. Therefore `self.form_factor.dE * 10` is numerically and dimensionally `1 eV`. This preserves the old QEDark convention where `vectorized_dRdE` grid values were multiplied by the ionization probabilities and summed directly over the electron-energy grid, with no `dE` factor.

Do not remove the factor or replace it with `self.form_factor.dE` without intentionally changing the validated QEDark legacy rates by a factor of 10.

## Evidence

| Check | Result | Interpretation |
|---|---:|---|
| QEDark Si/Ge `form_factor.dE` | `0.1 eV` | `dE * 10` is `1 eV` for the legacy path. |
| QCDark Si/Ge HDF5 `results.attrs["dE"]` | `0.1` | Same 0.1 eV energy-grid spacing in current QCDark inputs. |
| Current QEDark `dE*10` vs `dE` accumulation | exact factor `10` | Removing `* 10` would scale validated QEDark rates down by 10. |
| Old `old_for_comparison/QEDark/QEDark3.py` | `dRdne = torch.sum(dRdE * fn_tiled, axis=1)` | Historical QEDark path summed grid values directly, with no 0.1 eV bin width. |
| Old vs current QEDark Si, `mX=1000`, `FDMn=0`, step probabilities | old `8169.45`, current `8167.49` events/kg/year for `ne=1` | Current behavior with `dE*10` reproduces the legacy normalization at sub-percent level for Si. |
| QCDark `integrate=False` vs `integrate=True` | non-integrated path is hundreds to thousands of times larger | This is dominated by the separate q-grid convention/integral mismatch (`1/q` sum versus Simpson integral over `dq/q^2`), so it is not evidence that the energy `* 10` is a bug. |
| Integrated QCDark trapezoid vs rectangular `dE` over step probabilities | ratio `1.0` for tested bins | The integrated path is consistent with true 0.1 eV energy-bin integration; the non-integrated path follows the legacy QEDark convention instead. |

## Commands Run

```bash
source .venv/bin/activate && pytest tests/test_qedark.py tests/test_qcdark1.py -q
```

```bash
source .venv/bin/activate && python - <<'PY'
import h5py
for mat in ['Si','Ge']:
    with h5py.File(f'form_factors/QCDark/{mat}_final.hdf5','r') as f:
        print(mat, f['results/f2'].shape, f['results'].attrs['dE'])
PY
```

Additional ad hoc CPU scripts compared:

- current `integrate=False` accumulation with `dE` versus `dE*10`
- current QCDark `integrate=True` versus `integrate=False`
- old `old_for_comparison/QEDark/QEDark3.py` direct-sum QEDark rates versus current QEDark rates

## Open Risks

The comment assumes the semiconductor non-integrated grids remain at `dE = 0.1 eV`. If a future form-factor file uses a different energy spacing, `dE * 10` would no longer equal `1 eV`; that should be revisited with explicit baselines before changing behavior.

The large QCDark `integrate=False`/`integrate=True` disagreement appears to be a q-integration convention mismatch, not an energy-bin normalization issue. This pass did not attempt to fix or refactor that path.
