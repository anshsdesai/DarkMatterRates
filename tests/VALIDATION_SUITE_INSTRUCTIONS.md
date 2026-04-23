# DarkMatterRates Physics Validation Suite — Agent Instructions

## What you are building

A validation suite for the DarkMatterRates package at `/Users/ansh/Local/SENSEI/DarkMatterRates`. The suite has two tiers:

1. **Comparison notebooks** (primary): Visual + quantitative comparisons between DMeRates outputs and upstream reference codebases, in the style of `DMeRates_Examples.ipynb`. One notebook per backend.
2. **Pytest physics tests** (secondary): Automated invariant + parity assertions that can run in CI without human inspection.

The user has been doing informal validation by running modified code in `old_for_comparison/` and comparing figures in `DMeRates_Examples.ipynb`. This suite formalizes that workflow.

---

## Environment

```bash
source /Users/ansh/Local/SENSEI/DarkMatterRates/.venv/bin/activate
```

Upstream reference codebases are available at:
- `../QCDark/dark_matter_rates.py`
- `../QEdark/QEdark-python/` (has `Si_f2.txt`, `Ge_f2.txt`, `DM_halo_dist.py` but **no rate function** — use QCDark's `d_rate()` with QEDark form factors)
- `../QCDark2/qcdark2/dark_matter_rates.py`
- `../wimprates/wimprates/electron.py`

Pre-computed reference CSVs for wimprates: `old_for_comparison/wimprates_mod/` (format: two columns — energy_eV, rate or ne, rate).

---

## Critical upstream API references

### QCDark (../QCDark/dark_matter_rates.py)
```python
import sys; sys.path.insert(0, '../QCDark')
import dark_matter_rates as qcdark_dm
ff = qcdark_dm.form_factor('form_factors/QCDark/Si_final.hdf5')  # loads .dq, .dE, .mCell, .ff
Earr, dRdE = qcdark_dm.d_rate(mX_eV, ff, FDM_exp=0, screening=qcdark_dm.default_no_sreen, astro_model=qcdark_dm.default_astro)
# default_astro: v0=238, vEarth=250.2, vEscape=544, rhoX=0.3e9 eV/cm³, sigma_e=1e-39 cm²
# d_rate returns (Earr_eV, dRdE in events/kg/year/eV)
# Also: d_rate_FanoQ(mX, ff, E2Q, ...) → (Ebins, dRbins)  [step-fn ne bins]
# Also: d_rate_RamanathanQ(mX, ff, ionizationFile, ...) → (ne_arr, dRbins)  [RK ne bins]
```

### QEDark (no rate function — use QCDark engine with QEDark form factors)
```python
# Duck-typed adapter to pass QEDark ff to QCDark's d_rate():
class QEDarkFFAdapter:
    dq = 0.02 * 1/137 * 511e3          # eV: alpha * mElectron
    dE = 0.1                             # eV
    mCell = <Si_mCell or Ge_mCell>       # in eV (same value as QCDark HDF5 mCell)
    band_gap = 1.12  # Si or 0.67 Ge, eV
    ff = <loaded from f2.txt with wk/4 preprocessing — see note below>

Earr, dRdE = qcdark_dm.d_rate(mX_eV, adapter, FDM_exp=n, ...)
```
**CRITICAL — wk/4 preprocessing**: `form_factorQEDark` in DMeRates applies a `wk/4` factor when loading `f2.txt`. Before writing comparison code, verify by reading `DMeRates/form_factor.py` lines loading QEDark. The adapter must apply the same factor so both paths use identical ff arrays. Check: load one known (q_index, E_index) value from raw txt and compare to `dmrates_qedark.ff[q_index, E_index]`.

### QCDark2 (../QCDark2/qcdark2/dark_matter_rates.py)
```python
import sys; sys.path.insert(0, '../QCDark2')
from qcdark2 import dark_matter_rates as dm2
epsilon = dm2.load_epsilon('../QCDark2/dielectric_functions/composite/Si_comp.h5')
reference, energy_eV = dm2.get_dR_dE(epsilon, m_X=100e6, mediator='heavy', astro_model=dm2.default_astro, screening='RPA')
# mediator: 'heavy' (FDMn=0) or 'light' (FDMn=2)
# already tested in tests/test_qcdark2_reference.py — use that as reference implementation
```

### wimprates (../wimprates/wimprates/electron.py)
```python
import sys; sys.path.insert(0, '../wimprates')
import numericalunits as nu
import wimprates as wr

halo = wr.StandardHaloModel()   # defaults may differ from DMeRates — match v0, vEarth, v_esc
erec_array = np.linspace(1, 400, 100) * nu.eV
dRdE_shell = wr.rate_dme(erec_array, n=5, l=1, mw=mX_kg, sigma_dme=sigma_cm2, f_dm='1')
# Shells Xe: (5,1)=5p, (5,0)=5s, (4,2)=4d, (4,1)=4p, (4,0)=4s
# f_dm: '1'=heavy mediator, '1_q2'=light mediator (FDMn=2)
# returns dR/dE in wimprates' unit system
```

---

## File structure to create

```
tests/
  compare_qcdark.ipynb          ← NEW: Si/Ge QCDark comparison notebook
  compare_qedark.ipynb          ← NEW: Si/Ge QEDark comparison notebook
  compare_qcdark2.ipynb         ← NEW: all-material QCDark2 comparison notebook
  compare_wimprates.ipynb       ← NEW: Xe/Ar wimprates comparison notebook
  test_qcdark_physics.py        ← NEW: pytest physics invariants + parity, QCDark
  test_qedark_physics.py        ← NEW: pytest physics invariants + parity, QEDark
  test_qcdark2_physics.py       ← NEW: pytest physics invariants (extends test_qcdark2_reference.py)
  test_wimprates_physics.py     ← NEW: pytest physics invariants + parity, wimprates
  test_cross_backend_consistency.py  ← NEW: Si/Ge cross-backend sanity
  conftest.py                   ← EXTEND: add upstream paths, fix_units fixture
```

---

## Tier 1: Comparison notebooks (build these first)

Each notebook follows the same structure as `DMeRates_Examples.ipynb` but with an explicit reference overlay. The target audience is a physicist who wants to see whether DMeRates agrees with upstream by looking at the plots.

### Standard figure layout per notebook

**Figure 1: dR/dE spectrum overlay**
- Top panel: DMeRates dR/dE (solid) and upstream dR/dE (dashed) on log scale, same axes
- Bottom panel: ratio (DMeRates / upstream), with ±5% band shown
- X-axis: energy (eV), Y-axis: events/kg/year/eV
- Legend identifies each line

**Figure 2: dR/dne vs DM mass**
- DMeRates rates (solid) and upstream rates (dashed) for ne=1 through ne=5
- X-axis: DM mass (MeV), both axes log scale
- Y-axis: events/kg/year

**Figure 3: Residuals summary table**
Printed text block (not a figure) showing:
```
mX = 100 MeV, FDMn = 0, Si:
  total rate (DMeRates):   X.XXe+YY events/kg/year
  total rate (upstream):   X.XXe+YY events/kg/year
  rel. difference:         X.X%
  max bin residual:        X.X%
```

### Benchmark physics points (use in all notebooks)
```python
BENCHMARK_MASSES_MEV = [10.0, 100.0, 1000.0]
BENCHMARK_FDM = {'heavy': 0, 'light': 2}
```

---

### compare_qcdark.ipynb

**Materials**: Si and Ge  
**Reference code**: `../QCDark/dark_matter_rates.py`

**Cell structure**:

Cell 1 — Imports and setup:
```python
import sys
sys.path.insert(0, '../QCDark')
sys.path.insert(0, '..')  # for DMeRates

import numpy as np
import matplotlib.pyplot as plt
import numericalunits as nu
nu.reset_units('SI')    # deterministic units for comparison

import dark_matter_rates as qcdark_dm
from DMeRates.DMeRate import DMeRate

# Match SHM parameters: QCDark default_astro uses sigma_e=1e-39 cm²
# DMeRates default is 1e-36 cm²; always call update_params()
```

Cell 2 — dR/dE comparison at mX=100 MeV, FDMn=0, Si:
```python
mX_MeV = 100.0
mX_eV = mX_MeV * 1e6

# Upstream reference
ff_si = qcdark_dm.form_factor('form_factors/QCDark/Si_final.hdf5')
Earr_ref, dRdE_ref = qcdark_dm.d_rate(mX_eV, ff_si, FDM_exp=0,
                                        screening=qcdark_dm.default_no_sreen,
                                        astro_model=qcdark_dm.default_astro)

# DMeRates
obj = DMeRate('Si', form_factor_type='qcdark', device='cpu')
obj.update_params(qcdark_dm.default_astro['v0'], qcdark_dm.default_astro['vEarth'],
                  qcdark_dm.default_astro['vEscape'], qcdark_dm.default_astro['rhoX'],
                  qcdark_dm.default_astro['sigma_e'])
Earr_dm, spectra = obj.calculate_spectrum([mX_MeV], 'shm', 0, DoScreen=False)
dRdE_dm = spectra[0]  # events/kg/year/eV in DMeRates units
```

Cell 3 — Plot Figure 1 (overlay + ratio)

Cell 4 — dR/dne vs mass sweep:
```python
mX_array = np.concatenate([np.arange(0.5, 5, 0.2), np.arange(5, 11, 1), [20, 50, 100, 200, 1000]])
ne_bins = [1, 2, 3, 4, 5]
rates_dm = obj.calculate_ne_rates(mX_array, 'shm', 0, ne_bins, DoScreen=False)
# Reference: loop over mX calling d_rate_FanoQ()
```

Cell 5 — Plot Figure 2 (ne vs mass)

Cell 6 — Residuals summary (print table, not plot)

Cell 7 — Repeat for Ge, FDMn=2

**Note**: The RK ionization probability path (Si only) should also be validated:
```python
# Reference: d_rate_RamanathanQ(mX, ff_si, 'DMeRates/Rates/p100k.dat', ...)
# DMeRates: same but with ionization_func='RK' (default for Si)
```

---

### compare_qedark.ipynb

**Materials**: Si and Ge  
**Reference code**: QCDark `d_rate()` with QEDark form factors as adapter

**IMPORTANT — build QEDarkFFAdapter first**:

```python
# Step 1: Read DMeRates' form_factorQEDark source to see exactly how it loads the file
# (Read DMeRates/form_factor.py, find form_factorQEDark.__init__)
# Step 2: Replicate that loading in the adapter so ff arrays are identical
# Step 3: Verify: load one (q,E) value from raw txt and from form_factorQEDark
#   raw_data = np.loadtxt('form_factors/QEDark/Si_f2.txt')
#   dmr_ff = DMeRate('Si', form_factor_type='qedark').ff[q_index, E_index]
#   adapter_ff = QEDarkFFAdapter('Si').ff[q_index, E_index]
#   assert raw_data[q_index, E_index] * preprocessing_factor ≈ dmr_ff ≈ adapter_ff
```

Cell structure: Same as compare_qcdark.ipynb but:
- Reference = `qcdark_dm.d_rate(mX_eV, adapter, ...)` with QEDark adapter
- DMeRates: `DMeRate('Si', form_factor_type='qedark')`, use `integrate=True` for fair comparison
- Note visually whether QEDark and QCDark spectra differ (expected: they will, different physics)
- Add a third overlay cell comparing QCDark vs QEDark on the same axes (not just DMeRates vs reference, but also cross-backend)

---

### compare_qcdark2.ipynb

**Materials**: Si, Ge, GaAs, SiC, Diamond  
**Variants**: composite, lfe, nolfe  
**Reference code**: `../QCDark2/qcdark2/dark_matter_rates.py`

This notebook is an enhanced version of what `test_qcdark2_reference.py` does, but with visual output.

Cell structure:
- Cell 1: Imports + `nu.reset_units('SI')` + `import warnings; warnings.filterwarnings('ignore', ...)`
- Cell 2: Helper `compare_material_variant(material, variant, mX_MeV, FDMn)` → returns fig with overlay + ratio
- Cell 3-N: Call helper for each (material, variant) combination showing Figure 1
- Cell N+1: `DoScreen` no-op demonstration — show `rate(DoScreen=True) == rate(DoScreen=False)` numerically
- Cell N+2: Variant comparison for Si: composite vs lfe vs nolfe on same axes

**Note on existing tests**: `tests/test_qcdark2_reference.py` already runs the numerical parity checks. The notebook's goal is visualization, not duplication.

---

### compare_wimprates.ipynb

**Materials**: Xe, Ar  
**Reference**: 
  - Primary: `../wimprates/wimprates/electron.py` `rate_dme()` (live-run)
  - Secondary: pre-computed CSVs in `old_for_comparison/wimprates_mod/`

Cell structure:

Cell 1 — Imports + SHM parameter matching:
```python
import wimprates as wr
# wimprates StandardHaloModel defaults: v0=220 km/s; DMeRates uses 238 km/s
# Must construct halo_model with matching parameters:
halo_model = wr.StandardHaloModel()  
# Check wr.StandardHaloModel attributes and override v0 to match DMeRates
```

Cell 2 — Per-shell dR/dE comparison for Xe at mX=100 MeV:
```python
# DMeRates
obj = DMeRate('Xe', form_factor_type='wimprates', device='cpu')
# noble_dRdE returns dict keyed by shell name
shells_dm = obj.noble_dRdE(mX_MeV, 0, 'shm', ...)

# wimprates: call rate_dme() for each shell
Xe_shells = [(5, 1), (5, 0), (4, 2), (4, 1), (4, 0)]  # 5p, 5s, 4d, 4p, 4s
for n, l in Xe_shells:
    dRdE_ref = wr.rate_dme(erec_array, n, l, mw=mX, sigma_dme=sigma, f_dm='1', halo_model=halo_model)
```

Cell 3 — Figure 1: Per-shell overlay for Xe (one subplot per shell)

Cell 4 — dR/dne per shell (same style as `DMeRates_Examples.ipynb` Figure for Xe)

Cell 5 — Comparison against saved CSVs in `old_for_comparison/wimprates_mod/`:
```python
# Load darkside_100MeV_1e-36_fdm0_3p.csv (two columns: ne, rate)
# Overlay on DMeRates dR/dne
```

Cell 6 — Repeat for Ar

**Critical physical check**: The vmin formula for noble gases includes binding energy: `vmin = (E + Ebind)/q + q/(2mX)`. In a markdown cell, explain this and show that DMeRates `noble_dRdE()` rates are zero below the binding energy of each shell — verify this explicitly in the notebook.

---

## Tier 2: Pytest physics tests

These run in CI and catch regressions automatically. They reference the same upstream code as the notebooks but only check numerical assertions.

### conftest.py additions

Add to the existing `tests/conftest.py`:

```python
UPSTREAM_QCDARK_ROOT   = os.path.abspath(os.path.join(REPO_ROOT, '..', 'QCDark'))
UPSTREAM_QEDARK_ROOT   = os.path.abspath(os.path.join(REPO_ROOT, '..', 'QEdark', 'QEdark-python'))
UPSTREAM_WIMPRATES_ROOT = os.path.abspath(os.path.join(REPO_ROOT, '..', 'wimprates'))

def requires_upstream_qcdark():
    path = os.path.join(UPSTREAM_QCDARK_ROOT, 'dark_matter_rates.py')
    return pytest.mark.skipif(not os.path.exists(path), reason=f"QCDark not found at {path}")

def requires_upstream_wimprates():
    path = os.path.join(UPSTREAM_WIMPRATES_ROOT, 'wimprates', 'electron.py')
    return pytest.mark.skipif(not os.path.exists(path), reason=f"wimprates not found at {path}")

@pytest.fixture(autouse=False, scope='session')
def fix_units():
    """Deterministic unit scales for cross-run comparable parity tests."""
    import numericalunits as nu
    nu.reset_units('SI')
    yield
    nu.reset_units()

@pytest.fixture(params=[42, 137, 2718])
def random_nu_seed(request):
    import numericalunits as nu
    nu.reset_units(request.param)
    yield request.param
    nu.reset_units()
```

Shared benchmark constants:
```python
BENCHMARK_MASSES_MEV = [10.0, 100.0, 1000.0]
BENCHMARK_FDM = [0, 2]
DEFAULT_NE_BINS = [1, 2, 3, 4, 5]
```

### Tolerance policy

| Backend | integrate=False | total rate (integrate=True) | per-bin |
|---------|----------------|----------------------------|---------|
| QCDark  | exact (np.array_equal) | 2% | 5% |
| QEDark  | exact | 5% | 5% |
| QCDark2 | N/A — always Simpson | 2% | 5% |
| wimprates | exact | 5% | 5% |

### test_qcdark_physics.py

**Physics invariants** (no upstream, fast):
- `test_linearity_rho_x`: `rate(2ρ_X) == 2 × rate(ρ_X)` exactly
- `test_linearity_sigma_e`: `rate(2σ_e) == 2 × rate(σ_e)` exactly
- `test_band_gap_exact_zero`: `assert np.all(dRdE[Earr < Egap] == 0.0)` — not `allclose`, not `< 1e-10`, exactly zero
- `test_fdm_scaling`: At same (mX, halo), `rate(FDMn=2) / rate(FDMn=0)` matches analytic expectation from q-distribution
- `test_screening_lowers_rate`: `total_rate(DoScreen=True) < total_rate(DoScreen=False)` for Si at 100 MeV
- `test_ionization_sum_rule`: `np.sum(probabilities, axis=0) ≈ 1.0` for E > Egap (Si RK probabilities table)
- `test_ne_sum_equals_spectrum_integral`: `Σ_ne dR/dne ≈ ∫ dR/dE dE` within relative 1%

**Reference parity** (requires upstream, `fix_units` fixture):
- `test_reference_spectrum` — parametrize over `BENCHMARK_MASSES_MEV × BENCHMARK_FDM × ['Si', 'Ge']`
- `test_reference_ne_step` — compare `d_rate_FanoQ()` result
- `test_reference_ne_rk` — Si only, compare `d_rate_RamanathanQ()` result

**Integration determinism and convergence**:
- `test_integrate_false_deterministic`: `np.array_equal(result1, result2)` — call twice
- `test_integrate_true_vs_false_close`: `rtol=0.05` between modes

**Unit invariance**:
- `test_rate_unit_invariant(random_nu_seed)`: result divided by `nu.kg * nu.year` matches across three seeds to `rtol=1e-4`

### test_qedark_physics.py

Mirror structure of `test_qcdark_physics.py` with:
- No `test_reference_ne_rk` (RK not applicable to QEDark)
- Add `test_wk4_not_double_applied`: load raw Si_f2.txt, confirm `dmrates_qedark.ff[i,j]` matches raw value with exactly one wk/4 factor applied
- Reference tolerance: 5% (not 2%)
- Default `integrate=False` in QEDark backend — parity tests must explicitly pass `integrate=True` to get Simpson-vs-Simpson comparison

### test_qcdark2_physics.py

New physics invariants to add (on top of existing `test_qcdark2_reference.py`):
- `test_band_gap_exact_zero_all_materials` — parametrize over all 5 materials
- `test_doscreen_is_noop` — `rate(DoScreen=True) == rate(DoScreen=False)` for QCDark2
- `test_doscreen_raises_warning` — `with pytest.warns(UserWarning):`
- `test_s_matrix_identity` — `S[i,j] == Im(eps[i,j]) / abs(eps[i,j])^2 * q[i]^2 / (2*pi*alpha)` to float32 tolerance
- `test_linearity_rho_x`, `test_linearity_sigma_e`
- `test_ne_sum_rule`
- `test_variant_ordering` — for Si at 100 MeV: `rate(nolfe) ≠ rate(lfe)`, composite in between

New reference parity (extends existing test mass scan):
- `test_mass_scan_ge_composite` — Ge/composite at 4 masses × 2 FDMn
- `test_mass_scan_diamond_composite` — Diamond/composite (band-gap stress test)

### test_wimprates_physics.py

**Physics invariants**:
- `test_binding_energy_vmin_shift`: at fixed (q, E), noble vmin > semiconductor vmin by exactly `Ebind/q`
- `test_shell_threshold_enforced`: `dR/dE[E < Ebind] == 0.0` per shell, exactly
- `test_linearity_rho_x`, `test_ne_sum_rule`

**Reference parity** (requires upstream wimprates):
- `test_per_shell_parity_xe` — each (n,l) shell dR/dE at 100 MeV, FDMn=0, tolerance 5%
- `test_total_rate_parity_xe` — sum over shells
- `test_per_shell_parity_ar` — same for Ar
- `test_csv_reference` — compare DMeRates against `old_for_comparison/wimprates_mod/darkside_100MeV_1e-36_fdm0_*.csv`

**Integration determinism**:
- `test_deterministic`

### test_cross_backend_consistency.py

- `test_si_qcdark_vs_qedark_not_orders_of_magnitude_off` — ratio within [0.1, 10] at 100 MeV
- `test_si_qcdark_vs_qcdark2_within_20pct` — different physics, but same crystal
- `test_cell_mass_consistency` — QCDark HDF5 mCell == QCDark2 HDF5 M_cell for Si and Ge to 0.1%
- `test_fdm_ratio_decreases_with_mass` — `rate(FDMn=2)/rate(FDMn=0)` decreases from 10→1000 MeV for all Si backends (more phase-space at low mX for light mediator)

---

## Physics notes for the implementing agent

1. **`integrate=False` ≠ exact match with upstream.** QCDark `d_rate()` uses `scipy.integrate.simps`. DMeRates `integrate=False` is a Riemann sum on the same q-grid. They differ by O(dq²). Parity tests should use `integrate=True` in DMeRates.

2. **sigma_e units mismatch.** QCDark `default_astro['sigma_e'] = 1e-39 cm²`. DMeRates default is `1e-36 cm²`. Always call `obj.update_params(...)` with the upstream astro dict before parity comparisons.

3. **rhoX units mismatch.** QCDark `default_astro['rhoX'] = 0.3e9` eV/cm³. DMeRates uses numericalunits. When calling `update_params`, pass the value in numericalunits: `0.3e9 * nu.eV / nu.cm**3`.

4. **wimprates halo parameters.** `wr.StandardHaloModel()` defaults differ from DMeRates SHM defaults (v0=220 vs 238 km/s). Inspect the wimprates class attributes and set v0, vEarth, v_esc explicitly to match `Constants.py` values.

5. **Band gap masking is exact zero.** `dR/dE[E < Egap]` must be `0.0`, not small. Test with `assert np.all(...)`, not `np.allclose`. This is a bitwise check.

6. **QCDark2 and DoScreen.** RPA screening is embedded in the dielectric function ε. The `DoScreen` flag for QCDark2 must be a no-op (rate unchanged) AND should emit a `UserWarning`. Both behaviors must be tested separately.

7. **Notebook nu.reset_units('SI') placement.** Call `nu.reset_units('SI')` at the top of each comparison notebook, before any imports of DMeRates modules. The Constants.py randomizes units on import, so resetting after-the-fact may not help.

8. **Form factor grid alignment.** QCDark's `d_rate()` returns `Earr = dE * arange(numE) + dE/2` (bin centers). DMeRates `calculate_spectrum()` returns energy array from `obj.Earr / nu.eV`. These should align — verify they do at the first benchmark point before computing residuals.

9. **wimprates per-shell form factors.** The PKL file indexes shells by name ('4s', '4p', '4d', '5s', '5p'). DMeRates `noble_dRdE()` returns a dict keyed the same way. Match by name, not by list index.

---

## Verification workflow

After implementation:

```bash
# Verify notebooks run without error (kernel must be project venv):
cd /Users/ansh/Local/SENSEI/DarkMatterRates
source .venv/bin/activate
jupyter nbconvert --to notebook --execute tests/compare_qcdark.ipynb
jupyter nbconvert --to notebook --execute tests/compare_qedark.ipynb
jupyter nbconvert --to notebook --execute tests/compare_qcdark2.ipynb
jupyter nbconvert --to notebook --execute tests/compare_wimprates.ipynb

# Run physics invariant tests (fast, no upstream needed):
pytest tests/test_qcdark_physics.py tests/test_qedark_physics.py \
       tests/test_qcdark2_physics.py tests/test_wimprates_physics.py \
       -k "invariant or sum_rule or linearity or band_gap or screening or fdm_ratio or deterministic" -v

# Full reference parity:
pytest tests/test_qcdark_physics.py tests/test_qedark_physics.py \
       tests/test_qcdark2_physics.py tests/test_wimprates_physics.py \
       tests/test_cross_backend_consistency.py -v --tb=short

# QCDark2 parity suite (existing + new):
pytest tests/test_qcdark2_reference.py tests/test_qcdark2_physics.py -v
```

Expected: notebooks produce figure-rich outputs showing overlaid spectra with <5% residuals for QCDark/QCDark2, <10% for QEDark/wimprates. If residuals are large, investigate unit mismatches (rhoX, sigma_e, nu scales) before concluding a physics error.

---

## Build order

1. **Start with conftest.py additions** — needed by all tests
2. **compare_qcdark.ipynb** — simplest backend, establishes the figure template
3. **test_qcdark_physics.py** — parallel with notebook; invariants don't require upstream
4. **compare_wimprates.ipynb** — use existing CSVs in `old_for_comparison/wimprates_mod/` as quick sanity before live-run
5. **test_wimprates_physics.py**
6. **compare_qedark.ipynb** — requires solving wk/4 calibration question first
7. **test_qedark_physics.py**
8. **compare_qcdark2.ipynb** — existing parity tests tell you what to expect
9. **test_qcdark2_physics.py** — add invariants; parity already covered by test_qcdark2_reference.py
10. **test_cross_backend_consistency.py** — last, uses results from all backends
