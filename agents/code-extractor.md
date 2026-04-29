Role: code-extractor
Recommended model: Sonnet 4.6 or GPT-5.3-Codex high
Owns: Steps 2.1, 2.2, 2.3, 2.4, 2.5 (Phase 2) and Steps 3.1, 3.4, 3.5, 3.6 (Phase 3 wiring)
Prerequisites: Phase 0 complete (baselines green); Phase 1 complete (package installable)
Phase 3 prerequisites: Step 3.0 derivation notebook complete and validated (<5% agreement)

---

# code-extractor — Phases 2 and 3 (partial)

## What This Agent Does

This agent extracts the monolithic `DMeRate.py` into a modular structure without
changing any physics. Phase 3 wiring is only done after the physics-derivation agent
completes Step 3.0.

Steps **3.2** (dynamic structure factor utilities) and **3.3** (native dielectric
rate engine) are **not** in scope for this agent — they require the derivation
notebook's context and are done in the main conversation.

---

## Core Invariants — Never Violate These

1. **Run `pytest tests/ -v` after every sub-step.** If tests fail, fix before
   continuing. Do not accumulate failures.
2. **Run the modulation notebook at phase boundaries and after halo/API changes:**
   ```bash
   source .venv/bin/activate
   jupyter nbconvert --to notebook --execute \
       modulation_study/modulation_figures.ipynb \
       --output /tmp/modulation_exec.ipynb \
       --ExecutePreprocessor.timeout=600
   ```
   Pytest runs after every sub-step. The full notebook runs after Phase 2, after
   Step 2.4 halo extraction, after any public API/signature change, and after
   Phase 3 wiring. If it fails, fix before continuing. Pytest alone will miss
   notebook-path regressions, but executing the full notebook after every small
   file move is usually too brittle and slow.
3. **Option A (duplicate-then-delete).** The monolithic `DMeRate.py` logic stays
   fully intact and functional through all of Phase 2. New engines live in parallel
   under `DMeRates/engines/`. Do not make `DMeRate.py` delegate to new engines until
   Phase 2 baselines pass on all call paths, including modulated halos and
   halo-independent η. Only after that does `DMeRate.py` begin thinning.
4. **Do not rename any method or change any signature** used by `DMeRates_Examples.ipynb`
   or any notebook under `modulation_study/`. Any backward-incompatible change requires
   a compatibility shim.
5. **Do not refactor beyond extraction scope.** If surrounding code looks awkward,
   leave it. A bug fix requires a separate decision; this agent's job is faithful
   extraction.
6. **`_LEGACY_QEDARK_ENERGY_NORM`** — when extracting the non-integrated semiconductor
   path, replace the bare `* 10` with this constant. See below.

---

## Environment Setup

```bash
cd /Users/ansh/Local/SENSEI/DarkMatterRates
source .venv/bin/activate
```

The `torchinterp1d/` directory is a git submodule. If it is empty:
```bash
git submodule update --init
```

---

## Phase 2 — Extract Existing Physics Paths

### Step 2.1 — Extract response loaders

Source: `DMeRates/form_factor.py`

Create these files:
- `DMeRates/responses/__init__.py`
- `DMeRates/responses/qcdark1.py` — the `form_factor` class (HDF5 loader for Si/Ge)
- `DMeRates/responses/qedark.py` — the `form_factorQEDark` class (txt loader for Si/Ge)
- `DMeRates/responses/noble_gas.py` — the `formFactorNoble` class (pkl loader for Xe/Ar)

Add backward-compatible re-exports to `DMeRates/form_factor.py`:
```python
from DMeRates.responses.qcdark1 import form_factor
from DMeRates.responses.qedark import form_factorQEDark
from DMeRates.responses.noble_gas import formFactorNoble
```

Verification: `pytest tests/ -v` passes. Modulation notebook executes.

### Step 2.2 — Extract ionization/yield models

Source: RK probability and step-function logic in `DMeRates/DMeRate.py`
(methods: `RKProbabilities`, `change_to_step`, related helpers)

Create:
- `DMeRates/ionization/__init__.py`
- `DMeRates/ionization/rk_probabilities.py` — Si Ramanathan-Kurinsky interpolated model
- `DMeRates/ionization/step_function.py` — legacy/configurable step approximation

**Policy (must preserve):**
- Si QCDark1 defaults to RK probabilities.
- Ge QCDark1 uses the step-function approximation.
- QEDark legacy path remains reproducible, including `change_to_step()`.
- The extracted functions must accept explicit energy arrays, material metadata, and
  device/dtype rather than reading hidden state from `self`.

Keep `DMeRate.py` calling its own methods (monolith intact). Add thin wrappers
in `DMeRate.py` that delegate to the new functions — only if this is cleaner than
direct extraction. If in doubt, keep the monolith calls unchanged.

Verification: `pytest tests/ -v` passes. Modulation notebook executes.

### Step 2.3 — Extract screening

Source: Thomas-Fermi screening logic in `DMeRates/DMeRate.py`

Create:
- `DMeRates/screening/__init__.py`
- `DMeRates/screening/thomas_fermi.py` — QCDark1 modified Thomas-Fermi screening

Keep compatibility wrappers for any method names currently called by notebooks
(`DoScreen` parameter pathway must be preserved).

Verification: Screened and unscreened QCDark1 rates match baselines within 2%.

### Step 2.4 — Extract halo providers

Source: `DMeRates/DM_Halo.py` and halo-file portions of `DMeRate.setup_halo_data()`

Create:
- `DMeRates/halo/__init__.py`
- `DMeRates/halo/analytic.py` — SHM, Tsallis, DPL, and in-memory MB tensor path
- `DMeRates/halo/file_loader.py` — SHM text files and DaMaSCUS/Verne files
- `DMeRates/halo/independent.py` — halo-independent step eta

**Common interface** for all providers:
```python
eta(v_min_tensor) -> torch.Tensor
```

**Halo model string keys** (these must keep working without change):
- `'shm'`, `'tsa'`, `'dpl'` — analytic computation or file lookup
- `'modulated'`, `'summer'` — DaMaSCUS/Verne files indexed by `isoangle` (0–35, 5° steps)
- `'imb'` — in-memory Maxwell-Boltzmann tensor

Verification: Analytic, file-backed, modulated, and halo-independent paths all match
their baseline rates. Modulation notebook executes end-to-end.

### Step 2.5 — Extract rate engines

Source: `vectorized_dRdE`, `noble_dRdE`, `rate_dme_shell` in `DMeRates/DMeRate.py`

Create:
- `DMeRates/engines/__init__.py`
- `DMeRates/engines/form_factor.py` — QEDark/QCDark1 semiconductor rate engine
- `DMeRates/engines/noble_gas.py` — Xe/Ar shell rate engine

Engines must return `RateSpectrum` objects (from `DMeRates/spectrum.py`, created in Step 1.4).

**Critical — legacy energy normalization:**
In the non-integrated semiconductor path, the current code has `* self.form_factor.dE * 10`.
When extracting, replace this with:
```python
_LEGACY_QEDARK_ENERGY_NORM = 1.0 * nu.eV
```
and use `* _LEGACY_QEDARK_ENERGY_NORM` in place of `* self.form_factor.dE * 10`.
Add a comment citing `tests/phase0_2.md`:
```python
# Legacy QEDark convention: grid values summed directly without dE bin-width factor.
# Equivalent to dE * 10 for the current 0.1 eV grid. See tests/phase0_2.md.
```
Do not change the numerical behavior — this is documentation only.

**Compatibility:**
- `DMeRate.calculate_rates(...)` remains the public API throughout Phase 2.
- A new `RateCalculator` may be stubbed but must not replace the legacy class.
- Legacy argument names (`FDMn`, `DoScreen`, `integrate`, `isoangle`, `useVerne`,
  `calcErrors`) continue to work unchanged.

Verification: All existing baselines pass within physics-level tolerances (2% for
integrate=True, exact for integrate=False). Modulation notebook executes.

---

## Phase 3 Wiring — QCDark2 Integration

**STOP HERE if `tests/qcdark2_formula_derivation.ipynb` is not complete and validated.**
Steps 3.1 and later depend on the formula and unit conventions established in Step 3.0.
Check: does the notebook exist, run top-to-bottom, and show <5% agreement on dR/dE?
If not, wait for the physics-derivation agent to finish.

**Not in scope:** Steps 3.2 (DSF utilities) and 3.3 (dielectric rate engine) —
these are done in the main conversation after Step 3.0.

### Step 3.1 — QCDark2 dielectric response loader

Create `DMeRates/responses/dielectric.py`.

**Pre-check (do first, before any unit conversion code):**
```python
import h5py
with h5py.File('/Users/ansh/Local/SENSEI/QCDark2/dielectric_functions/composite/Si_comp.h5', 'r') as f:
    q = f['q'][:]
    V_cell = f.attrs['V_cell']
    M_cell = f.attrs['M_cell']
    print(f"q range: {q.min():.3f} – {q.max():.3f}")
    print(f"V_cell: {V_cell:.2f}")
    print(f"M_cell: {M_cell:.3e}")
```
Expected: q range ~0.01–25 (α·me units), V_cell ~130 (Bohr³), M_cell ~5e10 (eV).
Document the confirmed convention as a comment at the top of the loader.

The loader must use `DataRegistry` (Step 1.2) for file paths.

**Fields to load:**
- `epsilon` — complex dielectric function, shape (N_q, N_E)
- `q` — momentum in α·me units (raw); keep as `q_ame`
- `E` — energy in eV
- `M_cell` (attr) — cell mass in eV
- `V_cell` (attr) — cell volume in Bohr³
- `dE` (attr) — energy spacing in eV

**Unit conversions to apply at load time:**
```python
import numericalunits as nu

# Atomic momentum unit in SI
_q_amu = nu.alphaFS * nu.me * nu.c0

# Bohr radius in SI
_bohr = nu.hbar / (nu.alphaFS * nu.me * nu.c0)

# Applied to loaded data:
q_ame = h5['q'][:]                    # raw, dimensionless (α·me units)
q     = q_ame * _q_amu                # momentum in SI (nu units)
E     = h5['E'][:] * nu.eV
M_cell = h5.attrs['M_cell'] * nu.eV  # in energy units; divide by c² for mass
V_cell_bohr = h5.attrs['V_cell']
V_cell = V_cell_bohr * _bohr**3      # in SI volume (m³)
dE     = float(h5.attrs['dE']) * nu.eV
```

Keep `q_ame` available as a separate attribute alongside `q` — the dielectric formula
uses `q` in atomic units in some intermediate quantities (e.g., S(q)).

**Supported variants:** `'composite'`, `'lfe'`, `'nolfe'`

**Material keys:** `Si`, `Ge`, `GaAs`, `SiC`, `diamond` (lowercase for Diamond)

Verification: Loader returns correct numerical ranges after unit conversion. `pytest tests/` passes.

### Step 3.4 — QCDark2 screening selection

Create `DMeRates/screening/dielectric.py`.

**Initial modes:**
- `screening='rpa'` — use the loaded dielectric response as the screening dielectric.
- `screening='none'` — no screening (epsilon_screen = 1 everywhere).

QCDark2 calculations must raise a clear error if `screening` is not specified:
```python
if screening is None:
    raise ValueError(
        "QCDark2 calculations require an explicit screening choice. "
        "Pass screening='rpa' or screening='none'."
    )
```

Analytic screening models (Thomas-Fermi, Lindhard) may be added later only when
their material coverage and units are explicit.

### Step 3.5 — QCDark2 material metadata and yield policy

Add QCDark2 material metadata to `DMeRates/config.py` or a new
`DMeRates/responses/dielectric_materials.py`.

**Scissor-corrected bandgaps (from QCDark2 paper Table 1):**
```python
QCDARK2_BANDGAPS = {
    'Si':      1.1  * nu.eV,
    'Ge':      0.67 * nu.eV,
    'GaAs':    1.42 * nu.eV,
    'SiC':     2.36 * nu.eV,
    'Diamond': 5.5  * nu.eV,
}
```

**Yield policy (must enforce at call time):**
- `dR/dE` spectra and above-threshold rates are available for all QCDark2 materials.
- Si QCDark2 may use the existing RK yield model for `ne` rates.
- For GaAs, SiC, and Diamond, `ne` step-model rates require an explicit pair energy
  from the caller/config. If omitted, raise:
  ```python
  raise ValueError(
      f"QCDark2 ne rates for {material} require an explicit pair_energy (eV). "
      "The QCDark2 paper does not provide validated pair energies for this material."
  )
  ```
- Do not invent or default pair energies for GaAs, SiC, or Diamond.

### Step 3.6 — Wire QCDark2 into public APIs

Add QCDark2 selection to `DMeRate` and the new `RateCalculator`:

```python
# Existing compatibility class — new constructor option:
DMeRate("Si",    form_factor_type="qcdark2")
DMeRate("GaAs",  form_factor_type="qcdark2")

# New calculator (may be a stub that calls the same engine):
RateCalculator("Si",      backend="qcdark2", variant="composite", screening="rpa")
RateCalculator("Diamond", backend="qcdark2", variant="lfe",       screening="none")
```

**Requirement:** Legacy QEDark/QCDark1 constructor behavior is unchanged. Any call
path that worked before Step 3.6 must still work identically after it.

Verification: `pytest tests/ -v` passes. QCDark2 can calculate a rate for Si with
both constructor forms. Legacy paths are unaffected.

---

## After Each Step — Checklist

```
[ ] pytest tests/ -v — all pass
[ ] modulation_study/modulation_figures.ipynb executes end-to-end when required
    by the notebook checkpoint policy above
[ ] No method signatures used by DMeRates_Examples.ipynb or modulation_study/ were changed
[ ] If any test failed, it was fixed before moving to the next step
```
