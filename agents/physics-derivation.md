Role: physics-derivation
Recommended model: Opus 4.7, GPT-5.5 high/xhigh, or o3 for formal reasoning
Owns: Plan Step 3.0
Blocks: Steps 3.1, 3.2, 3.3, 3.4, 3.5, 3.6 (all of Phase 3)
Prerequisites: Phase 0 complete (pytest baselines green), Phase 1 complete (package installable)

---

# physics-derivation — Step 3.0

## Goal

Create `tests/qcdark2_formula_derivation.ipynb`. This notebook is the **primary
reproducibility artifact** for the entire QCDark2 integration. It must:

1. Write the complete QCDark2 differential rate formula with all terms explicit.
2. Track every unit conversion into `numericalunits`.
3. Validate each factor against QCDark2's reference implementation at a fixed test case.
4. Achieve <5% agreement on dR/dE before this step is considered done.
5. Produce 2–3 small pytest reference values for CI regression.

Do **not** move to any Phase 3 implementation step until this notebook exists,
runs top-to-bottom, and contains the final agreement numbers.

---

## Files to Read First

Before writing any notebook cells, read these files in full:

```
/Users/ansh/Local/SENSEI/QCDark2/qcdark2/dark_matter_rates.py  — reference implementation
/Users/ansh/Local/SENSEI/QCDark2/dielectric_functions/composite/Si_comp.h5  — data file
DMeRates/Constants.py     — numericalunits setup and physical constants
tests/conftest.py         — unit-seeding pattern (random.seed(0) before imports)
```

The QCDark2 paper is at `/Users/ansh/Local/SENSEI/QCDark2/2603.12326v1.pdf` if
you can read PDFs; otherwise derive from the code.

---

## Fixed Test Case

All formula validation uses this case:

| Parameter | Value |
|-----------|-------|
| Material | Si (composite dielectric: `Si_comp.h5`) |
| DM mass | mX = 1 GeV = 1e9 eV |
| Mediator | Heavy (F_DM = 1, `FDMn=0` in DMeRates, `mediator='heavy'` in QCDark2) |
| Halo | Maxwell-Boltzmann (`velocity_dist='MB'`, `astro_model=default_astro`) |
| Screening | RPA |
| Reference astro | v0=238 km/s, vEarth=250.2 km/s, vEscape=544 km/s, rhoX=0.3 GeV/cm³ |

Validate at 3–5 energy values: choose E ∈ {1, 5, 10, 20, 50} eV where
dR/dE is well above zero.

---

## QCDark2 Reference Formula

The production path in `dark_matter_rates.py` that you are replicating is `get_dR_dE()`.
Study it carefully before any cell:

```python
# From dark_matter_rates.py (reference only — do NOT import in production)
rho_T = epsilon.M_cell / kg / epsilon.V_cell  # kg/Bohr³; target material density
reduced_mass = m_X * m_e / (m_X + m_e)        # eV; DM-electron reduced mass

prefactor = (1/rho_T) * (rho_X/m_X) * (cross_section/reduced_mass**2) / (4*pi)

integrand = q * F_DM**2 * S(q,E) * eta_MB(q, E, m_X)  # q in α·me
# S(q) = elf(q) * q**2 / (2*pi*alpha)                   # q in α·me atomic units

q_eV = epsilon.q * alpha * m_e    # convert q to eV for the integration measure
dR_dE[i_E] = scipy.integrate.simpson(integrand[:, i_E], q_eV)

dR_dE = prefactor * dR_dE / cm2sec / sec2yr  # → events/kg/year/eV
```

The final `/cm2sec/sec2yr` converts the result from events·cm²·s/kg to
events/kg/year/eV. **Do not drop this factor** — it is easy to miss because it
appears at the very end as a unit rescaling, not as part of the physics.

---

## Unit Conventions in QCDark2 Data

The `Si_comp.h5` file stores:

| Field | Unit | Value range (Si) |
|-------|------|-----------------|
| `epsilon` (complex) | dimensionless | complex dielectric |
| `q` | α·me [atomic momentum units] | ~0.01–25 |
| `E` | eV | ~0.1–50 |
| `M_cell` (attr) | eV | ~5×10¹⁰ eV (Si 2-atom cell rest energy) |
| `V_cell` (attr) | Bohr³ = (α·me)⁻³ in natural units | ~130 Bohr³ for Si |
| `dE` (attr) | eV | 0.1 |

**Critical pre-check before writing any unit conversion:**

Open `Si_comp.h5` with h5py and verify:

1. q range: expect ~0.01–25 for α·me convention, not ~3–13000 for eV/c. Confirm.
2. V_cell numerical value: For crystalline Si with 2 atoms per unit cell,
   V_cell ≈ 130 Bohr³. If you see ~130, the Bohr³ convention is confirmed.
   If you see ~3e-29, it has already been converted to m³.
   Document the confirmed convention as a comment in the notebook before
   writing any conversion code.
3. M_cell: Si 2-atom cell ≈ 28 × 2 × 931.5e6 eV/c² × c² ≈ 5.2×10¹⁰ eV.
   Confirm the order of magnitude matches.

---

## numericalunits Conversion Idioms

Use these patterns in the notebook. Match the DMeRates idiom (divide to express in a unit):

```python
import random
random.seed(0)                    # must precede DMeRates import — fixes unit scales
import numericalunits as nu
from DMeRates.Constants import *  # imports v0, vEarth, vEscape, rhoX, crosssection

# Atomic momentum unit (a.u.) in SI:
q_amu_in_SI = nu.alphaFS * nu.me * nu.c0   # kg·m/s per atomic-momentum-unit

# Convert q from α·me to SI:
q_SI = q_raw * q_amu_in_SI

# Bohr radius in SI:
bohr = nu.hbar / (nu.alphaFS * nu.me * nu.c0)   # meters

# V_cell from Bohr³ to SI (m³):
V_cell_SI = V_cell_bohr * bohr**3

# M_cell from eV to kg:
M_cell_kg = M_cell_eV * nu.eV / nu.c0**2

# rhoX as an energy density in numericalunits:
rhoX_energy_density = 0.3e9 * nu.eV / (nu.cm**3)  # if using QCDark2 default
# Or use DMeRates default: rhoX (already in numericalunits SI)

# sigma_e from cm² to SI:
sigma_e_SI = 1e-39 * nu.cm**2

# Express result in events/kg/year/eV for comparison:
result_in_units = dRdE_SI / (1.0 / (nu.kg * nu.year * nu.eV))
```

Do NOT call `nu.reset_units()` at any point — `Constants.py` has already set
unit scales, and resetting them would break dimensional consistency.

---

## Notebook Structure

Organize the notebook with these sections:

1. **Setup** — imports, random.seed(0), h5py load of Si_comp.h5, pre-check of
   q/V_cell/M_cell ranges.
2. **Formula writeout** — the complete dR/dE formula with citations. Cite
   eq. numbers from the QCDark2 paper where possible.
3. **Term-by-term unit analysis** — for each factor in the prefactor and
   integrand, show its unit in QCDark2's convention and the numericalunits
   conversion.
4. **Reference run** — call QCDark2's `get_dR_dE()` directly (reference only)
   to get the target dR/dE values at the fixed test case.
5. **DMeRates reimplementation** — reimplement the same integral in numpy/torch
   using numericalunits. This is the proto-implementation that Step 3.3's
   engine will later formalize in torch.
6. **Agreement check** — compare at 3–5 energy values; target <5% relative
   disagreement at each.
7. **Pytest reference values** — extract 2–3 dR/dE values (in events/kg/year/eV)
   and display them in a format suitable for copy-paste into `tests/conftest.py`.

---

## Acceptance Criteria

- [ ] Notebook runs top-to-bottom with no errors.
- [ ] q range confirmed as α·me convention (not eV/c).
- [ ] V_cell confirmed as Bohr³ (not pre-converted); numerical value documented.
- [ ] Every numericalunits conversion is explicit — no bare `* 137` or `/ 0.511e6`.
- [ ] dR/dE agreement with QCDark2 reference is <5% at each validation energy.
- [ ] Final agreement numbers are in the notebook (not just "it works").
- [ ] 2–3 reference values extracted and formatted for pytest.
- [ ] No QCDark2 Python imports appear outside of the "Reference run" section.

---

## Hard Invariants

- **QCDark2 Python is reference-only.** Never import `qcdark2` modules in
  `DMeRates/` production code. The notebook's "Reference run" section is the
  only legitimate use of QCDark2 Python.
- **Do not guess units.** If a unit conversion is unclear, open the QCDark2 source
  and trace it explicitly. The pre-check in step 1 resolves ambiguity.
- **Do not proceed to Steps 3.1–3.6** until agreement is <5%. If you cannot achieve
  it, document the discrepancy and stop — do not paper over it.

---

## Handoff

When the notebook is complete, produce:

1. `tests/qcdark2_formula_derivation.ipynb` — the derivation notebook.
2. A short summary (5–10 sentences) of: the fixed test case, the agreement
   achieved, any unit convention that needed a pre-check to resolve, and the
   reference values suitable for adding to `tests/conftest.py`.

The main conversation will use this summary to proceed with Steps 3.2 and 3.3.
