Role: srdm-physics-derivation
Recommended model: Opus 4.7, GPT-5.5 high/xhigh, or o3 for formal reasoning
Owns: tests/qcdark2_srdm_derivation.ipynb
Blocks: srdm-dielectric-engine, srdm-form-factor-engine
Prerequisites: srdm-infrastructure complete; QCDark2 halo derivation notebook validated

---

# srdm-physics-derivation — SRDM Notebook

## Goal

Create `tests/qcdark2_srdm_derivation.ipynb` — the **source-of-truth derivation
artifact** for solar-reflected dark matter (SRDM) in DMeRates, mirroring the
existing `tests/qcdark2_formula_derivation.ipynb` for halo DM.

The notebook is the primary reproducibility artifact for the entire SRDM
implementation. It must:

1. Write the unified relativistic dR/dω formula from the QCDark2 paper
   (`/Users/ansh/Local/SENSEI/QCDark2/2603.12326v1.pdf`) §2.2 + Appendix A
   (eq. 2.2, 2.5–2.10, A.17–A.21), with full unit tracking via `numericalunits`.
2. Reproduce QCDark2's `dsigma_rel2` term-by-term and confirm match to <1e-3
   relative on a single (Si, m_X = 50 keV, σ_e = 1e-38, FDMn=2, mediator='vector')
   point.
3. Reproduce QCDark2's `get_rate_flux` end-to-end on the same point using the
   bundled flux file. Target <5 % agreement on dR/dE across the energy bins
   where flux > 0.
4. Add a "form-factor representation" cell block deriving the f²(q,ω) → S(ω,q)
   conversion (DarkELF eq. 16 / QCDark2 paper eq. 2.7 inverted) and verifying
   that the SRDM rate computed from QCDark1 and QEDark form factors is finite,
   positive, and within one order of magnitude of the QCDark2 SRDM result on
   the same kinematic case. **Per-engine percent-level agreement is not
   expected** — engines validate against themselves only.
5. Include a halo-limit sanity check: substitute a narrow Gaussian flux peaked
   at v = v_0 into the SRDM engine (dielectric path) and confirm the resulting
   dR/dE matches the existing halo dR/dE for the same m_X up to the documented
   relativistic correction (γ ≈ 1 + v_0²/2 ≈ 1.0000003 at v_0 = 238 km/s).
6. Capture 3 reference (E, dR/dE) values per engine (QCDark2 + QCDark1 + QEDark)
   in copy-pasteable form for `tests/conftest.py`.

Do not move to any SRDM engine implementation step until this notebook exists,
runs top-to-bottom, and contains the final agreement numbers.

---

## Files to Read First

Before writing any notebook cells, read in full:

```
/Users/ansh/Local/SENSEI/QCDark2/2603.12326v1.pdf  — Section 2.2 + Appendix A
/Users/ansh/Local/SENSEI/QCDark2/qcdark2/dark_matter_rates.py
    — dsigma_rel2 (lines 285–340) + get_rate_flux (lines 371–401) reference
/Users/ansh/Local/SENSEI/QCDark2/2404.10066v1.pdf  — flux derivation (background)
tests/qcdark2_formula_derivation.ipynb              — pattern to mirror
DMeRates/engines/dielectric.py                       — halo-path conventions
DMeRates/srdm/kinematics.py                          — bare-float kinematics
DMeRates/srdm/flux_loader.py                         — flux loader contract
DMeRates/Constants.py                                — numericalunits setup
tests/conftest.py                                    — unit-seeding pattern
```

---

## Fixed Test Case

All formula validation uses this case. **Do not pick a different one.**

| Parameter | Value |
|-----------|-------|
| Material | Si (composite dielectric: `Si_comp.h5`) |
| DM mass | m_X = 50 keV = 5×10⁴ eV |
| Reference σ_e | 1×10⁻³⁸ cm² |
| Mediator mass | m_A' = 0 (FDMn=2, light) |
| Mediator spin | vector |
| Halo / source | SRDM, flux from `halo_data/srdm/srdm_dphidv_mX5e4_sigma1e-38_FDMn2_medvector.txt` |
| Screening | RPA |

For the form-factor cells:
- QCDark1: `form_factors/QCDark/Si_final.hdf5`. Run with and without
  Thomas-Fermi screening to confirm the existing screening factor still
  composes correctly.
- QEDark: `form_factors/QEDark/Si_f2.txt`. Run unscreened (`DoScreen=False`).

Validate at 3–5 energy values where dR/dE is well above zero.

---

## QCDark2 Reference Formula

The production path you are replicating is `get_rate_flux` (with
`dsigma_rel2`). Study these carefully before any cell:

```python
# From dark_matter_rates.py (reference only — never imported in production):

def dsigma_rel2(epsilon, v, sigma_e, m_X, m_A, mediator='vector', screening='RPA'):
    # gamma = 1 / sqrt(1 - v^2);   E_X = gamma * m_X
    # mediator='vector' integrand uses H_V = (E_chi + E_chi')^2 - q^2
    # propagator denominator: (omega^2 - q^2 - mA^2)^2
    # q-bounds depend on v and E (eq. A.19)
    # prefactor = sigma_e / (32 pi^2 alpha v^2 E_X) * (mA^2 + (alpha me)^2)^2 / mu^2 / n
    ...

def get_rate_flux(epsilon, m_X, sigma_e, flux, v_list, m_A=0, mediator='vector', screening='RPA'):
    # for each v in v_list: dsigma[i] = dsigma_rel2(...)
    # integrand = dsigma * flux[:, None]
    # dR[i_E] = trapezoid(integrand[:, i_E], v_list)
    # final units conversion: * kg / sec2yr  -> events / kg / year / eV
    ...
```

The final `* kg / sec2yr` converts from events·cm²·s/atom·... to events/kg/year/eV.
**Do not drop this factor.**

---

## numericalunits Idioms

Same as `qcdark2_formula_derivation.ipynb`:

```python
import random
random.seed(0)                     # must precede DMeRates import
import numericalunits as nu
from DMeRates.Constants import *

# Velocity from km/s to v/c:
v_over_c = v_kms * (nu.km / nu.s) / nu.c0

# dPhi/dv from cm^-2 s^-1 / (km/s) to numericalunits SI:
dphi_dv_si = dphi_dv_per_kms * 1.0 / (nu.cm**2 * nu.s * (nu.km / nu.s))

# Express dR/dE as events / kg / year / eV:
dRdE_in_target = dRdE_SI / (1.0 / (nu.kg * nu.year * nu.eV))
```

Do NOT call `nu.reset_units()` at any point — `Constants.py` has already set
unit scales.

---

## Notebook Structure

Organize the notebook with these sections:

1. **Setup** — imports, `random.seed(0)`, h5py load of `Si_comp.h5`, flux loader
   call. Pre-check: v range (~0.001–0.05 v/c at 50 keV); dΦ/dv peak (~10⁻⁴–10⁻³).
2. **Formula writeout** — eq. 2.8 + 2.9 + Appendix A.21 with citations. Make γ,
   v_min, q_min, q_max, H_V, propagator factor explicit. State which version of
   the halo formula it reduces to in the v ≪ 1 limit (eq. 2.11).
3. **Term-by-term unit analysis** — for each factor in the prefactor and
   integrand, show its unit in QCDark2's bare-float convention and the
   numericalunits conversion.
4. **QCDark2 reference cell block** — call `dsigma_rel2(...)` + `get_rate_flux(...)`
   directly to obtain the target dR/dE array. **This is the only place QCDark2
   Python is allowed.**
5. **DMeRates dielectric reimplementation** — vectorized torch reproduction
   using `DMeRates/srdm/kinematics.py`. Compare against reference; target <5%
   relative agreement at validation energies.
6. **DMeRates form-factor reimplementation** — derive
   `S(ω,q) = 8π²(αm_e)²/V_cell · f²(q,ω)/q³` (DarkELF eq. 16 / QCDark2 eq. 2.7
   inverted); plug into the same SRDM rate kernel; produce QCDark1 + QEDark
   dR/dE for the same kinematic point. Confirm finite, positive, within one
   order of magnitude of the QCDark2 result.
7. **Halo-limit sanity check** — narrow Gaussian flux at v_0; agreement vs the
   existing halo path within the documented relativistic correction.
8. **Pytest reference values** — print 3 reference (E, dR/dE) tuples per engine
   in copy-pasteable form for `tests/conftest.py`.

---

## Acceptance Criteria

- [ ] Notebook runs top-to-bottom, no errors.
- [ ] QCDark2 SRDM dielectric agreement <5 % vs `get_rate_flux` reference at all validation energies.
- [ ] `dsigma_rel2` term-by-term reproduction agreement <1e-3 relative.
- [ ] QCDark1 + QEDark SRDM cells produce finite, positive dR/dE within one order of magnitude of the QCDark2 result.
- [ ] Halo-limit sanity check passes (Gaussian-flux SRDM dR/dE agrees with halo dR/dE within the documented γ ≈ 1 + v²/2 correction).
- [ ] 3 reference (E, dR/dE) values per engine printed in pytest-ready form.
- [ ] No reference to QCDark2 Python outside the explicit "Reference" cell block.

---

## Hard Invariants

- **QCDark2 Python imports appear only inside the labelled "Reference" cells.**
  Do not propagate `qcdark2.*` calls anywhere else in the notebook.
- **Do not change physics if the result disagrees** — diagnose the unit/factor
  and document. If you cannot reach <5 % agreement, stop and report — do not
  paper over it.
- **Do not generalize to scalar / approx mediators** in this notebook. Only
  `mediator_spin='vector'` is in scope here.
- **Do not change halo-path code.** This notebook is read-only with respect to
  the existing halo engines; any halo path it touches is for sanity-check
  comparison only.

---

## Handoff

When the notebook is complete, produce a 8–12 sentence summary covering:

1. Final agreement numbers per engine (QCDark2 dielectric, QCDark1, QEDark).
2. The 3 reference (E, dR/dE) values per engine.
3. Any unit subtlety encountered (especially the dΦ/dv conversion and the f² → S
   conversion).
4. Halo-limit sanity-check result (relative deviation in ppm).
5. Anything the units-numerics-reviewer should look at on Pass 1.

The main conversation hands these to `srdm-dielectric-engine` (and later
`srdm-form-factor-engine`).
