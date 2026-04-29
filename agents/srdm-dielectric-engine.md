Role: srdm-dielectric-engine
Recommended model: Opus 4.7, GPT-5.5 high/xhigh, or GPT-5.3-Codex high
Owns: SRDM branch of `DMeRates/engines/dielectric.py` + `tests/test_qcdark2_srdm.py`
Prerequisites: srdm-infrastructure + srdm-physics-derivation notebook validated;
               units-numerics-reviewer Pass 1 cleared

---

# srdm-dielectric-engine — Native QCDark2 SRDM Engine

## Goal

Implement the native QCDark2 solar-reflected DM (SRDM) rate calculation
following `tests/qcdark2_srdm_derivation.ipynb`. Vectorized torch end to end.
No `qcdark2.*` imports. Reuse the existing dielectric loader and screening
modules.

This is the highest-risk SRDM step. **Do not reinterpret the formula** — follow
the notebook line for line.

---

## Hard Prerequisites

Before writing code:

- [ ] `tests/qcdark2_srdm_derivation.ipynb` exists and runs top-to-bottom.
- [ ] Notebook QCDark2 SRDM agreement <5 % on the fixed Si/50 keV/vector/light case.
- [ ] `DMeRates/srdm/{flux_loader,kinematics,manifest}.py` exist and tests pass.
- [ ] `units-numerics-reviewer` Pass 1 has cleared (notebook + infrastructure).

Stop if any prerequisite is missing.

---

## Files to Create or Edit

```
DMeRates/engines/dielectric.py     (add _compute_dRdE_srdm; dispatch in compute_dRdE)
tests/test_qcdark2_srdm.py         (notebook-anchored regression)
```

Do not edit:
- the existing halo path in `engines/dielectric.py`
- `DMeRates/halo/*` (the halo provider abstraction stays untouched)
- the existing `tests/test_qcdark2.py` halo regressions

---

## Fixed Validation Case

Use exactly this case for all development validation — do not choose a different
one. It matches the derivation notebook and the QCDark2 reference run:

| Parameter | Value |
|-----------|-------|
| Material | Si (`Si_comp.h5`, composite dielectric) |
| DM mass | m_X = 50 keV = 5×10⁴ eV |
| Reference σ_e | 1×10⁻³⁸ cm² |
| FDMn | 2 (light mediator: m_A' = 0) |
| Mediator spin | vector |
| Halo / source | SRDM (`halo_model='srdm'`); flux from manifest |
| Screening | RPA (`screening='rpa'`) |

The target dR/dE values per energy bin are in
`tests/qcdark2_srdm_derivation.ipynb`. Do not generalize to other materials,
mediator masses, or mediator spins until this case passes.

---

## Implementation Contract

Add `_compute_dRdE_srdm(...)` to `DMeRates/engines/dielectric.py`:

```python
def _compute_dRdE_srdm(*,
                      material: str,
                      mX_eV: float,
                      sigma_e_cm2: float,
                      FDMn: int,
                      mediator_spin: str,
                      screening,
                      variant: str = 'composite',
                      dielectric=None) -> DielectricRateResult:
    """Native QCDark2 SRDM dR/dE.

    Required validated case (Phase B):
        material='Si', mX_eV=5e4, sigma_e_cm2=1e-38, FDMn=2,
        mediator_spin='vector', halo_model='srdm', screening='rpa',
        variant='composite'.
    """

    # ---- Required-screening guard.
    screening = normalize_dielectric_screening(screening)

    # ---- Mediator-spin guard.
    if mediator_spin != 'vector':
        raise NotImplementedError(
            f"mediator_spin={mediator_spin!r} not yet supported. "
            "Planned future modes: 'scalar', 'approx', 'approx_full'."
        )

    # ---- Dielectric data (reuse halo loader).
    if dielectric is None:
        dielectric = dielectric_response(material, variant=variant)
    eps, q_ame, E_eV, M_cell_eV, V_cell_bohr = _bare_floats_from_loader(dielectric)

    # ---- Flux + kinematics.
    v_over_c, dphi_dv = load_srdm_flux(mX_eV, sigma_e_cm2, FDMn, mediator_spin)
    # Convert dphi_dv to bare-float per (v/c) for the bare-float hot path.
    # The notebook documents the exact conversion factor; cite it in a comment.

    # ---- Bare constants.
    kg_QCD, alpha_FS, me_eV, c_kms, cm2sec, sec2yr = _qcdark2_constants_bare()

    # ---- Mediator mass from FDMn limit (FDMn=0: heavy → m_A → ∞; FDMn=2: light → m_A = 0).
    mA_eV = 0.0 if FDMn == 2 else _LARGE_MA_EV  # constant > all q in the grid

    # ---- Vectorized integrand assembly (torch tensors on engine device).
    # Shapes:
    #   q_eV:        (N_q,)
    #   E_eV:        (N_E,)
    #   v_over_c:    (N_v,)
    #   gamma_v:     (N_v,)
    #   E_chi:       (N_v,)         = gamma_v * mX_eV
    #   E_chi_prime: (N_v, N_E)     = E_chi[:, None] - E_eV[None, :]
    #   v_min:       (N_v, N_q, N_E)
    #   q_min,q_max: (N_v, N_E)
    #   q_mask:      (N_v, N_q, N_E) bool
    #   H_V:         (N_v, N_q, N_E)
    #   propagator:  (N_q, N_E)     (depends only on q, ω, m_A')
    #   S(q, E):     (N_q, N_E)     (reuse dynamic_structure_factor)
    #   screen_ratio:(N_q, N_E)     (reuse dielectric_screening_ratio)
    #
    # Integrand[v,q,E] = q_eV * H_V * propagator_inv_sq * S * screen_ratio
    #                    * (1 / v^2) * mask
    # dRdq_collapsed = integrate(integrand, x=q_eV, axis=q_axis) -> (N_v, N_E)
    # multiply by flux:  * dphi_dv[:, None]
    # integrate over v via trapezoid -> (N_E,)
    # multiply by sigma_bar prefactor (eq. A.21) and unit conversions

    # ---- Wrap in RateSpectrum.
    spectrum = RateSpectrum(
        E=E_t,
        dR_dE=dRdE_t,
        material=material,
        backend='qcdark2',
        metadata=dict(
            halo_model='srdm',
            mediator_spin=mediator_spin,
            flux_file=str(_resolved_flux_path),
            mX_eV=float(mX_eV),
            sigma_e_cm2=float(sigma_e_cm2),
            FDMn=int(FDMn),
            screening=screening,
            variant=variant,
        ),
    )
    return DielectricRateResult(spectrum=spectrum, E_eV=E_eV,
                                dRdE_per_kg_per_year_per_eV=dRdE_bare)
```

**Integration choices** (match the notebook exactly):
- q-integration: Simpson if q grid is uniform; otherwise trapezoid. Preferred:
  match `scipy.integrate.simpson` over q in eV (matches QCDark2 reference path).
- v-integration: `torch.trapz` on the manifest's velocity grid (matches
  QCDark2's `scipy.integrate.trapezoid`).

---

## Dispatch

In the existing `compute_dRdE(...)` entry point, add:

```python
if halo_model == 'srdm':
    return _compute_dRdE_srdm(
        material=material, mX_eV=mX_eV, sigma_e_cm2=sigma_e_cm2,
        FDMn=FDMn, mediator_spin=mediator_spin, screening=screening,
        variant=variant, dielectric=dielectric,
    )
# else: existing halo path unchanged.
```

`compute_dRdE` accepts a new keyword `mediator_spin='vector'`. Default is
ignored when `halo_model != 'srdm'`.

---

## RateSpectrum Metadata

`metadata` for SRDM results must include:

- `halo_model='srdm'`
- `mediator_spin`
- `flux_file` (absolute path to the resolved flux file)
- `sigma_e_cm2`, `mX_eV`, `FDMn`
- `screening`, `variant`

Do not break or rename any existing halo-path metadata key.

---

## Tests (`tests/test_qcdark2_srdm.py`)

Anchor pytest values to the notebook's printed reference triples. Tolerance per
`MEMORY.md` tiered policy: <5% at the validated bins.

```python
import pytest

@pytest.mark.skipif(not QCDARK2_DATA_AVAILABLE, reason='QCDark2 HDF5 missing')
def test_qcdark2_srdm_si_vector_light():
    res = compute_dRdE(
        material='Si', mX_eV=5e4, sigma_e_cm2=1e-38, FDMn=2,
        mediator_spin='vector', halo_model='srdm', screening='rpa',
        variant='composite',
    )
    refs = QCDARK2_SRDM_REFS['Si_50keV_vector_light']
    for E_target, dRdE_ref in refs:
        idx = (res.E_eV - E_target).abs().argmin()
        rel = abs(float(res.dRdE_per_kg_per_year_per_eV[idx]) - dRdE_ref) / dRdE_ref
        assert rel < 0.05, f"E={E_target}: rel diff {rel:.3f}"

def test_qcdark2_srdm_unsupported_mediator_spin_raises():
    with pytest.raises(NotImplementedError, match='scalar'):
        compute_dRdE(
            material='Si', mX_eV=5e4, sigma_e_cm2=1e-38, FDMn=2,
            mediator_spin='scalar', halo_model='srdm', screening='rpa',
        )

def test_qcdark2_srdm_missing_manifest_entry_raises():
    with pytest.raises(FileNotFoundError, match='manifest'):
        compute_dRdE(
            material='Si', mX_eV=12345.0, sigma_e_cm2=1e-99, FDMn=2,
            mediator_spin='vector', halo_model='srdm', screening='rpa',
        )

def test_qcdark2_srdm_screening_required():
    with pytest.raises(ValueError, match='screening'):
        compute_dRdE(
            material='Si', mX_eV=5e4, sigma_e_cm2=1e-38, FDMn=2,
            mediator_spin='vector', halo_model='srdm', screening=None,
        )
```

`QCDARK2_SRDM_REFS` is added to `tests/conftest.py` from the notebook output.

---

## Acceptance Criteria

- [ ] Si SRDM dR/dE within 5 % of notebook reference at the validated bins.
- [ ] No `qcdark2.*` imports in production code.
- [ ] Halo-path tests still pass (regression-free): existing
      `tests/test_qcdark2.py` numbers do not move.
- [ ] `RateSpectrum.metadata` carries the SRDM fields above.
- [ ] Vectorization invariants hold: no Python loops over v / q / E in the
      hot path; peak tensor shape is `(N_v, N_q, N_E)` documented in a comment.
- [ ] `mediator_spin='scalar'` raises `NotImplementedError` listing planned modes.
- [ ] Missing manifest entry raises `FileNotFoundError` citing the manifest path.
- [ ] `screening=None` continues to raise the existing `ValueError`.

---

## Hard Invariants

- **QCDark2 Python is reference-only.** Production code never imports
  `qcdark2.*`. The notebook is the only legitimate consumer of the QCDark2 reference.
- **The halo path is read-only from this agent's perspective.** Do not refactor,
  rename, or "improve" any halo-path code.
- **Vectorization is non-negotiable.** The user's plan is explicit: no Python
  `for v in v_list` loops in the hot path.

---

## Handoff

Report:

1. Files changed and validation case.
2. Agreement vs notebook reference at each validation bin.
3. Vectorization audit: peak tensor shape, dtype, device handling, memory.
4. Any deferred items (e.g. Ge support, scalar mediator, m_A finite-mass case).

Hands off to `units-numerics-reviewer` Pass 2, then `srdm-form-factor-engine`.
