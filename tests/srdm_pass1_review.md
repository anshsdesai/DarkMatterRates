=== Units/Numerics Review — SRDM Pass 1 ===
Artifact reviewed: tests/qcdark2_srdm_derivation.ipynb, DMeRates/srdm/*, DMeRates/data/registry.py, halo_data/srdm/manifest.json
Date: 2026-04-26

Blockers:
- mediator_spin='scalar'/'approx'/'approx_full' does not raise NotImplementedError.
  Current path: manifest.find_entry returns None -> flux_loader raises FileNotFoundError
  with "No SRDM flux file registered" message. Per checklist, unsupported mediator_spin
  values must raise NotImplementedError listing planned future modes BEFORE the manifest
  lookup. This is a contract the engine will rely on.
  File: DMeRates/srdm/flux_loader.py:33 (add guard before find_entry call)

Warnings:
- Propagator pole (omega^2 = q^2 + mA^2) for mA=0 is on-grid at (q[0]=37.3 eV,
  E=37.3 eV) where denom ~ 0.79 eV^2. Safety is provided by kinematic q_bounds
  excluding that region (verified: at v_max=0.165c, q_min=229 eV >> q[0]), but
  mediator_propagator_inv_sq has no regulator and no comment documenting that
  q_bounds is the pole-avoidance mechanism.
  File: DMeRates/srdm/kinematics.py:162-178
  Recommend: one-line comment noting that q_mask/q_bounds excludes the singularity.

- Peak integrand tensor shape at benchmark: (N_v=299, N_q=1251, N_E=501) ~ 1.5 GB
  float64. Fits on workstation RAM but will need v-axis chunking in the production
  engine if larger grids or batch mass sweeps are used. Note for Pass 2 / engine work.

- DMeRates/srdm/kinematics.py:17 and :76 preserve input dtype. Current loader and
  notebook pass float64, but the engine should enforce float64 at its public boundary
  before calling gamma/q_bounds to prevent silent float32 precision loss.

- halo_data/srdm/manifest.json uses actual upstream grid points as lookup keys.
  Nominal (50 keV, 1e-38) does not match. This is correctly documented in
  test_srdm_infrastructure.py:121-133, but the public API wiring should either
  expose nominal aliases or keep this grid-only behaviour explicit.

Looks good:
- gamma(v) in float64 is never NaN at v << 1. Verified: gamma(0)=1.0 exactly,
  gamma(1e-20)=1.0, no NaN or Inf at any tested value. (kinematics.py:17-27,
  test_srdm_infrastructure.py:TestGamma)

- v_min(q, omega, mX, gamma) reduces to halo v_min = q/(2mX) + omega/q in the
  gamma -> 1 limit. Tested at v=1e-4 with rel_diff < 1e-6.
  (kinematics.py:29-48, test_srdm_infrastructure.py:TestVmin)

- q_bounds returns q_min <= q_max for valid kinematics; degenerate cases
  ((gamma*mX - omega)^2 < mX^2) produce q_max < q_min -> empty q_mask, no NaN.
  (kinematics.py:51-96, test_srdm_infrastructure.py:TestQBounds)

- H_V reduces to 4*mX^2 - q^2 in the NR limit (E_chi ~ mX, tiny omega).
  Tested with rel_diff < 1e-5. (kinematics.py:120-137,
  test_srdm_infrastructure.py:TestHVector)

- sigma_bar_e prefactor (mA^2 + (alpha*me)^2)^2 is present in kinematics.py:181-196
  as reference_propagator_factor and used correctly in notebook Section 5 kernel.

- QCDark2 propagator-convention divergence (omega vs omega^2) is explicitly identified
  in notebook Section 2, contained to the labelled QCDark2-only reference cell
  (Section 4), and the production DMeRates convention follows the paper
  (omega^2 - q^2 - mA^2)^2. Term-by-term dsigma_rel2 reproduction under QCDark2's
  convention achieves 7.9e-16 (machine precision); the ~1e-3 end-to-end deviation is
  entirely attributable to the documented convention difference.

- Flux file unit conversion is correct and applied exactly once at the loader boundary.
  c_kms_bare is computed as bare float nu.c0/(nu.km/nu.s); no randomized-unit leakage.
  dPhi/d(v/c) = dPhi/dv_kms * c_kms_bare is applied once in flux_loader.py:63.
  Dedicated test confirms bare-c convention (test_srdm_infrastructure.py:166-184).

- v-integration uses torch.trapezoid over the flux-file grid (notebook Section 5),
  matching QCDark2's scipy.trapezoid. Simpson is not used over v.

- Halo-limit sanity check: full/NR kernel deviation = 6.5e-9, consistent with
  expected gamma correction v0^2/2 = 3.2e-7 at ppm scale. (Notebook Section 7)

- No nu.reset_units() after Constants import. Notebook seeds with random.seed(0)
  before import. Test fixtures use numericalunits.reset_units(seed=0) consistently.

- Manifest miss raises FileNotFoundError with both the lookup tuple AND the manifest
  path in the message. (flux_loader.py:36-40, test_srdm_infrastructure.py:206-220)

- No Python `for v in v_list` loop in the notebook's DMeRates reimplementation
  (Section 5 srdm_dielectric_rate). The per-v loop appears only in the numpy
  diagnostic (Section 5 term-by-term) and QCDark2 reference call (Section 4).

- QCDark2 Python imports appear only in Section 4 (the labelled reference cell).
  All other sections use local file reads and DMeRates.srdm helpers.

- f^2 -> S conversion (notebook Section 6, form_factor_to_elf_equiv) correctly
  implements 8*pi^2*(alpha*me)^2 * f^2 / (V_cell * q^3) for the structure factor,
  then 2*pi*alpha/q^2 * S for the ELF equivalent.

- All 14 tests in test_srdm_infrastructure.py pass. Notebook executes top-to-bottom.

Recommendation:
Do not proceed until the mediator_spin blocker is fixed. The fix is small:
add a guard in flux_loader.py (or a shared validation function) that raises
NotImplementedError for unsupported mediator_spin values before reaching the
manifest lookup. After that fix, proceed to srdm-dielectric-engine.
Re-check the warnings during SRDM Pass 2, especially propagator pole documentation,
engine-boundary float64 enforcement, and v-axis memory budget.
