=== Units/Numerics Review — SRDM Pass 2 ===
Artifact reviewed: DMeRates/engines/dielectric.py (_compute_dRdE_srdm), DMeRates/srdm/*, tests/test_qcdark2_srdm.py, tests/qcdark2_srdm_derivation.ipynb
Date: 2026-04-26

Blockers:
- None.

Warnings:
- Propagator pole (omega^2 = q^2 + mA^2) for mA=0 still has no inline comment
  documenting that q_bounds kinematic masking excludes the singularity.
  Carried from Pass 1; kinematic safety is unchanged (q_min=229 eV >> q[0]=37.3 eV
  at v_max=0.165c), but the invariant is not documented in code.
  File: DMeRates/srdm/kinematics.py:162-178 and DMeRates/engines/dielectric.py:323
  Recommend: one-line comment on mediator_propagator_inv_sq or at the engine's
  `prop_inv` call noting that q_bounds guarantees the pole is off-grid.

- Peak integrand tensor shape (299, 1251, 501) ≈ 1.5 GB float64 is not chunked.
  The shape is documented (engines/dielectric.py:301) and fits on workstation RAM
  for this single benchmark, but batch mass sweeps or larger grids will OOM.
  Carried from Pass 1. No v-axis chunking is implemented.
  Recommend: add a v-axis chunking path gated on estimated memory, or document the
  single-mass-point limitation before srdm-form-factor-engine work.

- Flux unit conversion is applied at the loader boundary (flux_loader.py:62-70)
  and then reversed inside the engine (engines/dielectric.py:286) to recover raw
  dPhi/dv_kms for QCDark2-convention matching. This is intentional and correct
  (the engine matches QCDark2's get_rate_flux exactly), but the "applied exactly
  once" spirit is weakened. A future refactor could let the engine accept raw flux
  directly or add an internal `_load_srdm_flux_bare()` that skips the nu round-trip.

Looks good:
- Pass 1 blocker FIXED: mediator_spin='scalar'/'approx'/'approx_full' now raises
  NotImplementedError before manifest lookup, in both flux_loader.py:34-39 and
  engines/dielectric.py:255-259. Test coverage in test_qcdark2_srdm.py:92-105
  and test_srdm_infrastructure.py:206-211. (flux_loader.py:34-39, engines/dielectric.py:255-259)

- Pass 1 warning RESOLVED: float64 enforced at engine boundary.
  engines/dielectric.py:309-313 casts all input tensors to torch.float64 before
  any kinematics call. kinematics.py functions preserve input dtype, so the
  float64 guarantee propagates through the entire hot path.

- gamma(v) in float64 is never NaN at v << 1. Engine computes gamma inline
  (engines/dielectric.py:315) using the same formula as kinematics.py:26.
  15 infrastructure tests + 5 engine tests all pass.

- v_min(q, omega, mX, gamma) reduces to halo v_min in gamma -> 1 limit.
  Verified in Pass 1 (kinematics.py:29-48) and still correct. The engine
  uses v_min implicitly via q_bounds, not called directly.

- q_bounds returns q_min <= q_max; degenerate cases produce empty q_mask.
  Engine calls q_bounds at line 327; the _qcdark2_half_open_mask at line 328
  zeros degenerate contributions without NaN. (kinematics.py:51-96,
  engines/dielectric.py:327-328)

- H_V reduces to 4*mX^2 - q^2 in the NR limit. Engine uses H_vector from
  kinematics.py:120-137 at line 322. Notebook Section 7 halo-limit sanity
  check confirms full/NR deviation = 6.5e-9, consistent with expected
  gamma correction at v0^2/2 = 3.2e-7. (engines/dielectric.py:322,
  notebook Section 7)

- sigma_bar_e prefactor (mA^2 + (alpha*me)^2)^2 present via
  reference_propagator_factor (kinematics.py:181-196), called at
  engines/dielectric.py:336.

- Propagator convention divergence from QCDark2 code is explicit and contained.
  QCDark2 code uses (omega - q^2 - mA^2)^2 [dark_matter_rates.py:311];
  DMeRates uses the paper convention (omega^2 - q^2 - mA^2)^2
  [kinematics.py:177]. Notebook Section 2 documents the divergence.
  dsigma_rel2 term-by-term reproduction under QCDark2's convention achieves
  7.9e-16 (machine precision); the ~1e-3 end-to-end deviation is entirely
  attributable to the convention difference. (notebook Section 5)

- Flux file unit conversion is correct. c_kms_bare is computed as bare float
  nu.c0/(nu.km/nu.s); no randomized-unit leakage. Applied at flux_loader.py:63.
  Engine reverses intentionally at line 286 to match QCDark2 convention.
  Dedicated test confirms bare-c convention (test_srdm_infrastructure.py:166-184).

- v-integration uses torch.trapezoid over the (non-uniform) flux-file grid
  (engines/dielectric.py:347), matching QCDark2's scipy.trapezoid. Simpson is
  NOT used over v.

- q-integration uses masked trapezoid with _qcdark2_half_open_mask
  (engines/dielectric.py:209-228, 330-333), matching QCDark2's Python-slice
  q_i:q_f convention. Both engine and notebook use the same masking function.

- No Python `for v in v_list` loop in the engine hot path. The entire
  SRDM computation is vectorized with torch broadcasting over (N_v, N_q, N_E).
  QCDark2's per-v for-loop (dark_matter_rates.py:393-394) is NOT copied.

- Agreement with QCDark2 reference: max relative diff = 1.0e-3 over 237
  energies (>5% of peak). Well within the 5% tolerance. The tolerance is
  justified by the documented propagator convention difference.
  (notebook Section 5, test_qcdark2_srdm.py:38-61)

- Agreement checks avoid zero-rate and threshold-edge bins. conftest.py
  reference values are at E=8.10, 14.90, 21.70 eV (all well within the
  positive-rate region; peak at ~18.4 eV). Notebook uses >5%-of-peak mask.

- No nu.reset_units() after Constants import in any production file.
  Notebook seeds with random.seed(0) before import. Test fixtures use
  conftest.py session-level seed. (Confirmed by grep across all SRDM files.)

- Results expressed by dividing by units: engines/dielectric.py:352-354
  divides bare dRdE by (nu.kg * nu.year * nu.eV) to produce nu-unit spectrum.

- RateSpectrum metadata includes halo_model='srdm', mediator_spin, flux_file
  path, mX_eV, sigma_e_cm2, FDMn, screening, variant. Test coverage at
  test_qcdark2_srdm.py:64-89.

- Manifest miss raises FileNotFoundError with lookup tuple AND manifest path.
  (flux_loader.py:42-48, test_qcdark2_srdm.py:108-121)

- Screening is required and explicit. normalize_dielectric_screening(None)
  raises ValueError. Test at test_qcdark2_srdm.py:124-137.

- Engine does not import or delegate to QCDark2 Python. All imports are from
  DMeRates.srdm.*, DMeRates.responses, DMeRates.screening, DMeRates.spectrum.
  QCDark2 Python appears only in notebook Section 4 reference cell.

- All 20 tests pass (15 infrastructure + 5 engine): confirmed in this review.

Recommendation:
Proceed. No blockers remain. The Pass 1 blocker (mediator_spin NotImplementedError)
is fixed and tested. The three warnings are non-blocking: (1) missing inline comment
for propagator pole safety, (2) no v-axis chunking for large grids, (3) flux
conversion round-trip. All are documentation/robustness improvements that can be
addressed before or during srdm-form-factor-engine work.
