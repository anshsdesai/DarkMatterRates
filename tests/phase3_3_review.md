=== Units/Numerics Review — Pass 2 ===
Artifact reviewed: DMeRates/engines/dielectric.py + tests/test_qcdark2.py
Date: 2026-04-25
Branch: qcdark2_integration

---

## Blockers

None.

---

## Warnings

1. **Light mediator (FDMn=2) unvalidated.**
   `DMeRates/engines/dielectric.py:97` returns `q_ame**-4` and the light path in
   `compute_dRdE` is algebraically equivalent to QCDark2 `dark_matter_rates.py:166-169`
   (which sets `F_DM = 1/q**2` then squares in `momentum_integrand`). No end-to-end
   reference comparison exists. Recommend a `('Si', 'light', 1e9)` entry in
   `tests/conftest.py:QCDARK2_REFS` plus a sister test in `tests/test_qcdark2.py`
   before Step 3.6 wiring lands. Not blocking because the algebra trivially reduces to
   a power-of-q substitution; risk is regression rather than incorrect physics today.

2. **Non-Si materials (Ge, GaAs, SiC, diamond) still smoke-tested only.**
   `DMeRates/responses/dielectric.py` accepts all five materials and the engine is
   material-agnostic, but no engine-level dR/dE comparison exists for any material
   besides Si. The risk surface is HDF5 schema differences (e.g., a missing `dE` attr
   triggers the `responses/dielectric.py:64` fallback). Pass 1 raised the same concern;
   status unchanged.

3. **Test tolerance too loose to act as a regression guard.**
   `tests/test_qcdark2.py:56` uses `rel < 0.05` (5%) while measured agreement is
   0.0001% (per Pass 1 / notebook section 7). Recommend tightening to `rel < 1e-4`
   (or at most `1e-3`). The 5% bound would silently absorb a major regression.

4. **Fragile conftest import in tests.**
   `tests/test_qcdark2.py:14` does `sys.path.insert(0, '.')` and line 22 does
   `from conftest import QCDARK2_REFS`. Works only when pytest runs from repo root.
   If invoked from inside `tests/`, the import fails. Standard fix: drop the sys.path
   hack and import via `tests.conftest` with a `tests/__init__.py`, or expose
   `QCDARK2_REFS` as a fixture. Not blocking since current CI uses repo root, but
   worth correcting before merge.

5. **`dynamic_structure_factor` default alpha is late-bound (cosmetic).**
   `DMeRates/engines/dielectric.py:67-68` defaults `alpha = nu.alphaFS` if None —
   robust to load-order issues given the Pass 1 history, but worth a one-line comment
   that this is intentional.

---

## Looks Good

- **M_cell handling (Pass 1 warning resolved).** `DMeRates/engines/dielectric.py:147`
  correctly recovers bare eV via `float(d.M_cell / nu.eV)`. The loader stores `M_cell`
  as energy in `nu.eV` (`DMeRates/responses/dielectric.py:58`). No c² confusion
  downstream — `rho_T = M_cell_eV / kg_QCD / V_cell_bohr` (`engines/dielectric.py:255`)
  is dimensionally kg/Bohr³, matching QCDark2 `dark_matter_rates.py:196` exactly.

- **`_qcdark2_constants_bare()` reproduces all six QCDark2 constants correctly.**
  (`engines/dielectric.py:155-167`) Cross-checked against QCDark2 hard-coded values:
  `kg = nu.kg*c²/eV` ↔ 5.6096e35; `alpha = nu.alphaFS` ↔ 1/137.03599908;
  `me_eV = nu.me*c²/eV` ↔ 5.1099894e5; `c_kms = nu.c0/(nu.km/nu.s)` ↔ 299792.458;
  `cm2sec = 1/c_kms*1e-5` ↔ QCDark2 line 17; `sec2yr = 1/(60*60*24*365.25)` literal-identical.
  Notebook section 5 already proved <1e-7 agreement.

- **Prefactor is line-identical to QCDark2.** (`engines/dielectric.py:257-262` ↔
  `dark_matter_rates.py:196-199`.) Dimensional trace: `1/rho_T [Bohr³/kg] × rhoX/mX [1/cm³]
  × sigma_e/mu² [cm²/eV²] / (4π)`. After Simpson over q_eV and `/cm2sec/sec2yr`, final
  units reduce to events/kg/yr/eV.

- **`screen_ratio` for both screening modes is correct.**
  RPA (`engines/dielectric.py:271`): `screen_ratio = 1.0` — short-circuits
  |ε|²/|ε|² = 1 without numerical noise.
  No screening (`engines/dielectric.py:274`): `screen_ratio = Im²+Re² = |ε|²`, giving
  final integrand `S × |ε|² = Im(ε) × q²/(2πα)` — exactly the QCDark2 `eps_screen=1`
  collapse at line 205.

- **Halo nu-boundary is correct.** (`engines/dielectric.py:126-128`)
  `vmin_over_c * nu.c0` lifts dimensionless v/c → nu-velocity (correct).
  `eta_nu * nu.c0` strips 1/v_nu → 1/(v/c) (correct; matches QCDark2's
  `vEscape/lightSpeed` convention in `dark_matter_rates.py:135-137`).

- **Simpson integration is correctly oriented.** (`engines/dielectric.py:283`)
  `axis=0` collapses (N_q, N_E) → (N_E,). q-grid passed as `q_eV` in eV — matches
  QCDark2 lines 207, 212 exactly.

- **RateSpectrum packing is correct.** (`engines/dielectric.py:289-290`)
  `dRdE_bare / (nu.kg * nu.year * nu.eV)` — dividing by the unit is the standard
  DMeRates "value carries units" convention. Round-trip validated by
  `tests/test_qcdark2.py:106-108` to rtol=1e-12.

- **No `nu.reset_units()` calls** in `DMeRates/engines/dielectric.py` or
  `DMeRates/responses/dielectric.py`. Confirmed.

- **No module-level `nu.*` captures in `DMeRates/engines/dielectric.py`.** All `nu.*`
  reads occur inside function bodies. The Pass 1 load-order pattern has not re-appeared.
  The engine reads only `d.q_ame` (bare) and `d.V_cell_bohr` (bare) from the loader,
  keeping it immune to load-order regressions.

- **`random.seed(0)` at module level** in `tests/conftest.py:6-7` before any DMeRates
  import — correct.

- **Validation guards** (`engines/dielectric.py:34-35, 224-239`) reject typos and
  unsupported values cleanly. `screening=None` produces a clear ValueError with a
  useful message.

- **`_SI_COMP_PATH.is_file()` at module level is safe** (`tests/test_qcdark2.py:25-26`).
  Constructs a Path object only — does not touch disk. `.is_file()` returns False
  without raising even if the parent directory is absent.

---

## Recommendation

**Proceed.** No blockers identified. The deferred Pass 1 warning on M_cell handling
is fully resolved. Light-mediator validation (warning 1) and non-Si material coverage
(warning 2) remain warnings tied to test coverage rather than physics correctness, and
may be addressed in parallel with Step 3.6 wiring without gating it. Warnings 3 and 4
(tolerance tightening, conftest import) should land before merge but are not gates
for API wiring.

Pass 2 review is cleared. Step 3.6 API wiring (`code-extractor`) may begin.
