Role: validator
Recommended model: Haiku 4.5 or GPT-5.4-Mini low/medium
Owns: Verification checkpoints after each phase
Prerequisites: The phase being validated must be complete per its agent runbook

---

# validator — Phase Verification Checkpoints

## Purpose

Run the canonical verification commands after each phase and report pass/fail.
This agent writes no code. If something fails, it reports the failure clearly
and stops — it does not attempt to fix the issue.

---

## Commands

Activate the environment first — every command below assumes it:

```bash
cd /Users/ansh/Local/SENSEI/DarkMatterRates
source .venv/bin/activate
```

**Pytest:**
```bash
pytest tests/ -v --tb=short 2>&1 | tee /tmp/pytest_output.txt
```

**Modulation notebook (end-to-end execution):**
```bash
jupyter nbconvert --to notebook --execute \
    modulation_study/modulation_figures.ipynb \
    --output /tmp/modulation_exec.ipynb \
    --ExecutePreprocessor.timeout=600 \
    2>&1 | tee /tmp/notebook_output.txt
```

---

## Phase Checkpoints

### After Phase 0

Expected state: baselines already established, `* 10` factor documented.

```
[ ] pytest tests/ -v passes (all tests in tests/test_qedark.py, test_qcdark1.py, test_noble_gas.py)
[ ] tests/phase0_2.md exists and contains the energy-normalization conclusion
[ ] tests/conftest.py contains QEDARK_REFS, QCDARK1_REFS, WIMPRATES_REFS with seed-0 values
```

Phase 0 is the baseline — if tests fail here, nothing else can proceed.

### After Phase 1

```
[ ] python -c "import DMeRates; print('OK')" succeeds from an installed environment
[ ] python -c "from DMeRates.DMeRate import DMeRate; print('OK')" succeeds
[ ] pytest tests/ -v passes
[ ] modulation_study/modulation_figures.ipynb executes end-to-end (see command above)
[ ] DMeRates/ is still named DMeRates/ (not dmerates/)
```

### After Phase 2

```
[ ] pytest tests/ -v passes — all QEDark, QCDark1, noble-gas baselines within tolerance
[ ] modulation_study/modulation_figures.ipynb executes end-to-end
[ ] QEDark legacy rates match QEDARK_REFS within 0.1% (integrate=False, exact path)
[ ] QCDark1 rates match QCDARK1_REFS within 2% (integrate=True, 2% tolerance)
[ ] Noble gas rates match WIMPRATES_REFS within 2%
[ ] Monolithic DMeRate.py is still functional (no regressions from extraction)
```

### After Phase 3

```
[ ] pytest tests/ -v passes — includes any new QCDark2 tests
[ ] modulation_study/modulation_figures.ipynb executes end-to-end
[ ] QCDark2 native engine matches QCDark2 numpy reference for Si/MB/RPA:
      relative difference < 5% at each validation energy
[ ] QCDark2 can use DMeRates halo providers: 'imb', file-backed 'shm',
      modulated (if data exists), halo-independent step eta
[ ] tests/qcdark2_formula_derivation.ipynb exists and runs top-to-bottom
[ ] Calling QCDark2 without explicit screening raises a clear error
[ ] Legacy QEDark/QCDark1 baselines still pass (no regression from Phase 3 wiring)
```

### After Phase 4

```
[ ] pytest tests/ -v passes — all tests including CI-skippable QCDark2 tests
[ ] .github/workflows/tests.yml exists and is syntactically valid YAML
[ ] generate_dat(...) produces both rates.dat and rates.dat.yaml
[ ] yaml.safe_load(open('rates.dat.yaml'))['physics'] round-trips via PhysicsConfig.from_dict()
[ ] DMeRates_Examples.ipynb has a QCDark2 section that executes
[ ] Functions from Modulation.py that rely on DMeRates calculations (such as plotRateComparisonSubplots()) execute end-to-end
```

### After SRDM Phase

```
[ ] pytest tests/ -v passes — including all SRDM tests
    (test_srdm_infrastructure.py, test_qcdark2_srdm.py, test_qcdark1_srdm.py, test_qedark_srdm.py)
[ ] tests/qcdark2_srdm_derivation.ipynb runs top-to-bottom
[ ] modulation_study/modulation_figures.ipynb still executes end-to-end
[ ] DMeRates_Examples.ipynb still executes end-to-end
[ ] SRDM smoke test (Si, mX=50 keV, FDMn=2, mediator_spin='vector') returns
    finite, positive dR/dE for all three backends (qcdark2, qcdark1, qedark)
[ ] mediator_spin='scalar' raises NotImplementedError
[ ] Calling halo_model='srdm' with a non-registered (mX, σ_e) tuple raises
    FileNotFoundError citing the manifest path
[ ] Calling halo_model='srdm' on the QCDark2 path without explicit screening
    raises ValueError
[ ] No qcdark2.* imports anywhere under DMeRates/ (grep check):
    grep -rn "import qcdark2\|from qcdark2" DMeRates/ should return no results
[ ] Halo-path numerical results unchanged (regression check on existing
    QEDARK_REFS, QCDARK1_REFS, WIMPRATES_REFS, and any QCDARK2 halo refs)
[ ] RateSpectrum.metadata for SRDM results carries: halo_model='srdm',
    mediator_spin, flux_file (absolute path), sigma_e_cm2, mX_eV, FDMn,
    screening, variant
```

---

## Report Template

After running checks for a given phase, produce a report in this format:

```
=== Validator Report — After Phase N ===
Date: YYYY-MM-DD
Branch: <current branch>

pytest: PASS / FAIL
  Failing tests (if any):
    - test_name: error summary

modulation notebook: PASS / FAIL
  Error (if any): <first error line>

Phase-specific checks:
  [X] check 1
  [ ] check 2 — FAILED: reason

Summary: PASS / FAIL
Next step: proceed to Phase N+1 / blocked on <failing check>
```

If any check fails, include the first 20 lines of pytest output or the notebook
error traceback. Do not truncate failures — the caller needs enough information
to diagnose the issue without re-running the commands.
