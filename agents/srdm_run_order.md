# Agent Run Order — SRDM Implementation

This runbook supersedes `agents/run_order.md` for the solar-reflected dark
matter (SRDM) phase. The QCDark2 integration runbook is unchanged.

## Environment (run once before any agent session)

```bash
cd /Users/ansh/Local/SENSEI/DarkMatterRates
source .venv/bin/activate
```

The `torchinterp1d/` directory is a git submodule. If it appears empty:
`git submodule update --init`.

---

## Phase A — Infrastructure

| Step | Agent | Runbook |
|------|-------|---------|
| Flux loader, manifest, kinematics module | `srdm-infrastructure` | `agents/srdm-infrastructure.md` |
| Checkpoint | `validator` | `agents/validator.md` (general invariants) |

Phase B cannot start until `DMeRates/srdm/{flux_loader,kinematics,manifest}.py`
exist, the manifest is populated, and `pytest tests/test_srdm_infrastructure.py`
is green.

---

## Phase B — QCDark2 SRDM

Order is strict. Each row blocks the one below it.

| Step | Agent | Runbook | Gate |
|------|-------|---------|------|
| Derivation notebook | `srdm-physics-derivation` | `agents/srdm-physics-derivation.md` | <5 % agreement vs `get_rate_flux` reference required |
| **Pass 1 review** | `units-numerics-reviewer` | `agents/units-numerics-reviewer.md` | **"Proceed" blocks step below** |
| QCDark2 SRDM engine | `srdm-dielectric-engine` | `agents/srdm-dielectric-engine.md` | Si / 50 keV / vector / light validated |
| **Pass 2 review** | `units-numerics-reviewer` | `agents/units-numerics-reviewer.md` | **"Proceed" blocks Phase C** |

---

## Phase C — QEDark + QCDark1 SRDM

| Step | Agent | Runbook |
|------|-------|---------|
| Form-factor SRDM engines | `srdm-form-factor-engine` | `agents/srdm-form-factor-engine.md` |

Optional Pass 3 review may be invoked here at the user's discretion. Otherwise,
the validator's regression checks catch structural issues.

---

## Phase D — Public API + Final Checkpoint

| Step | Agent | Runbook |
|------|-------|---------|
| API wiring | `srdm-api-wiring` | `agents/srdm-api-wiring.md` |
| Final SRDM checkpoint | `validator` | `agents/validator.md` → "After SRDM Phase" checklist |

---

## Hard Rules

- **Notebook (`tests/qcdark2_srdm_derivation.ipynb`) must exist and pass
  before any SRDM engine code is written.** This mirrors the QCDark2 halo
  policy where `qcdark2_formula_derivation.ipynb` blocked Phase 3.
- **Both `units-numerics-reviewer` passes are hard gates.** A "Do not proceed"
  verdict means the upstream agent fixes blockers before the next gate opens.
- **Halo-path tests are regression-protected.** Any halo-path number change
  requires explicit acknowledgement and re-validation. The SRDM branch is
  opt-in only.
- **`qcdark2.*` is reference-only.** Never imported in production SRDM code.
  The notebook's "Reference run" cell block is the only legitimate consumer.
- **The monolithic halo path is untouched.** SRDM is a parallel branch in the
  engines, not a refactor of the halo path.
- **Vectorization is non-negotiable.** No Python `for v in v_list` loops in
  the hot path of any engine.

---

## Agent Runbooks

```
agents/srdm-infrastructure.md       — Phase A
agents/srdm-physics-derivation.md   — Phase B notebook
agents/srdm-dielectric-engine.md    — Phase B QCDark2 engine
agents/srdm-form-factor-engine.md   — Phase C QEDark + QCDark1 engines
agents/srdm-api-wiring.md           — Phase D
agents/units-numerics-reviewer.md   — Pass 1 + Pass 2 SRDM gates (extended)
agents/validator.md                 — SRDM phase checkpoint (extended)
```

The plan that this runbook implements is at
`~/.claude/plans/i-have-successfully-ran-gentle-bonbon.md`.
