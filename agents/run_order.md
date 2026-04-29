# Agent Run Order — QCDark2 Integration Refactor

## Environment (run once before any agent session)

```bash
cd /Users/ansh/Local/SENSEI/DarkMatterRates
source .venv/bin/activate
```

---

## Phase 0 — Baselines (already done)

| Step | Agent | Runbook |
|------|-------|---------|
| Confirm baselines green | `validator` | `agents/validator.md` |

Phase 0 test files (`tests/conftest.py`, `tests/test_qedark.py`, etc.) are already on the branch.
Run `pytest tests/ -v` to confirm before proceeding.

---

## Phase 1 — Infrastructure (hard prerequisite for Phase 2)

| Step | Agent | Runbook |
|------|-------|---------|
| 1.1–1.4: pyproject.toml, DataRegistry, PhysicsConfig, RateSpectrum | `infrastructure` | `agents/infrastructure.md` |
| Checkpoint | `validator` | `agents/validator.md` → After Phase 1 checklist |

Phase 2 cannot start until `RateSpectrum` (Step 1.4) and `DataRegistry` (Step 1.2) exist.

---

## Phase 2 — Extract Existing Physics Paths

| Step | Agent | Runbook |
|------|-------|---------|
| 2.1: response loaders | `code-extractor` | `agents/code-extractor.md` |
| 2.2: ionization/yield models | `code-extractor` | |
| 2.3: screening | `code-extractor` | |
| 2.4: halo providers | `code-extractor` | |
| 2.5: rate engines | `code-extractor` | |
| Checkpoint | `validator` | `agents/validator.md` → After Phase 2 checklist |

Run `pytest tests/ -v` after each sub-step. Run modulation notebook after Step 2.4 and at phase end.

---

## Phase 3 — QCDark2 Integration

Order is strict. Each row blocks the one below it.

| Step | Agent | Runbook | Gate |
|------|-------|---------|------|
| 3.0: derivation notebook | `physics-derivation` | `agents/physics-derivation.md` | <5% agreement required |
| 3.1: dielectric loader | `code-extractor` | `agents/code-extractor.md` §3.1 | loader pre-check required |
| **Pass 1 review** | `units-numerics-reviewer` | `agents/units-numerics-reviewer.md` | **"Proceed" blocks step below** |
| 3.2–3.3: native engine | `dielectric-engine` | `agents/dielectric-engine.md` | validation case: Si/1 GeV/RPA |
| **Pass 2 review** | `units-numerics-reviewer` | `agents/units-numerics-reviewer.md` | **"Proceed" blocks step below** |
| 3.4–3.6: screening, metadata, API wiring | `code-extractor` | `agents/code-extractor.md` §3.4–3.6 | |
| Checkpoint | `validator` | `agents/validator.md` → After Phase 3 checklist | |

---

## Phase 4 — CI and Output

| Step | Agent | Runbook |
|------|-------|---------|
| 4.4–4.5: CI YAML, sidecar writer | `infrastructure` | `agents/infrastructure.md` |
| Final checkpoint | `validator` | `agents/validator.md` → After Phase 4 checklist |

---

## Hard Rules

- **Never skip validator checkpoints.** If tests fail, fix before continuing.
- **`units-numerics-reviewer` Pass 1 is a hard gate.** `dielectric-engine` does not start until it clears.
- **`units-numerics-reviewer` Pass 2 is a hard gate.** API wiring (Step 3.6) does not start until it clears.
- **`physics-derivation` notebook must exist and run top-to-bottom** before Steps 3.1–3.6 begin.
- **Monolith `DMeRate.py` stays functional throughout Phase 2** (Option A: duplicate-then-delete).
- **QCDark2 Python is reference-only** — never imported in `DMeRates/` production code.

## Agent Runbooks

All detailed instructions are in `agents/`:

```
agents/README.md                   — dependency diagram, model recommendations
agents/run_order.md                — this file
agents/validator.md                — verification checkpoints (all phases)
agents/infrastructure.md           — Phases 1 and 4
agents/code-extractor.md           — Phases 2 and 3 (wiring steps)
agents/physics-derivation.md       — Step 3.0 derivation notebook
agents/dielectric-engine.md        — Steps 3.2–3.3 native engine
agents/units-numerics-reviewer.md  — two-pass review gate
```
