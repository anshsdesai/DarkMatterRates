# DMeRates Refactor — Agent Runbooks

These runbooks describe the agent roles for the DMeRates architecture refactor.
The full plan lives at `~/.claude/plans/dmerates_synthesized_architecture_plan.md`.

## Dependency Order

```
Phase 0 (done — baselines green)
    │
    ▼
Phase 1  ←── infrastructure agent (Steps 1.1–1.4)
    │
    ▼
Phase 2  ←── code-extractor agent (Steps 2.1–2.5)
    │
    ├──▶ physics-derivation agent (Step 3.0)
    │         │
    │         ▼  [notebook complete + validated]
    │
    ├──▶ code-extractor agent (Steps 3.1, 3.4, 3.5, 3.6)
    │
    ├──▶ units-numerics-reviewer agent (review Steps 3.0–3.3)
    │
    ├──▶ MAIN CONVERSATION or dielectric-engine runbook
    │    (Steps 3.2, 3.3 — DSF utilities + dielectric engine)
    │
    └──▶ Phase 4  ←── infrastructure agent (Steps 4.4, 4.5)
                   code-extractor agent (Step 4.2, 4.3)

validator agent runs after each phase as a checkpoint.
```

## Agent Roles

### QCDark2 Integration (original)

| File | Owns | Recommended Model | Notes |
|------|------|-------------------|-------|
| `physics-derivation.md` | Step 3.0 | Opus 4.7 or o3 | Blocks all of Phase 3 |
| `code-extractor.md` | 2.1–2.5, 3.1, 3.4–3.6, 4.2–4.3 | Sonnet 4.6 | Longest agent; can split by phase |
| `infrastructure.md` | 1.1–1.4, 4.4–4.5 | Sonnet 4.6 | Non-physics Python; Phase 1 blocks Phase 2 |
| `units-numerics-reviewer.md` | Independent review of Steps 3.0–3.3 + SRDM gates | Opus 4.7 or GPT-5.5 | Units, dimensions, tolerances, dtype/device |
| `dielectric-engine.md` | Steps 3.2–3.3 runbook | Opus 4.7, GPT-5.5, or GPT-5.3-Codex high | Use only after Step 3.0 is validated |
| `validator.md` | Post-phase checkpoints (incl. SRDM) | Haiku 4.5 | Mechanical; run after every phase |

### SRDM Implementation (added)

Plan: `~/.claude/plans/i-have-successfully-ran-gentle-bonbon.md`.
Run order: `agents/srdm_run_order.md`.

| File | Owns | Recommended Model | Notes |
|------|------|-------------------|-------|
| `srdm-infrastructure.md` | SRDM Phase A (flux loader, manifest, kinematics) | Sonnet 4.6 / GPT-5.3-Codex medium | Pure Python + torch kinematics; small but physics-precise |
| `srdm-physics-derivation.md` | `tests/qcdark2_srdm_derivation.ipynb` | Opus 4.7 / GPT-5.5 high/xhigh / o3 | Highest-stakes SRDM step; blocks all SRDM engines |
| `srdm-dielectric-engine.md` | SRDM branch of `engines/dielectric.py` | Opus 4.7 / GPT-5.5 high / GPT-5.3-Codex high | Vectorized SRDM; reuses kinematics module |
| `srdm-form-factor-engine.md` | SRDM branch of `engines/form_factor.py` (QEDark + QCDark1) | Opus 4.7 / GPT-5.5 high | Reuses notebook-derived f² → S map |
| `srdm-api-wiring.md` | Public API plumbing for SRDM | Sonnet 4.6 / GPT-5.3-Codex medium | Mechanical kwargs + metadata plumbing |

## Special Handling Steps

Steps **3.2** (dynamic structure factor utilities) and **3.3** (native dielectric rate engine)
should be done either in the **main Claude/Codex conversation** or by a top-tier agent using
`dielectric-engine.md`. They require continuous context from Step 3.0's derivation. If delegated,
the full derivation notebook summary and the units-numerics review must be included in the prompt.

## Model Recommendations

- **Opus 4.7** for physics-derivation: Step 3.0 is the single highest-stakes step.
  A wrong prefactor or missed unit factor silently breaks every downstream QCDark2 rate.
  If you have access to **o3**, it is also a legitimate choice — this is a formal
  physics reasoning task with a verifiable numerical answer.
- **Sonnet 4.6** for code-extractor and infrastructure: careful code-movement and
  Python packaging, no deep physics reasoning required.
- **Haiku 4.5** for validator: purely mechanical — run commands, report results.

Codex equivalents:
- Opus/o3-tier physics reasoning → **GPT-5.5 high/xhigh**.
- Sonnet-tier implementation/refactor work → **GPT-5.3-Codex medium/high**.
- Infrastructure and docs → **GPT-5.4 medium** or **GPT-5.3-Codex medium**.
- Mechanical validation → **GPT-5.4-Mini low/medium**.

## Claude Code Sub-Agent vs Standalone Session

These files work as **standalone runbooks** for either Claude or Codex (copy into a
fresh session with sufficient context). They can also be used as Claude Code sub-agents
by copying into `.claude/agents/` and adding the appropriate YAML frontmatter.

**Where Claude Code sub-agents win:**
- `physics-derivation`: The `mcp__ide__executeCode` tool lets it iterate notebook cells
  in-session, validate unit conversions term-by-term, and return a validated artifact.
- `validator`: Haiku 4.5, spawned on demand after each phase — fast and cheap.

**Where a standalone session wins:**
- `code-extractor`: The multi-step Phase 2 work can exceed a sub-agent's useful context
  window. Running as an independent session (Codex or a fresh Claude session) is safer
  for long extraction sequences.
- `infrastructure`: Indifferent — both work.

## Environment

```bash
source .venv/bin/activate  # always activate before running commands
pytest tests/ -v           # baseline command
```

The `torchinterp1d/` directory is a git submodule — if it appears empty, run
`git submodule update --init`.
