Role: srdm-infrastructure
Recommended model: Sonnet 4.6 or GPT-5.3-Codex medium
Owns: SRDM Phase A — flux loader, manifest, kinematics module
Prerequisites: QCDark2 integration complete (Phases 0–4 of the original plan green); pyproject.toml installable; halo_data/ exists
Blocks: srdm-physics-derivation, srdm-dielectric-engine, srdm-form-factor-engine

---

# srdm-infrastructure — Phase A

## Goal

Add the data + kinematics scaffolding the SRDM engines need: flux-file loader,
manifest, and a small torch-native kinematics module. No rate calculation yet.

This is the foundation for the SRDM physics-derivation notebook and both
SRDM rate engines. It must be small, well-tested, and physics-precise.

---

## Files to Create

```
DMeRates/srdm/__init__.py
DMeRates/srdm/flux_loader.py
DMeRates/srdm/manifest.py
DMeRates/srdm/kinematics.py
halo_data/srdm/manifest.json
halo_data/srdm/srdm_dphidv_mX5e4_sigma1e-38_FDMn2_medvector.txt   # arXiv:2404.10066
halo_data/srdm/srdm_dphidv_mX5e5_sigma1e-37_FDMn2_medvector.txt   # arXiv:2404.10066
tests/test_srdm_infrastructure.py
```

The two flux files come from
https://github.com/hlxuhep/Solar-Reflected-Dark-Matter-Flux. Place them under
`halo_data/srdm/` with the canonical filenames above; do not check binary
copies of any other format.

---

## Filename + manifest convention

**Filename pattern:**
`srdm_dphidv_mX{mass_eV}_sigma{sigma_cm2}_FDMn{n}_med{spin}.txt`

Example: `srdm_dphidv_mX5e4_sigma1e-38_FDMn2_medvector.txt`.

**Two columns:**
- col 0: velocity (km/s)
- col 1: dΦ/dv (cm⁻²·s⁻¹ per (km/s))

**`manifest.json` schema:**

```json
{
  "files": [
    {
      "mX_eV": 5e4,
      "sigma_e_cm2": 1e-38,
      "FDMn": 2,
      "mediator_spin": "vector",
      "filename": "srdm_dphidv_mX5e4_sigma1e-38_FDMn2_medvector.txt",
      "source": "arXiv:2404.10066, Emken-Essig-Xu (2024)",
      "url": "https://github.com/hlxuhep/Solar-Reflected-Dark-Matter-Flux",
      "retrieved": "2026-04-25"
    },
    {
      "mX_eV": 5e5,
      "sigma_e_cm2": 1e-37,
      "FDMn": 2,
      "mediator_spin": "vector",
      "filename": "srdm_dphidv_mX5e5_sigma1e-37_FDMn2_medvector.txt",
      "source": "arXiv:2404.10066, Emken-Essig-Xu (2024)",
      "url": "https://github.com/hlxuhep/Solar-Reflected-Dark-Matter-Flux",
      "retrieved": "2026-04-25"
    }
  ]
}
```

---

## Module Contracts

### `DMeRates/srdm/manifest.py`

```python
def load_manifest() -> list[dict]:
    """Return parsed entries from halo_data/srdm/manifest.json (via DataRegistry)."""

def find_entry(mX_eV, sigma_e_cm2, FDMn, mediator_spin, rtol=1e-6) -> dict | None:
    """Tolerant lookup by (mX, sigma, FDMn, mediator_spin) tuple. None on miss."""
```

Use `DMeRates/data/registry.py` to resolve the manifest path; do not hard-code.

### `DMeRates/srdm/flux_loader.py`

```python
def load_srdm_flux(mX_eV, sigma_e_cm2, FDMn, mediator_spin):
    """Load (v_over_c, dphi_dv) tensors for the (mX, sigma, FDMn, spin) tuple.

    Returns:
        v_over_c: torch.Tensor, shape (N_v,), dimensionless v/c.
        dphi_dv:  torch.Tensor, shape (N_v,), in 1 / (nu.cm**2 * nu.s * (nu.km/nu.s)).

    Raises:
        FileNotFoundError: if no manifest entry matches. Message includes the
            full lookup tuple AND the absolute manifest path.
    """
```

**Unit conversions:**
- km/s → v/c: divide by `nu.c0 / (nu.km / nu.s)`.
- dΦ/dv per (km/s) → SI per (v/c): multiply by `nu.c0 / (nu.km / nu.s)` so the
  integration measure dv stays consistent.

**This module is the only place that translates km/s → v/c.** Do not let bare
km/s leak into the engines.

### `DMeRates/srdm/kinematics.py`

All functions are torch-native, broadcast-friendly, and dtype/device-aware.
**Inputs and outputs are bare floats (eV, dimensionless v/c)** per the existing
dielectric-engine convention. **No `numericalunits` inside this module** — the
boundary lives at the engine.

```python
def gamma(v_over_c: torch.Tensor) -> torch.Tensor:
    """1 / sqrt(1 - v^2). Same shape as input."""

def v_min_relativistic(q_eV, omega_eV, mX_eV, gamma_v) -> torch.Tensor:
    """v_min(q,ω) = q/(2 γ mX) + ω/q.

    Broadcasts over (..., N_q, ..., N_omega, ...) shapes.
    """

def q_bounds(v_over_c, omega_eV, mX_eV) -> tuple[torch.Tensor, torch.Tensor]:
    """q_min, q_max from QCDark2 paper eq. A.19.

    q_min = γ mX v - sqrt((γ mX - ω)^2 - mX^2)
    q_max = γ mX v + sqrt((γ mX - ω)^2 - mX^2)

    Returns shapes (N_v, N_omega), (N_v, N_omega). Where (γmX-ω)^2 < mX^2 the
    bounds are set so that q_max < q_min — the q_mask will then exclude the
    contribution rather than NaN.
    """

def q_mask(q_eV, q_min, q_max) -> torch.Tensor:
    """Bool tensor (N_v, N_q, N_omega) zeroing the integrand outside [q_min, q_max]."""

def H_vector(q_eV, E_chi_eV, E_chi_prime_eV) -> torch.Tensor:
    """H_V = (E_chi + E_chi')^2 - q^2  (eq. 2.6, paper)."""

def H_scalar(q_eV, E_chi_eV, E_chi_prime_eV, mX_eV) -> torch.Tensor:
    """H_phi = 4 mX^2 - (E_chi - E_chi')^2 + q^2  (eq. 2.6, paper)."""

def mediator_propagator_inv_sq(q_eV, omega_eV, mA_eV) -> torch.Tensor:
    """1 / (omega^2 - q^2 - mA^2)^2."""

def reference_propagator_factor(mA_eV, alpha_FS, me_eV) -> float:
    """(mA^2 + (alpha * me)^2)^2  (eq. A.20 prefactor numerator)."""
```

Each function ships with a 2–4 line docstring naming the QCDark2 paper equation
it implements.

---

## Tests (`tests/test_srdm_infrastructure.py`)

Add the following pytest cases. Tolerances are not physics-tier — these are
unit/structural checks:

```python
def test_gamma_limits():
    # gamma(0) == 1 exactly; gamma(0.5) ~ 1.1547
    ...

def test_v_min_recovers_halo_in_gamma_one():
    # In the v << 1 limit (gamma ~ 1), v_min_relativistic agrees with
    # the halo formula q/(2 mX) + omega/q to <1e-6 relative.
    ...

def test_q_bounds_omega_zero():
    # When omega -> 0, q_min -> 0 and q_max -> 2 gamma mX v (free-particle).
    ...

def test_q_bounds_degenerate_case():
    # When (gamma*mX - omega)^2 < mX^2, q_max < q_min and q_mask is empty.
    # No NaN should appear.
    ...

def test_H_vector_nr_limit():
    # In v << 1, H_V ~ 4 mX^2 (paper eq. 2.11 derivation).
    ...

def test_load_srdm_flux_smoke():
    # load_srdm_flux(5e4, 1e-38, 2, 'vector') returns sensible shapes and
    # finite positive values.
    ...

def test_load_srdm_flux_miss_raises():
    # Asking for an unregistered tuple raises FileNotFoundError with the
    # tuple AND the manifest path in the message.
    ...
```

---

## Acceptance Criteria

- [ ] `from DMeRates.srdm import flux_loader, kinematics, manifest` succeeds.
- [ ] `load_srdm_flux(5e4, 1e-38, 2, 'vector')` returns `(v_over_c, dphi_dv)` tensors with the expected shapes.
- [ ] Lookup miss raises `FileNotFoundError` with the lookup tuple AND the manifest path in the message.
- [ ] `kinematics.q_bounds(...)` reduces to `q_min, q_max ≈ 0, 2 γ mX v` in the ω → 0 limit (free-particle scattering check).
- [ ] `kinematics.gamma(0.0) == 1.0` exactly; `gamma(0.5) ≈ 1.1547` to 4 decimals.
- [ ] `kinematics.v_min_relativistic(...)` agrees with `q/(2 mX) + ω/q` to <1e-6 relative when v ≪ 1.
- [ ] `kinematics.H_vector(...)` reduces to `4 mX^2` in the v ≪ 1 limit.
- [ ] `pytest tests/ -v` still green; no halo-path numbers move.
- [ ] No `qcdark2.*` imports anywhere in `DMeRates/srdm/`.

---

## Hard Invariants

- **The flux loader is the only place that translates km/s → v/c.** Do not let bare km/s leak into engines.
- **`kinematics.py` stays bare-float (no `numericalunits`)** so it can sit inside the vectorized hot path with no overhead. The numericalunits boundary is the engine.
- **No `qcdark2.*` imports** anywhere. QCDark2 Python is reference-only and lives in notebooks.

---

## Handoff

Report in 5–8 sentences:
1. Files created and where the flux files were sourced from.
2. Manifest entries registered and pytest pass/fail state.
3. Any unit ambiguity that needed a pre-check (especially the dΦ/dv unit conversion).
4. Surface anything the units-numerics-reviewer should look at on Pass 1
   (e.g. the q_bounds degenerate-case behavior, the gamma → 1 limit recovery).

The next agent in the chain is `srdm-physics-derivation` (the notebook).
