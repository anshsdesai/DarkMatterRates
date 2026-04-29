Role: infrastructure
Recommended model: Sonnet 4.6, GPT-5.4 medium, or GPT-5.3-Codex medium
Owns: Steps 1.1, 1.2, 1.3, 1.4 (Phase 1) and Steps 4.4, 4.5 (Phase 4)
Prerequisites: Phase 0 complete (baselines green)
Phase ordering: Phase 1 is a hard prerequisite for Phase 2. Do not start
                code extraction until packaging, DataRegistry, PhysicsConfig,
                and RateSpectrum are complete and tested.

---

# infrastructure — Phases 1 and 4

## Goal

Add packaging, path management, configuration, and CI infrastructure without
touching any physics code. These steps do not change how rates are calculated.

---

## Core Invariants — Never Violate These

1. **Do NOT rename `DMeRates/` to `dmerates/`.** The package name in Python imports
   must remain `DMeRates` (capital D, capital M, capital R). Any pyproject.toml or
   setup.py must discover and expose `DMeRates`, not `dmerates`.
2. **No mass notebook import churn.** Every existing notebook must continue to work
   with `from DMeRates.DMeRate import DMeRate` and equivalent imports.
3. **Run `pytest tests/ -v` before and after each step** to confirm nothing broke.

---

## Environment Setup

```bash
cd /Users/ansh/Local/SENSEI/DarkMatterRates
source .venv/bin/activate
```

---

## Phase 1 Steps

### Step 1.1 — pyproject.toml

A `pyproject.toml` already exists in the repo root. Review it and ensure it:

- Discovers the `DMeRates` package (capital letters preserved).
- Lists production dependencies from `requirements.txt`. Do not introduce
  new heavyweight dependencies.
- Does NOT rename or alias the package.

Verify:
```bash
pip install -e . --quiet
python -c "import DMeRates; print('OK')"
python -c "from DMeRates.DMeRate import DMeRate; print('OK')"
```

If `pyproject.toml` already passes this check, document it and skip to Step 1.2.

### Step 1.2 — DataRegistry

Create `DMeRates/data/__init__.py` and `DMeRates/data/registry.py`.

The registry is a single source of truth for all file paths. Callers import it;
no module hard-codes a data path directly.

```python
# DMeRates/data/registry.py
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent  # DarkMatterRates/

class DataRegistry:
    """Centralized path management for DMeRates data files."""

    # --- External data roots (large; not bundled) ---
    form_factor_root: Path = _REPO_ROOT / "form_factors"
    halo_root: Path        = _REPO_ROOT / "halo_data"
    rate_output_root: Path = _REPO_ROOT / "DMeRates" / "Rates"

    # --- QCDark2 data root (external; override via env var) ---
    qcdark2_root: Path = Path(
        os.environ.get("DMERATES_QCDARK2_ROOT",
                       "/Users/ansh/Local/SENSEI/QCDark2")
    )

    # --- QCDark2 dielectric file paths ---
    @classmethod
    def qcdark2_dielectric(cls, material_key: str, variant: str) -> Path:
        """Return path to a QCDark2 HDF5 dielectric file.

        material_key: 'Si', 'Ge', 'GaAs', 'SiC', 'Diamond' or 'diamond'
        variant: 'composite', 'lfe', 'nolfe'
        """
        suffix = {
            "composite": "comp",
            "lfe": "lfe",
            "nolfe": "nolfe",
        }[variant]
        material_filename = "diamond" if material_key == "Diamond" else material_key
        return (
            cls.qcdark2_root
            / "dielectric_functions"
            / variant
            / f"{material_filename}_{suffix}.h5"
        )

    # --- QCDark1 HDF5 form factors ---
    @classmethod
    def qcdark1_ff(cls, material: str) -> Path:
        return cls.form_factor_root / "QCDark" / f"{material}_final.hdf5"

    # --- QEDark txt form factors ---
    @classmethod
    def qedark_ff(cls, material: str) -> Path:
        return cls.form_factor_root / "QEDark" / f"{material}_f2.txt"

    # --- Noble gas pkl form factors ---
    @classmethod
    def noble_ff(cls, material: str) -> Path:
        return cls.form_factor_root / "wimprates" / \
               f"{material}_dme_ionization_ff.pkl"

    # --- Small bundled data ---
    p100K_dat: Path = _REPO_ROOT / "DMeRates" / "p100K.dat"
```

**Verify the actual QCDark2 filename suffixes** before writing `qcdark2_dielectric()`:
```bash
ls /Users/ansh/Local/SENSEI/QCDark2/dielectric_functions/composite/
ls /Users/ansh/Local/SENSEI/QCDark2/dielectric_functions/lfe/
ls /Users/ansh/Local/SENSEI/QCDark2/dielectric_functions/nolfe/
```
Use the actual filenames — do not guess.

After creating the registry, update existing hard-coded paths in `DMeRate.py`
and `form_factor.py` to use `DataRegistry` — but only if this does not change
any behavior. If a change would alter a path that tests depend on, keep the old
path and add `DataRegistry` as the future-preferred route without removing the old one.

Verification: `pytest tests/ -v` passes.

### Step 1.3 — PhysicsConfig

Create `DMeRates/config.py`.

```python
# DMeRates/config.py
import numericalunits as nu
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhysicsConfig:
    """User-facing physics parameters in conventional units.

    All values stored in numericalunits SI internally. Construct with
    explicit keyword arguments; use PhysicsConfig.defaults() for the
    standard SHM parameters matching DMeRates/Constants.py.
    """
    v0_km_s: float = 238.0
    vEarth_km_s: float = 250.2
    vEscape_km_s: float = 544.0
    rhoX_GeV_cm3: float = 0.3
    sigma_e_cm2: float = 1e-36  # DMeRates legacy default

    # Derived numericalunits values (set in __post_init__)
    v0: float = field(init=False)
    vEarth: float = field(init=False)
    vEscape: float = field(init=False)
    rhoX: float = field(init=False)
    sigma_e: float = field(init=False)

    def __post_init__(self):
        self.v0      = self.v0_km_s      * nu.km / nu.s
        self.vEarth  = self.vEarth_km_s  * nu.km / nu.s
        self.vEscape = self.vEscape_km_s * nu.km / nu.s
        self.rhoX    = self.rhoX_GeV_cm3 * 1e9 * nu.eV / nu.c0**2 / nu.cm**3
        self.sigma_e = self.sigma_e_cm2  * nu.cm**2

    def to_dict(self) -> dict:
        return {
            'v0_km_s':      self.v0_km_s,
            'vEarth_km_s':  self.vEarth_km_s,
            'vEscape_km_s': self.vEscape_km_s,
            'rhoX_GeV_cm3': self.rhoX_GeV_cm3,
            'sigma_e_cm2':  self.sigma_e_cm2,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'PhysicsConfig':
        return cls(**d)

    @classmethod
    def defaults(cls) -> 'PhysicsConfig':
        return cls()
```

**Acceptance criteria:**
- A round-trip test: `PhysicsConfig.from_dict(cfg.to_dict())` produces the
  same numericalunits values as the original.
- Passing a config changes halo and cross-section values consistently in both
  the old `DMeRate` constructor and any new engine path.

Write a minimal pytest for the round-trip in `tests/test_config.py`:
```python
from DMeRates.config import PhysicsConfig
import numericalunits as nu

def test_physicsconfig_roundtrip():
    cfg = PhysicsConfig(v0_km_s=230.0, vEarth_km_s=240.0)
    cfg2 = PhysicsConfig.from_dict(cfg.to_dict())
    assert abs(cfg.v0 - cfg2.v0) / cfg.v0 < 1e-12
    assert abs(cfg.vEarth - cfg2.vEarth) / cfg.vEarth < 1e-12
```

### Step 1.4 — RateSpectrum

Create `DMeRates/spectrum.py`.

```python
# DMeRates/spectrum.py
from dataclasses import dataclass, field
from typing import Optional, Dict
import torch


@dataclass
class RateSpectrum:
    """Container for a differential rate spectrum before detector-response conversion.

    E and dR_dE carry numericalunits units throughout. To express in a
    specific unit, divide by it (e.g., dR_dE / (1/(nu.kg * nu.year * nu.eV))).
    """
    E: torch.Tensor               # energy array, shape (N_E,)
    dR_dE: torch.Tensor           # differential rate, shape (N_E,)
    material: str
    backend: str                  # 'qcdark1', 'qedark', 'qcdark2', 'noble_gas'
    metadata: Dict = field(default_factory=dict)
    # Optional shell decomposition for noble gases
    shell_spectra: Optional[Dict[str, torch.Tensor]] = None
    shell_labels: Optional[list] = None
```

`metadata` should capture at minimum: `mX`, `FDMn`, `halo_model`, `screening`,
`integrate` flag, and any other physics parameters relevant to the calculation.

---

## Phase 4 Steps

### Step 4.4 — GitHub Actions CI

Create `.github/workflows/tests.yml`.

**Policy:**
- CPU-only tests. Do not require CUDA.
- Run on Python 3.10 and 3.11.
- Skip QCDark2 dielectric tests if `DMERATES_QCDARK2_ROOT` is not set or the
  data files are absent (use `pytest.mark.skipif` in the test files, not in CI).
- Run: `pytest tests/ -v --tb=short`

```yaml
name: Tests

on:
  push:
    branches: [main, qcdark2_integration]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive        # needed for torchinterp1d
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest
      - name: Run tests
        run: pytest tests/ -v --tb=short
```

Verification: Push to the branch and confirm the Actions workflow appears on GitHub.

### Step 4.5 — Rate file provenance sidecar

Add sidecar writing to `DMeRate.generate_dat()` in `DMeRates/DMeRate.py`.

When `generate_dat(...)` writes a `.dat` file, it must also write a `.dat.yaml`
sidecar alongside it:

```python
import yaml
import DMeRates  # for __version__

def _write_sidecar(dat_path: str, config: dict):
    sidecar_path = dat_path + ".yaml"
    with open(sidecar_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
```

The sidecar dict must include:
```python
{
    'package_version': DMeRates.__version__,
    'backend':         self.form_factor_type,
    'material':        self.material,
    'physics': {
        'v0_km_s':      self.v0  / (nu.km / nu.s),
        'vEarth_km_s':  self.vEarth / (nu.km / nu.s),
        'vEscape_km_s': self.vEscape / (nu.km / nu.s),
        'rhoX_GeV_cm3': self.rhoX / (1e9 * nu.eV / nu.c0**2 / nu.cm**3),
        'sigma_e_cm2':  self.cross_section / nu.cm**2,
    },
    'halo_model':      halo_model,  # passed to generate_dat
    'FDMn':            FDMn,
    'screening':       DoScreen,
    'integrate':       integrate,
    # QCDark2 variant if applicable (add if form_factor_type == 'qcdark2'):
    # 'qcdark2_variant': self.qcdark2_variant,
}
```

**Acceptance criteria:**
- `generate_dat(...)` produces both `rates.dat` and `rates.dat.yaml`.
- `yaml.safe_load(open('rates.dat.yaml'))` returns a dict that passes
  `PhysicsConfig.from_dict(d['physics'])` without error.
- Write a pytest for this round-trip in `tests/test_sidecar.py`.

`PyYAML` must be in `requirements.txt` (or `pyproject.toml` dependencies).
Check first; if absent, add it.
