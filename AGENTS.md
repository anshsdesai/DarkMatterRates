# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What This Is

DMeRates is a Python library for vectorized calculation of dark matter (DM) electron scattering rates in Si, Ge, Xe, and Ar. It uses PyTorch for GPU-accelerated computation and was developed primarily to study daily modulation of DM signals due to Earth scattering. The associated paper is [arXiv:2507.00344](http://arxiv.org/abs/2507.00344).

## Environment Setup

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The code automatically detects and uses CUDA GPUs. MPS (Apple Silicon) is intentionally disabled due to float32 precision limitations — it can be re-enabled manually by setting `device='mps'` on the `DMeRate` constructor.

## Running Examples

The primary entry point for understanding usage is `DMeRates_Examples.ipynb`. Launch Jupyter and open it:

```bash
jupyter notebook DMeRates_Examples.ipynb
```

For modulation study figures (the paper's results), use:

```bash
jupyter notebook modulation_study/modulation_figures.ipynb
```

To regenerate modulated rates (requires halo data from Dryad):

```bash
jupyter notebook modulation_study/modulation_rates_generating.ipynb
```

## Architecture

### Core Package: `DMeRates/`

- **`Constants.py`** — All physical constants and SHM halo parameters (`v0`, `vEarth`, `vEscape`, `rhoX`), material properties (band gaps, atomic weights, Thomas-Fermi screening params, noble gas binding energies). Uses `numericalunits` with randomized unit scales to catch unit errors at runtime. Edit this file to change default SHM parameters.

- **`DM_Halo.py`** — `DM_Halo_Distributions` class. Implements SHM (`etaSHM`), Tsallis (`etaTsa`), and Double Power Law (`etaDPL`) velocity distributions. The tensor version (`eta_MB_tensor`) is used in GPU computation paths. `generate_halo_files()` writes precomputed η(v_min) data to `halo_data/`.

- **`form_factor.py`** — Three form factor classes:
  - `form_factor`: Loads QCDark `.hdf5` files for Si/Ge (primary)
  - `form_factorQEDark`: Loads QEDark `.txt` files for Si/Ge (legacy)
  - `formFactorNoble`: Loads wimprates `.pkl` files for Xe/Ar

- **`DMeRate.py`** — `DMeRate` class, the main calculation engine:
  - Constructor initializes device, loads form factors, precomputes ionization probabilities
  - `calculate_rates(mX_array, halo_model, FDMn, ne, ...)` — main public API, dispatches to semiconductor or noble gas paths
  - `vectorized_dRdE(...)` — computes dR/dE for semiconductors (Si/Ge)
  - `noble_dRdE(...)` → `rate_dme_shell(...)` — per-shell rates for nobles (Xe/Ar)
  - `calculate_semiconductor_rates(...)` / `calculate_nobleGas_rates(...)` — mass-array loops
  - `generate_dat(...)` — writes pre-calculated rates to `DMeRates/Rates/*.dat`
  - `setup_halo_data(mX, FDMn, halo_model, isoangle=...)` — loads the right η(v_min) file; generates it if missing

### Data Directories

- **`form_factors/`** — Crystal form factors: `QCDark/` (HDF5), `QEDark/` (txt), `wimprates/` (pkl)
- **`halo_data/`** — Precomputed η(v_min) files for SHM and other analytic models; `modulated/` subdirectory holds DaMaSCUS/Verne angle-dependent files for the modulation study
- **`DMeRates/Rates/`** — Pre-calculated rate `.dat` files, named by physics parameters
- **`limits/`** — Experimental constraint data (CSV files) and `Constraints.py` for loading them
- **`sensitivity_projections/`** — Expected sensitivity CSVs for Darkside-20k and Oscura
- **`modulation_study/`** — Analysis notebooks, `Modulation.py` (plotting/analysis utilities), and `isoangle.py`
- **`halo_independent/`** — Halo-independent analysis results and mock data
- **`torchinterp1d/`** — External submodule for GPU-compatible 1D interpolation (used in `get_halo_data` and `RKProbabilities`)

### Key Design Patterns

**Units**: All quantities carry `numericalunits` units throughout. To express a value in a specific unit, divide by it (e.g., `value / nu.km` gives km). The randomized unit scales in `Constants.py` act as a runtime unit-correctness test.

**Halo model string keys**: `'shm'`, `'tsa'`, `'dpl'` trigger analytic computation (or file lookup); `'modulated'` and `'summer'` use DaMaSCUS/Verne files indexed by `isoangle` (integer 0–35, representing 0°–175° in 5° steps); `'imb'` uses the in-memory Maxwell-Boltzmann tensor path.

**FDM form factor**: `FDMn=0` → heavy mediator (FDM=1); `FDMn=2` → light mediator (FDM∝1/q²). The parameter is the power `n` in `(α·me·c/q)^n`.

**Electron-hole pair probabilities**: Silicon uses interpolated Ramanathan-Kurinsky probabilities from `p100k.dat` (100K data). Germanium always uses the step function approximation (`change_to_step()` is called automatically).

**Integration**: The `integrate=True` path uses `torchquad.Simpson` for numerical q-integration; `integrate=False` uses a Riemann sum over the precomputed q-grid. QEDark form factors always use `integrate=False`.
