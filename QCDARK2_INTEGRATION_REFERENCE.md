# QCDark2 Integration Reference

This document records the first-pass `QCDark2` integration into
`DarkMatterRates`: what changed, how it was implemented, how it was validated,
and what tradeoffs were made.

It is meant to be a durable engineering note for future maintenance and
extensions, not just a short changelog.

Related notes:

- [`QCDARK2_PORTING_SCOPE.md`](QCDARK2_PORTING_SCOPE.md): what was intentionally
  left out of scope for this pass
- [`QCDARK2_FLUX_EXTENSION_NOTES.md`](QCDARK2_FLUX_EXTENSION_NOTES.md): how the
  current `eta(vmin)`-based integration differs from a true boosted / flux-based
  backend

## Goals Of This Pass

The integration work targeted four concrete outcomes:

1. Make `QCDark2` a first-class `DarkMatterRates` backend under the existing
   `DMeRate` interface.
2. Reproduce the non-relativistic halo results from the upstream sibling
   checkout at `../QCDark2`, treating upstream `QCDark2` as the accuracy
   reference.
3. Keep `DarkMatterRates` self-contained at runtime by vendoring the benchmark
   dielectric files needed for production use.
4. Preserve the faster vectorized / Torch-based execution model already used in
   this repository, while matching the upstream physics and unit conventions.

## High-Level Outcome

After this pass:

- `DMeRate(material, form_factor_type='qcdark2', qcdark2_variant=...)` is a
  supported production path for `Si`, `Ge`, `GaAs`, `SiC`, and `Diamond`.
- The native backend reproduces upstream `QCDark2` non-relativistic `dR/dE`
  spectra and total rates across `composite`, `lfe`, and `nolfe` variants.
- Public helpers now exist for spectrum-level work:
  - `calculate_spectrum(...)`
  - `calculate_total_rate(...)`
- Existing `calculate_rates(..., ne)` behavior remains available and folds the
  validated `qcdark2` recoil spectrum into electron/pair-yield bins.
- The runtime package is self-contained once the benchmark dielectric `.h5`
  files are present under `form_factors/QCDark2/`.

## Main Design Decisions

### 1. Upstream `QCDark2` was treated as the correctness oracle

The previous local implementation had drifted away from upstream normalization
and rate conventions. This integration deliberately aligned the native backend
to `../QCDark2/qcdark2/dark_matter_rates.py`, especially the non-relativistic
halo rate path in `get_dR_dE(...)`.

Legacy `qcdark` was not used as a parity target. It remains useful as a
comparison baseline for physics differences and performance, but not as the
acceptance reference for `qcdark2`.

### 2. A dedicated backend module was introduced

Rather than keeping the full `qcdark2` implementation inline in
`DMeRates/DMeRate.py`, the new rate kernel was moved into a dedicated module:

- `DMeRates/qcdark2_backend.py`

This keeps the `DMeRate` class focused on API orchestration and detector/halo
integration while the physics kernel stays isolated and easier to test.

### 3. Explicit physical units were used inside the new backend

The existing semiconductor code in this repository leans heavily on
`numericalunits`. The new backend instead mirrors the explicit numeric unit
conventions used in upstream `QCDark2`:

- energies in `eV`
- rates in `events / kg / year / eV`
- velocities in either `v/c` internally or `km/s` at the halo-file boundary

This was important for two reasons:

- it made direct parity checks against upstream straightforward
- it avoided inheriting old normalization assumptions from the legacy
  semiconductor path

### 4. The old downstream API contract was preserved

Internally, the new backend works in explicit physical units, but the
`DMeRate.calculate_rates(..., ne)` path still expects the repository's older
implicit unit convention downstream of `vectorized_dRdE(...)`.

To avoid breaking existing consumers, the `qcdark2` spectrum is converted back
into the repository's implicit internal units before the `n_e` folding step.
This is a deliberate compatibility shim, not a physics change.

## File-By-File Changes

## `DMeRates/qcdark2_backend.py`

This is the core new backend. It contains:

- explicit upstream-style constants:
  - `LIGHT_SPEED`
  - `ALPHA`
  - `M_E`
  - `KG`
  - `CM2SEC`
  - `SEC2YR`
- a local Simpson-rule integrator `simpson_uniform(...)` for evenly spaced grids
- `QCDark2Backend`, which owns:
  - the dielectric-function grids
  - cached Torch tensors by `(device, dtype)`
  - `vmin` grid generation
  - analytic Maxwell-Boltzmann `eta(vmin)`
  - halo-file interpolation onto the `QCDark2` `(q, E)` grid
  - the final non-relativistic differential-rate calculation

Implementation details:

- `q_raw`, `q_eV`, `E_eV`, and `S(q,E)` are cached on construction
- tensor copies are cached lazily per `(device, dtype)` pair
- band-gap masking is applied in the backend itself so spectra below the
  material threshold are zeroed consistently
- the rate kernel uses the upstream-style structure:
  - `integrand = q * |F_DM|^2 * S(q,E) * eta(vmin(q,E))`
  - followed by `q` integration and the explicit upstream prefactor

The backend currently targets the non-relativistic halo calculation only.

## `DMeRates/interpolation.py`

This small helper was added to fix a runtime import problem.

Before this change, source-tree execution could fail to resolve
`torchinterp1d` cleanly unless tests had already adjusted `sys.path`. The new
wrapper:

- first tries to import an installed `torchinterp1d`
- then falls back to the vendored source-tree copy under `./torchinterp1d`
- clears stale namespace-package imports when needed

This makes direct repo execution more reliable outside pytest.

## `DMeRates/DMeRate.py`

This file received the largest set of integration changes.

### Constructor and setup changes

- `form_factor_type='qcdark2'` remains supported
- `qcdark2_variant` remains part of the constructor contract
- the constructor now instantiates a `QCDark2Backend` when `qcdark2` is chosen
- the allowed material list is validated against `qcdark2_band_gaps`

### New internal helpers

The following orchestration helpers were added:

- `_coerce_1d_tensor(...)`
  - normalizes scalar/list/array inputs into 1D Torch tensors
- `_current_astro_model_numeric(...)`
  - returns the explicit numeric astrophysical model used by the backend
- `_qcdark2_eta_grid(...)`
  - evaluates `eta(vmin)` directly on the `QCDark2` `(q, E)` grid
  - supports:
    - analytic SHM-style MB halos
    - file-based halo tables
    - step-function halo-independent analyses

### Astro parameter handling

An explicit `_astro_numeric` state is tracked alongside the older
`numericalunits`-based attributes so the backend can operate in upstream-like
numeric units without ambiguity.

### `vectorized_dRdE_qcdark2(...)`

This method was rewritten to:

- assemble the `eta(vmin)` grid using `_qcdark2_eta_grid(...)`
- call the dedicated backend for the physical spectrum
- convert the explicit physical result back to the repository's legacy implicit
  units for compatibility with downstream `n_e` folding

This method is now the single production entry point for `qcdark2`
`dR/dE` inside `DMeRate`.

### `calculate_semiconductor_rates(...)`

The `qcdark2` branch was updated so that:

- SHM / IMB cases use the analytic MB path directly
- file-based halos still work through `setup_halo_data(...)`
- `n_e` folding uses the validated spectrum kernel rather than the old
  approximate normalization
- energy integration for `qcdark2` uses `simpson_uniform(...)` on the explicit
  energy grid

### Public APIs added

Two public helpers were added:

- `calculate_spectrum(...)`
  - returns `(energy_eV, spectra)`
  - units: `events / kg / year / eV`
- `calculate_total_rate(...)`
  - integrates `calculate_spectrum(...)`
  - units: `events / kg / year`

These were added to make direct spectrum-level validation and future benchmark
workflows straightforward.

### `DoScreen` behavior for `qcdark2`

For `qcdark2`, `DoScreen=True` is now explicitly treated as a warning/no-op
because the dielectric function already carries the RPA screening information.
No extra Thomas-Fermi screening is applied on top.

## `DMeRates/form_factor.py`

The `form_factorQCDark2` wrapper was extended so it now carries:

- `band_gap`
- `band_gap_eV`
- upstream dielectric grids and metadata in a form convenient for the new
  backend

This wrapper continues to expose:

- `elf()`
- `S()`

which are the material objects consumed by the rate kernel.

## `DMeRates/Constants.py`

New `QCDark2` material metadata were added:

- `qcdark2_band_gaps`
- `qcdark2_pair_energies`

These provide the material-specific values used by the constructor and by the
first-pass `n_e` folding model.

## `form_factors/QCDark2/`

The non-relativistic benchmark dielectric dataset was mirrored into the repo so
runtime use does not depend on the sibling `../QCDark2` checkout.

Vendored variants:

- `composite`
- `lfe`
- `nolfe`

Vendored materials:

- `Si`
- `Ge`
- `GaAs`
- `SiC`
- `Diamond`

The directory also includes:

- `form_factors/QCDark2/README.md`

which records the provenance of the vendored benchmark files.

## `tests/conftest.py`

Test helpers were extended to support:

- locating local vendored `qcdark2` dielectric files
- locating the sibling upstream `../QCDark2` dielectric files
- skip markers for missing local or upstream test data

This lets the parity suite run cleanly while still producing informative skips
when external reference data are absent.

## `tests/test_rate_calculation.py`

This file was updated to remove the old comparison that assumed `qcdark` and
`qcdark2` rates should be of the same order.

New sanity/API tests were added for:

- `calculate_spectrum(...)`
- `calculate_total_rate(...)`
- `DoScreen` warning semantics
- `DoScreen=True` and `DoScreen=False` equivalence for `qcdark2`
- positivity and band-gap behavior

## `tests/test_qcdark2_reference.py`

This is the main accuracy suite added for the integration.

It directly compares the native backend against the sibling upstream checkout
for:

- all supported materials
- all three dielectric variants
- heavy and light mediators
- `mX = 100 MeV`

It also includes an extended `Si composite` mass scan at:

- `0.5 MeV`
- `5 MeV`
- `100 MeV`
- `1000 MeV`

Acceptance checks:

- total rates within `2%`
- populated spectral bins within `5%`
- folded `n_e` rates within `5%` on populated bins when the same response model
  is applied to the upstream spectrum

## `scripts/reproduce_qcdark2_nr.py`

This script was added to reproduce upstream notebook-style non-relativistic halo
outputs inside `DarkMatterRates`.

It generates, for all supported material/variant combinations:

- `dR/dE` at `mX = 1 GeV` for heavy and light mediators
- sensitivity-style curves using the native total-rate path and upstream
  `dm.ex(...)`

It can optionally write `.npz` artifacts and a JSON summary.

## `scripts/benchmark_qcdark2.py`

This script benchmarks three paths:

- upstream `QCDark2`
- native `DarkMatterRates qcdark2`
- legacy `DarkMatterRates qcdark` for overlapping `Si` and `Ge` cases

It reports:

- single-spectrum wall time
- mass-sweep wall time
- CPU results by default
- GPU-native `qcdark2` timings if CUDA is available

## Physics And Numerical Details

### Material input

The new backend consumes precomputed dielectric functions only. No dielectric
generation machinery was integrated in this pass.

The dynamic structure factor is taken directly from the dielectric data:

- `ELF = Im(eps) / |eps|^2`
- `S(q,E) = ELF * q^2 / (2 * pi * alpha)`

This matches the structure used in the upstream code path.

### Halo input

Three `eta(vmin)` input modes now coexist for `qcdark2`:

1. Analytic MB evaluation for `shm` / `imb`
2. Interpolated halo-file `eta(vmin)` tables for file-based halo models
3. Step-function `eta(vmin)` support for halo-independent studies

The file-based path expects halo tables in the repository's established format:

- column 1: `vmin` in `km/s`
- column 2: `eta` in `s/km`

Internally, these are converted into the upstream `QCDark2` convention.

### Unit bridge

The main unit bridge in this pass is:

- explicit physical units in the backend
- repository-legacy implicit units downstream of `vectorized_dRdE_qcdark2(...)`

This is the main compatibility compromise that let the new backend integrate
without rewriting the rest of the semiconductor code.

### Detector response / `n_e` folding

The first-pass detector response model intentionally stays close to what the
repo already had:

- `Si`: existing RK-based ionization probability treatment
- `Ge`, `GaAs`, `SiC`, `Diamond`: step-function pair-creation model

The important change is that the folding now starts from the validated native
`qcdark2` recoil spectrum rather than from the older mismatched normalization.

## Validation Performed

## 1. Direct upstream parity suite

Command:

```bash
pytest -q tests/test_qcdark2_reference.py
```

Observed result:

- `38 passed` in about `53.45 s`

What this covers:

- all vendored materials and variants
- heavy and light mediators
- total-rate parity
- pointwise spectral parity on populated bins
- `n_e` folding parity against the same response model applied to the upstream
  reference spectrum

Notes:

- the warning volume is dominated by upstream `QCDark2` deprecation warnings
  from SciPy positional `simpson(...)` calls
- upstream also emits a few `SyntaxWarning`s on docstring escape sequences

## 2. Native-vs-upstream benchmark

Command:

```bash
python scripts/benchmark_qcdark2.py
```

Observed CPU results on this machine:

- `fdm_0`
  - upstream `qcdark2` single spectrum: `1.5321 s`
  - upstream `qcdark2` mass sweep: `7.2980 s`
  - native `qcdark2` single spectrum: `0.0112 s`
  - native `qcdark2` mass sweep: `0.0877 s`
- `fdm_2`
  - upstream `qcdark2` single spectrum: `1.4278 s`
  - upstream `qcdark2` mass sweep: `7.1937 s`
  - native `qcdark2` single spectrum: `0.0096 s`
  - native `qcdark2` mass sweep: `0.0769 s`

Also reported for comparison:

- legacy `qcdark` timings for `Si`
- legacy `qcdark` timings for `Ge`

GPU result during this run:

- `gpu: null`
- CUDA was not available on this machine at runtime

## 3. Full test suite during integration

During the main integration pass, the full repo suite was also run:

```bash
pytest -q
```

Observed result at that time:

- `78 passed`

This was the final end-to-end check after the new backend, tests, and scripts
had all been wired together.

## Known Warnings And Residual Issues

These are not blockers for the current integration, but they are worth
remembering:

- upstream `QCDark2` currently emits:
  - `SyntaxWarning`s from a few docstring escape sequences
  - `DeprecationWarning`s from positional `simpson(...)` calls
- the benchmark script also surfaces existing Torch / Torchquad warnings:
  - `torch.set_default_tensor_type()` deprecation
  - Torchquad odd-grid adjustment warnings
  - future `torch.meshgrid` indexing warning
- running some test paths can populate extra halo cache files in `halo_data/`

## What Was Deliberately Not Implemented

This integration intentionally did not include:

- dielectric generation via PySCF / DFT
- new-material build pipelines
- notebook/docs-site migration from upstream
- boosted / flux-driven DM workflows
- runtime dependence on the sibling `../QCDark2` checkout

Those boundaries are documented in more detail in
[`QCDARK2_PORTING_SCOPE.md`](QCDARK2_PORTING_SCOPE.md).

## Practical Takeaways For Future Work

- The native `qcdark2` kernel is now the right place to extend spectrum
  calculations in this repo.
- Upstream `QCDark2` remains the most useful parity reference when changing the
  non-relativistic backend.
- Future work should try to preserve the separation of concerns introduced here:
  - backend physics kernel in `DMeRates/qcdark2_backend.py`
  - orchestration / public API in `DMeRates/DMeRate.py`
  - reference validation in dedicated parity tests
- If future extensions target non-halo fluxes or boosted DM, they should likely
  be added as a parallel backend/API path rather than forcing those workflows
  through the current `eta(vmin)` abstraction.
