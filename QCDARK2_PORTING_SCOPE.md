# QCDark2 Porting Scope

This note captures what the first-pass `QCDark2` integration into `DarkMatterRates`
does and does not include.

## Included In This Pass

- Native `DarkMatterRates` support for `form_factor_type='qcdark2'`
- Vendored benchmark dielectric files for:
  - `Si`, `Ge`, `GaAs`, `SiC`, `Diamond`
  - `composite`, `lfe`, and `nolfe`
- Spectrum-level rate calculations through:
  - `calculate_spectrum(...)`
  - `calculate_total_rate(...)`
  - `calculate_rates(..., ne)` using the validated `qcdark2` spectrum kernel
- Direct validation against the sibling upstream `../QCDark2` repository
- Reproduction and benchmark scripts for non-relativistic halo DM

## Explicitly Not Ported In This Pass

- Dielectric-function generation from first principles
  - The upstream `qcdark2.dielectric_pyscf` workflow was not integrated.
  - No PySCF/DFT tooling or dielectric-building pipeline is part of the runtime
    `DarkMatterRates` package.
- New-material generation workflows
  - We support loading precomputed `.h5` benchmark files only.
  - Upstream material input files such as `qcdark2/materials/*.in` were not wired
    into a `DarkMatterRates` API for producing new dielectric datasets.
- Upstream notebook, plotting, and docs-site infrastructure
  - The upstream example notebook and documentation site were not ported into the
    library API.
  - Reproduction for this pass is handled by standalone scripts under `scripts/`.
- Boosted / flux-driven use cases
  - This pass targets non-relativistic halo dark matter only.
  - External flux or boosted-DM workflows were left out of scope.
- Runtime dependence on the sibling `../QCDark2` repository
  - `DarkMatterRates` is self-contained at runtime after vendoring the benchmark
    dielectric assets.
  - The upstream repository is used only as a validation and comparison reference.
- Legacy `qcdark` parity as an acceptance target
  - Legacy `qcdark` is treated as a comparison baseline for speed and physics
    differences, not as the correctness oracle for `qcdark2`.
- More detailed detector-response modeling beyond the current first pass
  - `Si` keeps the existing RK ionization-probability treatment.
  - `Ge`, `GaAs`, `SiC`, and `Diamond` currently use the step-function
    pair-creation model already present in `DarkMatterRates`.
  - More detailed material-specific response models can be added later without
    changing the validated spectrum kernel.

## Deliberate Simplifications

- Only the top-level benchmark dielectric `.h5` files are vendored.
- The native backend is optimized around reproducing upstream non-relativistic
  `dR/dE` and total rates accurately and quickly inside the existing
  `DarkMatterRates` API.
- Sensitivity-curve reproduction is provided as a script deliverable rather than a
  new stable public library interface.

## Good Next Steps

- Add a formal path for generating new dielectric datasets from upstream inputs
- Decide whether boosted/flux workflows should live in core or in separate tools
- Upgrade the non-Si detector response models if benchmark comparisons require it
