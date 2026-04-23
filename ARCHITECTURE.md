# DarkMatterRates Architecture

This repository uses `DMeRate` as the public facade and source-family backends
for target-specific rate kernels.

## Public Facade

`DMeRates.DMeRate.DMeRate` remains the main user entry point. It owns:

- public constructor compatibility, including `QEDark=True`
- halo parameter updates
- halo-file loading and modulation-file plumbing
- public rate APIs:
  - `calculate_spectrum(...)`
  - `calculate_total_rate(...)`
  - `calculate_ne_rates(...)`
  - `calculate_rates(...)`, kept as a compatibility alias

The facade should route to backends rather than accumulate new source-specific
physics kernels inline.

## Source-Family Backends

Backends live under `DMeRates/backends/` and are grouped by source family:

- `qcdark.py`: legacy QCDark semiconductor form factors
- `qedark.py`: QEdark semiconductor form factors
- `qcdark2.py`: QCDark2 dielectric-function backend
- `wimprates.py`: Xe/Ar noble-gas form factors
- `semiconductor_common.py`: shared legacy QCDark/QEdark setup

Each backend is responsible for loading its source-family data and attaching the
minimum state needed by the facade. Shared legacy math can remain shared rather
than duplicated between `qcdark` and `qedark`.

## Unit Policy

The codebase uses a hybrid unit policy:

- use `numericalunits` and `Constants.py` at public/module boundaries
- derive fixed numeric backend constants in `DMeRates/units.py`
- keep hot Torch kernels in plain numeric tensors once units are converted

This preserves dimensional checks and easy recasting at the outer layers while
keeping backend kernels simple to validate against upstream reference codes.

## Detector Response

Shared response helpers live in `DMeRates/response.py`.

Current defaults:

- Si semiconductor response keeps the RK probability model
- non-Si QCDark2 semiconductors use the step pair-creation model
- noble gases keep the shell-to-electron response model from the existing code

Future detector-response changes should be implemented in `response.py` or a
dedicated response module rather than inside backend kernels.

## Compatibility Rules

The following should remain stable unless intentionally deprecated:

- `from DMeRates.DMeRate import DMeRate`
- `DMeRate('Si')`
- `DMeRate('Si', QEDark=True)`
- `DMeRate(material, form_factor_type='qcdark2', qcdark2_variant='composite')`
- `calculate_rates(masses, halo_model, FDMn, ne)`

New code should prefer `calculate_ne_rates(...)` when requesting electron-count
or electron-hole-pair binned rates.

## Future Extensions

Use a new backend/API path when a future feature changes the physics
factorization. For example, a boosted or flux-based QCDark2 workflow should not
be forced through the existing `eta(vmin)` halo abstraction; it should be added
as a parallel `calculate_flux_*` path and backend method.
