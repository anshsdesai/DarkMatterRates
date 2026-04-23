# QCDark2 Flux / Boosted Extension Notes

This note records how the current `QCDark2` integration in `DarkMatterRates`
handles astrophysical inputs today, what that means for custom `eta(vmin)` file
workflows, and where that approach stops being equivalent to the upstream
boosted / flux-based calculation.

## Short Version

The current integration already supports more than just built-in SHM halos:

- analytic MB / SHM-style `eta(vmin)`
- halo-file `eta(vmin)` tables
- step-function `eta(vmin)` for halo-independent studies

That means the native backend already supports any non-relativistic scattering
calculation that can be expressed in the standard factorized form

`dR/dE ~ prefactor * integral dq [kernel(q,E)] * eta(vmin(q,E))`.

This is enough for:

- SHM-like halos
- daily modulation workflows based on precomputed `eta(vmin)` tables
- Verne / DaMaSCUS-style halo propagation outputs already reduced to `eta(vmin)`
- other custom non-relativistic speed distributions, provided they are supplied
  in the same normalization convention as the halo files used elsewhere in the
  repo

It is not automatically equivalent to the upstream `QCDark2` boosted / flux
workflow, because that workflow does not start from `eta(vmin)`.

## What The Current Native Backend Does

The native `qcdark2` backend computes the non-relativistic halo rate using the
same factorized structure as upstream `QCDark2` `get_dR_dE(...)`:

- build `vmin(q,E)`
- evaluate `eta(vmin)`
- compute the material kernel from `S(q,E)` and `F_DM(q)`
- integrate over `q`

The relevant implementation path in this repo is:

- `DMeRates/DMeRate.py`
  - `_qcdark2_eta_grid(...)`
  - `vectorized_dRdE_qcdark2(...)`
- `DMeRates/qcdark2_backend.py`
  - `eta_mb(...)`
  - `eta_from_file(...)`
  - `differential_rate(...)`

The halo-file path expects:

- `vmin` in `km/s`
- `eta` in `s/km`

with the same meaning used by the rest of `DarkMatterRates`: `rhoX` and
`sigma_e` still enter through the rate prefactor, not through the halo table.

## When A File-Based `eta(vmin)` Is The Right Abstraction

Using an `eta(vmin)` file is already the right abstraction if:

1. The scattering kinematics remain in the non-relativistic regime used by the
   standard halo calculation.
2. The astrophysical input can be reduced to an inverse-speed function
   `eta(vmin)` without losing information relevant to the rate kernel.
3. The normalization still matches the standard halo-style prefactor
   decomposition, where:
   - `rhoX` appears explicitly in the rate prefactor
   - the halo file contributes only the inverse-speed shape

For practical purposes, that covers most custom halo and modulation studies in
this codebase.

Examples:

- standard SHM
- modified halo shapes
- daily / annual modulation outputs from a separate transport solver after they
  have been reduced to `eta(vmin)`
- halo-independent step-function analyses

## Why This Is Not The Same As Upstream `get_rate_flux(...)`

Upstream `QCDark2` also has a different calculation path for external fluxes and
boosted populations:

- `dsigma_rel2(...)`
- `get_rate_flux(...)`

That path computes:

`dR/dE = integral dv [dPhi/dv] * [dσ/dE(v,E)]`

instead of:

`dR/dE = prefactor * integral dq [kernel(q,E)] * eta(vmin(q,E))`.

This distinction matters because the boosted / flux-based path keeps the full
velocity dependence of the differential cross section.

## The Three Biggest Differences

### 1. `eta(vmin)` compresses the astrophysics

In the non-relativistic halo derivation, the astrophysics enters only through
the inverse-speed integral `eta(vmin)`. That is a very convenient compression,
and it is exactly why the file-based halo workflow works so well.

But it is also a compression: only the information needed for that specific
non-relativistic factorization is retained.

### 2. The boosted path keeps explicit `v` and `gamma`

In upstream `QCDark2`, the flux workflow uses an explicit differential cross
section depending on:

- the incoming speed `v`
- the Lorentz factor `gamma`
- velocity-dependent kinematic limits `q_min(v,E)` and `q_max(v,E)`
- the mediator model used in `dsigma_rel2(...)`

That information is not, in general, recoverable from `eta(vmin)` alone.

### 3. The normalization object is different

The current halo backend assumes a standard halo-style prefactor:

- explicit `rhoX`
- explicit `sigma_e`
- `eta(vmin)` as a shape function

The flux workflow instead uses an absolute incoming flux:

- `dPhi/dv` in `cm^-2 s^-1`

That is a different input object. It does not naturally fit into the current
`rhoX * eta(vmin)` decomposition unless a separate reduction has already been
performed.

## So Can Solar Reflection Be Passed In As A File Today?

Sometimes, but only with an important caveat.

If a solar-reflected or otherwise nonstandard population has already been
reduced to a valid non-relativistic `eta(vmin)` under the same normalization
conventions as the rest of the halo machinery, then the current backend can use
it.

However, that is not the same thing as reproducing upstream `QCDark2`
`get_rate_flux(...)` directly.

The current backend does not natively accept:

- an absolute flux table `dPhi/dv`
- a mediator mass `m_A` for the flux kernel
- the explicit velocity-dependent relativistic cross section used upstream

So:

- custom non-relativistic `eta(vmin)` inputs are already supported
- true flux-driven / boosted calculations are not yet first-class in the native
  backend

## What Would A Proper Flux Extension Look Like?

The cleanest path is to add a parallel API rather than force everything through
the existing halo abstraction.

## Proposed public APIs

- `calculate_flux_spectrum(...)`
- `calculate_flux_total_rate(...)`
- `calculate_flux_ne_rates(...)`

Suggested inputs:

- `mX`
- `sigma_e`
- `flux`
- `v_list`
- `m_A`
- `mediator`
- `screening='RPA'`

This would mirror upstream `QCDark2` more closely while still reusing the local
material handling and detector-response layers.

## Proposed backend split

The existing backend can remain responsible for:

- non-relativistic `eta(vmin)`-based spectra

A new flux-specific backend path would handle:

- velocity-dependent cross sections
- `q_min(v,E)` / `q_max(v,E)` kinematics
- integration over supplied flux tables

That split keeps the current fast halo path clean and avoids mixing two
different physics factorizations into one interface.

## Reuse Opportunities

Even with a new flux path, a lot of the present integration can still be reused:

- dielectric data loading
- cached tensors for `q`, `E`, and `S(q,E)`
- band-gap masking
- `n_e` folding after the recoil spectrum is computed
- benchmark and parity harness structure

## Validation Plan For A Future Flux Extension

If a true boosted / flux backend is added later, the validation target should be
the sibling upstream checkout again.

Recommended parity tests:

1. Port upstream example flux tables and compare native
   `calculate_flux_spectrum(...)` against upstream `get_rate_flux(...)`.
2. Check both heavy and light mediator-like cases if the extension exposes the
   same mediator options as upstream `dsigma_rel2(...)`.
3. Reuse the existing detector-response folding to compare `n_e` spectra after
   the recoil spectrum matches upstream.
4. Keep halo `eta(vmin)` tests and flux tests separate, since they are checking
   different abstractions.

## Decision Rule For Future Work

Use the current `eta(vmin)` path when:

- the physics is non-relativistic halo-style scattering
- the astrophysical input is naturally an inverse-speed function

Add a flux-specific path when:

- the input object is an absolute flux `dPhi/dv`
- explicit velocity dependence in the cross section matters
- relativistic or boosted kinematics are part of the benchmark target

That boundary is the cleanest way to preserve the current fast, accurate halo
backend while making room for future boosted-DM support.
