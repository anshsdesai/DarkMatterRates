# QCDark2 Dielectric Function Files

This repository now vendors the non-relativistic benchmark dielectric-function set
used by the first-pass `qcdark2` integration in `DarkMatterRates`.

The original source files live under `dielectric_functions/` in the upstream
QCDark2 repository: https://github.com/meganhott/QCDark2

```
composite/Si_comp.h5
composite/Ge_comp.h5
composite/GaAs_comp.h5
composite/SiC_comp.h5
composite/diamond_comp.h5

lfe/Si_lfe.h5
lfe/Ge_lfe.h5
lfe/GaAs_lfe.h5
lfe/SiC_lfe.h5
lfe/diamond_lfe.h5

nolfe/Si_nolfe.h5
nolfe/Ge_nolfe.h5
nolfe/GaAs_nolfe.h5
nolfe/SiC_nolfe.h5
nolfe/diamond_nolfe.h5
```

The `composite` variant uses LFE up to a momentum cutoff and non-LFE above it; `lfe`
uses local field effects throughout; `nolfe` uses none. `composite` is the recommended
default and matches the rates published in arXiv:2603.12326.

Reference: Hott & Singal, arXiv:2603.12326
