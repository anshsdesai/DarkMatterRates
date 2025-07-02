[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15787086.svg)](https://doi.org/10.5281/zenodo.15787086)  [![arXiv](https://img.shields.io/badge/arXiv-2507.00344-B31B1B.svg)](http://arxiv.org/abs/2507.00344) [![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)

# DarkMatterRates
DMeRates is a tool that allows for vectorized calculation of DM-electron scattering rates in Silicon, Germanium, Xenon and Argon. It makes use of form factors from [QCDark](https://github.com/asingal14/QCDark) (see [arXiv:2306.14944](https://arxiv.org/abs/2306.14944)), [QEDark](https://github.com/tientienyu/QEdark) (see [arXiv:1509.01598](https://arxiv.org/abs/1509.01598)), [wimprates](https://github.com/JelleAalbers/wimprates) (see [arXiv:1703.00910](https://arxiv.org/abs/1703.00910)). It is fast, efficient, and takes advantage of a GPU if you have one. 

I wrote this to speed up rate calculations, and primarily for a study of daily modulation. You can calculate your own daily modulated rates by providing your own halo data. This code will make use of halo data in the form of a txt file with the left column being vmin (km/s), and the right column being η(vmin) (s/km).  Halo data used for the study of daily modulation due to Earth scattering [arXiv:2507.00344](http://arxiv.org/abs/2507.00344) is publicly available here: [Dryad](10.5061/dryad.8pk0p2p19), along with instructions on how to use it. You can also use tools such as [DaMaSCUS](https://github.com/temken/DaMaSCUS) and/or [Verne](https://github.com/bradkav/verne) to generate your own halo data. To use the data made public at [Dryad](10.5061/dryad.8pk0p2p19), simply copy the contents of the folder modulated into the folder titled halo_data.
You can use the previously generated rates available at the same link (copy contents of folder rates into the folder titled modulation_study), or you can regenerate the rates once the halo data is copied by running the notebook [modulation_rates_generating.ipynb](modulation_study/modulation_rates_generating.ipynb).



I am still working on documentation, so if you happen to see this before this code is well commented please reach out to me and I can help you with whatever you need. 

### Contents

- `form_factors`: All the form factor information used to calculate rates in Silicon, Germanium, Xenon and Argon. Information on these can be found in the above sources.
- `DMeRates`: The core of the software. Contains relevant Constants in Constants.py (editable if for example the SHM parameters change), the class for calculating DM halo distribution lives in DM_Halo.py, tools for loading in form factor information in form_factor.py, and the actual rate calculation in DMe_Rate.py.
- `limits`: Various experimental limits, used to generate constraint information.
- `senstivity_projections`: Similar to limits, but contains expected sensitivities from Darkside (see [arXiv:2407.05813](https://arxiv.org/abs/2407.05813)) and Oscura (see [arXiv:2202.10518](https://arxiv.org/abs/2202.10518)).
- `torchinterp1d`: submodule not written by me (see included license)
- `modulation_study`: rates, figures and notebook to generate plots for the associated paper.

### Getting started

There are examples in the [example notebook](DMeRates_Examples.ipynb). 


### Dependencies

Detailed dependencies can be found in `requirements.txt`.
Requires [numpy](http://www.numpy.org), [scipy](https://www.scipy.org) and uses [pytorch](https://www.pytorch.org) to speed up rate calculations and make use of GPUs (if you have one). Other requirements are fairly standard [matplotlib](https://matplotlib.org/), [jupyter](https://jupyter.org/install).
It also makes use of a useful package for tracking units called [numericalunits](https://pypi.org/project/numericalunits/)

### Citing
If you make use of this code or numerical results please see the CITATION.cff file as well as citing our relevant [arXiv:2507.00344](http://arxiv.org/abs/2507.00344)

## Papers that used this code
[arXiv:2507.00344](http://arxiv.org/abs/2507.00344)




