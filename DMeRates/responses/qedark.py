import numericalunits as nu

from DMeRates.Constants import ATOMIC_WEIGHT


class form_factorQEDark(object):
    """
    Class containing our form factor object, including data regarding input.

    To access crystal form factor, please use `form_factor.ff`.
    For other information, please view listed data.

    Early versions of results did not store if the energies had been scissor
    corrected to band gap, and so backwards compatibility requires allowing this
    to be skipped.
    """

    def __init__(self, filename):
        import numpy as np

        nq = 900
        wk = 2 / 137
        nE = 500
        fcrys = np.transpose(np.resize(np.loadtxt(filename, skiprows=1), (nE, nq)))
        if "Si" in filename:
            self.material = "Si"
        elif "Ge" in filename:
            self.material = "Ge"

        """
        materials = {name: [Mcell #eV, Eprefactor, Egap #eV, epsilon #eV, fcrys]}
        N.B. If you generate your own fcrys from QEdark, remove the factor of
        "wk/4" below.
        """
        materials = {
            "Si": [(2 * 28.0855), 2.0, 1.2, 3.8, wk / 4 * fcrys],
            "Ge": [2 * 72.64, 1.8, 0.7, 3, wk / 4 * fcrys],
        }
        self.ff = fcrys * wk / 4
        self.dq = 0.02 * nu.alphaFS * nu.me * nu.c0
        self.dE = 0.1 * nu.eV
        self.mCell = ATOMIC_WEIGHT[self.material]
        self.band_gap = materials[self.material][2] * nu.eV
