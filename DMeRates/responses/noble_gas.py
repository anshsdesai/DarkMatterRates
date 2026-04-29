from DMeRates.Constants import ATOMIC_WEIGHT


class formFactorNoble(object):
    # TODO:
    # should change from pickle files to better format to avoid compatibility issues
    def __init__(self, filename):
        from scipy.interpolate import RegularGridInterpolator
        import numpy as np
        import pickle

        if "Xe" in filename:
            self.material = "Xe"
        elif "Ar" in filename:
            self.material = "Ar"
        else:
            raise ValueError("Unknown material")

        with open(filename, mode="rb") as f:
            shell_data = pickle.load(f)
        keys = list(shell_data.keys())
        for _shell_, _sd_ in shell_data.items():
            _sd_["log10ffsquared_itp"] = RegularGridInterpolator(
                (_sd_["lnks"], _sd_["lnqs"]),
                np.log10(_sd_["ffsquared"]),
                bounds_error=False,
                fill_value=-float("inf"),
            )

        self.keys = keys
        self.shell_data = shell_data
        self.mCell = ATOMIC_WEIGHT[self.material]
