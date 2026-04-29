import numericalunits as nu
import torch

from DMeRates.Constants import me_eV, tf_screening


def tfscreening(
    material: str,
    Earr: torch.Tensor,
    qArr: torch.Tensor,
    do_screen: bool,
) -> torch.Tensor:
    """Thomas-Fermi screening on the native (E,q) grids used by legacy QCDark1."""
    tfdict = tf_screening[material]
    eps0 = tfdict["eps0"]
    qTF = tfdict["qTF"]
    omegaP = tfdict["omegaP"]
    alphaS = tfdict["alphaS"]

    Earr = Earr / nu.eV
    qArr = qArr / (nu.eV / nu.c0)
    omegaP_ = omegaP / nu.eV
    qTF_ = qTF / nu.eV
    mElectron_eV = me_eV / nu.eV

    q_arr_tiled = torch.tile(qArr, (len(Earr), 1))
    if do_screen:
        E_array_tiled = torch.tile(Earr, (len(qArr), 1)).T
        result = alphaS * ((q_arr_tiled / qTF_) ** 2)
        result += 1.0 / (eps0 - 1)
        result += q_arr_tiled**4 / (4.0 * (mElectron_eV**2) * (omegaP_**2))
        result -= (E_array_tiled / omegaP_) ** 2
        result = 1.0 / (1.0 + 1.0 / result)
    else:
        result = torch.ones_like(q_arr_tiled)
    return result


def thomas_fermi_screening(
    material: str,
    q: torch.Tensor,
    E: torch.Tensor,
    do_screen: bool = True,
):
    """Vectorized Thomas-Fermi screening used by the integrated path."""
    tfdict = tf_screening[material]
    eps0 = tfdict["eps0"]
    qTF = tfdict["qTF"]
    omegaP = tfdict["omegaP"]
    alphaS = tfdict["alphaS"]
    E_eV = E / nu.eV
    q_eV = q / (nu.eV / nu.c0)
    E_eV = E_eV.unsqueeze(0)
    q_eV = q_eV.unsqueeze(1)

    omegaP_ = omegaP / nu.eV
    qTF_ = qTF / nu.eV
    mElectron_eV = me_eV / nu.eV
    if do_screen:
        term1 = alphaS * (q_eV / qTF_) ** 2
        term2 = 1.0 / (eps0 - 1)
        term3 = q_eV**4 / (4.0 * (mElectron_eV**2) * (omegaP_**2))
        term4 = (E_eV / omegaP_) ** 2

        result = term1.expand(-1, E_eV.shape[1]) + term2
        result += term3.expand(-1, E_eV.shape[1])
        result -= term4.expand(q_eV.shape[0], -1)
        result = 1.0 / (1.0 + 1.0 / result)
    else:
        result = 1.0
    return result
