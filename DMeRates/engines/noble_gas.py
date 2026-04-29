import numericalunits as nu
import torch

from DMeRates.Constants import skip_keys
from DMeRates.spectrum import RateSpectrum


def noble_rate_dme_shell(
    *,
    mX,
    FDMn: int,
    halo_model: str,
    shell_key: str,
    form_factor,
    qArrdict: dict,
    Earr: torch.Tensor,
    dtype_str: str,
    reduced_mass_fn,
    fdm_fn,
    vmin_tensor_fn,
    get_parametrized_eta_fn,
    halo_id_params=None,
):
    """Extracted noble-gas shell engine from legacy rate_dme_shell()."""
    qArr = qArrdict[shell_key]
    qmax = qArr[-1]
    numq = len(qArr)
    rm = reduced_mass_fn(mX, nu.me)
    prefactor = 1 / (8 * form_factor.mCell * (rm) ** 2)
    prefactor /= nu.c0**2

    fdm_factor = (fdm_fn(qArr, FDMn)) ** 2
    vMins = vmin_tensor_fn(qArr, Earr, mX, shell_key)
    etas = get_parametrized_eta_fn(vMins, mX, halo_model, halo_id_params=halo_id_params)
    ff_arr = form_factor.ff[shell_key]
    result = torch.einsum("j,ij->ij", fdm_factor, etas)
    result *= ff_arr

    import torchquad
    from torchquad import Simpson, set_up_backend

    torchquad.set_log_level("ERROR")
    set_up_backend("torch", data_type=dtype_str)
    simp = Simpson()
    integration_domain = torch.Tensor([[0, qmax]])

    def momentum_integrand(q):
        qint = q.flatten()
        return torch.einsum("j,ij->ji", qint, result)

    integrated_result = (
        simp.integrate(momentum_integrand, dim=1, N=numq, integration_domain=integration_domain)
        / Earr
    )
    integrated_result *= prefactor
    return integrated_result


def noble_dRdE_spectrum(
    *,
    material: str,
    mX,
    FDMn: int,
    halo_model: str,
    form_factor,
    qArrdict: dict,
    Earr: torch.Tensor,
    dtype_str: str,
    reduced_mass_fn,
    fdm_fn,
    vmin_tensor_fn,
    get_parametrized_eta_fn,
    halo_id_params=None,
):
    """Return a shell-resolved RateSpectrum for noble-gas targets."""
    mX = mX * nu.MeV / nu.c0**2
    shell_spectra = {}
    for key in form_factor.keys:
        if key in skip_keys[material]:
            continue
        shell_spectra[key] = noble_rate_dme_shell(
            mX=mX,
            FDMn=FDMn,
            halo_model=halo_model,
            shell_key=key,
            form_factor=form_factor,
            qArrdict=qArrdict,
            Earr=Earr,
            dtype_str=dtype_str,
            reduced_mass_fn=reduced_mass_fn,
            fdm_fn=fdm_fn,
            vmin_tensor_fn=vmin_tensor_fn,
            get_parametrized_eta_fn=get_parametrized_eta_fn,
            halo_id_params=halo_id_params,
        )
    summed = torch.sum(torch.stack(list(shell_spectra.values())), axis=0)
    return RateSpectrum(
        E=Earr,
        dR_dE=summed,
        material=material,
        backend="noble_gas",
        metadata={"FDMn": FDMn},
        shell_spectra=shell_spectra,
        shell_labels=list(shell_spectra.keys()),
    )
