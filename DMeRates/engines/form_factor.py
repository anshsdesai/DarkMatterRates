import numericalunits as nu
import torch

from DMeRates.spectrum import RateSpectrum

# Legacy QEDark convention: grid values summed directly without dE bin-width factor.
# Equivalent to dE * 10 for the current 0.1 eV grid. See tests/phase0_2.md.
_LEGACY_QEDARK_ENERGY_NORM = 1.0 * nu.eV


def semiconductor_dRdE_spectrum(
    *,
    material: str,
    mX,
    FDMn: int,
    halo_model: str,
    DoScreen: bool,
    integrate: bool,
    QEDark: bool,
    form_factor,
    qArr: torch.Tensor,
    Earr: torch.Tensor,
    Ei_array: torch.Tensor,
    dtype_str: str,
    reduced_mass_fn,
    fdm_fn,
    get_parametrized_eta_fn,
    vmin_tensor_fn,
    tfscreening_fn,
    thomas_fermi_screening_fn,
    halo_id_params=None,
):
    """Extracted semiconductor differential-rate engine for QEDark/QCDark1."""
    mX = mX * nu.MeV / nu.c0**2
    rm = reduced_mass_fn(mX, nu.me)
    prefactor = nu.alphaFS * ((nu.me / rm) ** 2) * (1 / form_factor.mCell)
    ff_arr = torch.tensor(form_factor.ff, dtype=torch.get_default_dtype())

    if integrate:
        import torchquad
        from torchquad import Simpson, set_up_backend

        torchquad.set_log_level("ERROR")
        set_up_backend("torch", data_type=dtype_str)
        simp = Simpson()
        numq = len(qArr)
        qmin = qArr[0]
        qmax = qArr[-1]
        integration_domain = torch.tensor([[qmin, qmax]], dtype=torch.get_default_dtype())

        def vmin(q, E, mX_):
            term1 = E.unsqueeze(0) / q.unsqueeze(1)
            term2 = q.unsqueeze(1) / (2 * mX_)
            return term1 + term2

        def eta_func(vMin):
            return get_parametrized_eta_fn(vMin, mX, halo_model, halo_id_params=halo_id_params)

        def momentum_integrand(q):
            q = q.flatten()
            qdenom = 1 / q**2
            qdenom *= (fdm_fn(q, FDMn)) ** 2
            eta = eta_func(vmin(q, Earr, mX))
            tf_f = (thomas_fermi_screening_fn(q, Earr, doScreen=DoScreen)) ** 2
            ff_f = ff_arr[:-1, :]
            result = eta * tf_f * ff_f
            result = torch.einsum("i,ji->ji", Earr, result)
            result = torch.einsum("j,ji->ji", qdenom, result)
            return result

        integrated_result = (
            simp.integrate(momentum_integrand, dim=1, N=numq, integration_domain=integration_domain)
            / Earr
        )
    else:
        fdm_factor = (fdm_fn(qArr, FDMn)) ** 2
        vMins = vmin_tensor_fn(qArr, Earr, mX)
        etas = get_parametrized_eta_fn(vMins, mX, halo_model, halo_id_params=halo_id_params)
        if QEDark:
            ff_arr = ff_arr[:, Ei_array - 1]
        ff_arr = ff_arr.T
        tf_factor = (tfscreening_fn(DoScreen) ** 2)
        result = torch.einsum("i,ij->ij", Earr, torch.ones_like(etas))
        result *= etas
        result *= fdm_factor
        result *= ff_arr
        result *= tf_factor
        qdenom = 1 / qArr
        result = torch.einsum("j,ij->ij", qdenom, result)
        integrated_result = torch.sum(result, axis=1) / Earr

    integrated_result *= prefactor
    integrated_result /= nu.c0
    band_gap_result = torch.where(Earr < form_factor.band_gap, 0, integrated_result)

    backend = "qedark" if QEDark else "qcdark1"
    return RateSpectrum(
        E=Earr,
        dR_dE=band_gap_result,
        material=material,
        backend=backend,
        metadata={"integrate": integrate, "DoScreen": DoScreen, "FDMn": FDMn},
    )


def semiconductor_rates_from_spectrum(
    spectrum: RateSpectrum,
    prob_fn_tiled: torch.Tensor,
    *,
    integrate: bool,
) -> torch.Tensor:
    """Convert semiconductor dR/dE spectra into dR/dn_e for requested bins."""
    if integrate:
        return torch.trapezoid(spectrum.dR_dE * prob_fn_tiled, x=spectrum.E, axis=1)
    return torch.sum(spectrum.dR_dE * prob_fn_tiled * _LEGACY_QEDARK_ENERGY_NORM, axis=1)
