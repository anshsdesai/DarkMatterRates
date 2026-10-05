

class Earth_Density_Layer_NU:
    def __init__(self):
        self.Elements = None
        self.GeV = 1.0
        self.MeV	 = 1.0E-3 * self.GeV
        self.eV	 = 1.0E-9 * self.GeV
        self.gram = 5.617977528089887E23 * self.GeV
        self.cm			 = 5.067E13 / self.GeV
        self.meter		 = 100 * self.cm
        self.km			 = 1000 * self.meter
        self.EarthRadius = 6371 *self.km
        self.mNucleon  = 0.932 * self.GeV
        self.mProton  = 0.938 * self.GeV
        self.m  = 0.932 * self.GeV
        self.sec = 299792458 * self.meter
        self.Bohr_Radius = 5.291772083e-11 * self.meter
        self.mElectron = 0.511 * self.MeV
        self.alpha= 1.0 / 137.035999139


    def get_layer(self,r): #inner core
        #r in natural units
        x = r / self.EarthRadius
        # print('converted radius',r/self.km)
        if r < 1221.5*self.km: #km
            # print('Inner Core')
            self.Core()
            self.density = 13.0885 - 8.8381*x**2
        elif r >= 1221.5*self.km and r < 3480*self.km: #outer core
            # print('Outer Core')
            self.Core()
            self.density = 12.5815 - 1.2638*x - 3.6426*x**2 - 5.5281*x**3
        elif r >= 3480*self.km and r < 3630*self.km: #Lower Mantle 1 
            # print('Lower Mantle 1')

            self.Mantle()
            self.density = 7.9565 - 6.47618*x + 5.5283*x**2 - 3.0807*x**3
        elif r >= 3630*self.km and r < 5600*self.km: #Lower Mantle 2
            # print('Lower Mantle 2')

            self.Mantle()
            self.density = 7.9565 - 6.47618*x + 5.5283*x**2 - 3.0807*x**3
        elif r >= 5600*self.km and r < 5701*self.km: #Lower Mantle 3
            # print('Lower Mantle 3')

            self.Mantle()
            self.density = 7.9565 - 6.47618*x + 5.5283*x**2 - 3.0807*x**3

        elif r >= 5701*self.km and r < 5771*self.km:#Transition Zone 1
            # print('Transition Zone 1')

            self.Mantle()
            self.density = 5.3197 - 1.4836*x
        elif r >= 5771*self.km and r < 5971*self.km:#Transition Zone 2
            # print('Transition Zone 2')
            self.Mantle()
            self.density = 11.2494 - 8.0298*x
        elif r >= 5971*self.km and r < 6151*self.km: #Transition Zone 3
            # print('Transition Zone 3')

            self.Mantle()
            self.density = 7.1089-3.8405*x
        elif r >= 6151*self.km and r < 6291*self.km: #LVZ
            # print('LVZ')

            self.Mantle()
            self.density = 2.6910 + 0.6924*x
        elif r >= 6291*self.km and r < 6346.6*self.km: #LID
            # print('LID')
            self.Mantle()
            self.density = 2.6910 + 0.6924*x
        elif r >= 6346.6*self.km and r < 6356*self.km: #crust 1 
            # print('Inner Crust')
            self.Mantle()
            self.density = 2.9
        elif r >= 6356*self.km and r < 6368*self.km: #crust 2
            # print('Outer Crust')
            self.Mantle()
            self.density = 2.6
        # elif r >= 6368 and r < 6371: #ocean
        #     self.Mantle()
        self.density*= self.gram * (self.cm)**(-3) #[GeV^4]
        return

        
        

    def Core(self):
        self.Elements = [
            [26, 56, 0.855],  # # Iron			Fe
            [14, 28, 0.06],	   ## Silicon		Si
            [28, 58, 0.052],   ## Nickel		Ni
            [16, 32, 0.019],   ## Sulfur		S
            [24, 52, 0.009],   ## Chromium		Cr
            [25, 55, 0.003],   ## Manganese    Mn
            [15, 31, 0.002],   ## Phosphorus	P
            [6, 12, 0.002],	   ## Carbon		C
            [1, 1, 0.0006]	   ## Hydrogen		H
        ]
        return

    def Mantle(self):
        self.Elements = [
            [8, 16, 0.440],		# Oxygen		O
		[12, 24, 0.228],	# Magnesium		Mg
		[14, 28, 0.21],		# Silicon		Si
		[26, 56, 0.0626],	# Iron			Fe
		[20, 40, 0.0253],	# Calcium		Ca
		[13, 27, 0.0235],	# Aluminium		Al
		[11, 23, 0.0027],	# Natrium		Na
		[24, 52, 0.0026],	# Chromium		Cr
		[28, 58, 0.002],	# Nickel		Ni
		[25, 55, 0.001],	# Manganese		Mn
		[16, 32, 0.0003],	# Sulfur		S
		[6, 12, 0.0001],	# Carbon		C
		[1, 1, 0.0001],		# Hydrogen		H
		[15, 31, 0.00009]	# Phosphorus	P
        ]
        return 
    
    def NucleusMass(self,N):
        return N*self.mNucleon


 
    


    def muXElem(self,mX,mElem):     
      return mX*mElem/(mX+mElem)



    def sigma_i(self,v,isotope_mass,z,sigmaP,mX,FDMn,doScreen=True):
        import numpy as np
        qmax = 2 *  self.muXElem(mX,isotope_mass) * v
        q2max = qmax*qmax
        qref = self.alpha * self.mElectron
        # qref = 1e-3
        # qref = self.mElectron
        # sigmaP_bar = 18 * pi * alpha*alphaD * epsilon^2 * muXP ^2 / (qref + ma_prime^2)^2
        # a = 1 /4 (9 pi^2 / 2*Z) ^ 1/3
        # a0 = 0.89 *a0 / z^1/3
        a = (1/4)*((9*np.pi**2)/2/z)**(1/3)*self.Bohr_Radius
        
        x = a*a*q2max
        y = a*a * qref*qref
        
        if FDMn == 0 and not doScreen:
            fdm_factor = 1
        elif FDMn == 2:
            doScreen=False
            fdm_factor = y*y / (1+x)
        if doScreen and FDMn == 0:
            fdm_factor = (1+ (1/(1+x)) - (2/x)*np.log(1+x))
        si= sigmaP*((self.muXElem(mX,isotope_mass)/self.muXElem(mX,self.mNucleon))**2) *(z**2)* fdm_factor 

        return si #same units as sigmaP



    def Mean_Free_Path(self,r,mX,sigmaP,v,FDMn,doScreen=True):
        #r in natural units
        #mX in GeV
        #v in c
        #sigmaP in cm^2
        #convert sigmaP into energy units
        

        lambda_inv = 0
        sigmaP *= self.cm**2# [1/ev^2]
        self.get_layer(r) 
        density = self.density #natural units
        num_isotopes = len(self.Elements)
        for i in range(num_isotopes):
            Element= self.Elements[i]
            fractional_density = Element[2]
            Z = Element[0]
            N = Element[1]
            isotope_mass = self.NucleusMass(N) #GeV
            si = self.sigma_i(v,isotope_mass,Z,sigmaP,mX,FDMn,doScreen)
            # print('fractional_density,density/isotope_mass,si')
            # print(fractional_density,density/isotope_mass,si)
# 
            lambda_inv+= fractional_density * (density/isotope_mass) *si #in units of GeV
        
        mfp = 1/lambda_inv #[1/GeV]
        mfp /= self.EarthRadius
        return mfp
    




# ---------------------------------------------------------------------------
# Flux-weighted SRDM mean-free-path helpers
# ---------------------------------------------------------------------------

def _as_numpy(value):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=float)


def sigma_e_to_sigma_p_cm2(sigma_e_cm2, mX_MeV):
    """Convert sigma_e_bar to sigma_p_bar with the existing DMeRates convention."""
    earth = Earth_Density_Layer_NU()
    mX_GeV = float(mX_MeV) * earth.MeV
    return float(sigma_e_cm2) * (
        earth.muXElem(mX_GeV, earth.mProton) / earth.muXElem(mX_GeV, earth.mElectron)
    ) ** 2


def _direct_srdm_manifest_points(FDMn=2, mediator_spin="vector", grid_family="srdm_fdmq2_source_v1"):
    """Return direct SRDM manifest points for the requested light-mediator grid."""
    from DMeRates.srdm.manifest import load_manifest
    from DMeRates.srdm.mediators import normalize_mediator_spin

    spin = normalize_mediator_spin(mediator_spin)
    points = []
    for entry in load_manifest():
        if int(entry.get("FDMn", -1)) != int(FDMn):
            continue
        if entry.get("mediator_spin") != spin:
            continue
        if grid_family is not None and entry.get("grid_family") != grid_family:
            continue
        points.append(
            {
                "mX_MeV": float(entry["mX_eV"]) / 1.0e6,
                "mX_eV": float(entry["mX_eV"]),
                "sigma_e_cm2": float(entry["sigma_e_cm2"]),
                "FDMn": int(entry["FDMn"]),
                "mediator_spin": entry["mediator_spin"],
                "grid_family": entry.get("grid_family"),
                "filename": entry.get("filename"),
            }
        )
    return sorted(points, key=lambda row: (row["mX_MeV"], row["sigma_e_cm2"]))


def flux_weighted_srdm_mean_free_path(
    mX_MeV,
    sigma_e_cm2,
    *,
    r_fraction=0.8,
    FDMn=2,
    mediator_spin="vector",
    doScreen=True,
):
    """Return flux-weighted SRDM MFP in Earth radii for a direct SRDM flux file.

    This computes ``1 / <1/lambda(v)>_Phi`` using the direct incoming SRDM flux
    spectrum from ``halo_data/srdm/manifest.json``. It is intended as the SRDM
    analogue of the diagnostic MFP contours used for SHM Earth scattering.
    """
    import numpy as np
    from DMeRates.srdm.flux_loader import load_srdm_flux

    if int(FDMn) != 2:
        raise ValueError("flux_weighted_srdm_mean_free_path is currently defined for FDMn=2")

    earth = Earth_Density_Layer_NU()
    mX_GeV = float(mX_MeV) * earth.MeV
    sigma_p_cm2 = sigma_e_to_sigma_p_cm2(sigma_e_cm2, mX_MeV)
    v_over_c, dphi_dv = load_srdm_flux(float(mX_MeV) * 1.0e6, float(sigma_e_cm2), FDMn, mediator_spin)
    velocities = _as_numpy(v_over_c)
    weights = _as_numpy(dphi_dv)
    mask = np.isfinite(velocities) & np.isfinite(weights) & (velocities > 0.0) & (weights > 0.0)
    velocities = velocities[mask]
    weights = weights[mask]
    if velocities.size == 0:
        raise ValueError(f"No positive SRDM flux weights for mX={mX_MeV}, sigma_e={sigma_e_cm2}")

    r = float(r_fraction) * earth.EarthRadius
    mfps = np.asarray(
        [earth.Mean_Free_Path(r, mX_GeV, sigma_p_cm2, float(v), FDMn, doScreen=doScreen) for v in velocities],
        dtype=float,
    )
    valid = np.isfinite(mfps) & (mfps > 0.0)
    if not np.any(valid):
        raise ValueError(f"No finite MFP values for mX={mX_MeV}, sigma_e={sigma_e_cm2}")
    velocities = velocities[valid]
    weights = weights[valid]
    inv_mfp = 1.0 / mfps[valid]

    weight_integral = np.trapezoid(weights, velocities)
    if weight_integral <= 0.0:
        raise ValueError(f"SRDM flux integral is non-positive for mX={mX_MeV}, sigma_e={sigma_e_cm2}")
    mean_inv_mfp = np.trapezoid(weights * inv_mfp, velocities) / weight_integral
    return 1.0 / mean_inv_mfp


def get_srdm_flux_weighted_mfp_points(
    *,
    FDMn=2,
    mediator_spin="vector",
    grid_family="srdm_fdmq2_source_v1",
    r_fraction=0.8,
    doScreen=True,
    verbose=False,
):
    """Return arrays of direct-grid SRDM flux-weighted MFP values."""
    import numpy as np
    from tqdm.autonotebook import tqdm

    points = _direct_srdm_manifest_points(FDMn=FDMn, mediator_spin=mediator_spin, grid_family=grid_family)
    masses = []
    sigmaEs = []
    mfps = []
    for point in tqdm(points, desc="Calculating SRDM flux-weighted MFP"):
        try:
            mfp = flux_weighted_srdm_mean_free_path(
                point["mX_MeV"],
                point["sigma_e_cm2"],
                r_fraction=r_fraction,
                FDMn=FDMn,
                mediator_spin=mediator_spin,
                doScreen=doScreen,
            )
        except (FileNotFoundError, ValueError) as exc:
            if verbose:
                print(f"Skipping SRDM MFP point {point}: {exc}")
            continue
        masses.append(point["mX_MeV"])
        sigmaEs.append(point["sigma_e_cm2"])
        mfps.append(mfp)
    return np.asarray(masses), np.asarray(sigmaEs), np.asarray(mfps)


def get_srdm_flux_weighted_mfp_contour_data(
    *,
    FDMn=2,
    mediator_spin="vector",
    grid_family="srdm_fdmq2_source_v1",
    r_fraction=0.8,
    doScreen=True,
    grid_size=300,
    method="linear",
    verbose=False,
):
    """Interpolate flux-weighted SRDM MFP values onto a log-log contour grid."""
    import numpy as np
    from scipy.interpolate import griddata

    masses, sigmaEs, mfps = get_srdm_flux_weighted_mfp_points(
        FDMn=FDMn,
        mediator_spin=mediator_spin,
        grid_family=grid_family,
        r_fraction=r_fraction,
        doScreen=doScreen,
        verbose=verbose,
    )
    if masses.size < 3:
        raise ValueError("Need at least three SRDM MFP points to build a contour grid")
    log_masses = np.log10(masses)
    log_sigmas = np.log10(sigmaEs)
    log_mass_axis = np.linspace(log_masses.min(), log_masses.max(), grid_size)
    log_sigma_axis = np.linspace(log_sigmas.min(), log_sigmas.max(), grid_size)
    log_mass_grid, log_sigma_grid = np.meshgrid(log_mass_axis, log_sigma_axis)
    log_mfp_grid = griddata(
        points=(log_masses, log_sigmas),
        values=np.log10(mfps),
        xi=(log_mass_grid, log_sigma_grid),
        method=method,
    )
    return 10**log_mass_axis, 10**log_sigma_axis, 10**log_mfp_grid


def plot_srdm_flux_weighted_mfp(
    *,
    FDMn=2,
    mediator_spin="vector",
    grid_family="srdm_fdmq2_source_v1",
    r_fraction=0.8,
    doScreen=True,
    grid_size=300,
    overlay_fractional_modulation=True,
    fractional_threshold=0.03,
    modulation_kwargs=None,
    plot_solar_constraints=False,
    include_all_constraints=False,
    include_freeze_in=False,
    solar_constraint_kwargs=None,
    all_constraint_kwargs=None,
    freeze_in_kwargs=None,
    xlim=None,
    ylim=None,
    ax=None,
    savefig=False,
    outfile=None,
    verbose=False,
):
    """Plot the flux-weighted SRDM MFP contour for the direct SRDM light-mediator grid."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors

    masses, sigmaEs, mfp_grid = get_srdm_flux_weighted_mfp_contour_data(
        FDMn=FDMn,
        mediator_spin=mediator_spin,
        grid_family=grid_family,
        r_fraction=r_fraction,
        doScreen=doScreen,
        grid_size=grid_size,
        verbose=verbose,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    else:
        fig = ax.figure
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_\chi$ [MeV]")
    ax.set_ylabel(r"$\overline{\sigma}_e$ [cm$^2$]")

    finite = mfp_grid[np.isfinite(mfp_grid) & (mfp_grid > 0.0)]
    if finite.size == 0:
        raise ValueError("No finite positive SRDM MFP values were available")
    low_exp = np.floor(np.log10(np.nanmin(finite)))
    high_exp = np.ceil(np.log10(np.nanmax(finite)))
    levels = np.power(10.0, np.arange(low_exp, high_exp + 0.5, 0.5))
    contour = ax.contourf(
        masses,
        sigmaEs,
        mfp_grid,
        levels=levels,
        norm=colors.LogNorm(vmin=levels[0], vmax=levels[-1]),
        cmap="Reds_r",
        extend="both",
    )
    cbar = fig.colorbar(contour, ax=ax)
    cbar.ax.set_title(r"$\lambda_{\rm eff}/R_\oplus$", fontsize=16)

    line_levels = [level for level in [1e-4, 1e-2, 1.0, 1e2, 1e4] if finite.min() <= level <= finite.max()]
    if line_levels:
        lines = ax.contour(masses, sigmaEs, mfp_grid, levels=line_levels, colors="white", linewidths=1.5)
        ax.clabel(lines, fmt=lambda value: f"$10^{{{int(np.round(np.log10(value)))}}}$", fontsize=14)
    if finite.min() <= 1.0 <= finite.max():
        earth_line = ax.contour(masses, sigmaEs, mfp_grid, levels=[1.0], colors="black", linewidths=2.0)
        ax.clabel(earth_line, fmt={1.0: r"$\lambda_{\rm eff}=R_\oplus$"}, fontsize=14)

    if overlay_fractional_modulation:
        try:
            from modulation_study.Modulation import get_srdm_solar_reflection_contour_data

            kwargs = dict(
                material="Si",
                FDMn=FDMn,
                location="JUNO",
                date=[8, 8, 2024],
                ne=1,
                fractional=True,
                modulated_source="Verne",
                screening="rpa",
                variant="composite",
                mediator_spin=mediator_spin,
                form_factor_type="qcdark2",
                grid_size=grid_size,
            )
            if modulation_kwargs:
                kwargs.update(modulation_kwargs)
            mod_masses, mod_sigmas, frac_grid = get_srdm_solar_reflection_contour_data(**kwargs)
            frac_line = ax.contour(
                mod_masses,
                mod_sigmas,
                frac_grid,
                levels=[float(fractional_threshold)],
                colors="black",
                linestyles="--",
                linewidths=2.0,
            )
            ax.clabel(frac_line, fmt={float(fractional_threshold): f"{100*fractional_threshold:g}% daily"}, fontsize=14)
        except Exception as exc:
            if verbose:
                print(f"Could not overlay fractional modulation contour: {exc}")

    if plot_solar_constraints:
        try:
            from modulation_study.Modulation import plot_solar_constraints_overlay

            plot_solar_constraints_overlay(
                ax,
                FDMn,
                include_all_constraints=include_all_constraints,
                include_freeze_in=include_freeze_in,
                solar_kwargs=solar_constraint_kwargs,
                all_kwargs=all_constraint_kwargs,
                freeze_in_kwargs=freeze_in_kwargs,
                verbose=verbose,
            )
        except Exception as exc:
            if verbose:
                print(f"Could not overlay solar constraints: {exc}")

    ax.set_title("Flux-weighted SRDM Light-Mediator Mean Free Path")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if savefig:
        if outfile is None:
            outfile = "srdm_flux_weighted_mfp.png"
        fig.savefig(outfile, bbox_inches="tight", dpi=180)
    return fig, ax, contour
