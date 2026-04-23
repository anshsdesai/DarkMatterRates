from .Constants import *
from .DM_Halo import DM_Halo_Distributions
from .form_factor import form_factor
from .interpolation import interp1d
from .backends import QCDarkBackend, QCDark2Backend, QEDarkBackend, WimpratesBackend, simpson_uniform
from .response import build_probability_table, rebuild_step_probability_table
from .units import (
    LIGHT_SPEED_KM_PER_S as LIGHT_SPEED,
    qcdark2_astro_model_from_numeric,
    qcdark2_astro_model_from_unitful,
)
import numericalunits as nu











_SEMICONDUCTOR_MATERIALS = ('Si', 'Ge', 'GaAs', 'SiC', 'Diamond')
_NOBLE_MATERIALS = ('Xe', 'Ar')

class DMeRate:
    """Class for calculating dark matter-electron scattering rates in various materials.

    Provides methods for computing DM-electron interaction rates in:
    - Semiconductor crystals (Si, Ge, GaAs, SiC, Diamond)
    - Noble elements (Xe, Ar)

    Args:
        material (str): Target material ('Si', 'Ge', 'GaAs', 'SiC', 'Diamond', 'Xe', 'Ar')
        form_factor_type (str): Form factor source: 'qcdark' (default), 'qedark', or 'qcdark2'.
        qcdark2_variant (str): QCDark2 variant when form_factor_type='qcdark2'.
            One of 'composite' (default), 'lfe', or 'nolfe'.
        device (str, optional): Computation device ('cpu', 'cuda', etc.)
    """
    def __init__(self, material, form_factor_type=None,
                 qcdark2_variant='composite', device=None):
        """Initialize rate calculator with material properties and computation settings."""

        import torch
        import os

        # Resolve form_factor_type from legacy QEDark bool if not given explicitly
        if material in _NOBLE_MATERIALS:
            if form_factor_type not in (None, 'wimprates'):
                raise ValueError(
                    f"Noble gas material '{material}' uses form_factor_type='wimprates'."
                )
            form_factor_type = 'wimprates'
        elif form_factor_type is None:
            form_factor_type = 'qedark' if QEDark else 'qcdark'
        self.form_factor_type = form_factor_type
        self.qcdark2_variant = qcdark2_variant

        self.module_dir = os.path.dirname(__file__)
        self.v0 = v0
        self.vEarth = vEarth
        self.vEscape = vEscape
        self.rhoX = rhoX
        self.cross_section = crosssection
        self.material = material
        self.backend = None
        self.qcdark2_backend = None
        self._astro_numeric = qcdark2_astro_model_from_unitful(
            v0,
            vEarth,
            vEscape,
            rhoX,
            crosssection,
        )
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()

        if cuda_available:
            print("CUDA GPU found, performing calculations on GPU")
            self.device = 'cuda'
            self.default_dtype = torch.float64
            self.dtype_str = 'float64'
        # elif mps_available: commented out since mps is limited to 32 floats, which can cause issues
        #     print("MPS GPU found, performing calculations on GPU. Setting default dtype to float32.")
        #     self.device = 'mps'
        #     self.default_dtype = torch.float32
        #     self.dtype_str = 'float32'

        else:
            print("CUDA/MPS GPU not found, performing calculations on cpu (if you are doing this on apple silicon you can change your device to mps if you'd like)")
            self.device = 'cpu'
            self.default_dtype = torch.float64
            self.dtype_str = 'float64'

        if device is not None:
            print(f"You have manually specified your device to be: {device}. Overriding default")
            self.device = device

        torch.set_default_device(self.device)
        torch.set_default_dtype(self.default_dtype)
        self.DM_Halo = DM_Halo_Distributions(self.v0,self.vEarth,self.vEscape,self.rhoX,self.cross_section)

        if material in _SEMICONDUCTOR_MATERIALS:
            self.backend = self._build_semiconductor_backend()
            self.probabilities = build_probability_table(self)

        elif material in _NOBLE_MATERIALS:
            self.backend = WimpratesBackend(material, self.module_dir)
            self.backend.attach(self)

        else:
            raise ValueError(f"Material '{material}' is not supported. "
                             f"Semiconductors: {_SEMICONDUCTOR_MATERIALS}, "
                             f"Noble gases: {_NOBLE_MATERIALS}.")
        
    def _build_semiconductor_backend(self):
        """Select and attach the configured semiconductor source-family backend."""
        if self.form_factor_type == 'qcdark':
            backend = QCDarkBackend(self.material, self.module_dir)
            backend.attach(self)
            return backend
        if self.form_factor_type == 'qedark':
            backend = QEDarkBackend(self.material, self.module_dir)
            backend.attach(self)
            return backend
        if self.form_factor_type == 'qcdark2':
            return QCDark2Backend.build_for_rate(
                self,
                self.material,
                self.qcdark2_variant,
            )
        raise ValueError(
            f"Unknown form_factor_type '{self.form_factor_type}'. "
            f"Choose from: 'qcdark', 'qedark', 'qcdark2'."
        )


    

    def update_params(self,v0,vEarth,vEscape,rhoX,crosssection):
        """Update DM halo parameters.
        
        Args:
            v0 (float): Most probable DM velocity (km/s)
            vEarth (float): Earth's velocity (km/s)
            vEscape (float): Galactic escape velocity (km/s)
            rhoX (float): Local DM density (eV/cm^3)
            crosssection (float): DM cross section (cm^2)
        """
        #assuming values passed in are km/s,km/s,km/s,eV/cm^3,cm^2
        #masses must be in eV if passed in 
        self.v0 = v0* (nu.km / nu.s)
        self.vEarth = vEarth* (nu.km / nu.s)
        self.vEscape = vEscape* (nu.km / nu.s)
        self.rhoX = rhoX* nu.eV / (nu.cm**3)
        self.cross_section = crosssection* nu.cm**2
        self._astro_numeric.update(
            qcdark2_astro_model_from_numeric(v0, vEarth, vEscape, rhoX, crosssection)
        )
        self.DM_Halo = DM_Halo_Distributions(self.v0,self.vEarth,self.vEscape,self.rhoX)

   
    def step_probabilities(self,ne):
        """Step function approximation for electron-hole pair creation probabilities.
        
        Args:
            ne (int): Number of electron-hole pairs
            
        Returns:
            probabilities for being in certain bin
        """
        import torch
        i = ne - 1
        dE, E_gap = self.form_factor.dE, self.form_factor.band_gap
        dE /=nu.eV
        E_gap /=nu.eV
        E2Q = self.bin_size / nu.eV

        initE, binE = int((E_gap)/(dE)), int(round(E2Q/dE))
        # bounds = (i*binE + initE,(i+1)*binE + initE)
        # if self.QEDark:
        bounds = (i*binE + initE + 1,(i+1)*binE + initE + 1)
        probabilities = torch.zeros_like(self.Earr)
        probabilities[bounds[0]:bounds[1]] = 1
        return probabilities

    def RKProbabilities(self,ne): #using values at 100k
        """Interpolated probabilities from Ramanathan Kurinsky data for electron-hole pair creation.
           See https://arxiv.org/abs/2004.10709
        Args:
            ne (int): Number of electron-hole pairs
            
        Returns:
            Array of probabilities for each energy bin
        """
        from numpy import loadtxt
        import torch
        import os
        filepath = os.path.join(self.module_dir,'p100k.dat')
        p100data = loadtxt(filepath)
        pEV = torch.tensor(p100data[:,0],dtype=torch.get_default_dtype()) *nu.eV
        
        file_probabilities = torch.tensor(p100data.T,dtype=torch.get_default_dtype())#[:,:]
        file_probabilities = file_probabilities[ne]

        probabilities = interp1d(pEV, file_probabilities,self.Earr).flatten()# kind = 'linear',bounds_error=False,fill_value=0)
        # probabilities = p100_func(self.Earr)
        # probabilities = torch.from_numpy(probabilities)
        # probabilities = probabilities.to(self.device)
        return probabilities

    def update_crosssection(self,crosssection):
        """Update DM-electron cross section.
        
        Args:
            crosssection (float): New cross section in cm^2
        """
        #assuming value is in cm*2
        self.cross_section = crosssection *nu.cm**2
        self._astro_numeric['sigma_e'] = float(crosssection)

    def _coerce_1d_tensor(self, values):
        """Convert scalars/sequences/arrays to a 1D tensor on the active device."""
        import numpy as np
        import torch

        if isinstance(values, torch.Tensor):
            tensor = values.to(self.device)
        elif isinstance(values, np.ndarray):
            tensor = torch.from_numpy(values).to(self.device)
        elif isinstance(values, (list, tuple)):
            tensor = torch.tensor(values, device=self.device)
        else:
            tensor = torch.tensor([values], device=self.device)

        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _current_astro_model_numeric(self):
        """Return the explicit numeric astro model used by the QCDark2 backend."""
        return dict(self._astro_numeric)

    def _qcdark2_eta_grid(self, mX_mev, halo_model, halo_id_params=None):
        """Return eta(vmin) on the QCDark2 q/E grid in upstream units."""
        import torch

        astro_model = self._current_astro_model_numeric()
        mX_eV = float(mX_mev) * 1e6
        device = self.device
        dtype = self.default_dtype

        if halo_id_params is not None:
            vmins_kms = self.qcdark2_backend.vmin_grid_kms(mX_eV, device, dtype)
            params = halo_id_params.to(self.device) if isinstance(halo_id_params, torch.Tensor) else torch.tensor(
                halo_id_params,
                device=self.device,
                dtype=dtype,
            )
            eta_unitful = self.DM_Halo.step_function_eta(vmins_kms * (nu.km / nu.s), params)
            return eta_unitful / (nu.s / nu.km) * LIGHT_SPEED

        if halo_model in ('shm', 'imb'):
            return self.qcdark2_backend.eta_mb(mX_eV, astro_model, device, dtype)

        if not hasattr(self, 'file_vmins') or not hasattr(self, 'file_etas'):
            raise RuntimeError(
                "Halo data has not been loaded for this qcdark2 calculation. "
                "Call setup_halo_data(...) before evaluating file-based halo models."
            )

        file_vmins = (self.file_vmins / (nu.km / nu.s)).detach().cpu().numpy()
        file_etas = (self.file_etas / (nu.s / nu.km)).detach().cpu().numpy()
        return self.qcdark2_backend.eta_from_file(mX_eV, file_vmins, file_etas, device, dtype)


    def FDM(self,q,n):
        """Dark matter form factor calculation.
        
        Args:
            q: Momentum transfer
            n: Form factor model (0=FDM1, 1=FDMq, 2=FDMq2)
            
        Returns:
            Form factor value
        """
        # me_energy = (nu.me * nu.c0**2)
        """
        DM form factor
        n = 0: FDM=1, heavy mediator
        n = 1: FDM~1/q, electric dipole
        n = 2: FDM~1/q^2, light mediator
        """
        return (nu.alphaFS*  nu.me * nu.c0 /q)**n
    
    # def vmin_tensor(self,E,q,mX):
    #     import torch
    #     q_tiled = torch.tile(q,(len(E),1))
    #     EE_tiled = torch.tile(E,(len(q),1)).T
    #     vmins = ((EE_tiled/q_tiled)+(q_tiled/(2*mX)))
    #     return  vmins

    def read_output(self,fileName):
        """Read form factor data from file.
        
        Args:
            fileName: Path to form factor file
            
        Returns:
            Form factor object
        """
        """Read Input File"""
        return form_factor(fileName)
    
    def reduced_mass(self,mass1,mass2):
        """Calculate reduced mass of two particles.
        
        Args:
            mass1: First particle mass
            mass2: Second particle mass
            
        Returns:
            Reduced mass
        """
        return mass1*mass2/(mass1+mass2)
    
    def change_to_step(self):
        """Switch to step function probability model."""
        rebuild_step_probability_table(self)
    

    def TFscreening(self,DoScreen):
        """Thomas-Fermi screening calculation for semiconductors.
        
        Args:
            DoScreen (bool): Whether to apply screening
            
        Returns:
            Screening factor array
        """
        import torch
        tfdict = tf_screening[self.material]
        eps0,qTF,omegaP,alphaS = tfdict['eps0'],tfdict['qTF'],tfdict['omegaP'],tfdict['alphaS']
        Earr = self.Earr / nu.eV
        qArr = self.qArr / (nu.eV / nu.c0)
        omegaP_ = omegaP/nu.eV
        qTF_ = qTF/nu.eV
        
        mElectron_eV = me_eV/nu.eV

        q_arr_tiled = torch.tile(qArr,(len(Earr),1))
        if DoScreen:
            E_array_tiled= torch.tile(Earr,(len(qArr),1)).T
            result = alphaS*((q_arr_tiled/qTF_)**2)
            result += 1.0/(eps0 - 1)
            result += q_arr_tiled**4/(4.*(mElectron_eV**2)*(omegaP_**2))
            result -= (E_array_tiled/omegaP_)**2
            result = 1. / (1. + 1. / result)
        else:
            result = torch.ones_like(q_arr_tiled)
        return result
    
    def thomas_fermi_screening(self,q,E,doScreen=True):
        """Vectorized Thomas-Fermi screening calculation.
        
        Args:
            q: Momentum transfer values
            E: Energy values
            doScreen: Whether to apply screening
            
        Returns:
            Screening factor tensor
        """
        #param q, tensor with shape 1250
        #param E, tensor with shape 500
        tfdict = tf_screening[self.material]
        eps0,qTF,omegaP,alphaS = tfdict['eps0'],tfdict['qTF'],tfdict['omegaP'],tfdict['alphaS']
        E_eV = E / nu.eV
        q_eV = q/ (nu.eV / nu.c0)
        E_eV = E_eV.unsqueeze(0)
        q_eV = q_eV.unsqueeze(1)


        omegaP_ = omegaP/nu.eV
        qTF_ = qTF/nu.eV
        
        mElectron_eV = me_eV/nu.eV
        if doScreen:

            term1 = alphaS * (q_eV / qTF_)**2  # [1249,1] * [1,1] → [1249,1]
            term2 = 1.0 / (eps0 - 1)           # Scalar → [1,1]
            term3 = q_eV**4 / (4. * (mElectron_eV**2) * (omegaP_**2))  # [1249,1]
            term4 = (E_eV / omegaP_)**2        # [1,500] → [1,500]
            
            # Explicitly broadcast all terms to [1249,500]
            result = term1.expand(-1, E_eV.shape[1]) + term2
            result += term3.expand(-1, E_eV.shape[1])
            result -= term4.expand(q_eV.shape[0], -1)
            
            result = 1. / (1. + 1. / result)



            # result = alphaS*((q_eV/qTF_)**2)
            # result += 1.0/(eps0 - 1)
            # result += q_eV**4/(4.*(mElectron_eV**2)*(omegaP_**2))
            # result -= (E_eV/omegaP_)**2
            # result = 1. / (1. + 1. / result)
        else:
            result = 1.
        return result
    
    

    def setup_halo_data(self,mX,FDMn,halo_model,isoangle=None,useVerne=False,calcErrors=None):
        """Load or generate velocity distribution data for given parameters.
        
        Args:
            mX: DM mass
            FDMn: Form factor model [0 or 2]
            halo_model: Velocity distribution model ('shm' for standard halo model, 'modulated' for march data, 'summer' for june data)
            isoangle: isoangle (for modulated distributions) (number from 0 -35 with the true isoangle being 5x that value)
            useVerne: Use Verne distribution
            calcErrors: Calculate errors ('High'/'Low') (only works for DaMaSCUS data)
        """
        import os
        import torch
        torch.set_default_device(self.device)
        torch.set_default_dtype(self.default_dtype)
        if halo_model == 'imb' or halo_model == 'step':
            return
        if isoangle is None:
            lightSpeed_kmpers = nu.s / nu.km #inverse to output it in units i want

            geVconversion = 1 / (nu.GeV / nu.c0**2/ nu.cm**3)
            halo_prefix = '../halo_data/'

            halo_dir_prefix = os.path.join(self.module_dir,halo_prefix) 
            file = halo_dir_prefix + f'{halo_model}_v0{round(self.v0*lightSpeed_kmpers,1)}_vE{round(self.vEarth*lightSpeed_kmpers,1)}_vEsc{round(self.vEscape*lightSpeed_kmpers,1)}_rhoX{round(self.rhoX*geVconversion,1)}.txt'
            try:

                temp =open(file,'r')
                temp.close()
            except FileNotFoundError:
                self.DM_Halo.generate_halo_files(halo_model)
            # print(f'found halo file: {file}')
            from numpy import loadtxt
            try:
                data = loadtxt(file,delimiter='\t')
            except ValueError:
                try:
                    self.DM_Halo.generate_halo_files(halo_model)
                except:
                    raise ValueError("Unknown halo type")
                data = loadtxt(file,delimiter='\t')
                
            if len(data) == 0:
                raise ValueError('file is empty!')
            
            #default file units
            file_etas = torch.tensor(data[:,1],dtype=torch.get_default_dtype()) * nu.s / nu.km
            file_vmins = torch.tensor(data[:,0],dtype=torch.get_default_dtype()) * nu.km / nu.s

            #clearly this was hardcoded to catch something but don't remember what
            if file_etas[-1] == file_etas[-2]:
                file_etas = file_etas[:-1]
                file_vmins = file_vmins[:-1]
        else:
            import re
            import os

            # mass_string = mX / (nu.MeV / nu.c0**2) #turn into MeV
            mass_string = float(mX)
            from numpy import round as npround
            mass_string = npround(mass_string,3)

            mass_string = str(mass_string)
            mass_string = mass_string.replace('.',"_")
            sigmaE = float(format(self.cross_section / nu.cm**2, '.3g'))
            sigmaE_str = str(sigmaE)
            # sigmaE_str.replace('.',"_")
            fdm_str = 'FDM1' if FDMn ==0 else 'FDMq2'
            summer_str ='_summer' if halo_model == 'summer' else '' #only works for Verne
       
            halo_prefix = f'../halo_data/modulated/{fdm_str}/'

            halo_dir_prefix = os.path.join(self.module_dir,halo_prefix) 
            if useVerne:
                dir = halo_dir_prefix + f'Verne{summer_str}/'
            
            else:
                dir = halo_dir_prefix + f'DaMaSCUS/'
            
            # if 'summer' in halo_model or 'winter' in halo_model: 
            #     file = f'{dir}DM_Eta_theta_{isoangle}.txt'
                
            # else:
            file = f'{dir}mDM_{mass_string}_MeV_sigmaE_{sigmaE_str}_cm2/DM_Eta_theta_{isoangle}.txt'
            
            # print(f"Using Halo Data from: {file}")
            if not os.path.isfile(file):
                print(file)
                raise FileNotFoundError('sigmaE file not found')
            
            from numpy import loadtxt
            try:
                data = loadtxt(file,delimiter='\t')
            except ValueError:
                print(file)
                raise ValueError(f'file not found! tried {file}')
            if len(data) == 0:
                raise ValueError('file is empty!')
            
            file_etas = torch.tensor(data[:,1],dtype=self.default_dtype) * nu.s / nu.km
            file_vmins = torch.tensor(data[:,0],dtype=self.default_dtype)* nu.km / nu.s
            if isoangle is not None:
                if calcErrors is not None:
                    file_eta_err = torch.tensor(data[:,2]) * nu.s / nu.km
                    if calcErrors == 'High':
                        file_etas += file_eta_err
                    if calcErrors == 'Low':
                        file_etas -= file_eta_err


            #this was hardcoded to catch duplicates at end of verne file randomly
            if file_etas[-1] == file_etas[-2]:
                file_etas = file_etas[:-1]
                file_vmins = file_vmins[:-1]

        self.file_etas = file_etas
        self.file_vmins = file_vmins
        return


    def get_halo_data(self,vMins,halo_model,halo_id_params=None):
        """Retrieve velocity distribution for given velocities.
        
        Args:
            vMins: Minimum velocity values
            halo_model: Velocity distribution model (shm, tsa, etc. See DM_Halo.py)
            halo_id_params: Parameters for step function model (used for halo independent analysis)
            
        Returns:
            Integrated velocity distribution values
        """
        import torch
        import os
        import re
        #Etas are very sensitive to numerical deviations, so leaving these units in units of c
        


        if halo_id_params is not None: #doing halo idp analysis
            etas = self.DM_Halo.step_function_eta(vMins, halo_id_params) 


        elif halo_model == 'imb':
            # vMins = self.DM_Halo.vmin_tensor(self.Earr,qArr,mX) #in velocity units
            etas = self.DM_Halo.eta_MB_tensor(vMins) 

        
        else: #from file
            import torch
            file_vmins = self.file_vmins
            file_etas = self.file_etas
 
            etas = interp1d(file_vmins,file_etas,vMins) # inverse velocity
            #make sure to avoid interpolation issues where there isn't data
            etas = torch.where((vMins<file_vmins[0]) | (vMins > file_vmins[-1]) | (torch.isnan(etas)) ,0,etas)
            
        return etas  #inverse velocity units

    
    

        

    def vMin_tensor(self,qArr,Earr,mX,shell_key=None):
        """Calculate minimum velocities in tensor form.
        
        Args:
            qArr: Momentum transfer values
            Earr: Energy values
            mX: DM mass
            shell_key: Electron shell (for noble elements)
            
        Returns:
            Tensor of minimum velocities
        """
        import torch
        q_tiled = torch.tile(qArr,(len(Earr),1))

        EE_tiled = torch.tile(Earr,(len(qArr),1)).T
        if shell_key is not None:
            EE_tiled += binding_es[self.material][shell_key]

        vMin = ((EE_tiled/q_tiled)+(q_tiled/(2*mX)))
        

        return vMin
    
        
            # qpart = q / 2*mX
            # epart = torch.einsum('i,j->ij',E,1/q)
            # vMin = qpart + epart

            # return vMin/nu.c0

                    # result = torch.einsum("i,ij->ij",self.Earr,etas)

            #q / (2 * mX) + test_eE / qtest

            #  qtest/(2.0*test_mX * nu.MeV) + test_Ee/qtest
        
    def get_parametrized_eta(self,vMins,mX,halo_model,halo_id_params=None):
        """Calculate parametrized velocity distribution.
        
        Args:
            vMins: Minimum velocity values
            mX: DM mass
            halo_model: Velocity distribution model
            halo_id_params: Parameters for step function model (used for halo independent analysis)
            
        Returns:
            Parametrized integrated velocity distribution values (units of inverse time)
        """
        #stupid way for me to set units 
        import torch

        etas = self.get_halo_data(vMins,halo_model,halo_id_params=halo_id_params) #inverse velocity
        #ccms**2*sec2yr
        etas*=self.rhoX/mX * self.cross_section * nu.c0**2
        # etas = etas.to(torch.double)
        return etas # inverse time






    def vectorized_dRdE(self,mX,FDMn,halo_model,DoScreen=True,halo_id_params=None,integrate=True,debug=False,unitize=False):
        """Calculate differential rate (dR/dE) using vectorized operations for semiconductor materials.
        
        Args:
            mX: DM mass
            FDMn: Form factor model
            halo_model: Velocity distribution model
            DoScreen: Apply screening effects
            halo_id_params: Parameters for step function model (used for halo independent analysis)
            integrate: Perform numerical integration
            debug: Return debug information
            unitize: Convert to standard units
            
        Returns:
            Differential rate values
        """
        import torch
        
        torch.set_default_device(self.device)
        torch.set_default_dtype(self.default_dtype)

        mX = mX*nu.MeV  / nu.c0**2
        rm = self.reduced_mass(mX,nu.me)
        prefactor = nu.alphaFS * ((nu.me/rm)**2) * (1 / self.form_factor.mCell)

        ff_arr = torch.tensor(self.form_factor.ff,dtype=torch.get_default_dtype())

        if integrate:
            import torchquad
            torchquad.set_log_level('ERROR')
            from torchquad import set_up_backend
            # Enable GPU support if available and set the floating point precision
            set_up_backend("torch", data_type=self.dtype_str)

            from torchquad import Simpson
            simp = Simpson()
            numq = len(self.qArr)
            numE = len(self.Earr)
            qmin = self.qArr[0]
            qmax = self.qArr[-1]

            

            integration_domain = torch.tensor([[qmin,qmax]],dtype=torch.get_default_dtype())
            def vmin(q,E,mX):
                term1 = term1 = E.unsqueeze(0) / q.unsqueeze(1)
                term2 = q.unsqueeze(1) / (2 * mX)
                v = term1 + term2
                return  v
            def eta_func(vMin):
                return self.get_parametrized_eta(vMin,mX,halo_model,halo_id_params=halo_id_params)
            
            def momentum_integrand(q):
                q = q.flatten()
                #q only parts
                qdenom = 1/q**2
                qdenom *=(self.FDM(q,FDMn))**2
                #parts that depend on q and E
                eta = eta_func(vmin(q,self.Earr,mX))
                tf_f = (self.thomas_fermi_screening(q,self.Earr,doScreen=DoScreen))**2
                ff_f = ff_arr[:-1,:]
                # ff_f = ff_interp((q,Earr_unit))
                result = eta * tf_f * ff_f
                result = torch.einsum("i,ji->ji",self.Earr,result)
                result = torch.einsum("j,ji->ji",qdenom,result)

                return result
            integrated_result = simp.integrate(momentum_integrand, dim=1, N=numq, integration_domain=integration_domain) / self.Earr

          
                        
        else:
            fdm_factor = (self.FDM(self.qArr,FDMn))**2 #unitless
            vMins = self.vMin_tensor(self.qArr,self.Earr,mX)
            etas = self.get_parametrized_eta(vMins,mX,halo_model,halo_id_params=halo_id_params)
            if self.QEDark:
                ff_arr = ff_arr[:,self.Ei_array-1]
            ff_arr = ff_arr.T
            # ff_arr = torch.from_numpy(ff_arr)
            # ff_arr = ff_arr.to(self.device) #form factor (unitless)
            tf_factor = (self.TFscreening(DoScreen)**2) #unitless
            result = torch.einsum("i,ij->ij",self.Earr,torch.ones_like(etas))
            result *=etas
            result *= fdm_factor     
            result*=ff_arr
            result *=tf_factor
            qdenom = 1 / self.qArr
            result = torch.einsum("j,ij->ij",qdenom,result)
            integrated_result = (torch.sum(result,axis=1) / self.Earr)

           
        
        integrated_result *= prefactor
        integrated_result /=nu.c0
        
        band_gap_result = torch.where(self.Earr < self.form_factor.band_gap,0,integrated_result)
        if unitize:
            band_gap_result *= nu.year * nu.kg * nu.eV #return in correct units for comparison, otherwise keep it implicit for drdne
        if debug:  
            if integrate:
                vMins = self.vMin_tensor(self.qArr,self.Earr,mX)
                etas = self.get_parametrized_eta(vMins,mX,halo_model,halo_id_params=halo_id_params)
                tf_factor = (self.TFscreening(DoScreen)**2) #unitless
                fdm_factor = (self.FDM(self.qArr,FDMn))**2 #unitless
                ff_arr = ff_arr.T

            returndict = {
                'drde': band_gap_result,
                'vMins': vMins / (nu.km / nu.s),
                'etas': etas * nu.year,
                'prefactor': prefactor *nu.kg,
                'fdm_factor':fdm_factor,
                'ff_arr':ff_arr,
                'tf_factor': tf_factor,
                'qdenom': (1/self.qArr**2)

            }

            return returndict

        return band_gap_result  #result is in R / kg /year / eV

    def vectorized_dRdE_qcdark2(self, mX, FDMn, halo_model, halo_id_params=None):
        """Calculate dR/dE using QCDark2 dielectric function (RPA screening embedded in ε).

        Uses the dynamic structure factor S(q,E) = ELF × q²/(2πα) directly, matching the
        QCDark2 rate formula (arXiv:2603.12326). No additional Thomas-Fermi screening is
        applied — RPA screening is already encoded in ε.

        Args:
            mX: DM mass in MeV (scalar tensor)
            FDMn: Form factor model (0=heavy mediator, 2=light mediator)
            halo_model: Velocity distribution model string
            halo_id_params: Step-function parameters for halo-independent analysis

        Returns:
            dR/dE tensor of shape (N_E,), units implicit 1/kg/year/eV
        """
        import torch

        torch.set_default_device(self.device)
        torch.set_default_dtype(self.default_dtype)

        mX_mev = float(mX.item()) if isinstance(mX, torch.Tensor) else float(mX)
        mX_eV = mX_mev * 1e6
        astro_model = self._current_astro_model_numeric()
        eta_grid = self._qcdark2_eta_grid(mX_mev, halo_model, halo_id_params=halo_id_params)
        spectrum_numeric = self.qcdark2_backend.differential_rate(
            mX_eV,
            FDMn,
            astro_model,
            eta_grid,
            self.device,
            self.default_dtype,
        )

        # Convert from explicit physical units back into the repository's implicit
        # units so downstream n_e folding remains backward compatible.
        return spectrum_numeric / (nu.kg * nu.year * nu.eV)

    def calculate_semiconductor_rates(self,mX_array,halo_model,FDMn,ne,integrate=True,DoScreen=True,isoangle=None,halo_id_params=None,useVerne=False,calcErrors=None,debug=False):
        """Calculate rates for semiconductor crystals (Si, Ge).
        
        Args:
            mX_array: Array of DM masses
            halo_model: Velocity distribution model
            FDMn: Form factor model
            ne: Number of electron-hole pairs
            integrate: Perform numerical integration vs using preintegrated version
            DoScreen: Apply screening effects
            isoangle: isoangle (for modulated distributions) (number from 0 -35 with the true isoangle being 5x that value)
            halo_id_params: Parameters for step function model (used for halo independent analysis)
            useVerne: Use Verne distribution
            calcErrors: Calculate errors ('High'/'Low') (only for DaMaSCUS)
            debug: Return debug information
            
        Returns:
            Calculated rates
        """
        import torch
        if self.material == 'Ge' and self.form_factor_type != 'qcdark2':
            if self.ionization_func is not self.step_probabilities:
                self.change_to_step()

        nes = self._coerce_1d_tensor(ne).long()
        prob_fn_tiled = self.probabilities[nes-1,:]
        mX_array = self._coerce_1d_tensor(mX_array).to(torch.get_default_dtype())
        dRdnEs = torch.zeros((len(mX_array),len(nes)))

        for m,mX in enumerate(mX_array):
            if self.form_factor_type == 'qcdark2':
                if halo_id_params is None and halo_model not in ('shm', 'imb'):
                    self.setup_halo_data(
                        mX,
                        FDMn,
                        halo_model,
                        isoangle=isoangle,
                        useVerne=useVerne,
                        calcErrors=calcErrors,
                    )
                dRdE = self.vectorized_dRdE_qcdark2(mX,FDMn,halo_model,halo_id_params=halo_id_params)
                dRdne = simpson_uniform(
                    dRdE * prob_fn_tiled,
                    self.qcdark2_backend.E_step_eV,
                    dim=1,
                ) * nu.eV
            else:
                self.setup_halo_data(
                    mX,
                    FDMn,
                    halo_model,
                    isoangle=isoangle,
                    useVerne=useVerne,
                    calcErrors=calcErrors,
                )
                dRdE = self.vectorized_dRdE(mX,FDMn,halo_model,DoScreen=DoScreen,halo_id_params=halo_id_params,integrate=integrate)
                if integrate:
                    dRdne = torch.trapezoid(dRdE*prob_fn_tiled,x=self.Earr, axis = 1)
                else:
                # if self.QEDark:
                # dRdne = torch.sum(dRdE*prob_fn_tiled, axis = 1) * (1*nu.eV)
                # else:
                    dRdne = torch.sum(dRdE*prob_fn_tiled*self.form_factor.dE*10, axis = 1) #still not sure why I need a factor of 10 here to match

            dRdnEs[m,:] = dRdne
        if debug:
            return dRdnEs.T,dRdE,prob_fn_tiled
        return dRdnEs.T #should be in kg/year
    
    def rate_dme_shell(self,mX,FDMn,halo_model,shell_key,halo_id_params=None,debug=False,unitize=False):
        """Calculate rate for specific electron shell in noble elements.
        
        Args:
            mX: DM mass
            FDMn: Form factor model
            halo_model: Velocity distribution model
            shell_key: Electron shell identifier
            halo_id_params: Parameters for step function model (used for halo independent analysis)
            debug: Return debug information
            unitize: Convert to standard units
            
        Returns:
            Rate for specified shell
        """
        import torch
        qArr = self.qArrdict[shell_key]
        # qiArr = self.qArrdict[shell_key] / (me_eV * nu.alphaFS)
        qmin = qArr[0]
        qmax = qArr[-1]
        numq = len(qArr)

        



        rm = self.reduced_mass(mX,nu.me)


        prefactor = 1 / (8*self.form_factor.mCell * (rm)**2 )  
        prefactor /= nu.c0**2


        #eb = get_binding_es(shell_key)
        fdm_factor = (self.FDM(qArr,FDMn))**2 #unitless
        vMins = self.vMin_tensor(qArr,self.Earr,mX,shell_key)
        etas = self.get_parametrized_eta(vMins,mX,halo_model,halo_id_params=halo_id_params)
            
        try:
            ff_arr = self.form_factor.ff[shell_key]
        except:
            print("ValueError('you have given an invalid or unimplemented shell for this material')")

        # qmax = (torch.exp(shell_data[shell_key]['lnqs'].max())) * me_eV * nu.alphaFS #this might also be a defined quantity

        result = torch.einsum("j,ij->ij",fdm_factor,etas)
        result*=ff_arr  




        import torchquad
        torchquad.set_log_level('ERROR')
        from torchquad import set_up_backend,set_precision
        # Enable GPU support if available and set the floating point precision
        set_up_backend("torch", data_type=self.dtype_str)

        from torchquad import Simpson
        simp = Simpson()
        
        integration_domain = torch.Tensor([[0,qmax]])
        if debug:
            print(integration_domain)
        def momentum_integrand(q):
            qint = q.flatten()
            # qint = qint * self.FDM(qint,FDMn) 
            return torch.einsum("j,ij->ji",qint,result)
        
        integrated_result = simp.integrate(momentum_integrand, dim=1, N=numq, integration_domain=integration_domain) / self.Earr

        integrated_result *= prefactor

        if unitize:
            integrated_result *= nu.kg *nu.day * nu.keV


        if debug:
            print('returning debug output:')
            print('integrated_result,prefactor,fdm_factor,ff_arr,etas,qArr,qmin,qmax')
            return integrated_result,prefactor,fdm_factor,ff_arr,etas,qArr,qmin,qmax
        return integrated_result
    
    def noble_dRdE(self,mX,FDMn,halo_model,halo_id_params=None,debug=False,unitize=False):
        """Calculate differential rates for all shells in noble elements.
        
        Args:
            mX: DM mass
            FDMn: Form factor model
            halo_model: Velocity distribution model
            halo_id_params: Parameters for step function model (used for halo independent analysis)
            debug: Return debug information
            unitize: Convert to standard units
            
        Returns:
            Dictionary of differential rates by shell
        """

        mX = mX*nu.MeV / nu.c0**2
        drs = dict()
        for key in self.form_factor.keys:
            if key in skip_keys[self.material]:
                continue
            dr = self.rate_dme_shell(mX,FDMn,halo_model,key,halo_id_params=halo_id_params,debug=False,unitize=unitize)
            drs[key] = dr
        return drs
    

    def energy_to_ne_pmf(self,rates,shell,nes_tensor,p_primary,p_secondary):
        """Convert energy differential rates to electron count probabilities.
        
        Args:
            rates: Differential rates
            shell: Electron shell
            nes_tensor: Electron count values
            p_primary: Primary probability
            p_secondary: Secondary probability
            
        Returns:
            Probability mass function values
        """
        import torch
        from torch.distributions.binomial import Binomial
        # nes_tensor = nes_tensor.to(torch.double)
        p_primary = torch.tensor(p_primary)
        fact = additional_quanta[self.material][shell]
        W = work_function[self.material]

        n_secondary = torch.floor(self.Earr / W).int() + fact  # (N,)


        binsizes = torch.tensor(torch.diff(self.Earr).tolist() + [self.Earr[-1] - self.Earr[-2]])
        # rates = rates.to(torch.float32)
        weights = rates * binsizes

        N = n_secondary.shape[0]
        M = nes_tensor.shape[0]

        r_n_tensor = torch.zeros(M)

        # Initialize PMF matrix
        pmf = torch.zeros((N, M))

        # Get unique values for efficiency
        unique_n, inverse_idx = torch.unique(n_secondary, return_inverse=True)
        for idx, n in enumerate(unique_n):
            n = int(n.item())  # Convert to Python int


            row_mask = (n_secondary == n)
            dist = Binomial(total_count=n, probs=p_secondary)

            # Determine which nes_tensor values are valid for this n
            valid_idx = nes_tensor <= n
            valid_nes = nes_tensor[valid_idx]  # Only pass valid values into log_prob

            # Compute PMF(nes)
            pmf_n = torch.zeros_like(nes_tensor)
            pmf_n[valid_idx] = torch.exp(dist.log_prob(valid_nes))

            # Compute PMF(nes - 1)
            shifted = nes_tensor - 1
            valid_shift_idx = (shifted >= 0) & (shifted <= n)
            shifted_nes = shifted[valid_shift_idx]

            pmf_n_minus_1 = torch.zeros_like(nes_tensor)
            pmf_n_minus_1[valid_shift_idx] = torch.exp(dist.log_prob(shifted_nes))

            # Combine PMFs
            weighted_pmf = p_primary * pmf_n_minus_1 + (1 - p_primary) * pmf_n

            # Assign weighted PMF to all rows with this n
            pmf[row_mask] = weighted_pmf
        # weights = weights.to(torch.double)
    
        # print(weights.type(),pmf.type())
        r_n_tensor = torch.matmul(weights, pmf)
        return r_n_tensor

    def vectorized_energy_to_ne_pmf(self, rates, shell, nes_tensor, p_primary, p_secondary):
        """Vectorized version of energy_to_ne_pmf.
        
        Args:
            rates: Differential rates
            shell: Electron shell
            nes_tensor: Electron count values
            p_primary: Primary probability
            p_secondary: Secondary probability
            
        Returns:
            Probability mass function values
        """
        import torch
        
        # Convert inputs to appropriate types
        # nes_tensor = nes_tensor.to(torch.double)
        p_primary = torch.tensor(p_primary)
        p_secondary = torch.tensor(p_secondary)
        fact = additional_quanta[self.material][shell]
        W = work_function[self.material]

        # Calculate n_secondary (shape: [N])
        n_secondary = (torch.floor(self.Earr / W) + fact).int()
        
        # Calculate binsizes and weights (shape: [N])
        binsizes = torch.tensor(torch.diff(self.Earr).tolist() + [self.Earr[-1] - self.Earr[-2]])
        weights = rates * binsizes
        # weights = weights.to(torch.double)

        # Prepare for vectorized calculations
        N = n_secondary.shape[0]
        M = nes_tensor.shape[0]
        
        # Expand dimensions for broadcasting
        # nes_tensor: [M] -> [1, M]
        # n_secondary: [N] -> [N, 1]
        k = nes_tensor.unsqueeze(0)          # [1, M]
        n = n_secondary.unsqueeze(1)         # [N, 1]
        p = p_secondary
        
        # Calculate log binomial coefficients using log factorial
        # log_comb(n,k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
        log_comb = (torch.lgamma(n+1) - torch.lgamma(k+1) - torch.lgamma(n-k+1))
        
        # Calculate log probabilities
        log_pmf = log_comb + k*torch.log(p) + (n-k)*torch.log(1-p)
        
        # Calculate pmf_n (set to 0 where k > n)
        pmf_n = torch.exp(log_pmf) * (k <= n).double()
        
        # Calculate pmf_n_minus_1 (for k-1)
        k_minus_1 = k - 1
        valid_shift = (k_minus_1 >= 0) & (k_minus_1 <= n)
        log_comb_shift = (torch.lgamma(n+1) - torch.lgamma(k_minus_1+1) - 
                        torch.lgamma(n-k_minus_1+1))
        log_pmf_shift = log_comb_shift + k_minus_1*torch.log(p) + (n-k_minus_1)*torch.log(1-p)
        pmf_n_minus_1 = torch.exp(log_pmf_shift) * valid_shift.double()
        
        # Combine PMFs
        weighted_pmf = p_primary * pmf_n_minus_1 + (1 - p_primary) * pmf_n
        
        # Compute final result
        r_n_tensor = torch.matmul(weights, weighted_pmf)
    
        return r_n_tensor

    


    def rates_to_ne(self,drs,nes,p_primary = 1,p_secondary = 0.83, swap_4s4p = False):
        """Convert differential rates to electron count rates.
        
        Args:
            drs: Dictionary of differential rates by shell
            nes: Electron values to calculate for (list/int)
            p_primary: Primary probability
            p_secondary: Secondary probability
            swap_4s4p: Special handling for Xenon shells because the form factor data was swapped
            
        Returns:
            Dictionary of rates by shell
        """
        import torch
        # We need an "energy bin size" to multiply with (or do some fancy integration)
        # I'll use the differences between the points at which the differential 
        # rates were computed.
        # To ensure this doesn't give a bias, nearby bins can't differ too much 
        # (e.g. use a linspace or a high-n logspace/geomspace)
        # import torch.distributions.binomial as binom
        
        result = dict()
        
        for shell, rates in drs.items():
            if swap_4s4p and self.material == 'Xe':
                # Somehow we can reproduce 1703.00910
                # if we swap 4s <-> 4p here??
                if shell == '4s':
                    rates = drs['4p']
                elif shell == '4p':
                    rates = drs['4s']
            result[shell] = self.vectorized_energy_to_ne_pmf(rates,shell,nes,p_primary,p_secondary)


        return result
            
                



            







    def calculate_nobleGas_rates(self,mX_array,halo_model,FDMn,ne,isoangle=None,halo_id_params=None,useVerne=False,calcErrors=None,debug=False,returnShells=False):
        """Calculate rates for noble gas targets (Xe, Ar).
        
            Args:
                mX_array: Array of DM masses
                halo_model: Velocity distribution model
                FDMn: Form factor model
                ne: Number of electrons (list,int)
                isoangle: isoangle (for modulated distributions) (number from 0 -35 with the true isoangle being 5x that value)
                halo_id_params: Parameters for step function model (used for halo independent analysis)
                useVerne: Use Verne distribution
                calcErrors: Calculate errors ('High'/'Low')
                debug: Return debug information
                returnShells: Return shell-by-shell breakdown
                
            Returns:
                Calculated rates (and shell breakdown if requested)
        """
        import torch
        import numpy

        if type(mX_array) != torch.tensor:
            if type(mX_array) == int or type(mX_array) == float:
                    mX_array = torch.tensor([mX_array])
            elif type(mX_array) == list:
                mX_array = torch.tensor(mX_array)
            elif type(mX_array) == numpy.ndarray:
                mX_array = torch.from_numpy(mX_array)
                mX_array = mX_array.to(self.device)
            else:
                try:
                    mX_array = torch.tensor([mX_array])
                except:
                    print('unknown data type')

        if type(ne) != torch.Tensor:
            if type(ne) == int:
                nes = torch.tensor([ne])
            elif type(ne) == list:
                nes = torch.tensor(ne)
            elif type(ne) == numpy.ndarray:
                nes = torch.from_numpy(ne)
                nes = nes.to(self.device)
            else:
                try:
                    nes = torch.tensor(ne)
                except:
                    print('unknown data type')
        else:
            nes = ne

        if returnShells:        

            keys = self.form_factor.keys
            skpkeys = skip_keys[self.material]
            totalkeys = [k for k in keys if k not in skpkeys]
            numkeys = len(totalkeys)
            shells = ['Summed'] + totalkeys
            dRdnEs = torch.zeros((len(mX_array),len(nes),numkeys + 1))
        else:        
            dRdnEs = torch.zeros((len(mX_array),len(nes)))


        
        for m,mX in enumerate(mX_array):
            self.setup_halo_data(mX,FDMn,halo_model,isoangle=isoangle,useVerne=useVerne,calcErrors=calcErrors)
            drs = self.noble_dRdE(mX,FDMn,halo_model,halo_id_params=halo_id_params,debug=False,unitize=False)
            dRdnEs_by_shell =  self.rates_to_ne(drs,nes,p_primary = 1,p_secondary = 0.83, swap_4s4p = True)
            dRdnEs_by_shell = torch.stack(list(dRdnEs_by_shell.values()))
            dRdnE_sum = torch.sum(dRdnEs_by_shell,axis=0)
            if returnShells:
                dRdnEs[m,:,0] = dRdnE_sum
                dRdnEs[m,:,1:] = dRdnEs_by_shell.T


            else:
                dRdnEs[m,:] = dRdnE_sum

        
        if returnShells:
            return dRdnEs,shells
        return dRdnEs.T

    def calculate_spectrum(self,mX_array,halo_model,FDMn,integrate=True,DoScreen=True,isoangle=None,halo_id_params=None,useVerne=False,calcErrors=None):
        """Return semiconductor recoil spectra in physical units."""
        import warnings
        import numpy as np

        if self.material not in _SEMICONDUCTOR_MATERIALS:
            raise ValueError("calculate_spectrum is currently implemented for semiconductor targets only.")

        if self.form_factor_type == 'qcdark2' and DoScreen:
            warnings.warn(
                "DoScreen=True has no effect for form_factor_type='qcdark2': RPA screening "
                "is already embedded in the dielectric function. No Thomas-Fermi screening "
                "will be applied.", UserWarning, stacklevel=2)

        mX_array = self._coerce_1d_tensor(mX_array).to(self.default_dtype)
        spectra = []

        for mX in mX_array:
            if self.form_factor_type == 'qcdark2':
                if halo_id_params is None and halo_model not in ('shm', 'imb'):
                    self.setup_halo_data(
                        mX,
                        FDMn,
                        halo_model,
                        isoangle=isoangle,
                        useVerne=useVerne,
                        calcErrors=calcErrors,
                    )
                dRdE = self.vectorized_dRdE_qcdark2(mX, FDMn, halo_model, halo_id_params=halo_id_params)
            else:
                self.setup_halo_data(
                    mX,
                    FDMn,
                    halo_model,
                    isoangle=isoangle,
                    useVerne=useVerne,
                    calcErrors=calcErrors,
                )
                dRdE = self.vectorized_dRdE(
                    mX,
                    FDMn,
                    halo_model,
                    DoScreen=DoScreen,
                    halo_id_params=halo_id_params,
                    integrate=integrate,
                )
            spectra.append((dRdE * nu.kg * nu.year * nu.eV).detach().cpu())

        energy_eV = (self.Earr / nu.eV).detach().cpu().numpy()
        spectrum_array = np.stack([spec.numpy() for spec in spectra], axis=0)
        return energy_eV, spectrum_array

    def calculate_total_rate(self,mX_array,halo_model,FDMn,integrate=True,DoScreen=True,isoangle=None,halo_id_params=None,useVerne=False,calcErrors=None):
        """Return semiconductor total rates in events/kg/year."""
        from scipy.integrate import simpson

        energy_eV, spectra = self.calculate_spectrum(
            mX_array,
            halo_model,
            FDMn,
            integrate=integrate,
            DoScreen=DoScreen,
            isoangle=isoangle,
            halo_id_params=halo_id_params,
            useVerne=useVerne,
            calcErrors=calcErrors,
        )
        return simpson(spectra, x=energy_eV, axis=1)











    def calculate_ne_rates(self,mX_array,halo_model,FDMn,ne,integrate=True,DoScreen=True,isoangle=None,halo_id_params=None,useVerne=False,calcErrors=None,debug=False):
        """Calculate electron-count/electron-hole-pair binned rates.
        
        Args:
            mX_array: Array of DM masses
            halo_model: Velocity distribution model
            FDMn: Form factor model
            ne: Number of electrons/electron-hole pairs (list/int)
            integrate: Perform numerical integration
            DoScreen: Apply screening effects
            isoangle: isoangle (for modulated distributions) (number from 0 -35 with the true isoangle being 5x that value)
            halo_id_params: Parameters for step function model (used for halo independent analysis)
            useVerne: Use Verne distribution
            calcErrors: Calculate errors ('High'/'Low')
            debug: Return debug information
            
        Returns:
            Calculated rates binned by detected electron count
        """

        import warnings
        if self.form_factor_type == 'qcdark2' and DoScreen:
            warnings.warn(
                "DoScreen=True has no effect for form_factor_type='qcdark2': RPA screening "
                "is already embedded in the dielectric function. No Thomas-Fermi screening "
                "will be applied.", UserWarning, stacklevel=2)
        if self.material in _SEMICONDUCTOR_MATERIALS:
            return self.calculate_semiconductor_rates(mX_array,halo_model,FDMn,ne,integrate,DoScreen,isoangle=isoangle,halo_id_params=halo_id_params,useVerne=useVerne,calcErrors=calcErrors,debug=debug)
        if self.material in _NOBLE_MATERIALS:
            return self.calculate_nobleGas_rates(mX_array,halo_model,FDMn,ne,isoangle=isoangle,halo_id_params=halo_id_params,useVerne=useVerne,calcErrors=calcErrors,debug=debug,returnShells=False)
    
    
    def calculate_rates(self,mX_array,halo_model,FDMn,ne,integrate=True,DoScreen=True,isoangle=None,halo_id_params=None,useVerne=False,calcErrors=None,debug=False):
        """Backward-compatible alias for ``calculate_ne_rates``."""
        return self.calculate_ne_rates(
            mX_array,
            halo_model,
            FDMn,
            ne,
            integrate=integrate,
            DoScreen=DoScreen,
            isoangle=isoangle,
            halo_id_params=halo_id_params,
            useVerne=useVerne,
            calcErrors=calcErrors,
            debug=debug,
        )

    

    def generate_dat(self,dm_masses,ne_bins,fdm,dm_halo_model,DoScreen=False,write=True,tag=""):
        """Generate rate data files for storage and analysis.
        
        Args:
            dm_masses: Array of DM masses
            ne_bins: Electron count bins
            fdm: Form factor model 
            dm_halo_model: Velocity distribution model
            DoScreen: Apply screening effects
            write: Save to file
            tag: Additional filename tag
            
        Returns:
            Array of calculated rates
        """

        import torch
        from tqdm.autonotebook import tqdm
        import numpy as np
        function_name = 'p100k' if self.ionization_func == self.RKProbabilities else 'step'
        fdm_dict = {0:'1',1:'q',2:'q2'}
        rho_X =self.rhoX / (nu.GeV / nu.c0**2 / nu.cm**3)
        vE = self.vEarth  / (nu.km / nu.s)
        vesc = self.vEscape    / (nu.km / nu.s)
        v0 = self.v0 / (nu.km / nu.s)
        rho_X_print = str(np.round(rho_X,1))
        ebin_print = str(self.bin_size)
        vesc_print = str(np.round(vesc,1))
        v0_print = str(np.round(v0,1))
        vE_print = str(np.round(vE,1))


        ebin_print = ebin_print.replace('.','pt')
        rho_X_print = rho_X_print.replace('.','pt')
        vesc_print = vesc_print.replace('.','pt')
        v0_print = v0_print.replace('.','pt')
        vE_print = vE_print.replace('.','pt')


        import os
        FDM_Dir = os.path.join(self.module_dir,'Rates/')
        screen = '_screened' if DoScreen else ''
        integrate = True
        ff_tag_map = {
            'qcdark':  '_qcdark',
            'qedark':  '_qedark',
            'qcdark2': f'_qcdark2_{self.qcdark2_variant}',
        }
        qestr = ff_tag_map.get(self.form_factor_type, f'_{self.form_factor_type}')
        if self.form_factor_type == 'qedark':
            integrate = False

        
      
        filename = FDM_Dir + f'FDM{fdm_dict[fdm]}_vesc{vesc_print}-v0{v0_print}-vE{vE_print}-rhoX{rho_X_print}_nevents_func{function_name}_maxne{np.max(ne_bins)}_unscaled{screen}{qestr}_dmerates_{tag}.dat'
        
        lines = []
        data = np.zeros((len(dm_masses),len(ne_bins)))
        for m in tqdm(range(len(dm_masses))):
            mX = dm_masses[m]
            line = f'{mX}\t'
            gday = self.calculate_rates(mX,dm_halo_model,fdm,ne_bins,integrate=integrate,DoScreen=DoScreen).T * nu.g * nu.day
            gday = gday[0]
            gday = torch.where(torch.isnan(gday),0,gday)
            g_day = gday.numpy()
            for ne in ne_bins:
                data[m,ne - 1] = g_day[ne-1]
                line+=str(g_day[ne-1])+'\t'
                
            line += f'\n'
            lines.append(line)
        if write: 
            f = open(filename,'w')
            first_line = f'FDM{fdm_dict[fdm]}:\tmX [MeV]\tsigmae={self.cross_section / nu.cm**2} [cm^2]\t'
            for ne in ne_bins:
                first_line += f'ne{ne}\t'
            first_line+='\n'
            f.write(first_line)
            for line in lines:
                f.write(line)
        return data

        

    
