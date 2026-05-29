import logging
import math
from time import perf_counter
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
from math import degrees

import matplotlib.pyplot as plt

from plot_funcs import plotFuncs
plt.rcParams.update({'font.size': 10})
#plt.rcParams.update({'figure.autolayout':True})

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utility.coordtrans import XYZ_to_RTP, RTP_XYZ_JAC#, RTP_to_XYZ
#from plot_funcs import plotFuncs

## SOME PHYSICAL CONSTANTS
kg_per_amu = 1.660_539_068E-27
kboltz = 1.602_176_634E-19 # Joules/eV
eps0 = 8.854_187_8128E-12
sqrt_pi = math.sqrt(math.pi)
Li_mass = 6.941 #amu
He_mass = 4.002602 #amu

class Boris():
    """Class to handle Boris analysis of magnetic field lines."""
    def __init__(self, io_handler, anlys_name='Boris', tag=None):
        """Initializes the Boris class with the specified solver parameters and writes to log.

        Args:
            io_handler: An object responsible for handling output operations, such as logging and directory creation.
            anlys_name (str, optional): Name of the analysis or subdirectory. Defaults to 'Poincare'.
        """

        # attach plotting function to class instance
        for name in dir(plotFuncs):
                    func = getattr(plotFuncs, name)
                    if callable(func) and not name.startswith("__"):
                        if name.startswith("global_"):
                            new_name = name.replace("global_", "")  # Remove prefix
                        elif name.startswith("boris_"):
                            new_name = name.replace("boris_", "")  # Remove prefix
                            setattr(self, new_name, func)  # Attach to the instance with the new name

        self.IO = io_handler
        self.anlys_name = anlys_name
        self.solver = 'boris_buneman'
        self.tag = tag

        # self.IO.createSubDir(anlys_name)
        # self.IO.log.info("+----------------+-------------------------+")
        # self.IO.log.info("| Parameter      | Value                   |")
        # self.IO.log.info("+----------------+-------------------------+")
        # self.IO.log.info(f"| SOLVER         | {self.solver:<23} |")
        # self.IO.log.info(f"| ANLYS_NAME     | {str(self.anlys_name):<23} |")
        # self.IO.log.info(f"| TAG            | {str(self.tag):<23} |")
        # self.IO.log.info("+----------------+-------------------------+")


    def setConditions(self, ion_list, cond_string, dt=1e-8, tmax=1e-3, T_gas_eV=0.025):
        """Sets the initial conditions and events for Poincare analysis.

        Args:
            ion_list (list): List of Ion objects containing initial conditions and properties.
            dt (float, optional): Time step for the simulation. Defaults to 1e-8.
            tmax (float, optional): Maximum time for the simulation. Defaults to 1e-3.

        Returns:
            None
        """
        self.dt = dt
        self.tmax = tmax
        self.nsteps = int(tmax // dt) + 1
        self.ion_list = ion_list
        self.T_gas_eV = T_gas_eV # eV, room temperature
        self.m_gas_amu = 4.002602 #amu, Helium
        self.Ti_eV = 2.0 # eV, ion temperature for ion-ion collision model
        self.m_ion_amu = 4.002602 #amu, ion mass for ion-ion collision model

        self.cond_string = cond_string
        ## SET OUTPUT
        # for ion in ion_list:
        #     ion.initOutput(dt, tmax)

        # self.IO.log.info("+----------------+-------------------------+")
        # self.IO.log.info(f"| DT             | {self.dt:<23} |")
        # self.IO.log.info(f"| TMAX           | {self.tmax:<23} |")
        # self.IO.log.info(f"| NSTEPS         | {self.nsteps:<23} |")
        # self.IO.log.info("+----------------+-------------------------+")

    def viscous_drag_hstep(self, x, v, n_gas=1e17, sigma_mt=1e-19):
        """Function to apply a half-step of viscous drag to the ion velocities, simulating ion-neutral collisions.
        Constant cross-section and cold (motionless) neutrals are assumed for simplicity.

        Parameters:
            -x (torch.Tensor): Current positions of the ions, shape (Nparticles, 3).
            -v (torch.Tensor): Current velocities of the ions, shape (Nparticles, 3).
            -n_gas (float, optional): Neutral gas density in m^-3. Defaults to 1e18.
            -sigma_mt (float, optional): Momentum transfer cross-section in m^2. Defaults to 1e-19 m^2.
         Returns:
            -v_new (torch.Tensor): Updated velocities of the ions after applying viscous drag, shape (Nparticles, 3).
        """

        v_mag = torch.linalg.norm(v, axis=-1)
        nu = n_gas * sigma_mt * v_mag
        alpha = torch.exp(-nu * self.dt / 2)


        v_new = v * alpha[:, None]  # Apply the viscous drag factor to the velocities

        return v_new

    def langevin_in_hstep(self, x, v, n_gas=1e17, sigma_mt=1e-17, kbTgasqMi=(kboltz*0.025) / (4.002602 * kg_per_amu) ):

        """Function to apply a half-step of viscous drag to the ion velocities, simulating ion-neutral collisions.
        Constant cross-section and cold (motionless) neutrals are assumed for simplicity.
        
        Parameters:
            -x (torch.Tensor): Current positions of the ions, shape (Nparticles, 3).
            -v (torch.Tensor): Current velocities of the ions, shape (Nparticles, 3).
            -n_gas (float, optional): Neutral gas density in m^-3. Defaults to 1e18.
            -sigma_mt (float, optional): Momentum transfer cross-section in m^2. Defaults to 1e-19 m^2.
            -kbTgasqMi (float, optional): Gas temperature in eV divided by ion mass in amu. Defaults to 0.025 eV/amu.
         Returns:
            -v_new (torch.Tensor): Updated velocities of the ions after applying viscous drag, shape (Nparticles, 3).
        """

        v_mag = torch.linalg.norm(v, axis=-1)
        nu = n_gas * sigma_mt * v_mag
        alpha = torch.exp(-nu * self.dt / 2)

        sigma = torch.sqrt(kbTgasqMi * (1.0 - alpha**2))

        eta = torch.randn_like(v)
        v_new = v * alpha[:, None] + sigma[:, None] * eta # Apply the viscous drag factor to the velocities

        return v_new

    def linearFP_ii_hstep(self, x, v, n_e, Ti_ev=2.0):

        """Function to apply a half-step of viscous drag to the ion velocities, simulating ion-ion collisions.
        Constant cross-section and cold (motionless) neutrals are assumed for simplicity.

        Parameters:
            -x (torch.Tensor): Current positions of the ions, shape (Nparticles, 3).
            -v (torch.Tensor): Current velocities of the ions, shape (Nparticles, 3).
            -n_e (float, optional): Electron density in m^-3. Defaults to 1e18.
            -Ti_ev (float, optional): Ion temperature in eV. Defaults to 2.0 eV.
        Returns:
            -v_new (torch.Tensor): Updated velocities of the ions after applying viscous drag, shape (Nparticles, 3).
        """

        n_e = torch.as_tensor(n_e, dtype=v.dtype, device=v.device)

        nu = 3.61e-10 * n_e * Ti_ev**(-1.5)
        alpha = torch.exp(-nu * self.dt / 2)

        sigma = torch.sqrt(kboltz * Ti_ev / (self.m_ion_amu * kg_per_amu) * (1.0 - alpha**2))
        eta = torch.randn_like(v)

        #v_new = v_therm + (v - v_therm) * alpha[..., None] + sigma[..., None] * eta # Apply the viscous drag factor to the velocities
        v_new = v * alpha[:, None] + sigma[:, None] * eta # Apply the viscous drag factor to the velocities

        return v_new

    def chandrasekhar_psi(self, x):
        """Stable Chandrasekhar function psi(x) for Coulomb FP rates."""

        x = torch.clamp(x, min=0.0)
        sqrt_x = torch.sqrt(x)

        psi_direct = torch.erf(sqrt_x) - (2.0 / sqrt_pi) * sqrt_x * torch.exp(-x)
        psi_series = (
            4.0 / (3.0 * sqrt_pi)
            * x * sqrt_x
            * (1.0 - 3.0 * x / 5.0 + 3.0 * x * x / 14.0 - x * x * x / 18.0)
        )

        return torch.where(x < 1.0e-3, psi_series, psi_direct)

    def chandrasekhar_psi_prime(self, x):
        """Derivative psi'(x) for Coulomb FP rates."""

        x = torch.clamp(x, min=0.0)
        return (2.0 / sqrt_pi) * torch.sqrt(x) * torch.exp(-x)

    def coulomb_fp_rates_li_he(
        self,
        w,
        n_b,
        T_b_eV,
        lnLambda=10.0,
        Z_a=1.0,
        Z_b=1.0,
        m_a_amu=Li_mass,
        m_b_amu=He_mass,
        w_floor=1.0e-30,
        x_floor=1.0e-30,
    ):
        """Coulomb FP rates for Li test ions on a Maxwellian helium background.

        Parameters:
            -w (torch.Tensor): Relative velocity v_Li - u_He, shape (..., 3), in m/s.
            -n_b (torch.Tensor or float): Background helium ion density in m^-3.
            -T_b_eV (torch.Tensor or float): Background helium ion temperature in eV.
            -lnLambda (torch.Tensor or float): Coulomb logarithm.
            -Z_a (float): Test ion charge state.
            -Z_b (float): Background helium charge state.
            -m_a_amu (float): Test ion mass in amu.
            -m_b_amu (float): Background ion mass in amu.
        Returns:
            -nu_s, nu_perp, nu_parallel (torch.Tensor): FP rates in s^-1.
        """

        dtype = w.dtype
        device = w.device

        n_b = torch.as_tensor(n_b, dtype=dtype, device=device)
        T_b_eV = torch.as_tensor(T_b_eV, dtype=dtype, device=device)
        lnLambda = torch.as_tensor(lnLambda, dtype=dtype, device=device)

        m_a = m_a_amu * kg_per_amu
        m_b = m_b_amu * kg_per_amu

        w2 = torch.sum(w * w, dim=-1)
        wmag = torch.sqrt(torch.clamp(w2, min=w_floor * w_floor))

        kT_b = kboltz * T_b_eV
        x = m_b * w2 / (2.0 * kT_b)
        x_safe = torch.clamp(x, min=x_floor)

        psi = self.chandrasekhar_psi(x_safe)
        psi_prime = self.chandrasekhar_psi_prime(x_safe)

        prefactor = (
            n_b
            * (Z_a * Z_b * kboltz**2)**2
            * lnLambda
            / (4.0 * math.pi * eps0**2 * m_a**2)
        )
        nu0 = prefactor / (wmag**3)

        nu_s = (1.0 + m_a / m_b) * psi * nu0
        nu_perp = 2.0 * ((1.0 - 1.0 / (2.0 * x_safe)) * psi + psi_prime) * nu0
        nu_parallel = (psi / x_safe) * nu0

        nu_s = torch.clamp(nu_s, min=0.0)
        nu_perp = torch.clamp(nu_perp, min=0.0)
        nu_parallel = torch.clamp(nu_parallel, min=0.0)

        return nu_s, nu_perp, nu_parallel

    def _resolve_collision_model(self, collisions):
        if collisions is None or collisions is False:
            return None
        if collisions is True:
            return 'linearFP_ii_hstep'

        collision_key = str(collisions).strip().lower()
        collision_models = {
            'none': None,
            'false': None,
            'off': None,
            'true': 'linearFP_ii_hstep',
            'viscous_drag_hstep': 'viscous_drag_hstep',
            'langevin_in_hstep': 'langevin_in_hstep',
            'linearfp_ii_hstep': 'linearFP_ii_hstep',
            'fokkerplanck_ii_hstep': 'fokker_planck_ii_hstep',
            'fokker_planck_ii_hstep': 'fokker_planck_ii_hstep',
            'fokker_planck_ii_hstepp': 'fokker_planck_ii_hstep',
        }

        if collision_key not in collision_models:
            valid = ', '.join([
                'viscous_drag_hstep',
                'langevin_in_hstep',
                'linearFP_ii_hstep',
                'fokker_planck_ii_hstep',
                'None',
            ])
            raise ValueError(f'Unknown collisions option {collisions!r}. Use one of: {valid}.')

        return collision_models[collision_key]

    def _collision_uses_density(self, collision_model):
        return collision_model in ('linearFP_ii_hstep', 'fokker_planck_ii_hstep')

    def _apply_collision_hstep(self, collision_model, x, v, n_e=None, Ti_ev=2.0, kbTgasqMi=None):
        if collision_model == 'viscous_drag_hstep':
            return self.viscous_drag_hstep(x, v)
        if collision_model == 'langevin_in_hstep':
            if kbTgasqMi is None:
                return self.langevin_in_hstep(x, v)
            return self.langevin_in_hstep(x, v, kbTgasqMi=kbTgasqMi)

        if n_e is None:
            n_e = torch.full((v.shape[0],), 1e18, dtype=v.dtype, device=v.device)

        if collision_model == 'linearFP_ii_hstep':
            return self.linearFP_ii_hstep(x, v, n_e, Ti_ev=Ti_ev)
        if collision_model == 'fokker_planck_ii_hstep':
            return self.fokker_planck_ii_hstep(x, v, n_e, Ti_ev=Ti_ev)

        return v

    def fokker_planck_ii_hstep(
        self,
        x,
        v,
        n_e,
        Ti_ev=2.0,
        lnLambda=10.0,
        u_b=None,
        Z_a=1.0,
        Z_b=1.0,
        m_a_amu=Li_mass,
        m_b_amu=He_mass,
        w_small=1.0e-6,
    ):

        """Apply a half-step Li-He Coulomb Fokker-Planck collision operator.

        Parameters:
            -x (torch.Tensor): Current positions of the ions, shape (Nparticles, 3).
            -v (torch.Tensor): Current velocities of the ions, shape (Nparticles, 3).
            -n_e (float or torch.Tensor): Background helium ion density in m^-3.
            -Ti_ev (float or torch.Tensor): Background helium ion temperature in eV.
            -lnLambda (float or torch.Tensor): Coulomb logarithm.
            -u_b (torch.Tensor, optional): Background helium flow velocity in m/s.
        Returns:
            -v_new (torch.Tensor): Updated velocities of the ions after applying viscous drag, shape (Nparticles, 3).
        """

        if u_b is None:
            u_b = torch.zeros_like(v)
        else:
            u_b = torch.as_tensor(u_b, dtype=v.dtype, device=v.device)

        w = v - u_b
        w2 = torch.sum(w * w, dim=-1, keepdim=True)
        wmag = torch.sqrt(torch.clamp(w2, min=w_small * w_small))

        nu_s, nu_perp, nu_parallel = self.coulomb_fp_rates_li_he(
            w,
            n_e,
            Ti_ev,
            lnLambda=lnLambda,
            Z_a=Z_a,
            Z_b=Z_b,
            m_a_amu=m_a_amu,
            m_b_amu=m_b_amu,
        )

        nu_s = nu_s[:, None]
        nu_perp = nu_perp[:, None]
        nu_parallel = nu_parallel[:, None]

        w_hat = w / wmag
        eta = torch.randn_like(v)
        eta_parallel = torch.sum(eta * w_hat, dim=-1, keepdim=True)
        eta_perp = eta - eta_parallel * w_hat

        dt_h = self.dt / 2
        var_parallel = torch.clamp(nu_parallel * w2 * dt_h, min=0.0)
        var_perp_each = torch.clamp(0.5 * nu_perp * w2 * dt_h, min=0.0)

        dw_drag = -nu_s * w * dt_h
        dw_parallel = torch.sqrt(var_parallel) * eta_parallel * w_hat
        dw_perp = torch.sqrt(var_perp_each) * eta_perp

        return u_b + w + dw_drag + dw_parallel + dw_perp

    def parallel_solver(self, ions, Bfield, Efield=None, nfield=None, trace_IDs=[], freq_corr=False, collisions=None):
        """
        Function to take in a particle and field object and solves the particle path until termination event or tmax
        using a fixed-step Boris-Buneman Solver, based on (Birdsall, 4-3&4).

        Parameters:
            -ions (list): List of ion objects containing initial conditions and properties.
            -Bfield (object): Magnetic field object providing field interpolation methods.
            -Efield (object, optional): Electric field object providing field interpolation methods. Defaults to None.
            -nfield (object, optional): Neutral field object providing field interpolation methods. Defaults to None.
            -track_ID (list, optional): List of particle IDs to track. Defaults to [10, 20].
            -freq_corr (bool, optional): Flag to enable frequency correction. Defaults to False.
            -collisions (str or None): Collision model. Valid options are 'viscous_drag_hstep',
                'langevin_in_hstep', 'linearFP_ii_hstep', 'fokker_planck_ii_hstep', or None.
        Returns:
            -wallPts (torch.Tensor): XYZ Positions where particles terminate (e.g., hit the wall), shape (Nparticles, 3).
            -wallVelocities (torch.Tensor): Velocities of particles at termination, shape (Nparticles, 3).
            -maxStep (torch.Tensor): Step index at which each particle terminated, shape (Nparticles,).
        """
        log = logging.getLogger()
        log.info('Start ICs: {}-{}'.format(ions[0].particleID, ions[-1].particleID))

        collision_model = self._resolve_collision_model(collisions)
        collision_uses_density = self._collision_uses_density(collision_model)
        if collision_model:
            log.info('Collision model: {}'.format(collision_model))

        t_startInd = perf_counter()
        Nparticles = len(ions)
        trace_output = torch.zeros([self.nsteps+1, len(trace_IDs), 3], dtype=torch.float64, device=device)
        with torch.no_grad():
            wallPts = torch.zeros([Nparticles, 3], dtype=torch.float64, device=device)
            wallVelocities = torch.zeros([Nparticles, 3], dtype=torch.float64, device=device)
            maxStep = torch.zeros(Nparticles, dtype=torch.int, device=device)
            tvec = torch.empty([Nparticles, 3], dtype=torch.float64, device=device)

            qdt2m = torch.tensor([ion.charge_mass_ratio * self.dt / 2 for ion in ions], dtype=torch.float64, device=device)
            kbTgasqMi = torch.tensor(
                (kboltz * self.T_gas_eV) / (self.m_gas_amu * kg_per_amu),
                dtype=torch.float64,
                device=device,
            ) # convert to Joules and divide by mass to get velocity squared units
            # kbTqMi = torch.tensor(
            #     (kboltz * 2.0) / (self.m_gas_amu * kg_per_amu),
            #     dtype=torch.float64,
            #     device=device,
            # ) # convert to Joules and divide by mass to get velocity squared units


            v_k = torch.tensor(np.array([ion.vel0_XYZ for ion in ions]), dtype=torch.float64, device=device)

            [ion.setPosition(0, ion.pos0_XYZ) for ion in ions]
            pos_k = torch.tensor(np.array([ion.pos0_XYZ for ion in ions]), dtype=torch.float64, device=device)

            # NEED v_n-1/2 TO START
            if Efield:
                Evec = (Efield.interpField(pos_k) * qdt2m).T
            else:
                Evec = torch.zeros([Nparticles, 3], dtype=torch.float64, device=device)

            if freq_corr:
                Bvec = torch.empty([Nparticles, 3], dtype=torch.float64, device=device)
                Bvec = Bfield.interpField(pos_k).T
                Bmag = torch.linalg.norm(Bvec, axis=-1)
                Bhat = Bvec / Bmag[:, None]
                tvec = torch.tan(qdt2m * Bmag)[:, None] * Bhat
            else:
                tvec = (Bfield.interpField(pos_k) * qdt2m).T

            tmag = torch.linalg.norm(tvec, axis=-1)

            vminus = v_k + Evec
            vprime = vminus + torch.linalg.cross(vminus, tvec)
            svec = 2 * tvec / (1 + (tmag * tmag)[:, None])
            vplus = vminus - torch.linalg.cross(vprime, svec) / 2
            v_k = vplus + Evec

            x2 = pos_k.T[0] * pos_k.T[0]
            y2 = pos_k.T[1] * pos_k.T[1]
            z2 = pos_k.T[2] * pos_k.T[2]
            r_k = torch.sqrt(x2 + y2 + z2 + Bfield.R0 * Bfield.R0
                              - 2 * Bfield.R0 * torch.sqrt(x2 + y2))

            running = torch.arange(0, Nparticles, 1, dtype=torch.int, device=device)
            Nrunning = Nparticles

            # ADD SELECTED PARTICLE TRACING
            trace_output[0] = pos_k[trace_IDs]

            log.info('START STEPPING...')
            logging.basicConfig(level=logging.INFO)
            with logging_redirect_tqdm(loggers=[log]):
                pbar = tqdm(range(1, self.nsteps), ncols=100, mininterval=2.0)



                for k in pbar:
                    pos_active = pos_k[running]
                    qdt2m_active = qdt2m[running]
                    v_k_active = v_k[running]


                    actv_weights, actv_corner_indices, actv_ph_localN = Bfield.get_weights(pos_active)
                    b_vecs_active = Bfield.return_vecs(actv_weights, actv_corner_indices, actv_ph_localN)
                    full_phi_corner_indices = None
                    if Efield or (collision_uses_density and nfield):
                        sector = torch.remainder(actv_ph_localN.to(torch.long), Bfield.periodicity[2])
                        phi_offset = sector.unsqueeze(0) * Bfield.nphi
                        full_phi_corner_indices = torch.stack([
                            actv_corner_indices[0],
                            actv_corner_indices[1],
                            actv_corner_indices[2] + phi_offset,
                        ])

                    if Efield:
                        e_vecs_active = Efield.return_vecs(actv_weights, full_phi_corner_indices, ph_localN=None)
                        Evec_active = (e_vecs_active * qdt2m_active).T
                    else:
                        Evec_active = torch.zeros_like(v_k_active)

                    # INSERT (FIRST) COLLISION HALF-STEP HERE IF DESIRED
                    ne_active = None
                    if collision_model:
                        if collision_uses_density:
                            if nfield:
                                ne_active = nfield.return_scalars(actv_weights, full_phi_corner_indices)
                            else:
                                ne_active = torch.full((pos_active.shape[0],), 1e18, dtype=v_k_active.dtype, device=v_k_active.device)

                        v_k_active = self._apply_collision_hstep(
                            collision_model,
                            pos_active,
                            v_k_active,
                            n_e=ne_active,
                            Ti_ev=self.Ti_eV,
                            kbTgasqMi=kbTgasqMi,
                        )

                    if freq_corr:
                        Bmag_active = torch.linalg.norm(b_vecs_active, axis=-1)
                        Bhat_active = b_vecs_active / Bmag_active[:, None]
                        tvec_active = torch.tan(qdt2m_active * Bmag_active)[:, None] * Bhat_active
                    else:
                        tvec_active = (b_vecs_active * qdt2m_active).T

                    tmag_active = torch.linalg.norm(tvec_active, axis=-1)



                    vminus_active = v_k_active + Evec_active
                    vprime_active = vminus_active + torch.linalg.cross(vminus_active, tvec_active)
                    svec_active = 2 * tvec_active / (1 + (tmag_active * tmag_active)[:, None])
                    vplus_active = vminus_active + torch.linalg.cross(vprime_active, svec_active)
                    
                    v_k_active = vplus_active + Evec_active

                    pos_active += v_k_active * self.dt
    
                    # INSERT (SECOND)COLLISION HALF-STEP HERE IF DESIRED
                    ##-----------------------------------------##
                    if collision_model:
                        v_k_active = self._apply_collision_hstep(
                            collision_model,
                            pos_active,
                            v_k_active,
                            n_e=ne_active,
                            Ti_ev=self.Ti_eV,
                            kbTgasqMi=kbTgasqMi,
                        )
                    ##-----------------------------------------##

                    # Update particle positions and velocities
                    pos_k[running] = pos_active
                    v_k[running] = v_k_active



                    x2 = pos_k.T[0]**2
                    y2 = pos_k.T[1]**2
                    z2 = pos_k.T[2]**2
                    r_k = torch.sqrt(x2 + y2 + z2 + Bfield.R0 * Bfield.R0
                                               - 2 * Bfield.R0 * torch.sqrt(x2 + y2))
                    running = torch.where(r_k < Bfield.a)[0]


                    maxStep[running] = k # +1?


                    # ADD SELECTED PARTICLE TRACING
                    trace_output[k] = pos_k[trace_IDs]

                    Nrunning = running.size(0)
                    if Nrunning == 0:
                        log.info('All particles terminated at step {}'.format(k))
                        break

                    pbar.set_postfix({'#Particles running': Nrunning}, refresh=False)




            terminated = torch.where(r_k >= Bfield.a)[0]
            wallPts[terminated] = pos_k[terminated]
            wallVelocities[terminated] = v_k[terminated]

        t_stopInd = perf_counter()
        elapsed_timeInd = t_stopInd - t_startInd
        min_, sec_ = divmod(elapsed_timeInd, 60)
        hr_, min_ = divmod(min_, 60)

        log.info(
            'ELAPSED TIME({} Particles): {:02.0f}H:{:02.0f}M:{:02.3f}S'.format(
                Nparticles, hr_, min_, sec_
            )
        )

        return wallPts, wallVelocities, maxStep, trace_output

    def single_solver(self, particle):
        pass

    def post_solver(self, solver_output, Bfield):
        """Processes the solver output to extract path lengths and Poincare data,
        and prepares the data for plotting and output.

        Args:
            solver_output (iterator): The output from the solver, containing tuples of path lengths and event data.

        Returns:
            tuple: (path_lengths, poincare_points, wall_points)
                path_lengths (list): List of path lengths for each particle.
                poincare_points (list): List of Poincare data for each particle.
                wall_points (list): List of wall intersection data for each particle.
        """
        ## SOME PHYSICAL CONSTANTS
        #kg_per_amu = 1.660_539_068E-27
        kboltz = 1.602_176_634E-19 # Joules/eV

        wallPts_, wallVelocities_, maxStep_, trace_output_ = solver_output

        tic = perf_counter()
        wallPt_output = wallPts_.cpu().numpy()
        velocity_output = wallVelocities_.cpu().numpy()
        max_timeStep = maxStep_.cpu().numpy()
        ion_traces = trace_output_.cpu().numpy()

        # filter out rows containing all zeros
        wallPt_output = wallPt_output[~np.all(wallPt_output == 0, axis=1)]
        # Filter velocity_output and get the indices of nonzero rows
        nonzero_indices = ~np.all(velocity_output == 0, axis=1)
        velocity_output = velocity_output[nonzero_indices]
        max_timeStep = max_timeStep[nonzero_indices]

        speed_output = np.linalg.norm(velocity_output, axis=1)
        ion_mass_kg = self.ion_list[0].mass #* kg_per_amu
        energy_output = 0.5 * ion_mass_kg * speed_output**2 / kboltz #convert speed to energy in eV
        self.IO.log.info('Energy output stats: min={:.2f} eV, max={:.2f} eV, avg={:.2f} eV'.format(
            np.min(energy_output), np.max(energy_output), np.mean(energy_output)))

        wallPtArray = np.asarray( [XYZ_to_RTP(wall_point, Bfield.R0) for wall_point in wallPt_output] ).T
        outputArray = np.vstack((wallPtArray, velocity_output.T, max_timeStep[None, :]))

        ## CALCULATE UNIT VECTORS
        unit_vec_xyz = velocity_output/speed_output[:, None]  # Normalize the velocity vectors to get unit vectors
        radial_vec_xyz = np.asarray( [RTP_XYZ_JAC(wall_point, np.array([1,0,0]), form='rtp2xyz') for wall_point in wallPtArray.T] )# Convert unit vectors to RTP coordinates
        toroidal_vec_xyz = np.asarray( [RTP_XYZ_JAC(wall_point, np.array([0,0,1]), form='rtp2xyz') for wall_point in wallPtArray.T] )# Convert unit vectors to RTP coordinates

        ## CALCULATE ANGLE FROM NORMAL    
        deposition_angles = np.arccos(np.einsum('ij,ij->i', unit_vec_xyz, radial_vec_xyz))  # Calculate angles between unit vectors and radial vectors
        deposition_angles_deg = np.degrees(deposition_angles)  # Convert angles to degrees
        ## CALCULATE TOROIDAL ANGLE    
        cos_toroidal_angles = np.einsum('ij,ij->i', unit_vec_xyz, toroidal_vec_xyz)
        toroidal_angles = np.arccos(cos_toroidal_angles)
        toroidal_angles_deg = np.degrees(toroidal_angles)  # Convert angles to degrees


        self.IO.log.info('deposition_angles_deg min: {:.2f} deg, max: {:.2f} deg, avg: {:.2f} deg'.format(
            np.min(deposition_angles_deg), np.max(deposition_angles_deg), np.mean(deposition_angles_deg)))
        self.IO.log.info('toroidal_angles_deg min: {:.2f} deg, max: {:.2f} deg, avg: {:.2f} deg'.format(
            np.min(toroidal_angles_deg), np.max(toroidal_angles_deg), np.mean(toroidal_angles_deg)))
        toc = perf_counter()
        self.IO.log.info('OUTPUT SENT TO CPU AND CONVERTED TO RTP IN {}SEC'.format(toc-tic))

        return outputArray, energy_output, deposition_angles_deg, toroidal_angles_deg, ion_traces

    def save_output(self, outputArray, ion_traces):
        """Saves the output data to files in the specified output directory."""
        trace_filename = 'Ion_traces_' + self.cond_string+self.tag
        self.IO.saveNumpyData(ion_traces, trace_filename)
        self.IO.log.info('OUTPUT ION TRACES: {}'.format(trace_filename))

        wallpts_filename = 'Wallpt_OUTPUT_' + self.cond_string+self.tag
        self.IO.saveNumpyData(outputArray, wallpts_filename)
        self.IO.log.info('OUTPUT RESULT DATA: {}'.format(wallpts_filename))

    def run(self, Bfield, Efield=None, nfield=None, collisions=None, trace_IDs=[]):
        """Runs the Boris solver and processes the results.

        Args:
            Bfield: Magnetic field object providing field interpolation methods.
            Efield: Electric field object providing field interpolation methods. Defaults to None.
            nfield: Neutral field object providing field interpolation methods. Defaults to None.
            collisions: Collision model name, True for the legacy linear FP model, or None.
            trace_IDs: List of particle IDs to trace. Defaults to [].

        Returns:
            Tuple containing:
                outputArray (np.ndarray): Array of wall point and velocity data.
                energy_output (np.ndarray): Array of particle energies at termination (in eV).
                deposition_angles_deg (np.ndarray): Array of deposition angles (in degrees).
                toroidal_angles_deg (np.ndarray): Array of toroidal angles (in degrees).
                ion_traces (np.ndarray): Array of traced particle positions.
        """
        solv_out = self.parallel_solver(
            ions = self.ion_list,
            Bfield = Bfield,
            Efield = Efield,
            nfield = nfield,
            collisions = collisions,
            trace_IDs = trace_IDs
        )

        outputArray, energy_output, deposition_angles_deg, toroidal_angles_deg, ion_traces = self.post_solver(solv_out, Bfield)

        self.save_output(outputArray, ion_traces)

        return outputArray, energy_output, deposition_angles_deg, toroidal_angles_deg, ion_traces
