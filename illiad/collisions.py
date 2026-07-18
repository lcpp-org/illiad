"""Ion-neutral and ion-ion collision operators."""

import math

import torch

## SOME PHYSICAL CONSTANTS
kg_per_amu = 1.660_539_068E-27
kboltz = 1.602_176_634E-19 # Joules/eV
eps0 = 8.854_187_8128E-12
sqrt_pi = math.sqrt(math.pi)
Li_mass = 6.941 #amu
He_mass = 4.002602 #amu

class Collisions():
    def viscous_drag_hstep(self, x, v, n_gas=3e18, sigma_mt=1e-19):
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

    def langevin_in_hstep(self, x, v, n_gas=3e18, sigma_mt=1e-19, kbTgasqMi=(kboltz*0.025) / (4.002602 * kg_per_amu) ):

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

    def linearFP_ii_hstep(self, x, v, n_e=1e18, Ti_ev=2.0):

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

    def _resolve_ion_neutral_collision_model(self, collisions):
        collision_model = self._resolve_collision_model(collisions)
        if collision_model in (None, 'viscous_drag_hstep', 'langevin_in_hstep'):
            return collision_model
        raise ValueError(f'Unknown ion-neutral collisions option {collisions!r}.')

    def _resolve_ion_ion_collision_model(self, collisions):
        collision_model = self._resolve_collision_model(collisions)
        if collision_model in (None, 'linearFP_ii_hstep', 'fokker_planck_ii_hstep'):
            return collision_model
        raise ValueError(f'Unknown ion-ion collisions option {collisions!r}.')

    def _collision_uses_density(self, collision_model):
        return collision_model in ('linearFP_ii_hstep', 'fokker_planck_ii_hstep')

    def _apply_collision_hstep(self, collision_model, x, v, n_e=None, n_gas=3e18, Ti_ev=2.0, kbTgasqMi=None):
        if collision_model == 'viscous_drag_hstep':
            return self.viscous_drag_hstep(x, v, n_gas=n_gas)
        if collision_model == 'langevin_in_hstep':
            if kbTgasqMi is None:
                return self.langevin_in_hstep(x, v, n_gas=n_gas)
            return self.langevin_in_hstep(x, v, n_gas=n_gas, kbTgasqMi=kbTgasqMi)

        if n_e is None:
            n_e = torch.full((v.shape[0],), 1e18, dtype=v.dtype, device=v.device)

        if collision_model == 'linearFP_ii_hstep':
            return self.linearFP_ii_hstep(x, v, n_e, Ti_ev=Ti_ev)
        if collision_model == 'fokker_planck_ii_hstep':
            return self.fokker_planck_ii_hstep(x, v, n_e, Ti_ev=Ti_ev)

        return v

    def fokker_planck_ii_hstep( self, x, v, n_e=1e18, Ti_ev=2.0, lnLambda=10.0, u_b=None,
                               Z_a=1.0, Z_b=1.0, m_a_amu=Li_mass, m_b_amu=He_mass, w_small=1.0e-6):

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
