"""
Cosmological Background Evolution with FDM + Quintessence
Based on the CLASS modification by Silvio A. Corrêa Junior
Paper: Dynamical Dark Sector: FDM + Quintessence (arXiv)

This module implements the background cosmological equations for:
- Standard ΛCDM components (radiation, baryons, CDM, cosmological constant)
- Fuzzy Dark Matter (FDM) - treated as a fluid with effective EoS
- Quintessence (PNGB) - axion-like scalar field as dynamic dark energy
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

# Physical constants
C_LIGHT = 299792.458  # km/s
H0_UNITS = 100.0  # km/s/Mpc per h
M_PL = 2.435e18  # Reduced Planck mass in GeV
EV_TO_GEV = 1e-9
MPC_TO_KM = 3.0857e19  # Mpc in km


@dataclass
class CosmologyParams:
    """Cosmological parameters for background evolution"""
    # Hubble parameter
    h: float = 0.67
    
    # Density parameters (Omega0)
    Omega_b: float = 0.05       # Baryons
    Omega_cdm: float = 0.25     # Cold Dark Matter
    Omega_g: float = 5.38e-5    # Photons
    Omega_ur: float = 3.76e-5   # Ultra-relativistic neutrinos
    
    # Curvature
    Omega_k: float = 0.0
    
    # Quintessence parameters (PNGB axion)
    has_quintessence: bool = False
    M_quint: float = 1.5e-3     # Mass scale in M_pl units
    f_quint: float = 1.0        # Decay constant in M_pl units
    phi_ini: float = 0.1        # Initial field value
    phi_prime_ini: float = 0.0  # Initial field derivative
    
    # FDM parameters
    has_fdm: bool = False
    m_Psi: float = 1e-22        # FDM mass in eV
    Omega_fdm: float = 0.25     # FDM density (replaces CDM when has_fdm=True)
    
    # Integration settings
    a_ini: float = 1e-10        # Initial scale factor
    a_end: float = 1.0          # Final scale factor (today)
    N_points: int = 2000        # Number of output points
    
    @property
    def H0(self) -> float:
        """Hubble constant in natural units (1/Mpc)"""
        return self.h * H0_UNITS / C_LIGHT  # in 1/Mpc
    
    @property
    def H0_SI(self) -> float:
        """Hubble constant in km/s/Mpc"""
        return self.h * H0_UNITS
    
    @property
    def Omega_lambda(self) -> float:
        """Dark energy density from closure"""
        omega_m = self.Omega_b + self.Omega_cdm
        if self.has_fdm:
            omega_m += self.Omega_fdm
        omega_r = self.Omega_g + self.Omega_ur
        return max(0, 1.0 - omega_m - omega_r - self.Omega_k)
    
    @property
    def rho_crit_0(self) -> float:
        """Critical density today (in natural units)"""
        return 3.0 * self.H0**2


class BackgroundEvolution:
    """
    Solves the cosmological background equations with FDM and Quintessence.
    
    Uses log(a) as the integration variable for better numerical stability.
    FDM is treated using the fluid approximation (w ≈ 0 for late times).
    """
    
    def __init__(self, params: CosmologyParams):
        self.params = params
        self.solution = None
        self._setup_indices()
    
    def _setup_indices(self):
        """Setup array indices for the ODE system"""
        idx = 0
        
        if self.params.has_quintessence:
            self.idx_phi = idx; idx += 1
            self.idx_phi_dot = idx; idx += 1
        else:
            self.idx_phi = None
            self.idx_phi_dot = None
            
        self.n_vars = max(idx, 1)
    
    def _E_squared(self, a: float, phi: float = 0, phi_dot: float = 0) -> float:
        """
        Compute E²(a) = H²(a)/H₀² (dimensionless Hubble parameter squared)
        
        Using the Friedmann equation in terms of density parameters.
        """
        p = self.params
        
        # Standard components
        E2 = 0.0
        
        # Radiation (photons + relativistic neutrinos)
        E2 += (p.Omega_g + p.Omega_ur) / a**4
        
        # Baryons
        E2 += p.Omega_b / a**3
        
        # Cold Dark Matter
        E2 += p.Omega_cdm / a**3
        
        # Fuzzy Dark Matter (behaves like CDM at late times, radiation-like at early times)
        if p.has_fdm:
            # Transition scale where m = H
            # For ultra-light masses, FDM behaves like DE at early times, CDM at late times
            a_osc = self._fdm_transition_scale()
            if a < a_osc:
                # Early times: frozen field, behaves like dark energy
                E2 += p.Omega_fdm  # w ≈ -1
            else:
                # Late times: oscillating field, behaves like matter
                E2 += p.Omega_fdm / a**3
        
        # Curvature
        E2 += p.Omega_k / a**2
        
        # Cosmological constant (or quintessence replaces it)
        if p.has_quintessence:
            # Quintessence contribution
            M4 = p.M_quint**4
            V = M4 * (1.0 + np.cos(phi / p.f_quint))
            
            # Energy density in units of ρ_crit,0
            rho_phi = 0.5 * phi_dot**2 + V
            E2 += rho_phi / (3.0 * p.H0**2)
        else:
            # Cosmological constant
            E2 += p.Omega_lambda
        
        return max(E2, 1e-30)  # Prevent negative values
    
    def _fdm_transition_scale(self) -> float:
        """
        Compute the scale factor at which FDM transitions from DE-like to matter-like.
        This occurs when m ≈ H, i.e., the field starts oscillating.
        """
        p = self.params
        
        # Convert FDM mass to Hubble units
        # m in eV, H0 in km/s/Mpc
        # m/H0 = (m * eV_to_Hz) / (H0 * km/s/Mpc_to_Hz)
        m_eV = p.m_Psi
        H0_eV = p.H0_SI * 1e3 / MPC_TO_KM * 6.582e-16  # Convert to eV
        
        # m = H = H0 * E(a)
        # For radiation domination: E ∝ a^-2, so m = H0/a^2 → a_osc = sqrt(H0/m)
        # For matter domination: E ∝ a^-3/2, so a_osc = (H0/m)^(2/3)
        
        if m_eV > 0:
            ratio = H0_eV / m_eV
            # Use radiation-era formula as approximation
            a_osc = min(np.sqrt(ratio), 0.1)
        else:
            a_osc = 1e-10
        
        return max(a_osc, 1e-12)
    
    def _derivs(self, log_a: float, y: np.ndarray) -> np.ndarray:
        """
        Compute derivatives with respect to log(a) = ln(a).
        
        d/d(ln a) = (1/H) d/dt
        """
        p = self.params
        a = np.exp(log_a)
        dy = np.zeros(self.n_vars)
        
        if p.has_quintessence and self.idx_phi is not None:
            phi = y[self.idx_phi]
            phi_dot = y[self.idx_phi_dot]  # d(phi)/dt in units of H0
            
            E2 = self._E_squared(a, phi, phi_dot)
            E = np.sqrt(E2)
            
            M4 = p.M_quint**4
            f = p.f_quint
            
            dVdphi = -M4 / f * np.sin(phi / f)
            
            # d(phi)/d(ln a) = (1/H) d(phi)/dt = phi_dot / (H0 * E)
            dy[self.idx_phi] = phi_dot / (p.H0 * E)
            
            # Klein-Gordon: φ'' + 3Hφ' + dV/dφ = 0
            # In terms of d/d(ln a):
            # d(phi_dot)/d(ln a) = -3 * phi_dot - dVdphi / (H0 * E)
            dy[self.idx_phi_dot] = -3.0 * phi_dot - dVdphi / (p.H0 * E)
        
        return dy
    
    def solve(self) -> Dict:
        """
        Solve the background evolution equations.
        
        Returns a dictionary with all background quantities.
        """
        p = self.params
        
        # Create log(a) array for output
        log_a_array = np.linspace(np.log(p.a_ini), np.log(p.a_end), p.N_points)
        a_array = np.exp(log_a_array)
        
        # Initialize results arrays
        E_array = np.zeros(p.N_points)
        
        if p.has_quintessence:
            phi_array = np.zeros(p.N_points)
            phi_dot_array = np.zeros(p.N_points)
            
            # Initial conditions
            y0 = np.array([p.phi_ini, p.phi_prime_ini])
            
            # Solve ODE
            result = solve_ivp(
                self._derivs,
                (log_a_array[0], log_a_array[-1]),
                y0,
                method='RK45',
                t_eval=log_a_array,
                rtol=1e-8,
                atol=1e-10,
                max_step=0.1
            )
            
            if not result.success:
                raise RuntimeError(f"Integration failed: {result.message}")
            
            phi_array = result.y[self.idx_phi]
            phi_dot_array = result.y[self.idx_phi_dot]
            
            # Compute E(a) with quintessence
            for i, a in enumerate(a_array):
                E_array[i] = np.sqrt(self._E_squared(a, phi_array[i], phi_dot_array[i]))
        else:
            phi_array = None
            phi_dot_array = None
            
            # Compute E(a) without quintessence
            for i, a in enumerate(a_array):
                E_array[i] = np.sqrt(self._E_squared(a))
        
        # Compute all quantities
        z_array = 1.0 / a_array - 1.0
        H_array = E_array * p.H0_SI  # in km/s/Mpc
        
        # Density parameters
        results = {
            'a': a_array,
            'z': z_array,
            'E': E_array,
            'H': H_array,
            'H_conf': a_array * H_array / C_LIGHT,  # Conformal H in 1/Mpc
            'H_phys': H_array / C_LIGHT,  # Physical H in 1/Mpc
        }
        
        # Compute Omega parameters
        E2 = E_array**2
        
        results['Omega_g'] = p.Omega_g / (a_array**4 * E2)
        results['Omega_ur'] = p.Omega_ur / (a_array**4 * E2)
        results['Omega_r'] = results['Omega_g'] + results['Omega_ur']
        
        results['Omega_b'] = p.Omega_b / (a_array**3 * E2)
        results['Omega_cdm'] = p.Omega_cdm / (a_array**3 * E2)
        
        if p.has_fdm:
            a_osc = self._fdm_transition_scale()
            Omega_fdm = np.zeros_like(a_array)
            w_fdm = np.zeros_like(a_array)
            
            for i, a in enumerate(a_array):
                if a < a_osc:
                    Omega_fdm[i] = p.Omega_fdm / E2[i]
                    w_fdm[i] = -1.0
                else:
                    Omega_fdm[i] = p.Omega_fdm / (a**3 * E2[i])
                    w_fdm[i] = 0.0
            
            results['Omega_fdm'] = Omega_fdm
            results['w_fdm'] = w_fdm
            results['a_osc_fdm'] = a_osc
            results['Omega_m'] = results['Omega_b'] + results['Omega_cdm'] + Omega_fdm
        else:
            results['Omega_m'] = results['Omega_b'] + results['Omega_cdm']
        
        results['Omega_k'] = p.Omega_k / (a_array**2 * E2)
        
        if p.has_quintessence:
            M4 = p.M_quint**4
            V = M4 * (1.0 + np.cos(phi_array / p.f_quint))
            
            rho_quint = 0.5 * phi_dot_array**2 + V
            p_quint = 0.5 * phi_dot_array**2 - V
            
            results['phi'] = phi_array
            results['phi_prime'] = phi_dot_array
            results['V_quint'] = V
            results['rho_quint'] = rho_quint
            results['Omega_quint'] = rho_quint / (3.0 * p.H0**2 * E2)
            results['w_quint'] = np.where(rho_quint > 1e-30, p_quint / rho_quint, -1.0)
            results['Omega_lambda'] = np.zeros_like(a_array)
            results['Omega_de'] = results['Omega_quint']
        else:
            results['Omega_lambda'] = p.Omega_lambda / E2
            results['Omega_de'] = results['Omega_lambda']
        
        # Add FDM to dark sector if present
        if p.has_fdm:
            # FDM is dark matter, not dark energy in late universe
            pass
        
        self.solution = results
        self.params_used = p
        return results


def compute_cosmology(
    h: float = 0.67,
    Omega_b: float = 0.05,
    Omega_cdm: float = 0.25,
    has_quintessence: bool = False,
    M_quint: float = 1.5e-3,
    f_quint: float = 1.0,
    phi_ini: float = 0.1,
    has_fdm: bool = False,
    m_Psi: float = 1e-22,
    Omega_fdm: float = 0.25,
) -> Dict:
    """
    Convenience function to compute cosmological background evolution.
    
    Returns dictionary with all background quantities.
    """
    params = CosmologyParams(
        h=h,
        Omega_b=Omega_b,
        Omega_cdm=Omega_cdm,
        has_quintessence=has_quintessence,
        M_quint=M_quint,
        f_quint=f_quint,
        phi_ini=phi_ini,
        has_fdm=has_fdm,
        m_Psi=m_Psi,
        Omega_fdm=Omega_fdm,
    )
    
    bg = BackgroundEvolution(params)
    return bg.solve()
