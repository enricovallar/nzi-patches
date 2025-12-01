import numpy as np
from scipy.optimize import brentq

def solve_slab_modes(d_top, d_mid, d_bot, eps_top, eps_mid, eps_bot, lambda0, z_grid_res=1000):
    """
    Calculates all guided TE modes for a symmetric/asymmetric slab waveguide.
    
    Parameters:
    - d_top, d_mid, d_bot: Thicknesses of top cladding, core, and bottom cladding (m).
      (Note: d_top and d_bot are treated as semi-infinite for the mode solution, 
       but used for plotting/integration limits).
    - eps_top, eps_mid, eps_bot: Relative permittivities of the layers.
    - lambda0: Vacuum wavelength (m).
    - z_grid_res: Number of points in the z-grid for the field profile.
    
    Returns:
    - modes: A list of dictionaries. Each dict contains:
        {'neff': float, 'theta': numpy array, 'z_grid': numpy array, 'confinement': float}
    - N_modes: Integer count of found modes.
    """
    
    k0 = 2 * np.pi / lambda0
    
    # Refractive indices
    n_top = np.sqrt(eps_top)
    n_core = np.sqrt(eps_mid)
    n_bot = np.sqrt(eps_bot)
    
    # The guided modes exist in the range max(n_top, n_bot) < neff < n_core
    n_min = max(n_top, n_bot)
    n_max = n_core
    
    if n_min >= n_max:
        print("Warning: No guided modes possible (Cladding index >= Core index).")
        return [], 0

    # Define the dispersion function for an asymmetric slab
    # Condition: tan(h * d) = (p + q) / (h * (1 - p*q/h^2))
    # Where:
    # h = k0 * sqrt(n_core^2 - neff^2)
    # q = k0 * sqrt(neff^2 - n_top^2)
    # p = k0 * sqrt(neff^2 - n_bot^2)
    
    def dispersion_func(neff):
        # Avoid singularities at neff = n_min or n_max
        if neff <= n_min + 1e-9 or neff >= n_max - 1e-9:
            return np.nan
            
        h = k0 * np.sqrt(n_core**2 - neff**2)
        q = k0 * np.sqrt(neff**2 - n_top**2)
        p = k0 * np.sqrt(neff**2 - n_bot**2)
        
        lhs = np.tan(h * d_mid)
        rhs = (p + q) / (h * (1 - (p * q) / h**2))
        return lhs - rhs

    # Root Finding: Scan the range [n_min, n_max] for sign changes
    # We use a fine grid to detect brackets for roots
    # The tangent function has singularities, so we must be careful.
    
    scan_points = np.linspace(n_min + 1e-6, n_max - 1e-6, 500)
    f_vals = [dispersion_func(n) for n in scan_points]
    
    roots = []
    for i in range(len(scan_points) - 1):
        if np.isnan(f_vals[i]) or np.isnan(f_vals[i+1]):
            continue
            
        # Check for sign change
        if f_vals[i] * f_vals[i+1] < 0:
            # Check if this is a root or a singularity (asymptote of tan)
            # If the difference is huge, it's likely a singularity
            if np.abs(f_vals[i] - f_vals[i+1]) < 10.0: 
                try:
                    root = brentq(dispersion_func, scan_points[i], scan_points[i+1])
                    roots.append(root)
                except ValueError:
                    pass
    
    # Sort roots descending (highest neff = fundamental mode TE0)
    roots = sorted(roots, reverse=True)
    
    # Construct Output
    found_modes = []
    
    # Define z-grid for field profile
    # Center the core at z=0
    z_total_half_width = d_mid + max(d_top, d_bot) # Sufficient padding
    z = np.linspace(-(d_mid/2 + d_bot), (d_mid/2 + d_top), z_grid_res)
    
    for i, neff in enumerate(roots):
        # Calculate Field Profile Theta(z)
        h = k0 * np.sqrt(n_core**2 - neff**2)
        q = k0 * np.sqrt(neff**2 - n_top**2)
        p = k0 * np.sqrt(neff**2 - n_bot**2)
        
        theta = np.zeros_like(z)
        
        # Coefficients (assuming Amplitude in core A = 1)
        # We need to match boundary conditions at z = -d_mid/2 and z = d_mid/2
        # General solution in core: A cos(hz) + B sin(hz)
        # Usually easier to shift coordinate system to interface for calculation, 
        # then shift back. Let's use the standard asymmetric slab form relative to core center.
        # But for arbitrary asymmetry, phase shift phi is easier: cos(hz - phi)
        
        # phi = arctan(p/h) - m*pi/2? No, let's use explicit matching.
        # At z = -d/2 (bottom interface): E = C_bot * exp(p(z + d/2))
        # At z = +d/2 (top interface):    E = C_top * exp(-q(z - d/2))
        # Inside: cos(h*z - phi)
        
        phi = np.arctan(p / h) # Phase shift determined by bottom cladding asymmetry
        # The mode profile is proportional to cos(h(z + d_mid/2) - phi)
        # Let's verify: at boundary z = -d_mid/2 (local coordinate 0), arg is -phi.
        # cos(-phi) = cos(phi). Slope is h*sin(phi).
        # Decay is exp(p*z'). Value 1. Slope p. Match: p/h = tan(phi). Correct.
        
        for j, z_val in enumerate(z):
            if z_val < -d_mid/2:
                # Bottom Cladding
                dist_from_interface = -d_mid/2 - z_val
                # Amplitude at interface
                amp_surf = np.cos(-phi)
                theta[j] = amp_surf * np.exp(-p * dist_from_interface)
                
            elif z_val > d_mid/2:
                # Top Cladding
                dist_from_interface = z_val - d_mid/2
                # Amplitude at interface
                amp_surf = np.cos(h * d_mid - phi)
                theta[j] = amp_surf * np.exp(-q * dist_from_interface)
                
            else:
                # Core
                # z goes from -d/2 to d/2
                # Argument goes from 0 to d
                theta[j] = np.cos(h * (z_val + d_mid/2) - phi)
        
        # Normalize (Integral |Theta|^2 dz = 1)
        norm_factor = np.trapz(theta**2, z)
        theta_norm = theta / np.sqrt(norm_factor)
        
        # Confinement Factor (Fraction of energy in core)
        mask_core = (z >= -d_mid/2) & (z <= d_mid/2)
        confinement = np.trapz(theta_norm[mask_core]**2, z[mask_core])
        
        mode_data = {
            'mode_index': i,
            'neff': neff,
            'theta': theta_norm,
            'z_grid': z,
            'confinement': confinement
        }
        found_modes.append(mode_data)
        
    return found_modes, len(found_modes)