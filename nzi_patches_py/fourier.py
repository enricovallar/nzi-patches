import numpy as np
from scipy.special import j1

def get_g_vectors(gmax, a):
    """
    Generate G-vectors for a square lattice up to a cutoff gmax.
    """
    b = 2 * np.pi / a
    # Estimate max index
    n_max = int(np.ceil(gmax / b))
    
    g_vecs = []
    
    for m in range(-n_max, n_max + 1):
        for n in range(-n_max, n_max + 1):
            gx = m * b
            gy = n * b
            g_mag = np.sqrt(gx**2 + gy**2)
            if g_mag <= gmax + 1e-9: # Add small tolerance
                g_vecs.append([gx, gy])
                
    return np.array(g_vecs).T # Shape (2, N_g)

def get_circle_ft(g_vecs, r, center, a):
    """
    Analytic FT of a circle in a unit cell.
    FT = integral( exp(-i G.r) ) over circle
    """
    Gx = g_vecs[0, :]
    Gy = g_vecs[1, :]
    G_mag = np.sqrt(Gx**2 + Gy**2)
    
    cx, cy = center
    # Phase factor exp(-i G . r_c)
    phase = np.exp(-1j * (Gx * cx + Gy * cy))
    
    ft_shape = np.zeros_like(G_mag, dtype=complex)
    
    # Handle G=0
    mask_zero = (G_mag < 1e-9)
    ft_shape[mask_zero] = np.pi * r**2
    
    # Handle G!=0
    mask_nonzero = ~mask_zero
    G_nz = G_mag[mask_nonzero]
    ft_shape[mask_nonzero] = 2 * np.pi * r * j1(G_nz * r) / G_nz
    
    return ft_shape * phase

def get_epsilon_coefficients_analytic(g_vecs, eps_bg, shapes, a):
    """
    Calculate epsilon coefficients for a background with shapes.
    shapes: list of dicts {'eps': float, 'r': float, 'center': (float, float)}
    """
    Gx = g_vecs[0, :]
    Gy = g_vecs[1, :]
    G_mag = np.sqrt(Gx**2 + Gy**2)
    
    # Term 1: eps_bg * delta(G)
    eps_coeffs = np.zeros_like(G_mag, dtype=complex)
    mask_zero = (G_mag < 1e-9)
    eps_coeffs[mask_zero] = eps_bg
    
    area = a**2
    
    for shape in shapes:
        eps_shape = shape['eps']
        r = shape['r']
        center = shape['center']
        
        ft_shape = get_circle_ft(g_vecs, r, center, a)
        
        # Add contribution: (eps_shape - eps_bg) * FT_shape / Area
        eps_coeffs += (eps_shape - eps_bg) * ft_shape / area
        
    return eps_coeffs

def get_xi_mn(m, n, a, g_vecs, coeffs):
    """
    Retrieve Xi_mn.
    """
    b = 2 * np.pi / a
    target_Gx = m * b
    target_Gy = n * b
    
    dist_sq = (g_vecs[0, :] - target_Gx)**2 + (g_vecs[1, :] - target_Gy)**2
    min_dist_idx = np.argmin(dist_sq)
    
    if dist_sq[min_dist_idx] > 1e-6:
        print(f"Warning: G_({m},{n}) not found in expansion")
        return 0.0
        
    return coeffs[min_dist_idx]