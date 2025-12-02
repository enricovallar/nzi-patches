import numpy as np
from scipy.integrate import simpson as simps

def get_beta0(a):
    """Basic wave propagation constant (2nd order Gamma point)."""
    return 2 * np.pi / a

def calculate_greens_integral(theta_z, z_grid, n0_z, k0):
    """
    Calculates the overlap integral for the vertical coupling.
    In the 5-wave model, this determines the coupling strength (kappa_v)
    between the guided dipoles and the monopole mode.
    """
    if len(z_grid) > 1:
        dz = z_grid[1] - z_grid[0]
    else:
        return 0.0 
    
    beta_z = k0 * n0_z
    Z, ZP = np.meshgrid(z_grid, z_grid, indexing='ij')
    BZ, BZP = np.meshgrid(beta_z, beta_z, indexing='ij')
    TH, THP = np.meshgrid(theta_z, theta_z, indexing='ij')
    
    # Green's function approximation for vertical radiation/coupling
    with np.errstate(divide='ignore', invalid='ignore'):
        green_kernel = (-1j / (2 * BZ)) * np.exp(-1j * BZ * np.abs(Z - ZP))
    
    green_kernel[~np.isfinite(green_kernel)] = 0.0
    integrand = green_kernel * THP * np.conj(TH)
    
    inner = np.trapz(integrand, z_grid, axis=1)
    result = np.trapz(inner, z_grid)
    
    return result

def construct_cwt_matrices(params):
    """
    Constructs the 5x5 CWT Hamiltonian for NZI/Dirac-Cone analysis.
    
    Basis: [Rx, Sx, Ry, Sy, A0]
    - Rx, Sx, Ry, Sy: The 4 fundamental dipole waves (TE-like).
    - A0: The Monopole (breathing) mode (G=0).
    """
    a = params['a']
    lam = params['lambda0']
    k0 = 2 * np.pi / lam
    beta0 = 2 * np.pi / a
    
    # Prefactor from Liang (Eq 19)
    prefactor = (k0**2) / (2 * beta0)
    
    # Fourier Coefficients
    xi_20 = params['xi'].get((2,0), 0j)
    xi_02 = params['xi'].get((0,2), 0j)
    xi_10 = params['xi'].get((1,0), 0j)
    xi_01 = params['xi'].get((0,1), 0j)
    xi_11 = params['xi'].get((1,1), 0j) # Needed for C_2D
    
    gamma = params['conf']
    
    # Initialize 5x5 Matrix
    C = np.zeros((5, 5), dtype=complex)
    
    # --- 1. Dipole-Dipole Feedback (Standard Bragg) ---
    kappa_x = -prefactor * xi_20 * gamma
    kappa_y = -prefactor * xi_02 * gamma
    
    C[0, 1] = kappa_x      # Rx <- Sx
    C[1, 0] = np.conj(kappa_x) 
    C[2, 3] = kappa_y      # Ry <- Sy
    C[3, 2] = np.conj(kappa_y) 
    
    # --- 2. Monopole-Dipole Coupling (Vertical) ---
    # We treat A0 as a basis mode. The coupling depends on the overlap integral.
    g_int = calculate_greens_integral(params['theta_z'], params['z_grid'], params['n0_z'], k0)
    
    # Heuristic coupling strength derived from the radiation term
    # This replaces the imaginary "loss" with a real/complex coupling coefficient
    kappa_v = -(k0**2 / (2 * beta0)) * np.sqrt(np.abs(g_int))
    
    # Coupling Rx <-> A0 (via xi_10)
    C[0, 4] = kappa_v * xi_10
    C[4, 0] = np.conj(kappa_v * xi_10)
    
    # Coupling Sx <-> A0 (via xi_-10)
    C[1, 4] = kappa_v * np.conj(xi_10)
    C[4, 1] = np.conj(kappa_v * np.conj(xi_10))
    
    # Coupling Ry <-> A0 (via xi_01)
    C[2, 4] = kappa_v * xi_01
    C[4, 2] = np.conj(kappa_v * xi_01)
    
    # Coupling Sy <-> A0 (via xi_0-1)
    C[3, 4] = kappa_v * np.conj(xi_01)
    C[4, 3] = np.conj(kappa_v * np.conj(xi_01))

    # --- 3. Monopole Detuning (Diagonal) ---
    # The user can manually tune the A0 frequency to simulate changing hole radius.
    # detuning = 0 means A0 is perfectly degenerate with the Bragg condition (unlikely).
    C[4, 4] = params.get('monopole_detuning', 0.0)
    
    # --- 4. 2D Cross-Polarization Coupling (C_2D) ---
    # This couples x-dipoles to y-dipoles via (1,1) order terms.
    # Critical for finding the specific ellipse shape for the Dirac cone.
    if params.get('include_C2D', True):
        # We approximate the propagator for the (1,1) mode
        # In full theory, this is integral(Green_11 ...). 
        # Here we use a perturbative constant approximation.
        chi_2D = -prefactor * xi_11 * 0.5 
        
        # Rx <-> Ry (via xi_11)
        C[0, 2] = chi_2D
        C[2, 0] = np.conj(chi_2D)
        
        # Sx <-> Sy
        C[1, 3] = chi_2D
        C[3, 1] = np.conj(chi_2D)
        
        # Rx <-> Sy (via xi_1-1)
        xi_1m1 = params['xi'].get((1, -1), 0j)
        chi_2D_mix = -prefactor * xi_1m1 * 0.5
        C[0, 3] = chi_2D_mix
        C[3, 0] = np.conj(chi_2D_mix)
        
        # Sx <-> Ry
        C[1, 2] = np.conj(chi_2D_mix)
        C[2, 1] = chi_2D_mix

    return C

def solve_cwt_eigenproblem(C_total):
    """
    Solves V = C V for the 5x5 system.
    Returns sorted eigenvalues and eigenvectors.
    """
    eigvals, eigvecs = np.linalg.eig(C_total)
    
    # Sort by real part (Frequency Detuning)
    idx = np.argsort(np.real(eigvals))
    return eigvals[idx], eigvecs[:, idx]

def calculate_field_distributions(eigvecs, a, Nx, Ny, resolution=50):
    """
    Reconstructs the field distribution for the 5-wave basis.
    """
    beta0 = 2 * np.pi / a
    x = np.linspace(-a*Nx/2, a*Nx/2, resolution * Nx)
    y = np.linspace(-a*Ny/2, a*Ny/2, resolution * Ny)
    X, Y = np.meshgrid(x, y)
    
    fields = []
    num_modes = eigvecs.shape[1]
    
    for i in range(num_modes):
        vec = eigvecs[:, i]
        # Unpack 5 components
        Rx, Sx, Ry, Sy = vec[0], vec[1], vec[2], vec[3]
        A0 = vec[4] if len(vec) > 4 else 0.0
        
        # Dipole traveling waves
        term_Rx = Rx * np.exp(-1j * beta0 * X)
        term_Sx = Sx * np.exp(1j * beta0 * X)
        term_Ry = Ry * np.exp(-1j * beta0 * Y)
        term_Sy = Sy * np.exp(1j * beta0 * Y)
        
        # Monopole term (Spatially uniform envelope approx)
        # Note: In reality, this is modulated by u_00(r) which concentrates in veins.
        # But for envelope visualization, it's the "background" DC term.
        term_A0 = A0 * np.ones_like(X) 
        
        field_dist = term_Rx + term_Sx + term_Ry + term_Sy + term_A0
        fields.append(field_dist)
        
    return fields


def add_k_dependence(C_total, kx, ky):
    """
    Adds the wavevector detuning terms to the CWT matrix diagonals.
    This allows calculating the band structure E(k).
    
    Parameters:
    - C_total: The 5x5 Base Hamiltonian (at Gamma)
    - kx, ky: Detuning wavevectors (in units of m^-1)
    
    Returns:
    - C_k: The k-dependent Hamiltonian
    """
    # Create a copy to avoid modifying the base matrix
    C_k = C_total.copy()
    
    # The diagonal terms shift by -/+ k (Linear Dispersion of Basic Waves)
    # Based on the derivative term -i * d/dx -> k
    # Rx (Forward x):  Delta -> Delta - kx
    # Sx (Backward x): Delta -> Delta + kx
    # Ry (Forward y):  Delta -> Delta - ky
    # Sy (Backward y): Delta -> Delta + ky
    # A0 (Monopole):   Flat band assumption (approx. zero group velocity near Gamma)
    
    # Rx (Index 0)
    C_k[0, 0] += -kx
    
    # Sx (Index 1)
    C_k[1, 1] += kx
    
    # Ry (Index 2)
    C_k[2, 2] += -ky
    
    # Sy (Index 3)
    C_k[3, 3] += ky
    
    # A0 (Index 4)
    # The monopole mode is "flat" in the first-order approximation
    # so we do not add a linear k-term.
    
    return C_k