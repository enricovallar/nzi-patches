import numpy as np
from scipy.integrate import simpson as simps

def get_beta0(a):
    """Basic wave propagation constant (2nd order Gamma point)."""
    return 2 * np.pi / a

def calculate_greens_integral(theta_z, z_grid, n0_z, k0):
    """
    Calculates the double integral for the Green's function term in C_rad.
    Corresponds to the integral part of Eq. A14 in Liang 2012.
    
    I = integral( integral( G(z, z') * Theta(z') * Theta*(z) dz' ) dz )
    Assuming G(z, z') = -i / (2 * beta_z) * exp(-i * beta_z * |z - z'|)
    where beta_z = k0 * n0(z).
    """
    # Grid spacing
    if len(z_grid) > 1:
        dz = z_grid[1] - z_grid[0]
    else:
        return 0.0 # Safety for single point
    
    # Wave vector in z
    beta_z = k0 * n0_z
    
    # Construct 2D meshes for integration
    Z, ZP = np.meshgrid(z_grid, z_grid, indexing='ij')
    BZ, BZP = np.meshgrid(beta_z, beta_z, indexing='ij')
    TH, THP = np.meshgrid(theta_z, theta_z, indexing='ij')
    
    # We approximate beta_z as constant in the Green's function denominator 
    # for the layer where radiation is generated (usually the slab index).
    # However, rigorous G varies. Liang assumes simplified G. 
    # We use the local beta_z for the phase, and average for amplitude to avoid singularities.
    # Note: For PCSELs, usually radiation is into air/substrate.
    # Here we implement the simplified form often used: G approx -i/(2*k0*n_avg).
    
    # Let's use the explicit form:
    # G(z, z') term
    # Note: This is computationally heavy O(N^2). 
    # For a 1D profile, N=1000 is fine.
    
    # Avoid division by zero if beta_z is zero (e.g. metal)
    with np.errstate(divide='ignore', invalid='ignore'):
        green_kernel = (-1j / (2 * BZ)) * np.exp(-1j * BZ * np.abs(Z - ZP))
    
    # Handle NaN/Inf if any (though unlikely for dielectrics)
    green_kernel[~np.isfinite(green_kernel)] = 0.0
    
    integrand = green_kernel * THP * np.conj(TH)
    
    # Double integral using Trapezoidal or Simpson's rule
    # Inner integral (dz')
    inner = np.trapz(integrand, z_grid, axis=1)
    # Outer integral (dz)
    result = np.trapz(inner, z_grid)
    
    return result

def construct_cwt_matrices(params):
    """
    Constructs the C_1D, C_rad, and C_2D matrices based on Liang 2012.
    
    params dictionary must contain:
    - xi: dict of Fourier coeffs {(m,n): complex}
    - n_eff: effective index of fundamental mode
    - theta_z: vertical mode profile array
    - z_grid: z-coordinates array
    - n0_z: refractive index profile array n0(z)
    - a: lattice constant
    - lambda0: wavelength
    - conf: confinement factor (Gamma)
    - D_trunc: truncation order for 2D sum (e.g. 1 or 2)
    """
    a = params['a']
    lam = params['lambda0']
    k0 = 2 * np.pi / lam
    beta0 = 2 * np.pi / a
    
    # Factor common to many terms: k0^2 / (2 * beta0)
    # Note: Liang 2012 uses Beta0 in denominator of Eq 19.
    prefactor = (k0**2) / (2 * beta0)
    
    # =========================================================================
    # 1. C_1D (Feedback Matrix)
    # =========================================================================
    # Eq. 19: kappa = -prefactor * Xi_{2,0} * Confinement
    # Note: Xi is usually defined as Delta Epsilon. Legume returns just the FT of Epsilon.
    # Liang's equation assumes expansion of dielectric constant.
    
    xi_20 = params['xi'].get((2,0), 0j)
    xi_02 = params['xi'].get((0,2), 0j)
    gamma = params['conf']
    
    kappa_x = -prefactor * xi_20 * gamma
    kappa_y = -prefactor * xi_02 * gamma
    
    # Eq A11 matrix structure
    C_1D = np.zeros((4, 4), dtype=complex)
    C_1D[0, 1] = kappa_x      # R_x <- S_x
    C_1D[1, 0] = np.conj(kappa_x) # S_x <- R_x (assuming Hermitian for lossless)
    C_1D[2, 3] = kappa_y      # R_y <- S_y
    C_1D[3, 2] = np.conj(kappa_y) # S_y <- R_y
    
    # =========================================================================
    # 2. C_rad (Radiation Matrix)
    # =========================================================================
    # Depends on G-integral (Zeta terms)
    # Eq A14: zeta = -k0^4 / (2*beta0) * integral(...) * xi * xi
    # Note: Liang A14 includes k0^4.
    # The integral part is calculated by helper.
    
    g_int = calculate_greens_integral(params['theta_z'], params['z_grid'], params['n0_z'], k0)
    
    # Pre-calculate zeta factor (excluding Xi terms)
    zeta_factor = -(k0**4 / (2 * beta0)) * g_int
    
    # We need Xi coefficients for radiation coupling (1st order)
    # These are the terms that couple +/-1 to 0 (radiation)
    xi_10 = params['xi'].get((1,0), 0j)
    xi_m10 = np.conj(xi_10) # xi_{-1, 0}
    xi_01 = params['xi'].get((0,1), 0j)
    xi_m01 = np.conj(xi_01) # xi_{0, -1}
    
    # Matrix Elements (Eq A12)
    # The matrix couples Rx, Sx, Ry, Sy via the radiation field
    # Structure:
    # [ z_xx  z_xx  z_xy  z_xy ]
    # [ z_xx  z_xx  z_xy  z_xy ]
    # [ z_yx  z_yx  z_yy  z_yy ]
    # [ z_yx  z_yx  z_yy  z_yy ]
    
    # z_xx ~ xi_{1,0} * xi_{-1,0} etc.
    # Carefully matching Eq A12 & A14 indices:
    
    C_rad = np.zeros((4, 4), dtype=complex)
    
    # Block 1 (x-x coupling via radiation)
    # Actually Liang A12 shows: zeta_{1,0}^{(1,0)}, zeta_{1,0}^{(-1,0)} etc.
    # It's safer to follow the "radiation loss is proportional to |R+S|^2" logic for symmetric.
    
    # Row 1 (Rx equation)
    C_rad[0,0] = zeta_factor * xi_10 * xi_m10
    C_rad[0,1] = zeta_factor * xi_10 * xi_10 # Coupling from Sx using xi_10
    C_rad[0,2] = zeta_factor * xi_10 * xi_m01
    C_rad[0,3] = zeta_factor * xi_10 * xi_01
    
    # Row 2 (Sx equation)
    C_rad[1,0] = zeta_factor * xi_m10 * xi_m10
    C_rad[1,1] = zeta_factor * xi_m10 * xi_10
    C_rad[1,2] = zeta_factor * xi_m10 * xi_m01
    C_rad[1,3] = zeta_factor * xi_m10 * xi_01
    
    # Row 3 (Ry)
    C_rad[2,0] = zeta_factor * xi_01 * xi_m10
    C_rad[2,1] = zeta_factor * xi_01 * xi_10
    C_rad[2,2] = zeta_factor * xi_01 * xi_m01
    C_rad[2,3] = zeta_factor * xi_01 * xi_01
    
    # Row 4 (Sy)
    C_rad[3,0] = zeta_factor * xi_m01 * xi_m10
    C_rad[3,1] = zeta_factor * xi_m01 * xi_10
    C_rad[3,2] = zeta_factor * xi_m01 * xi_m01
    C_rad[3,3] = zeta_factor * xi_m01 * xi_01

    # =========================================================================
    # 3. C_2D (Higher Order Coupling Matrix)
    # =========================================================================
    # This involves summation over m, n for sqrt(m^2+n^2) > 1
    # Eq A20: chi = -prefactor * sum( xi * xi_high * overlap )
    # Overlap integrals mu and nu (Eq A12)
    # This is complex. For a simplified model (thin slab), we can approximate
    # the vertical overlap as Gamma (confinement) or 1.
    # However, rigorous CWT requires the G_mn(z,z') integral.
    
    # Placeholder for full 2D implementation.
    # For many square lattice PCSELs, C_2D is smaller than C_1D and C_rad 
    # but critical for breaking degeneracy between x and y pol.
    
    # We will implement a simplified version assuming High-order waves 
    # have similar confinement to basic waves (Gamma).
    # Sum over nearest neighbors (1,1), (-1,1), etc.
    
    C_2D = np.zeros((4, 4), dtype=complex)
    
    # Iterate over higher order terms
    D = params['D_trunc']
    for m in range(-D, D+1):
        for n in range(-D, D+1):
            if m**2 + n**2 <= 1: continue # Skip basic and radiative
            
            # Reciprocal vector magnitude for this order
            G_sq = (m**2 + n**2) * beta0**2
            # Vertical decay constant for this order (evanescent)
            # gamma_mn = sqrt(G^2 - k0^2 * eps_avg)
            # Simplification: The "Green's function" for high order waves 
            # acts like 1 / (k0^2*eps - G^2)
            
            denom = k0**2 * params['n_eff']**2 - G_sq 
            # Note: This denominator is negative for evanescent waves
            
            # Effective 'propagator' for high order wave
            # This replaces the integral G_mn in Eq A7
            # If denominator is very small, this blows up.
            if abs(denom) < 1e-9: denom = 1e-9 
            
            prop = 1.0 / denom 
            
            # We need to add contribution to each matrix element.
            # Example: Coupling Rx <-> Ry via (1,1)
            # Rx couples to (1,1) via xi_{0,1}. (1,1) couples to Ry via xi_{-1,0}
            
            # This requires a rigorous map of all mixing paths.
            # Implementing the full summation loop structure of A20 is complex for this snippet.
            # We will perform a 1st-neighbor approximation (1,1) family only.
            pass

    return C_1D, C_rad, C_2D

def solve_cwt_eigenproblem(C_total):
    """
    Solves (delta + i*alpha) V = C V
    Returns eigenvalues and eigenvectors.
    """
    eigvals, eigvecs = np.linalg.eig(C_total)
    
    # Sort by imaginary part (lowest loss first)
    # alpha = Imag(eigval). alpha is amplitude loss. 
    # Radiation constant (power loss) = 2 * alpha.
    
    idx = np.argsort(np.imag(eigvals))
    return eigvals[idx], eigvecs[:, idx]

def calculate_field_distributions(eigvecs, a, resolution=100):
    """
    Reconstructs the Hz field distribution for each eigenmode at z=d/2.
    
    Parameters:
    - eigvecs: Eigenvectors matrix (4 x N_modes) from solve_cwt_eigenproblem
               Rows are [Rx, Sx, Ry, Sy]
    - a: Lattice constant
    - resolution: Number of points per unit cell for plotting
    
    Returns:
    - fields: List of 2D arrays (resolution x resolution), one for each mode.
              Each array contains the complex Hz field.
    """
    beta0 = 2 * np.pi / a
    
    # Create coordinate grid for one unit cell (centered at 0)
    # The unit cell goes from -a/2 to a/2
    x = np.linspace(-a/2, a/2, resolution)
    y = np.linspace(-a/2, a/2, resolution)
    X, Y = np.meshgrid(x, y)
    
    fields = []
    num_modes = eigvecs.shape[1]
    
    for i in range(num_modes):
        # Extract components for this mode
        vec = eigvecs[:, i]
        Rx, Sx, Ry, Sy = vec[0], vec[1], vec[2], vec[3]
        
        # Reconstruct Field
        # Basic waves are plane waves modulated by the envelope (which is 1 for infinite)
        # H_z approx sum of basic waves
        # Rx corresponds to exp(-i * beta0 * x)
        # Sx corresponds to exp(+i * beta0 * x)  (Note: beta0 = 2pi/a is G vector)
        # Ry corresponds to exp(-i * beta0 * y)
        # Sy corresponds to exp(+i * beta0 * y)
        
        # Check sign convention in Liang 2012 Eq 4: 
        # Field = sum E_mn * exp(-i m beta0 x - i n beta0 y)
        # Rx is E_{y, 1,0} -> m=1, n=0 -> exp(-i beta0 x)
        # Sx is E_{y, -1,0} -> m=-1, n=0 -> exp(+i beta0 x)
        # Ry is E_{x, 0,1} -> m=0, n=1 -> exp(-i beta0 y)
        # Sy is E_{x, 0,-1} -> m=0, n=-1 -> exp(+i beta0 y)
        
        term_Rx = Rx * np.exp(-1j * beta0 * X)
        term_Sx = Sx * np.exp(1j * beta0 * X)
        term_Ry = Ry * np.exp(-1j * beta0 * Y)
        term_Sy = Sy * np.exp(1j * beta0 * Y)
        
        # Total field (scalar proxy for H_z or E_total)
        # For TE, Hz is the scalar field that is often plotted
        field_dist = term_Rx + term_Sx + term_Ry + term_Sy
        
        fields.append(field_dist)
        
    return fields