import numpy as np
from scipy.linalg import eig
from scipy.integrate import simpson
import cmath
import matplotlib
matplotlib.use('Agg') # Headless plotting
import matplotlib.pyplot as plt
import fourier # Requires fourier.py in the same directory

class Waveguide1D:
    """
    Solves the 1D vertical waveguide structure to obtain the fundamental TE mode
    Theta_0(z) and the effective refractive index n_eff.
    """
    def __init__(self, layers, wavelength):
        """
        layers: List of dicts [{'n': float, 'thickness': float, 'type': str}, ...]
                Ordered from bottom to top. 'type'='pc' for the photonic crystal layer.
        wavelength: Operational wavelength (vacuum).
        """
        self.layers = layers
        self.wavelength = wavelength
        self.k0 = 2 * np.pi / wavelength
        
        # Locate PC layer
        self.pc_layer_index = next((i for i, L in enumerate(layers) if L.get('type') == 'pc'), None)
        if self.pc_layer_index is None:
            raise ValueError("One layer must be marked as type='pc'")

    def solve_mode(self, dz=1e-3):
        """
        Solves for the fundamental TE mode.
        Returns: n_eff, z_grid, field_profile (Theta_0)
        """
        # 1. Discretize space
        total_thickness = sum(L['thickness'] for L in self.layers)
        if dz > total_thickness / 10:
             dz = total_thickness / 100
             
        z_grid = np.arange(0, total_thickness, dz)
        n_profile = np.zeros_like(z_grid)
        
        current_z = 0
        pc_z_start = 0
        pc_z_end = 0
        
        for i, layer in enumerate(self.layers):
            idx = (z_grid >= current_z) & (z_grid < current_z + layer['thickness'])
            n_profile[idx] = layer['n']
            if i == self.pc_layer_index:
                pc_z_start = current_z
                pc_z_end = current_z + layer['thickness']
            current_z += layer['thickness']
            
        # 2. Finite Difference Mode Solver
        N = len(z_grid)
        diag = -2 * np.ones(N)
        off_diag = np.ones(N - 1)
        
        # Construct operator matrix for d2/dz2 / dz^2
        H = (np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)) / (dz**2)
        
        # Add refractive index term: k0^2 * n(z)^2
        n_term = (self.k0 * n_profile)**2
        
        # We solve (d2/dz2 + k0^2 n^2) E = beta^2 E
        A = H + np.diag(n_term)
        
        vals, vecs = np.linalg.eigh(A)
        
        # Sort eigenvalues (beta^2) descending
        idx_sort = np.argsort(vals)[::-1]
        vals = vals[idx_sort]
        vecs = vecs[:, idx_sort]
        
        # Fundamental mode
        beta_sq = vals[0]
        field = vecs[:, 0]
        
        if beta_sq >= 0:
            n_eff = np.sqrt(beta_sq) / self.k0
        else:
            n_eff = 1j * np.sqrt(-beta_sq) / self.k0
        
        # Normalize field: Integral |Theta_0|^2 dz = 1
        norm = np.sqrt(simpson(np.abs(field)**2, z_grid))
        field = field / norm
        
        self.n_eff = n_eff
        self.z_grid = z_grid
        self.field = field
        self.n_profile = n_profile
        self.pc_bounds = (pc_z_start, pc_z_end)
        
        return n_eff, z_grid, field

class AnalyticFourierProvider:
    """
    Handles dynamic calculation of xi_{m,n} using the fourier module.
    """
    def __init__(self, a, gvecs, eps_bg, shapes):
        self.a = a
        self.gvecs = gvecs
        self.eps_bg = eps_bg
        self.shapes = shapes
        self.cache = {}
        
        # Pre-calculate epsilon Fourier coefficients
        print("Calculating analytic epsilon coefficients...")
        self.eps_ft_coeffs = fourier.get_epsilon_coefficients_analytic(
            gvecs, eps_bg, shapes, a
        )

    def get_xi(self, m, n):
        """Returns xi_{m,n} (cached)."""
        if (m, n) in self.cache:
            return self.cache[(m, n)]
        
        val = fourier.get_xi_mn(m, n, self.a, self.gvecs, self.eps_ft_coeffs)
        self.cache[(m, n)] = val
        return val

class CWTSolver:
    """
    Implements the 3D Coupled-Wave Model based on the Appendix of Liang et al. (2011).
    Calculates coupling matrices dynamically using analytic Fourier transforms.
    """
    def __init__(self, waveguide, gvecs, eps_bg, shapes, truncation_order=3, lattice_constant=1.0):
        self.wg = waveguide
        self.D = truncation_order
        self.a = lattice_constant
        
        # Initialize Fourier Provider
        self.xi_prov = AnalyticFourierProvider(self.a, gvecs, eps_bg, shapes)
        
        # Physical constants
        self.k0 = self.wg.k0
        self.beta0 = 2 * np.pi / self.a
        
        # Extract PC region data
        z_start, z_end = self.wg.pc_bounds
        idx_pc = (self.wg.z_grid >= z_start) & (self.wg.z_grid <= z_end)
        
        if not np.any(idx_pc):
             raise ValueError("PC Layer index mismatch or empty.")
             
        self.z_pc = self.wg.z_grid[idx_pc]
        self.theta0_pc = self.wg.field[idx_pc]
        self.n0_pc = self.wg.n_profile[idx_pc][0]
        
    def _integral_G_rad(self, z, zp):
        """Green's function for radiative waves (Eq A2)."""
        beta_z = self.k0 * self.n0_pc 
        if beta_z == 0: return 0j 
        # Liang Eq A2: exp(i * kz * |z-z'|) / (2i * kz)
        # Here kz = beta_z. 1/(2i) = -i/2 = -1j/2.
        return (-1j / (2 * beta_z)) * np.exp(1j * beta_z * np.abs(z - zp))

    def _integral_G_high(self, m, n, z, zp):
        """Green's function for high-order waves (Eq A6)."""
        val = (m**2 + n**2) * self.beta0**2 - (self.k0 * self.n0_pc)**2
        beta_z_mn = cmath.sqrt(val) 
        # Liang Eq A2 with kz = i * gamma.
        # 1/(2i * i*gamma) = 1/(-2*gamma) = -1/(2*gamma)
        # exp(i * i*gamma * |z|) = exp(-gamma * |z|)
        return (-1.0 / (2 * beta_z_mn)) * np.exp(-beta_z_mn * np.abs(z - zp))

    def calculate_matrices(self):
        """Builds C1D, Crad, and C2D and sums them."""
        print("Building C1D...")
        self.C1D = self._build_C1D()
        print("Building Crad...")
        self.Crad = self._build_Crad()
        print("Building C2D...")
        self.C2D = self._build_C2D()
        self.C = self.C1D + self.Crad + self.C2D
        return self.C

    def _build_C1D(self):
        """Constructs C1D matrix (Eq A16)."""
        Gamma = simpson(np.abs(self.theta0_pc)**2, self.z_pc)
        prefactor = - (self.k0**2) / (2 * self.beta0) * Gamma
        
        # Coupling constants
        k_20 = prefactor * self.xi_prov.get_xi(2, 0)
        k_m20 = prefactor * self.xi_prov.get_xi(-2, 0)
        k_02 = prefactor * self.xi_prov.get_xi(0, 2)
        k_0m2 = prefactor * self.xi_prov.get_xi(0, -2)
        
        C = np.zeros((4,4), dtype=complex)
        C[0, 1] = k_20
        C[1, 0] = k_m20
        C[2, 3] = k_02
        C[3, 2] = k_0m2
        return C

    def _zeta(self, p, q, r, s):
        """Calculates zeta terms for Crad (Eq A19)."""
        Z, ZP = np.meshgrid(self.z_pc, self.z_pc, indexing='ij')
        
        beta_z = self.k0 * self.n0_pc
        if beta_z == 0:
            G_mat = np.zeros_like(Z, dtype=complex)
        else:
            G_mat = (-1j / (2 * beta_z)) * np.exp(-1j * beta_z * np.abs(Z - ZP))
        
        T_col = self.theta0_pc[np.newaxis, :]  
        T_row = np.conj(self.theta0_pc[:, np.newaxis]) 
        
        integrand = G_mat * T_col * T_row
        int_zp = simpson(integrand, self.z_pc, axis=1)
        integral_val = simpson(int_zp, self.z_pc, axis=0)
        
        xi_pq = self.xi_prov.get_xi(p, q)
        xi_rs_conj = self.xi_prov.get_xi(-r, -s)
        
        const_factor = - (self.k0**4) / (2 * self.beta0)
        
        return const_factor * xi_pq * xi_rs_conj * integral_val

    def _build_Crad(self):
        """Constructs Crad matrix (Eq A17)."""
        C = np.zeros((4,4), dtype=complex)
        
        # Block 1 (Rx, Sx) -> Delta Ey
        C[0, 0] = self._zeta(1, 0, 1, 0)
        C[0, 1] = self._zeta(1, 0, -1, 0)
        C[1, 0] = self._zeta(-1, 0, 1, 0)
        C[1, 1] = self._zeta(-1, 0, -1, 0)
        
        # Block 2 (Ry, Sy) -> Delta Ex
        C[2, 2] = self._zeta(0, 1, 0, 1)
        C[2, 3] = self._zeta(0, 1, 0, -1)
        C[3, 2] = self._zeta(0, -1, 0, 1)
        C[3, 3] = self._zeta(0, -1, 0, -1)
        
        return C

    def _mu_nu(self, m, n, r, s):
        """Calculates mu and nu terms (Eq A11, A12)."""
        xi_val = self.xi_prov.get_xi(m-r, n-s)
        if xi_val == 0:
            return 0j, 0j

        # mu: Double integral
        Z, ZP = np.meshgrid(self.z_pc, self.z_pc, indexing='ij')
        
        val = (m**2 + n**2) * self.beta0**2 - (self.k0 * self.n0_pc)**2
        beta_z_mn = cmath.sqrt(val)
        # Liang Eq A7: -1/(2*gamma) * exp(-gamma*|z-z'|)
        # Here beta_z_mn is gamma.
        G_mat = (1.0 / (2 * beta_z_mn)) * np.exp(-beta_z_mn * np.abs(Z - ZP))
        
        T_col = self.theta0_pc[np.newaxis, :]
        T_row = np.conj(self.theta0_pc[:, np.newaxis])
        
        integrand_mu = G_mat * T_col * T_row
        integral_mu = simpson(simpson(integrand_mu, self.z_pc, axis=1), self.z_pc, axis=0)
        
        mu = (self.k0**2) * xi_val * integral_mu
        
        # nu: Single integral
        integral_nu_inner = simpson(np.abs(self.theta0_pc)**2, self.z_pc)
        if self.n0_pc == 0:
             nu = 0j 
        else:
             nu = - (1.0 / self.n0_pc**2) * xi_val * integral_nu_inner
        
        return mu, nu

    def _get_high_order_coeffs(self, m, n, r, s):
        """
        Helper to retrieve Ex_mn and Ey_mn coefficients for a specific source wave (r,s).
        Based on Eq A9 and A10.
        """
        if m**2 + n**2 <= 1: return 0j, 0j
        
        mu_val, nu_val = self._mu_nu(m, n, r, s)
        if mu_val == 0 and nu_val == 0: return 0j, 0j
        
        # Source polarization logic
        if s == 0: # Source X-pol
            term_E_minus = -m * mu_val
            term_E_plus  = n * nu_val
        else:      # Source Y-pol
            term_E_minus = n * mu_val
            term_E_plus  = m * nu_val
        
        M2 = float(m**2 + n**2)
        
        coeff_Ex = (1.0 / M2) * (n * term_E_minus + m * term_E_plus)
        coeff_Ey = (1.0 / M2) * (-m * term_E_minus + n * term_E_plus)
        
        return coeff_Ex, coeff_Ey

    def _chi(self, i_pol, p, q, r, s):
        """Calculates chi elements for C2D (Eq A20)."""
        sum_val = 0j
        
        for m in range(-self.D, self.D + 1):
            for n in range(-self.D, self.D + 1):
                if m**2 + n**2 <= 1: continue
                
                coeff_Ex, coeff_Ey = self._get_high_order_coeffs(m, n, r, s)
                if coeff_Ex == 0 and coeff_Ey == 0: continue
                
                relevant_coeff = coeff_Ex if i_pol == 'x' else coeff_Ey
                
                xi_back = self.xi_prov.get_xi(p - m, q - n)
                sum_val += xi_back * relevant_coeff
                
        return - (self.k0**2) / (2 * self.beta0) * sum_val

    def _build_C2D(self):
        C = np.zeros((4,4), dtype=complex)
        vecs = [(1,0), (-1,0), (0,1), (0,-1)]
        
        for row_idx in range(4):
            p, q = vecs[row_idx]
            coupling_field_pol = 'y' if row_idx < 2 else 'x'
            
            for col_idx in range(4):
                r, s = vecs[col_idx]
                val = self._chi(coupling_field_pol, p, q, r, s)
                C[row_idx, col_idx] = val
        return C

    def solve(self):
        """Solves the eigenvalue problem at Gamma."""
        self.calculate_matrices()
        eigenvalues, eigenvectors = eig(self.C)
        return eigenvalues, eigenvectors
    
    def solve_band_diagram(self, k_points):
        if not hasattr(self, 'C'):
            print("Calculating coupling matrices (Gamma point)...")
            self.calculate_matrices()
            
        all_eigenvalues = []
        for kx, ky in k_points:
            perturbation = np.diag([-kx, kx, -ky, ky])
            C_k = self.C + perturbation
            vals, _ = eig(C_k)
            vals = np.sort(vals)
            all_eigenvalues.append(vals)
            
        return np.array(all_eigenvalues)
        
    def calculate_field_distributions(self, eigenvecs, Nx=100, Ny=100, D_display=None):
        """
        Reconstructs Hz(x,y) including every term required by the formalism.
        
        Args:
            eigenvecs: (4, 4) matrix of eigenvectors [Rx, Sx, Ry, Sy].
            Nx, Ny: Grid dimensions.
            D_display: Truncation order for spatial harmonics (m, n) for visualization. 
                       If None, uses self.D (the truncation used for the matrix calculation).
        """
        D_display = D_display if D_display is not None else self.D
        
        x = np.linspace(-self.a, self.a, Nx)
        y = np.linspace(-self.a, self.a, Ny)
        X, Y = np.meshgrid(x, y)
        fields = []
        num_modes = eigenvecs.shape[1]
        
        # Sources corresponding to vector indices 0..3: [Rx, Sx, Ry, Sy]
        sources = [(1,0), (-1,0), (0,1), (0,-1)]
        
        for i in range(num_modes):
            vec = eigenvecs[:, i] # [Rx, Sx, Ry, Sy]
            Hz_total = np.zeros_like(X, dtype=complex)
            
            # --- Radiative Field Contribution (m=0, n=0) ---
            # The field Delta E is spatially uniform in x,y, meaning its curl (Hz) is zero.
            # Delta H_z is proportional to i * (d Delta E_y / dx - d Delta E_x / dy).
            # If Delta E is uniform, d/dx and d/dy are zero. 
            # We skip this term as it is zero for H_z in the plane, as is standard practice.
            
            # --- Sum over all relevant harmonics (Basic + High Order) ---
            for m in range(-D_display, D_display + 1):
                for n in range(-D_display, D_display + 1):
                    
                    # Skip m=0, n=0 (Radiative term, Hz contribution is zero)
                    if m == 0 and n == 0: continue 
                    
                    Ex_mn = 0j
                    Ey_mn = 0j
                    
                    if m**2 + n**2 <= 1:
                        # Basic Waves: Direct amplitude from eigenvector (at Gamma k=0)
                        
                        idx = None
                        try:
                            idx = sources.index((m, n))
                        except ValueError:
                            # Skip terms like (+/-2, 0), (0, +/-2) which are non-basic low order
                            continue 
                        
                        amp = vec[idx]
                        
                        if m != 0: 
                            Ey_mn = amp
                        else: 
                            Ex_mn = amp
                            
                    else:
                        # High Order Waves: Calculated from basic wave components
                        
                        # Sum contributions from all 4 basic sources (r,s)
                        for idx, (r, s) in enumerate(sources):
                            amp = vec[idx]
                            coeff_Ex, coeff_Ey = self._get_high_order_coeffs(m, n, r, s)
                            Ex_mn += coeff_Ex * amp
                            Ey_mn += coeff_Ey * amp

                    # --- Compute Hz contribution for this (m,n) harmonic ---
                    # Hz_mn ~ beta0 * (m * Ey_mn - n * Ex_mn)
                    term_high = self.beta0 * (m * Ey_mn - n * Ex_mn)
                    phase = np.exp(-1j * self.beta0 * (m * X + n * Y))
                    
                    Hz_total += term_high * phase

            # Rotate global phase to maximize real part contrast
            max_idx = np.argmax(np.abs(Hz_total))
            max_val = Hz_total.flat[max_idx]
            if np.abs(max_val) > 1e-10:
                Hz_total *= np.exp(-1j * np.angle(max_val))

            fields.append(Hz_total)
            
        return fields