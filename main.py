import numpy as np
import matplotlib
matplotlib.use('Agg')
import fourier
import matplotlib.pyplot as plt
import mode_solver
import cwt_solver


# ==========================================
# 1. DEFINE GEOMETRY (Same as before)
# ==========================================
a = 1e-6     
d = 0.25*a     
eps_InP = 3.17**2 
eps_air = 1.0
r1 = 0.23 * a    
r2 = 0.24 * a 
lambda0 = 1.55e-6  

eps_avg = (eps_InP * (1 - np.pi * (r1**2 + r2**2) / a**2) + eps_air * (np.pi * (r1**2 + r2**2) / a**2))
print(f"Average Permittivity: {eps_avg:.4f}")

# Initialize Lattice
# lattice = legume.Lattice([a, 0], [0, a])
# phc = legume.PhotCryst(lattice, eps_l=eps_air, eps_u=eps_air)

# ==========================================
# EXTRACT ANALYTIC FOURIER COEFFICIENTS
# ==========================================

# Define shapes (circles)
# Center coordinates in real space
shapes = [
    {'eps': eps_air, 'r': r1, 'center': (0.5 * a, 0)},
    {'eps': eps_air, 'r': r2, 'center': (0, 0.5 * a)}
]

# 1. Get the list of Reciprocal Vectors (G-vectors)
gmax = 5 * 2 * np.pi / a
gvecs = fourier.get_g_vectors(gmax, a)

# 2. Compute the FT of the slab layer
eps_ft_coeffs = fourier.get_epsilon_coefficients_analytic(gvecs, eps_InP, shapes, a)

xi_00 = fourier.get_xi_mn(0, 0, a, gvecs, eps_ft_coeffs)
xi_10 = fourier.get_xi_mn(1, 0, a, gvecs, eps_ft_coeffs)
xi_01 = fourier.get_xi_mn(0, 1, a, gvecs, eps_ft_coeffs)
xi_20 = fourier.get_xi_mn(2, 0, a, gvecs, eps_ft_coeffs)
xi_02 = fourier.get_xi_mn(0, 2, a, gvecs, eps_ft_coeffs)

print(f"Xi_0,0 (Avg Eps): {np.real(xi_00):.4f}")
print(f"Xi_1,0 (Coupling): {np.abs(xi_10):.4f}") # Should now be non-zero
print(f"Xi_0,1 (Coupling): {np.abs(xi_01):.4f}")


# ==========================================
# 3. EXTRACT VERTICAL MODE (Theta_0)
# ==========================================
# Now we can extract the vertical mode profiles using the average permittivity term
modes, num_modes = mode_solver.solve_slab_modes(
    d_top=2e-6, 
    d_mid=0.25e-6,
    d_bot=2e-6, 
    eps_top=eps_air, 
    eps_mid=eps_avg, 
    eps_bot=eps_air, 
    lambda0=lambda0
)

print(f"Found {num_modes} guided mode(s) using average permittivity.")
for m in modes:
    print(f"Mode TE{m['mode_index']}: neff = {m['neff']:.4f}, Confinement = {m['confinement']:.4f}")


# ==========================================
# 4. PREPARE DATA FOR CWT SOLVER
# ==========================================
if num_modes > 0:
    te0 = modes[0]  # Use fundamental TE mode
    
    # 1. Package Fourier Coefficients
    # The solver expects a dictionary keys (m, n) -> complex value
    # We assume symmetric holes (real structure) => xi_{-m} = conj(xi_m)
    xi_coeffs = {
        (0, 0): xi_00,
        (1, 0): xi_10,
        (0, 1): xi_01,
        (2, 0): xi_20,
        (0, 2): xi_02,
        # Add conjugates for negative orders required by the solver
        (-1, 0): np.conj(xi_10),
        (0, -1): np.conj(xi_01),
        (-2, 0): np.conj(xi_20),
        (0, -2): np.conj(xi_02)
    }

    # 2. Construct Refractive Index Profile n0(z)
    # Required for the Green's function integral in C_rad
    n_clad_val = np.sqrt(eps_air)
    n_core_val = np.sqrt(eps_avg)
    
    z_grid = te0['z_grid']
    n0_z = np.ones_like(z_grid) * n_clad_val
    
    # Apply core index to slab region
    # Note: The slab in mode_solver was centered at 0 with thickness d
    slab_mask = (z_grid >= -d/2) & (z_grid <= d/2)
    n0_z[slab_mask] = n_core_val

    # ==========================================
    # 5. SOLVE CWT EIGENPROBLEM
    # ==========================================
    print("\n--- Solving CWT Band Edge Modes ---")

    cwt_params = {
        'xi': xi_coeffs,
        'n_eff': te0['neff'],
        'theta_z': te0['theta'],
        'z_grid': z_grid,
        'n0_z': n0_z,
        'a': a,
        'lambda0': lambda0,
        'conf': te0['confinement'],
        'D_trunc': 2  # Truncation for higher-order waves
    }

    # Construct Matrices
    C_1D, C_rad, C_2D = cwt_solver.construct_cwt_matrices(cwt_params)
    C_total = C_1D + C_rad + C_2D

    # Solve Eigenvalues
    eigvals, eigvecs = cwt_solver.solve_cwt_eigenproblem(C_total)

    # ==========================================
    # 6. DISPLAY RESULTS
    # ==========================================
    print(f"\nResults for lattice constant a = {a*1e9:.0f} nm:")
    
    c_light = 3e8
    for i, val in enumerate(eigvals):
        delta = np.real(val)  # Frequency detuning (m^-1)
        alpha = np.imag(val)  # Radiation loss (m^-1)
        
        # Convert delta to Frequency Shift (Hz)
        # delta = n_eff * (omega - omega_Bragg) / c
        d_freq = delta * c_light / (2 * np.pi * te0['neff'])
        
        # Analyze Vector Character (Rx, Sx, Ry, Sy)
        vec = eigvecs[:, i]
        mag = np.abs(vec)
        # Normalize magnitude
        mag = mag / np.max(mag)
        
        mode_type = "Unknown"
        if mag[0] > 0.9 and mag[1] > 0.9: mode_type = "Mode A/B (Antisymmetric/Symmetric)"
        
        print(f"Mode {i+1}:")
        print(f"  Eigenvalue: {val:.2e}")
        print(f"  Loss (alpha): {alpha/100:.2f} cm^-1")
        print(f"  Detuning:     {delta:.2f} m^-1 ({d_freq/1e9:.2f} GHz)")
        print(f"  Vector Comp:  [Rx:{vec[0]:.2f}, Sx:{vec[1]:.2f}, Ry:{vec[2]:.2f}, Sy:{vec[3]:.2f}]")

    # Plot Eigenvalues
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(eigvals), np.imag(eigvals), c='red', marker='x', s=100)
    plt.xlabel(r'Detuning $\delta$ ($m^{-1}$)')
    plt.ylabel(r'Radiation Loss $\alpha$ ($m^{-1}$)')
    plt.title('CWT Band Edge Eigenvalues')
    plt.grid(True, alpha=0.3)
    plt.savefig('cwt_band_structure.png')
    print("\nSaved plot: cwt_band_structure.png")

    # ==========================================
    # 7. PLOT FIELD DISTRIBUTIONS
    # ==========================================
    fields = cwt_solver.calculate_field_distributions(eigvecs, None,  a, 2,2)
    
    for i, field in enumerate(fields):
        # Normalize field by max amplitude
        max_val = np.max(np.abs(field))
        if max_val > 0:
            field = field / max_val
            
        # Rotate phase to maximize real part at the point of maximum intensity
        # This handles the arbitrary phase of eigenvectors
        idx_max = np.unravel_index(np.argmax(np.abs(field)), field.shape)
        phase_factor = np.exp(-1j * np.angle(field[idx_max]))
        field_rotated = field * phase_factor
        
        plt.figure(figsize=(10, 4))
        
        # Plot Real part of Hz (after phase rotation)
        plt.subplot(1, 2, 1)
        extent = [-a/2*1e6, a/2*1e6, -a/2*1e6, a/2*1e6]
        plt.imshow(np.real(field_rotated), extent=extent, cmap='RdBu', origin='lower')
        plt.colorbar(label='Re(Hz)')
        plt.title(f'Mode {i+1} Re(Hz)')
        plt.xlabel('x (um)')
        plt.ylabel('y (um)')
        
        # Plot Intensity |Hz|^2
        plt.subplot(1, 2, 2)
        plt.imshow(np.abs(field)**2, extent=extent, cmap='inferno', origin='lower')
        plt.colorbar(label='|Hz|^2')
        plt.title(f'Mode {i+1} Intensity')
        plt.xlabel('x (um)')
        plt.ylabel('y (um)')
        
        plt.tight_layout()
        plt.savefig(f'mode_{i+1}_hz.png')
        print(f"Saved plot: mode_{i+1}_hz.png")
        plt.close()

else:
    print("Error: No vertical guided modes found. Cannot run CWT.")