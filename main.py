import numpy as np
import matplotlib
matplotlib.use('Agg')
import fourier
import matplotlib.pyplot as plt
import mode_solver
import cwt_solver

# ==========================================
# 1. DEFINE GEOMETRY (NZI Candidate)
# ==========================================
a = 1e-6     
d = 0.25*a     
eps_InP = 3.17**2 
eps_air = 1.0
r1 = 0.23 * a    
r2 = 0.24 * a 
lambda0 = 1.55e-6  

# Calculate average epsilon
eps_avg = (eps_InP * (1 - np.pi * (r1**2 + r2**2) / a**2) + eps_air * (np.pi * (r1**2 + r2**2) / a**2))

# ==========================================
# 2. EXTRACT FOURIER COEFFICIENTS
# ==========================================
shapes = [
    {'eps': eps_air, 'r': r1, 'center': (0.5 * a, 0)},
    {'eps': eps_air, 'r': r2, 'center': (0, 0.5 * a)}
]

gmax = 5 * 2 * np.pi / a
gvecs = fourier.get_g_vectors(gmax, a)
eps_ft_coeffs = fourier.get_epsilon_coefficients_analytic(gvecs, eps_InP, shapes, a)

# Extract specific coefficients needed for 5-wave basis
xi_coeffs = {}
orders = [(0,0), (1,0), (0,1), (2,0), (0,2), (1,1), (1,-1)]

for m, n in orders:
    val = fourier.get_xi_mn(m, n, a, gvecs, eps_ft_coeffs)
    xi_coeffs[(m, n)] = val
    xi_coeffs[(-m, -n)] = np.conj(val) # Hermitian symmetry
    # Also need mixed conjugates for 1,-1 terms if not symmetric
    if m != 0 and n != 0:
        xi_coeffs[(m, -n)] = fourier.get_xi_mn(m, -n, a, gvecs, eps_ft_coeffs)
        xi_coeffs[(-m, n)] = np.conj(xi_coeffs[(m, -n)])

print(f"Xi_1,1 (Cross-Coupling): {np.abs(xi_coeffs.get((1,1), 0)):.4f}")

# ==========================================
# 3. VERTICAL MODE
# ==========================================
modes, num_modes = mode_solver.solve_slab_modes(
    d_top=2e-6, d_mid=d, d_bot=2e-6, 
    eps_top=eps_air, eps_mid=eps_avg, eps_bot=eps_air, 
    lambda0=lambda0
)

if num_modes > 0:
    te0 = modes[0]
    
    # Grid for Green's function
    z_grid = te0['z_grid']
    n0_z = np.ones_like(z_grid) * np.sqrt(eps_air)
    slab_mask = (z_grid >= -d/2) & (z_grid <= d/2)
    n0_z[slab_mask] = np.sqrt(eps_avg)

    # ==========================================
    # 4. SOLVE CWT (5-Wave Basis)
    # ==========================================
    print("\n--- Solving NZI CWT Modes ---")
    
    cwt_params = {
        'xi': xi_coeffs,
        'n_eff': te0['neff'],
        'theta_z': te0['theta'],
        'z_grid': z_grid,
        'n0_z': n0_z,
        'a': a,
        'lambda0': lambda0,
        'conf': te0['confinement'],
        'D_trunc': 2,
        'include_C2D': True,
        # Monopole Detuning:
        # 0.0 means the monopole is perfectly degenerate with the Bragg frequency.
        # In a real sweep, you would vary this from -2e4 to +2e4 to find the crossing.
        'monopole_detuning': 0.0 
    }

    C_total = cwt_solver.construct_cwt_matrices(cwt_params)
    eigvals, eigvecs = cwt_solver.solve_cwt_eigenproblem(C_total)

    # ==========================================
    # 5. RESULTS & PLOTTING
    # ==========================================
    print(f"\nResults for 5-Wave Basis (a={a*1e9:.0f}nm):")
    
    for i, val in enumerate(eigvals):
        delta = np.real(val)
        vec = eigvecs[:, i]
        mag = np.abs(vec)
        mag = mag / np.max(mag) # Normalize
        
        # Identify mode character
        char_str = ""
        if mag[4] > 0.8: char_str = "[Monopole-Like]"
        elif mag[0] > 0.5: char_str = "[Dipole-Like]"
        
        print(f"Mode {i+1} {char_str}:")
        print(f"  Detuning (Real): {delta:.2e}")
        print(f"  Loss (Imag):     {np.imag(val):.2e}")
        print(f"  Basis [Rx, Sx, Ry, Sy, A0]: {np.round(mag, 2)}")

    # Plot Fields
    fields = cwt_solver.calculate_field_distributions(eigvecs, a, Nx=2, Ny=2)
    
    for i, field in enumerate(fields):
        plt.figure(figsize=(5, 4))
        extent = [-a, a, -a, a] # 2 unit cells
        plt.imshow(np.real(field), extent=extent, cmap='RdBu', origin='lower')
        plt.title(f'Mode {i+1} Re(Hz)')
        plt.colorbar()
        plt.savefig(f'mode_{i+1}_Hz.png')
        plt.close()
        print(f"Saved mode_{i+1}_Hz.png")

else:
    print("No guided modes found.")