import numpy as np
import matplotlib.pyplot as plt
import fourier
import mode_solver
import cwt_solver

# ==========================================
# 1. SETUP GEOMETRY
# ==========================================
a = 1e-6           # Lattice constant
d = 250e-9         # Slab thickness
r1 = 0.21 * a      # Hole 1 radius
r2 = 0.20 * a     # Hole 2 radius (Optimized)
lambda0 = 1.55e-6  # Target wavelength

eps_InP = 9.46     # Permittivity
eps_air = 1.0

# Calculate average permittivity
eps_avg = (eps_InP * (1 - np.pi*(r1**2 + r2**2)/a**2) + 
           eps_air * (np.pi*(r1**2 + r2**2)/a**2))

# ==========================================
# 2. INITIALIZE CWT MATRICES
# ==========================================
# --- Fourier Coefficients ---
shapes = [{'eps': eps_air, 'r': r1, 'center': (0.5*a, 0)},
          {'eps': eps_air, 'r': r2, 'center': (0, 0.5*a)}]

gmax = 5 * 2 * np.pi / a
gvecs = fourier.get_g_vectors(gmax, a)
eps_ft = fourier.get_epsilon_coefficients_analytic(gvecs, eps_InP, shapes, a)

xi_coeffs = {}
orders = [(0,0), (1,0), (0,1), (2,0), (0,2), (1,1), (1,-1)]
for m, n in orders:
    val = fourier.get_xi_mn(m, n, a, gvecs, eps_ft)
    xi_coeffs[(m, n)] = val
    xi_coeffs[(-m, -n)] = np.conj(val)
    if m != 0 and n != 0:
        xi_coeffs[(m, -n)] = fourier.get_xi_mn(m, -n, a, gvecs, eps_ft)
        xi_coeffs[(-m, n)] = np.conj(xi_coeffs[(m, -n)])

# --- Vertical Mode ---
modes, _ = mode_solver.solve_slab_modes(2e-6, d, 2e-6, eps_air, eps_avg, eps_air, lambda0)
te0 = modes[0]

# --- Green's Function Grid ---
z_grid = te0['z_grid']
n0_z = np.ones_like(z_grid) * np.sqrt(eps_air)
n0_z[(z_grid >= -d/2) & (z_grid <= d/2)] = np.sqrt(eps_avg)

# --- Construct Base Hamiltonian (at k=0) ---
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
    'monopole_detuning': 0.0 
}

C_base = cwt_solver.construct_cwt_matrices(cwt_params)

# ==========================================
# 3. COMPUTE BAND STRUCTURE
# ==========================================
num_points = 51
k_max = 0.1 * (2 * np.pi / a) 
kx_range = np.linspace(-k_max, k_max, num_points)

bands_real = []

print("Computing Band Structure...")
for kx in kx_range:
    # Use the function from your updated cwt_solver
    C_k = cwt_solver.add_k_dependence(C_base, kx, 0.0)
    
    eigvals, _ = cwt_solver.solve_cwt_eigenproblem(C_k)
    bands_real.append(np.real(eigvals))

bands_real = np.array(bands_real)

# ==========================================
# 4. PLOT NORMALIZED FREQUENCY
# ==========================================
# Convert Detuning -> Norm Freq
n_eff = te0['neff']
bands_norm = (1.0 / n_eff) * (1 + bands_real * a / (2 * np.pi))
k_norm = kx_range / (2 * np.pi / a)

plt.figure(figsize=(10, 7))
labels = ["Mode 1", "Mode 2", "Mode 3", "Mode 4", "Mode 5"]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

for i in range(5):
    plt.plot(k_norm, bands_norm[:, i], '.-', label=labels[i], color=colors[i], markersize=4)

plt.xlabel(r'Wavevector $k_x a / 2\pi$')
plt.ylabel(r'Normalized Frequency $\omega a / 2\pi c$')
plt.title(f'NZI Band Structure near $\Gamma$ (a={a*1e9:.0f}nm)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(min(k_norm), max(k_norm))

plt.savefig('nzi_band_structure_norm.png')
print(f"Plot saved to nzi_band_structure_norm.png")
plt.show()
