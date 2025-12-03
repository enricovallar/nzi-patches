import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fourier
from cwt_solver import Waveguide1D, CWTSolver

# ==========================================
# 1. SETUP GEOMETRY
# ==========================================
a = 1e-6           # Lattice constant
d = 250e-9         # Slab thickness
r1 = 0.24 * a      # Hole 1 radius
r2 = 0.21 * a      # Hole 2 radius (Optimized)
lambda0 = 1.55e-6  # Target wavelength

eps_InP = 9.46     # Permittivity
eps_air = 1.0

# Calculate average permittivity
f1 = np.pi * r1**2 / a**2
f2 = np.pi * r2**2 / a**2
eps_avg = eps_InP * (1 - (f1 + f2)) + eps_air * (f1 + f2)

# ==========================================
# 2. INITIALIZE SOLVER
# ==========================================

# Define shapes (Same as original plot_band_structure.py)
shapes = [
    {'eps': eps_air, 'r': r1, 'center': (0.5 * a, 0)},
    {'eps': eps_air, 'r': r2, 'center': (0, 0.5 * a)}
]

# G-vectors
gmax = 5 * 2 * np.pi / a
gvecs = fourier.get_g_vectors(gmax, a)

# Solve 1D Vertical Mode
layers = [
    {'n': np.sqrt(eps_air), 'thickness': 2.0e-6, 'type': 'clad_bot'},
    {'n': np.sqrt(eps_avg), 'thickness': d,      'type': 'pc'},
    {'n': np.sqrt(eps_air), 'thickness': 2.0e-6, 'type': 'clad_top'}
]
wg = Waveguide1D(layers, lambda0)
n_eff, _, _ = wg.solve_mode()

print(f"Effective Index: {n_eff:.4f}")

# Initialize CWT Solver
# Note: eps_bg is eps_InP
solver = CWTSolver(wg, gvecs, eps_InP, shapes, truncation_order=3, lattice_constant=a)

# ==========================================
# 3. COMPUTE BAND STRUCTURE
# ==========================================
num_points = 51
k_max = 0.1 * (2 * np.pi / a) 
kx_range = np.linspace(-k_max, k_max, num_points)
k_points = [(kx, 0) for kx in kx_range]

print("Computing Band Structure...")
eigvals = solver.solve_band_diagram(k_points)
bands_real = np.real(eigvals)

# ==========================================
# 4. PLOT NORMALIZED FREQUENCY
# ==========================================
# Convert Detuning -> Norm Freq
# Formula: a/lambda = 1/n_eff + delta * a / (2 * pi * n_eff)
# The base frequency is the Bragg frequency where n_eff * k0 = 2*pi/a
# => a/lambda_Bragg = 1/n_eff
bands_norm = bands_real * a / (2 * np.pi * np.real(n_eff)) + 1.0 / np.real(n_eff)
k_norm = kx_range / (2 * np.pi / a)

plt.figure(figsize=(10, 7))
num_bands = bands_norm.shape[1]
labels = [f"Band {i+1}" for i in range(num_bands)]
colors = plt.cm.viridis(np.linspace(0, 1, num_bands))

for i in range(num_bands):
    plt.plot(k_norm, bands_norm[:, i], '.-', label=labels[i], color=colors[i], markersize=4)

plt.xlabel(r'Wavevector $k_x a / 2\pi$')
plt.ylabel(r'Normalized Frequency $a / \lambda$')
plt.title(f'NZI Band Structure near $\Gamma$ (a={a*1e9:.0f}nm)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(min(k_norm), max(k_norm))

plt.savefig('nzi_band_structure_norm.png')
print(f"Plot saved to nzi_band_structure_norm.png")

# ==========================================
# 5. PLOT FIELDS AT GAMMA
# ==========================================
print("Calculating fields at Gamma...")
vals_gamma, vecs_gamma = solver.solve()

# Sort to match band structure order (np.sort behavior for complex is Real then Imag)
idx_sort = np.argsort(vals_gamma)
vals_gamma = vals_gamma[idx_sort]
vecs_gamma = vecs_gamma[:, idx_sort]

fields = solver.calculate_field_distributions(vecs_gamma)

for i, field in enumerate(fields):
    plt.figure(figsize=(10, 4))
    
    # Plot Real part of Hz
    plt.subplot(1, 2, 1)
    extent = [-a/2*1e6, a/2*1e6, -a/2*1e6, a/2*1e6] # Convert to microns
    plt.imshow(np.real(field), extent=extent, cmap='RdBu', origin='lower')
    plt.colorbar(label='Re(Hz)')
    plt.title(f'Band {i+1} Re(Hz) @ Gamma')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    
    # Plot Intensity |Hz|^2
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(field)**2, extent=extent, cmap='inferno', origin='lower')
    plt.colorbar(label='|Hz|^2')
    plt.title(f'Band {i+1} Intensity @ Gamma')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    
    plt.tight_layout()
    plt.savefig(f'field_band_{i+1}_gamma.png')
    print(f"Saved field plot: field_band_{i+1}_gamma.png")
    plt.close()
