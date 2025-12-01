import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import mode_solver

# ==========================================
# EXAMPLE USAGE
# ==========================================

# Parameters
a = 1000e-9
d_core = 250e-9
# Using "semi-infinite" padding for calculation, but visualizing specific range
d_clad_padding = 1000e-9 

lambda0 = 1550e-9
r1 = 0.23 * a
r2 = 0.24 * a

# Material Permittivities
eps_InP = 3.17**2
eps_air = 1.0

# Calculate Average Permittivity for CWT "Basic Wave" definition
fill_factor = (np.pi * (r1**2 + r2**2)) / a**2
eps_avg_core = eps_InP * (1 - fill_factor) + eps_air * fill_factor
print(f"Average Permittivity of Core: {eps_avg_core:.4f}")

# Run Solver
modes, num_modes = mode_solver.solve_slab_modes(
    d_top=d_clad_padding, 
    d_mid=d_core, 
    d_bot=d_clad_padding, 
    eps_top=eps_air, 
    eps_mid=eps_avg_core, 
    eps_bot=eps_air, # Symmetric for now, but function handles asymmetric
    lambda0=lambda0
)

# Output Results
print(f"Found {num_modes} guided mode(s).")
for m in modes:
    print(f"Mode TE{m['mode_index']}: neff = {m['neff']:.4f}, Confinement = {m['confinement']:.4f}")

# Plotting
if num_modes > 0:
    plt.figure(figsize=(8, 6))
    colors = ['r', 'b', 'g', 'm']
    
    for m in modes:
        idx = m['mode_index']
        color = colors[idx % len(colors)]
        plt.plot(m['z_grid']*1e9, m['theta']**2, color=color, linewidth=2, 
                 label=f'TE{idx} ($n_{{eff}}$={m["neff"]:.3f})')
        
    # Draw Slab Boundaries
    plt.axvspan(-d_core/2*1e9, d_core/2*1e9, color='gray', alpha=0.2, label='Core')
    plt.xlabel('z (nm)')
    plt.ylabel('Normalized Intensity $|\Theta(z)|^2$')
    plt.title(f'Analytical Slab Modes (Average Permittivity)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
