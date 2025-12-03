import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import minimize
import cwt_solver
import fourier

# ==========================================
# 1. SETUP & OBJECTIVE FUNCTION
# ==========================================

def get_solver_for_radii(x, a, d, eps_InP, eps_air, lambda0, gvecs):
    """
    Helper to reconstruct geometry and solver for given radii x = [r1, r2].
    """
    r1, r2 = x
    
    # 1. Update Shapes
    # Corrected Geometry based on Thesis Fig 5.2 (Corner and Center)
    # This preserves C4v symmetry to support E-mode degeneracy
    shapes = [
        {'eps': eps_air, 'r': r1, 'center': (0.5 * a, 0.5 * a)}, # Center Hole
        {'eps': eps_air, 'r': r2, 'center': (0, 0)}              # Corner Hole
    ]

    # 2. Recalculate Average Epsilon for 1D Mode Solver
    f1 = np.pi * r1**2 / a**2
    f2 = np.pi * r2**2 / a**2
    # Ensure filling factor isn't too high (physical constraint check roughly)
    if f1 + f2 > 0.9: # slightly relaxed constraint
        return None
        
    eps_avg = eps_InP * (1 - (f1 + f2)) + eps_air * (f1 + f2)

    # 3. Solve 1D Vertical Mode
    layers = [
        {'n': np.sqrt(eps_air), 'thickness': 2.0e-6, 'type': 'clad_bot'},
        {'n': np.sqrt(eps_avg), 'thickness': d,      'type': 'pc'},
        {'n': np.sqrt(eps_air), 'thickness': 2.0e-6, 'type': 'clad_top'}
    ]
    wg = cwt_solver.Waveguide1D(layers, lambda0)
    wg.solve_mode() # Solve in-place
    
    # 4. Initialize CWT Solver
    # D=10 for better accuracy in optimization
    solver = cwt_solver.CWTSolver(wg, gvecs, eps_InP, shapes, truncation_order=10, lattice_constant=a)
    return solver

def objective_function(x, a, d, eps_InP, eps_air, lambda0, gvecs):
    """
    Returns the gap between Real(Mode 4) and Real(Mode 2) at Gamma.
    """
    print(f"Optimizing: r1/a={x[0]/a:.4f}, r2/a={x[1]/a:.4f}...", end="")
    
    solver = get_solver_for_radii(x, a, d, eps_InP, eps_air, lambda0, gvecs)
    if solver is None:
        return 1e9 # Penalty for invalid geometry
        
    vals, _ = solver.solve()
    
    # Sort by real part (Detuning)
    reals = np.sort(np.real(vals))
    
    # Objective: Minimize gap between Mode 4 (index 3) and Mode 2 (index 1)
    # We want band 1 (A1/B1) and bands 3,4 (E doublet) to cross.
    # Note: Mode indices are 0, 1, 2, 3.
    # We want to minimize the distance between the singlet (which might be index 0 or 1)
    # and the doublet (usually indices 2 and 3 in CWT if degenerate).
    # Let's target minimizing the standard deviation of the top 3 eigenvalues 
    # or simply the gap between 2nd and 4th to force them close.
    # Based on thesis, we want degeneracy of 3 modes. 
    # Let's minimize the variance of the top 3 real eigenvalues.
    
    gap = np.std(reals[1:]) # Minimize spread of top 3 modes
    
    print(f" Gap (std)={gap:.2e}")
    return gap

# ==========================================
# 2. MAIN OPTIMIZATION LOOP
# ==========================================
if __name__ == "__main__":
    # Constants
    a = 1e-6
    d = 0.25 * a
    eps_InP = 9.46
    eps_air = 1.0
    lambda0 = 1.55e-6
    
    # Pre-calculate G-vectors (Geometry independent)
    gmax = 20 * 2 * np.pi / a
    gvecs = fourier.get_g_vectors(gmax, a)

    # Initial Guess (Based on Thesis sweet spot)
    r1_init = 0.24 * a
    r2_init = 0.24 * a
    x0 = [r1_init, r2_init]
    
    print("--- Starting Optimization ---")
    print(f"Target: Minimize Detuning Gap for Triple Degeneracy")
    
    # Bounds: Radii between 0.1a and 0.4a
    bounds = [(0.15*a, 0.35*a), (0.15*a, 0.35*a)]
    
    # Optimization
    res = minimize(
        objective_function, 
        x0, 
        args=(a, d, eps_InP, eps_air, lambda0, gvecs),
        method='Nelder-Mead',
        bounds=bounds,
        tol=1e-5,
        options={'maxiter': 40, 'disp': True}
    )
    
    r1_opt, r2_opt = res.x
    print("\n--- Optimization Complete ---")
    print(f"Optimized r1: {r1_opt/a:.4f} a")
    print(f"Optimized r2: {r2_opt/a:.4f} a")
    print(f"Final Gap: {res.fun:.2e}")

    # ==========================================
    # 3. PLOT OPTIMIZED RESULTS
    # ==========================================
    print("\nCalculated Optimized Band Diagram...")
    
    # Re-build solver with optimal parameters for High-Res Plotting
    solver_opt = get_solver_for_radii(res.x, a, d, eps_InP, eps_air, lambda0, gvecs)
    
    # Calculate Band Diagram
    k_range_norm = np.linspace(-0.015, 0.015, 61)
    beta0 = 2 * np.pi / a
    k_points = [(kn * beta0, 0) for kn in k_range_norm]
    
    bands = solver_opt.solve_band_diagram(k_points)
    
    # Convert to Normalized Frequency
    n_eff = solver_opt.wg.n_eff
    freqs_norm = (1.0 / n_eff) * (1.0 + np.real(bands) * a / (2 * np.pi))
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Band Diagram
    ax = axes[0]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for mode_idx in range(4):
        ax.plot(k_range_norm, freqs_norm[:, mode_idx], '.-', color=colors[mode_idx], label=f'Mode {mode_idx+1}')
    ax.set_title(f'Optimized Band Diagram\n$r_1={r1_opt/a:.3f}a, r_2={r2_opt/a:.3f}a$')
    ax.set_xlabel('$k_x a / 2\pi$')
    ax.set_ylabel('Normalized Frequency $\omega a / 2\pi c$')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Mode Fields at Gamma (for the optimized structure)
    _, vec_gamma = solver_opt.solve() # Solve at Gamma
    fields = solver_opt.calculate_field_distributions(vec_gamma, Nx=60, Ny=60)
    
    # Combine 4 mode plots into sub-subplots on the right
    import matplotlib.gridspec as gridspec
    # Use .get_subplotspec() to ensure compatibility
    gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=axes[1].get_subplotspec())
    
    for i in range(4):
        ax_sub = plt.subplot(gs[i])
        extent = [-a, a, -a, a] # Plot range (2x2 unit cells for visualization if desired, or -a/2 to a/2)
        # Note: Solver returns field on [-a, a] x [-a, a] by default in previous setup
        im = ax_sub.imshow(np.real(fields[i]), extent=extent, cmap='RdBu', origin='lower')
        ax_sub.set_title(f'Mode {i+1}', fontsize=9)
        ax_sub.axis('off')
        
        # Add colorbar for each subplot
        plt.colorbar(im, ax=ax_sub, fraction=0.046, pad=0.04)
        
        # Superimpose Circles
        # Hole 1 (Center): (0.5a, 0.5a)
        # Hole 2 (Corner): (0, 0)
        # We plot in range [-a, a] which covers 2 unit cells (-1 to 1).
        
        for ix in range(-1, 1):
            for iy in range(-1, 1):
                # Shift for lattice sites
                x_shift = ix * a
                y_shift = iy * a
                
                # Hole 1 (Center) relative to unit cell origin (0,0) is (0.5, 0.5)
                # Global coords:
                c1_x = x_shift + 0.5 * a
                c1_y = y_shift + 0.5 * a
                
                # Hole 2 (Corner) relative to unit cell origin is (0,0)
                # Global coords:
                c2_x = x_shift
                c2_y = y_shift
                
                # Add patches
                circle1 = Circle((c1_x, c1_y), r1_opt, fill=False, edgecolor='k', linewidth=0.8, linestyle='--')
                ax_sub.add_patch(circle1)
                
                circle2 = Circle((c2_x, c2_y), r2_opt, fill=False, edgecolor='k', linewidth=0.8, linestyle='--')
                ax_sub.add_patch(circle2)
                
                # Also add the corner holes for the upper/right boundaries to complete visual
                circle2_ur = Circle((c2_x + a, c2_y + a), r2_opt, fill=False, edgecolor='k', linewidth=0.8, linestyle='--')
                ax_sub.add_patch(circle2_ur)

    axes[1].set_title("Mode Profiles $Re(H_z)$ @ $\Gamma$")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('optimized_band_structure.png')
    print("Saved optimized_band_structure.png")