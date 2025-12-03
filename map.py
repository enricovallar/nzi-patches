import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import minimize
# Import necessary classes from cwt_solver.py for local use
from cwt_solver import Waveguide1D, AnalyticFourierProvider, CWTSolver
import fourier

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def get_solver_for_radii(r1, r2, a, d, eps_InP, eps_air, lambda0, gvecs):
    """
    Helper to reconstruct geometry and solver for given radii r1, r2.
    """
    # 1. Define Shapes: r1=Center, r2=Corner (C4v geometry)
    shapes = [
        {'eps': eps_air, 'r': r1, 'center': (0.5 * a, 0.5 * a)}, # Center Hole
        {'eps': eps_air, 'r': r2, 'center': (0, 0)}              # Corner Hole
    ]

    # 2. Recalculate Average Epsilon for 1D Mode Solver
    f1 = np.pi * r1**2 / a**2
    f2 = np.pi * r2**2 / a**2
    
    if f1 + f2 > 0.9: 
        return None, "High Filling Factor"
        
    eps_avg = eps_InP * (1 - (f1 + f2)) + eps_air * (f1 + f2)

    # 3. Solve 1D Vertical Mode
    layers = [
        {'n': np.sqrt(eps_air), 'thickness': 2.0e-6, 'type': 'clad_bot'},
        {'n': np.sqrt(eps_avg), 'thickness': d,      'type': 'pc'},
        {'n': np.sqrt(eps_air), 'thickness': 2.0e-6, 'type': 'clad_top'}
    ]
    wg = Waveguide1D(layers, lambda0)
    wg.solve_mode()
    
    # 4. Initialize CWT Solver (D=5 is usually sufficient for accurate mapping)
    solver = CWTSolver(wg, gvecs, eps_InP, shapes, truncation_order=3, lattice_constant=a)
    return solver, None

def calculate_mode_gap(r1, r2, a, d, eps_InP, eps_air, lambda0, gvecs):
    """
    Calculates the absolute frequency difference between Mode 4 and Mode 2.
    """
    solver, error = get_solver_for_radii(r1, r2, a, d, eps_InP, eps_air, lambda0, gvecs)
    if error:
        return np.nan 
        
    try:
        vals, _ = solver.solve()
        reals = np.sort(np.real(vals))
        
        # Calculate gap between Mode 4 (index 3) and Mode 2 (index 1)
        # Note: We use the absolute difference between the frequencies (detuning delta)
        # Since omega = delta + const, minimizing delta gap minimizes omega gap.
        gap = np.abs(reals[3] - reals[1])
        return gap
        
    except Exception as e:
        # Catch numerical instability or complex beta
        return np.nan

# ==========================================
# 2. MAIN GRID SCAN
# ==========================================
if __name__ == "__main__":
    # Constants (based on thesis parameters)
    a = 1e-6
    d = 0.25 * a
    eps_InP = 9.46
    eps_air = 1.0
    lambda0 = 1.55e-6
    
    # Pre-calculate G-vectors 
    gmax = 20 * 2 * np.pi / a
    gvecs = fourier.get_g_vectors(gmax, a)

    # Define Scan Space (0.2a to 0.3a)
    R_min = 0.21 * a
    R_max = 0.25 * a
    N_points = 10 # Number of points in each dimension
    
    r_ratios = np.linspace(R_min / a, R_max / a, N_points)
    r1_values = r_ratios * a
    r2_values = r_ratios * a
    
    gap_map = np.zeros((N_points, N_points))
    total_iterations = N_points * N_points
    
    print(f"--- Starting {N_points}x{N_points} Grid Scan ({R_min/a:.2f}a to {R_max/a:.2f}a) ---")
    
    for i, r1 in enumerate(r1_values):
        for j, r2 in enumerate(r2_values):
            # Progress tracking
            current_iteration = i * N_points + j + 1
            progress_percent = (current_iteration / total_iterations) * 100
            
            # Calculate gap. Note: i -> r1 (Y-axis in plot), j -> r2 (X-axis in plot)
            gap = calculate_mode_gap(r1, r2, a, d, eps_InP, eps_air, lambda0, gvecs)
            gap_map[i, j] = gap
            
            # Print progress to console (using \r to overwrite line)
            print(f"\rProcessing {current_iteration}/{total_iterations} ({progress_percent:.1f}% complete) - Current Gap: {gap:.2e}", end='', flush=True)

    print("\n--- Grid Scan Complete ---")

    # ==========================================
    # 3. PLOT HEATMAP
    # ==========================================
    
    # The locus of degeneracy should appear as a region of very low gap values (blue/dark)
    
    # Clip map for better visual contrast (e.g., max gap to display is 1000)
    gap_map_clipped = np.clip(gap_map, 0, np.nanmedian(gap_map[gap_map > 0]) * 3) 
    
    plt.figure(figsize=(9, 7))
    
    # Ensure Ratios are used for Ticks
    extent = [r_ratios.min(), r_ratios.max(), r_ratios.min(), r_ratios.max()]
    
    # Plotting the gap map
    im = plt.imshow(gap_map_clipped, cmap='viridis', extent=extent, aspect='auto', interpolation="spline16", origin='lower')
    
    # Plot elliptical fit if desired (optional based on thesis knowledge)
    
    plt.title(f'Detuning Gap $|\delta_4 - \delta_2|$ in Parameter Space (2D Scan)')
    plt.xlabel('$r_2 / a$ (Corner Hole)')
    plt.ylabel('$r_1 / a$ (Center Hole)')
    
    cbar = plt.colorbar(im, label='Absolute Detuning Gap $|\delta_4 - \delta_2|$ (1/m)')
    

    plt.grid(False)
    plt.tight_layout()
    plt.savefig('r_r_gap_heatmap.png')
    print("\nSaved r_r_gap_heatmap.png showing the locus of the Dirac points.")
    plt.show()