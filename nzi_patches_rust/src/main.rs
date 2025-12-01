mod fourier;
mod mode_solver;
mod cwt_solver;

use num_complex::Complex;
use std::f64::consts::PI;
use std::collections::HashMap;
use plotters::prelude::*;

fn main() {
    // ==========================================
    // 1. DEFINE GEOMETRY
    // ==========================================
    let a: f64 = 1e-6;
    let d = 0.25 * a;
    let eps_inp = 3.17f64.powi(2);
    let eps_air = 1.0;
    let r1: f64 = 0.23 * a;
    let r2: f64 = 0.24 * a;
    let lambda0 = 1.55e-6;

    let eps_avg = eps_inp * (1.0 - PI * (r1.powi(2) + r2.powi(2)) / a.powi(2)) 
                + eps_air * (PI * (r1.powi(2) + r2.powi(2)) / a.powi(2));
    
    println!("Average Permittivity: {:.4}", eps_avg);

    // ==========================================
    // 2. FOURIER COEFFICIENTS
    // ==========================================
    let shapes = vec![
        fourier::CircleShape { eps: eps_air, r: r1, center: (0.5 * a, 0.0) },
        fourier::CircleShape { eps: eps_air, r: r2, center: (0.0, 0.5 * a) },
    ];

    let gmax = 5.0 * 2.0 * PI / a;
    let (gx, gy) = fourier::get_g_vectors(gmax, a);
    let eps_ft_coeffs = fourier::get_epsilon_coefficients_analytic(&gx, &gy, eps_inp, &shapes, a);

    let xi_00 = fourier::get_xi_mn(0, 0, a, &gx, &gy, &eps_ft_coeffs);
    let xi_10 = fourier::get_xi_mn(1, 0, a, &gx, &gy, &eps_ft_coeffs);
    let xi_01 = fourier::get_xi_mn(0, 1, a, &gx, &gy, &eps_ft_coeffs);
    let xi_20 = fourier::get_xi_mn(2, 0, a, &gx, &gy, &eps_ft_coeffs);
    let xi_02 = fourier::get_xi_mn(0, 2, a, &gx, &gy, &eps_ft_coeffs);

    println!("Xi_0,0 (Avg Eps): {:.4}", xi_00.re);
    println!("Xi_1,0 (Coupling): {:.4}", xi_10.norm());
    println!("Xi_0,1 (Coupling): {:.4}", xi_01.norm());

    // ==========================================
    // 3. MODE SOLVER
    // ==========================================
    let (modes, num_modes) = mode_solver::solve_slab_modes(
        2e-6, 0.25e-6, 2e-6,
        eps_air, eps_avg, eps_air,
        lambda0, 1000
    );

    println!("Found {} guided mode(s).", num_modes);
    if num_modes == 0 {
        return;
    }

    let te0 = &modes[0];
    println!("Mode TE0: neff = {:.4}, Confinement = {:.4}", te0.neff, te0.confinement);

    // ==========================================
    // 4. CWT SOLVER
    // ==========================================
    let mut xi_coeffs = HashMap::new();
    xi_coeffs.insert((0, 0), xi_00);
    xi_coeffs.insert((1, 0), xi_10);
    xi_coeffs.insert((0, 1), xi_01);
    xi_coeffs.insert((2, 0), xi_20);
    xi_coeffs.insert((0, 2), xi_02);
    xi_coeffs.insert((-1, 0), xi_10.conj());
    xi_coeffs.insert((0, -1), xi_01.conj());
    xi_coeffs.insert((-2, 0), xi_20.conj());
    xi_coeffs.insert((0, -2), xi_02.conj());

    let n_clad_val = eps_air.sqrt();
    let n_core_val = eps_avg.sqrt();
    
    let mut n0_z = vec![0.0; te0.z_grid.len()];
    for (i, &z) in te0.z_grid.iter().enumerate() {
        if z >= -d/2.0 && z <= d/2.0 {
            n0_z[i] = n_core_val;
        } else {
            n0_z[i] = n_clad_val;
        }
    }

    let cwt_params = cwt_solver::CwtParams {
        xi: xi_coeffs,
        n_eff: te0.neff,
        theta_z: te0.theta.clone(),
        z_grid: te0.z_grid.clone(),
        n0_z,
        a,
        lambda0,
        conf: te0.confinement,
        d_trunc: 2,
    };

    let (c_1d, c_rad, c_2d) = cwt_solver::construct_cwt_matrices(&cwt_params);
    let c_total = c_1d + c_rad + c_2d;

    let (eigvals, eigvecs) = cwt_solver::solve_cwt_eigenproblem(&c_total);

    println!("\nResults for lattice constant a = {:.0} nm:", a * 1e9);
    let c_light = 3e8;
    
    for (i, val) in eigvals.iter().enumerate() {
        let delta = val.re;
        let alpha = val.im;
        let d_freq = delta * c_light / (2.0 * PI * te0.neff);
        
        let vec = eigvecs.column(i);
        let mag: Vec<f64> = vec.iter().map(|c| c.norm()).collect();
        
        println!("Mode {}:", i + 1);
        println!("  Eigenvalue: {:.2e}", val);
        println!("  Loss (alpha): {:.2} cm^-1", alpha / 100.0);
        println!("  Detuning:     {:.2} m^-1 ({:.2} GHz)", delta, d_freq / 1e9);
        println!("  Vector Comp:  [Rx:{:.2}, Sx:{:.2}, Ry:{:.2}, Sy:{:.2}]", 
                 mag[0], mag[1], mag[2], mag[3]);
    }

    // ==========================================
    // 5. PLOTTING
    // ==========================================
    // Plot Eigenvalues
    let root = BitMapBackend::new("cwt_band_structure_rust.png", (600, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    
    let min_x = eigvals.iter().map(|v| v.re).fold(f64::INFINITY, f64::min);
    let max_x = eigvals.iter().map(|v| v.re).fold(f64::NEG_INFINITY, f64::max);
    let min_y = eigvals.iter().map(|v| v.im).fold(f64::INFINITY, f64::min);
    let max_y = eigvals.iter().map(|v| v.im).fold(f64::NEG_INFINITY, f64::max);
    
    let x_range = (min_x - 1e4)..(max_x + 1e4);
    let y_range = (min_y - 1e-10)..(max_y + 1e2); // Adjust for log scale or similar if needed

    let mut chart = ChartBuilder::on(&root)
        .caption("CWT Band Edge Eigenvalues", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_range)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart.draw_series(
        eigvals.iter().map(|v| Circle::new((v.re, v.im), 5, RED.filled()))
    ).unwrap();
    
    println!("\nSaved plot: cwt_band_structure_rust.png");
    
    // Field Plots
    let fields = cwt_solver::calculate_field_distributions(&eigvecs, a, 100);
    
    for (i, field) in fields.iter().enumerate() {
        let filename = format!("mode_{}_hz_rust.png", i + 1);
        let root = BitMapBackend::new(&filename, (600, 500)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        
        // Find max for normalization
        let max_val = field.iter().flatten().map(|c| c.norm()).fold(0.0, f64::max);
        
        // Rotate phase
        // Find index of max
        let mut max_idx = (0, 0);
        let mut max_mag = 0.0;
        for (iy, row) in field.iter().enumerate() {
            for (ix, val) in row.iter().enumerate() {
                if val.norm() > max_mag {
                    max_mag = val.norm();
                    max_idx = (ix, iy);
                }
            }
        }
        let phase_factor = (-Complex::new(0.0, field[max_idx.1][max_idx.0].arg())).exp();
        
        // Prepare data for heatmap
        // Plotters heatmap is a bit verbose, let's just plot points or use a crate like `plotters`
        // For simplicity, we will skip complex heatmap implementation in this snippet 
        // and just print that we calculated it. 
        // Implementing a heatmap in plotters requires creating rectangles for each pixel.
        
        let mut chart = ChartBuilder::on(&root)
            .caption(format!("Mode {} Re(Hz)", i+1), ("sans-serif", 30).into_font())
            .build_cartesian_2d(0usize..100usize, 0usize..100usize)
            .unwrap();
            
        chart.draw_series(
            (0usize..100usize).flat_map(|y| (0usize..100usize).map(move |x| (x, y)))
            .map(|(x, y)| {
                let val = field[y][x] * phase_factor;
                let norm_re = val.re / max_val; // -1 to 1
                
                // Color map RdBu
                let color = if norm_re > 0.0 {
                    RGBColor(
                        (255.0 * (1.0 - norm_re)) as u8, 
                        (255.0 * (1.0 - norm_re)) as u8, 
                        255
                    ) // Blue-ish (actually white to blue)
                } else {
                    RGBColor(
                        255, 
                        (255.0 * (1.0 + norm_re)) as u8, 
                        (255.0 * (1.0 + norm_re)) as u8
                    ) // Red-ish
                };
                
                Rectangle::new([(x, y), (x + 1, y + 1)], color.filled())
            })
        ).unwrap();
        
        println!("Saved plot: {}", filename);
    }
}
