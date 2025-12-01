use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct Mode {
    pub mode_index: usize,
    pub neff: f64,
    pub theta: Vec<f64>,
    pub z_grid: Vec<f64>,
    pub confinement: f64,
}

pub fn solve_slab_modes(
    d_top: f64,
    d_mid: f64,
    d_bot: f64,
    eps_top: f64,
    eps_mid: f64,
    eps_bot: f64,
    lambda0: f64,
    z_grid_res: usize,
) -> (Vec<Mode>, usize) {
    let k0 = 2.0 * PI / lambda0;
    let n_top = eps_top.sqrt();
    let n_core = eps_mid.sqrt();
    let n_bot = eps_bot.sqrt();

    let n_min = n_top.max(n_bot);
    let n_max = n_core;

    if n_min >= n_max {
        println!("Warning: No guided modes possible.");
        return (vec![], 0);
    }

    let dispersion_func = |neff: f64| -> f64 {
        if neff <= n_min + 1e-9 || neff >= n_max - 1e-9 {
            return f64::NAN;
        }
        let h = k0 * (n_core.powi(2) - neff.powi(2)).sqrt();
        let q = k0 * (neff.powi(2) - n_top.powi(2)).sqrt();
        let p = k0 * (neff.powi(2) - n_bot.powi(2)).sqrt();

        let lhs = (h * d_mid).tan();
        let rhs = (p + q) / (h * (1.0 - (p * q) / h.powi(2)));
        lhs - rhs
    };

    // Root finding
    let mut roots = Vec::new();
    let scan_points = 500;
    let step = (n_max - n_min - 2.0e-6) / scan_points as f64;
    
    for i in 0..scan_points {
        let x1 = n_min + 1e-6 + i as f64 * step;
        let x2 = x1 + step;
        
        let y1 = dispersion_func(x1);
        let y2 = dispersion_func(x2);
        
        if !y1.is_nan() && !y2.is_nan() && y1 * y2 < 0.0 {
            if (y1 - y2).abs() < 10.0 {
                // Simple bisection
                let mut a = x1;
                let mut b = x2;
                for _ in 0..50 {
                    let c = (a + b) / 2.0;
                    if dispersion_func(c).abs() < 1e-9 {
                        break;
                    }
                    if dispersion_func(a) * dispersion_func(c) < 0.0 {
                        b = c;
                    } else {
                        a = c;
                    }
                }
                roots.push((a + b) / 2.0);
            }
        }
    }
    
    roots.sort_by(|a, b| b.partial_cmp(a).unwrap());
    
    let mut found_modes = Vec::new();
    let z_start = -(d_mid/2.0 + d_bot);
    let z_end = d_mid/2.0 + d_top;
    let dz = (z_end - z_start) / (z_grid_res as f64 - 1.0);
    let z_grid: Vec<f64> = (0..z_grid_res).map(|i| z_start + i as f64 * dz).collect();
    
    for (i, &neff) in roots.iter().enumerate() {
        let h = k0 * (n_core.powi(2) - neff.powi(2)).sqrt();
        let q = k0 * (neff.powi(2) - n_top.powi(2)).sqrt();
        let p = k0 * (neff.powi(2) - n_bot.powi(2)).sqrt();
        
        let phi = (p / h).atan();
        
        let mut theta = Vec::with_capacity(z_grid_res);
        
        for &z_val in &z_grid {
            let val = if z_val < -d_mid/2.0 {
                let dist = -d_mid/2.0 - z_val;
                let amp = (-phi).cos();
                amp * (-p * dist).exp()
            } else if z_val > d_mid/2.0 {
                let dist = z_val - d_mid/2.0;
                let amp = (h * d_mid - phi).cos();
                amp * (-q * dist).exp()
            } else {
                (h * (z_val + d_mid/2.0) - phi).cos()
            };
            theta.push(val);
        }
        
        // Normalize
        let norm_sq: f64 = theta.iter().map(|x| x.powi(2)).sum::<f64>() * dz; // Simple trapz approx
        let norm = norm_sq.sqrt();
        let theta_norm: Vec<f64> = theta.iter().map(|x| x / norm).collect();
        
        // Confinement
        let mut conf_sum = 0.0;
        for (j, &z_val) in z_grid.iter().enumerate() {
            if z_val >= -d_mid/2.0 && z_val <= d_mid/2.0 {
                conf_sum += theta_norm[j].powi(2);
            }
        }
        let confinement = conf_sum * dz;
        
        found_modes.push(Mode {
            mode_index: i,
            neff,
            theta: theta_norm,
            z_grid: z_grid.clone(),
            confinement,
        });
    }
    
    let count = found_modes.len();
    (found_modes, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_slab_modes() {
        let modes = solve_slab_modes(
            2e-6, 0.25e-6, 2e-6,
            1.0, 3.17*3.17, 1.0,
            1.55e-6, 100
        );
        assert!(modes.1 > 0);
    }
}
