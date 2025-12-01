use num_complex::Complex;
use std::f64::consts::PI;
use scilib::math::bessel::j;

pub fn get_g_vectors(gmax: f64, a: f64) -> (Vec<f64>, Vec<f64>) {
    let b = 2.0 * PI / a;
    let n_max = (gmax / b).ceil() as i32;
    
    let mut gx_vec = Vec::new();
    let mut gy_vec = Vec::new();
    
    for m in -n_max..=n_max {
        for n in -n_max..=n_max {
            let gx = m as f64 * b;
            let gy = n as f64 * b;
            let g_mag = (gx.powi(2) + gy.powi(2)).sqrt();
            
            if g_mag <= gmax + 1e-9 {
                gx_vec.push(gx);
                gy_vec.push(gy);
            }
        }
    }
    (gx_vec, gy_vec)
}

pub struct CircleShape {
    pub eps: f64,
    pub r: f64,
    pub center: (f64, f64),
}

pub fn get_circle_ft(gx: &[f64], gy: &[f64], r: f64, center: (f64, f64)) -> Vec<Complex<f64>> {
    let mut ft = Vec::with_capacity(gx.len());
    let (cx, cy) = center;
    
    for i in 0..gx.len() {
        let g_mag = (gx[i].powi(2) + gy[i].powi(2)).sqrt();
        let phase = Complex::new(0.0, -(gx[i] * cx + gy[i] * cy)).exp();
        
        let val = if g_mag < 1e-9 {
            PI * r.powi(2)
        } else {
            2.0 * PI * r * j(g_mag * r, 1).re / g_mag
        };
        
        ft.push(Complex::new(val, 0.0) * phase);
    }
    ft
}

pub fn get_epsilon_coefficients_analytic(
    gx: &[f64], 
    gy: &[f64], 
    eps_bg: f64, 
    shapes: &[CircleShape], 
    a: f64
) -> Vec<Complex<f64>> {
    let area = a.powi(2);
    let mut coeffs = vec![Complex::new(0.0, 0.0); gx.len()];
    
    for i in 0..gx.len() {
        let g_mag = (gx[i].powi(2) + gy[i].powi(2)).sqrt();
        if g_mag < 1e-9 {
            coeffs[i] = Complex::new(eps_bg, 0.0);
        }
    }
    
    for shape in shapes {
        let ft_shape = get_circle_ft(gx, gy, shape.r, shape.center);
        for i in 0..gx.len() {
            coeffs[i] += (shape.eps - eps_bg) * ft_shape[i] / area;
        }
    }
    
    coeffs
}

pub fn get_xi_mn(m: i32, n: i32, a: f64, gx: &[f64], gy: &[f64], coeffs: &[Complex<f64>]) -> Complex<f64> {
    let b = 2.0 * PI / a;
    let target_gx = m as f64 * b;
    let target_gy = n as f64 * b;
    
    let mut min_dist_sq = f64::MAX;
    let mut min_idx = 0;
    
    for i in 0..gx.len() {
        let dist_sq = (gx[i] - target_gx).powi(2) + (gy[i] - target_gy).powi(2);
        if dist_sq < min_dist_sq {
            min_dist_sq = dist_sq;
            min_idx = i;
        }
    }
    
    if min_dist_sq > 1e-6 {
        println!("Warning: G_({},{}) not found in expansion", m, n);
        return Complex::new(0.0, 0.0);
    }
    
    coeffs[min_idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_g_vectors() {
        let a = 1.0;
        let gmax = 2.0 * PI / a * 1.1; // Should include -1, 0, 1
        let (gx, gy) = get_g_vectors(gmax, a);
        // (0,0), (1,0), (-1,0), (0,1), (0,-1) -> 5 vectors
        assert_eq!(gx.len(), 5);
    }
}
