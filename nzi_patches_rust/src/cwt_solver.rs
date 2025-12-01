use num_complex::Complex;
use std::f64::consts::PI;
use std::collections::HashMap;
use nalgebra::{DMatrix, DVector, ComplexField};
use nalgebra::Schur;

pub struct CwtParams {
    pub xi: HashMap<(i32, i32), Complex<f64>>,
    pub n_eff: f64,
    pub theta_z: Vec<f64>,
    pub z_grid: Vec<f64>,
    pub n0_z: Vec<f64>,
    pub a: f64,
    pub lambda0: f64,
    pub conf: f64,
    pub d_trunc: i32,
}

fn simpson_integrate(y: &[Complex<f64>], x: &[f64]) -> Complex<f64> {
    if y.len() < 2 { return Complex::new(0.0, 0.0); }
    let mut sum = Complex::new(0.0, 0.0);
    for i in 0..y.len()-1 {
        let dx = x[i+1] - x[i];
        sum += (y[i] + y[i+1]) * dx / 2.0; // Trapezoidal for simplicity/robustness with non-uniform grid
    }
    sum
}

fn calculate_greens_integral(theta_z: &[f64], z_grid: &[f64], n0_z: &[f64], k0: f64) -> Complex<f64> {
    let n = z_grid.len();
    if n < 2 { return Complex::new(0.0, 0.0); }
    
    let mut outer_integral = Complex::new(0.0, 0.0);
    
    // Precompute beta_z
    let beta_z: Vec<f64> = n0_z.iter().map(|&n| k0 * n).collect();
    
    // Double integral
    // I = integral( integral( G(z, z') * Theta(z') * Theta*(z) dz' ) dz )
    // G(z, z') = -i / (2 * beta_z) * exp(-i * beta_z * |z - z'|)
    
    // We can optimize this loop
    for i in 0..n {
        let z = z_grid[i];
        let th_z = theta_z[i];
        let bz = beta_z[i];
        
        let mut inner_integral = Complex::new(0.0, 0.0);
        
        for j in 0..n {
            let zp = z_grid[j];
            let th_zp = theta_z[j];
            // let bzp = beta_z[j]; // Use local beta? Or average? Python code used meshgrid BZ (beta_z[i])
            
            let green = if bz.abs() > 1e-9 {
                Complex::new(0.0, -1.0) / (2.0 * bz) * Complex::new(0.0, -bz * (z - zp).abs()).exp()
            } else {
                Complex::new(0.0, 0.0)
            };
            
            let integrand = green * th_zp * th_z; // th_z is real
            
            let dzp = if j < n - 1 { z_grid[j+1] - z_grid[j] } else { z_grid[j] - z_grid[j-1] };
            inner_integral += integrand * dzp; // Simple rect/trapz
        }
        
        let dz = if i < n - 1 { z_grid[i+1] - z_grid[i] } else { z_grid[i] - z_grid[i-1] };
        outer_integral += inner_integral * dz;
    }
    
    outer_integral
}

pub fn construct_cwt_matrices(params: &CwtParams) -> (DMatrix<Complex<f64>>, DMatrix<Complex<f64>>, DMatrix<Complex<f64>>) {
    let k0 = 2.0 * PI / params.lambda0;
    let beta0 = 2.0 * PI / params.a;
    let prefactor = k0.powi(2) / (2.0 * beta0);
    
    // C_1D
    let xi_20 = params.xi.get(&(2, 0)).cloned().unwrap_or(Complex::new(0.0, 0.0));
    let xi_02 = params.xi.get(&(0, 2)).cloned().unwrap_or(Complex::new(0.0, 0.0));
    let gamma = params.conf;
    
    let kappa_x = -Complex::new(prefactor, 0.0) * xi_20 * gamma;
    let kappa_y = -Complex::new(prefactor, 0.0) * xi_02 * gamma;
    
    let mut c_1d = DMatrix::from_element(4, 4, Complex::new(0.0, 0.0));
    c_1d[(0, 1)] = kappa_x;
    c_1d[(1, 0)] = kappa_x.conj();
    c_1d[(2, 3)] = kappa_y;
    c_1d[(3, 2)] = kappa_y.conj();
    
    // C_rad
    let g_int = calculate_greens_integral(&params.theta_z, &params.z_grid, &params.n0_z, k0);
    let zeta_factor = -Complex::new(k0.powi(4) / (2.0 * beta0), 0.0) * g_int;
    
    let xi_10 = params.xi.get(&(1, 0)).cloned().unwrap_or(Complex::new(0.0, 0.0));
    let xi_m10 = xi_10.conj();
    let xi_01 = params.xi.get(&(0, 1)).cloned().unwrap_or(Complex::new(0.0, 0.0));
    let xi_m01 = xi_01.conj();
    
    let mut c_rad = DMatrix::from_element(4, 4, Complex::new(0.0, 0.0));
    
    // Row 1
    c_rad[(0, 0)] = zeta_factor * xi_10 * xi_m10;
    c_rad[(0, 1)] = zeta_factor * xi_10 * xi_10;
    c_rad[(0, 2)] = zeta_factor * xi_10 * xi_m01;
    c_rad[(0, 3)] = zeta_factor * xi_10 * xi_01;
    
    // Row 2
    c_rad[(1, 0)] = zeta_factor * xi_m10 * xi_m10;
    c_rad[(1, 1)] = zeta_factor * xi_m10 * xi_10;
    c_rad[(1, 2)] = zeta_factor * xi_m10 * xi_m01;
    c_rad[(1, 3)] = zeta_factor * xi_m10 * xi_01;
    
    // Row 3
    c_rad[(2, 0)] = zeta_factor * xi_01 * xi_m10;
    c_rad[(2, 1)] = zeta_factor * xi_01 * xi_10;
    c_rad[(2, 2)] = zeta_factor * xi_01 * xi_m01;
    c_rad[(2, 3)] = zeta_factor * xi_01 * xi_01;
    
    // Row 4
    c_rad[(3, 0)] = zeta_factor * xi_m01 * xi_m10;
    c_rad[(3, 1)] = zeta_factor * xi_m01 * xi_10;
    c_rad[(3, 2)] = zeta_factor * xi_m01 * xi_m01;
    c_rad[(3, 3)] = zeta_factor * xi_m01 * xi_01;
    
    // C_2D (Placeholder)
    let c_2d = DMatrix::from_element(4, 4, Complex::new(0.0, 0.0));
    
    (c_1d, c_rad, c_2d)
}

pub fn solve_cwt_eigenproblem(c_total: &DMatrix<Complex<f64>>) -> (Vec<Complex<f64>>, DMatrix<Complex<f64>>) {
    // Convert 4x4 Complex to 8x8 Real
    // M_real = [[Re, -Im], [Im, Re]]
    let mut m_real = DMatrix::<f64>::zeros(8, 8);
    for i in 0..4 {
        for j in 0..4 {
            let val = c_total[(i, j)];
            m_real[(i, j)] = val.re;
            m_real[(i, j + 4)] = -val.im;
            m_real[(i + 4, j)] = val.im;
            m_real[(i + 4, j + 4)] = val.re;
        }
    }

    let schur = Schur::new(m_real);
    let (q, t) = schur.unpack();
    
    // Extract eigenvalues from T (quasi-upper triangular)
    let mut eigenvalues = Vec::new();
    let n = 8;
    let mut i = 0;
    while i < n {
        if i == n - 1 || t[(i + 1, i)].abs() < 1e-9 {
            // Real eigenvalue
            eigenvalues.push(Complex::new(t[(i, i)], 0.0));
            i += 1;
        } else {
            // 2x2 block
            let a = t[(i, i)];
            let b = t[(i, i + 1)];
            let c = t[(i + 1, i)];
            let d = t[(i + 1, i + 1)];
            // Trace = a + d, Det = ad - bc
            // lambda^2 - tr lambda + det = 0
            let tr = a + d;
            let det = a * d - b * c;
            let delta = tr * tr - 4.0 * det;
            let sqrt_delta = Complex::new(delta, 0.0).sqrt();
            let l1 = (Complex::new(tr, 0.0) + sqrt_delta) * 0.5;
            let l2 = (Complex::new(tr, 0.0) - sqrt_delta) * 0.5;
            eigenvalues.push(l1);
            eigenvalues.push(l2);
            i += 2;
        }
    }
    
    // Compute eigenvectors for each eigenvalue
    let mut valid_pairs = Vec::new();
    
    for &lambda in &eigenvalues {
        // Solve (T - lambda I) x = 0
        // Back-substitution
        let mut x = DMatrix::<Complex<f64>>::zeros(n, 1);
        // We can set the last component to 1 (or the one corresponding to the block)
        // But T is quasi-upper.
        // For complex lambda, diagonal elements T[k,k] - lambda are never zero (unless lambda is real and matches).
        // If lambda is complex, T - lambda I is invertible? No, lambda IS an eigenvalue.
        // So T - lambda I is singular.
        
        // We need to find a non-trivial solution.
        // Since T is quasi-upper, we can solve from bottom up?
        // Actually, for a specific eigenvalue lambda, we can find an eigenvector.
        // If lambda comes from block at index k, then x[j] = 0 for j > k+1.
        // We can solve the block equation to find x[k], x[k+1].
        // Then back substitute for j < k.
        
        // Find the index of the block corresponding to lambda
        // We iterate blocks again or just search?
        // Let's assume we know the block index `blk_idx`.
        // But we are iterating eigenvalues.
        // Let's find the first block that produces this lambda (approx).
        
        let mut blk_idx = 0;
        let mut found = false;
        let mut i = 0;
        while i < n {
             if i == n - 1 || t[(i + 1, i)].abs() < 1e-9 {
                if (Complex::new(t[(i, i)], 0.0) - lambda).norm() < 1e-5 {
                    blk_idx = i;
                    found = true;
                    break;
                }
                i += 1;
            } else {
                // Check if lambda matches this block
                let a = t[(i, i)];
                let b = t[(i, i + 1)];
                let c = t[(i + 1, i)];
                let d = t[(i + 1, i + 1)];
                let tr = a + d;
                let det = a * d - b * c;
                let delta = tr * tr - 4.0 * det;
                let sqrt_delta = Complex::new(delta, 0.0).sqrt();
                let l1 = (Complex::new(tr, 0.0) + sqrt_delta) * 0.5;
                let l2 = (Complex::new(tr, 0.0) - sqrt_delta) * 0.5;
                
                if (l1 - lambda).norm() < 1e-5 || (l2 - lambda).norm() < 1e-5 {
                    blk_idx = i; // Start of block
                    found = true;
                    break;
                }
                i += 2;
            }
        }
        
        if !found { continue; }
        
        // Solve
        // If 1x1 block at blk_idx: x[blk_idx] = 1, x[j] = 0 for j > blk_idx.
        // If 2x2 block at blk_idx: solve (Block - lambda I) [u; v] = 0.
        
        if blk_idx == n - 1 || t[(blk_idx + 1, blk_idx)].abs() < 1e-9 {
            // 1x1
            x[blk_idx] = Complex::new(1.0, 0.0);
        } else {
            // 2x2
            // [[a-L, b], [c, d-L]] [[u], [v]] = 0
            // Pick u=1, solve for v? Or pick v=1?
            // If b is not zero, v = -(a-L)u / b.
            // If c is not zero, u = -(d-L)v / c.
            let a = t[(blk_idx, blk_idx)];
            let b = t[(blk_idx, blk_idx + 1)];
            // let c = t[(blk_idx + 1, blk_idx)];
            // let d = t[(blk_idx + 1, blk_idx + 1)];
            
            if b.abs() > 1e-9 {
                x[blk_idx] = Complex::new(1.0, 0.0);
                x[blk_idx + 1] = -(Complex::new(a, 0.0) - lambda) / b;
            } else {
                x[blk_idx] = Complex::new(0.0, 0.0);
                x[blk_idx + 1] = Complex::new(1.0, 0.0);
            }
        }
        
        // Back substitute for k < blk_idx
        for k in (0..blk_idx).rev() {
            let mut sum = Complex::new(0.0, 0.0);
            for j in k+1..=blk_idx+1 { // Only go up to where x is non-zero
                if j < n {
                    sum += t[(k, j)] * x[j];
                }
            }
            
            // (T[k,k] - lambda) x[k] + sum = 0
            // But T[k,k] might be part of a 2x2 block?
            // Wait, back substitution must handle 2x2 blocks too!
            // If k is part of a 2x2 block (k-1, k), we solve them together.
            // But we iterate k downwards.
            // Check if k is the second element of a 2x2 block.
            // t[(k, k-1)] != 0 implies k is second element.
            
            if k > 0 && t[(k, k-1)].abs() > 1e-9 {
                // Skip this k, handle at k-1
                continue;
            }
            
            // Check if k is first element of 2x2 block
            if k < n - 1 && t[(k+1, k)].abs() > 1e-9 {
                // 2x2 block at k, k+1
                // [[Tkk-L, Tkk+1], [Tk+1k, Tk+1k+1-L]] [[xk], [xk+1]] = [[-sum_k], [-sum_k+1]]
                let mut sum_k = Complex::new(0.0, 0.0);
                for j in k+2..n {
                    sum_k += t[(k, j)] * x[j];
                }
                let mut sum_kp1 = Complex::new(0.0, 0.0);
                for j in k+2..n {
                    sum_kp1 += t[(k+1, j)] * x[j];
                }
                
                // Solve 2x2 system
                let m11 = Complex::new(t[(k, k)], 0.0) - lambda;
                let m12 = Complex::new(t[(k, k+1)], 0.0);
                let m21 = Complex::new(t[(k+1, k)], 0.0);
                let m22 = Complex::new(t[(k+1, k+1)], 0.0) - lambda;
                
                let rhs1 = -sum_k;
                let rhs2 = -sum_kp1;
                
                // Cramer's rule or inverse
                let det = m11 * m22 - m12 * m21;
                if det.norm() > 1e-9 {
                    x[k] = (rhs1 * m22 - rhs2 * m12) / det;
                    x[k+1] = (m11 * rhs2 - m21 * rhs1) / det;
                }
            } else {
                // 1x1 block
                let denom = Complex::new(t[(k, k)], 0.0) - lambda;
                if denom.norm() > 1e-9 {
                    x[k] = -sum / denom;
                }
            }
        }
        
        // Transform x by Q to get eigenvector of M
        // Q is real 8x8. x is complex 8x1.
        // res = Q * x
        let mut res = DMatrix::<Complex<f64>>::zeros(n, 1);
        for r in 0..n {
            let mut val = Complex::new(0.0, 0.0);
            for c in 0..n {
                val += q[(r, c)] * x[c];
            }
            res[r] = val;
        }
        
        // Reconstruct v = x_vec + i y_vec
        let mut v = DVector::<Complex<f64>>::zeros(4);
        for i in 0..4 {
            v[i] = res[i] + Complex::new(0.0, 1.0) * res[i+4];
        }
        
        if v.norm() > 1e-3 {
            v.normalize_mut();
            valid_pairs.push((lambda, v));
        }
    }
    
    // Sort and return top 4
    valid_pairs.sort_by(|a, b| a.0.im.partial_cmp(&b.0.im).unwrap());
    
    // Deduplicate?
    // Simple deduplication: if lambda close to previous, skip?
    // But we might have multiplicity.
    // Let's just take top 4.
    
    let count = valid_pairs.len().min(4);
    let mut final_vals = Vec::new();
    let mut final_vecs_cols = Vec::new();
    
    for i in 0..count {
        final_vals.push(valid_pairs[i].0);
        final_vecs_cols.push(valid_pairs[i].1.clone());
    }
    
    if count < 4 {
        for _ in count..4 {
            final_vals.push(Complex::new(0.0, 0.0));
            final_vecs_cols.push(DVector::zeros(4));
        }
    }
    
    let sorted_vecs = DMatrix::from_columns(&final_vecs_cols);
    // let sorted_vecs = DMatrix::zeros(4, 4); // Placeholder
    
    (final_vals, sorted_vecs)
}

pub fn calculate_field_distributions(eigvecs: &DMatrix<Complex<f64>>, a: f64, resolution: usize) -> Vec<Vec<Vec<Complex<f64>>>> {
    let beta0 = 2.0 * PI / a;
    let x_vals: Vec<f64> = (0..resolution).map(|i| -a/2.0 + i as f64 * a / (resolution as f64 - 1.0)).collect();
    let y_vals: Vec<f64> = (0..resolution).map(|i| -a/2.0 + i as f64 * a / (resolution as f64 - 1.0)).collect();
    
    let mut fields = Vec::new();
    
    for i in 0..eigvecs.ncols() {
        let vec = eigvecs.column(i);
        let rx = vec[0];
        let sx = vec[1];
        let ry = vec[2];
        let sy = vec[3];
        
        let mut field_grid = vec![vec![Complex::new(0.0, 0.0); resolution]; resolution];
        
        for (ix, &x) in x_vals.iter().enumerate() {
            for (iy, &y) in y_vals.iter().enumerate() {
                let term_rx = rx * Complex::new(0.0, -beta0 * x).exp();
                let term_sx = sx * Complex::new(0.0, beta0 * x).exp();
                let term_ry = ry * Complex::new(0.0, -beta0 * y).exp();
                let term_sy = sy * Complex::new(0.0, beta0 * y).exp();
                
                field_grid[iy][ix] = term_rx + term_sx + term_ry + term_sy;
            }
        }
        fields.push(field_grid);
    }
    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construct_matrices() {
        // Basic test to ensure no panic
        let params = CwtParams {
            xi: HashMap::new(),
            n_eff: 2.0,
            theta_z: vec![1.0, 1.0],
            z_grid: vec![0.0, 1.0],
            n0_z: vec![1.0, 1.0],
            a: 1.0,
            lambda0: 1.55,
            conf: 0.5,
            d_trunc: 1,
        };
        let (c1, cr, c2) = construct_cwt_matrices(&params);
        assert_eq!(c1.shape(), (4, 4));
    }
}
