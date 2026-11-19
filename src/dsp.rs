use num_complex::Complex;
use rayon::prelude::*;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::f64::consts::PI;
use std::sync::Arc;

/// Smallest even 2/3/5-smooth integer >= n (optimal for realfft).
pub fn next_fast_len(n: usize) -> usize {
    if n <= 2 {
        return n.max(1);
    }
    let mut best = n.next_power_of_two();
    let mut p5 = 1usize;
    while p5 < best {
        let mut p3 = p5;
        while p3 < best {
            let mut p2 = p3 * 2; // enforce even length for RealToComplexEven
            while p2 < n {
                if let Some(next) = p2.checked_mul(2) {
                    p2 = next;
                } else {
                    p2 = usize::MAX;
                    break;
                }
            }
            if p2 >= n && p2 < best {
                best = p2;
            }
            if let Some(next) = p3.checked_mul(3) {
                p3 = next;
            } else {
                break;
            }
        }
        if let Some(next) = p5.checked_mul(5) {
            p5 = next;
        } else {
            break;
        }
    }
    best
}

pub fn mean_std_f32(vals: &[f32]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    let n = vals.len() as f64;
    let sum: f64 = vals.iter().map(|&v| v as f64).sum();
    let mean = sum / n;
    let var: f64 = vals
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    (mean, var.sqrt())
}

pub fn median_f64(vals: &[f64]) -> f64 {
    assert!(!vals.is_empty());
    let mut tmp = vals.to_vec();
    let n = tmp.len();
    let mid = n / 2;
    let (_, m, _) = tmp.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let hi = *m;
    if n % 2 == 1 {
        hi
    } else {
        let lo = tmp[..mid]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lo + hi)
    }
}

pub fn polyfit_linear(t: &[f64], d: &[f64]) -> (f64, f64) {
    let n = t.len() as f64;
    let tm: f64 = t.iter().sum::<f64>() / n;
    let dm: f64 = d.iter().sum::<f64>() / n;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&ti, &di) in t.iter().zip(d.iter()) {
        let dt = ti - tm;
        num += dt * (di - dm);
        den += dt * dt;
    }
    let b = if den > 0.0 { num / den } else { 0.0 };
    let a = dm - b * tm;
    (b, a)
}

/// Precomputed FFT of the reference signal for a given FFT length.
/// Allows evaluating `detect_tempo` (7 tempos x all tracks) and `global_offset` (all tracks)
/// without ever recomputing the FFT of `ref`.
pub struct RefFftCache {
    pub nfft: usize,
    pub ref_len: usize,
    pub ref_spec: Vec<Complex<f32>>,
    pub r2c: Arc<dyn RealToComplex<f32>>,
    pub c2r: Arc<dyn ComplexToReal<f32>>,
}

pub struct RefFftScratch {
    pub buf: Vec<f32>,
    pub spec: Vec<Complex<f32>>,
    pub scratch: Vec<Complex<f32>>,
}

impl RefFftCache {
    pub fn new(ref_audio: &[f32], min_nfft: usize) -> Self {
        let nfft = next_fast_len(min_nfft.max(2));
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(nfft);
        let c2r = planner.plan_fft_inverse(nfft);

        let mut buf = r2c.make_input_vec();
        buf[..ref_audio.len()].copy_from_slice(ref_audio);
        let mut ref_spec = r2c.make_output_vec();
        let mut scratch = r2c.make_scratch_vec();
        r2c.process_with_scratch(&mut buf, &mut ref_spec, &mut scratch)
            .expect("rfft ref failed");

        Self {
            nfft,
            ref_len: ref_audio.len(),
            ref_spec,
            r2c,
            c2r,
        }
    }

    pub fn make_scratch(&self) -> RefFftScratch {
        RefFftScratch {
            buf: self.r2c.make_input_vec(),
            spec: self.r2c.make_output_vec(),
            scratch: vec![
                Complex::new(0.0f32, 0.0f32);
                self.r2c.get_scratch_len().max(self.c2r.get_scratch_len())
            ],
        }
    }

    /// Valid cross-correlation `peak(ref, src_seg)` reusing caller-provided scratch buffers.
    pub fn peak_valid_with_scratch(
        &self,
        sc: &mut RefFftScratch,
        src_seg: &[f32],
    ) -> (usize, f64) {
        assert!(src_seg.len() <= self.ref_len);
        assert!(self.ref_len <= self.nfft);

        sc.buf[..src_seg.len()].copy_from_slice(src_seg);
        sc.buf[src_seg.len()..].fill(0.0);

        self.r2c
            .process_with_scratch(&mut sc.buf, &mut sc.spec, &mut sc.scratch)
            .expect("rfft src failed");

        for (s, &r) in sc.spec.iter_mut().zip(self.ref_spec.iter()) {
            *s = r * s.conj();
        }
        sc.spec[0].im = 0.0;
        if let Some(last) = sc.spec.last_mut() {
            last.im = 0.0;
        }

        self.c2r
            .process_with_scratch(&mut sc.spec, &mut sc.buf, &mut sc.scratch)
            .expect("irfft valid failed");

        let valid_len = self.ref_len - src_seg.len() + 1;
        let inv_n = 1.0f32 / (self.nfft as f32);
        let mut max_idx = 0usize;
        let mut max_val = -1.0f32;
        let mut sum = 0.0f64;
        for (i, v) in sc.buf[..valid_len].iter_mut().enumerate() {
            let av = v.abs() * inv_n;
            *v = av;
            sum += av as f64;
            if av > max_val {
                max_val = av;
                max_idx = i;
            }
        }
        let n = valid_len as f64;
        let mean = sum / n;
        let var: f64 = sc.buf[..valid_len]
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let conf = (max_val as f64 - mean) / (var.sqrt() + 1e-12);
        (max_idx, conf)
    }

    pub fn peak_valid(&self, src_seg: &[f32]) -> (usize, f64) {
        let mut sc = self.make_scratch();
        self.peak_valid_with_scratch(&mut sc, src_seg)
    }

    /// Full cross-correlation `global_offset(ref, src)` reusing scratch buffers.
    pub fn global_offset_with_scratch(&self, sc: &mut RefFftScratch, src: &[f32]) -> isize {
        assert!(self.ref_len + src.len() - 1 <= self.nfft);

        sc.buf[..src.len()].copy_from_slice(src);
        sc.buf[src.len()..].fill(0.0);

        self.r2c
            .process_with_scratch(&mut sc.buf, &mut sc.spec, &mut sc.scratch)
            .expect("rfft src full failed");

        for (s, &r) in sc.spec.iter_mut().zip(self.ref_spec.iter()) {
            *s = r * s.conj();
        }
        sc.spec[0].im = 0.0;
        if let Some(last) = sc.spec.last_mut() {
            last.im = 0.0;
        }

        self.c2r
            .process_with_scratch(&mut sc.spec, &mut sc.buf, &mut sc.scratch)
            .expect("irfft full failed");

        let mut best_lag = 0isize;
        let mut best_val = -1.0f32;
        for lag_abs in (1..src.len()).rev() {
            let v = sc.buf[self.nfft - lag_abs].abs();
            if v > best_val {
                best_val = v;
                best_lag = -(lag_abs as isize);
            }
        }
        for (lag, &val) in sc.buf[..self.ref_len].iter().enumerate() {
            let v = val.abs();
            if v > best_val {
                best_val = v;
                best_lag = lag as isize;
            }
        }
        best_lag
    }

    pub fn global_offset(&self, src: &[f32]) -> isize {
        let mut sc = self.make_scratch();
        self.global_offset_with_scratch(&mut sc, src)
    }
}

/// Thread-sharable FFT plan + per-thread scratch for windowed cross-correlations (`scan`, `dense_map`, `ncc`).
#[derive(Clone)]
pub struct SmallCorrPlan {
    pub nfft: usize,
    pub r2c: Arc<dyn RealToComplex<f32>>,
    pub c2r: Arc<dyn ComplexToReal<f32>>,
}

pub struct SmallCorrScratch {
    buf_r: Vec<f32>,
    buf_s: Vec<f32>,
    spec_r: Vec<Complex<f32>>,
    spec_s: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}

impl SmallCorrPlan {
    pub fn new(max_ref_len: usize) -> Self {
        let nfft = next_fast_len(max_ref_len.max(2));
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(nfft);
        let c2r = planner.plan_fft_inverse(nfft);
        Self { nfft, r2c, c2r }
    }

    pub fn make_scratch(&self) -> SmallCorrScratch {
        let slen = self.r2c.get_scratch_len().max(self.c2r.get_scratch_len());
        SmallCorrScratch {
            buf_r: self.r2c.make_input_vec(),
            buf_s: self.r2c.make_input_vec(),
            spec_r: self.r2c.make_output_vec(),
            spec_s: self.r2c.make_output_vec(),
            scratch: vec![Complex::new(0.0, 0.0); slen],
        }
    }

    fn correlate_valid_into<'a>(
        &self,
        sc: &'a mut SmallCorrScratch,
        ref_seg: &[f32],
        src_seg: &[f32],
    ) -> &'a mut [f32] {
        debug_assert!(src_seg.len() <= ref_seg.len());
        debug_assert!(ref_seg.len() <= self.nfft);

        sc.buf_r[..ref_seg.len()].copy_from_slice(ref_seg);
        sc.buf_r[ref_seg.len()..].fill(0.0);
        sc.buf_s[..src_seg.len()].copy_from_slice(src_seg);
        sc.buf_s[src_seg.len()..].fill(0.0);

        self.r2c
            .process_with_scratch(&mut sc.buf_r, &mut sc.spec_r, &mut sc.scratch)
            .unwrap();
        self.r2c
            .process_with_scratch(&mut sc.buf_s, &mut sc.spec_s, &mut sc.scratch)
            .unwrap();

        for (sr, &ss) in sc.spec_r.iter_mut().zip(sc.spec_s.iter()) {
            *sr = *sr * ss.conj();
        }
        sc.spec_r[0].im = 0.0;
        if let Some(last) = sc.spec_r.last_mut() {
            last.im = 0.0;
        }

        self.c2r
            .process_with_scratch(&mut sc.spec_r, &mut sc.buf_r, &mut sc.scratch)
            .unwrap();

        let valid_len = ref_seg.len() - src_seg.len() + 1;
        &mut sc.buf_r[..valid_len]
    }

    pub fn peak(&self, sc: &mut SmallCorrScratch, ref_seg: &[f32], src_seg: &[f32]) -> (usize, f64) {
        let inv_n = 1.0f32 / (self.nfft as f32);
        let valid = self.correlate_valid_into(sc, ref_seg, src_seg);
        let mut max_idx = 0usize;
        let mut max_val = -1.0f32;
        let mut sum = 0.0f64;
        for (i, v) in valid.iter_mut().enumerate() {
            let av = v.abs() * inv_n;
            *v = av;
            sum += av as f64;
            if av > max_val {
                max_val = av;
                max_idx = i;
            }
        }
        let n = valid.len() as f64;
        let mean = sum / n;
        let var: f64 = valid
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        (max_idx, (max_val as f64 - mean) / (var.sqrt() + 1e-12))
    }

    pub fn peak_subpixel(
        &self,
        sc: &mut SmallCorrScratch,
        ref_seg: &[f32],
        src_seg: &[f32],
    ) -> (f64, f64) {
        let (max_idx, conf) = self.peak(sc, ref_seg, src_seg);
        let valid_len = ref_seg.len() - src_seg.len() + 1;
        let frac = if max_idx > 0 && max_idx + 1 < valid_len {
            let yl = sc.buf_r[max_idx - 1] as f64;
            let yc = sc.buf_r[max_idx] as f64;
            let yr = sc.buf_r[max_idx + 1] as f64;
            let denom = yl - 2.0 * yc + yr;
            if denom.abs() > 1e-12 {
                (0.5 * (yl - yr) / denom).clamp(-0.5, 0.5)
            } else {
                0.0
            }
        } else {
            0.0
        };
        (max_idx as f64 + frac, conf)
    }

    pub fn max_abs_valid(&self, sc: &mut SmallCorrScratch, ref_seg: &[f32], src_seg: &[f32]) -> f64 {
        let inv_n = 1.0f32 / (self.nfft as f32);
        let valid = self.correlate_valid_into(sc, ref_seg, src_seg);
        let mut max_val = 0.0f32;
        for &v in valid.iter() {
            let av = v.abs();
            if av > max_val {
                max_val = av;
            }
        }
        (max_val * inv_n) as f64
    }
}

/// Signed `correlate(a, b, mode="full")` for subtitle measurement (~70k frames).
pub fn correlate_full_signed(a: &[f32], b: &[f32]) -> Vec<f32> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let full_len = a.len() + b.len() - 1;
    let nfft = next_fast_len(full_len);
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(nfft);
    let c2r = planner.plan_fft_inverse(nfft);

    let mut buf_a = r2c.make_input_vec();
    buf_a[..a.len()].copy_from_slice(a);
    let mut buf_b = r2c.make_input_vec();
    buf_b[..b.len()].copy_from_slice(b);

    let mut spec_a = r2c.make_output_vec();
    let mut spec_b = r2c.make_output_vec();
    let mut scratch = vec![
        Complex::new(0.0f32, 0.0f32);
        r2c.get_scratch_len().max(c2r.get_scratch_len())
    ];

    r2c.process_with_scratch(&mut buf_a, &mut spec_a, &mut scratch)
        .unwrap();
    r2c.process_with_scratch(&mut buf_b, &mut spec_b, &mut scratch)
        .unwrap();

    for (sa, &sb) in spec_a.iter_mut().zip(spec_b.iter()) {
        *sa = *sa * sb.conj();
    }
    spec_a[0].im = 0.0;
    if let Some(last) = spec_a.last_mut() {
        last.im = 0.0;
    }

    c2r.process_with_scratch(&mut spec_a, &mut buf_a, &mut scratch)
        .unwrap();

    let inv_n = 1.0f32 / (nfft as f32);
    let mut out = Vec::with_capacity(full_len);
    for lag_abs in (1..b.len()).rev() {
        out.push(buf_a[nfft - lag_abs] * inv_n);
    }
    for &v in &buf_a[..a.len()] {
        out.push(v * inv_n);
    }
    out
}

pub fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

pub fn limit_denominator(val: f64, max_denom: usize) -> (usize, usize) {
    let mut p0 = 0usize;
    let mut q0 = 1usize;
    let mut p1 = 1usize;
    let mut q1 = 0usize;
    let mut x = val;
    loop {
        let a = x.floor() as usize;
        let q2 = q0 + a * q1;
        if q2 > max_denom {
            break;
        }
        let p2 = p0 + a * p1;
        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;
        let rem = x - (a as f64);
        if rem < 1e-12 {
            return (p1, q1);
        }
        x = 1.0 / rem;
    }
    let k = (max_denom - q0) / q1;
    let p_bound = p0 + k * p1;
    let q_bound = q0 + k * q1;
    let err1 = ((p1 as f64) / (q1 as f64) - val).abs();
    let err2 = ((p_bound as f64) / (q_bound as f64) - val).abs();
    if err2 < err1 {
        (p_bound, q_bound)
    } else {
        (p1, q1)
    }
}

/// Modified Bessel function I0(x) for Kaiser window.
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y
            * (3.5156229
                + y * (3.0899424
                    + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + y * (0.01328592
                    + y * (0.00225319
                        + y * (-0.00157565
                            + y * (0.00916281
                                + y * (-0.02057706
                                    + y * (0.02635537
                                        + y * (-0.01647633 + y * 0.00392377))))))))
    }
}

/// Exact `scipy.signal.resample_poly(x, up, down)` using Kaiser(5.0) sinc FIR.
pub fn resample_poly(x: &[f32], up: usize, down: usize) -> Vec<f32> {
    if x.is_empty() {
        return vec![];
    }
    let g = gcd(up, down);
    let up = up / g;
    let down = down / g;
    if up == 1 && down == 1 {
        return x.to_vec();
    }

    let max_rate = up.max(down);
    let f_c = 1.0 / (max_rate as f64);
    let half_len = 10 * max_rate;
    let n_taps = 2 * half_len + 1;
    let denom_i0 = bessel_i0(5.0);
    let mut h = Vec::with_capacity(n_taps);
    let mut h_sum = 0.0f64;
    for m in 0..n_taps {
        let t = (m as f64) - (half_len as f64);
        let sinc = if t == 0.0 {
            f_c
        } else {
            (PI * f_c * t).sin() / (PI * t)
        };
        let r = t / (half_len as f64);
        let arg = 5.0 * (1.0 - r * r).max(0.0).sqrt();
        let win = bessel_i0(arg) / denom_i0;
        let val = sinc * win;
        h.push(val);
        h_sum += val;
    }
    let scale = (up as f64) / h_sum;
    let h_f32: Vec<f32> = h.into_iter().map(|v| (v * scale) as f32).collect();

    let n_out = ((x.len() as u128 * up as u128 + down as u128 - 1) / down as u128) as usize;
    let mut out = vec![0.0f32; n_out];
    let x_len = x.len();

    let compute_chunk = |base: usize, slice: &mut [f32]| {
        for (idx, dst) in slice.iter_mut().enumerate() {
            let j = base + idx;
            let k = j * down + half_len;
            let m_start = k % up;
            let mut acc = 0.0f32;
            let mut m = m_start;
            while m < n_taps && m <= k {
                let xi = (k - m) / up;
                if xi < x_len {
                    acc += h_f32[m] * x[xi];
                }
                m += up;
            }
            *dst = acc;
        }
    };

    if n_out > 50_000 {
        let chunk_sz = 16_384;
        out.par_chunks_mut(chunk_sz)
            .enumerate()
            .for_each(|(ci, sl)| compute_chunk(ci * chunk_sz, sl));
    } else {
        compute_chunk(0, &mut out);
    }
    out
}

/// Fast 2:1 Kaiser-sinc decimation from 16 kHz to 8 kHz (exact match to `resample_poly(x, 1, 2)`).
pub fn decimate_2x(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return vec![];
    }
    if x.len() < 48 {
        return resample_poly(x, 1, 2);
    }
    let half_len = 20usize;
    let n_taps = 41usize;
    let f_c = 0.5f64;
    let denom_i0 = bessel_i0(5.0);
    let mut h = [0.0f64; 41];
    let mut h_sum = 0.0f64;
    for m in 0..n_taps {
        let t = (m as f64) - (half_len as f64);
        let sinc = if t == 0.0 {
            f_c
        } else {
            (PI * f_c * t).sin() / (PI * t)
        };
        let r = t / (half_len as f64);
        let arg = 5.0 * (1.0 - r * r).max(0.0).sqrt();
        let win = bessel_i0(arg) / denom_i0;
        let val = sinc * win;
        h[m] = val;
        h_sum += val;
    }
    let scale = 1.0 / h_sum;
    let c0 = (h[20] * scale) as f32;
    // Non-zero odd offsets d = 1, 3, 5, ..., 19
    let c_odd: [f32; 10] = [
        (h[21] * scale) as f32,
        (h[23] * scale) as f32,
        (h[25] * scale) as f32,
        (h[27] * scale) as f32,
        (h[29] * scale) as f32,
        (h[31] * scale) as f32,
        (h[33] * scale) as f32,
        (h[35] * scale) as f32,
        (h[37] * scale) as f32,
        (h[39] * scale) as f32,
    ];

    let n_out = (x.len() + 1) / 2;
    let mut out = vec![0.0f32; n_out];
    let x_len = x.len();
    let interior_start = 10usize;
    let interior_end = (x_len.saturating_sub(20)) / 2;

    // Boundary left
    for j in 0..interior_start.min(n_out) {
        let center = (j * 2) as isize;
        let mut acc = c0 * x[center as usize];
        for (idx, &coeff) in c_odd.iter().enumerate() {
            let d = (idx * 2 + 1) as isize;
            let l = center - d;
            let r = center + d;
            let vl = if l >= 0 { x[l as usize] } else { 0.0 };
            let vr = if (r as usize) < x_len { x[r as usize] } else { 0.0 };
            acc += coeff * (vl + vr);
        }
        out[j] = acc;
    }

    // Fast branch-free interior in parallel
    if interior_end > interior_start {
        let chunk_sz = 32_768usize;
        out[interior_start..interior_end]
            .par_chunks_mut(chunk_sz)
            .enumerate()
            .for_each(|(ci, slice)| {
                let base = interior_start + ci * chunk_sz;
                for (k, dst) in slice.iter_mut().enumerate() {
                    let c = (base + k) * 2;
                    let mut acc = c0 * x[c];
                    acc += c_odd[0] * (x[c - 1] + x[c + 1]);
                    acc += c_odd[1] * (x[c - 3] + x[c + 3]);
                    acc += c_odd[2] * (x[c - 5] + x[c + 5]);
                    acc += c_odd[3] * (x[c - 7] + x[c + 7]);
                    acc += c_odd[4] * (x[c - 9] + x[c + 9]);
                    acc += c_odd[5] * (x[c - 11] + x[c + 11]);
                    acc += c_odd[6] * (x[c - 13] + x[c + 13]);
                    acc += c_odd[7] * (x[c - 15] + x[c + 15]);
                    acc += c_odd[8] * (x[c - 17] + x[c + 17]);
                    acc += c_odd[9] * (x[c - 19] + x[c + 19]);
                    *dst = acc;
                }
            });
    }

    // Boundary right
    for j in interior_end.max(interior_start)..n_out {
        let center = (j * 2) as isize;
        let mut acc = if (center as usize) < x_len {
            c0 * x[center as usize]
        } else {
            0.0
        };
        for (idx, &coeff) in c_odd.iter().enumerate() {
            let d = (idx * 2 + 1) as isize;
            let l = center - d;
            let r = center + d;
            let vl = if l >= 0 && (l as usize) < x_len {
                x[l as usize]
            } else {
                0.0
            };
            let vr = if r >= 0 && (r as usize) < x_len {
                x[r as usize]
            } else {
                0.0
            };
            acc += coeff * (vl + vr);
        }
        out[j] = acc;
    }
    out
}

pub fn stretch(x: &[f32], tempo: f64) -> Vec<f32> {
    if (tempo - 1.0).abs() < 1e-12 {
        return x.to_vec();
    }
    let (up, down) = limit_denominator(1.0 / tempo, 5000);
    resample_poly(x, up, down)
}

/// Exact SOS coefficients for `scipy.signal.butter(4, [300, 3000], "bandpass", fs=8000, output="sos")`.
const BUTTER_BP_300_3000_SOS: [[f64; 6]; 4] = [
    [
        2.44405685234252545e-01,
        4.88811370468505091e-01,
        2.44405685234252545e-01,
        1.00000000000000000e+00,
        7.85594617515539051e-01,
        1.89869136715115405e-01,
    ],
    [
        1.00000000000000000e+00,
        2.00000000000000000e+00,
        1.00000000000000000e+00,
        1.00000000000000000e+00,
        1.12132295001660620e+00,
        5.99760135691436824e-01,
    ],
    [
        1.00000000000000000e+00,
        -2.00000000000000000e+00,
        1.00000000000000000e+00,
        1.00000000000000000e+00,
        -1.57191755881355899e+00,
        6.25340792248505251e-01,
    ],
    [
        1.00000000000000000e+00,
        -2.00000000000000000e+00,
        1.00000000000000000e+00,
        1.00000000000000000e+00,
        -1.79466752834514387e+00,
        8.46945906146255756e-01,
    ],
];

/// Direct Form II Transposed SOS filter matching `scipy.signal.sosfilt`.
pub fn butter_bandpass_sosfilt(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    let mut z = [[0.0f64; 2]; 4];
    for (dst, &sample) in out.iter_mut().zip(x.iter()) {
        let mut v = sample as f64;
        for (s, &[b0, b1, b2, _a0, a1, a2]) in BUTTER_BP_300_3000_SOS.iter().enumerate() {
            let y = b0 * v + z[s][0];
            z[s][0] = b1 * v - a1 * y + z[s][1];
            z[s][1] = b2 * v - a2 * y;
            v = y;
        }
        *dst = v as f32;
    }
    out
}

/// STFT magnitude in bins `[bin_lo..=bin_hi]` matching `scipy.signal.stft(..., boundary=None)` (periodic Hann window).
/// Returns `(n_frames, data)` where `data` is `[n_frames * (bin_hi - bin_lo + 1)]`.
pub fn stft_hann_mag_band(
    x: &[f32],
    nperseg: usize,
    hop: usize,
    bin_lo: usize,
    bin_hi: usize,
) -> (usize, Vec<f32>) {
    if x.len() < nperseg {
        return (0, vec![]);
    }
    let n_frames = (x.len() - nperseg + hop - 1) / hop + 1;
    let n_bins = bin_hi - bin_lo + 1;
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(nperseg);

    let win: Vec<f32> = (0..nperseg)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f64) / (nperseg as f64)).cos()) as f32)
        .collect();
    let inv_wsum = 1.0f32 / win.iter().sum::<f32>();

    let mut out = vec![0.0f32; n_frames * n_bins];
    let chunk_frames = 2048usize;
    out.par_chunks_mut(chunk_frames * n_bins)
        .enumerate()
        .for_each_init(
            || {
                (
                    r2c.make_input_vec(),
                    r2c.make_output_vec(),
                    r2c.make_scratch_vec(),
                )
            },
            |(buf, spec, scratch), (ci, dst)| {
                let start_f = ci * chunk_frames;
                let count = dst.len() / n_bins;
                for fi in 0..count {
                    let fidx = start_f + fi;
                    let off = fidx * hop;
                    let avail = (x.len().saturating_sub(off)).min(nperseg);
                    for i in 0..avail {
                        buf[i] = x[off + i] * win[i];
                    }
                    buf[avail..nperseg].fill(0.0);
                    r2c.process_with_scratch(buf, spec, scratch).unwrap();
                    let row = &mut dst[fi * n_bins..(fi + 1) * n_bins];
                    for (bi, c) in spec[bin_lo..=bin_hi].iter().enumerate() {
                        row[bi] = c.norm() * inv_wsum;
                    }
                }
            },
        );
    (n_frames, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_poly_matches_scipy() {
        let x: Vec<f32> = (0..1000)
            .map(|i| ((i as f64) * 50.0 / 999.0).sin() as f32)
            .collect();
        let y1 = resample_poly(&x, 1, 2);
        assert_eq!(y1.len(), 500);
        let exp1 = [0.00779524, 0.09857585, 0.19916433, 0.29547548, 0.38987634];
        for (a, b) in y1[..5].iter().zip(exp1.iter()) {
            assert!((a - b).abs() < 1e-5, "dec2x diff: {a} vs {b}");
        }

        let y1_fast = decimate_2x(&x);
        assert_eq!(y1_fast.len(), y1.len());
        for (i, (&a, &b)) in y1_fast.iter().zip(y1.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "idx {i}: {a} vs {b}");
        }
        let x_odd: Vec<f32> = x[..999].to_vec();
        let y_odd_ref = resample_poly(&x_odd, 1, 2);
        let y_odd_fast = decimate_2x(&x_odd);
        assert_eq!(y_odd_fast.len(), y_odd_ref.len());
        for (i, (&a, &b)) in y_odd_fast.iter().zip(y_odd_ref.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "odd idx {i}: {a} vs {b}");
        }

        let y2 = resample_poly(&x, 1001, 960);
        assert_eq!(y2.len(), 1043);
        let exp2 = [3.5e-19, 4.75576e-2, 9.622745e-2, 0.14300504, 0.19104581];
        for (a, b) in y2[..5].iter().zip(exp2.iter()) {
            assert!((a - b).abs() < 1e-5, "up1001 diff: {a} vs {b}");
        }
    }

    #[test]
    fn test_correlate_matches_direct() {
        let mut seed = 123456789u64;
        let r: Vec<f32> = (0..300)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 32) as f32) / (u32::MAX as f32) - 0.5
            })
            .collect();
        let s: Vec<f32> = r[42..142].to_vec();
        let cache = RefFftCache::new(&r, r.len());
        let (k, conf) = cache.peak_valid(&s);
        assert_eq!(k, 42);
        assert!(conf > 5.0, "conf={conf}");
        let cache_full = RefFftCache::new(&r, r.len() + s.len() - 1);
        assert_eq!(cache_full.global_offset(&s), 42);

        // Negative lag test: s starts with r[0..] shifted earlier
        let r2: Vec<f32> = s[30..].to_vec();
        let cache_full2 = RefFftCache::new(&r2, r2.len() + s.len() - 1);
        assert_eq!(cache_full2.global_offset(&s), -30);

        // Also test SmallCorrPlan against direct brute-force sum
        let plan = SmallCorrPlan::new(r.len());
        let mut sc = plan.make_scratch();
        let (k2, conf2) = plan.peak(&mut sc, &r, &s);
        assert_eq!(k2, 42);
        assert!((conf - conf2).abs() < 1e-4, "{conf} vs {conf2}");
        let direct_max: f64 = (0..=r.len() - s.len())
            .map(|lag| {
                r[lag..lag + s.len()]
                    .iter()
                    .zip(s.iter())
                    .map(|(&a, &b)| (a as f64) * (b as f64))
                    .sum::<f64>()
                    .abs()
            })
            .fold(0.0f64, f64::max);
        let fft_max = plan.max_abs_valid(&mut sc, &r, &s);
        assert!((direct_max - fft_max).abs() < 1e-3, "{direct_max} vs {fft_max}");
    }
}
