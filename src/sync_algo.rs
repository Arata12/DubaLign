use crate::dsp::{
    butter_bandpass_sosfilt, correlate_full_signed, mean_std_f32, median_f64, polyfit_linear,
    stft_hann_mag_band, stretch, RefFftCache, RefFftScratch, SmallCorrPlan,
};
use crate::media;
use crate::model::{
    round1, round2, round3, round7, AudioPlanEntry, AudioReport, MeasurePoint, Plan, ProbeInfo,
    Report, Segment, SubPlanEntry, SubReport, TrackInfo,
};
use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use realfft::RealFftPlanner;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use tempfile::NamedTempFile;

pub const SR: usize = 8000;
pub const ENV_SR: usize = 200;
pub const TOL: f64 = 0.040;
pub const THRESHOLD_MS: i64 = 40;
pub const EDGE_MIN: f64 = 0.150;
pub const SUB_THRESHOLD_MS: i64 = 400;
pub const SUB_MIN_CONF: f64 = 0.35;
pub const FR: usize = 50;

pub const FILM: f64 = 24000.0 / 1001.0;
pub const TEMPOS: [f64; 7] = [
    1.0,
    FILM / 25.0,
    25.0 / FILM,
    24.0 / 25.0,
    25.0 / 24.0,
    1000.0 / 1001.0,
    1001.0 / 1000.0,
];

fn match_standard_fps(fps: f64, tol: f64) -> Option<f64> {
    const STD: &[(f64, f64)] = &[
        (24000.0 / 1001.0, 23.976),
        (24.0, 24.0),
        (25.0, 25.0),
        (30000.0 / 1001.0, 29.97),
        (30.0, 30.0),
        (50.0, 50.0),
        (60000.0 / 1001.0, 59.94),
        (60.0, 60.0),
    ];
    for &(exact, label) in STD {
        if (fps - exact).abs() <= tol {
            return Some(label);
        }
    }
    None
}

/// Snaps a frame rate to standard broadcast/cinema values (`23.976`, `24`, `25`, `29.97`, `30`, `50`, `59.94`, `60`) when within `0.005` fps.
pub fn snap_fps(fps: f64) -> f64 {
    match_standard_fps(fps, 0.005).unwrap_or_else(|| round3(fps))
}

fn exact_fps(fps: f64) -> f64 {
    if (fps - 23.976).abs() < 0.005 {
        24000.0 / 1001.0
    } else if (fps - 29.97).abs() < 0.005 {
        30000.0 / 1001.0
    } else if (fps - 59.94).abs() < 0.005 {
        60000.0 / 1001.0
    } else {
        fps
    }
}

/// Infers `(ref_fps, src_fps)` ONLY when `tempo` corresponds to a genuine difference between distinct standard FPS rates
/// (e.g. `23.976 <-> 25`, `24 <-> 25`, `23.976 <-> 24`, `29.97 <-> 30`).
/// Returns `(None, None)` when both videos share the same FPS (e.g. non-FPS capture clock drift), never returning equal FPS pairs.
pub fn infer_fps_pair(video_fps: Option<f64>, tempo: f64) -> (Option<f64>, Option<f64>) {
    // Smallest real FPS conversion is 1000/1001 (~0.001 = 1000 ppm)
    if tempo <= 0.05 || (tempo - 1.0).abs() < 0.0007 {
        return (None, None);
    }
    if let Some(vf) = video_fps.filter(|&f| f > 1.0) {
        let vf_ex = exact_fps(vf);
        let r_fps = snap_fps(vf_ex);
        let raw_src = vf_ex / tempo;
        if let Some(s_fps) = match_standard_fps(raw_src, 0.008) {
            if (r_fps - s_fps).abs() >= 0.01 {
                return (Some(r_fps), Some(s_fps));
            }
        }
        return (None, None);
    }
    const PAIRS: &[(f64, f64, f64, f64)] = &[
        (FILM / 25.0, 23.976, 25.0, 0.0003),
        (25.0 / FILM, 25.0, 23.976, 0.0003),
        (24.0 / 25.0, 24.0, 25.0, 0.0003),
        (25.0 / 24.0, 25.0, 24.0, 0.0003),
        (1000.0 / 1001.0, 23.976, 24.0, 0.0002),
        (1001.0 / 1000.0, 24.0, 23.976, 0.0002),
    ];
    for &(ratio, r_fps, s_fps, tol) in PAIRS {
        if (tempo - ratio).abs() < tol {
            return (Some(r_fps), Some(s_fps));
        }
    }
    (None, None)
}

/// Pitch difference in semitones of a varispeed-converted source relative to reference (`+0.72` for PAL 25 fps speedup).
pub fn pitch_semitones_from_tempo(tempo: f64) -> f64 {
    if tempo <= 0.0 {
        return 0.0;
    }
    round2(12.0 * (1.0 / tempo).log2())
}

/// Populates missing `ref_fps`, `src_fps`, and `pitch_semitones` fields on `tempo` tracks in `rep`,
/// and cleans up any legacy false-positive micro-tempo (`|tempo - 1| < 0.0005` with `< 180 ms` span drift) in cached reports.
pub fn populate_fps_info(rep: &mut Report) -> bool {
    let mut changed = false;
    if let Some(vf) = rep.video_fps {
        let snapped = snap_fps(vf);
        if (vf - snapped).abs() > 1e-6 {
            rep.video_fps = Some(snapped);
            changed = true;
        }
    }
    for a in &mut rep.audio {
        if a.status == "tempo" && (a.tempo - 1.0).abs() < 0.0005 && a.points.len() >= 3 {
            let pts: Vec<(f64, f64, f64)> = a
                .points
                .iter()
                .map(|p| (p.0, (p.1 as f64) / 1000.0, p.2))
                .collect();
            let (d_sec, tp_new, kind_new, _) = diagnose(&pts);
            if kind_new != "drift" || (tp_new - 1.0).abs() < 1e-12 {
                let med_ms = (d_sec * 1000.0).round() as i64;
                a.tempo = 1.0;
                a.delay_ms = Some(med_ms);
                a.status = if med_ms.abs() < THRESHOLD_MS {
                    "ok".to_string()
                } else {
                    "offset".to_string()
                };
                a.ref_fps = None;
                a.src_fps = None;
                a.pitch_shift = None;
                a.pitch_semitones = None;
                changed = true;
                continue;
            }
        }
        if a.status == "tempo" && (a.tempo - 1.0).abs() > 1e-9 {
            let (rf, sf) = infer_fps_pair(rep.video_fps, a.tempo);
            if a.ref_fps != rf || a.src_fps != sf {
                a.ref_fps = rf;
                a.src_fps = sf;
                changed = true;
            }
            let semi = Some(pitch_semitones_from_tempo(a.tempo));
            if a.pitch_semitones != semi {
                a.pitch_semitones = semi;
                changed = true;
            }
        }
    }
    if changed {
        apply_sub_twin(&rep.audio, &mut rep.subs);
        rep.status = recompute_worst_status(&rep.audio, &rep.subs);
    }
    changed
}

/// Computes a 200 Hz (`ENV_SR`) zero-mean log-energy modulation envelope invariant to musical pitch.
pub fn onset_envelope(x: &[f32]) -> Vec<f32> {
    let hop = SR / ENV_SR; // 40 samples = 5 ms
    let n = x.len() / hop;
    if n < 8 {
        return vec![];
    }
    let mut log_e = Vec::with_capacity(n);
    let mut prev = x[0] as f64;
    for chunk in x[..n * hop].chunks_exact(hop) {
        let mut sum_sq = 0.0f64;
        for &s in chunk {
            let cur = s as f64;
            let hp = cur - 0.95 * prev;
            prev = cur;
            sum_sq += hp * hp;
        }
        log_e.push(10.0 * (sum_sq / (hop as f64) + 1e-9).log10());
    }
    // Subtract local 41-frame (±100 ms) moving average to isolate 2..100 Hz syllable/beat modulations
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(0.0f64);
    let mut acc = 0.0f64;
    for &v in &log_e {
        acc += v;
        prefix.push(acc);
    }
    let half_w = 20usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let lo = i.saturating_sub(half_w);
        let hi = (i + half_w + 1).min(n);
        let local_mean = (prefix[hi] - prefix[lo]) / ((hi - lo) as f64);
        out.push((log_e[i] - local_mean) as f32);
    }
    out
}

pub fn peak_single(ref_seg: &[f32], src_seg: &[f32]) -> (usize, f64) {
    let plan = SmallCorrPlan::new(ref_seg.len());
    let mut sc = plan.make_scratch();
    plan.peak(&mut sc, ref_seg, src_seg)
}

pub fn detect_tempo_scores(
    ref_cache: &RefFftCache,
    sc: &mut RefFftScratch,
    ref_len: usize,
    src: &[f32],
    sr: usize,
) -> (f64, f64) {
    let n = (60 * sr).min(src.len() / 3);
    let half = n / 2;
    let mid = src.len() / 2;
    if half == 0 || mid < half || mid + half > src.len() {
        return (1.0, 0.0);
    }
    let chunk = &src[mid - half..mid + half];

    let stretched: Vec<(f64, Vec<f32>)> = TEMPOS
        .par_iter()
        .map(|&tp| (tp, stretch(chunk, tp)))
        .collect();

    let mut scores: Vec<(f64, f64)> = Vec::with_capacity(TEMPOS.len());
    for (tp, seg) in stretched {
        if seg.len() < ref_len {
            let (_, conf) = ref_cache.peak_valid_with_scratch(sc, &seg);
            scores.push((tp, conf));
        }
    }

    if scores.is_empty() {
        return (1.0, 0.0);
    }
    let score_1 = scores
        .iter()
        .find(|&&(tp, _)| (tp - 1.0).abs() < 1e-12)
        .map(|&(_, c)| c)
        .unwrap_or(0.0);
    let (best_tp, best_conf) = scores
        .iter()
        .copied()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    let chosen = if best_conf > 1.5 * score_1 && best_conf >= 7.0 {
        best_tp
    } else {
        1.0
    };
    (chosen, score_1)
}

pub fn detect_tempo_with_scratch(
    ref_cache: &RefFftCache,
    sc: &mut RefFftScratch,
    ref_len: usize,
    src: &[f32],
) -> f64 {
    detect_tempo_scores(ref_cache, sc, ref_len, src, SR).0
}

pub fn detect_tempo(ref_cache: &RefFftCache, ref_len: usize, src: &[f32]) -> f64 {
    let mut sc = ref_cache.make_scratch();
    detect_tempo_with_scratch(ref_cache, &mut sc, ref_len, src)
}

/// Pitch-invariant tempo detection on `ENV_SR` (`200 Hz`) onset envelopes (catches pitch-locked 25<->23.976 conversions).
pub fn detect_tempo_envelope(ref_env: &[f32], src_env: &[f32]) -> f64 {
    if ref_env.len() < ENV_SR * 15 || src_env.len() < ENV_SR * 15 {
        return 1.0;
    }
    let cache = RefFftCache::new(ref_env, ref_env.len());
    let mut sc = cache.make_scratch();
    let (tp, _) = detect_tempo_scores(&cache, &mut sc, ref_env.len(), src_env, ENV_SR);
    tp
}

pub fn scan(
    ref_audio: &[f32],
    src: &[f32],
    g: isize,
    win: f64,
    step: f64,
    search: f64,
    min_conf: f64,
    t0: f64,
    t1: Option<f64>,
) -> Vec<(f64, f64, f64)> {
    let l = (win * SR as f64) as usize;
    let s = (search * SR as f64) as usize;
    let step_samples = ((step * SR as f64) as usize).max(1);
    let max_stop = src.len().saturating_sub(l);
    let stop = match t1 {
        None => max_stop,
        Some(v) => ((v.max(0.0) * SR as f64) as usize).min(max_stop),
    };
    let start = (t0.max(0.0) * SR as f64) as usize;
    if start >= stop || l == 0 {
        return vec![];
    }

    let ts: Vec<usize> = (start..stop).step_by(step_samples).collect();
    let plan = SmallCorrPlan::new(l + 2 * s);

    ts.par_iter()
        .map_init(
            || plan.make_scratch(),
            |sc, &t| {
                let seg = &src[t..t + l];
                let lo = (t as isize + g - s as isize).max(0) as usize;
                let hi = ((t as isize + g + l as isize + s as isize).max(0) as usize)
                    .min(ref_audio.len());
                if mean_std_f32(seg).1 < 1e-3 || hi < lo + l + SR {
                    return None;
                }
                let (k, conf) = plan.peak(sc, &ref_audio[lo..hi], seg);
                if conf >= min_conf {
                    let t_sec = (t as f64) / (SR as f64);
                    let delay = ((lo as isize + k as isize - t as isize) as f64) / (SR as f64);
                    Some((t_sec, delay, conf))
                } else {
                    None
                }
            },
        )
        .filter_map(|x| x)
        .collect()
}

pub fn scan_env(
    ref_env: &[f32],
    src_env: &[f32],
    g: isize,
    win: f64,
    step: f64,
    search: f64,
    min_conf: f64,
) -> Vec<(f64, f64, f64)> {
    let l = (win * ENV_SR as f64) as usize;
    let s = (search * ENV_SR as f64) as usize;
    let step_samples = ((step * ENV_SR as f64) as usize).max(1);
    let stop = src_env.len().saturating_sub(l);
    if stop == 0 || l == 0 {
        return vec![];
    }
    let ts: Vec<usize> = (0..stop).step_by(step_samples).collect();
    let plan = SmallCorrPlan::new(l + 2 * s);

    ts.par_iter()
        .map_init(
            || plan.make_scratch(),
            |sc, &t| {
                let seg = &src_env[t..t + l];
                let lo = (t as isize + g - s as isize).max(0) as usize;
                let hi = ((t as isize + g + l as isize + s as isize).max(0) as usize)
                    .min(ref_env.len());
                if mean_std_f32(seg).1 < 1e-2 || hi < lo + l + ENV_SR {
                    return None;
                }
                let (k_sub, conf) = plan.peak_subpixel(sc, &ref_env[lo..hi], seg);
                if conf >= min_conf {
                    let t_sec = (t as f64) / (ENV_SR as f64);
                    let delay = ((lo as f64) + k_sub - (t as f64)) / (ENV_SR as f64);
                    Some((t_sec, delay, conf))
                } else {
                    None
                }
            },
        )
        .filter_map(|x| x)
        .collect()
}

/// Determines whether a `tempo`-mismatched track also has shifted musical pitch (`true` = varispeed, e.g. PAL +0.72 semitones)
/// vs already matching the reference's pitch (`false` = pitch-locked time-stretch or <0.05-semitone micro-drift).
pub fn detect_pitch_shifted(
    ref_audio: &[f32],
    src_in: &[f32],
    tempo: f64,
    pts: &[(f64, f64, f64)],
    via_envelope: bool,
) -> bool {
    // Below ~0.043 semitones (e.g. 23.976 <-> 24.0 is 0.017 semitones = 1.7 cents), pitch difference is imperceptible
    if (tempo - 1.0).abs() < 0.0025 {
        return false;
    }
    if pts.is_empty() {
        return !via_envelope;
    }

    let mut ranked = pts.to_vec();
    ranked.sort_by(|a, b| b.2.total_cmp(&a.2));
    ranked.truncate(8);

    const NFFT: usize = 2048; // 3.9 Hz/bin at 8 kHz
    const BIN_LO: usize = 50; // ~195 Hz
    const BIN_HI: usize = 768; // 3000 Hz
    const NBINS: usize = BIN_HI - BIN_LO + 1;

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(NFFT);
    let mut fft_in = r2c.make_input_vec();
    let mut fft_out = r2c.make_output_vec();
    let mut fft_sc = r2c.make_scratch_vec();
    let win: Vec<f32> = (0..NFFT)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f64) / (NFFT as f64)).cos()) as f32)
        .collect();

    let mut whiten = |sig: &[f32], center_sec: f64, dst: &mut [f64; NBINS]| -> bool {
        let start_i = ((center_sec * SR as f64).round() as isize) - (NFFT as isize / 2);
        if start_i < 0 || (start_i as usize) + NFFT > sig.len() {
            return false;
        }
        let sl = &sig[start_i as usize..start_i as usize + NFFT];
        if mean_std_f32(sl).1 < 2e-3 {
            return false;
        }
        for i in 0..NFFT {
            fft_in[i] = sl[i] * win[i];
        }
        if r2c
            .process_with_scratch(&mut fft_in, &mut fft_out, &mut fft_sc)
            .is_err()
        {
            return false;
        }
        let mut raw_log = [0.0f64; NBINS];
        let mut pref = [0.0f64; NBINS + 1];
        for b in 0..NBINS {
            let v = (fft_out[BIN_LO + b].norm() as f64 + 1e-6).ln();
            raw_log[b] = v;
            pref[b + 1] = pref[b] + v;
        }
        let mut max_peak = 0.0f64;
        let mut sum_sq = 0.0f64;
        for b in 0..NBINS {
            let lo = b.saturating_sub(7);
            let hi = (b + 8).min(NBINS);
            let wv = raw_log[b] - (pref[hi] - pref[lo]) / ((hi - lo) as f64);
            dst[b] = wv;
            if wv > max_peak {
                max_peak = wv;
            }
            sum_sq += wv * wv;
        }
        if max_peak < 0.35 || sum_sq < 1e-9 {
            return false;
        }
        let inv_norm = 1.0 / sum_sq.sqrt();
        for v in dst.iter_mut() {
            *v *= inv_norm;
        }
        true
    };

    let mut sum_vari = 0.0f64;
    let mut sum_raw = 0.0f64;
    let mut count = 0usize;
    let mut w_ref = [0.0f64; NBINS];
    let mut w_vari = [0.0f64; NBINS];
    let mut w_raw = [0.0f64; NBINS];

    for &(t_str, d_str, _) in &ranked {
        let s0 = ((t_str.max(0.0) * tempo * SR as f64) as usize).min(src_in.len());
        let s1 = (((t_str.max(0.0) + 4.2) * tempo * SR as f64).ceil() as usize).min(src_in.len());
        if s1 <= s0 + NFFT {
            continue;
        }
        let local_vari = stretch(&src_in[s0..s1], tempo);
        let t_base = (s0 as f64) / ((SR as f64) * tempo);

        for k in 0..12 {
            let dt = 0.35 + (k as f64) * 0.30;
            let t_actual_str = t_base + dt;
            let t_r = t_actual_str + d_str;
            let t_s = t_actual_str * tempo;
            if whiten(ref_audio, t_r, &mut w_ref)
                && whiten(&local_vari, dt, &mut w_vari)
                && whiten(src_in, t_s, &mut w_raw)
            {
                let mut dv = 0.0f64;
                let mut dr = 0.0f64;
                for b in 0..NBINS {
                    dv += w_ref[b] * w_vari[b];
                    dr += w_ref[b] * w_raw[b];
                }
                sum_vari += dv;
                sum_raw += dr;
                count += 1;
            }
        }
    }

    if count >= 3 {
        let mv = sum_vari / (count as f64);
        let mr = sum_raw / (count as f64);
        if (mv - mr).abs() >= 0.03 {
            return mv > mr;
        }
    }
    !via_envelope
}

pub fn ncc(ref_audio: &[f32], src: &[f32], t: f64, d: f64, win: f64, slack: f64) -> f64 {
    if t < 0.0 {
        return 0.0;
    }
    let a = (t * SR as f64) as usize;
    let l = (win * SR as f64) as usize;
    let s = (slack * SR as f64) as usize;
    if a + l > src.len() {
        return 0.0;
    }
    let w = &src[a..a + l];
    let lo_raw = a as isize + ((d * SR as f64) as isize) - s as isize;
    let lo = lo_raw.max(0) as usize;
    let hi_raw = lo_raw + l as isize + 2 * s as isize;
    if hi_raw <= 0 {
        return 0.0;
    }
    let hi = (hi_raw as usize).min(ref_audio.len());
    if hi <= lo {
        return 0.0;
    }
    let r = &ref_audio[lo..hi];
    if w.len() < l || r.len() < l || mean_std_f32(w).1 < 1e-4 {
        return 0.0;
    }
    let plan = SmallCorrPlan::new(r.len());
    let mut sc = plan.make_scratch();
    let c = plan.max_abs_valid(&mut sc, r, w);
    let norm_w: f64 = w.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
    let norm_r: f64 = r.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
    c / (norm_w * norm_r + 1e-9)
}

fn quiet_point(src: &[f32], lo: f64, hi: f64, length: f64) -> f64 {
    let a = (lo.max(0.0) * SR as f64) as usize;
    let b = ((hi.max(0.0) * SR as f64) as usize).min(src.len());
    let n = ((length * SR as f64) as usize).max(SR / 20);
    if b <= a || b - a <= n {
        return lo;
    }
    let len = b - a;
    let mut cum = Vec::with_capacity(len + 1);
    cum.push(0.0f64);
    let mut acc = 0.0f64;
    for &v in &src[a..b] {
        acc += (v as f64) * (v as f64);
        cum.push(acc);
    }
    let mut best_i = 0usize;
    let mut best_e = f64::INFINITY;
    for i in 0..=(len - n) {
        let e = cum[i + n] - cum[i];
        if e < best_e {
            best_e = e;
            best_i = i;
        }
    }
    lo.max(0.0) + (best_i as f64) / (SR as f64)
}

fn level_db(src: &[f32], t: f64, length: f64) -> f64 {
    let a = (t.max(0.0) * SR as f64) as usize;
    let n = ((length * SR as f64) as usize).max(1);
    let b = (a + n).min(src.len());
    if b <= a {
        return 0.0;
    }
    let mean_sq = src[a..b]
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        / ((b - a) as f64);
    20.0 * (mean_sq.sqrt() + 1e-9).log10()
}

fn cut_point(ref_audio: &[f32], src: &[f32], d_a: f64, d_b: f64, t_lo: f64, t_hi: f64) -> f64 {
    let mut ts = Vec::new();
    let mut cur = t_lo.max(0.0);
    while cur < t_hi {
        ts.push(cur);
        cur += 0.5;
    }
    if ts.is_empty() {
        return t_lo.max(0.0);
    }
    let is_b: Vec<bool> = ts
        .par_iter()
        .map(|&t| ncc(ref_audio, src, t, d_b, 2.0, 0.02) > ncc(ref_audio, src, t, d_a, 2.0, 0.02))
        .collect();

    let total_b: usize = is_b.iter().filter(|&&b| b).count();
    let mut not_b_before = 0usize;
    let mut b_before = 0usize;
    let mut best_j = 0usize;
    let mut best_score = usize::MIN;
    for j in 0..=ts.len() {
        let b_after = total_b - b_before;
        let score = not_b_before + b_after;
        if j == 0 || score > best_score {
            best_score = score;
            best_j = j;
        }
        if j < ts.len() {
            if is_b[j] {
                b_before += 1;
            } else {
                not_b_before += 1;
            }
        }
    }
    let tc = ts[best_j.min(ts.len() - 1)] + 1.0;
    let gap = (d_b - d_a).abs();
    let length = if d_b < d_a { gap } else { 0.05 };
    let mut x = quiet_point(src, tc - 1.0, tc + 1.0, length);
    let x3 = quiet_point(src, tc - 3.0, tc + 3.0, length);
    if level_db(src, x3, length) < -90.0 && level_db(src, x, length) >= -90.0 {
        x = x3;
    }
    x + if d_b < d_a { gap } else { 0.025 }
}

pub fn robust(resid: &[f64], tol: f64) -> (bool, Vec<bool>) {
    let bad: Vec<bool> = resid.iter().map(|&r| r.abs() >= tol).collect();
    let bad_cnt = bad.iter().filter(|&&b| b).count();
    if (bad_cnt as f64) / (resid.len() as f64) > 0.2 {
        return (false, bad);
    }
    let bad_idx: Vec<usize> = bad
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i) } else { None })
        .collect();
    let step = bad_idx
        .windows(2)
        .any(|w| w[1] - w[0] == 1 && (resid[w[1]] - resid[w[0]]).abs() < tol);
    (!step, bad)
}

pub fn coverage_ok(points: &[(f64, f64, f64)], duration: f64) -> bool {
    if duration <= 0.0 {
        return false;
    }
    let mut thirds = [false; 3];
    for &(t, _, _) in points {
        let idx = ((3.0 * t / duration).floor() as isize).clamp(0, 2) as usize;
        thirds[idx] = true;
    }
    thirds.iter().all(|&x| x)
}

pub fn diagnose(points: &[(f64, f64, f64)]) -> (f64, f64, &'static str, usize) {
    let t: Vec<f64> = points.iter().map(|p| p.0).collect();
    let d: Vec<f64> = points.iter().map(|p| p.1).collect();
    let med = median_f64(&d);
    let resid1: Vec<f64> = d.iter().map(|&v| v - med).collect();
    let (ok1, bad1) = robust(&resid1, 0.040);
    if ok1 {
        let good_d: Vec<f64> = d
            .iter()
            .zip(bad1.iter())
            .filter_map(|(&v, &b)| if !b { Some(v) } else { None })
            .collect();
        let dropped = bad1.iter().filter(|&&b| b).count();
        return (median_f64(&good_d), 1.0, "constante", dropped);
    }
    let (b, a) = polyfit_linear(&t, &d);
    let resid2: Vec<f64> = t
        .iter()
        .zip(d.iter())
        .map(|(&ti, &di)| di - (a + b * ti))
        .collect();
    let (ok2, bad2) = robust(&resid2, 0.040);
    if ok2 {
        let gt: Vec<f64> = t
            .iter()
            .zip(bad2.iter())
            .filter_map(|(&v, &bad)| if !bad { Some(v) } else { None })
            .collect();
        let gd: Vec<f64> = d
            .iter()
            .zip(bad2.iter())
            .filter_map(|(&v, &bad)| if !bad { Some(v) } else { None })
            .collect();
        if gt.len() >= 3 {
            let (b2, a2) = polyfit_linear(&gt, &gd);
            let span_t = gt.last().copied().unwrap_or(0.0) - gt.first().copied().unwrap_or(0.0);
            let total_drift = (b2 * span_t).abs();
            let gd_med = median_f64(&gd);
            let rms_const = (gd.iter().map(|&v| (v - gd_med).powi(2)).sum::<f64>()
                / (gd.len() as f64))
                .sqrt();
            let rms_line = (gt
                .iter()
                .zip(gd.iter())
                .map(|(&ti, &di)| (di - (a2 + b2 * ti)).powi(2))
                .sum::<f64>()
                / (gd.len() as f64))
                .sqrt();
            // Only classify as continuous speed drift if accumulated drift >= 180 ms AND linear fit markedly beats constant
            if total_drift >= 0.180 && rms_line < 0.45 * rms_const.max(1e-9) {
                let dropped = bad2.iter().filter(|&&bad| bad).count();
                return (a2, 1.0 / (1.0 + b2), "drift", dropped);
            }
        }
    }
    (med, 1.0, "cortes", 0)
}

fn edge_jump(
    ref_audio: &[f32],
    src: &[f32],
    d_main: f64,
    side: &str,
    span: f64,
) -> Option<(f64, f64)> {
    let n = (src.len() as f64) / (SR as f64);
    let (t0, t1) = if side == "tail" {
        ((n - span).max(0.0), None)
    } else {
        (0.0, Some(span.min(n)))
    };
    let pts = scan(
        ref_audio,
        src,
        (d_main * SR as f64) as isize,
        4.0,
        2.0,
        15.0,
        8.0,
        t0,
        t1,
    );
    let other: Vec<(f64, f64, f64)> = pts
        .iter()
        .copied()
        .filter(|q| (q.1 - d_main).abs() >= TOL)
        .collect();
    if other.len() < 3 {
        return None;
    }
    let best = other
        .iter()
        .copied()
        .max_by_key(|q| other.iter().filter(|o| (o.1 - q.1).abs() < 0.02).count())?;
    let cluster: Vec<(f64, f64, f64)> = other
        .iter()
        .copied()
        .filter(|o| (o.1 - best.1).abs() < 0.02)
        .collect();
    if cluster.len() < 3 {
        return None;
    }
    let cluster_d: Vec<f64> = cluster.iter().map(|o| o.1).collect();
    let d_edge = median_f64(&cluster_d);
    if (d_edge - d_main).abs() < EDGE_MIN {
        return None;
    }
    let main_t: Vec<f64> = pts
        .iter()
        .filter(|q| (q.1 - d_main).abs() < TOL)
        .map(|q| q.0)
        .collect();
    let (cut, a, b) = if side == "tail" {
        let tc = cluster
            .iter()
            .map(|o| o.0)
            .fold(f64::INFINITY, f64::min);
        if main_t.iter().any(|&mt| mt > tc) {
            return None;
        }
        let cut = cut_point(ref_audio, src, d_main, d_edge, tc - 14.0, tc + 4.0);
        (cut, cut, n)
    } else {
        let tc = cluster
            .iter()
            .map(|o| o.0)
            .fold(f64::NEG_INFINITY, f64::max);
        if main_t.iter().any(|&mt| mt < tc) {
            return None;
        }
        let cut = cut_point(ref_audio, src, d_edge, d_main, tc - 4.0, tc + 14.0);
        (cut, 0.0, cut)
    };

    let mut ts = Vec::new();
    let mut cur = a;
    while cur < b - 2.0 {
        ts.push(cur);
        cur += 1.0;
    }
    if ts.len() < 3 {
        return None;
    }
    let em: Vec<(f64, f64)> = ts
        .par_iter()
        .map(|&t| {
            (
                ncc(ref_audio, src, t, d_edge, 2.0, 0.02),
                ncc(ref_audio, src, t, d_main, 2.0, 0.02),
            )
        })
        .collect();
    let e_vals: Vec<f64> = em.iter().map(|p| p.0).collect();
    let m_vals: Vec<f64> = em.iter().map(|p| p.1).collect();
    let win_frac = (em.iter().filter(|&&(ev, mv)| ev > mv + 0.05).count() as f64)
        / (em.len() as f64);
    if win_frac < 0.3 || median_f64(&e_vals) < 1.5 * median_f64(&m_vals) {
        return None;
    }
    Some((cut, d_edge))
}

fn add_edges(ref_audio: &[f32], src: &[f32], segments: &[Segment]) -> Vec<Segment> {
    let mut segs = segments.to_vec();
    if let Some((cut, d_edge)) = edge_jump(ref_audio, src, segs[0].delay, "head", 60.0) {
        if cut < segs[0].end.unwrap_or(1e9) {
            let s0 = round3(cut);
            segs[0].start = s0;
            segs.insert(
                0,
                Segment {
                    start: 0.0,
                    end: Some(s0),
                    delay: round3(d_edge),
                },
            );
        }
    }
    let last_idx = segs.len() - 1;
    if let Some((cut, d_edge)) = edge_jump(ref_audio, src, segs[last_idx].delay, "tail", 60.0) {
        if cut > segs[last_idx].start {
            let e_last = round3(cut);
            segs[last_idx].end = Some(e_last);
            segs.push(Segment {
                start: e_last,
                end: None,
                delay: round3(d_edge),
            });
        }
    }
    segs
}

fn find_segments(ref_audio: &[f32], src: &[f32], g: isize) -> Vec<Segment> {
    let pts = scan(ref_audio, src, g, 10.0, 10.0, 15.0, 10.0, 0.0, None);
    let mut groups: Vec<Vec<(f64, f64, f64)>> = Vec::new();
    let mut i = 0usize;
    while i < pts.len() {
        let p = pts[i];
        let matches_last = groups.last().is_some_and(|grp| {
            let ds: Vec<f64> = grp.iter().map(|q| q.1).collect();
            (p.1 - median_f64(&ds)).abs() < TOL
        });
        if matches_last {
            groups.last_mut().unwrap().push(p);
        } else if i + 1 < pts.len() && (pts[i + 1].1 - p.1).abs() < TOL {
            groups.push(vec![p]);
        }
        i += 1;
    }
    if groups.len() < 2 {
        return vec![];
    }
    let delays: Vec<f64> = groups
        .iter()
        .map(|grp| {
            let ds: Vec<f64> = grp.iter().map(|q| q.1).collect();
            median_f64(&ds)
        })
        .collect();
    let cuts: Vec<f64> = (0..groups.len() - 1)
        .map(|k| {
            cut_point(
                ref_audio,
                src,
                delays[k],
                delays[k + 1],
                groups[k].last().unwrap().0,
                groups[k + 1][0].0 + 8.0,
            )
        })
        .collect();
    let mut bounds = Vec::with_capacity(groups.len() + 1);
    bounds.push(Some(0.0));
    for c in cuts {
        bounds.push(Some(c));
    }
    bounds.push(None);

    (0..groups.len())
        .map(|k| Segment {
            start: round3(bounds[k].unwrap()),
            end: bounds[k + 1].map(round3),
            delay: round3(delays[k]),
        })
        .collect()
}

pub fn detect_cut_segments(ref_audio: &[f32], src: &[f32], g: isize) -> Vec<Segment> {
    let found = find_segments(ref_audio, src, g);
    if found.is_empty() {
        vec![]
    } else {
        add_edges(ref_audio, src, &found)
    }
}

pub fn with_edge_jumps(ref_audio: &[f32], src: &[f32], base_delay: f64) -> Vec<Segment> {
    let base = vec![Segment {
        start: 0.0,
        end: None,
        delay: round3(base_delay),
    }];
    let segs = add_edges(ref_audio, src, &base);
    if segs.len() > 1 {
        segs
    } else {
        vec![]
    }
}

pub fn dense_map(ref_audio: &[f32], src: &[f32], g: isize) -> Vec<MeasurePoint> {
    scan(ref_audio, src, g, 4.0, 2.0, 3.0, 8.0, 0.0, None)
        .into_iter()
        .map(|(t, d, c)| MeasurePoint(round1(t), (d * 1000.0).round() as i64, round1(c)))
        .collect()
}

pub fn dense_map_env(ref_env: &[f32], src_env: &[f32], g: isize) -> Vec<MeasurePoint> {
    scan_env(ref_env, src_env, g, 4.0, 2.0, 3.0, 8.0)
        .into_iter()
        .map(|(t, d, c)| MeasurePoint(round1(t), (d * 1000.0).round() as i64, round1(c)))
        .collect()
}

pub struct AudioAnalysis {
    pub status: String,
    pub delay_ms: Option<i64>,
    pub tempo: f64,
    pub pitch_shift: Option<bool>,
    pub points: Vec<MeasurePoint>,
    pub dense: Vec<MeasurePoint>,
    pub segments: Vec<Segment>,
    pub reason: Option<String>,
}

pub fn analyze_audio(
    ref_audio: &[f32],
    ref_tempo_cache: &RefFftCache,
    tempo_sc: &mut RefFftScratch,
    ref_full_cache: &RefFftCache,
    full_sc: &mut RefFftScratch,
    src_in: &[f32],
    windows: usize,
    window: f64,
    search: f64,
    min_conf: f64,
) -> AudioAnalysis {
    let (mut tempo0, score_1) =
        detect_tempo_scores(ref_tempo_cache, tempo_sc, ref_audio.len(), src_in, SR);
    let mut via_envelope = false;
    let mut env_pair: Option<(Vec<f32>, Vec<f32>)> = None;

    if (tempo0 - 1.0).abs() < 1e-12 && score_1 < 12.0 {
        let r_env = onset_envelope(ref_audio);
        let s_env = onset_envelope(src_in);
        let env_tp = detect_tempo_envelope(&r_env, &s_env);
        if (env_tp - 1.0).abs() > 1e-12 {
            tempo0 = env_tp;
            via_envelope = true;
        }
        env_pair = Some((r_env, s_env));
    }

    let stretched;
    let mut wave_src: &[f32] = src_in;
    let mut wave_g: isize = 0;

    let (pts, dense, stretched_dur_s) = if via_envelope {
        let (r_env, s_env) = env_pair.take().unwrap();
        let s_str = stretch(&s_env, tempo0);
        let g_env = if !r_env.is_empty() && !s_str.is_empty() {
            RefFftCache::new(&r_env, r_env.len() + s_str.len()).global_offset(&s_str)
        } else {
            0
        };
        let l_env = (window * ENV_SR as f64) as usize;
        let step_env = ((((s_str.len() as isize - l_env as isize).max(0) as f64) / ENV_SR as f64)
            / ((windows + 1) as f64))
            .max(1.0);
        let p = scan_env(&r_env, &s_str, g_env, window, step_env, search, min_conf);
        let dm = dense_map_env(&r_env, &s_str, g_env);
        let dur_s = (s_str.len() as f64) / (ENV_SR as f64);
        (p, dm, dur_s)
    } else {
        if (tempo0 - 1.0).abs() > 1e-12 {
            stretched = stretch(src_in, tempo0);
            wave_src = &stretched;
        }
        wave_g = if ref_audio.len() + wave_src.len() - 1 <= ref_full_cache.nfft {
            ref_full_cache.global_offset_with_scratch(full_sc, wave_src)
        } else {
            let tmp_cache = RefFftCache::new(ref_audio, ref_audio.len() + wave_src.len() - 1);
            tmp_cache.global_offset(wave_src)
        };
        let l = (window * SR as f64) as usize;
        let step = ((((wave_src.len() as isize - l as isize).max(0) as f64) / SR as f64)
            / ((windows + 1) as f64))
            .max(1.0);
        let mut p = scan(ref_audio, wave_src, wave_g, window, step, search, min_conf, 0.0, None);
        let mut dm = dense_map(ref_audio, wave_src, wave_g);
        let mut dur_s = (wave_src.len() as f64) / (SR as f64);

        // Fallback to pitch-invariant envelope when carrier frequencies differ at tempo=1.0 (e.g. pitch-shifted master)
        if p.len() < 3 && (tempo0 - 1.0).abs() < 1e-12 {
            let (r_env, s_env) = env_pair
                .take()
                .unwrap_or_else(|| (onset_envelope(ref_audio), onset_envelope(src_in)));
            if r_env.len() >= ENV_SR * 10 && s_env.len() >= ENV_SR * 10 {
                let g_env =
                    RefFftCache::new(&r_env, r_env.len() + s_env.len()).global_offset(&s_env);
                let l_env = (window * ENV_SR as f64) as usize;
                let step_env =
                    ((((s_env.len() as isize - l_env as isize).max(0) as f64) / ENV_SR as f64)
                        / ((windows + 1) as f64))
                        .max(1.0);
                let p_env = scan_env(&r_env, &s_env, g_env, window, step_env, search, min_conf);
                if p_env.len() >= 3 {
                    p = p_env;
                    dm = dense_map_env(&r_env, &s_env, g_env);
                    dur_s = (s_env.len() as f64) / (ENV_SR as f64);
                    via_envelope = true;
                }
            }
        }
        (p, dm, dur_s)
    };

    let points_json: Vec<MeasurePoint> = pts
        .iter()
        .map(|&(t, d, c)| MeasurePoint(round1(t), (d * 1000.0).round() as i64, round1(c)))
        .collect();

    if pts.len() < 3 {
        return AudioAnalysis {
            status: "unknown".to_string(),
            delay_ms: None,
            tempo: 1.0,
            pitch_shift: None,
            points: points_json,
            dense,
            segments: vec![],
            reason: Some(
                "casi nada coincide con la referencia: ¿es el mismo episodio?".to_string(),
            ),
        };
    }

    if !coverage_ok(&pts, stretched_dur_s) {
        return AudioAnalysis {
            status: "unknown".to_string(),
            delay_ms: None,
            tempo: 1.0,
            pitch_shift: None,
            points: points_json,
            dense,
            segments: vec![],
            reason: Some(
                "solo coincide en un tramo (¿opening/ending compartido?): probablemente otro episodio"
                    .to_string(),
            ),
        };
    }

    let (delay, mut tempo, kind, _) = diagnose(&pts);
    tempo *= tempo0;
    let delay_ms = (delay * 1000.0).round() as i64;

    if kind == "cortes" {
        if (tempo0 - 1.0).abs() > 1e-12 {
            // Standard FPS tempo0 already matched; refine residual slope if linear, else use median offset + tempo0
            let t_vec: Vec<f64> = pts.iter().map(|p| p.0).collect();
            let d_vec: Vec<f64> = pts.iter().map(|p| p.1).collect();
            let (b_r, a_r) = polyfit_linear(&t_vec, &d_vec);
            let res_r: Vec<f64> = t_vec
                .iter()
                .zip(d_vec.iter())
                .map(|(&ti, &di)| di - (a_r + b_r * ti))
                .collect();
            let (d_fin, tp_fin) = if robust(&res_r, 0.060).0 {
                ((a_r * 1000.0).round() as i64, round7(tempo0 / (1.0 + b_r)))
            } else {
                (delay_ms, round7(tempo0))
            };
            let pitch_shift = Some(detect_pitch_shifted(
                ref_audio,
                src_in,
                tp_fin,
                &pts,
                via_envelope,
            ));
            return AudioAnalysis {
                status: "tempo".to_string(),
                delay_ms: Some(d_fin),
                tempo: tp_fin,
                pitch_shift,
                points: points_json,
                dense,
                segments: vec![],
                reason: None,
            };
        }
        let segs = if !via_envelope {
            let found = find_segments(ref_audio, wave_src, wave_g);
            if found.is_empty() {
                vec![]
            } else {
                add_edges(ref_audio, wave_src, &found)
            }
        } else {
            vec![]
        };
        if !segs.is_empty()
            && segs
                .iter()
                .all(|s| ((s.delay * 1000.0).round() as i64).abs() < THRESHOLD_MS)
        {
            return AudioAnalysis {
                status: "ok".to_string(),
                delay_ms: Some(delay_ms),
                tempo: 1.0,
                pitch_shift: None,
                points: points_json,
                dense,
                segments: vec![],
                reason: None,
            };
        }
        if segs.is_empty() {
            // Check if points are simply constant with ~40-75 ms jitter (not multi-second unlocalizable cuts)
            let resid_med: Vec<f64> = pts.iter().map(|p| p.1 - delay).collect();
            if robust(&resid_med, 0.075).0 {
                let st = if delay_ms.abs() < THRESHOLD_MS {
                    "ok"
                } else {
                    "offset"
                };
                return AudioAnalysis {
                    status: st.to_string(),
                    delay_ms: Some(delay_ms),
                    tempo: 1.0,
                    pitch_shift: None,
                    points: points_json,
                    dense,
                    segments: vec![],
                    reason: None,
                };
            }
        }
        let status = if segs.is_empty() { "unknown" } else { "cuts" }.to_string();
        let reason = if segs.is_empty() {
            Some("el desfase cambia a saltos y no se pudieron localizar los cortes".to_string())
        } else {
            None
        };
        AudioAnalysis {
            status,
            delay_ms: Some(delay_ms),
            tempo: 1.0,
            pitch_shift: None,
            points: points_json,
            dense,
            segments: segs,
            reason,
        }
    } else if (tempo - 1.0).abs() > 1e-12 {
        let tp_round = round7(tempo);
        let pitch_shift = Some(detect_pitch_shifted(
            ref_audio,
            src_in,
            tp_round,
            &pts,
            via_envelope,
        ));
        AudioAnalysis {
            status: "tempo".to_string(),
            delay_ms: Some(delay_ms),
            tempo: tp_round,
            pitch_shift,
            points: points_json,
            dense,
            segments: vec![],
            reason: None,
        }
    } else {
        let mut status = if delay_ms.abs() < THRESHOLD_MS {
            "ok"
        } else {
            "offset"
        }
        .to_string();
        let out_segs = if via_envelope {
            vec![]
        } else {
            let base_seg = vec![Segment {
                start: 0.0,
                end: None,
                delay: round3(delay),
            }];
            let segs = add_edges(ref_audio, wave_src, &base_seg);
            if segs.len() > 1 {
                status = "cuts".to_string();
                segs
            } else {
                vec![]
            }
        };
        AudioAnalysis {
            status,
            delay_ms: Some(delay_ms),
            tempo: 1.0,
            pitch_shift: None,
            points: points_json,
            dense,
            segments: out_segs,
            reason: None,
        }
    }
}

// ---------------------------------------------------------------- Subtitles & Activity

pub fn speech_activity(x: &[f32]) -> Vec<f32> {
    let y = butter_bandpass_sosfilt(x);
    let hop = SR / FR;
    let n = y.len() / hop;
    if n == 0 {
        return vec![];
    }
    let mut e = Vec::with_capacity(n);
    for chunk in y[..n * hop].chunks_exact(hop) {
        let mean_sq: f64 = chunk.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / (hop as f64);
        e.push((10.0 * (mean_sq + 1e-10).log10()) as f32);
    }
    let e_f64: Vec<f64> = e.iter().map(|&v| v as f64).collect();
    let med = median_f64(&e_f64);
    let (_, std) = mean_std_f32(&e);
    e.into_iter()
        .map(|v| (((v as f64 - med) / (std + 1e-9)).clamp(-2.0, 2.0)) as f32)
        .collect()
}

pub fn voice_difference(ref_audio: &[f32], others: &[(&[f32], f64)]) -> Vec<f32> {
    let nperseg = 2 * SR / FR; // 320
    let hop = SR / FR;         // 160
    let bin_lo = 12usize;      // 300 Hz
    let bin_hi = 136usize;     // 3400 Hz
    let n_bins = bin_hi - bin_lo + 1;

    let (n_frames, a) = stft_hann_mag_band(ref_audio, nperseg, hop, bin_lo, bin_hi);
    if n_frames == 0 {
        return vec![];
    }
    let a_sum: Vec<f64> = a
        .par_chunks_exact(n_bins)
        .map(|row| row.iter().map(|&v| v as f64).sum())
        .collect();

    let mut acc = vec![0.0f64; n_frames];
    for &(src, delay) in others {
        let k = (delay * SR as f64).round() as isize;
        let mut y = vec![0.0f32; ref_audio.len()];
        if k >= 0 {
            let ku = k as usize;
            if ku < y.len() {
                let copy_len = src.len().min(y.len() - ku);
                y[ku..ku + copy_len].copy_from_slice(&src[..copy_len]);
            }
        } else {
            let skip = (-k) as usize;
            if skip < src.len() {
                let copy_len = (src.len() - skip).min(y.len());
                y[..copy_len].copy_from_slice(&src[skip..skip + copy_len]);
            }
        }
        let (_, b) = stft_hann_mag_band(&y, nperseg, hop, bin_lo, bin_hi);
        let contrib: Vec<f64> = (0..n_frames)
            .into_par_iter()
            .map(|t| {
                let ra = &a[t * n_bins..(t + 1) * n_bins];
                let rb = &b[t * n_bins..(t + 1) * n_bins];
                let mut diff_sum = 0.0f64;
                let mut b_sum = 0.0f64;
                for i in 0..n_bins {
                    diff_sum += (ra[i] - rb[i]).abs() as f64;
                    b_sum += rb[i] as f64;
                }
                diff_sum / (a_sum[t] + b_sum + 1e-3)
            })
            .collect();
        for (dst, val) in acc.iter_mut().zip(contrib.into_iter()) {
            *dst += val;
        }
    }

    let n_oth = others.len() as f64;
    for v in &mut acc {
        *v /= n_oth;
    }
    let med = median_f64(&acc);
    let mean = acc.iter().sum::<f64>() / (acc.len() as f64);
    let var = acc
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / (acc.len() as f64);
    let std = var.sqrt();
    acc.into_iter()
        .map(|v| (((v - med) / (std + 1e-9)).clamp(-2.0, 3.0)) as f32)
        .collect()
}

pub fn measure_subs(activity: &[f32], events: &[(f64, f64)], max_lag: f64) -> (f64, f64) {
    let n = activity.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut s = vec![0.0f32; n];
    for &(a, b) in events {
        let i0 = ((a * FR as f64).max(0.0) as usize).min(n);
        let i1 = ((b * FR as f64).max(0.0) as usize).min(n);
        for x in &mut s[i0..i1] {
            *x = 1.0;
        }
    }
    let s_sum: f64 = s.iter().map(|&v| v as f64).sum();
    let (a_mean, a_std) = mean_std_f32(activity);
    if s_sum < (FR * 20) as f64 || a_std == 0.0 {
        return (0.0, 0.0);
    }
    let (s_mean, s_std) = mean_std_f32(&s);
    let a0: Vec<f32> = activity.iter().map(|&v| (v as f64 - a_mean) as f32).collect();
    let s0: Vec<f32> = s.iter().map(|&v| (v as f64 - s_mean) as f32).collect();
    let c = correlate_full_signed(&a0, &s0);
    let mid = n - 1;
    let m = (max_lag * FR as f64) as usize;
    let lo = mid.saturating_sub(m);
    let hi = (mid + m + 1).min(c.len());
    let denom = (n as f64) * a_std * s_std;
    let mut best_k = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    for (idx, &val) in c[lo..hi].iter().enumerate() {
        let norm = (val as f64) / denom;
        if norm > best_val {
            best_val = norm;
            best_k = idx;
        }
    }
    let lag = ((best_k as isize) - ((mid - lo) as isize)) as f64 / (FR as f64);
    (lag, best_val)
}

pub fn transient(x: &[f32], sr: usize) -> f64 {
    let hop = sr / 100;
    let n = x.len() / hop;
    if n <= 31 {
        return 0.0;
    }
    let mut e = Vec::with_capacity(n);
    for chunk in x[..n * hop].chunks_exact(hop) {
        let mean_sq: f64 =
            chunk.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / (hop as f64);
        e.push(10.0 * (mean_sq + 1e-12).log10());
    }
    // running 30-sample average (convolution mode="full"[:len(e)])
    let mut quiet = vec![0.0f64; n];
    let mut acc = 0.0f64;
    for i in 0..n {
        acc += e[i];
        if i >= 30 {
            acc -= e[i - 30];
        }
        quiet[i] = acc / 30.0;
    }
    let mut best_i = 0usize;
    let mut best_jump = f64::NEG_INFINITY;
    for i in 31..n {
        let jump = e[i] - quiet[i - 1];
        if jump > best_jump {
            best_jump = jump;
            best_i = i;
        }
    }
    ((best_i * hop) as f64) / (sr as f64)
}

/// Analyzes file using already-decoded 8 kHz tracks and optional pre-extracted subtitle events.
pub fn analyze_file_with_tracks(
    path: &Path,
    info: &ProbeInfo,
    tracks_8k: &[Vec<f32>],
    ref_track: usize,
    preextracted_subs: Option<std::collections::HashMap<usize, Option<Vec<(f64, f64)>>>>,
    progress: &(dyn Fn(f64, &str) + Sync + Send),
) -> Result<Report> {
    let audios: Vec<&TrackInfo> = info.tracks.iter().filter(|t| t.kind == "audio").collect();
    let subs: Vec<&TrackInfo> = info
        .tracks
        .iter()
        .filter(|t| t.kind == "subtitles")
        .collect();
    let steps = (audios.len() + subs.len() + 1).max(1) as f64;
    let done = AtomicUsize::new(0);

    let tick = |msg: &str| {
        let d = done.fetch_add(1, Ordering::Relaxed);
        progress((d as f64) / steps, msg);
    };

    tick("leyendo audio de referencia");
    if ref_track >= tracks_8k.len() {
        bail!("pista de referencia {ref_track} no existe");
    }
    let ref_audio = &tracks_8k[ref_track];

    let max_src_len = tracks_8k
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != ref_track)
        .map(|(_, v)| ((v.len() as f64) * 1.05) as usize)
        .max()
        .unwrap_or(ref_audio.len());

    // Build RefFftCache(tempo) and RefFftCache(full) + extract subs concurrently
    let ((ref_tempo_cache, ref_full_cache), all_sub_events) = rayon::join(
        || {
            rayon::join(
                || RefFftCache::new(ref_audio, ref_audio.len()),
                || RefFftCache::new(ref_audio, ref_audio.len() + max_src_len),
            )
        },
        || match preextracted_subs {
            Some(m) => m,
            None => media::extract_all_sub_events(path, &subs),
        },
    );

    // Pool of up to 3 reusable scratch pairs for parallel track analysis without memory bloat
    let non_ref: Vec<&TrackInfo> = audios.iter().copied().filter(|t| t.n != ref_track).collect();
    let pool_sz = non_ref.len().clamp(1, 3);
    let pool = Mutex::new(
        (0..pool_sz)
            .map(|_| (ref_tempo_cache.make_scratch(), ref_full_cache.make_scratch()))
            .collect::<Vec<_>>(),
    );

    // Analyze non-ref tracks in parallel (capped by pool_sz via semaphore-like pool or chunking)
    let mut non_ref_results: Vec<(usize, AudioAnalysis)> = Vec::with_capacity(non_ref.len());
    for chunk in non_ref.chunks(pool_sz) {
        let chunk_res: Vec<(usize, AudioAnalysis)> = chunk
            .par_iter()
            .map(|t| {
                let (mut t_sc, mut f_sc) = {
                    let mut g = pool.lock().unwrap();
                    g.pop().unwrap()
                };
                tick(&format!("audio {}", t.lang));
                let a = analyze_audio(
                    ref_audio,
                    &ref_tempo_cache,
                    &mut t_sc,
                    &ref_full_cache,
                    &mut f_sc,
                    &tracks_8k[t.n],
                    20,
                    30.0,
                    5.0,
                    12.0,
                );
                pool.lock().unwrap().push((t_sc, f_sc));
                (t.n, a)
            })
            .collect();
        non_ref_results.extend(chunk_res);
    }

    let mut analysis_by_n: BTreeMap<usize, AudioAnalysis> = non_ref_results.into_iter().collect();
    let mut out_audio = Vec::with_capacity(audios.len());
    let mut aligned: Vec<(&[f32], f64)> = Vec::new();
    for t in &audios {
        if t.n == ref_track {
            out_audio.push(AudioReport {
                n: t.n,
                id: t.id,
                lang: t.lang.clone(),
                name: t.name.clone(),
                codec: t.codec.clone(),
                channels: t.channels,
                status: "ref".to_string(),
                delay_ms: Some(0),
                tempo: 1.0,
                points: vec![],
                dense: vec![],
                segments: vec![],
                reason: None,
                duration_s: Some(round3((tracks_8k[t.n].len() as f64) / (SR as f64))),
                ref_fps: info.video_fps.map(snap_fps),
                src_fps: info.video_fps.map(snap_fps),
                pitch_shift: None,
                pitch_semitones: None,
            });
            continue;
        }
        let a = analysis_by_n.remove(&t.n).unwrap();
        if a.status == "ok" || a.status == "offset" {
            if let Some(ms) = a.delay_ms {
                aligned.push((&tracks_8k[t.n], (ms as f64) / 1000.0));
            }
        }
        let (ref_fps, src_fps, pitch_semitones) = if a.status == "tempo" {
            let (rf, sf) = infer_fps_pair(info.video_fps, a.tempo);
            (rf, sf, Some(pitch_semitones_from_tempo(a.tempo)))
        } else {
            (None, None, None)
        };
        out_audio.push(AudioReport {
            n: t.n,
            id: t.id,
            lang: t.lang.clone(),
            name: t.name.clone(),
            codec: t.codec.clone(),
            channels: t.channels,
            status: a.status,
            delay_ms: a.delay_ms,
            tempo: a.tempo,
            points: a.points,
            dense: a.dense,
            segments: a.segments,
            reason: a.reason,
            duration_s: Some(round3((tracks_8k[t.n].len() as f64) / (SR as f64))),
            ref_fps,
            src_fps,
            pitch_shift: a.pitch_shift,
            pitch_semitones,
        });
    }

    let activity = if !aligned.is_empty() {
        voice_difference(ref_audio, &aligned)
    } else {
        speech_activity(ref_audio)
    };

    let mut out_subs = Vec::with_capacity(subs.len());
    for t in &subs {
        tick(&format!("subtítulos {}", t.lang));
        let events = all_sub_events
            .get(&t.n)
            .and_then(|o| o.as_ref())
            .cloned()
            .unwrap_or_default();
        let (status, offset_ms, conf) = if events.is_empty() {
            ("unknown", None, 0.0)
        } else {
            let (off, c) = measure_subs(&activity, &events, 20.0);
            let ms = (off * 1000.0).round() as i64;
            let st = if c < SUB_MIN_CONF {
                "unknown"
            } else if ms.abs() < SUB_THRESHOLD_MS {
                "ok"
            } else {
                "offset"
            };
            (st, Some(ms), round2(c))
        };
        out_subs.push(SubReport {
            n: t.n,
            id: t.id,
            lang: t.lang.clone(),
            name: t.name.clone(),
            codec: t.codec.clone(),
            status: status.to_string(),
            offset_ms,
            conf,
            events,
            twin_delay_ms: None,
            twin_segments: vec![],
            twin_tempo: None,
        });
    }

    apply_sub_twin(&out_audio, &mut out_subs);
    let worst_status = recompute_worst_status(&out_audio, &out_subs);

    progress(1.0, "listo");
    let suggest_start = crate::model::round2((transient(ref_audio, SR) - 1.6).max(0.0));
    Ok(Report {
        file: path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        duration: info.duration,
        video_fps: info.video_fps.map(snap_fps),
        ref_track,
        status: worst_status,
        suggest_start,
        audio: out_audio,
        subs: out_subs,
    })
}

fn has_signs_keyword(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    if lower.contains("s&s") {
        return true;
    }
    lower
        .split(|c: char| !c.is_alphanumeric())
        .any(|tok| {
            matches!(
                tok,
                "sign"
                    | "signs"
                    | "forced"
                    | "forzado"
                    | "forzados"
                    | "forzada"
                    | "forzadas"
                    | "cartel"
                    | "carteles"
                    | "letrero"
                    | "letreros"
                    | "titulo"
                    | "titulos"
                    | "título"
                    | "títulos"
            )
        })
}

fn has_dialogue_keyword(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower
        .split(|c: char| !c.is_alphanumeric())
        .any(|tok| matches!(tok, "full" | "dialogue" | "dialogo" | "diálogo" | "completo" | "completos" | "normal"))
}

/// Inherits audio timing adjustments (`offset`, `cuts`, or `tempo`) onto paired Signs/Forced or dub-aligned CC tracks.
///
/// Normal dialogue subtitles timed to the reference (`ok`, `|offset| < SUB_THRESHOLD_MS`, or
/// normal dialogue tracks without Signs/Forced markers) are NEVER overridden by dub audio offsets.
/// Unmeasurable Signs/Forced tracks inherit matching-language `"offset"`, `"cuts"`, or `"tempo"` audio adjustments.
pub fn apply_sub_twin(audio: &[AudioReport], subs: &mut [SubReport]) -> bool {
    let mut lang_has_named_signs = std::collections::HashSet::new();
    for s in subs.iter() {
        if has_signs_keyword(&s.name) {
            lang_has_named_signs.insert(s.lang.clone());
        }
    }

    let mut changed = false;
    for s in subs.iter_mut() {
        let unmeasurable = s.events.is_empty() || s.offset_ms.is_none() || s.conf < SUB_MIN_CONF;
        let is_signs = has_signs_keyword(&s.name)
            || (!has_dialogue_keyword(&s.name)
                && !lang_has_named_signs.contains(&s.lang)
                && s.events.len() <= 15);

        let (twin_delay, twin_segs, twin_tempo, inherited_status): (Option<i64>, Vec<Segment>, Option<f64>, Option<&'static str>) =
            if s.lang == "und" {
                (None, vec![], None, None)
            } else if unmeasurable {
                if is_signs {
                    audio
                        .iter()
                        .find_map(|a| {
                            if a.lang != s.lang {
                                return None;
                            }
                            match a.status.as_str() {
                                "offset" => {
                                    let ad = a.delay_ms.filter(|&d| d != 0)?;
                                    Some((Some(ad), vec![], None, Some("offset")))
                                }
                                "cuts" if !a.segments.is_empty() => {
                                    Some((None, a.segments.clone(), None, Some("cuts")))
                                }
                                "tempo" if (a.tempo - 1.0).abs() > 1e-12 || a.delay_ms.is_some_and(|d| d != 0) => {
                                    Some((a.delay_ms, vec![], Some(a.tempo), Some("tempo")))
                                }
                                _ => None,
                            }
                        })
                        .unwrap_or((None, vec![], None, None))
                } else {
                    (None, vec![], None, None)
                }
            } else if let Some(ms) = s.offset_ms {
                // Never override ref-synced `ok` dialogue subs (e.g. +100..+350 ms)
                if ms.abs() >= SUB_THRESHOLD_MS {
                    let matched = audio.iter().find_map(|a| {
                        if a.lang == s.lang && a.status == "offset" {
                            let ad = a.delay_ms?;
                            if ad.abs() >= 300
                                && (ms - ad).abs() < ms.abs()
                                && (ms - ad).abs() <= 350
                            {
                                return Some(ad);
                            }
                        }
                        None
                    });
                    (matched, vec![], None, matched.map(|_| "offset"))
                } else {
                    (None, vec![], None, None)
                }
            } else {
                (None, vec![], None, None)
            };

        let new_status = if let Some(st) = inherited_status {
            st
        } else if unmeasurable {
            "unknown"
        } else if s.offset_ms.is_some_and(|ms| ms.abs() < SUB_THRESHOLD_MS) {
            "ok"
        } else {
            "offset"
        };
        if s.status != new_status
            || s.twin_delay_ms != twin_delay
            || s.twin_segments != twin_segs
            || s.twin_tempo != twin_tempo
        {
            s.status = new_status.to_string();
            s.twin_delay_ms = twin_delay;
            s.twin_segments = twin_segs;
            s.twin_tempo = twin_tempo;
            changed = true;
        }
    }
    changed
}

/// Computes the overall report status across audio tracks and actionable subtitle tracks.
pub fn recompute_worst_status(audio: &[AudioReport], subs: &[SubReport]) -> String {
    let status_rank = |s: &str| -> usize {
        match s {
            "ok" => 0,
            "ref" => 1,
            "unknown" => 2,
            "offset" => 3,
            "tempo" => 4,
            "cuts" => 5,
            _ => 0,
        }
    };
    let mut worst_rank = 0usize;
    let mut worst_str = "ok";
    for a in audio {
        let r = status_rank(&a.status);
        if r >= worst_rank {
            worst_rank = r;
            worst_str = &a.status;
        }
    }
    for s in subs {
        if matches!(s.status.as_str(), "offset" | "tempo" | "cuts") {
            let r = status_rank(&s.status);
            if r > worst_rank {
                worst_rank = r;
                worst_str = &s.status;
            }
        }
    }
    if worst_str == "ref" {
        "ok".to_string()
    } else {
        worst_str.to_string()
    }
}

pub fn analyze_file(
    path: &Path,
    ref_track: usize,
    progress: &(dyn Fn(f64, &str) + Sync + Send),
) -> Result<Report> {
    let info = media::probe(path)?;
    let n_audio = info.tracks.iter().filter(|t| t.kind == "audio").count();
    let subs: Vec<&TrackInfo> = info
        .tracks
        .iter()
        .filter(|t| t.kind == "subtitles")
        .collect();
    progress(0.0, "leyendo pistas de audio");
    let (tracks_res, subs_map) = rayon::join(
        || media::load_audio_all_single_pass(path, n_audio, SR as u32),
        || media::extract_all_sub_events(path, &subs),
    );
    let tracks_8k = tracks_res?;
    analyze_file_with_tracks(path, &info, &tracks_8k, ref_track, Some(subs_map), progress)
}

// ---------------------------------------------------------------- Plan & Apply

pub fn propose(report: &Report) -> Plan {
    let mut audio = BTreeMap::new();
    let mut subs = BTreeMap::new();
    let ref_dur = report
        .audio
        .iter()
        .find(|a| a.n == report.ref_track)
        .and_then(|a| a.duration_s);
    for a in &report.audio {
        match a.status.as_str() {
            "offset" => {
                audio.insert(
                    a.n.to_string(),
                    AudioPlanEntry {
                        mode: "delay".to_string(),
                        delay_ms: a.delay_ms,
                        ..Default::default()
                    },
                );
            }
            "cuts" => {
                audio.insert(
                    a.n.to_string(),
                    AudioPlanEntry {
                        mode: "segments".to_string(),
                        segments: Some(a.segments.clone()),
                        codec: Some("auto".to_string()),
                        ..Default::default()
                    },
                );
            }
            "tempo" => {
                audio.insert(
                    a.n.to_string(),
                    AudioPlanEntry {
                        mode: "tempo".to_string(),
                        tempo: Some(a.tempo),
                        delay_ms: a.delay_ms,
                        pitch_correct: Some(a.pitch_shift.unwrap_or(false)),
                        codec: Some("flac".to_string()),
                        ..Default::default()
                    },
                );
            }
            "ok" => {
                if let (Some(rd), Some(sd)) = (ref_dur, a.duration_s) {
                    if sd > rd + 0.25 {
                        audio.insert(
                            a.n.to_string(),
                            AudioPlanEntry {
                                mode: "trim".to_string(),
                                ..Default::default()
                            },
                        );
                    }
                }
            }
            _ => {}
        }
    }
    for s in &report.subs {
        match s.status.as_str() {
            "offset" => {
                let d = s.twin_delay_ms.or(s.offset_ms);
                subs.insert(
                    s.n.to_string(),
                    SubPlanEntry {
                        delay_ms: d,
                        ..Default::default()
                    },
                );
            }
            "cuts" if !s.twin_segments.is_empty() => {
                subs.insert(
                    s.n.to_string(),
                    SubPlanEntry {
                        segments: Some(s.twin_segments.clone()),
                        ..Default::default()
                    },
                );
            }
            "tempo" if s.twin_tempo.is_some() => {
                subs.insert(
                    s.n.to_string(),
                    SubPlanEntry {
                        delay_ms: s.twin_delay_ms,
                        tempo: s.twin_tempo,
                        ..Default::default()
                    },
                );
            }
            _ => {}
        }
    }
    Plan { audio, subs }
}

pub fn sub_entry_is_active(p: &SubPlanEntry) -> bool {
    p.delay_ms.unwrap_or(0) != 0
        || p.segments.as_ref().is_some_and(|v| !v.is_empty())
        || p.tempo.is_some_and(|tp| (tp - 1.0).abs() > 1e-12)
}

pub fn plan_is_empty(plan: &Plan) -> bool {
    let has_audio = plan.audio.values().any(|p| {
        p.mode != "none" && !(p.mode == "delay" && p.delay_ms.unwrap_or(0) == 0)
    });
    let has_subs = plan.subs.values().any(sub_entry_is_active);
    !has_audio && !has_subs
}

/// Computes the subtitle shift in seconds at source timestamp `t` (supports constant `delay_ms`,
/// piecewise `segments`, and `tempo` scaling).
pub fn sub_source_shift_at(entry: &SubPlanEntry, t: f64) -> f64 {
    if let Some(segs) = &entry.segments {
        for s in segs {
            let end = s.end.unwrap_or(f64::INFINITY);
            if t >= s.start && t < end {
                return s.delay;
            }
        }
        if let Some(last) = segs.last() {
            if t >= last.start {
                return last.delay;
            }
        }
        if let Some(first) = segs.first() {
            return first.delay;
        }
    }
    let d = (entry.delay_ms.unwrap_or(0) as f64) / 1000.0;
    if let Some(tp) = entry.tempo {
        if tp > 0.0 && (tp - 1.0).abs() > 1e-12 {
            return t * (1.0 / tp - 1.0) + d;
        }
    }
    d
}

pub fn shift_at(entry: Option<&AudioPlanEntry>, t: f64) -> f64 {
    let Some(e) = entry else {
        return 0.0;
    };
    if e.mode == "none" {
        return 0.0;
    }
    if e.mode == "segments" {
        if let Some(segs) = &e.segments {
            if !segs.is_empty() {
                for s in segs {
                    let end = s.end.unwrap_or(f64::INFINITY);
                    if s.start + s.delay <= t && t < end + s.delay {
                        return s.delay;
                    }
                }
                if let Some(last) = segs.last() {
                    if t >= last.start + last.delay {
                        return last.delay;
                    }
                }
                return segs[0].delay;
            }
        }
        return 0.0;
    }
    let d = (e.delay_ms.unwrap_or(0) as f64) / 1000.0;
    if e.mode == "tempo" {
        if let Some(tp) = e.tempo {
            if tp > 0.0 && (tp - 1.0).abs() > 1e-12 {
                return t - (t - d) * tp;
            }
        }
    }
    d
}

fn encode_args(codec: &str, channels: Option<u32>) -> Vec<String> {
    if codec == "aac" {
        let br = if channels.unwrap_or(2) <= 2 {
            "192k"
        } else {
            "384k"
        };
        vec!["-c:a".into(), "aac".into(), "-b:a".into(), br.into()]
    } else {
        vec![
            "-c:a".into(),
            "flac".into(),
            "-sample_fmt".into(),
            "s32".into(),
            "-compression_level".into(),
            "12".into(),
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawFrameCodec {
    Adts,
    Ac3,
    Eac3,
}

impl RawFrameCodec {
    fn mux_fmt(self) -> &'static str {
        match self {
            RawFrameCodec::Adts => "adts",
            RawFrameCodec::Ac3 => "ac3",
            RawFrameCodec::Eac3 => "eac3",
        }
    }

    fn demux_fmt(self) -> &'static str {
        match self {
            RawFrameCodec::Adts => "aac",
            RawFrameCodec::Ac3 => "ac3",
            RawFrameCodec::Eac3 => "eac3",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitstreamFrame {
    pub start: usize,
    pub end: usize,
    pub samples: usize,
}

/// Parses raw AAC ADTS frames, rejecting HE-AAC/inline-PCE or trailing corrupt bytes.
pub fn parse_adts_frames(data: &[u8]) -> Result<(Vec<BitstreamFrame>, u32)> {
    const ADTS_SR: [u32; 13] = [
        96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
    ];
    let mut frames = Vec::new();
    let mut i = 0usize;
    let mut expected_sr = None;
    let mut expected_ch = None;
    while i < data.len() {
        if i + 7 > data.len() {
            bail!("ADTS truncado en byte {i}");
        }
        if data[i] != 0xFF || (data[i + 1] & 0xF6) != 0xF0 {
            bail!("syncword ADTS inválido en byte {i}");
        }
        let prot_absent = data[i + 1] & 0x01;
        let hdr_len = if prot_absent == 1 { 7 } else { 9 };
        let sf_idx = ((data[i + 2] >> 2) & 0x0F) as usize;
        if sf_idx >= ADTS_SR.len() {
            bail!("sample rate ADTS no soportado ({sf_idx})");
        }
        let sr = ADTS_SR[sf_idx];
        let ch_cfg = ((data[i + 2] & 0x01) << 2) | ((data[i + 3] >> 6) & 0x03);
        if ch_cfg == 0 {
            bail!("ADTS channel_config=0 (PCE inline) no seguro para splicing");
        }
        let flen = (((data[i + 3] & 0x03) as usize) << 11)
            | ((data[i + 4] as usize) << 3)
            | (((data[i + 5] & 0xE0) as usize) >> 5);
        if flen < hdr_len || i + flen > data.len() {
            bail!("tamaño de frame ADTS inválido ({flen}) en byte {i}");
        }
        let num_rdb = data[i + 6] & 0x03;
        if num_rdb != 0 {
            bail!("ADTS multibloque no soportado");
        }
        if let Some(esr) = expected_sr {
            if esr != sr || expected_ch != Some(ch_cfg) {
                bail!("cambio de parámetros en stream ADTS");
            }
        } else {
            expected_sr = Some(sr);
            expected_ch = Some(ch_cfg);
        }
        frames.push(BitstreamFrame {
            start: i,
            end: i + flen,
            samples: 1024,
        });
        i += flen;
    }
    let sr = expected_sr.context("stream ADTS vacío")?;
    Ok((frames, sr))
}

/// Parses raw AC-3 or independent E-AC-3 syncframes without dependent substreams.
pub fn parse_ac3_or_eac3_frames(
    data: &[u8],
    expect_eac3: bool,
) -> Result<(Vec<BitstreamFrame>, u32)> {
    // ETSI TS 102 366 Table 4.13 frame size in 16-bit words for fscod 0..=2 and frmsizecod 0..=37
    #[rustfmt::skip]
    const AC3_WORDS: [[usize; 38]; 3] = [
        [64,64,80,80,96,96,112,112,128,128,160,160,192,192,224,224,256,256,320,320,384,384,448,448,512,512,640,640,768,768,896,896,1024,1024,1152,1152,1280,1280],
        [69,70,87,88,104,105,121,122,139,140,174,175,208,209,243,244,278,279,348,349,417,418,487,488,557,558,696,697,835,836,975,976,1114,1115,1253,1254,1393,1394],
        [96,96,120,120,144,144,168,168,192,192,240,240,288,288,336,336,384,384,480,480,576,576,672,672,768,768,960,960,1152,1152,1344,1344,1536,1536,1728,1728,1920,1920],
    ];
    const SR_MAIN: [u32; 3] = [48000, 44100, 32000];
    const SR_HALF: [u32; 3] = [24000, 22050, 16000];
    const BLKS: [usize; 4] = [1, 2, 3, 6];

    let mut frames = Vec::new();
    let mut i = 0usize;
    let mut expected_sr = None;
    let mut expected_samples = None;
    while i < data.len() {
        if i + 7 > data.len() {
            bail!("syncframe AC3/EAC3 truncado en byte {i}");
        }
        if data[i] != 0x0B || data[i + 1] != 0x77 {
            bail!("syncword AC3/EAC3 inválido en byte {i}");
        }
        let bsid = (data[i + 5] >> 3) & 0x1F;
        let (flen, sr, samples) = if bsid <= 10 {
            if expect_eac3 {
                bail!("AC3 mezclado en flujo EAC3");
            }
            let fscod = (data[i + 4] >> 6) as usize;
            let frmsizecod = (data[i + 4] & 0x3F) as usize;
            if fscod >= 3 || frmsizecod >= 38 {
                bail!("cabecera AC3 inválida en byte {i}");
            }
            (AC3_WORDS[fscod][frmsizecod] * 2, SR_MAIN[fscod], 1536)
        } else if bsid <= 16 {
            if !expect_eac3 {
                bail!("EAC3 en flujo AC3");
            }
            let strmtyp = (data[i + 2] >> 6) & 0x03;
            if strmtyp != 0 {
                bail!("E-AC3 con substream dependiente (strmtyp={strmtyp}) no seguro");
            }
            let frmsiz = (((data[i + 2] & 0x07) as usize) << 8) | (data[i + 3] as usize);
            let flen = (frmsiz + 1) * 2;
            let fscod = ((data[i + 4] >> 6) & 0x03) as usize;
            let (sr, numblks) = if fscod == 3 {
                let fscod2 = ((data[i + 4] >> 4) & 0x03) as usize;
                if fscod2 >= 3 {
                    bail!("fscod2 E-AC3 inválido");
                }
                (SR_HALF[fscod2], 6usize)
            } else {
                let nb = ((data[i + 4] >> 4) & 0x03) as usize;
                (SR_MAIN[fscod], BLKS[nb])
            };
            (flen, sr, numblks * 256)
        } else {
            bail!("bsid AC3/EAC3 desconocido ({bsid})");
        };
        if i + flen > data.len() {
            bail!("frame AC3/EAC3 truncado en byte {i}");
        }
        if let Some(esr) = expected_sr {
            if esr != sr || expected_samples != Some(samples) {
                bail!("frame rate/tamaño variable en flujo AC3/EAC3");
            }
        } else {
            expected_sr = Some(sr);
            expected_samples = Some(samples);
        }
        frames.push(BitstreamFrame {
            start: i,
            end: i + flen,
            samples,
        });
        i += flen;
    }
    let sr = expected_sr.context("flujo AC3/EAC3 vacío")?;
    Ok((frames, sr))
}

/// Plans output frame index sequence (`drop` & `duplicate`) for lossless segment splicing.
/// Returns `(emitted_frame_indices, initial_sync_delay_ms)`.
pub fn plan_frame_edits<F>(
    total_frames: usize,
    frame_dur: f64,
    first_pts: f64,
    segments: &[Segment],
    max_out_s: Option<f64>,
    mut pick_quiet_frame: F,
) -> Result<(Vec<usize>, i64)>
where
    F: FnMut(usize, usize) -> Result<(usize, f64)>,
{
    if total_frames == 0 || segments.is_empty() || frame_dur <= 0.0 {
        bail!("sin frames o tramos para splicing");
    }
    if frame_dur > 0.050 {
        bail!("duración de frame demasiado grande ({:.1} ms)", frame_dur * 1000.0);
    }
    let src_dur = first_pts + (total_frames as f64) * frame_dur;

    let seg0 = &segments[0];
    let mut src_t0 = seg0.start.max(first_pts);
    if src_t0 + seg0.delay < 0.0 {
        src_t0 = (first_pts - seg0.delay).max(seg0.start);
    }
    let mut cur_frame = (((src_t0 - first_pts) / frame_dur).round().max(0.0) as usize)
        .min(total_frames.saturating_sub(1));
    let out_offset_s = first_pts + (cur_frame as f64) * frame_dur + seg0.delay;
    let init_delay_ms = (out_offset_s * 1000.0).round() as i64;
    let mut cur_out_t = out_offset_s;

    let time_to_frame = |t: f64| -> usize {
        (((t - first_pts) / frame_dur).round().max(0.0) as usize).min(total_frames)
    };

    let mut emit = Vec::with_capacity(total_frames + 256);
    for (i, seg) in segments.iter().enumerate() {
        let seg_src_end = seg.end.unwrap_or(src_dur).min(src_dur);
        let seg_end_frame = time_to_frame(seg_src_end);
        if i + 1 < segments.len() {
            let next = &segments[i + 1];
            let next_src_end = next.end.unwrap_or(src_dur).min(src_dur);
            let next_end_frame = time_to_frame(next_src_end);
            let next_out_t = next.start + next.delay;
            let seg_out_end = seg_src_end + seg.delay;
            let jump = next_out_t - seg_out_end;

            if jump < -0.5 * frame_dur {
                // Drop overlapped excess frames cleanly up to target output handoff
                let stop_src_t = (next_out_t - seg.delay).min(seg_src_end);
                let stop_frame = time_to_frame(stop_src_t).min(total_frames);
                if stop_frame < cur_frame {
                    bail!("solapamiento entre tramos requiere PCM 32-bit");
                }
                while cur_frame < stop_frame {
                    emit.push(cur_frame);
                    cur_frame += 1;
                    cur_out_t += frame_dur;
                }
                let next_start_frame = time_to_frame(next.start);
                if next_start_frame >= next_end_frame {
                    bail!("tramo destino vacío tras recorte");
                }
                cur_frame = next_start_frame;
                let rem = (first_pts + (cur_frame as f64) * frame_dur + next.delay) - cur_out_t;
                let pad = (rem / frame_dur).round() as isize;
                if pad > 0 {
                    let quiet_f = cur_frame.min(total_frames - 1);
                    for _ in 0..pad {
                        emit.push(quiet_f);
                        cur_out_t += frame_dur;
                    }
                } else if pad < 0 {
                    let skip = (-pad) as usize;
                    if cur_frame + skip >= next_end_frame {
                        bail!("ajuste excede longitud del tramo");
                    }
                    cur_frame += skip;
                }
            } else {
                // Positive gap (or tiny drift): pick quietest boundary frame around cut and duplicate
                let cut_frame = time_to_frame(seg_src_end).clamp(cur_frame, next_end_frame.saturating_sub(1));
                let radius = (0.25 / frame_dur).round() as usize;
                let search_lo = cut_frame.saturating_sub(radius).max(cur_frame);
                let search_hi = (cut_frame + radius).min(next_end_frame.saturating_sub(1));
                let (seam_frame, rms_db) = if search_lo <= search_hi {
                    pick_quiet_frame(search_lo, search_hi)?
                } else {
                    (cut_frame, -90.0)
                };
                while cur_frame <= seam_frame && cur_frame < total_frames {
                    emit.push(cur_frame);
                    cur_frame += 1;
                    cur_out_t += frame_dur;
                }
                let resume_frame = (seam_frame + 1).min(next_end_frame);
                let desired_resume_t = first_pts + (resume_frame as f64) * frame_dur + next.delay;
                let gap = desired_resume_t - cur_out_t;
                let n_adj = (gap / frame_dur).round() as isize;
                if n_adj >= 3 && rms_db > -28.0 {
                    bail!(
                        "pausa de {} frames sobre audio activo ({:.1} dBFS): usar PCM 32-bit -> FLAC",
                        n_adj,
                        rms_db
                    );
                }
                if n_adj > 0 {
                    let dup_idx = seam_frame.min(total_frames - 1);
                    for _ in 0..n_adj {
                        emit.push(dup_idx);
                        cur_out_t += frame_dur;
                    }
                    cur_frame = resume_frame;
                } else {
                    let drop = (-n_adj) as usize;
                    if resume_frame + drop >= next_end_frame {
                        bail!("recorte en seam excede tramo siguiente");
                    }
                    cur_frame = resume_frame + drop;
                }
            }
        } else {
            while cur_frame < seg_end_frame && cur_frame < total_frames {
                emit.push(cur_frame);
                cur_frame += 1;
                cur_out_t += frame_dur;
            }
        }
    }

    if emit.is_empty() {
        bail!("splicing sin frames de salida");
    }
    if let Some(cap) = max_out_s {
        if cur_out_t > cap + 0.25 {
            let keep = (((cap - out_offset_s) / frame_dur).round().max(1.0)) as usize;
            emit.truncate(keep.min(emit.len()));
        }
    }
    Ok((emit, init_delay_ms))
}

fn measure_window_quiet_frame(
    raw: &[u8],
    frames: &[BitstreamFrame],
    fmt: &str,
    sr: u32,
    lo: usize,
    hi: usize,
) -> Result<(usize, f64)> {
    let warm_lo = lo.saturating_sub(1);
    let slice = &raw[frames[warm_lo].start..frames[hi].end];
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-f", fmt, "-i", "-", "-f", "f32le", "-ac", "1", "-ar", &sr.to_string(), "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("ffmpeg window decode for quiet frame")?;
    {
        let mut stdin = child.stdin.take().unwrap();
        let _ = stdin.write_all(slice);
    }
    let out = child.wait_with_output()?;
    if !out.status.success() {
        bail!("ffmpeg window decode falló");
    }
    let samples: Vec<f32> = out
        .stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut pos = if warm_lo < lo { frames[warm_lo].samples } else { 0 };
    let mut best_idx = (lo + hi) / 2;
    let mut best_db = f64::INFINITY;
    for idx in lo..=hi {
        let len = frames[idx].samples;
        if pos >= samples.len() {
            break;
        }
        let end = (pos + len).min(samples.len());
        let win = &samples[pos..end];
        let mean_sq = if win.is_empty() {
            1.0
        } else {
            win.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / (win.len() as f64)
        };
        let db = 20.0 * (mean_sq.sqrt() + 1e-9).log10();
        if db < best_db {
            best_db = db;
            best_idx = idx;
        }
        pos += len;
    }
    if best_db.is_infinite() {
        bail!("no se pudieron medir frames en la ventana de corte");
    }
    Ok((best_idx, best_db))
}

fn render_segments_lossless_frames(
    path: &Path,
    track: &TrackInfo,
    segments: &[Segment],
    out_mka: &Path,
    max_out_s: Option<f64>,
    expected_end_s: Option<f64>,
) -> Result<i64> {
    let detail = media::probe_audio_stream_detail(path, track.n)?;
    let prof_up = detail.profile.to_ascii_uppercase();
    let raw_codec = match detail.codec_name.as_str() {
        "aac" => {
            if prof_up.contains("HE") || prof_up.contains("SBR") || prof_up.contains("PS") || prof_up.contains("LTP") {
                bail!("perfil AAC ({}) requiere PCM 32-bit", detail.profile);
            }
            RawFrameCodec::Adts
        }
        "ac3" => RawFrameCodec::Ac3,
        "eac3" => RawFrameCodec::Eac3,
        other => bail!("codec {other} no soporta splicing sin recodificar"),
    };

    let raw_out = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(path)
        .args(["-map", &format!("0:a:{}", track.n), "-c:a", "copy", "-f", raw_codec.mux_fmt(), "-"])
        .output()
        .context("ffmpeg raw stream dump")?;
    if !raw_out.status.success() || raw_out.stdout.is_empty() {
        bail!("no se pudo extraer bitstream comprimido");
    }
    let raw = raw_out.stdout;

    let (frames, sr) = match raw_codec {
        RawFrameCodec::Adts => parse_adts_frames(&raw)?,
        RawFrameCodec::Ac3 => parse_ac3_or_eac3_frames(&raw, false)?,
        RawFrameCodec::Eac3 => parse_ac3_or_eac3_frames(&raw, true)?,
    };
    let spf = frames[0].samples;
    if frames.iter().any(|f| f.samples != spf) {
        bail!("muestras por frame variables");
    }
    let frame_dur = (spf as f64) / (sr as f64);
    let raw_dur = (frames.len() as f64) * frame_dur;
    if let Some(end_s) = expected_end_s {
        let container_span = (end_s - detail.first_pts).max(0.0);
        if (raw_dur - container_span).abs() > 0.25 {
            bail!(
                "duración de frames ({raw_dur:.2}s) difiere del contenedor ({container_span:.2}s)"
            );
        }
    }

    let (emit, init_delay_ms) = plan_frame_edits(
        frames.len(),
        frame_dur,
        detail.first_pts,
        segments,
        max_out_s,
        |lo, hi| measure_window_quiet_frame(&raw, &frames, raw_codec.demux_fmt(), sr, lo, hi),
    )?;

    let mut mux = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            raw_codec.demux_fmt(),
            "-i",
            "-",
            "-c:a",
            "copy",
            "-f",
            "matroska",
        ])
        .arg(out_mka)
        .stdin(Stdio::piped())
        .spawn()
        .context("ffmpeg remux lossless frames")?;
    {
        let mut stdin = mux.stdin.take().unwrap();
        let mut buf = Vec::with_capacity(65536);
        for &idx in &emit {
            let fr = frames[idx];
            buf.extend_from_slice(&raw[fr.start..fr.end]);
            if buf.len() >= 65536 {
                stdin.write_all(&buf)?;
                buf.clear();
            }
        }
        if !buf.is_empty() {
            stdin.write_all(&buf)?;
        }
    }
    let st = mux.wait()?;
    if !st.success() {
        let _ = std::fs::remove_file(out_mka);
        bail!("ffmpeg falló al remuxar frames sin recodificar");
    }
    Ok(init_delay_ms)
}

pub fn render_segments_pcm32(
    path: &Path,
    track: &TrackInfo,
    segments: &[Segment],
    out: &Path,
    codec: &str,
    max_out_s: Option<f64>,
) -> Result<()> {
    let sr = track.sample_rate.unwrap_or(48000) as usize;
    let ch = track.channels.unwrap_or(2) as usize;
    let parent = out.parent().unwrap_or_else(|| Path::new("."));
    let tmp_raw = NamedTempFile::new_in(parent).context("temp raw 32-bit PCM file")?;

    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(path)
        .args([
            "-map",
            &format!("0:a:{}", track.n),
            "-af",
            "aresample=async=1:first_pts=0",
            "-ar",
            &sr.to_string(),
            "-ac",
            &ch.to_string(),
            "-f",
            "f32le",
        ])
        .arg(tmp_raw.path())
        .status()
        .context("ffmpeg decode 32-bit PCM for segments")?;
    if !status.success() {
        bail!("ffmpeg no pudo decodificar la pista en PCM 32 bits");
    }

    let file = File::open(tmp_raw.path())?;
    let mmap = unsafe { Mmap::map(&file)? };
    let total_frames = mmap.len() / (4 * ch);

    let mut spans = Vec::new();
    for s in segments {
        let mut a = (s.start * sr as f64) as isize;
        let b = match s.end {
            None => total_frames as isize,
            Some(e) => ((e * sr as f64) as isize).min(total_frames as isize),
        };
        let mut pos = a + (s.delay * sr as f64).round() as isize;
        if pos < 0 {
            a -= pos;
            pos = 0;
        }
        if b > a {
            spans.push((a as usize, b as usize, pos as usize));
        }
    }
    if spans.is_empty() {
        bail!("no hay tramos válidos para renderizar");
    }

    let raw_out_frames = spans
        .iter()
        .map(|&(a, b, pos)| pos + b - a)
        .max()
        .unwrap_or(0);
    let (out_frames, was_trimmed) = match max_out_s {
        Some(cap) => {
            let cap_f = (cap * sr as f64).round().max(1.0) as usize;
            if raw_out_frames > cap_f + sr / 4 {
                (cap_f, true)
            } else {
                (raw_out_frames, false)
            }
        }
        None => (raw_out_frames, false),
    };
    let mut dst = vec![0.0f32; out_frames * ch];
    let fade_len = sr / 200;
    let fade: Vec<f64> = if fade_len > 1 {
        (0..fade_len)
            .map(|k| (k as f64) / ((fade_len - 1) as f64))
            .collect()
    } else {
        vec![1.0; fade_len]
    };

    for (i, &(a, b, pos)) in spans.iter().enumerate() {
        if pos >= out_frames {
            continue;
        }
        let seg_frames = (b - a).min(out_frames - pos);
        let apply_fade = seg_frames > 2 * fade_len && fade_len > 0;
        for f in 0..seg_frames {
            let mut gain = 1.0f64;
            if apply_fade {
                if i > 0 && f < fade_len {
                    gain *= fade[f];
                }
                if i + 1 < spans.len() && f >= seg_frames - fade_len {
                    gain *= fade[seg_frames - 1 - f];
                }
            }
            let src_base = (a + f) * ch;
            let dst_base = (pos + f) * ch;
            for c in 0..ch {
                let byte_idx = (src_base + c) * 4;
                let sample = f32::from_le_bytes([
                    mmap[byte_idx],
                    mmap[byte_idx + 1],
                    mmap[byte_idx + 2],
                    mmap[byte_idx + 3],
                ]);
                dst[dst_base + c] = if (gain - 1.0).abs() < 1e-12 {
                    sample
                } else {
                    ((sample as f64) * gain) as f32
                };
            }
        }
    }
    if was_trimmed {
        let end_fade = (sr / 50).min(out_frames); // 20 ms fade out at cut
        for k in 0..end_fade {
            let g = (k as f64) / (end_fade.max(1) as f64);
            let f = out_frames - 1 - k;
            for c in 0..ch {
                dst[f * ch + c] = ((dst[f * ch + c] as f64) * g) as f32;
            }
        }
    }
    drop(mmap);
    drop(file);
    drop(tmp_raw);

    let enc = encode_args(codec, Some(ch as u32));
    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-v",
        "error",
        "-y",
        "-f",
        "f32le",
        "-ar",
        &sr.to_string(),
        "-ac",
        &ch.to_string(),
        "-i",
        "-",
    ])
    .args(&enc);
    if codec != "aac" {
        cmd.args(["-f", "flac"]);
    } else {
        cmd.args(["-f", "matroska"]);
    }
    let mut child = cmd
        .arg(out)
        .stdin(Stdio::piped())
        .spawn()
        .context("ffmpeg encode 32-bit PCM segments")?;

    {
        let mut stdin = child.stdin.take().unwrap();
        let mut buf = Vec::with_capacity(65536);
        for chunk in dst.chunks(16384) {
            buf.clear();
            for &s in chunk {
                buf.extend_from_slice(&s.clamp(-1.0, 1.0).to_le_bytes());
            }
            stdin.write_all(&buf)?;
        }
    }
    let st = child.wait()?;
    if !st.success() {
        bail!("ffmpeg no pudo codificar la pista corregida desde PCM 32 bits");
    }
    Ok(())
}

/// Renders segmented audio edits cleanly: tries lossless packet frame drop/duplication first when
/// `codec` is `"auto"` / `"copy"` / `"frames"`, otherwise jumps directly to 32-bit PCM + max-compression FLAC.
pub fn render_segments(
    path: &Path,
    track: &TrackInfo,
    segments: &[Segment],
    work_dir: &Path,
    stem: &str,
    codec: &str,
    max_out_s: Option<f64>,
    expected_end_s: Option<f64>,
) -> Result<(PathBuf, i64)> {
    let try_frames = matches!(codec, "auto" | "copy" | "frames" | "");
    if try_frames {
        let mka_out = work_dir.join(format!(".{stem}.a{}.mka", track.n));
        match render_segments_lossless_frames(path, track, segments, &mka_out, max_out_s, expected_end_s) {
            Ok(delay_ms) => {
                eprintln!(
                    "[segments] a:{} ({}): frame drop/dup sin recodificar OK (sync {:+} ms)",
                    track.n, track.lang, delay_ms
                );
                return Ok((mka_out, delay_ms));
            }
            Err(e) => {
                let _ = std::fs::remove_file(&mka_out);
                eprintln!(
                    "[segments] a:{} ({}): sin recodificar no posible ({}), usando PCM 32 bits -> FLAC compresión máx.",
                    track.n, track.lang, e
                );
            }
        }
    }

    let pcm_codec = if codec == "aac" { "aac" } else { "flac" };
    let ext = if pcm_codec == "aac" { ".mka" } else { ".flac" };
    let out = work_dir.join(format!(".{stem}.a{}{ext}", track.n));
    render_segments_pcm32(path, track, segments, &out, pcm_codec, max_out_s)?;
    Ok((out, 0))
}

pub fn render_tempo(
    path: &Path,
    track: &TrackInfo,
    tempo: f64,
    pitch_correct: bool,
    out: &Path,
    codec: &str,
    max_out_s: Option<f64>,
) -> Result<()> {
    let enc = encode_args(codec, track.channels);
    let core_af = if pitch_correct {
        // Inverse varispeed: simultaneously corrects speed and restores original pitch via 28-bit SoXR resampling in 32-bit float
        let sr = track.sample_rate.unwrap_or(48000).max(8000);
        let new_sr = (sr as f64) * tempo;
        format!("aformat=sample_fmts=flt,aresample={sr},asetrate={new_sr:.6},aresample={sr}:resampler=soxr:precision=28")
    } else {
        // Time-stretch preserving existing pitch in 32-bit float
        format!("aformat=sample_fmts=flt,atempo={tempo:.8}")
    };
    let af = if codec == "aac" {
        core_af
    } else {
        // Guarantee full 32-bit float intermediate processing before s32 max-compression FLAC
        format!("{core_af},aformat=sample_fmts=s32")
    };
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-v", "error", "-y", "-i"])
        .arg(path)
        .args(["-map", &format!("0:a:{}", track.n), "-af", &af]);
    if let Some(cap) = max_out_s {
        cmd.args(["-t", &format!("{cap:.6}")]);
    }
    let st = cmd.args(&enc).arg(out).status().context("ffmpeg tempo")?;
    if !st.success() {
        bail!("ffmpeg no pudo corregir la velocidad");
    }
    Ok(())
}

pub fn render_trim_pcm32_flac(
    path: &Path,
    track_n: usize,
    trim_dur: f64,
    out: &Path,
) -> Result<()> {
    let st = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(path)
        .args([
            "-map",
            &format!("0:a:{track_n}"),
            "-af",
            "aresample=async=1:first_pts=0,aformat=sample_fmts=s32",
            "-t",
            &format!("{trim_dur:.6}"),
            "-c:a",
            "flac",
            "-sample_fmt",
            "s32",
            "-compression_level",
            "12",
            "-f",
            "flac",
        ])
        .arg(out)
        .status()
        .context("ffmpeg trim PCM32 -> FLAC")?;
    if !st.success() {
        bail!("ffmpeg no pudo recortar a:{track_n} vía PCM 32 bits -> FLAC");
    }
    Ok(())
}

pub fn trim_stream_copy(
    path: &Path,
    kind_char: char,
    track_n: usize,
    trim_dur: f64,
    out: &Path,
) -> Result<()> {
    let st = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(path)
        .args([
            "-map",
            &format!("0:{kind_char}:{track_n}"),
            "-t",
            &format!("{trim_dur:.6}"),
            &format!("-c:{kind_char}"),
            "copy",
            "-f",
            "matroska",
        ])
        .arg(out)
        .status()
        .context("ffmpeg trim copy")?;
    if !st.success() {
        bail!("ffmpeg -t no pudo recortar la pista {kind_char}:{track_n}");
    }
    Ok(())
}

fn parse_hms_frac(s: &str, sep: char) -> Option<f64> {
    let mut parts = s.trim().split(':');
    let h: f64 = parts.next()?.parse().ok()?;
    let m: f64 = parts.next()?.parse().ok()?;
    let sec_str = parts.next()?.replace(sep, ".");
    if parts.next().is_some() {
        return None;
    }
    let sec: f64 = sec_str.parse().ok()?;
    Some(h * 3600.0 + m * 60.0 + sec)
}

fn format_ass_time(t: f64) -> String {
    let cs = (t.max(0.0) * 100.0).round() as u64;
    let total_s = cs / 100;
    let frac = cs % 100;
    let h = total_s / 3600;
    let m = (total_s % 3600) / 60;
    let s = total_s % 60;
    format!("{h}:{m:02}:{s:02}.{frac:02}")
}

fn format_srt_time(t: f64) -> String {
    let ms = (t.max(0.0) * 1000.0).round() as u64;
    let total_s = ms / 1000;
    let frac = ms % 1000;
    let h = total_s / 3600;
    let m = (total_s % 3600) / 60;
    let s = total_s % 60;
    format!("{h:02}:{m:02}:{s:02},{frac:03}")
}

fn shift_cue_bounds(entry: &SubPlanEntry, a: f64, b: f64) -> (f64, f64) {
    let has_segs = entry.segments.as_ref().is_some_and(|v| !v.is_empty());
    let sha = sub_source_shift_at(entry, a);
    let shb = if has_segs {
        // Preserve exact display duration & relative \fad/\move/\t tags when crossing cut seams
        sha
    } else {
        sub_source_shift_at(entry, b)
    };
    let na = (a + sha).max(0.0);
    let nb = (b + shb).max(na);
    (na, nb)
}

/// Shifts ASS dialogue/comment timestamps according to `entry` (`segments` or `tempo`), clipping/dropping beyond `max_out_s`.
pub fn shift_ass_text(content: &str, entry: &SubPlanEntry, max_out_s: Option<f64>) -> String {
    let nl = if content.contains("\r\n") { "\r\n" } else { "\n" };
    let mut out = String::with_capacity(content.len());
    let mut first = true;
    for raw_line in content.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        let trimmed = line.trim_start_matches('\u{feff}');
        let is_dialogue = trimmed.starts_with("Dialogue:");
        let shifted = if is_dialogue {
            let cols: Vec<&str> = line.splitn(10, ',').collect();
            if cols.len() >= 10 {
                if let (Some(a), Some(b)) =
                    (parse_hms_frac(cols[1], '.'), parse_hms_frac(cols[2], '.'))
                {
                    let (na, mut nb) = shift_cue_bounds(entry, a, b);
                    if let Some(cap) = max_out_s {
                        if na >= cap + 0.25 {
                            continue;
                        }
                        if nb > cap {
                            nb = cap.max(na);
                        }
                    }
                    let fa = format_ass_time(na);
                    let fb = format_ass_time(nb);
                    Some(format!("{},{fa},{fb},{}", cols[0], cols[3..].join(",")))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        if !first {
            out.push_str(nl);
        }
        first = false;
        if let Some(s) = shifted {
            out.push_str(&s);
        } else {
            out.push_str(line);
        }
    }
    if content.ends_with('\n') && !out.ends_with('\n') {
        out.push_str(nl);
    }
    out
}

/// Shifts SRT cue timestamps according to `entry` (`segments` or `tempo`), clipping/dropping beyond `max_out_s`.
pub fn shift_srt_text(content: &str, entry: &SubPlanEntry, max_out_s: Option<f64>) -> String {
    let nl = if content.contains("\r\n") { "\r\n" } else { "\n" };
    let mut blocks: Vec<String> = Vec::new();
    let normalized = content.replace("\r\n", "\n");
    for block in normalized.split("\n\n") {
        let lines: Vec<&str> = block.split('\n').collect();
        if lines.is_empty() || lines.iter().all(|l| l.trim().is_empty()) {
            continue;
        }
        let mut drop = false;
        let mut rebuilt = Vec::with_capacity(lines.len());
        for line in lines {
            if let Some((lhs, rhs_all)) = line.split_once("-->") {
                let rhs = rhs_all.split_whitespace().next().unwrap_or("");
                let tail = rhs_all
                    .trim_start()
                    .strip_prefix(rhs)
                    .unwrap_or("");
                if let (Some(a), Some(b)) =
                    (parse_hms_frac(lhs, ','), parse_hms_frac(rhs, ','))
                {
                    let (na, mut nb) = shift_cue_bounds(entry, a, b);
                    if let Some(cap) = max_out_s {
                        if na >= cap + 0.25 {
                            drop = true;
                            break;
                        }
                        if nb > cap {
                            nb = cap.max(na);
                        }
                    }
                    rebuilt.push(format!(
                        "{} --> {}{}",
                        format_srt_time(na),
                        format_srt_time(nb),
                        tail
                    ));
                    continue;
                }
            }
            rebuilt.push(line.to_string());
        }
        if !drop {
            blocks.push(rebuilt.join(nl));
        }
    }
    let mut res = blocks.join(&format!("{nl}{nl}"));
    if !res.is_empty() {
        res.push_str(nl);
    }
    res
}

/// Shifts Blu-ray PGS (`.sup`) presentation/decoding timestamps in 90 kHz ticks without touching bitmaps.
pub fn shift_pgs_sup(
    data: &[u8],
    entry: &SubPlanEntry,
    max_out_s: Option<f64>,
) -> Result<Vec<u8>> {
    let mut segs: Vec<(u32, u32, u8, &[u8])> = Vec::new();
    let mut i = 0usize;
    while i < data.len() {
        if i + 13 > data.len() || &data[i..i + 2] != b"PG" {
            bail!("cabecera PGS (.sup) inválida en byte {i}");
        }
        let pts = u32::from_be_bytes([data[i + 2], data[i + 3], data[i + 4], data[i + 5]]);
        let dts = u32::from_be_bytes([data[i + 6], data[i + 7], data[i + 8], data[i + 9]]);
        let kind = data[i + 10];
        let sz = u16::from_be_bytes([data[i + 11], data[i + 12]]) as usize;
        if i + 13 + sz > data.len() {
            bail!("segmento PGS truncado en byte {i}");
        }
        segs.push((pts, dts, kind, &data[i + 13..i + 13 + sz]));
        i += 13 + sz;
    }
    let mut out = Vec::with_capacity(data.len());
    let mut prev_pts: u32 = 0;
    let mut ds_start = 0usize;
    while ds_start < segs.len() {
        let mut ds_end = ds_start;
        while ds_end < segs.len() {
            let is_end = segs[ds_end].2 == 0x80;
            ds_end += 1;
            if is_end {
                break;
            }
        }
        let base_pts_s = (segs[ds_start].0 as f64) / 90_000.0;
        let sh = sub_source_shift_at(entry, base_pts_s);
        let new_pts_s = (base_pts_s + sh).max(0.0);
        if max_out_s.is_some_and(|cap| new_pts_s >= cap + 0.25) {
            ds_start = ds_end;
            continue;
        }
        let new_pts = ((new_pts_s * 90_000.0).round().clamp(0.0, u32::MAX as f64) as u32).max(prev_pts);
        prev_pts = new_pts;
        for &(pts, dts, kind, payload) in &segs[ds_start..ds_end] {
            let seg_pts_s = ((pts as f64) / 90_000.0 + sh).max(0.0);
            let seg_pts = ((seg_pts_s * 90_000.0).round().clamp(0.0, u32::MAX as f64) as u32).max(new_pts);
            let seg_dts = if dts == 0 {
                0
            } else {
                let d_s = ((dts as f64) / 90_000.0 + sh).max(0.0);
                ((d_s * 90_000.0).round().clamp(0.0, seg_pts as f64)) as u32
            };
            out.extend_from_slice(b"PG");
            out.extend_from_slice(&seg_pts.to_be_bytes());
            out.extend_from_slice(&seg_dts.to_be_bytes());
            out.push(kind);
            out.extend_from_slice(&(payload.len() as u16).to_be_bytes());
            out.extend_from_slice(payload);
        }
        ds_start = ds_end;
    }
    Ok(out)
}

/// Shifts DVD VobSub (`.idx`) cue timestamps while keeping `.sub` packet `filepos` offsets intact.
pub fn shift_vobsub_idx(content: &str, entry: &SubPlanEntry, max_out_s: Option<f64>) -> String {
    let nl = if content.contains("\r\n") { "\r\n" } else { "\n" };
    let mut out_lines = Vec::new();
    for raw_line in content.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if let Some(rest) = line.trim_start().strip_prefix("timestamp:") {
            if let Some((ts_part, comma_tail)) = rest.split_once(',') {
                let raw_ts = ts_part.trim();
                let (neg, clean_ts) = match raw_ts.strip_prefix('-') {
                    Some(s) => (true, s),
                    None => (false, raw_ts),
                };
                let parts: Vec<&str> = clean_ts.split(':').collect();
                if parts.len() == 4 {
                    if let (Ok(h), Ok(m), Ok(s), Ok(ms)) = (
                        parts[0].parse::<f64>(),
                        parts[1].parse::<f64>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<f64>(),
                    ) {
                        let sign = if neg { -1.0 } else { 1.0 };
                        let a = sign * (h * 3600.0 + m * 60.0 + s + ms / 1000.0);
                        let na = (a + sub_source_shift_at(entry, a.max(0.0))).max(0.0);
                        if max_out_s.is_some_and(|cap| na >= cap + 0.25) {
                            continue;
                        }
                        let total_ms = (na * 1000.0).round() as u64;
                        let tot_s = total_ms / 1000;
                        let f_ms = total_ms % 1000;
                        let hh = tot_s / 3600;
                        let mm = (tot_s % 3600) / 60;
                        let ss = tot_s % 60;
                        out_lines.push(format!("timestamp: {hh:02}:{mm:02}:{ss:02}:{f_ms:03},{comma_tail}"));
                        continue;
                    }
                }
            }
        }
        out_lines.push(line.to_string());
    }
    out_lines.join(nl)
}

fn render_shifted_sub(
    path: &Path,
    track: &TrackInfo,
    entry: &SubPlanEntry,
    work_dir: &Path,
    stem: &str,
    max_out_s: Option<f64>,
) -> Result<(PathBuf, Vec<PathBuf>)> {
    let cu = track.codec.to_ascii_uppercase();
    let is_ass = cu.contains("ASS") || cu.contains("SSA") || cu.contains("SUBSTATION");
    let is_srt = cu.contains("SRT") || cu.contains("UTF-8") || cu.contains("SUBRIP");
    let is_pgs = cu.contains("PGS") || cu.contains("HDMV");
    let is_vobsub = cu.contains("VOBSUB");
    if !is_ass && !is_srt && !is_pgs && !is_vobsub {
        bail!("subtítulo {} ({}) no soportado para tramos", track.n, track.codec);
    }
    let ext = if is_ass {
        "ass"
    } else if is_srt {
        "srt"
    } else if is_pgs {
        "sup"
    } else {
        "idx"
    };
    let out = work_dir.join(format!(".{stem}.s{}.shift.{ext}", track.n));
    let mut cleanup = vec![out.clone()];
    if is_vobsub {
        cleanup.push(out.with_extension("sub"));
    }
    let status = Command::new("mkvextract")
        .arg(path)
        .arg("tracks")
        .arg(format!("{}:{}", track.id, out.to_string_lossy()))
        .output()
        .context("mkvextract subtitle")?;
    if !status.status.success() {
        for p in &cleanup {
            let _ = std::fs::remove_file(p);
        }
        bail!("mkvextract no pudo extraer s:{}", track.n);
    }
    let write_res: Result<()> = (|| {
        let raw = std::fs::read(&out)?;
        if is_pgs {
            let shifted = shift_pgs_sup(&raw, entry, max_out_s)?;
            std::fs::write(&out, shifted)?;
        } else {
            let text = String::from_utf8(raw).context("subtítulo no UTF-8")?;
            let shifted = if is_ass {
                shift_ass_text(&text, entry, max_out_s)
            } else if is_srt {
                shift_srt_text(&text, entry, max_out_s)
            } else {
                shift_vobsub_idx(&text, entry, max_out_s)
            };
            std::fs::write(&out, shifted.as_bytes())?;
        }
        Ok(())
    })();
    if let Err(e) = write_res {
        for p in &cleanup {
            let _ = std::fs::remove_file(p);
        }
        return Err(e);
    }
    Ok((out, cleanup))
}

fn push_extra_track(
    extra_args: &mut Vec<String>,
    order: &mut Vec<String>,
    file_index: usize,
    t: &TrackInfo,
    delay_ms: i64,
    f: &Path,
) {
    extra_args.extend([
        "--language".into(),
        format!("0:{}", t.lang),
        "--track-name".into(),
        format!("0:{}", t.name),
        "--default-track-flag".into(),
        format!("0:{}", if t.default { 1 } else { 0 }),
        "--sync".into(),
        format!("0:{delay_ms}"),
        f.to_string_lossy().into_owned(),
    ]);
    order.push(format!("{file_index}:0"));
}

pub fn apply_with_ref(
    path: &Path,
    plan: &Plan,
    out: &Path,
    work: Option<&Path>,
    ref_track: usize,
) -> Result<PathBuf> {
    if let (Ok(cp), Ok(co)) = (std::fs::canonicalize(path), std::fs::canonicalize(out)) {
        if cp == co {
            bail!("la salida no puede ser el archivo original");
        }
    }
    let out_parent = out.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(out_parent)?;
    let work_dir = work.unwrap_or(out_parent);
    std::fs::create_dir_all(work_dir)?;

    let info = media::probe(path)?;
    let ends = media::stream_end_times(path, &info).unwrap_or_default();
    let ref_dur = info
        .tracks
        .iter()
        .find(|t| t.kind == "audio" && t.n == ref_track)
        .and_then(|t| ends.get(&t.id).copied());

    let mut main_args: Vec<String> = Vec::new();
    let mut extra_args: Vec<String> = Vec::new();
    let mut order: Vec<String> = Vec::new();
    let mut keep_audio: Vec<String> = Vec::new();
    let mut keep_subs: Vec<String> = Vec::new();
    let mut any_extra_audio = false;
    let mut any_extra_subs = false;
    let mut temps: Vec<PathBuf> = Vec::new();
    let mut cleanup_only: Vec<PathBuf> = Vec::new();

    let stem = out
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out".to_string());

    let res = (|| -> Result<PathBuf> {
        for t in &info.tracks {
            if t.kind == "audio" {
                let p = plan.audio.get(&t.n.to_string());
                let mode = p.map(|e| e.mode.as_str()).unwrap_or("none");
                if mode == "segments" || mode == "tempo" {
                    let entry = p.unwrap();
                    let (f, delay) = if mode == "segments" {
                        let codec = entry.codec.as_deref().unwrap_or("auto");
                        let segs = entry.segments.as_deref().unwrap_or(&[]);
                        render_segments(
                            path,
                            t,
                            segs,
                            work_dir,
                            &stem,
                            codec,
                            ref_dur,
                            ends.get(&t.id).copied(),
                        )?
                    } else {
                        let codec = entry.codec.as_deref().unwrap_or("flac");
                        let ext = if codec == "aac" { ".mka" } else { ".flac" };
                        let f = work_dir.join(format!(".{stem}.a{}{ext}", t.n));
                        let tp = entry.tempo.unwrap_or(1.0);
                        let pitch_correct = entry.pitch_correct.unwrap_or(false);
                        let d_ms = entry.delay_ms.unwrap_or(0);
                        let cap = ref_dur.map(|rd| (rd - (d_ms as f64 / 1000.0)).max(0.1));
                        if let Err(e) = render_tempo(path, t, tp, pitch_correct, &f, codec, cap) {
                            let _ = std::fs::remove_file(&f);
                            return Err(e);
                        }
                        (f, d_ms)
                    };
                    temps.push(f.clone());
                    any_extra_audio = true;
                    extra_args.extend([
                        "--language".into(),
                        format!("0:{}", t.lang),
                        "--track-name".into(),
                        format!("0:{}", t.name),
                        "--default-track-flag".into(),
                        format!("0:{}", if t.default { 1 } else { 0 }),
                        "--sync".into(),
                        format!("0:{delay}"),
                        f.to_string_lossy().into_owned(),
                    ]);
                    order.push(format!("{}:0", temps.len()));
                    continue;
                }

                let delay = if mode == "delay" {
                    p.and_then(|e| e.delay_ms).unwrap_or(0)
                } else {
                    0
                };
                let delay_s = (delay as f64) / 1000.0;
                let need_trim = t.n != ref_track
                    && ref_dur.is_some()
                    && ends
                        .get(&t.id)
                        .is_some_and(|&sd| sd + delay_s > ref_dur.unwrap() + 0.25);
                if need_trim {
                    let rd = ref_dur.unwrap();
                    let trim_dur = (rd - delay_s).max(0.1);
                    let mka_f = work_dir.join(format!(".{stem}.a{}.trim.mka", t.n));
                    let f = if trim_stream_copy(path, 'a', t.n, trim_dur, &mka_f).is_ok() {
                        mka_f
                    } else {
                        let _ = std::fs::remove_file(&mka_f);
                        let flac_f = work_dir.join(format!(".{stem}.a{}.trim.flac", t.n));
                        render_trim_pcm32_flac(path, t.n, trim_dur, &flac_f)?;
                        flac_f
                    };
                    temps.push(f.clone());
                    any_extra_audio = true;
                    extra_args.extend([
                        "--language".into(),
                        format!("0:{}", t.lang),
                        "--track-name".into(),
                        format!("0:{}", t.name),
                        "--default-track-flag".into(),
                        format!("0:{}", if t.default { 1 } else { 0 }),
                        "--sync".into(),
                        format!("0:{delay}"),
                        f.to_string_lossy().into_owned(),
                    ]);
                    order.push(format!("{}:0", temps.len()));
                    continue;
                }

                keep_audio.push(t.id.to_string());
                if mode == "delay" && delay != 0 {
                    main_args.extend(["--sync".into(), format!("{}:{delay}", t.id)]);
                }
            } else if t.kind == "subtitles" {
                let sub_entry = plan.subs.get(&t.n.to_string());
                let has_non_const = sub_entry.is_some_and(|e| {
                    e.segments.as_ref().is_some_and(|v| !v.is_empty())
                        || e.tempo.is_some_and(|tp| (tp - 1.0).abs() > 1e-12)
                });
                if has_non_const {
                    let entry = sub_entry.unwrap();
                    if let Ok((f, extra_temps)) =
                        render_shifted_sub(path, t, entry, work_dir, &stem, ref_dur)
                    {
                        any_extra_subs = true;
                        // Note: for VobSub (.idx + .sub), mkvmerge reads ONLY the .idx input file!
                        // Keep file_index aligned with input files passed to mkvmerge (`mux_inputs` count).
                        temps.push(f.clone());
                        let mux_file_idx = temps.len();
                        push_extra_track(&mut extra_args, &mut order, mux_file_idx, t, 0, &f);
                        for extra_p in extra_temps {
                            if extra_p != f {
                                cleanup_only.push(extra_p);
                            }
                        }
                        continue;
                    }
                }
                let ms = sub_entry.and_then(|e| e.delay_ms).unwrap_or(0);
                let delay_s = (ms as f64) / 1000.0;
                let need_trim = ref_dur.is_some()
                    && ends
                        .get(&t.id)
                        .is_some_and(|&sd| sd + delay_s > ref_dur.unwrap() + 0.25);
                if need_trim {
                    let rd = ref_dur.unwrap();
                    let trim_dur = (rd - delay_s).max(0.1);
                    let f = work_dir.join(format!(".{stem}.s{}.trim.mks", t.n));
                    if trim_stream_copy(path, 's', t.n, trim_dur, &f).is_ok() {
                        temps.push(f.clone());
                        any_extra_subs = true;
                        push_extra_track(&mut extra_args, &mut order, temps.len(), t, ms, &f);
                        continue;
                    }
                    let _ = std::fs::remove_file(&f);
                }

                keep_subs.push(t.id.to_string());
                if ms != 0 {
                    main_args.extend(["--sync".into(), format!("{}:{ms}", t.id)]);
                }
            }
            order.push(format!("0:{}", t.id));
        }

        if any_extra_audio {
            if !keep_audio.is_empty() {
                main_args.extend(["--audio-tracks".into(), keep_audio.join(",")]);
            } else {
                main_args.push("--no-audio".into());
            }
        }
        if any_extra_subs {
            if !keep_subs.is_empty() {
                main_args.extend(["--subtitle-tracks".into(), keep_subs.join(",")]);
            } else {
                main_args.push("--no-subtitles".into());
            }
        }

        let tmp_out = out.with_file_name(format!(".{stem}.tmp.mkv"));
        let mut cmd = Command::new("mkvmerge");
        cmd.arg("-q")
            .arg("-o")
            .arg(&tmp_out)
            .arg("--track-order")
            .arg(order.join(","))
            .args(&main_args)
            .arg(path)
            .args(&extra_args);

        let out_res = cmd.output().context("failed to run mkvmerge")?;
        let code = out_res.status.code().unwrap_or(2);
        if code > 1 {
            let _ = std::fs::remove_file(&tmp_out);
            let stdout_s = String::from_utf8_lossy(&out_res.stdout);
            let stderr_s = String::from_utf8_lossy(&out_res.stderr);
            let msg = if !stdout_s.trim().is_empty() {
                stdout_s.trim()
            } else {
                stderr_s.trim()
            };
            bail!("mkvmerge falló: {msg}");
        }
        std::fs::rename(&tmp_out, out)?;
        Ok(out.to_path_buf())
    })();

    for f in temps {
        let _ = std::fs::remove_file(f);
    }
    for f in cleanup_only {
        let _ = std::fs::remove_file(f);
    }
    res
}

pub fn apply(path: &Path, plan: &Plan, out: &Path, work: Option<&Path>) -> Result<PathBuf> {
    apply_with_ref(path, plan, out, work, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnose_constante_drift_cortes() {
        // Constante with 1 outlier
        let pts_const: Vec<(f64, f64, f64)> = (0..10)
            .map(|i| {
                let d = if i == 4 { 0.500 } else { 1.242 + (i as f64) * 0.0005 };
                (i as f64 * 100.0, d, 20.0)
            })
            .collect();
        let (d, tp, kind, dropped) = diagnose(&pts_const);
        assert_eq!(kind, "constante");
        assert!((tp - 1.0).abs() < 1e-9);
        assert_eq!(dropped, 1);
        assert!((d - 1.2445).abs() < 0.005);

        // Drift (linear)
        let pts_drift: Vec<(f64, f64, f64)> = (0..10)
            .map(|i| {
                let t = i as f64 * 100.0;
                (t, 0.015 + 0.001 * t, 20.0)
            })
            .collect();
        let (d2, tp2, kind2, _) = diagnose(&pts_drift);
        assert_eq!(kind2, "drift");
        assert!((d2 - 0.015).abs() < 1e-4);
        assert!((tp2 - 1.0 / 1.001).abs() < 1e-6);

        // Cortes (two adjacent windows shifted)
        let pts_cuts: Vec<(f64, f64, f64)> = (0..10)
            .map(|i| (i as f64 * 100.0, if i < 5 { 0.0 } else { 0.55 }, 20.0))
            .collect();
        let (_, _, kind3, _) = diagnose(&pts_cuts);
        assert_eq!(kind3, "cortes");
    }

    #[test]
    fn test_coverage_ok() {
        let pts = vec![(10.0, 0.0, 20.0), (500.0, 0.0, 20.0), (1100.0, 0.0, 20.0)];
        assert!(coverage_ok(&pts, 1400.0));
        let pts_bad = vec![(10.0, 0.0, 20.0), (50.0, 0.0, 20.0), (100.0, 0.0, 20.0)];
        assert!(!coverage_ok(&pts_bad, 1400.0));
    }

    #[test]
    fn test_plan_is_empty_and_shift_at() {
        let mut plan = Plan::default();
        assert!(plan_is_empty(&plan));
        plan.audio.insert(
            "1".into(),
            AudioPlanEntry {
                mode: "delay".into(),
                delay_ms: Some(0),
                ..Default::default()
            },
        );
        assert!(plan_is_empty(&plan));
        plan.audio.insert(
            "2".into(),
            AudioPlanEntry {
                mode: "trim".into(),
                ..Default::default()
            },
        );
        assert!(!plan_is_empty(&plan));

        let seg_entry = AudioPlanEntry {
            mode: "segments".into(),
            segments: Some(vec![
                Segment {
                    start: 0.0,
                    end: Some(100.0),
                    delay: 0.0,
                },
                Segment {
                    start: 100.0,
                    end: None,
                    delay: 0.55,
                },
            ]),
            ..Default::default()
        };
        assert!((shift_at(Some(&seg_entry), 50.0) - 0.0).abs() < 1e-9);
        assert!((shift_at(Some(&seg_entry), 150.0) - 0.55).abs() < 1e-9);
    }

    fn mk_audio(n: usize, lang: &str, status: &str, delay_ms: Option<i64>) -> AudioReport {
        AudioReport {
            n,
            id: n as u64,
            lang: lang.to_string(),
            name: String::new(),
            codec: "aac".to_string(),
            channels: Some(2),
            status: status.to_string(),
            delay_ms,
            tempo: 1.0,
            points: vec![],
            dense: vec![],
            segments: vec![],
            reason: None,
            duration_s: Some(1420.0),
            ref_fps: None,
            src_fps: None,
            pitch_shift: None,
            pitch_semitones: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mk_sub(
        n: usize,
        lang: &str,
        name: &str,
        status: &str,
        offset_ms: Option<i64>,
        conf: f64,
        events: Vec<(f64, f64)>,
    ) -> SubReport {
        SubReport {
            n,
            id: (10 + n) as u64,
            lang: lang.to_string(),
            name: name.to_string(),
            codec: "ass".to_string(),
            status: status.to_string(),
            offset_ms,
            conf,
            events,
            twin_delay_ms: None,
            twin_segments: vec![],
            twin_tempo: None,
        }
    }

    #[test]
    fn test_apply_sub_twin_inherits_unknown_signs_and_preserves_normal_subs() {
        let audio = vec![
            mk_audio(0, "jpn", "ref", Some(0)),
            mk_audio(1, "eng", "offset", Some(1242)),
            mk_audio(2, "spa", "offset", Some(-180)),
            mk_audio(3, "por", "ok", Some(10)),
            mk_audio(4, "und", "offset", Some(900)),
            mk_audio(5, "fra", "offset", Some(500)),
        ];
        let many_events = vec![(1.0, 3.0); 25];
        let mut subs = vec![
            // s:0 spa normal dialogue synced to jpn (+100 ms, high conf) -> must stay ok
            mk_sub(0, "spa", "Latin American Spanish", "ok", Some(100), 0.62, many_events.clone()),
            // s:1 spa Signs (low conf noisy +19220 ms) -> must inherit -180 ms from spa audio
            mk_sub(1, "spa", "Latin American Spanish Signs", "unknown", Some(19220), 0.08, vec![(5.0, 7.0)]),
            // s:2 spa empty Signs (100% \pos/\move filtered) -> must inherit -180 ms
            mk_sub(2, "spa", "Signs PosOnly", "unknown", None, 0.0, vec![]),
            // s:3 eng CC measurable near eng audio offset (+980 vs +1242) -> inherit 1242 ms
            mk_sub(3, "eng", "English CC", "offset", Some(980), 0.57, vec![(2.0, 4.0)]),
            // s:4 por Signs when por audio is ok -> stays unknown
            mk_sub(4, "por", "Portuguese Signs", "unknown", Some(4500), 0.10, vec![(4.0, 6.0)]),
            // s:5 und Signs -> never inherit und
            mk_sub(5, "und", "Unknown Signs", "unknown", None, 0.0, vec![]),
            // s:6 fra normal dialogue at +260 ms (`ok` vs jpn) when fra audio is +500 ms -> MUST NOT inherit!
            mk_sub(6, "fra", "French", "ok", Some(260), 0.60, many_events.clone()),
            // s:7 spa normal dialogue with low conf alongside Signs sibling -> MUST NOT inherit!
            mk_sub(7, "spa", "Latin American Spanish", "unknown", Some(320), 0.25, vec![(2.0, 4.0)]),
            // s:8 eng normal dialogue at +410 ms while eng audio is +1242 ms -> keep own offset, no twin override
            mk_sub(8, "eng", "English Full", "offset", Some(410), 0.55, many_events),
        ];

        assert!(apply_sub_twin(&audio, &mut subs));
        assert!(!apply_sub_twin(&audio, &mut subs)); // idempotent

        assert_eq!(subs[0].status, "ok");
        assert_eq!(subs[0].twin_delay_ms, None);

        assert_eq!(subs[1].status, "offset");
        assert_eq!(subs[1].twin_delay_ms, Some(-180));

        assert_eq!(subs[2].status, "offset");
        assert_eq!(subs[2].twin_delay_ms, Some(-180));

        assert_eq!(subs[3].status, "offset");
        assert_eq!(subs[3].twin_delay_ms, Some(1242));

        assert_eq!(subs[4].status, "unknown");
        assert_eq!(subs[4].twin_delay_ms, None);

        assert_eq!(subs[5].status, "unknown");
        assert_eq!(subs[5].twin_delay_ms, None);

        assert_eq!(subs[6].status, "ok");
        assert_eq!(subs[6].twin_delay_ms, None);

        assert_eq!(subs[7].status, "unknown");
        assert_eq!(subs[7].twin_delay_ms, None);

        assert_eq!(subs[8].status, "offset");
        assert_eq!(subs[8].twin_delay_ms, None);

        let report = Report {
            file: "ep.mkv".into(),
            duration: Some(1420.0),
            video_fps: Some(23.976),
            ref_track: 0,
            status: recompute_worst_status(&audio, &subs),
            suggest_start: 0.0,
            audio,
            subs,
        };
        assert_eq!(report.status, "offset");
        let plan = propose(&report);
        assert!(plan.subs.get("0").is_none());
        assert_eq!(plan.subs.get("1").and_then(|e| e.delay_ms), Some(-180));
        assert_eq!(plan.subs.get("2").and_then(|e| e.delay_ms), Some(-180));
        assert_eq!(plan.subs.get("3").and_then(|e| e.delay_ms), Some(1242));
        assert!(plan.subs.get("4").is_none());
        assert!(plan.subs.get("5").is_none());
        assert!(plan.subs.get("6").is_none());
        assert!(plan.subs.get("7").is_none());
        assert_eq!(plan.subs.get("8").and_then(|e| e.delay_ms), Some(410));
    }

    #[test]
    fn test_recompute_worst_status_sub_offset_only() {
        let audio = vec![mk_audio(0, "jpn", "ref", Some(0)), mk_audio(1, "spa", "ok", Some(0))];
        let mut subs = vec![mk_sub(0, "spa", "Signs", "offset", Some(19220), 0.08, vec![(1.0, 2.0)])];
        subs[0].twin_delay_ms = Some(1253);
        assert_eq!(recompute_worst_status(&audio, &subs), "offset");
    }

    #[test]
    fn test_signs_inherit_cuts_and_shift_ass_piecewise() {
        let mut audio_spa = mk_audio(2, "spa", "cuts", Some(-1));
        audio_spa.segments = vec![
            Segment { start: 0.0, end: Some(689.195), delay: -0.003 },
            Segment { start: 689.195, end: Some(959.022), delay: 0.498 },
            Segment { start: 959.022, end: None, delay: 1.456 },
        ];
        let audio = vec![
            mk_audio(0, "jpn", "ref", Some(0)),
            mk_audio(1, "eng", "ok", Some(-10)),
            audio_spa,
        ];
        let mut subs = vec![
            mk_sub(0, "spa", "Latin American Spanish", "ok", Some(40), 0.64, vec![(1.0, 2.0); 25]),
            mk_sub(1, "spa", "Latin American Spanish Signs", "unknown", Some(-5820), 0.05, vec![(60.0, 62.0), (1413.47, 1416.89)]),
            mk_sub(4, "eng", "English", "ok", Some(40), 0.63, vec![(1.0, 2.0); 25]),
            mk_sub(5, "eng", "English Signs", "unknown", Some(19280), -0.24, vec![(10.0, 12.0)]),
        ];

        assert!(apply_sub_twin(&audio, &mut subs));
        assert!(!apply_sub_twin(&audio, &mut subs));
        assert_eq!(subs[0].status, "ok");
        assert_eq!(subs[1].status, "cuts");
        assert_eq!(subs[1].twin_segments.len(), 3);
        assert_eq!(subs[2].status, "ok");
        assert_eq!(subs[3].status, "unknown"); // eng audio is ok -> eng Signs stays untouched

        let report = Report {
            file: "s02e18.mkv".into(),
            duration: Some(1420.0),
            video_fps: Some(23.976),
            ref_track: 0,
            status: recompute_worst_status(&audio, &subs),
            suggest_start: 0.0,
            audio,
            subs,
        };
        let plan = propose(&report);
        assert!(!plan_is_empty(&plan));
        let sub_plan = plan.subs.get("1").expect("spa Signs must have cuts plan");
        assert_eq!(sub_plan.segments.as_ref().unwrap().len(), 3);
        assert!(plan.subs.get("0").is_none());
        assert!(plan.subs.get("5").is_none());

        let ass_in = "\u{feff}[Script Info]\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:01:00.00,0:01:02.00,Default,,0,0,0,,Inicio\nDialogue: 0,0:13:50.01,0:13:50.91,Default,,0,0,0,,Mitad\nDialogue: 0,0:23:33.47,0:23:36.89,Default,,0,0,0,,Final\nDialogue: 0,0:24:03.04,0:24:04.46,Default,,0,0,0,,Sobrante\n";
        let ass_out = shift_ass_text(ass_in, sub_plan, Some(1420.053));
        assert!(ass_out.contains("Dialogue: 0,0:01:00.00,0:01:02.00,Default,,0,0,0,,Inicio"));
        assert!(ass_out.contains("Dialogue: 0,0:13:50.51,0:13:51.41,Default,,0,0,0,,Mitad"));
        assert!(ass_out.contains("Dialogue: 0,0:23:34.93,0:23:38.35,Default,,0,0,0,,Final"));
        assert!(!ass_out.contains("Sobrante"));
    }

    #[test]
    fn test_parse_adts_and_ac3_frames() {
        // Build two minimal valid 48 kHz stereo ADTS frames (sf_idx = 3, ch_cfg = 2, lengths 10 and 12)
        let mk_adts = |len: usize| -> Vec<u8> {
            let mut v = vec![0u8; len];
            v[0] = 0xFF;
            v[1] = 0xF1; // MPEG-4 ADTS, protection_absent=1
            v[2] = (1 << 6) | (3 << 2) | 0; // AAC-LC, 48kHz (idx 3), ch bit2=0
            v[3] = (2 << 6) | (((len >> 11) & 3) as u8);
            v[4] = ((len >> 3) & 0xFF) as u8;
            v[5] = (((len & 7) << 5) as u8) | 0x1F;
            v[6] = 0xFC; // num_raw_data_blocks = 0
            v
        };
        let mut adts = mk_adts(10);
        adts.extend_from_slice(&mk_adts(12));
        let (frames, sr) = parse_adts_frames(&adts).unwrap();
        assert_eq!(sr, 48000);
        assert_eq!(frames.len(), 2);
        assert_eq!((frames[0].start, frames[0].end, frames[0].samples), (0, 10, 1024));
        assert_eq!((frames[1].start, frames[1].end), (10, 22));
        assert!(parse_adts_frames(&adts[..21]).is_err());

        // Minimal AC3 frame: fscod=0 (48kHz), frmsizecod=0 -> 64 words = 128 bytes; bsid=8
        let mut ac3 = vec![0u8; 128];
        ac3[0] = 0x0B;
        ac3[1] = 0x77;
        ac3[4] = 0x00;
        ac3[5] = 8 << 3;
        let (ac3_frames, ac3_sr) = parse_ac3_or_eac3_frames(&ac3, false).unwrap();
        assert_eq!(ac3_sr, 48000);
        assert_eq!(ac3_frames.len(), 1);
        assert_eq!(ac3_frames[0].end, 128);
        assert_eq!(ac3_frames[0].samples, 1536);

        // E-AC3 independent frame (bsid=16, strmtyp=0, frmsiz=63 -> 128 bytes, fscod=0, numblkscod=3 -> 6 blocks = 1536)
        let mut eac3 = vec![0u8; 128];
        eac3[0] = 0x0B;
        eac3[1] = 0x77;
        eac3[2] = 0x00;
        eac3[3] = 63;
        eac3[4] = 0x30; // fscod=0, numblkscod=3
        eac3[5] = 16 << 3;
        let (e_frames, e_sr) = parse_ac3_or_eac3_frames(&eac3, true).unwrap();
        assert_eq!(e_sr, 48000);
        assert_eq!(e_frames[0].samples, 1536);

        // Dependent E-AC3 substream (strmtyp=1) must be rejected
        eac3[2] |= 1 << 6;
        assert!(parse_ac3_or_eac3_frames(&eac3, true).is_err());
    }

    #[test]
    fn test_plan_frame_edits_dup_drop_trim_and_loud_fallback() {
        let fd = 0.020; // 50 fps exact = 20 ms/frame for deterministic testing
        // 1) Positive gap (+0.200s = 10 frames) + negative cut (-0.100s = 5 frames drop) + tail trim
        let segs = vec![
            Segment { start: 0.0, end: Some(2.0), delay: 0.0 },
            Segment { start: 2.0, end: Some(4.0), delay: 0.200 },
            Segment { start: 4.0, end: None, delay: 0.100 },
        ];
        // Total source = 6.0s (300 frames). Target ref_dur = 5.0s -> trims trailing >5.25s cleanly
        let (emit, init_ms) = plan_frame_edits(300, fd, 0.0, &segs, Some(5.0), |lo, hi| {
            Ok(((lo + hi) / 2, -60.0))
        })
        .unwrap();
        assert_eq!(init_ms, 0);
        // 5.0s / 0.020 = 250 frames
        assert_eq!(emit.len(), 250);
        // Check duplicated frames occurred around frame 100 (10 duplicates of frame 100)
        assert_eq!(emit.iter().filter(|&&f| f == 100).count(), 11);

        // 2) Negative initial delay (-0.100s) drops first 5 frames + positive gap later
        let segs_lead = vec![
            Segment { start: 0.0, end: Some(1.0), delay: -0.100 },
            Segment { start: 1.0, end: None, delay: 0.0 },
        ];
        let (emit2, init_ms2) =
            plan_frame_edits(100, fd, 0.0, &segs_lead, None, |lo, hi| Ok((lo.min(hi), -55.0)))
                .unwrap();
        assert_eq!(init_ms2, 0);
        assert_eq!(emit2[0], 5);

        // 3) Loud audio (-15 dBFS) across a >=3 frame gap rejects lossless dup so PCM32->FLAC takes over
        let res_loud = plan_frame_edits(300, fd, 0.0, &segs, None, |lo, hi| Ok(((lo + hi) / 2, -15.0)));
        assert!(res_loud.is_err());
    }

    #[test]
    fn test_shift_at_tempo_and_seam_crossing_ass_pgs_vobsub() {
        // 1) shift_at with tempo=2.0 (atempo=2x faster): output t=50s came from source t=100s -> shift = -50s
        let tempo_entry = AudioPlanEntry {
            mode: "tempo".into(),
            tempo: Some(2.0),
            delay_ms: Some(0),
            ..Default::default()
        };
        assert!((shift_at(Some(&tempo_entry), 50.0) - (-50.0)).abs() < 1e-9);

        // 2) Cue crossing segment seam [9.0..11.0s] across cut at 10.0s (delay 0.0 -> +1.5s) preserves 2.0s duration!
        let sub_plan = SubPlanEntry {
            delay_ms: None,
            segments: Some(vec![
                Segment { start: 0.0, end: Some(10.0), delay: 0.0 },
                Segment { start: 10.0, end: None, delay: 1.5 },
            ]),
            tempo: None,
        };
        let ass = "Dialogue: 0,0:00:09.00,0:00:11.00,Default,,0,0,0,,{\\fad(500,500)}Seam\nDialogue: 0,0:00:12.00,0:00:14.00,Default,,0,0,0,,After\n";
        let ass_out = shift_ass_text(ass, &sub_plan, Some(20.0));
        assert!(ass_out.contains("Dialogue: 0,0:00:09.00,0:00:11.00,Default,,0,0,0,,{\\fad(500,500)}Seam"));
        assert!(ass_out.contains("Dialogue: 0,0:00:13.50,0:00:15.50,Default,,0,0,0,,After"));

        // 3) PGS (.sup) 90kHz PTS shifting & cap dropping
        let mk_pgs_ds = |pts_s: f64| -> Vec<u8> {
            let ticks = (pts_s * 90_000.0).round() as u32;
            let mut v = Vec::new();
            // PCS segment (0x16) with 2-byte dummy payload + END segment (0x80)
            v.extend_from_slice(b"PG");
            v.extend_from_slice(&ticks.to_be_bytes());
            v.extend_from_slice(&0u32.to_be_bytes());
            v.push(0x16);
            v.extend_from_slice(&2u16.to_be_bytes());
            v.extend_from_slice(&[0xAA, 0xBB]);
            v.extend_from_slice(b"PG");
            v.extend_from_slice(&ticks.to_be_bytes());
            v.extend_from_slice(&0u32.to_be_bytes());
            v.push(0x80);
            v.extend_from_slice(&0u16.to_be_bytes());
            v
        };
        let mut sup = mk_pgs_ds(12.0); // in seg 2 (+1.5s -> 13.5s = 1_215_000 ticks)
        sup.extend_from_slice(&mk_pgs_ds(25.0)); // past cap=20.0 -> dropped!
        let sup_out = shift_pgs_sup(&sup, &sub_plan, Some(20.0)).unwrap();
        assert_eq!(sup_out.len(), 13 + 2 + 13);
        let new_ticks = u32::from_be_bytes([sup_out[2], sup_out[3], sup_out[4], sup_out[5]]);
        assert_eq!(new_ticks, 1_215_000);

        // 4) VobSub (.idx) timestamp shifting & cap dropping
        let idx_in = "# VobSub index\ntimestamp: 00:00:12:000, filepos: 000000800\ntimestamp: 00:00:25:000, filepos: 000001000\n";
        let idx_out = shift_vobsub_idx(idx_in, &sub_plan, Some(20.0));
        assert!(idx_out.contains("timestamp: 00:00:13:500, filepos: 000000800"));
        assert!(!idx_out.contains("000001000"));
    }

    #[test]
    fn test_fps_inference_and_varispeed_vs_pitch_locked_tempo() {
        // 1) FPS inference & semitones math
        assert_eq!(infer_fps_pair(Some(24000.0 / 1001.0), FILM / 25.0), (Some(23.976), Some(25.0)));
        assert_eq!(infer_fps_pair(Some(25.0), 25.0 / FILM), (Some(25.0), Some(23.976)));
        assert_eq!(infer_fps_pair(Some(24.0), 24.0 / 25.0), (Some(24.0), Some(25.0)));
        assert_eq!(infer_fps_pair(Some(23.976), 1000.0 / 1001.0), (Some(23.976), Some(24.0)));
        assert_eq!(infer_fps_pair(None, FILM / 25.0), (Some(23.976), Some(25.0)));
        // Same FPS / non-FPS clock drift MUST NOT return equal (23.976, 23.976)!
        assert_eq!(infer_fps_pair(Some(23.976), 0.9999172), (None, None));
        assert_eq!(infer_fps_pair(Some(23.976), 0.99970), (None, None));
        assert_eq!(pitch_semitones_from_tempo(FILM / 25.0), 0.72);
        assert_eq!(pitch_semitones_from_tempo(25.0 / FILM), -0.72);

        // False-positive micro-drift (-26 ms .. +58 ms step, x0.9999172) must NOT be classified as "drift"!
        let pts_micro_step: Vec<(f64, f64, f64)> = (0..10)
            .map(|i| (i as f64 * 140.0, if i < 5 { -0.025 } else { 0.058 }, 20.0))
            .collect();
        let (_, tp_ms, kind_ms, _) = diagnose(&pts_micro_step);
        assert_ne!(kind_ms, "drift");
        assert!((tp_ms - 1.0).abs() < 1e-9);

        // Cached report with legacy x0.9999172 micro-tempo is cleaned up automatically by populate_fps_info
        let mut legacy_audio = mk_audio(1, "spa", "tempo", Some(-26));
        legacy_audio.tempo = 0.9999172;
        legacy_audio.ref_fps = Some(23.976);
        legacy_audio.src_fps = Some(23.976);
        legacy_audio.points = (0..10)
            .map(|i| MeasurePoint(i as f64 * 140.0, if i < 5 { -25 } else { 35 }, 20.0))
            .collect();
        let mut legacy_rep = Report {
            file: "legacy.mkv".into(),
            duration: Some(1420.0),
            video_fps: Some(23.976),
            ref_track: 0,
            status: "tempo".into(),
            suggest_start: 0.0,
            audio: vec![mk_audio(0, "jpn", "ref", Some(0)), legacy_audio],
            subs: vec![],
        };
        assert!(populate_fps_info(&mut legacy_rep));
        assert_eq!(legacy_rep.audio[1].status, "ok");
        assert!((legacy_rep.audio[1].tempo - 1.0).abs() < 1e-9);
        assert_eq!(legacy_rep.audio[1].ref_fps, None);
        assert_eq!(legacy_rep.status, "ok");

        // 2) Synthesize 180s reference at 23.976 fps with unique rhythmic bursts + A-major harmonic chord
        let dur_s = 180.0f64;
        let n_ref = (dur_s * SR as f64) as usize;
        let n_slots = 1800usize; // 10 slots/sec
        let make_gates = |mut seed: u64| -> Vec<f64> {
            (0..n_slots + 16)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let r = ((seed >> 33) as f64) / ((1u64 << 31) as f64);
                    if r > 0.62 { 0.15 + 0.85 * r } else { 0.02 }
                })
                .collect()
        };
        let gates_ep1 = make_gates(0xCAFE_BABE_1234_5678);
        let gates_ep2 = make_gates(0xDEAD_BEEF_8765_4321);

        let env_at = |gates: &[f64], u_sec: f64| -> f64 {
            let pos = (u_sec * 10.0).max(0.0);
            let idx = (pos as usize).min(gates.len() - 2);
            let frac = pos - (idx as f64);
            // Smooth raised-cosine envelope across each 100 ms slot
            let w = 0.5 - 0.5 * (PI * frac).cos();
            gates[idx] * (1.0 - w) + gates[idx + 1] * w
        };
        let chord_at = |t: f64| -> f64 {
            0.45 * (2.0 * PI * 440.0 * t).sin()
                + 0.30 * (2.0 * PI * 554.37 * t).sin()
                + 0.25 * (2.0 * PI * 659.25 * t).sin()
                + 0.15 * (2.0 * PI * 880.0 * t).sin()
                + 0.10 * (2.0 * PI * 1318.5 * t).sin()
        };

        let ref_sig: Vec<f32> = (0..n_ref)
            .map(|i| {
                let t = (i as f64) / (SR as f64);
                (env_at(&gates_ep1, t) * chord_at(t)) as f32
            })
            .collect();

        let tp_pal = FILM / 25.0; // 0.95904096

        // Varispeed PAL 25 fps track (both timing AND pitch sped up by 25 / 23.976)
        let src_vari = stretch(&ref_sig, 1.0 / tp_pal);

        // Pitch-locked PAL 25 fps track (timing sped up by 25 / 23.976, chord pitch kept at 440 Hz!)
        let n_pal = ((dur_s * tp_pal) * SR as f64) as usize;
        let src_locked: Vec<f32> = (0..n_pal)
            .map(|i| {
                let t = (i as f64) / (SR as f64);
                let u_ep = t / tp_pal;
                (env_at(&gates_ep1, u_ep) * chord_at(t)) as f32
            })
            .collect();

        // Unrelated episode (must stay "unknown")
        let wrong_ep: Vec<f32> = (0..n_ref)
            .map(|i| {
                let t = (i as f64) / (SR as f64);
                (env_at(&gates_ep2, t) * chord_at(t * 1.12)) as f32
            })
            .collect();

        let tempo_cache = RefFftCache::new(&ref_sig, ref_sig.len());
        let full_cache = RefFftCache::new(&ref_sig, ref_sig.len() * 2 + 16);
        let mut t_sc = tempo_cache.make_scratch();
        let mut f_sc = full_cache.make_scratch();

        let a_vari = analyze_audio(
            &ref_sig, &tempo_cache, &mut t_sc, &full_cache, &mut f_sc, &src_vari, 12, 15.0, 5.0, 10.0,
        );
        assert_eq!(a_vari.status, "tempo");
        assert!((a_vari.tempo - tp_pal).abs() < 0.0005, "vari tempo={}", a_vari.tempo);
        assert_eq!(a_vari.pitch_shift, Some(true), "varispeed PAL must detect pitch_shift=true");

        let a_locked = analyze_audio(
            &ref_sig, &tempo_cache, &mut t_sc, &full_cache, &mut f_sc, &src_locked, 12, 15.0, 5.0, 10.0,
        );
        assert_eq!(a_locked.status, "tempo");
        assert!((a_locked.tempo - tp_pal).abs() < 0.0005, "locked tempo={}", a_locked.tempo);
        assert_eq!(a_locked.pitch_shift, Some(false), "pitch-locked PAL must detect pitch_shift=false");

        let a_wrong = analyze_audio(
            &ref_sig, &tempo_cache, &mut t_sc, &full_cache, &mut f_sc, &wrong_ep, 12, 15.0, 5.0, 10.0,
        );
        assert_eq!(a_wrong.status, "unknown");
    }
}
