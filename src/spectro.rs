use crate::dsp::mean_std_f32;
use crate::media;
use crate::sync_algo::transient;
use crate::timeline::MAGMA;
use anyhow::Result;
use fontdue::layout::{CoordinateSystem, Layout, LayoutSettings, TextStyle};
use fontdue::Font;
use rayon::prelude::*;
use realfft::RealFftPlanner;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;
use std::sync::OnceLock;
use tiny_skia::{Color, Paint, PathBuilder, Pixmap, Rect, Stroke, Transform};

pub const SR: usize = 16000;

const TAB10: [[u8; 3]; 10] = [
    [31, 119, 180],
    [255, 127, 14],
    [44, 160, 44],
    [214, 39, 40],
    [148, 103, 189],
    [140, 86, 75],
    [227, 119, 194],
    [127, 127, 127],
    [188, 189, 34],
    [23, 190, 207],
];

fn get_font() -> Option<&'static Font> {
    static FONT: OnceLock<Option<Font>> = OnceLock::new();
    FONT.get_or_init(|| {
        let mut candidates = Vec::new();
        if let Ok(p) = std::env::var("DUBALIGN_FONT") {
            candidates.push(p);
        }
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf".to_string(),
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf".to_string(),
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf".to_string(),
        ]);
        for path in candidates {
            if let Ok(bytes) = fs::read(&path) {
                if let Ok(f) = Font::from_bytes(bytes, fontdue::FontSettings::default()) {
                    return Some(f);
                }
            }
        }
        None
    })
    .as_ref()
}

#[derive(Clone, Copy)]
enum Align {
    Left,
    Center,
    Right,
}

fn measure_text_width(font: &Font, text: &str, px: f32) -> f32 {
    let mut layout = Layout::new(CoordinateSystem::PositiveYDown);
    layout.reset(&LayoutSettings::default());
    layout.append(&[font], &TextStyle::new(text, px, 0));
    layout
        .glyphs()
        .iter()
        .map(|g| g.x + g.width as f32)
        .fold(0.0f32, f32::max)
}

fn blend_pixel(data: &mut [u8], w: usize, h: usize, x: i32, y: i32, r: u8, g: u8, b: u8, a: u8) {
    if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 || a == 0 {
        return;
    }
    let idx = ((y as usize) * w + (x as usize)) * 4;
    let sa = (a as f32) / 255.0;
    let inv = 1.0 - sa;
    // tiny-skia pixmap stores premultiplied RGBA
    let dr = data[idx] as f32;
    let dg = data[idx + 1] as f32;
    let db = data[idx + 2] as f32;
    let da = data[idx + 3] as f32;

    data[idx] = (r as f32 * sa + dr * inv).round().clamp(0.0, 255.0) as u8;
    data[idx + 1] = (g as f32 * sa + dg * inv).round().clamp(0.0, 255.0) as u8;
    data[idx + 2] = (b as f32 * sa + db * inv).round().clamp(0.0, 255.0) as u8;
    data[idx + 3] = (255.0 * sa + da * inv).round().clamp(0.0, 255.0) as u8;
}

fn draw_text(
    pixmap: &mut Pixmap,
    text: &str,
    x: f32,
    y: f32,
    px: f32,
    rgb: [u8; 3],
    align: Align,
) {
    let Some(font) = get_font() else {
        return;
    };
    let mut layout = Layout::new(CoordinateSystem::PositiveYDown);
    layout.reset(&LayoutSettings::default());
    layout.append(&[font], &TextStyle::new(text, px, 0));
    let total_w = layout
        .glyphs()
        .iter()
        .map(|g| g.x + g.width as f32)
        .fold(0.0f32, f32::max);
    let dx = match align {
        Align::Left => x,
        Align::Center => x - total_w * 0.5,
        Align::Right => x - total_w,
    };
    let pw = pixmap.width() as usize;
    let ph = pixmap.height() as usize;
    let data = pixmap.data_mut();
    for g in layout.glyphs() {
        if g.width == 0 || g.height == 0 {
            continue;
        }
        let (metrics, bitmap) = font.rasterize_config(g.key);
        let gx = (dx + g.x).round() as i32;
        let gy = (y + g.y).round() as i32;
        for row in 0..metrics.height {
            for col in 0..metrics.width {
                let cov = bitmap[row * metrics.width + col];
                if cov > 0 {
                    blend_pixel(
                        data,
                        pw,
                        ph,
                        gx + col as i32,
                        gy + row as i32,
                        rgb[0],
                        rgb[1],
                        rgb[2],
                        cov,
                    );
                }
            }
        }
    }
}

/// Draws text rotated 90 degrees counter-clockwise, centered at (cx, cy).
fn draw_text_vertical_ccw(
    pixmap: &mut Pixmap,
    text: &str,
    cx: f32,
    cy: f32,
    px: f32,
    rgb: [u8; 3],
) {
    let Some(font) = get_font() else {
        return;
    };
    let mut layout = Layout::new(CoordinateSystem::PositiveYDown);
    layout.reset(&LayoutSettings::default());
    layout.append(&[font], &TextStyle::new(text, px, 0));
    let total_w = layout
        .glyphs()
        .iter()
        .map(|g| g.x + g.width as f32)
        .fold(0.0f32, f32::max);
    let total_h = px;
    let pw = pixmap.width() as usize;
    let ph = pixmap.height() as usize;
    let data = pixmap.data_mut();

    for g in layout.glyphs() {
        if g.width == 0 || g.height == 0 {
            continue;
        }
        let (metrics, bitmap) = font.rasterize_config(g.key);
        for row in 0..metrics.height {
            for col in 0..metrics.width {
                let cov = bitmap[row * metrics.width + col];
                if cov > 0 {
                    // Local coords relative to text center
                    let lx = g.x + col as f32 - total_w * 0.5;
                    let ly = g.y + row as f32 - total_h * 0.5;
                    // Rotate -90 deg (CCW): (x', y') = (ly, -lx)
                    let px_x = (cx + ly).round() as i32;
                    let px_y = (cy - lx).round() as i32;
                    blend_pixel(data, pw, ph, px_x, px_y, rgb[0], rgb[1], rgb[2], cov);
                }
            }
        }
    }
}

fn fill_rect(pixmap: &mut Pixmap, x: f32, y: f32, w: f32, h: f32, r: u8, g: u8, b: u8, a: u8) {
    if let Some(rect) = Rect::from_xywh(x, y, w.max(0.5), h.max(0.5)) {
        let mut paint = Paint::default();
        paint.set_color_rgba8(r, g, b, a);
        paint.anti_alias = false;
        pixmap.fill_rect(rect, &paint, Transform::identity(), None);
    }
}

fn stroke_line(
    pixmap: &mut Pixmap,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    width: f32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) {
    let mut pb = PathBuilder::new();
    pb.move_to(x0, y0);
    pb.line_to(x1, y1);
    if let Some(path) = pb.finish() {
        let mut paint = Paint::default();
        paint.set_color_rgba8(r, g, b, a);
        paint.anti_alias = true;
        let stroke = Stroke {
            width,
            ..Default::default()
        };
        pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
    }
}

fn nice_step(min_step: f64) -> f64 {
    for &s in &[
        0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0,
    ] {
        if s >= min_step {
            return s;
        }
    }
    1200.0
}

fn envelope(x: &[f32], hop: usize) -> Vec<f64> {
    let n = x.len() / hop;
    x[..n * hop]
        .chunks_exact(hop)
        .map(|ch| {
            let mean_sq =
                ch.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / (hop as f64);
            (mean_sq + 1e-12).sqrt()
        })
        .collect()
}

fn compute_specgram_db(x: &[f32], nfft: usize, hop: usize) -> (usize, usize, Vec<f32>) {
    let n_bins = nfft / 2 + 1;
    if x.len() < nfft {
        return (n_bins, 0, vec![]);
    }
    let n_frames = (x.len() - nfft) / hop + 1;
    let win: Vec<f32> = (0..nfft)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f64) / ((nfft - 1) as f64)).cos()) as f32)
        .collect();
    let win_sq_sum: f64 = win.iter().map(|&w| (w as f64) * (w as f64)).sum();
    let scale_interior = 2.0 / ((SR as f64) * win_sq_sum);
    let scale_edge = 1.0 / ((SR as f64) * win_sq_sum);

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(nfft);

    let mut out = vec![-120.0f32; n_frames * n_bins];
    out.par_chunks_mut(n_bins)
        .enumerate()
        .for_each_init(
            || (r2c.make_input_vec(), r2c.make_output_vec(), r2c.make_scratch_vec()),
            |(buf, spec, scratch), (fi, dst)| {
                let off = fi * hop;
                for i in 0..nfft {
                    buf[i] = x[off + i] * win[i];
                }
                r2c.process_with_scratch(buf, spec, scratch).unwrap();
                for (bi, c) in spec.iter().enumerate() {
                    let p2 = (c.re as f64) * (c.re as f64) + (c.im as f64) * (c.im as f64);
                    let sc = if bi == 0 || bi == n_bins - 1 {
                        scale_edge
                    } else {
                        scale_interior
                    };
                    let pxx = p2 * sc;
                    dst[bi] = if pxx > 0.0 {
                        (10.0 * pxx.log10()) as f32
                    } else {
                        -200.0
                    };
                }
            },
        );
    (n_bins, n_frames, out)
}

pub fn render(
    path: &Path,
    out: &Path,
    start_in: Option<f64>,
    dur: f64,
    full: bool,
    shifts: &HashMap<usize, f64>,
    subs: &[(String, Vec<(f64, f64)>, f64)],
    title: Option<&str>,
) -> Result<f64> {
    let names = media::ffprobe_audio_labels(path)?;
    let n_audio = names.len();
    if n_audio == 0 {
        anyhow::bail!("no hay pistas de audio en {}", path.display());
    }

    let (start, dur_opt, audio) = if full {
        let raw: Vec<Vec<f32>> = (0..n_audio)
            .into_par_iter()
            .map(|i| media::load_audio_slice(path, i, None, None, SR as u32))
            .collect::<Result<_>>()?;
        let shifted: Vec<Vec<f32>> = raw
            .into_iter()
            .enumerate()
            .map(|(i, x)| {
                let k = ((shifts.get(&i).copied().unwrap_or(0.0)) * SR as f64) as isize;
                if k >= 0 {
                    let mut v = vec![0.0f32; k as usize];
                    v.extend_from_slice(&x);
                    v
                } else {
                    let sk = (-k) as usize;
                    if sk < x.len() {
                        x[sk..].to_vec()
                    } else {
                        vec![]
                    }
                }
            })
            .collect();
        (0.0, None, shifted)
    } else {
        let st = match start_in {
            Some(s) => s,
            None => {
                let x0 = media::load_audio_slice(path, 0, None, None, SR as u32)?;
                (transient(&x0, SR) - dur * 0.4).max(0.0)
            }
        };
        let loaded: Vec<Vec<f32>> = (0..n_audio)
            .into_par_iter()
            .map(|i| {
                let sh = shifts.get(&i).copied().unwrap_or(0.0);
                media::load_audio_slice(path, i, Some(st - sh), Some(dur), SR as u32)
            })
            .collect::<Result<_>>()?;
        (st, Some(dur), loaded)
    };

    let end = start
        + dur_opt.unwrap_or_else(|| {
            audio.iter().map(|x| x.len()).max().unwrap_or(0) as f64 / (SR as f64)
        });
    let span = (end - start).max(1e-3);

    // Layout dimensions
    let has_subs = !subs.is_empty();
    let mut ratios = vec![1.0f64; n_audio + 1];
    if has_subs {
        ratios.push(0.12 * (subs.len() as f64) + 0.25);
    }
    let ratio_sum: f64 = ratios.iter().sum();

    let width = 1600u32;
    let height = (2.6 * ratio_sum * 100.0).round().max(320.0) as u32;

    let mut pixmap = Pixmap::new(width, height).expect("pixmap alloc");
    pixmap.fill(Color::WHITE);

    let ml = 78.0f32;
    let mr = 22.0f32;
    let mt = 44.0f32;
    let mb = if dur_opt.is_some_and(|d| d <= 6.0) {
        56.0f32
    } else {
        42.0f32
    };
    let gap_y = 28.0f32;
    let plot_w = (width as f32) - ml - mr;
    let rows = ratios.len();
    let avail_h = (height as f32) - mt - mb - gap_y * ((rows - 1) as f32);

    let mut row_rects = Vec::with_capacity(rows);
    let mut cur_y = mt;
    for r in &ratios {
        let rh = ((avail_h as f64) * (*r / ratio_sum)).max(40.0) as f32;
        row_rects.push((ml, cur_y, plot_w, rh));
        cur_y += rh + gap_y;
    }

    // Determine shared X ticks
    let tick_step = if dur_opt.is_some_and(|d| d <= 6.0) {
        0.1
    } else {
        nice_step(span / 10.0)
    };
    let mut x_ticks = Vec::new();
    let mut tk = (start / tick_step).ceil() * tick_step;
    while tk <= end + 1e-9 {
        x_ticks.push(tk);
        tk += tick_step;
    }
    let time_to_x = |t: f64| -> f32 { ml + (((t - start) / span) as f32) * plot_w };

    // 1. Spectrogram subplots
    let (nfft, hop) = if dur_opt.is_none() || span > 120.0 {
        (4096, 2048)
    } else if span > 12.0 {
        (1024, 256)
    } else {
        (512, 32)
    };

    let spec_results: Vec<(usize, usize, Vec<f32>)> = audio
        .par_iter()
        .map(|x| compute_specgram_db(x, nfft, hop))
        .collect();

    for (i, ((n_bins, n_frames, db_data), name)) in
        spec_results.iter().zip(names.iter()).enumerate()
    {
        let (px, py, pw, ph) = row_rects[i];
        let x0 = px.round() as usize;
        let y0 = py.round() as usize;
        let iw = pw.round() as usize;
        let ih = ph.round() as usize;
        fill_rect(&mut pixmap, px, py, pw, ph, 0, 0, 0, 255);

        if *n_frames > 0 && *n_bins > 1 {
            let x_end = start + (audio[i].len() as f64) / (SR as f64);
            let pix_data = pixmap.data_mut();
            let canvas_w = width as usize;

            for py_off in 0..ih {
                // Top row = Nyquist (n_bins - 1), bottom row = 0 Hz
                let freq_frac = 1.0 - ((py_off as f64 + 0.5) / (ih as f64));
                let bin = ((freq_frac * ((*n_bins - 1) as f64)).round() as usize).min(n_bins - 1);
                let row_y = y0 + py_off;

                for px_off in 0..iw {
                    let t_lo = start + ((px_off as f64) / (iw as f64)) * span;
                    let t_hi = start + (((px_off + 1) as f64) / (iw as f64)) * span;
                    if t_lo >= x_end || t_hi <= start {
                        continue;
                    }
                    let track_dur = (x_end - start).max(1e-6);
                    let f_lo = (((t_lo - start) / track_dur) * (*n_frames as f64))
                        .floor()
                        .max(0.0) as usize;
                    let f_hi = (((t_hi - start) / track_dur) * (*n_frames as f64))
                        .ceil()
                        .min(*n_frames as f64) as usize;
                    let f_hi = f_hi.max(f_lo + 1).min(*n_frames);

                    let mut max_db = -200.0f32;
                    for fi in f_lo..f_hi {
                        let v = db_data[fi * (*n_bins) + bin];
                        if v > max_db {
                            max_db = v;
                        }
                    }
                    if max_db <= -199.0 {
                        continue; // digital silence stays black
                    }
                    let ci = (((max_db as f64 + 110.0) / 80.0) * 255.0).clamp(0.0, 255.0) as usize;
                    let rgb = MAGMA[ci];
                    let dst_idx = (row_y * canvas_w + (x0 + px_off)) * 4;
                    pix_data[dst_idx] = rgb[0];
                    pix_data[dst_idx + 1] = rgb[1];
                    pix_data[dst_idx + 2] = rgb[2];
                    pix_data[dst_idx + 3] = 255;
                }
            }
        }

        // Grid lines (white alpha 0.35)
        for &tk in &x_ticks {
            let gx = time_to_x(tk);
            if gx > px && gx < px + pw {
                stroke_line(&mut pixmap, gx, py, gx, py + ph, 0.7, 255, 255, 255, 90);
            }
        }
        // Border
        stroke_line(&mut pixmap, px, py, px + pw, py, 1.0, 80, 80, 80, 255);
        stroke_line(&mut pixmap, px, py + ph, px + pw, py + ph, 1.0, 80, 80, 80, 255);
        stroke_line(&mut pixmap, px, py, px, py + ph, 1.0, 80, 80, 80, 255);
        stroke_line(&mut pixmap, px + pw, py, px + pw, py + ph, 1.0, 80, 80, 80, 255);

        // Title & Y-axis labels
        draw_text(&mut pixmap, name, px, py - 15.0, 12.0, [28, 28, 26], Align::Left);
        for &hz in &[0, 2000, 4000, 6000, 8000] {
            let ty = py + ph * (1.0 - (hz as f32) / 8000.0);
            stroke_line(&mut pixmap, px - 4.0, ty, px, ty, 1.0, 60, 60, 60, 255);
            draw_text(
                &mut pixmap,
                &hz.to_string(),
                px - 7.0,
                ty - 5.0,
                10.0,
                [60, 60, 60],
                Align::Right,
            );
        }
        draw_text_vertical_ccw(&mut pixmap, "Hz", px - 56.0, py + ph * 0.5, 11.0, [28, 28, 26]);
    }

    // 2. Volume envelopes subplot
    {
        let (px, py, pw, ph) = row_rects[n_audio];
        let ehop = (SR / 500).max(((SR as f64) * span / 3000.0) as usize);
        let envs: Vec<Vec<f64>> = audio
            .iter()
            .map(|x| {
                envelope(x, ehop)
                    .into_iter()
                    .map(|e| 20.0 * e.log10())
                    .collect()
            })
            .collect();

        let mut db_min = f64::INFINITY;
        let mut db_max = f64::NEG_INFINITY;
        for env in &envs {
            for &v in env {
                if v.is_finite() {
                    db_min = db_min.min(v);
                    db_max = db_max.max(v);
                }
            }
        }
        if !db_min.is_finite() || !db_max.is_finite() || (db_max - db_min) < 1.0 {
            db_min = -90.0;
            db_max = 0.0;
        } else {
            let pad = (db_max - db_min) * 0.06;
            db_min -= pad;
            db_max += pad;
        }

        let db_to_y = |db: f64| -> f32 {
            let frac = ((db - db_min) / (db_max - db_min)).clamp(0.0, 1.0);
            py + ph * (1.0 - frac as f32)
        };

        // Grid
        for &tk in &x_ticks {
            let gx = time_to_x(tk);
            if gx > px && gx < px + pw {
                stroke_line(&mut pixmap, gx, py, gx, py + ph, 0.7, 210, 210, 210, 255);
            }
        }
        let db_step = nice_step((db_max - db_min) / 4.0).max(5.0);
        let mut yv = (db_min / db_step).ceil() * db_step;
        while yv <= db_max {
            let gy = db_to_y(yv);
            stroke_line(&mut pixmap, px, gy, px + pw, gy, 0.7, 210, 210, 210, 255);
            stroke_line(&mut pixmap, px - 4.0, gy, px, gy, 1.0, 60, 60, 60, 255);
            draw_text(
                &mut pixmap,
                &format!("{:.0}", yv),
                px - 7.0,
                gy - 5.0,
                10.0,
                [60, 60, 60],
                Align::Right,
            );
            yv += db_step;
        }

        // Plot each track envelope
        for (i, env) in envs.iter().enumerate() {
            if env.is_empty() {
                continue;
            }
            let mut pb = PathBuilder::new();
            for (k, &db) in env.iter().enumerate() {
                let t = start + ((k * ehop) as f64) / (SR as f64);
                let x = time_to_x(t).clamp(px, px + pw);
                let y = db_to_y(db);
                if k == 0 {
                    pb.move_to(x, y);
                } else {
                    pb.line_to(x, y);
                }
            }
            if let Some(p) = pb.finish() {
                let c = TAB10[i % 10];
                let mut paint = Paint::default();
                paint.set_color_rgba8(c[0], c[1], c[2], 235);
                paint.anti_alias = true;
                let stroke = Stroke {
                    width: 1.0,
                    ..Default::default()
                };
                pixmap.stroke_path(&p, &paint, &stroke, Transform::identity(), None);
            }
        }

        // Border
        stroke_line(&mut pixmap, px, py, px + pw, py, 1.0, 80, 80, 80, 255);
        stroke_line(&mut pixmap, px, py + ph, px + pw, py + ph, 1.0, 80, 80, 80, 255);
        stroke_line(&mut pixmap, px, py, px, py + ph, 1.0, 80, 80, 80, 255);
        stroke_line(&mut pixmap, px + pw, py, px + pw, py + ph, 1.0, 80, 80, 80, 255);

        draw_text(
            &mut pixmap,
            "Envolvente de volumen (todas superpuestas)",
            px,
            py - 15.0,
            12.0,
            [28, 28, 26],
            Align::Left,
        );
        draw_text_vertical_ccw(
            &mut pixmap,
            "dBFS",
            px - 56.0,
            py + ph * 0.5,
            11.0,
            [28, 28, 26],
        );

        // Legend in lower right
        if let Some(font) = get_font() {
            let item_widths: Vec<f32> = names
                .iter()
                .map(|nm| 20.0 + measure_text_width(font, nm, 10.0) + 12.0)
                .collect();
            let leg_w = item_widths.iter().sum::<f32>() + 8.0;
            let leg_h = 20.0f32;
            let lx = (px + pw - leg_w - 8.0).max(px + 4.0);
            let ly = py + ph - leg_h - 6.0;
            fill_rect(&mut pixmap, lx, ly, leg_w, leg_h, 255, 255, 255, 225);
            stroke_line(&mut pixmap, lx, ly, lx + leg_w, ly, 0.8, 180, 180, 180, 255);
            stroke_line(&mut pixmap, lx, ly + leg_h, lx + leg_w, ly + leg_h, 0.8, 180, 180, 180, 255);
            stroke_line(&mut pixmap, lx, ly, lx, ly + leg_h, 0.8, 180, 180, 180, 255);
            stroke_line(&mut pixmap, lx + leg_w, ly, lx + leg_w, ly + leg_h, 0.8, 180, 180, 180, 255);
            let mut cx = lx + 6.0;
            for (i, nm) in names.iter().enumerate() {
                let c = TAB10[i % 10];
                stroke_line(&mut pixmap, cx, ly + 10.0, cx + 14.0, ly + 10.0, 2.0, c[0], c[1], c[2], 255);
                draw_text(&mut pixmap, nm, cx + 18.0, ly + 5.0, 10.0, [28, 28, 26], Align::Left);
                cx += item_widths[i];
            }
        }
    }

    // 3. Subtitles subplot
    if has_subs {
        let (px, py, pw, ph) = row_rects[rows - 1];
        for &tk in &x_ticks {
            let gx = time_to_x(tk);
            if gx > px && gx < px + pw {
                stroke_line(&mut pixmap, gx, py, gx, py + ph, 0.7, 210, 210, 210, 255);
            }
        }
        let n_sub = subs.len() as f32;
        let row_step = ph / n_sub;
        for (i, (label, events, shift)) in subs.iter().enumerate() {
            let c = TAB10[i % 10];
            let bar_h = (row_step * 0.65).max(4.0);
            let by = py + (i as f32 + 0.5) * row_step - bar_h * 0.5;
            for &(a, b) in events {
                let sa = a + *shift;
                let sb = b + *shift;
                if sa < end && sb > start {
                    let x_a = time_to_x(sa.max(start));
                    let x_b = time_to_x(sb.min(end));
                    fill_rect(&mut pixmap, x_a, by, (x_b - x_a).max(1.0), bar_h, c[0], c[1], c[2], 235);
                }
            }
            let short_lab: String = label.chars().take(12).collect();
            draw_text(
                &mut pixmap,
                &short_lab,
                px - 6.0,
                py + (i as f32 + 0.5) * row_step - 4.0,
                9.0,
                [60, 60, 60],
                Align::Right,
            );
        }
        stroke_line(&mut pixmap, px, py, px + pw, py, 1.0, 80, 80, 80, 255);
        stroke_line(&mut pixmap, px, py + ph, px + pw, py + ph, 1.0, 80, 80, 80, 255);
        stroke_line(&mut pixmap, px, py, px, py + ph, 1.0, 80, 80, 80, 255);
        stroke_line(&mut pixmap, px + pw, py, px + pw, py + ph, 1.0, 80, 80, 80, 255);
        draw_text(
            &mut pixmap,
            "Subtítulos (cada barra es una línea en pantalla)",
            px,
            py - 15.0,
            12.0,
            [28, 28, 26],
            Align::Left,
        );
    }

    // 4. Shared X-axis ticks & labels on bottom row
    {
        let (px, py, pw, ph) = row_rects[rows - 1];
        let bot_y = py + ph;
        let fine = dur_opt.is_some_and(|d| d <= 6.0);
        for &tk in &x_ticks {
            let tx = time_to_x(tk);
            if tx < px - 1.0 || tx > px + pw + 1.0 {
                continue;
            }
            stroke_line(&mut pixmap, tx, bot_y, tx, bot_y + 4.0, 1.0, 60, 60, 60, 255);
            let txt = if fine {
                format!("{tk:.1}")
            } else if tk.fract().abs() < 1e-6 {
                format!("{:.0}", tk)
            } else {
                format!("{tk:.1}")
            };
            if fine {
                draw_text_vertical_ccw(&mut pixmap, &txt, tx, bot_y + 20.0, 9.0, [40, 40, 40]);
            } else {
                draw_text(&mut pixmap, &txt, tx, bot_y + 7.0, 10.0, [40, 40, 40], Align::Center);
            }
        }
        draw_text(
            &mut pixmap,
            "tiempo (s)",
            px + pw * 0.5,
            (height as f32) - 16.0,
            11.0,
            [28, 28, 26],
            Align::Center,
        );
    }

    // 5. Suptitle
    let fname = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let default_title = if full {
        format!("{fname}  —  completo")
    } else {
        format!("{fname}  —  {start:.2}s a {end:.2}s")
    };
    let sup = title.unwrap_or(&default_title);
    draw_text(
        &mut pixmap,
        sup,
        (width as f32) * 0.5,
        10.0,
        13.0,
        [20, 20, 20],
        Align::Center,
    );

    if let Some(p) = out.parent() {
        fs::create_dir_all(p)?;
    }
    let tmp = out.with_extension("png.tmp");
    {
        let rgba = pixmap.take();
        let mut rgb = Vec::with_capacity((width as usize) * (height as usize) * 3);
        for px in rgba.chunks_exact(4) {
            rgb.push(px[0]);
            rgb.push(px[1]);
            rgb.push(px[2]);
        }
        let file = File::create(&tmp)?;
        let w = BufWriter::new(file);
        let mut encoder = png::Encoder::new(w, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&rgb)?;
    }
    fs::rename(tmp, out)?;
    let _ = mean_std_f32;
    Ok(start)
}
