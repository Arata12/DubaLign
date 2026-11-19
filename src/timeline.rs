use crate::media;
use crate::model::{EnvelopeData, TrackInfo};
use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use memmap2::Mmap;
use rayon::prelude::*;
use realfft::RealFftPlanner;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};

pub const SR: usize = 16000;
pub const NFFT: usize = 512;
pub const SPEC_FPS: usize = 50;
pub const SPEC_BINS: usize = 128;
pub const ENV_FPS: usize = 200;
pub const LEVELS: [(f64, f64); 3] = [(50.0, 40.0), (5.0, 400.0), (0.5, 4000.0)];
pub const DB_LO: f64 = -100.0;
pub const DB_HI: f64 = -25.0;

pub const MAGMA: [[u8; 3]; 256] = [
    [0,0,3], [0,0,4], [0,0,6], [1,0,7],
    [1,1,9], [1,1,11], [2,2,13], [2,2,15],
    [3,3,17], [4,3,19], [4,4,21], [5,4,23],
    [6,5,25], [7,5,27], [8,6,29], [9,7,31],
    [10,7,34], [11,8,36], [12,9,38], [13,10,40],
    [14,10,42], [15,11,44], [16,12,47], [17,12,49],
    [18,13,51], [20,13,53], [21,14,56], [22,14,58],
    [23,15,60], [24,15,63], [26,16,65], [27,16,68],
    [28,16,70], [30,16,73], [31,17,75], [32,17,77],
    [34,17,80], [35,17,82], [37,17,85], [38,17,87],
    [40,17,89], [42,17,92], [43,17,94], [45,16,96],
    [47,16,98], [48,16,101], [50,16,103], [52,16,104],
    [53,15,106], [55,15,108], [57,15,110], [59,15,111],
    [60,15,113], [62,15,114], [64,15,115], [66,15,116],
    [67,15,117], [69,15,118], [71,15,119], [72,16,120],
    [74,16,121], [75,16,121], [77,17,122], [79,17,123],
    [80,18,123], [82,18,124], [83,19,124], [85,19,125],
    [87,20,125], [88,21,126], [90,21,126], [91,22,126],
    [93,23,126], [94,23,127], [96,24,127], [97,24,127],
    [99,25,127], [101,26,128], [102,26,128], [104,27,128],
    [105,28,128], [107,28,128], [108,29,128], [110,30,129],
    [111,30,129], [113,31,129], [115,31,129], [116,32,129],
    [118,33,129], [119,33,129], [121,34,129], [122,34,129],
    [124,35,129], [126,36,129], [127,36,129], [129,37,129],
    [130,37,129], [132,38,129], [133,38,129], [135,39,129],
    [137,40,129], [138,40,129], [140,41,128], [141,41,128],
    [143,42,128], [145,42,128], [146,43,128], [148,43,128],
    [149,44,128], [151,44,127], [153,45,127], [154,45,127],
    [156,46,127], [158,46,126], [159,47,126], [161,47,126],
    [163,48,126], [164,48,125], [166,49,125], [167,49,125],
    [169,50,124], [171,51,124], [172,51,123], [174,52,123],
    [176,52,123], [177,53,122], [179,53,122], [181,54,121],
    [182,54,121], [184,55,120], [185,55,120], [187,56,119],
    [189,57,119], [190,57,118], [192,58,117], [194,58,117],
    [195,59,116], [197,60,116], [198,60,115], [200,61,114],
    [202,62,114], [203,62,113], [205,63,112], [206,64,112],
    [208,65,111], [209,66,110], [211,66,109], [212,67,109],
    [214,68,108], [215,69,107], [217,70,106], [218,71,105],
    [220,72,105], [221,73,104], [222,74,103], [224,75,102],
    [225,76,102], [226,77,101], [228,78,100], [229,80,99],
    [230,81,98], [231,82,98], [232,84,97], [234,85,96],
    [235,86,96], [236,88,95], [237,89,95], [238,91,94],
    [238,93,93], [239,94,93], [240,96,93], [241,97,92],
    [242,99,92], [243,101,92], [243,103,91], [244,104,91],
    [245,106,91], [245,108,91], [246,110,91], [246,112,91],
    [247,113,91], [247,115,92], [248,117,92], [248,119,92],
    [249,121,92], [249,123,93], [249,125,93], [250,127,94],
    [250,128,94], [250,130,95], [251,132,96], [251,134,96],
    [251,136,97], [251,138,98], [252,140,99], [252,142,99],
    [252,144,100], [252,146,101], [252,147,102], [253,149,103],
    [253,151,104], [253,153,105], [253,155,106], [253,157,107],
    [253,159,108], [253,161,110], [253,162,111], [253,164,112],
    [254,166,113], [254,168,115], [254,170,116], [254,172,117],
    [254,174,118], [254,175,120], [254,177,121], [254,179,123],
    [254,181,124], [254,183,125], [254,185,127], [254,187,128],
    [254,188,130], [254,190,131], [254,192,133], [254,194,134],
    [254,196,136], [254,198,137], [254,199,139], [254,201,141],
    [254,203,142], [253,205,144], [253,207,146], [253,209,147],
    [253,210,149], [253,212,151], [253,214,152], [253,216,154],
    [253,218,156], [253,220,157], [253,221,159], [253,223,161],
    [253,225,163], [252,227,165], [252,229,166], [252,230,168],
    [252,232,170], [252,234,172], [252,236,174], [252,238,176],
    [252,240,177], [252,241,179], [252,243,181], [252,245,183],
    [251,247,185], [251,249,187], [251,250,189], [251,252,191],
];

fn lock_for(key: &str) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<Mutex<HashMap<String, Arc<Mutex<()>>>>> = OnceLock::new();
    let map = LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap();
    guard
        .entry(key.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

pub fn compute_features(x: &[f32]) -> (Vec<u8>, usize, Vec<u8>) {
    let hop = SR / SPEC_FPS; // 320
    let n = if x.len() >= NFFT {
        (x.len() - NFFT) / hop + 1
    } else {
        0
    };

    let win: Vec<f32> = (0..NFFT)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f64) / ((NFFT - 1) as f64)).cos()) as f32)
        .collect();
    let scale = 2.0f32 / win.iter().sum::<f32>();

    let mut spec = vec![0u8; SPEC_BINS * n];
    if n > 0 {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(NFFT);
        // Compute columns in parallel chunks -> transpose into [SPEC_BINS, n]
        let chunk_cols = 2000usize;
        let col_chunks: Vec<(usize, Vec<u8>)> = (0..n)
            .step_by(chunk_cols)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map_init(
                || (r2c.make_input_vec(), r2c.make_output_vec(), r2c.make_scratch_vec()),
                |(buf, fft_out, scratch), start_col| {
                    let end_col = (start_col + chunk_cols).min(n);
                    let cols = end_col - start_col;
                    let mut local = vec![0u8; SPEC_BINS * cols];
                    for c_off in 0..cols {
                        let off = (start_col + c_off) * hop;
                        for i in 0..NFFT {
                            buf[i] = x[off + i] * win[i];
                        }
                        r2c.process_with_scratch(buf, fft_out, scratch).unwrap();
                        for b in 0..SPEC_BINS {
                            let m0 = fft_out[2 * b].norm() * scale;
                            let m1 = fft_out[2 * b + 1].norm() * scale;
                            let mag = m0.max(m1) as f64;
                            let db = 20.0 * (mag + 1e-9).log10();
                            let v = ((db - DB_LO) / (DB_HI - DB_LO) * 255.0).clamp(0.0, 255.0);
                            local[b * cols + c_off] = v as u8;
                        }
                    }
                    (start_col, local)
                },
            )
            .collect();

        for (start_col, local) in col_chunks {
            let cols = local.len() / SPEC_BINS;
            for b in 0..SPEC_BINS {
                let dst = &mut spec[b * n + start_col..b * n + start_col + cols];
                let src = &local[b * cols..(b + 1) * cols];
                dst.copy_from_slice(src);
            }
        }
    }

    let eh = SR / ENV_FPS; // 80
    let m = x.len() / eh;
    let env: Vec<u8> = x[..m * eh]
        .par_chunks_exact(eh)
        .map(|ch| {
            let mean_sq = ch.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / (eh as f64);
            let rms = (mean_sq + 1e-12).sqrt();
            let v = ((20.0 * rms.log10() + 90.0) / 90.0 * 255.0).clamp(0.0, 255.0);
            v as u8
        })
        .collect();

    (spec, n, env)
}

fn write_npy_u8(path: &Path, shape: &[usize], data: &[u8]) -> Result<()> {
    let shape_str = match shape {
        [a] => format!("({a},)"),
        [a, b] => format!("({a}, {b})"),
        _ => bail!("unsupported npy shape"),
    };
    let mut header = format!(
        "{{'descr': '|u1', 'fortran_order': False, 'shape': {shape_str}, }}"
    );
    let total_prefix = 10 + header.len() + 1; // +1 for trailing '\n'
    let pad = (64 - (total_prefix % 64)) % 64;
    for _ in 0..pad {
        header.push(' ');
    }
    header.push('\n');
    let hlen = header.len() as u16;

    let tmp = path.with_extension("npy.tmp");
    {
        let f = File::create(&tmp)?;
        let mut w = BufWriter::new(f);
        w.write_all(b"\x93NUMPY\x01\x00")?;
        w.write_all(&hlen.to_le_bytes())?;
        w.write_all(header.as_bytes())?;
        w.write_all(data)?;
        w.flush()?;
    }
    fs::rename(tmp, path)?;
    Ok(())
}

fn parse_npy_header(mmap: &[u8]) -> Result<(usize, Vec<usize>)> {
    if mmap.len() < 10 || &mmap[..6] != b"\x93NUMPY" {
        bail!("not a valid .npy file");
    }
    let major = mmap[6];
    let (hlen, data_off) = if major == 1 {
        let l = u16::from_le_bytes([mmap[8], mmap[9]]) as usize;
        (l, 10 + l)
    } else if major == 2 {
        if mmap.len() < 12 {
            bail!("truncated .npy v2");
        }
        let l = u32::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11]]) as usize;
        (l, 12 + l)
    } else {
        bail!("unsupported .npy version {major}");
    };
    if data_off > mmap.len() {
        bail!("truncated .npy header");
    }
    let hdr = std::str::from_utf8(&mmap[data_off - hlen..data_off])?;
    let s_start = hdr.find("'shape'").context("missing shape in npy")?;
    let p_open = hdr[s_start..].find('(').context("missing (")? + s_start + 1;
    let p_close = hdr[p_open..].find(')').context("missing )")? + p_open;
    let dims: Vec<usize> = hdr[p_open..p_close]
        .split(',')
        .filter_map(|t| {
            let s = t.trim();
            if s.is_empty() {
                None
            } else {
                s.parse::<usize>().ok()
            }
        })
        .collect();
    Ok((data_off, dims))
}

pub fn ensure_from_samples(
    key: &str,
    tracks_16k: &[Vec<f32>],
    root: &Path,
) -> Result<PathBuf> {
    let d = root.join(key);
    if d.join("done").is_file() {
        return Ok(d);
    }
    let lk = lock_for(key);
    let _guard = lk.lock().unwrap();
    if d.join("done").is_file() {
        return Ok(d);
    }
    fs::create_dir_all(&d)?;
    tracks_16k
        .par_iter()
        .enumerate()
        .try_for_each(|(n, x)| -> Result<()> {
            let (spec, cols, env) = compute_features(x);
            write_npy_u8(&d.join(format!("spec{n}.npy")), &[SPEC_BINS, cols], &spec)?;
            write_npy_u8(&d.join(format!("env{n}.npy")), &[env.len()], &env)?;
            Ok(())
        })?;
    fs::write(d.join("done"), b"ok")?;
    Ok(d)
}

pub fn ensure(path: &Path, key: &str, n_tracks: usize, root: &Path) -> Result<PathBuf> {
    let d = root.join(key);
    if d.join("done").is_file() {
        return Ok(d);
    }
    let lk = lock_for(key);
    let _guard = lk.lock().unwrap();
    if d.join("done").is_file() {
        return Ok(d);
    }
    let tracks_16k = media::load_audio_all_single_pass(path, n_tracks, SR as u32)?;
    fs::create_dir_all(&d)?;
    tracks_16k
        .par_iter()
        .enumerate()
        .try_for_each(|(n, x)| -> Result<()> {
            let (spec, cols, env) = compute_features(x);
            write_npy_u8(&d.join(format!("spec{n}.npy")), &[SPEC_BINS, cols], &spec)?;
            write_npy_u8(&d.join(format!("env{n}.npy")), &[env.len()], &env)?;
            Ok(())
        })?;
    fs::write(d.join("done"), b"ok")?;
    Ok(d)
}

pub fn tile_png(d: &Path, track: usize, level: usize, index: usize) -> Result<Vec<u8>> {
    let (fps, span) = LEVELS[level];
    let file = File::open(d.join(format!("spec{track}.npy")))?;
    let mmap = unsafe { Mmap::map(&file)? };
    let (off, dims) = parse_npy_header(&mmap)?;
    if dims.len() != 2 || dims[0] != SPEC_BINS {
        bail!("unexpected spec shape");
    }
    let total_cols = dims[1];
    let data = &mmap[off..];
    let step = ((SPEC_FPS as f64) / fps).round() as usize;
    let a = ((index as f64) * span * (SPEC_FPS as f64)) as usize;
    let b = (((index + 1) as f64) * span * (SPEC_FPS as f64)) as usize;
    let b = b.min(total_cols);

    let out_w = if b > a {
        let w = b - a;
        if step > 1 {
            let m = (w / step) * step;
            if m > 0 {
                m / step
            } else {
                1
            }
        } else {
            w
        }
    } else {
        1
    };

    let mut rgb = vec![0u8; SPEC_BINS * out_w * 3];
    if b > a {
        let w = b - a;
        if step > 1 {
            let m = (w / step) * step;
            for r in 0..SPEC_BINS {
                let row_src = &data[r * total_cols + a..r * total_cols + b];
                let out_r = SPEC_BINS - 1 - r;
                let dst_row = &mut rgb[out_r * out_w * 3..(out_r + 1) * out_w * 3];
                if m > 0 {
                    for x in 0..out_w {
                        let base = x * step;
                        let mut mx = 0u8;
                        for k in 0..step {
                            let v = row_src[base + k];
                            if v > mx {
                                mx = v;
                            }
                        }
                        let c = MAGMA[mx as usize];
                        dst_row[x * 3] = c[0];
                        dst_row[x * 3 + 1] = c[1];
                        dst_row[x * 3 + 2] = c[2];
                    }
                } else {
                    let c = MAGMA[row_src[0] as usize];
                    dst_row[0] = c[0];
                    dst_row[1] = c[1];
                    dst_row[2] = c[2];
                }
            }
        } else {
            for r in 0..SPEC_BINS {
                let row_src = &data[r * total_cols + a..r * total_cols + b];
                let out_r = SPEC_BINS - 1 - r;
                let dst_row = &mut rgb[out_r * out_w * 3..(out_r + 1) * out_w * 3];
                for (x, &v) in row_src.iter().enumerate() {
                    let c = MAGMA[v as usize];
                    dst_row[x * 3] = c[0];
                    dst_row[x * 3 + 1] = c[1];
                    dst_row[x * 3 + 2] = c[2];
                }
            }
        }
    } else {
        let c = MAGMA[0];
        for r in 0..SPEC_BINS {
            rgb[r * 3] = c[0];
            rgb[r * 3 + 1] = c[1];
            rgb[r * 3 + 2] = c[2];
        }
    }

    let mut buf = Vec::with_capacity(out_w * SPEC_BINS);
    {
        let mut encoder = png::Encoder::new(&mut buf, out_w as u32, SPEC_BINS as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&rgb)?;
    }
    Ok(buf)
}

pub fn envelopes(d: &Path, n_tracks: usize) -> Result<EnvelopeData> {
    let mut tracks = Vec::with_capacity(n_tracks);
    for n in 0..n_tracks {
        let file = File::open(d.join(format!("env{n}.npy")))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let (off, _) = parse_npy_header(&mmap)?;
        tracks.push(B64.encode(&mmap[off..]));
    }
    Ok(EnvelopeData {
        fps: ENV_FPS,
        db_lo: -90,
        tracks,
    })
}

pub fn audio_file(
    path: &Path,
    key: &str,
    root: &Path,
    ref_track: usize,
    track: usize,
    mode: &str,
) -> Result<PathBuf> {
    let out = root.join(key).join(format!("audio_{mode}_{ref_track}_{track}.m4a"));
    if out.is_file() {
        return Ok(out);
    }
    let lk = lock_for(&format!("{key}/audio/{mode}/{ref_track}/{track}"));
    let _guard = lk.lock().unwrap();
    if out.is_file() {
        return Ok(out);
    }
    if let Some(p) = out.parent() {
        fs::create_dir_all(p)?;
    }
    let norm = "aresample=async=1:first_pts=0";
    let mut args: Vec<String> = vec![
        "-v".into(),
        "error".into(),
        "-y".into(),
        "-i".into(),
        path.to_string_lossy().into_owned(),
    ];
    if mode == "split" {
        let fc = format!(
            "[0:a:{ref_track}]{norm},aformat=channel_layouts=mono[l];[0:a:{track}]{norm},aformat=channel_layouts=mono[r];[l][r]join=inputs=2:channel_layout=stereo[a]"
        );
        args.extend(["-filter_complex".into(), fc, "-map".into(), "[a]".into()]);
    } else {
        args.extend([
            "-map".into(),
            format!("0:a:{track}"),
            "-af".into(),
            norm.into(),
            "-ac".into(),
            "2".into(),
        ]);
    }
    let tmp = out.with_extension("tmp.m4a");
    args.extend([
        "-c:a".into(),
        "aac".into(),
        "-b:a".into(),
        "160k".into(),
        "-movflags".into(),
        "+faststart".into(),
        tmp.to_string_lossy().into_owned(),
    ]);
    let res = Command::new("ffmpeg").args(&args).output()?;
    if !res.status.success() {
        let _ = fs::remove_file(&tmp);
        let err = String::from_utf8_lossy(&res.stderr).trim().to_string();
        bail!("{}", if err.is_empty() { "ffmpeg falló".to_string() } else { err });
    }
    fs::rename(&tmp, &out)?;
    Ok(out)
}

pub fn track_names(tracks: &[TrackInfo]) -> Vec<String> {
    tracks
        .iter()
        .filter(|t| t.kind == "audio")
        .map(|t| format!("a:{} {} {}", t.n, t.lang, t.name).trim().to_string())
        .collect()
}
