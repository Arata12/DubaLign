use clap::{Parser, ValueEnum};
use dubalign::{
    dsp::{stretch, RefFftCache},
    media,
    sync_algo::{
        coverage_ok, detect_cut_segments, detect_pitch_shifted, detect_tempo_envelope,
        detect_tempo_scores, diagnose, infer_fps_pair, onset_envelope, pitch_semitones_from_tempo,
        render_segments, render_tempo, render_trim_pcm32_flac, scan, scan_env, trim_stream_copy,
        with_edge_jumps, ENV_SR, SR,
    },
};
use std::path::{Path, PathBuf};
use std::process::Command;
use walkdir::WalkDir;

const MEDIA_EXT: &[&str] = &[
    "mkv", "mp4", "m4v", "avi", "ts", "m2ts", "webm", "mka", "flac", "ac3", "eac3", "dts", "aac",
    "m4a", "opus", "mp3", "wav",
];

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum PitchMode {
    /// Detecta automáticamente si el tono está alterado (p. ej. speedup PAL) o ya es correcto
    Auto,
    /// Corrige velocidad y tono simultáneamente (varispeed / SoXR 28-bit)
    Correct,
    /// Cambia solo la velocidad manteniendo el tono actual (atempo)
    Keep,
}

#[derive(Parser)]
#[command(about = "Sincroniza el audio de un archivo (src) contra el video de referencia (ref).")]
struct Args {
    #[arg(help = "video de referencia (archivo o carpeta)")]
    r#ref: PathBuf,
    #[arg(help = "archivo/carpeta con el audio a sincronizar")]
    src: PathBuf,
    #[arg(short, long, default_value = "output")]
    output: PathBuf,
    #[arg(long, default_value = "work")]
    work: PathBuf,
    #[arg(long, default_value_t = 0)]
    ref_track: usize,
    #[arg(long, default_value_t = 0)]
    src_track: usize,
    #[arg(long, default_value = "spa")]
    lang: String,
    #[arg(long, default_value = "")]
    name: String,
    #[arg(long)]
    default: bool,
    #[arg(long)]
    src_subs: bool,
    #[arg(long, value_enum, default_value_t = PitchMode::Auto)]
    pitch: PitchMode,
    #[arg(long)]
    dry_run: bool,
    #[arg(long, default_value_t = 20)]
    windows: usize,
    #[arg(long, default_value_t = 30.0)]
    window: f64,
    #[arg(long, default_value_t = 5.0)]
    search: f64,
    #[arg(long, default_value_t = 12.0)]
    min_conf: f64,
}

fn process(ref_path: &Path, src_path: &Path, out_path: &Path, args: &Args) -> anyhow::Result<bool> {
    let ref_name = ref_path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let src_name = src_path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    println!("\n== {ref_name}  <-  {src_name}");

    let ref_info = media::probe(ref_path).ok();
    let (ref_res, src_res) = rayon::join(
        || media::load_audio(ref_path, args.ref_track, SR as u32),
        || media::load_audio(src_path, args.src_track, SR as u32),
    );
    let ref_audio = ref_res?;
    let src_raw = src_res?;
    let ref_dur = (ref_audio.len() as f64) / (SR as f64);
    let src_raw_dur = (src_raw.len() as f64) / (SR as f64);

    let ref_tempo_cache = RefFftCache::new(&ref_audio, ref_audio.len());
    let mut t_sc = ref_tempo_cache.make_scratch();
    let (mut tempo0, score_1) =
        detect_tempo_scores(&ref_tempo_cache, &mut t_sc, ref_audio.len(), &src_raw, SR);
    let mut via_envelope = false;
    let mut env_pair: Option<(Vec<f32>, Vec<f32>)> = None;
    if (tempo0 - 1.0).abs() < 1e-12 && score_1 < 12.0 {
        let r_env = onset_envelope(&ref_audio);
        let s_env = onset_envelope(&src_raw);
        let env_tp = detect_tempo_envelope(&r_env, &s_env);
        if (env_tp - 1.0).abs() > 1e-12 {
            tempo0 = env_tp;
            via_envelope = true;
        }
        env_pair = Some((r_env, s_env));
    }

    let stretched_wave;
    let mut wave_src: &[f32] = &src_raw;
    let mut wave_g: isize = 0;

    let (pts, stretched_dur_s) = if via_envelope {
        println!("  velocidad distinta detectada (tono conservado): tempo x{tempo0:.5}");
        let (r_env, s_env) = env_pair.take().unwrap();
        let s_str = stretch(&s_env, tempo0);
        let g_env = RefFftCache::new(&r_env, r_env.len() + s_str.len()).global_offset(&s_str);
        println!(
            "  desfase global aprox.: {:+.0} ms",
            (g_env as f64) / (ENV_SR as f64) * 1000.0
        );
        let l_env = (args.window * ENV_SR as f64) as usize;
        let step_env = ((((s_str.len() as isize - l_env as isize).max(0) as f64) / ENV_SR as f64)
            / ((args.windows + 1) as f64))
            .max(1.0);
        let p = scan_env(&r_env, &s_str, g_env, args.window, step_env, args.search, args.min_conf);
        (p, (s_str.len() as f64) / (ENV_SR as f64))
    } else {
        if (tempo0 - 1.0).abs() > 1e-12 {
            println!("  velocidad distinta detectada: tempo x{tempo0:.5}");
            stretched_wave = stretch(&src_raw, tempo0);
            wave_src = &stretched_wave;
        }
        let ref_full_cache = RefFftCache::new(&ref_audio, ref_audio.len() + wave_src.len() - 1);
        wave_g = ref_full_cache.global_offset(wave_src);
        println!(
            "  desfase global aprox.: {:+.0} ms",
            (wave_g as f64) / (SR as f64) * 1000.0
        );
        let l = (args.window * SR as f64) as usize;
        let step = ((((wave_src.len() as isize - l as isize).max(0) as f64) / SR as f64)
            / ((args.windows + 1) as f64))
            .max(1.0);
        let p = scan(
            &ref_audio,
            wave_src,
            wave_g,
            args.window,
            step,
            args.search,
            args.min_conf,
            0.0,
            None,
        );
        (p, (wave_src.len() as f64) / (SR as f64))
    };

    if pts.len() < 3 {
        println!("  ERROR: muy pocas ventanas fiables; ¿son el mismo episodio?");
        return Ok(false);
    }
    for &(t, d, conf) in &pts {
        println!(
            "    t={:5.1} min  delay={:+8.0} ms  conf={conf:5.1}",
            t / 60.0,
            d * 1000.0
        );
    }
    if !coverage_ok(&pts, stretched_dur_s) {
        println!("  ERROR: solo encaja en un tramo del episodio (¿opening/ending compartido?); probablemente no es el mismo episodio.");
        return Ok(false);
    }

    let (delay, mut tempo, mut kind, dropped) = diagnose(&pts);
    if dropped > 0 {
        println!("  ({dropped} ventana(s) suelta(s) descartada(s) por no cuadrar con el resto)");
    }
    tempo *= tempo0;
    if kind == "constante" && (tempo - 1.0).abs() > 1e-12 {
        kind = "drift";
    }
    let mut ms = (delay * 1000.0).round() as i64;
    let cut_segs = if (tempo0 - 1.0).abs() < 1e-12 && !via_envelope {
        if kind == "cortes" {
            detect_cut_segments(&ref_audio, wave_src, wave_g)
        } else if kind == "constante" {
            let e = with_edge_jumps(&ref_audio, wave_src, delay);
            if !e.is_empty() {
                kind = "cortes";
            }
            e
        } else {
            vec![]
        }
    } else {
        vec![]
    };

    let mut pitch_correct = false;
    if kind == "cortes" {
        if cut_segs.is_empty() {
            println!("  ERROR: el desfase cambia a saltos y no se pudieron localizar los cortes.");
            return Ok(false);
        }
        let desc: Vec<String> = cut_segs
            .iter()
            .map(|s| format!("({}, {:+} ms)", s.start, (s.delay * 1000.0).round() as i64))
            .collect();
        println!("  resultado: cortes por tramos [{}]", desc.join(", "));
    } else if (tempo - 1.0).abs() > 1e-12 {
        let auto_pitch = detect_pitch_shifted(&ref_audio, &src_raw, tempo, &pts, via_envelope);
        pitch_correct = match args.pitch {
            PitchMode::Auto => auto_pitch,
            PitchMode::Correct => true,
            PitchMode::Keep => false,
        };
        let (rf, sf) = infer_fps_pair(ref_info.as_ref().and_then(|i| i.video_fps), tempo);
        let fps_msg = match (rf, sf) {
            (Some(r), Some(s)) => format!(", FPS referencia: {r} · FPS esta pista: {s}"),
            _ => ", mismos FPS (deriva gradual de reloj)".to_string(),
        };
        let semi = pitch_semitones_from_tempo(tempo);
        let pitch_msg = if pitch_correct {
            format!("corregir tono y velocidad ({semi:+.2} semitonos)")
        } else {
            "mantener tono (solo velocidad)".to_string()
        };
        println!("  resultado: {kind}, delay {ms:+} ms, tempo x{tempo:.6}{fps_msg}, {pitch_msg}");
    } else {
        println!("  resultado: {kind}, delay {ms:+} ms");
    }
    if args.dry_run {
        return Ok(true);
    }

    let src_info = media::probe(src_path)?;
    let audio_tracks: Vec<&dubalign::model::TrackInfo> = src_info
        .tracks
        .iter()
        .filter(|t| t.kind == "audio")
        .collect();
    let src_tr = *audio_tracks
        .get(args.src_track)
        .ok_or_else(|| anyhow::anyhow!("pista de audio {} no existe en {}", args.src_track, src_name))?;
    let mut audio_file = src_path.to_path_buf();
    let mut tid = src_tr.id;
    let mut temp_file: Option<PathBuf> = None;
    let delay_s = (ms as f64) / 1000.0;
    let trim_dur = (ref_dur - delay_s).max(0.1);
    let stem = src_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "src".into());

    if !cut_segs.is_empty() {
        std::fs::create_dir_all(&args.work)?;
        let src_end_s = media::stream_end_times(src_path, &src_info)
            .ok()
            .and_then(|m| m.get(&src_tr.id).copied());
        let (seg_file, seg_delay_ms) = render_segments(
            src_path,
            src_tr,
            &cut_segs,
            &args.work,
            &stem,
            "auto",
            Some(ref_dur),
            src_end_s,
        )?;
        audio_file = seg_file.clone();
        temp_file = Some(seg_file);
        tid = 0;
        ms = seg_delay_ms;
    } else if (tempo - 1.0).abs() > 1e-12 {
        std::fs::create_dir_all(&args.work)?;
        audio_file = args.work.join(format!("{stem}.stretched.flac"));
        temp_file = Some(audio_file.clone());
        tid = 0;
        if render_tempo(
            src_path,
            src_tr,
            tempo,
            pitch_correct,
            &audio_file,
            "flac",
            Some(trim_dur),
        )
        .is_err()
        {
            return Ok(false);
        }
    } else if src_raw_dur + delay_s > ref_dur + 0.25 {
        std::fs::create_dir_all(&args.work)?;
        let trimmed = args.work.join(format!("{stem}.trimmed.mka"));
        if trim_stream_copy(src_path, 'a', args.src_track, trim_dur, &trimmed).is_ok() {
            audio_file = trimmed.clone();
            temp_file = Some(trimmed);
        } else {
            let _ = std::fs::remove_file(&trimmed);
            let flac_trim = args.work.join(format!("{stem}.trimmed.flac"));
            render_trim_pcm32_flac(src_path, args.src_track, trim_dur, &flac_trim)?;
            audio_file = flac_trim.clone();
            temp_file = Some(flac_trim);
        }
        tid = 0;
    }

    if let Some(par) = out_path.parent() {
        std::fs::create_dir_all(par)?;
    }

    let mut cmd = Command::new("mkvmerge");
    cmd.args(["-q", "-o"])
        .arg(out_path)
        .arg(ref_path)
        .args([
            "--no-video",
            "--no-chapters",
            "--no-attachments",
            "--no-global-tags",
            "--audio-tracks",
            &tid.to_string(),
            "--sync",
            &format!("{tid}:{ms}"),
            "--language",
            &format!("{tid}:{}", args.lang),
            "--default-track-flag",
            &format!("{tid}:{}", if args.default { 1 } else { 0 }),
        ]);
    if !args.name.is_empty() {
        cmd.args(["--track-name", &format!("{tid}:{}", args.name)]);
    }
    if !args.src_subs {
        cmd.arg("--no-subtitles");
    }
    let st = cmd.arg(&audio_file).status()?;
    if let Some(tf) = temp_file {
        let _ = std::fs::remove_file(tf);
    }
    if st.code().unwrap_or(2) > 1 {
        return Ok(false);
    }
    println!("  -> {}", out_path.display());
    Ok(true)
}

fn media_files_recursive(dir: &Path) -> Vec<(PathBuf, PathBuf)> {
    let mut files: Vec<(PathBuf, PathBuf)> = WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|ext| MEDIA_EXT.contains(&ext.to_ascii_lowercase().as_str()))
        })
        .filter_map(|e| {
            let rel = e.path().strip_prefix(dir).ok()?.to_path_buf();
            Some((rel, e.path().to_path_buf()))
        })
        .collect();
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.r#ref.is_dir() != args.src.is_dir() {
        anyhow::bail!("ref y src deben ser ambos archivos o ambos carpetas");
    }
    let pairs: Vec<(PathBuf, PathBuf, PathBuf)> = if args.r#ref.is_dir() {
        let refs = media_files_recursive(&args.r#ref);
        let srcs = media_files_recursive(&args.src);
        if refs.len() != srcs.len() || refs.is_empty() {
            anyhow::bail!("las carpetas no cuadran: {} ref vs {} src", refs.len(), srcs.len());
        }
        refs.into_iter()
            .zip(srcs)
            .map(|((rel, r), (_, s))| {
                let mut out_rel = rel;
                out_rel.set_extension("mkv");
                (r, s, args.output.join(out_rel))
            })
            .collect()
    } else {
        let ref_stem = args
            .r#ref
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "out".into());
        vec![(
            args.r#ref.clone(),
            args.src.clone(),
            args.output.join(format!("{ref_stem}.mkv")),
        )]
    };

    let mut failed = Vec::new();
    for (r, s, o) in &pairs {
        if !process(r, s, o, &args)? {
            failed.push(
                r.file_name()
                    .map(|x| x.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            );
        }
    }
    if !failed.is_empty() {
        anyhow::bail!("Fallaron: {}", failed.join(", "));
    }
    Ok(())
}
