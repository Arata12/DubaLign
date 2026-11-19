use clap::{Parser, ValueEnum};
use dubalign::{
    model::Report,
    sync_algo::{analyze_file, apply_with_ref, plan_is_empty, propose, SUB_MIN_CONF},
};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const VIDEO_EXT: &[&str] = &["mkv", "mp4", "m4v", "avi", "ts", "m2ts", "webm"];

#[derive(Clone, ValueEnum, PartialEq, Eq)]
enum Codec {
    /// Sin recodificar por duplicación/dropeo de frames cuando sea posible; si no, PCM 32 bits -> FLAC compresión máxima
    Auto,
    /// Forzar PCM 32 bits -> FLAC compresión máxima
    Flac,
    /// Recodificar a AAC
    Aac,
}

#[derive(Parser)]
#[command(about = "Revisa archivos o carpetas con varios audios y subtítulos, y corrige los que estén desfasados respecto a la pista de referencia.")]
struct Args {
    #[arg(required = true)]
    files: Vec<PathBuf>,
    #[arg(short, long, default_value = "output")]
    output: PathBuf,
    #[arg(long, default_value_t = 0)]
    ref_track: usize,
    #[arg(long, value_enum, default_value_t = Codec::Auto)]
    codec: Codec,
    #[arg(long)]
    dry_run: bool,
}

fn label(status: &str) -> &str {
    match status {
        "ref" => "referencia",
        "ok" => "ok",
        "offset" => "DESFASADO",
        "cuts" => "CORTES",
        "tempo" => "VELOCIDAD DISTINTA",
        "unknown" => "no medible",
        s => s,
    }
}

fn show(report: &Report) {
    let ref_dur = report
        .audio
        .iter()
        .find(|a| a.n == report.ref_track)
        .and_then(|a| a.duration_s);
    for a in &report.audio {
        let ms = match a.delay_ms {
            None => "".to_string(),
            Some(v) => format!("{v:+} ms"),
        };
        let extra = if !a.segments.is_empty() {
            let list: Vec<String> = a
                .segments
                .iter()
                .map(|s| format!("({}, {})", s.start, (s.delay * 1000.0).round() as i64))
                .collect();
            format!("  tramos: [{}]", list.join(", "))
        } else if a.status == "tempo" {
            let fps_txt = match (a.ref_fps, a.src_fps) {
                (Some(rf), Some(sf)) => format!("FPS ref: {rf} · FPS esta pista: {sf} · "),
                _ => match report.video_fps {
                    Some(vf) => format!("mismos FPS ({vf}), deriva de reloj · "),
                    None => "deriva de reloj · ".to_string(),
                },
            };
            let semi = a.pitch_semitones.unwrap_or(0.0);
            let pitch_txt = if a.pitch_shift == Some(true) {
                format!("tono alterado ({semi:+.2} st) -> corregir tono")
            } else if semi.abs() < 0.05 {
                "deriva mínima -> mantener tono".to_string()
            } else {
                "tono correcto -> mantener tono".to_string()
            };
            format!("  [{fps_txt}x{:.6} · {pitch_txt}]", a.tempo)
        } else {
            String::new()
        };
        let trim_note = match (ref_dur, a.duration_s) {
            (Some(rd), Some(sd)) if a.n != report.ref_track && sd > rd + 0.25 => {
                format!("  [+{:.1}s final -> recorte a {:.1}s]", sd - rd, rd)
            }
            _ => String::new(),
        };
        let reason = a
            .reason
            .as_deref()
            .map(|r| format!("  ({r})"))
            .unwrap_or_default();
        println!(
            "  audio a:{} {:<4} {:>9}  {}{}{}{}",
            a.n,
            a.lang,
            ms,
            label(&a.status),
            extra,
            trim_note,
            reason
        );
    }
    for s in &report.subs {
        let ms = if s.conf < SUB_MIN_CONF || s.offset_ms.is_none() {
            String::new()
        } else {
            format!("{:+} ms", s.offset_ms.unwrap())
        };
        let twin_note = if !s.twin_segments.is_empty() {
            format!("  [heredado audio {}: cortes ({} tramos)]", s.lang, s.twin_segments.len())
        } else if let Some(tp) = s.twin_tempo {
            format!("  [heredado audio {}: velocidad x{tp:.6}]", s.lang)
        } else {
            s.twin_delay_ms
                .map(|td| format!("  [heredado audio {}: {:+} ms]", s.lang, td))
                .unwrap_or_default()
        };
        println!(
            "  subs  s:{} {:<4} {:>9}  {}  (corr {}){}  {}",
            s.n,
            s.lang,
            ms,
            label(&s.status),
            s.conf,
            twin_note,
            s.name
        );
    }
}

fn collect_inputs(inputs: &[PathBuf], out_dir: &Path) -> Vec<(PathBuf, PathBuf)> {
    let media_env = std::env::var("MEDIA_DIR").ok().and_then(|s| std::fs::canonicalize(s).ok());
    let out_env = std::env::var("OUTPUT_DIR").ok().map(PathBuf::from);
    let using_default_out = out_dir == Path::new("output")
        || out_env.as_ref().is_some_and(|eo| eo == out_dir);

    let mut pairs = Vec::new();
    for inp in inputs {
        if inp.is_dir() {
            let dir_canon = std::fs::canonicalize(inp).unwrap_or_else(|_| inp.clone());
            let rel_root = if using_default_out {
                if let Some(ref m) = media_env {
                    if dir_canon.starts_with(m) {
                        m.clone()
                    } else {
                        dir_canon.clone()
                    }
                } else {
                    dir_canon.clone()
                }
            } else {
                dir_canon.clone()
            };
            let mut found: Vec<PathBuf> = WalkDir::new(&dir_canon)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
                .filter(|e| {
                    e.path()
                        .extension()
                        .and_then(|s| s.to_str())
                        .is_some_and(|ext| VIDEO_EXT.contains(&ext.to_ascii_lowercase().as_str()))
                })
                .map(|e| e.path().to_path_buf())
                .collect();
            found.sort();
            for f in found {
                let rel = f.strip_prefix(&rel_root).unwrap_or_else(|_| Path::new(f.file_name().unwrap()));
                pairs.push((f.clone(), out_dir.join(rel)));
            }
        } else {
            let name = inp
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "out.mkv".to_string());
            pairs.push((inp.clone(), out_dir.join(name)));
        }
    }
    pairs
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let codec_override = match args.codec {
        Codec::Auto => None,
        Codec::Flac => Some("flac"),
        Codec::Aac => Some("aac"),
    };

    let targets = collect_inputs(&args.files, &args.output);
    if targets.is_empty() {
        anyhow::bail!("no se encontraron videos en las rutas indicadas");
    }

    for (f, out_path) in &targets {
        let label_name = f
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| f.display().to_string());
        println!("\n== {label_name}");
        let report = analyze_file(f, args.ref_track, &|_, _| {})?;
        show(&report);
        let mut plan = propose(&report);
        if let Some(c) = codec_override {
            for p in plan.audio.values_mut() {
                if p.codec.is_some() {
                    p.codec = Some(c.to_string());
                }
            }
        }
        if plan_is_empty(&plan) || args.dry_run {
            continue;
        }
        let out = apply_with_ref(f, &plan, out_path, None, args.ref_track)?;
        println!("  -> {}\n  verificación del corregido:", out.display());
        let verify = analyze_file(&out, args.ref_track, &|_, _| {})?;
        show(&verify);
    }
    Ok(())
}
