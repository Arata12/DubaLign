use crate::model::{round2, ProbeInfo, TrackInfo};
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use tempfile::TempDir;

pub const TEXT_SUBS: &[&str] = &["SubStationAlpha", "SubRip/SRT", "WebVTT", "Timed Text"];

pub fn is_text_sub(codec: &str) -> bool {
    TEXT_SUBS.iter().any(|p| codec.starts_with(p))
}

#[derive(Deserialize)]
struct MkvMergeOutput {
    #[serde(default)]
    tracks: Vec<MkvTrack>,
    container: Option<MkvContainer>,
}

#[derive(Deserialize)]
struct MkvContainer {
    properties: Option<MkvContainerProps>,
}

#[derive(Deserialize)]
struct MkvContainerProps {
    duration: Option<u64>,
}

#[derive(Deserialize)]
struct MkvTrack {
    id: u64,
    #[serde(rename = "type")]
    kind: String,
    codec: String,
    #[serde(default)]
    properties: MkvTrackProps,
}

#[derive(Deserialize, Default)]
struct MkvTrackProps {
    language: Option<String>,
    track_name: Option<String>,
    default_track: Option<bool>,
    default_duration: Option<u64>,
    audio_channels: Option<u32>,
    audio_sampling_frequency: Option<u32>,
}

pub fn probe(path: &Path) -> Result<ProbeInfo> {
    let out = Command::new("mkvmerge")
        .arg("-J")
        .arg(path)
        .output()
        .context("failed to execute mkvmerge")?;
    if !out.status.success() && out.stdout.is_empty() {
        bail!(
            "mkvmerge -J failed on {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let parsed: MkvMergeOutput =
        serde_json::from_slice(&out.stdout).context("invalid mkvmerge -J JSON")?;

    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut tracks = Vec::with_capacity(parsed.tracks.len());
    let mut video_fps = None;
    for t in parsed.tracks {
        if t.kind == "video" && video_fps.is_none() {
            if let Some(ns) = t.properties.default_duration.filter(|&d| d > 0) {
                video_fps = Some(1_000_000_000.0 / (ns as f64));
            }
        }
        let n = {
            let e = counts.entry(t.kind.clone()).or_insert(0);
            let cur = *e;
            *e += 1;
            cur
        };
        tracks.push(TrackInfo {
            id: t.id,
            kind: t.kind,
            n,
            codec: t.codec,
            lang: t.properties.language.unwrap_or_else(|| "und".to_string()),
            name: t.properties.track_name.unwrap_or_default(),
            default: t.properties.default_track.unwrap_or(false),
            channels: t.properties.audio_channels,
            sample_rate: t.properties.audio_sampling_frequency,
        });
    }
    let duration = parsed
        .container
        .and_then(|c| c.properties)
        .and_then(|p| p.duration)
        .map(|ns| (ns as f64) / 1e9);
    Ok(ProbeInfo {
        duration,
        video_fps,
        tracks,
    })
}

#[derive(Deserialize)]
struct FfprobeStreamMeta {
    index: usize,
    codec_type: String,
    #[serde(default)]
    tags: HashMap<String, String>,
}

#[derive(Deserialize)]
struct FfprobeOut {
    #[serde(default)]
    streams: Vec<FfprobeStreamMeta>,
}

#[derive(Deserialize)]
struct FfprobePacketsOut {
    #[serde(default)]
    packets: Vec<FfprobePacket>,
}

#[derive(Deserialize)]
struct FfprobePacket {
    #[serde(default)]
    stream_index: usize,
    pts_time: Option<String>,
    duration_time: Option<String>,
}

#[derive(Deserialize)]
struct FfprobeAudioDetailOut {
    #[serde(default)]
    streams: Vec<FfprobeAudioStream>,
    #[serde(default)]
    packets: Vec<FfprobePacket>,
}

#[derive(Deserialize)]
struct FfprobeAudioStream {
    codec_name: Option<String>,
    profile: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AudioStreamDetail {
    pub codec_name: String,
    pub profile: String,
    pub first_pts: f64,
}

/// Probes codec name, profile, and initial PTS of audio stream `track_n`.
pub fn probe_audio_stream_detail(path: &Path, track_n: usize) -> Result<AudioStreamDetail> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            &format!("a:{track_n}"),
            "-read_intervals",
            "%+#1",
            "-show_entries",
            "stream=codec_name,profile:packet=pts_time",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .context("failed to execute ffprobe for audio detail")?;
    if !out.status.success() {
        bail!("ffprobe detail failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    let parsed: FfprobeAudioDetailOut =
        serde_json::from_slice(&out.stdout).context("invalid ffprobe audio detail JSON")?;
    let st = parsed
        .streams
        .into_iter()
        .next()
        .context("missing audio stream in ffprobe detail")?;
    let first_pts = parsed
        .packets
        .first()
        .and_then(|pk| pk.pts_time.as_deref())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    Ok(AudioStreamDetail {
        codec_name: st.codec_name.unwrap_or_default().to_ascii_lowercase(),
        profile: st.profile.unwrap_or_else(|| "unknown".to_string()),
        first_pts,
    })
}

pub fn ffprobe_audio_labels(path: &Path) -> Result<Vec<String>> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index,codec_type:stream_tags=language,title",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .context("failed to execute ffprobe")?;
    if !out.status.success() {
        bail!("ffprobe failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    let parsed: FfprobeOut = serde_json::from_slice(&out.stdout).context("invalid ffprobe JSON")?;
    Ok(parsed
        .streams
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            let lang = s.tags.get("language").map(String::as_str).unwrap_or("?");
            let title = s.tags.get("title").map(String::as_str).unwrap_or("");
            format!("a:{i} {lang} {title}").trim().to_string()
        })
        .collect())
}

/// Detects the exact end timestamp (in seconds) for every stream in `path` (mapped by `TrackInfo.id`).
pub fn stream_end_times(path: &Path, info: &ProbeInfo) -> Result<HashMap<u64, f64>> {
    let st_out = Command::new("ffprobe")
        .args(["-v", "error", "-show_entries", "stream=index,codec_type", "-of", "json"])
        .arg(path)
        .output()
        .context("ffprobe streams")?;
    let parsed_st: FfprobeOut = serde_json::from_slice(&st_out.stdout).unwrap_or(FfprobeOut { streams: vec![] });

    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut ff_idx_to_track_id: HashMap<usize, u64> = HashMap::new();
    for s in &parsed_st.streams {
        let kind = if s.codec_type == "subtitle" {
            "subtitles"
        } else {
            s.codec_type.as_str()
        };
        let n = {
            let e = counts.entry(kind.to_string()).or_insert(0);
            let cur = *e;
            *e += 1;
            cur
        };
        if let Some(t) = info.tracks.iter().find(|t| t.kind == kind && t.n == n) {
            ff_idx_to_track_id.insert(s.index, t.id);
        }
    }

    let scan_packets = |read_from: Option<f64>| -> Result<HashMap<usize, f64>> {
        let mut cmd = Command::new("ffprobe");
        cmd.args(["-v", "error"]);
        if let Some(t0) = read_from {
            cmd.args(["-read_intervals", &format!("{t0:.0}%")]);
        }
        let out = cmd
            .args(["-show_entries", "packet=stream_index,pts_time,duration_time", "-of", "json"])
            .arg(path)
            .output()
            .context("ffprobe packets")?;
        if !out.status.success() {
            bail!("ffprobe packets failed: {}", String::from_utf8_lossy(&out.stderr));
        }
        let pkts: FfprobePacketsOut = serde_json::from_slice(&out.stdout)?;
        let mut ends: HashMap<usize, f64> = HashMap::new();
        for pk in pkts.packets {
            let pts = pk.pts_time.as_deref().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            let dur = pk.duration_time.as_deref().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            let end = pts + dur;
            let entry = ends.entry(pk.stream_index).or_insert(0.0);
            if end > *entry {
                *entry = end;
            }
        }
        Ok(ends)
    };

    let mut ends = if let Some(d) = info.duration.filter(|&d| d > 150.0) {
        let fast = scan_packets(Some((d - 120.0).max(0.0)))?;
        let audio_missing = parsed_st
            .streams
            .iter()
            .any(|s| s.codec_type == "audio" && !fast.contains_key(&s.index));
        if audio_missing {
            scan_packets(None)?
        } else {
            fast
        }
    } else {
        scan_packets(None)?
    };

    let mut res = HashMap::new();
    for (ff_idx, tid) in ff_idx_to_track_id {
        if let Some(end) = ends.remove(&ff_idx) {
            res.insert(tid, end);
        }
    }
    Ok(res)
}

/// Detects the exact duration (seconds) of audio track `track_n` via `ffprobe`.
pub fn audio_duration(path: &Path, track_n: usize) -> Result<f64> {
    let info = probe(path)?;
    let tid = info
        .tracks
        .iter()
        .find(|t| t.kind == "audio" && t.n == track_n)
        .map(|t| t.id)
        .context("pista de audio no encontrada")?;
    let ends = stream_end_times(path, &info)?;
    if let Some(&dur) = ends.get(&tid) {
        return Ok(dur);
    }
    if let Some(d) = info.duration {
        return Ok(d);
    }
    bail!("no se pudo determinar la duración del audio {track_n}")
}

pub fn load_audio(path: &Path, track: usize, sr: u32) -> Result<Vec<f32>> {
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-map",
            &format!("0:a:{track}"),
            "-af",
            "aresample=async=1:first_pts=0",
            "-ac",
            "1",
            "-ar",
            &sr.to_string(),
            "-f",
            "s16le",
            "-",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn ffmpeg")?;

    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let err_thread = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s);
        s
    });

    // Reserve roughly ~24 min of mono audio to avoid repeated reallocs
    let mut out = Vec::with_capacity((sr as usize) * 1450);
    let mut buf = [0u8; 65536];
    let mut leftover: Option<u8> = None;
    let inv = 1.0f32 / 32768.0f32;
    loop {
        let n = stdout.read(&mut buf)?;
        if n == 0 {
            break;
        }
        let mut start = 0usize;
        if let Some(prev) = leftover.take() {
            let v = i16::from_le_bytes([prev, buf[0]]);
            out.push((v as f32) * inv);
            start = 1;
        }
        let slice = &buf[start..n];
        for chunk in slice.chunks_exact(2) {
            let v = i16::from_le_bytes([chunk[0], chunk[1]]);
            out.push((v as f32) * inv);
        }
        if slice.len() % 2 == 1 {
            leftover = Some(slice[slice.len() - 1]);
        }
    }

    let status = child.wait()?;
    let err_msg = err_thread.join().unwrap_or_default();
    if !status.success() || out.is_empty() {
        bail!(
            "ffmpeg no pudo leer el audio {track} de {}:\n{err_msg}",
            path.display(),
        );
    }
    Ok(out)
}

/// Decodes all audio tracks `[0..n_tracks)` at `sr` Hz in parallel via stdout pipes (zero disk I/O).
pub fn load_audio_all_single_pass(
    path: &Path,
    n_tracks: usize,
    sr: u32,
) -> Result<Vec<Vec<f32>>> {
    if n_tracks == 0 {
        return Ok(vec![]);
    }
    if n_tracks == 1 {
        return Ok(vec![load_audio(path, 0, sr)?]);
    }
    (0..n_tracks)
        .into_par_iter()
        .map(|i| load_audio(path, i, sr))
        .collect()
}

/// Loads mono f32 audio slice matching `spectro.load`.
pub fn load_audio_slice(
    path: &Path,
    track: usize,
    start: Option<f64>,
    dur: Option<f64>,
    sr: u32,
) -> Result<Vec<f32>> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-v", "error"]);
    let mut pad = 0.0f64;
    if let Some(st) = start {
        let d = dur.expect("dur required when start is provided");
        pad = (-st).max(0.0).min(d);
        if d - pad <= 0.0 {
            return Ok(vec![0.0f32; (d * (sr as f64)) as usize]);
        }
        cmd.args([
            "-ss",
            &format!("{:.4}", st.max(0.0)),
            "-t",
            &format!("{:.4}", d - pad),
        ]);
    }
    let out = cmd
        .arg("-i")
        .arg(path)
        .args([
            "-map",
            &format!("0:a:{track}"),
            "-af",
            "aresample=async=1:first_pts=0",
            "-ac",
            "1",
            "-ar",
            &sr.to_string(),
            "-f",
            "s16le",
            "-",
        ])
        .output()
        .context("failed to run ffmpeg slice")?;
    if !out.status.success() {
        bail!("{}", String::from_utf8_lossy(&out.stderr));
    }
    let mut x = Vec::with_capacity(out.stdout.len() / 2);
    let inv = 1.0f32 / 32768.0f32;
    for chunk in out.stdout.chunks_exact(2) {
        let v = i16::from_le_bytes([chunk[0], chunk[1]]);
        x.push((v as f32) * inv);
    }
    let Some(_) = start else {
        return Ok(x);
    };
    let d = dur.unwrap();
    let pad_n = (pad * (sr as f64)) as usize;
    let target_n = (d * (sr as f64)) as usize;
    let mut res = Vec::with_capacity(target_n);
    res.resize(pad_n.min(target_n), 0.0);
    let take = (target_n.saturating_sub(res.len())).min(x.len());
    res.extend_from_slice(&x[..take]);
    if res.len() < target_n {
        res.resize(target_n, 0.0);
    }
    Ok(res)
}

/// Loads mono f32 audio slice applying `tempo`, `delay_s`, and `pitch_correct` (varispeed vs atempo).
#[allow(clippy::too_many_arguments)]
pub fn load_audio_slice_tempo(
    path: &Path,
    track: usize,
    out_start: f64,
    out_dur: f64,
    sr: u32,
    tempo: f64,
    delay_s: f64,
    pitch_correct: bool,
) -> Result<Vec<f32>> {
    let target_n = (out_dur * (sr as f64)) as usize;
    if tempo <= 0.05 {
        return load_audio_slice(path, track, Some(out_start - delay_s), Some(out_dur), sr);
    }
    let st_str = out_start - delay_s;
    let pad = (-st_str).max(0.0).min(out_dur);
    let rem_out = out_dur - pad;
    if rem_out <= 0.0 {
        return Ok(vec![0.0f32; target_n]);
    }
    let src_ss = st_str.max(0.0) * tempo;
    let src_dur = rem_out * tempo;
    let af = if pitch_correct {
        let new_sr = (sr as f64) * tempo;
        format!(
            "aresample=async=1:first_pts=0,aformat=sample_fmts=flt,aresample={sr},asetrate={new_sr:.4},aresample={sr}:resampler=soxr:precision=28"
        )
    } else {
        format!("aresample=async=1:first_pts=0,aformat=sample_fmts=flt,atempo={tempo:.8}")
    };
    let out = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-ss",
            &format!("{src_ss:.4}"),
            "-t",
            &format!("{src_dur:.4}"),
            "-i",
        ])
        .arg(path)
        .args([
            "-map",
            &format!("0:a:{track}"),
            "-af",
            &af,
            "-ac",
            "1",
            "-ar",
            &sr.to_string(),
            "-f",
            "s16le",
            "-",
        ])
        .output()
        .context("failed to run ffmpeg tempo slice")?;
    if !out.status.success() {
        bail!("{}", String::from_utf8_lossy(&out.stderr));
    }
    let inv = 1.0f32 / 32768.0f32;
    let pad_n = (pad * (sr as f64)) as usize;
    let mut res = Vec::with_capacity(target_n);
    res.resize(pad_n.min(target_n), 0.0);
    for chunk in out.stdout.chunks_exact(2) {
        if res.len() >= target_n {
            break;
        }
        let v = i16::from_le_bytes([chunk[0], chunk[1]]);
        res.push((v as f32) * inv);
    }
    if res.len() < target_n {
        res.resize(target_n, 0.0);
    }
    Ok(res)
}

fn parse_ass_time(s: &str) -> Option<f64> {
    let mut parts = s.trim().split(':');
    let h: f64 = parts.next()?.parse().ok()?;
    let m: f64 = parts.next()?.parse().ok()?;
    let sec: f64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some(h * 3600.0 + m * 60.0 + sec)
}

pub fn parse_ass_dialogue(text: &str) -> Vec<(f64, f64)> {
    let mut ev = Vec::new();
    for line in text.lines() {
        if !line.starts_with("Dialogue:") {
            continue;
        }
        let parts: Vec<&str> = line.splitn(10, ',').collect();
        if parts.len() < 10 {
            continue;
        }
        if parts[9].contains("\\pos(") || parts[9].contains("\\move(") {
            continue;
        }
        if let (Some(a), Some(b)) = (parse_ass_time(parts[1]), parse_ass_time(parts[2])) {
            let dur = b - a;
            if dur > 0.2 && dur < 12.0 {
                ev.push((round2(a), round2(b)));
            }
        }
    }
    ev.sort_by(|x, y| x.0.total_cmp(&y.0).then_with(|| x.1.total_cmp(&y.1)));
    ev
}

pub fn sub_events_single(path: &Path, n: usize) -> Option<Vec<(f64, f64)>> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-map", &format!("0:s:{n}"), "-f", "ass", "-"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Some(parse_ass_dialogue(&text))
}

/// Extracts all text subtitle tracks in one ffmpeg pass (falls back per-track on error).
pub fn extract_all_sub_events(
    path: &Path,
    subs: &[&TrackInfo],
) -> HashMap<usize, Option<Vec<(f64, f64)>>> {
    let text_tracks: Vec<usize> = subs
        .iter()
        .filter(|t| is_text_sub(&t.codec))
        .map(|t| t.n)
        .collect();
    let mut res = HashMap::new();
    if text_tracks.is_empty() {
        return res;
    }
    if text_tracks.len() == 1 {
        let n = text_tracks[0];
        res.insert(n, sub_events_single(path, n));
        return res;
    }

    if let Ok(tmp) = TempDir::new() {
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-v", "error", "-y", "-i"]).arg(path);
        let mut files = Vec::with_capacity(text_tracks.len());
        for &n in &text_tracks {
            let f = tmp.path().join(format!("s{n}.ass"));
            cmd.args(["-map", &format!("0:s:{n}"), "-f", "ass"]).arg(&f);
            files.push((n, f));
        }
        if let Ok(out) = cmd.stdin(Stdio::null()).output() {
            if out.status.success() {
                for (n, f) in files {
                    if let Ok(bytes) = std::fs::read(&f) {
                        let text = String::from_utf8_lossy(&bytes);
                        res.insert(n, Some(parse_ass_dialogue(&text)));
                    } else {
                        res.insert(n, None);
                    }
                }
                return res;
            }
        }
    }

    // Fallback per-track
    for &n in &text_tracks {
        res.insert(n, sub_events_single(path, n));
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ass_dialogue_filters_signs_and_short_long() {
        let ass = "\
Dialogue: 0,0:00:07.73,0:00:13.11,Default,,0,0,0,,Hola mundo
Dialogue: 0,0:00:08.00,0:00:09.00,Sign,,0,0,0,,{\\pos(100,200)}Cartel
Dialogue: 0,0:00:10.00,0:00:11.00,Sign,,0,0,0,,{\\move(0,0,10,10)}Mov
Dialogue: 0,0:00:12.00,0:00:12.10,Default,,0,0,0,,Too short
Dialogue: 0,0:00:15.00,0:00:30.00,Default,,0,0,0,,Too long
Dialogue: 0,0:00:01.50,0:00:03.25,Default,,0,0,0,,Primero
";
        let ev = parse_ass_dialogue(ass);
        assert_eq!(ev, vec![(1.5, 3.25), (7.73, 13.11)]);
    }

    #[test]
    fn test_ffprobe_audio_detail_deserializes_without_stream_index() {
        let json = r#"{
            "packets": [{"pts_time": "0.021000"}],
            "streams": [{"codec_name": "aac", "profile": "LC"}]
        }"#;
        let parsed: FfprobeAudioDetailOut = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.streams[0].codec_name.as_deref(), Some("aac"));
        assert_eq!(parsed.streams[0].profile.as_deref(), Some("LC"));
        assert_eq!(parsed.packets[0].pts_time.as_deref(), Some("0.021000"));
    }
}
