use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeMap;

pub fn clean_zero(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x
    }
}

pub fn round1(x: f64) -> f64 {
    clean_zero((x * 10.0).round() / 10.0)
}

pub fn round2(x: f64) -> f64 {
    clean_zero((x * 100.0).round() / 100.0)
}

pub fn round3(x: f64) -> f64 {
    clean_zero((x * 1000.0).round() / 1000.0)
}

pub fn round7(x: f64) -> f64 {
    clean_zero((x * 10_000_000.0).round() / 10_000_000.0)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackInfo {
    pub id: u64,
    #[serde(rename = "type")]
    pub kind: String,
    pub n: usize,
    pub codec: String,
    pub lang: String,
    pub name: String,
    pub default: bool,
    pub channels: Option<u32>,
    pub sample_rate: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeInfo {
    pub duration: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_fps: Option<f64>,
    pub tracks: Vec<TrackInfo>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub end: Option<f64>,
    pub delay: f64,
}

/// (t_sec, delay_ms, conf)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MeasurePoint(pub f64, pub i64, pub f64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioReport {
    pub n: usize,
    pub id: u64,
    pub lang: String,
    pub name: String,
    pub codec: String,
    pub channels: Option<u32>,
    pub status: String,
    pub delay_ms: Option<i64>,
    pub tempo: f64,
    pub points: Vec<MeasurePoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dense: Vec<MeasurePoint>,
    pub segments: Vec<Segment>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ref_fps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub src_fps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch_shift: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch_semitones: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubReport {
    pub n: usize,
    pub id: u64,
    pub lang: String,
    pub name: String,
    pub codec: String,
    pub status: String,
    pub offset_ms: Option<i64>,
    pub conf: f64,
    pub events: Vec<(f64, f64)>,
    pub twin_delay_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub twin_segments: Vec<Segment>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub twin_tempo: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub file: String,
    pub duration: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_fps: Option<f64>,
    pub ref_track: usize,
    pub status: String,
    pub suggest_start: f64,
    pub audio: Vec<AudioReport>,
    pub subs: Vec<SubReport>,
}

fn default_mode() -> String {
    "none".to_string()
}

fn de_opt_i64_flex<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: Deserializer<'de>,
{
    let val = Option::<serde_json::Value>::deserialize(deserializer)?;
    match val {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                Ok(Some(i))
            } else if let Some(f) = n.as_f64() {
                Ok(Some(f.round() as i64))
            } else {
                Err(serde::de::Error::custom("invalid number"))
            }
        }
        Some(serde_json::Value::String(s)) => {
            if s.trim().is_empty() {
                Ok(None)
            } else if let Ok(f) = s.trim().parse::<f64>() {
                Ok(Some(f.round() as i64))
            } else {
                Err(serde::de::Error::custom("invalid numeric string"))
            }
        }
        _ => Err(serde::de::Error::custom("expected number or null")),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AudioPlanEntry {
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default, deserialize_with = "de_opt_i64_flex", skip_serializing_if = "Option::is_none")]
    pub delay_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<Segment>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tempo: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch_correct: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codec: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SubPlanEntry {
    #[serde(default, deserialize_with = "de_opt_i64_flex")]
    pub delay_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<Segment>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tempo: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Plan {
    #[serde(default)]
    pub audio: BTreeMap<String, AudioPlanEntry>,
    #[serde(default)]
    pub subs: BTreeMap<String, SubPlanEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedRecord {
    pub plan: Plan,
    pub verify: Report,
    pub t: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: String,
    pub folder: String,
    pub name: String,
    pub size: u64,
    pub status: Option<String>,
    pub saved: bool,
    pub saved_status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobInfo {
    pub id: String,
    pub kind: String,
    pub file: String,
    pub state: String,
    pub progress: f64,
    pub msg: String,
    pub error: Option<String>,
    pub t: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateResponse {
    pub files: Vec<FileEntry>,
    pub jobs: Vec<JobInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportResponse {
    pub report: Report,
    pub proposed: Plan,
    pub saved: Option<SavedRecord>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnalyzeReq {
    pub files: Vec<String>,
    #[serde(default)]
    pub force: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProcessReq {
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default)]
    pub folder: Option<String>,
    #[serde(default)]
    pub force: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SaveReq {
    pub file: String,
    pub plan: Plan,
}

#[derive(Debug, Clone, Serialize)]
pub struct EnvelopeData {
    pub fps: usize,
    pub db_lo: i32,
    pub tracks: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TimelineMeta {
    pub key: String,
    pub duration: Option<f64>,
    pub tracks: Vec<String>,
    pub spec_fps: usize,
    pub levels: Vec<(f64, f64)>,
    pub env: EnvelopeData,
}
