use axum::{
    body::Body,
    extract::{Query, Request, State},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use dubalign::{
    dsp, file_key, media,
    model::{
        round3, AnalyzeReq, FileEntry, JobInfo, Plan, ProcessReq, Report, ReportResponse,
        SaveReq, SavedRecord, StateResponse, TimelineMeta,
    },
    spectro, sync_algo, timeline,
};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::json;
use sha1::{Digest, Sha1};
use std::{
    collections::{HashMap, VecDeque},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Condvar, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};
use tower::ServiceExt;
use tower_http::services::ServeFile;
use walkdir::WalkDir;

const VIDEO_EXT: &[&str] = &["mkv", "mp4", "m4v", "avi", "ts", "m2ts", "webm"];

fn now_sec() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn now_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

#[derive(Clone)]
struct AppState {
    media_dir: PathBuf,
    output_dir: PathBuf,
    work_dir: PathBuf,
    static_dir: PathBuf,
    jobs: Arc<Mutex<Vec<JobInfo>>>,
    queue: Arc<(Mutex<VecDeque<(String, Option<Plan>)>>, Condvar)>,
}

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "detail": self.1 }))).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
    }
}

fn resolve_file(base: &Path, rel: &str) -> Result<PathBuf, ApiError> {
    if Path::new(rel).is_absolute() {
        return Err(ApiError(
            StatusCode::NOT_FOUND,
            "archivo no encontrado".into(),
        ));
    }
    let joined = base.join(rel);
    let canon = fs::canonicalize(&joined).map_err(|_| {
        ApiError(
            StatusCode::NOT_FOUND,
            "archivo no encontrado".into(),
        )
    })?;
    if !canon.starts_with(base) || !canon.is_file() {
        return Err(ApiError(
            StatusCode::NOT_FOUND,
            "archivo no encontrado".into(),
        ));
    }
    Ok(canon)
}

fn read_json<T: serde::de::DeserializeOwned>(p: &Path) -> Option<T> {
    let bytes = fs::read(p).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn report_for(work: &Path, p: &Path) -> Option<Report> {
    let k = file_key(p).ok()?;
    let rp = work.join("reports").join(format!("{k}.json"));
    let mut rep: Report = read_json(&rp)?;
    let mut dirty = sync_algo::populate_fps_info(&mut rep);
    if sync_algo::apply_sub_twin(&rep.audio, &mut rep.subs) {
        rep.status = sync_algo::recompute_worst_status(&rep.audio, &rep.subs);
        dirty = true;
    }
    if dirty {
        if let Ok(bytes) = serde_json::to_vec(&rep) {
            let _ = fs::write(&rp, bytes);
        }
    }
    Some(rep)
}

fn saved_for(work: &Path, p: &Path) -> Option<SavedRecord> {
    let k = file_key(p).ok()?;
    read_json(&work.join("saved").join(format!("{k}.json")))
}

fn enrich_durations(p: &Path, rep: &mut Report) -> bool {
    let need_dur = rep.audio.iter().any(|a| a.duration_s.is_none());
    let need_fps = rep.video_fps.is_none();
    if !need_dur && !need_fps {
        return false;
    }
    let Ok(info) = media::probe(p) else {
        return false;
    };
    let mut changed = false;
    if need_fps {
        if let Some(vf) = info.video_fps {
            rep.video_fps = Some(sync_algo::snap_fps(vf));
            changed = true;
        }
    }
    if sync_algo::populate_fps_info(rep) {
        changed = true;
    }
    if need_dur {
        if let Ok(ends) = media::stream_end_times(p, &info) {
            for a in &mut rep.audio {
                if a.duration_s.is_none() {
                    if let Some(&end) = ends.get(&a.id) {
                        a.duration_s = Some(round3(end));
                        changed = true;
                    }
                }
            }
        }
    }
    changed
}

fn list_media_files(media_dir: &Path) -> Vec<(String, PathBuf)> {
    let mut entries: Vec<(String, PathBuf)> = WalkDir::new(media_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|ext| VIDEO_EXT.contains(&ext.to_ascii_lowercase().as_str()))
        })
        .filter_map(|e| {
            let rel = e
                .path()
                .strip_prefix(media_dir)
                .ok()?
                .to_string_lossy()
                .into_owned();
            Some((rel, e.path().to_path_buf()))
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

fn update_job(jobs: &Arc<Mutex<Vec<JobInfo>>>, id: &str, f: impl FnOnce(&mut JobInfo)) {
    let mut guard = jobs.lock().unwrap();
    if let Some(j) = guard.iter_mut().find(|j| j.id == id) {
        f(j);
    }
}

fn enqueue_job(state: &AppState, kind: &str, rel: &str, plan: Option<Plan>) -> String {
    let mut jobs = state.jobs.lock().unwrap();
    for j in jobs.iter() {
        if j.kind == kind && j.file == rel && (j.state == "queued" || j.state == "running") {
            return j.id.clone();
        }
    }
    let jid = format!("{:x}", now_ns());
    jobs.push(JobInfo {
        id: jid.clone(),
        kind: kind.to_string(),
        file: rel.to_string(),
        state: "queued".to_string(),
        progress: 0.0,
        msg: "en cola".to_string(),
        error: None,
        t: now_sec(),
    });
    if jobs.len() > 500 {
        let drain = jobs.len() - 500;
        jobs.drain(..drain);
    }
    drop(jobs);

    let (lock, cvar) = &*state.queue;
    let mut q = lock.lock().unwrap();
    q.push_back((jid.clone(), plan));
    cvar.notify_one();
    jid
}

fn analyze_and_cache(
    path: &Path,
    key: &str,
    timeline_root: &Path,
    on_progress: impl Fn(f64, &str) + Sync + Send,
) -> anyhow::Result<Report> {
    let info = media::probe(path)?;
    let n_audio = info.tracks.iter().filter(|t| t.kind == "audio").count();
    let subs: Vec<&dubalign::model::TrackInfo> = info
        .tracks
        .iter()
        .filter(|t| t.kind == "subtitles")
        .collect();
    on_progress(0.02, "decodificando pistas de audio");
    let (tracks_res, subs_map) = rayon::join(
        || media::load_audio_all_single_pass(path, n_audio, timeline::SR as u32),
        || media::extract_all_sub_events(path, &subs),
    );
    let tracks_16k = tracks_res?;
    let tracks_8k: Vec<Vec<f32>> = tracks_16k.par_iter().map(|x| dsp::decimate_2x(x)).collect();

    let (rep_res, tl_res) = rayon::join(
        || {
            sync_algo::analyze_file_with_tracks(
                path,
                &info,
                &tracks_8k,
                0,
                Some(subs_map),
                &|f, msg| {
                    on_progress(0.10 + f * 0.80, msg);
                },
            )
        },
        || timeline::ensure_from_samples(key, &tracks_16k, timeline_root),
    );
    tl_res?;
    rep_res
}

fn do_analyze_inner(
    state: &AppState,
    jid: &str,
    rel: &str,
    p_lo: f64,
    p_hi: f64,
) -> anyhow::Result<Report> {
    let p = fs::canonicalize(state.media_dir.join(rel))?;
    let k = file_key(&p)?;
    let tl_root = state.work_dir.join("timeline");
    let rep = analyze_and_cache(&p, &k, &tl_root, |f, m| {
        update_job(&state.jobs, jid, |j| {
            j.progress = p_lo + f * (p_hi - p_lo);
            j.msg = m.to_string();
        });
    })?;
    let rep_path = state.work_dir.join("reports").join(format!("{k}.json"));
    let tmp = rep_path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_vec(&rep)?)?;
    fs::rename(tmp, rep_path)?;
    Ok(rep)
}

fn do_analyze(state: &AppState, jid: &str, rel: &str) -> anyhow::Result<()> {
    let _ = do_analyze_inner(state, jid, rel, 0.0, 1.0)?;
    Ok(())
}

fn do_save_inner(
    state: &AppState,
    jid: &str,
    rel: &str,
    plan: Plan,
    p_lo: f64,
    p_hi: f64,
) -> anyhow::Result<()> {
    let p = fs::canonicalize(state.media_dir.join(rel))?;
    let pk = file_key(&p)?;
    let out = state.output_dir.join(rel);
    if out.is_file() {
        if let Ok(ok) = file_key(&out) {
            let _ = fs::remove_dir_all(state.work_dir.join("timeline").join(ok));
        }
    }
    let ref_track = report_for(&state.work_dir, &p)
        .map(|r| r.ref_track)
        .unwrap_or(0);
    let span = p_hi - p_lo;
    update_job(&state.jobs, jid, |j| {
        j.progress = p_lo + span * 0.05;
        j.msg = "escribiendo archivo corregido".to_string();
    });
    sync_algo::apply_with_ref(&p, &plan, &out, Some(&state.work_dir), ref_track)?;

    update_job(&state.jobs, jid, |j| {
        j.progress = p_lo + span * 0.45;
        j.msg = "verificando el resultado".to_string();
    });
    let out_canon = fs::canonicalize(&out)?;
    let out_key = file_key(&out_canon)?;
    let tl_root = state.work_dir.join("timeline");
    let verify = analyze_and_cache(&out_canon, &out_key, &tl_root, |f, m| {
        update_job(&state.jobs, jid, |j| {
            j.progress = p_lo + span * (0.45 + f * 0.50);
            j.msg = format!("verificando: {m}");
        });
    })?;

    let saved = SavedRecord {
        plan,
        verify,
        t: now_sec(),
    };
    let sv_path = state.work_dir.join("saved").join(format!("{pk}.json"));
    let tmp = sv_path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_vec(&saved)?)?;
    fs::rename(tmp, sv_path)?;
    Ok(())
}

fn do_save(state: &AppState, jid: &str, rel: &str, plan: Plan) -> anyhow::Result<()> {
    do_save_inner(state, jid, rel, plan, 0.0, 1.0)
}

fn do_process(state: &AppState, jid: &str, rel: &str, force: bool) -> anyhow::Result<()> {
    let p = fs::canonicalize(state.media_dir.join(rel))?;
    let rep = match report_for(&state.work_dir, &p) {
        Some(mut r) if !force => {
            if enrich_durations(&p, &mut r) {
                if let Ok(k) = file_key(&p) {
                    let rep_path = state.work_dir.join("reports").join(format!("{k}.json"));
                    let _ = fs::write(rep_path, serde_json::to_vec(&r)?);
                }
            }
            r
        }
        _ => do_analyze_inner(state, jid, rel, 0.0, 0.45)?,
    };
    let plan = sync_algo::propose(&rep);
    if sync_algo::plan_is_empty(&plan) {
        update_job(&state.jobs, jid, |j| {
            j.progress = 1.0;
            j.msg = "ya estaba sincronizado".to_string();
        });
        return Ok(());
    }
    let saved_ok = if !force && state.output_dir.join(rel).is_file() {
        if let Some(mut sv) = saved_for(&state.work_dir, &p) {
            let out_p = state.output_dir.join(rel);
            let _ = enrich_durations(&out_p, &mut sv.verify);
            let v_ref = sv
                .verify
                .audio
                .iter()
                .find(|a| a.n == sv.verify.ref_track)
                .and_then(|a| a.duration_s);
            let needs_trim = v_ref.is_some_and(|rd| {
                sv.verify
                    .audio
                    .iter()
                    .any(|a| a.n != sv.verify.ref_track && a.duration_s.is_some_and(|sd| sd > rd + 0.25))
            });
            let missing_proposed_sub = plan.subs.iter().any(|(k, v)| {
                sync_algo::sub_entry_is_active(v)
                    && !sv.plan.subs.get(k).is_some_and(sync_algo::sub_entry_is_active)
            });
            !needs_trim && !missing_proposed_sub
        } else {
            false
        }
    } else {
        false
    };
    if saved_ok {
        update_job(&state.jobs, jid, |j| {
            j.progress = 1.0;
            j.msg = "ya estaba guardado".to_string();
        });
        return Ok(());
    }
    do_save_inner(state, jid, rel, plan, 0.45, 1.0)
}

fn spawn_workers(state: AppState, n_workers: usize) {
    for _ in 0..n_workers.max(1) {
        let st = state.clone();
        std::thread::spawn(move || loop {
            let (jid, plan) = {
                let (lock, cvar) = &*st.queue;
                let mut q = lock.lock().unwrap();
                while q.is_empty() {
                    q = cvar.wait(q).unwrap();
                }
                q.pop_front().unwrap()
            };
            let (kind, rel) = {
                let mut jobs = st.jobs.lock().unwrap();
                let j = jobs.iter_mut().find(|j| j.id == jid).unwrap();
                j.state = "running".to_string();
                j.msg = "empezando".to_string();
                (j.kind.clone(), j.file.clone())
            };
            let res = match kind.as_str() {
                "analyze" => do_analyze(&st, &jid, &rel),
                "process" => do_process(&st, &jid, &rel, plan.is_some()),
                _ => do_save(&st, &jid, &rel, plan.unwrap_or_default()),
            };
            match res {
                Ok(()) => update_job(&st.jobs, &jid, |j| {
                    j.state = "done".to_string();
                    j.progress = 1.0;
                    if j.msg == "empezando" || j.msg == "en cola" || j.msg.starts_with("verificando") || j.msg.starts_with("preparando") {
                        j.msg = "listo".to_string();
                    }
                }),
                Err(e) => {
                    eprintln!("Job error ({kind} {rel}): {e:#}");
                    update_job(&st.jobs, &jid, |j| {
                        j.state = "error".to_string();
                        j.error = Some(e.to_string());
                        j.msg = "error".to_string();
                    });
                }
            }
        });
    }
}

// ---------------------------------------------------------------- Handlers

async fn api_state(State(st): State<AppState>) -> Result<Json<StateResponse>, ApiError> {
    let media_dir = st.media_dir.clone();
    let output_dir = st.output_dir.clone();
    let work_dir = st.work_dir.clone();

    let files = tokio::task::spawn_blocking(move || -> Vec<FileEntry> {
        list_media_files(&media_dir)
            .into_iter()
            .filter_map(|(rel, p)| {
                let size = fs::metadata(&p).ok()?.len();
                let rep = report_for(&work_dir, &p);
                let sv = saved_for(&work_dir, &p);
                let has_out = output_dir.join(&rel).is_file();
                let folder = Path::new(&rel)
                    .parent()
                    .map(|par| par.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let name = Path::new(&rel)
                    .file_name()
                    .map(|nm| nm.to_string_lossy().into_owned())
                    .unwrap_or_else(|| rel.clone());
                Some(FileEntry {
                    path: rel,
                    folder,
                    name,
                    size,
                    status: rep.map(|r| r.status),
                    saved: sv.is_some() && has_out,
                    saved_status: if has_out {
                        sv.map(|s| s.verify.status)
                    } else {
                        None
                    },
                })
            })
            .collect()
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let recent = {
        let mut jobs = st.jobs.lock().unwrap().clone();
        jobs.sort_by(|a, b| a.t.total_cmp(&b.t));
        if jobs.len() > 200 {
            jobs.split_off(jobs.len() - 200)
        } else {
            jobs
        }
    };

    Ok(Json(StateResponse {
        files,
        jobs: recent,
    }))
}

async fn api_analyze(
    State(st): State<AppState>,
    Json(req): Json<AnalyzeReq>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mut ids = Vec::new();
    for rel in req.files {
        let p = resolve_file(&st.media_dir, &rel)?;
        if req.force || report_for(&st.work_dir, &p).is_none() {
            ids.push(enqueue_job(&st, "analyze", &rel, None));
        }
    }
    Ok(Json(json!({ "jobs": ids })))
}

async fn api_process(
    State(st): State<AppState>,
    Json(req): Json<ProcessReq>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let targets: Vec<String> = if let Some(folder) = req.folder {
        let folder_trim = folder.trim_matches('/').to_string();
        list_media_files(&st.media_dir)
            .into_iter()
            .filter(|(rel, _)| {
                let f = Path::new(rel)
                    .parent()
                    .map(|par| par.to_string_lossy().into_owned())
                    .unwrap_or_default();
                if folder_trim.is_empty() {
                    f.is_empty()
                } else {
                    f == folder_trim || f.starts_with(&format!("{folder_trim}/"))
                }
            })
            .map(|(rel, _)| rel)
            .collect()
    } else if req.files.is_empty() {
        list_media_files(&st.media_dir)
            .into_iter()
            .map(|(rel, _)| rel)
            .collect()
    } else {
        req.files
    };

    let mut ids = Vec::new();
    let force_marker = if req.force { Some(Plan::default()) } else { None };
    for rel in targets {
        resolve_file(&st.media_dir, &rel)?;
        ids.push(enqueue_job(&st, "process", &rel, force_marker.clone()));
    }
    Ok(Json(json!({ "jobs": ids })))
}

#[derive(Deserialize)]
struct ReportQuery {
    file: String,
}

async fn api_report(
    State(st): State<AppState>,
    Query(q): Query<ReportQuery>,
) -> Result<Json<ReportResponse>, ApiError> {
    let p = resolve_file(&st.media_dir, &q.file)?;
    let mut rep = report_for(&st.work_dir, &p)
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "sin analizar".into()))?;
    let out_p = st.output_dir.join(&q.file);
    let mut sv = if out_p.is_file() {
        saved_for(&st.work_dir, &p)
    } else {
        None
    };

    if rep.video_fps.is_none()
        || rep.audio.iter().any(|a| a.duration_s.is_none())
        || sv
            .as_ref()
            .is_some_and(|s| s.verify.audio.iter().any(|a| a.duration_s.is_none()))
    {
        let work_dir = st.work_dir.clone();
        let p_c = p.clone();
        let out_c = out_p.clone();
        let (rep_up, sv_up) = tokio::task::spawn_blocking(move || {
            if enrich_durations(&p_c, &mut rep) {
                if let Ok(k) = file_key(&p_c) {
                    let rp = work_dir.join("reports").join(format!("{k}.json"));
                    if let Ok(bytes) = serde_json::to_vec(&rep) {
                        let _ = fs::write(rp, bytes);
                    }
                }
            }
            if let Some(mut s) = sv {
                if out_c.is_file() && enrich_durations(&out_c, &mut s.verify) {
                    if let Ok(k) = file_key(&p_c) {
                        let sp = work_dir.join("saved").join(format!("{k}.json"));
                        if let Ok(bytes) = serde_json::to_vec(&s) {
                            let _ = fs::write(sp, bytes);
                        }
                    }
                }
                sv = Some(s);
            }
            (rep, sv)
        })
        .await
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        rep = rep_up;
        sv = sv_up;
    }

    let proposed = sync_algo::propose(&rep);
    Ok(Json(ReportResponse {
        report: rep,
        proposed,
        saved: sv,
    }))
}

async fn api_save(
    State(st): State<AppState>,
    Json(req): Json<SaveReq>,
) -> Result<Json<serde_json::Value>, ApiError> {
    resolve_file(&st.media_dir, &req.file)?;
    if sync_algo::plan_is_empty(&req.plan) {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "el plan no cambia nada".into(),
        ));
    }
    let jid = enqueue_job(&st, "save", &req.file, Some(req.plan));
    Ok(Json(json!({ "job": jid })))
}

fn parse_plan(plan_str: &str) -> Result<Plan, ApiError> {
    if plan_str.is_empty() {
        return Ok(Plan::default());
    }
    serde_json::from_str(plan_str)
        .map_err(|_| ApiError(StatusCode::BAD_REQUEST, "plan inválido".into()))
}

#[derive(Deserialize)]
struct SpectroQuery {
    file: String,
    #[serde(default)]
    start: f64,
    #[serde(default = "default_spec_dur")]
    dur: f64,
    #[serde(default = "default_src")]
    src: String,
    #[serde(default)]
    plan: String,
}

fn default_spec_dur() -> f64 {
    4.0
}
fn default_src() -> String {
    "media".to_string()
}

async fn api_spectrogram(
    State(st): State<AppState>,
    Query(q): Query<SpectroQuery>,
) -> Result<Response, ApiError> {
    let base = if q.src == "output" {
        &st.output_dir
    } else {
        &st.media_dir
    };
    let p = resolve_file(base, &q.file)?;
    let dur = q.dur.clamp(1.0, 120.0);
    let pl = parse_plan(&q.plan)?;
    let k = file_key(&p).map_err(anyhow::Error::from)?;
    let pl_json = serde_json::to_string(&pl).map_err(anyhow::Error::from)?;

    let mut hasher = Sha1::new();
    hasher.update(format!("{k}|{}|{dur}|{pl_json}", q.start).as_bytes());
    let img_hash: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    let img = st.work_dir.join("img").join(format!("{img_hash}.png"));

    if !img.is_file() {
        let st_clone = st.clone();
        let img_clone = img.clone();
        let p_clone = p.clone();
        let q_file = q.file.clone();
        let q_src = q.src.clone();
        let q_start = q.start;
        let has_plan = !q.plan.is_empty();

        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let mid = q_start + dur / 2.0;
            let rep = if q_src == "output" {
                resolve_file(&st_clone.media_dir, &q_file)
                    .ok()
                    .and_then(|mp| saved_for(&st_clone.work_dir, &mp))
                    .map(|sv| sv.verify)
            } else {
                report_for(&st_clone.work_dir, &p_clone)
            };

            let mut shifts = HashMap::new();
            for (n_str, entry) in &pl.audio {
                if let Ok(n) = n_str.parse::<usize>() {
                    shifts.insert(n, sync_algo::shift_at(Some(entry), mid));
                }
            }

            let mut subs = Vec::new();
            if let Some(r) = rep {
                for s in r.subs {
                    if !s.events.is_empty() {
                        let sh = pl
                            .subs
                            .get(&s.n.to_string())
                            .map(|e| sync_algo::sub_source_shift_at(e, mid))
                            .unwrap_or(0.0);
                        subs.push((format!("s:{} {} {}", s.n, s.lang, s.name), s.events, sh));
                    }
                }
            }

            let label = if q_src == "output" {
                "corregido (archivo guardado)"
            } else if has_plan {
                "simulación de la corrección"
            } else {
                "original"
            };
            let fname = p_clone
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let title = format!(
                "{fname}  —  {label}  —  {:.2}s a {:.2}s",
                q_start,
                q_start + dur
            );
            spectro::render(
                &p_clone,
                &img_clone,
                Some(q_start),
                dur,
                false,
                &shifts,
                &subs,
                Some(&title),
            )?;

            // Prune keeping newest 300 PNGs
            let mut pngs: Vec<(std::time::SystemTime, PathBuf)> =
                fs::read_dir(st_clone.work_dir.join("img"))?
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
                    .filter_map(|p| {
                        let mt = fs::metadata(&p).ok()?.modified().ok()?;
                        Some((mt, p))
                    })
                    .collect();
            if pngs.len() > 300 {
                pngs.sort_by_key(|x| x.0);
                let del_cnt = pngs.len() - 300;
                for (_, old) in pngs.into_iter().take(del_cnt) {
                    let _ = fs::remove_file(old);
                }
            }
            Ok(())
        })
        .await
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??;
    }

    let bytes = fs::read(&img).map_err(anyhow::Error::from)?;
    Ok((
        [
            (header::CONTENT_TYPE, HeaderValue::from_static("image/png")),
            (header::CACHE_CONTROL, HeaderValue::from_static("no-store")),
        ],
        bytes,
    )
        .into_response())
}

#[derive(Deserialize)]
struct ClipQuery {
    file: String,
    track: usize,
    #[serde(default)]
    start: f64,
    #[serde(default = "default_clip_dur")]
    dur: f64,
    #[serde(default = "default_src")]
    src: String,
    #[serde(default)]
    plan: String,
}

fn default_clip_dur() -> f64 {
    8.0
}

fn encode_stereo_wav(left: &[f32], right: &[f32], sr: u32) -> Vec<u8> {
    let n = left.len().min(right.len());
    let data_bytes = (n * 4) as u32;
    let riff_size = 36 + data_bytes;
    let mut out = Vec::with_capacity(44 + data_bytes as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&2u16.to_le_bytes()); // stereo
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&(sr * 4).to_le_bytes()); // byte rate
    out.extend_from_slice(&4u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for i in 0..n {
        let l = (left[i].clamp(-1.0, 1.0) * 32767.0) as i16;
        let r = (right[i].clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&l.to_le_bytes());
        out.extend_from_slice(&r.to_le_bytes());
    }
    out
}

async fn api_clip(
    State(st): State<AppState>,
    Query(q): Query<ClipQuery>,
) -> Result<Response, ApiError> {
    let base = if q.src == "output" {
        &st.output_dir
    } else {
        &st.media_dir
    };
    let p = resolve_file(base, &q.file)?;
    let dur = q.dur.clamp(1.0, 30.0);
    let sr = 22050u32;
    let pl = parse_plan(&q.plan)?;
    let media_p = resolve_file(&st.media_dir, &q.file).ok();
    let ref_track = media_p
        .as_deref()
        .and_then(|mp| report_for(&st.work_dir, mp))
        .map(|r| r.ref_track)
        .unwrap_or(0);
    let audio_entry = pl.audio.get(&q.track.to_string()).cloned();
    let shift = sync_algo::shift_at(audio_entry.as_ref(), q.start + dur / 2.0);
    let track = q.track;
    let start = q.start;

    let wav = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<u8>> {
        let (l_res, r_res) = rayon::join(
            || media::load_audio_slice(&p, ref_track, Some(start), Some(dur), sr),
            || {
                if let Some(ref e) = audio_entry {
                    if e.mode == "tempo" {
                        let tp = e.tempo.unwrap_or(1.0);
                        let d_s = (e.delay_ms.unwrap_or(0) as f64) / 1000.0;
                        let pc = e.pitch_correct.unwrap_or(false);
                        return media::load_audio_slice_tempo(&p, track, start, dur, sr, tp, d_s, pc);
                    }
                }
                media::load_audio_slice(&p, track, Some(start - shift), Some(dur), sr)
            },
        );
        Ok(encode_stereo_wav(&l_res?, &r_res?, sr))
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??;

    Ok((
        [
            (header::CONTENT_TYPE, HeaderValue::from_static("audio/wav")),
            (header::CACHE_CONTROL, HeaderValue::from_static("no-store")),
        ],
        wav,
    )
        .into_response())
}

fn timeline_dir_sync(
    st: &AppState,
    file: &str,
    src: &str,
) -> Result<(PathBuf, PathBuf, usize, dubalign::model::ProbeInfo), ApiError> {
    let base = if src == "output" {
        &st.output_dir
    } else {
        &st.media_dir
    };
    let p = resolve_file(base, file)?;
    let info = media::probe(&p).map_err(anyhow::Error::from)?;
    let n = info.tracks.iter().filter(|t| t.kind == "audio").count();
    let k = file_key(&p).map_err(anyhow::Error::from)?;
    let d = timeline::ensure(&p, &k, n, &st.work_dir.join("timeline"))
        .map_err(anyhow::Error::from)?;
    Ok((p, d, n, info))
}

#[derive(Deserialize)]
struct TimelineMetaQuery {
    file: String,
    #[serde(default = "default_src")]
    src: String,
}

async fn api_timeline_meta(
    State(st): State<AppState>,
    Query(q): Query<TimelineMetaQuery>,
) -> Result<Json<TimelineMeta>, ApiError> {
    let st_c = st.clone();
    let meta = tokio::task::spawn_blocking(move || -> Result<TimelineMeta, ApiError> {
        let (p, d, n, info) = timeline_dir_sync(&st_c, &q.file, &q.src)?;
        let k = file_key(&p).map_err(anyhow::Error::from)?;
        let names = timeline::track_names(&info.tracks);
        let env = timeline::envelopes(&d, n).map_err(anyhow::Error::from)?;
        Ok(TimelineMeta {
            key: k,
            duration: info.duration,
            tracks: names,
            spec_fps: timeline::SPEC_FPS,
            levels: timeline::LEVELS.to_vec(),
            env,
        })
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??;
    Ok(Json(meta))
}

#[derive(Deserialize)]
struct TimelineTileQuery {
    file: String,
    track: usize,
    level: usize,
    index: isize,
    #[serde(default = "default_src")]
    src: String,
}

async fn api_timeline_tile(
    State(st): State<AppState>,
    Query(q): Query<TimelineTileQuery>,
) -> Result<Response, ApiError> {
    if q.level >= timeline::LEVELS.len() || q.index < 0 {
        return Err(ApiError(StatusCode::BAD_REQUEST, "tesela inválida".into()));
    }
    let st_c = st.clone();
    let png_bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, ApiError> {
        let (_, d, n, _) = timeline_dir_sync(&st_c, &q.file, &q.src)?;
        if q.track >= n {
            return Err(ApiError(StatusCode::NOT_FOUND, "pista inexistente".into()));
        }
        timeline::tile_png(&d, q.track, q.level, q.index as usize)
            .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??;

    Ok((
        [
            (header::CONTENT_TYPE, HeaderValue::from_static("image/png")),
            (
                header::CACHE_CONTROL,
                HeaderValue::from_static("private, max-age=86400"),
            ),
        ],
        png_bytes,
    )
        .into_response())
}

#[derive(Deserialize)]
struct TimelineAudioQuery {
    file: String,
    track: usize,
    #[serde(default = "default_audio_mode")]
    mode: String,
    #[serde(default = "default_src")]
    src: String,
}

fn default_audio_mode() -> String {
    "split".to_string()
}

async fn api_timeline_audio(
    State(st): State<AppState>,
    Query(q): Query<TimelineAudioQuery>,
    req: Request<Body>,
) -> Result<Response, ApiError> {
    if q.mode != "split" && q.mode != "solo" {
        return Err(ApiError(StatusCode::BAD_REQUEST, "modo inválido".into()));
    }
    let base = if q.src == "output" {
        &st.output_dir
    } else {
        &st.media_dir
    };
    let p = resolve_file(base, &q.file)?;
    let ref_track = resolve_file(&st.media_dir, &q.file)
        .ok()
        .and_then(|mp| report_for(&st.work_dir, &mp))
        .map(|r| r.ref_track)
        .unwrap_or(0);
    let k = file_key(&p).map_err(anyhow::Error::from)?;
    let tl_root = st.work_dir.join("timeline");
    let track = q.track;
    let mode = q.mode.clone();

    let audio_path = tokio::task::spawn_blocking(move || {
        timeline::audio_file(&p, &k, &tl_root, ref_track, track, &mode)
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(anyhow::Error::from)?;

    let Ok(resp) = ServeFile::new(&audio_path).oneshot(req).await;
    let mut resp = resp.into_response();
    resp.headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/mp4"));
    Ok(resp)
}

async fn index(State(st): State<AppState>) -> Result<Response, ApiError> {
    let bytes = fs::read(st.static_dir.join("index.html"))
        .map_err(|_| ApiError(StatusCode::NOT_FOUND, "index.html no encontrado".into()))?;
    Ok((
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/html; charset=utf-8"),
            ),
            (header::CACHE_CONTROL, HeaderValue::from_static("no-store")),
        ],
        bytes,
    )
        .into_response())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let media_raw = std::env::var("MEDIA_DIR").unwrap_or_else(|_| "/media".into());
    let output_raw = std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into());
    let work_raw = std::env::var("WORK_DIR").unwrap_or_else(|_| "/work".into());
    let static_raw = std::env::var("STATIC_DIR").unwrap_or_else(|_| "static".into());
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7341);
    let workers: usize = std::env::var("DUBALIGN_WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    fs::create_dir_all(&output_raw)?;
    for sub in ["reports", "saved", "img", "timeline"] {
        fs::create_dir_all(Path::new(&work_raw).join(sub))?;
    }

    let media_dir = fs::canonicalize(&media_raw).unwrap_or_else(|_| PathBuf::from(&media_raw));
    let output_dir = fs::canonicalize(&output_raw).unwrap_or_else(|_| PathBuf::from(&output_raw));
    let work_dir = fs::canonicalize(&work_raw).unwrap_or_else(|_| PathBuf::from(&work_raw));
    let static_dir = PathBuf::from(&static_raw);

    let state = AppState {
        media_dir,
        output_dir,
        work_dir,
        static_dir,
        jobs: Arc::new(Mutex::new(Vec::new())),
        queue: Arc::new((Mutex::new(VecDeque::new()), Condvar::new())),
    };
    spawn_workers(state.clone(), workers);

    let app = Router::new()
        .route("/", get(index))
        .route("/api/state", get(api_state))
        .route("/api/analyze", post(api_analyze))
        .route("/api/process", post(api_process))
        .route("/api/report", get(api_report))
        .route("/api/save", post(api_save))
        .route("/api/spectrogram", get(api_spectrogram))
        .route("/api/clip", get(api_clip))
        .route("/api/timeline/meta", get(api_timeline_meta))
        .route("/api/timeline/tile", get(api_timeline_tile))
        .route("/api/timeline/audio", get(api_timeline_audio))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    println!("dubalign web listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
