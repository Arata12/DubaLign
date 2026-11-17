"""Interfaz web: lista los originales de MEDIA_DIR, los analiza, enseña
espectrogramas antes/después, deja ajustar la corrección y guarda en OUTPUT_DIR.

Se arranca con `docker compose up` (ver docker-compose.yml).
"""
import hashlib
import io
import json
import os
import queue
import shutil
import threading
import time
import traceback
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

import engine
import spectro
import timeline

MEDIA = Path(os.environ.get("MEDIA_DIR", "/media")).resolve()
OUTPUT = Path(os.environ.get("OUTPUT_DIR", "/output")).resolve()
WORK = Path(os.environ.get("WORK_DIR", "/work")).resolve()
VIDEO_EXT = {".mkv", ".mp4", ".m4v", ".avi", ".ts", ".m2ts", ".webm"}
for d in ("reports", "saved", "img", "timeline"):
    (WORK / d).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="dubalign")
jobs, jobs_lock, todo = {}, threading.Lock(), queue.Queue()


# ---------------------------------------------------------------- utilidades

def resolve(rel, src="media"):
    base = OUTPUT if src == "output" else MEDIA
    p = (base / rel).resolve()
    if not p.is_relative_to(base) or not p.is_file():
        raise HTTPException(404, "archivo no encontrado")
    return p


def key(p):
    st = p.stat()
    return hashlib.sha1(f"{p}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:20]


def read_json(p):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def report_for(p):
    return read_json(WORK / "reports" / f"{key(p)}.json")


def saved_for(p):
    return read_json(WORK / "saved" / f"{key(p)}.json")


# ---------------------------------------------------------------- trabajos

def enqueue(kind, rel, **data):
    with jobs_lock:
        for j in jobs.values():  # no duplicar lo que ya está en cola
            if j["kind"] == kind and j["file"] == rel and j["state"] in ("queued", "running"):
                return j["id"]
        jid = f"{time.time_ns():x}"
        jobs[jid] = {"id": jid, "kind": kind, "file": rel, "state": "queued", "progress": 0.0, "msg": "en cola",
                     "error": None, "t": time.time()}
    todo.put((jid, data))
    return jid


def do_analyze(job, data):
    p = resolve(job["file"])
    rep = engine.analyze_file(p, progress=lambda f, m: job.update(progress=f * 0.85, msg=m))
    (WORK / "reports" / f"{key(p)}.json").write_text(json.dumps(rep))
    job.update(progress=0.9, msg="preparando el visor del episodio")
    timeline.ensure(p, key(p), len(rep["audio"]), WORK / "timeline")


def do_save(job, data):
    p = resolve(job["file"])
    out = OUTPUT / job["file"]
    if out.is_file():  # datos del visor de la versión anterior
        shutil.rmtree(WORK / "timeline" / key(out), ignore_errors=True)
    job.update(progress=0.05, msg="escribiendo archivo corregido")
    engine.apply(p, data["plan"], out, WORK)
    job.update(progress=0.45, msg="verificando el resultado")
    verify = engine.analyze_file(out, progress=lambda f, m: job.update(progress=0.45 + f * 0.45, msg=f"verificando: {m}"))
    job.update(progress=0.92, msg="preparando el visor del episodio")
    timeline.ensure(out, key(out), len(verify["audio"]), WORK / "timeline")
    (WORK / "saved" / f"{key(p)}.json").write_text(json.dumps({"plan": data["plan"], "verify": verify, "t": time.time()}))


def worker():
    while True:
        jid, data = todo.get()
        job = jobs[jid]
        job.update(state="running", msg="empezando")
        try:
            (do_analyze if job["kind"] == "analyze" else do_save)(job, data)
            job.update(state="done", progress=1.0, msg="listo")
        except BaseException as e:  # sync.load_audio usa sys.exit
            traceback.print_exc()
            job.update(state="error", error=str(e) or type(e).__name__, msg="error")


threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------- API

class AnalyzeReq(BaseModel):
    files: list[str]
    force: bool = False


class SaveReq(BaseModel):
    file: str
    plan: dict


@app.get("/api/state")
def state():
    files = []
    for p in sorted(MEDIA.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXT or not p.is_file():
            continue
        rel = str(p.relative_to(MEDIA))
        rep, sv = report_for(p), saved_for(p)
        has_out = (OUTPUT / rel).is_file()
        files.append({"path": rel, "size": p.stat().st_size, "status": rep["status"] if rep else None,
                      "saved": bool(sv and has_out),
                      "saved_status": sv["verify"]["status"] if sv and has_out else None})
    with jobs_lock:
        recent = sorted(jobs.values(), key=lambda j: j["t"])[-200:]
    return {"files": files, "jobs": recent}


@app.post("/api/analyze")
def analyze(req: AnalyzeReq):
    ids = []
    for rel in req.files:
        p = resolve(rel)
        if req.force or not report_for(p):
            ids.append(enqueue("analyze", rel))
    return {"jobs": ids}


@app.get("/api/report")
def report(file: str):
    p = resolve(file)
    rep = report_for(p)
    if not rep:
        raise HTTPException(404, "sin analizar")
    return {"report": rep, "proposed": engine.propose(rep), "saved": saved_for(p) if (OUTPUT / file).is_file() else None}


@app.post("/api/save")
def save(req: SaveReq):
    resolve(req.file)
    if engine.plan_is_empty(req.plan):
        raise HTTPException(400, "el plan no cambia nada")
    return {"job": enqueue("save", req.file, plan=req.plan)}


def _plan(plan):
    try:
        return json.loads(plan) if plan else {"audio": {}, "subs": {}}
    except ValueError:
        raise HTTPException(400, "plan inválido")


@app.get("/api/spectrogram")
def spectrogram(file: str, start: float = 0.0, dur: float = 4.0, src: str = "media", plan: str = ""):
    """PNG de todas las pistas. Con `plan` simula la corrección sin escribir el archivo."""
    p = resolve(file, src)
    dur = min(max(dur, 1.0), 120.0)
    pl = _plan(plan)
    img = WORK / "img" / (hashlib.sha1(f"{key(p)}|{start}|{dur}|{json.dumps(pl, sort_keys=True)}".encode()).hexdigest() + ".png")
    if not img.is_file():
        mid = start + dur / 2
        rep = (saved_for(resolve(file)) or {}).get("verify") if src == "output" else report_for(p)
        shifts = {int(n): engine.shift_at(e, mid) for n, e in pl.get("audio", {}).items()}
        subs = [(f"s:{s['n']} {s['lang']} {s['name']}", s["events"],
                 pl.get("subs", {}).get(str(s["n"]), {}).get("delay_ms", 0) / 1000)
                for s in (rep or {}).get("subs", []) if s["events"]]
        label = "corregido (archivo guardado)" if src == "output" else "simulación de la corrección" if plan else "original"
        try:
            spectro.render(p, img, start, dur, shifts=shifts, subs=subs,
                           title=f"{p.name}  —  {label}  —  {start:.2f}s a {start + dur:.2f}s")
        except RuntimeError as e:
            raise HTTPException(500, str(e))
        for old in sorted((WORK / "img").glob("*.png"), key=lambda f: f.stat().st_mtime)[:-300]:
            old.unlink(missing_ok=True)
    return FileResponse(img, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/clip")
def clip(file: str, track: int, start: float = 0.0, dur: float = 8.0, src: str = "media", plan: str = ""):
    """WAV estéreo: referencia en el canal izquierdo, la pista elegida en el derecho."""
    p = resolve(file, src)
    dur, sr = min(max(dur, 1.0), 30.0), 22050
    rep = report_for(resolve(file)) or {}
    shift = engine.shift_at(_plan(plan).get("audio", {}).get(str(track)), start + dur / 2)
    try:
        left = spectro.load(p, rep.get("ref_track", 0), start, dur, sr)
        right = spectro.load(p, track, start - shift, dur, sr)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    pcm = (np.clip(np.stack([left, right], axis=1), -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return Response(buf.getvalue(), media_type="audio/wav", headers={"Cache-Control": "no-store"})


def _timeline_dir(file, src):
    p = resolve(file, src)
    n = len([t for t in engine.probe(p)["tracks"] if t["type"] == "audio"])
    return p, timeline.ensure(p, key(p), n, WORK / "timeline"), n


@app.get("/api/timeline/meta")
def timeline_meta(file: str, src: str = "media"):
    """Datos del visor de episodio completo (los calcula la primera vez, ~10 s)."""
    p, d, n = _timeline_dir(file, src)
    info = engine.probe(p)
    names = [f"a:{t['n']} {t['lang']} {t['name']}".strip() for t in info["tracks"] if t["type"] == "audio"]
    return {"key": key(p), "duration": info["duration"], "tracks": names, "spec_fps": timeline.SPEC_FPS,
            "levels": timeline.LEVELS, "env": timeline.envelopes(d, n)}


@app.get("/api/timeline/tile")
def timeline_tile(file: str, track: int, level: int, index: int, src: str = "media"):
    if not 0 <= level < len(timeline.LEVELS) or index < 0:
        raise HTTPException(400, "tesela inválida")
    _, d, n = _timeline_dir(file, src)
    if not 0 <= track < n:
        raise HTTPException(404, "pista inexistente")
    return Response(timeline.tile_png(d, track, level, index), media_type="image/png",
                    headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/timeline/audio")
def timeline_audio(file: str, track: int, mode: str = "split", src: str = "media"):
    """Audio del episodio entero: mode=split (izq. referencia, der. la pista) o solo."""
    if mode not in ("split", "solo"):
        raise HTTPException(400, "modo inválido")
    p = resolve(file, src)
    rep = report_for(resolve(file)) or {}
    try:
        f = timeline.audio_file(p, key(p), WORK / "timeline", rep.get("ref_track", 0), track, mode)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return FileResponse(f, media_type="audio/mp4")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html", headers={"Cache-Control": "no-store"})
