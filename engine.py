"""Motor común (CLI y web): analiza un archivo con varios audios/subtítulos contra
la pista de referencia, propone una corrección y la aplica.

Informe (analyze_file):
  audio[n]: status ref|ok|offset|cuts|tempo|unknown, delay_ms, tempo, points, segments
  subs[n]:  status ok|offset|unknown, offset_ms, conf, events
Plan (propose / apply):
  {"audio": {"1": {"mode": "delay", "delay_ms": 1242},
             "3": {"mode": "segments", "segments": [...], "codec": "aac"}},
   "subs":  {"0": {"delay_ms": -500}}}
"""
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import butter, correlate, sosfilt, stft

import sync

SR = sync.SR
TOL = 0.040          # s: dos medidas "coinciden" si difieren menos que esto
THRESHOLD_MS = 40    # desfase de audio por debajo del cual no se toca
EDGE_MIN = 0.150     # s: salto mínimo de un tramo al principio/final para recodificar la pista
SUB_THRESHOLD_MS = 400  # los subtítulos suelen ir ~150 ms por delante de la voz; eso es normal
SUB_MIN_CONF = 0.35     # correlación mínima (real: 0.5-0.6; emparejamiento falso: <0.2)
TEXT_SUBS = ("SubStationAlpha", "SubRip/SRT", "WebVTT", "Timed Text")


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], **kw)


# ---------------------------------------------------------------- probe

def probe(path):
    j = json.loads(run(["mkvmerge", "-J", path], capture_output=True, text=True).stdout)
    count, tracks = {}, []
    for t in j.get("tracks", []):
        p = t["properties"]
        n = count.get(t["type"], 0)
        count[t["type"]] = n + 1
        tracks.append({"id": t["id"], "type": t["type"], "n": n, "codec": t["codec"],
                       "lang": p.get("language", "und"), "name": p.get("track_name", ""),
                       "default": bool(p.get("default_track")), "channels": p.get("audio_channels"),
                       "sample_rate": p.get("audio_sampling_frequency")})
    dur = j.get("container", {}).get("properties", {}).get("duration")
    return {"duration": dur / 1e9 if dur else None, "tracks": tracks}


# ---------------------------------------------------------------- audio

def global_offset(ref, src):
    c = np.abs(correlate(ref, src, mode="full", method="fft"))
    return int(np.argmax(c)) - (len(src) - 1)


def scan(ref, src, g, win, step, search, min_conf, t0=0.0, t1=None):
    """[(t_src, delay, conf)] con ventanas de `win` s cada `step` s. delay = s a sumar a src."""
    L, S, out = int(win * SR), int(search * SR), []
    stop = max(len(src) - L, 0) if t1 is None else min(int(t1 * SR), max(len(src) - L, 0))
    for t in range(int(t0 * SR), stop, int(step * SR)):
        seg = src[t:t + L]
        lo, hi = max(t + g - S, 0), min(t + g + L + S, len(ref))
        if seg.std() < 1e-3 or hi - lo < L + SR:
            continue
        k, conf = sync.peak(ref[lo:hi], seg)
        if conf >= min_conf:
            out.append((t / SR, (lo + k - t) / SR, conf))
    return out


def _ncc(ref, src, t, d, win=2.0, slack=0.02):
    """Parecido (0..1) entre src[t:t+win] y ref alrededor de t+d."""
    a, L, s = int(t * SR), int(win * SR), int(slack * SR)
    w = src[a:a + L]
    lo = a + int(d * SR) - s
    r = ref[max(lo, 0):lo + L + 2 * s]
    if len(w) < L or len(r) < L or w.std() < 1e-4:
        return 0.0
    c = np.abs(correlate(r, w, mode="valid", method="fft")).max()
    return float(c / (np.linalg.norm(w) * np.linalg.norm(r) + 1e-9))


def _quiet_point(src, lo, hi, length):
    """Inicio (s) del tramo más silencioso de `length` s dentro de [lo, hi]."""
    a, b, n = int(lo * SR), int(hi * SR), max(int(length * SR), SR // 20)
    x = src[a:b].astype(np.float64) ** 2
    if len(x) <= n:
        return lo
    c = np.cumsum(np.concatenate([[0.0], x]))
    return lo + int(np.argmin(c[n:] - c[:-n])) / SR


def _cut_point(ref, src, dA, dB, t_lo, t_hi):
    """Instante (s de src) entre t_lo y t_hi donde el delay pasa de dA a dB, en el punto más silencioso."""
    ts = np.arange(max(t_lo, 0.0), t_hi, 0.5)
    isB = np.array([_ncc(ref, src, t, dB) > _ncc(ref, src, t, dA) for t in ts])
    # punto de cambio que mejor separa "A antes / B después"
    score = [np.sum(~isB[:j]) + np.sum(isB[j:]) for j in range(len(ts) + 1)]
    j = int(np.argmax(score))
    tc = ts[min(j, len(ts) - 1)] + 1.0  # las ventanas miden 2 s: el cambio cae hacia su centro
    gap = abs(dB - dA)
    length = gap if dB < dA else 0.05
    x = _quiet_point(src, tc - 1.0, tc + 1.0, length)
    # la estimación puede desviarse en fundidos o música (encajan igual con los dos delays):
    # si cerca hay silencio digital, el corte casi seguro está ahí
    x3 = _quiet_point(src, tc - 3.0, tc + 3.0, length)
    if _level_db(src, x3, length) < -90 <= _level_db(src, x, length):
        x = x3
    return x + (gap if dB < dA else 0.025)


def _level_db(src, t, length):
    a = int(t * SR)
    w = src[a:a + max(int(length * SR), 1)]
    return 20 * np.log10(np.sqrt((w.astype(np.float64) ** 2).mean()) + 1e-9) if len(w) else 0.0


def edge_jump(ref, src, d_main, side, span=60.0):
    """Busca en el primer/último minuto un tramo corto con otro delay (intro, avance del
    siguiente episodio...) que las ventanas largas no ven. -> (corte_s, delay) o None."""
    n = len(src) / SR
    t0, t1 = (max(n - span, 0.0), None) if side == "tail" else (0.0, min(span, n))
    pts = scan(ref, src, int(d_main * SR), 4, 2, 15, 8, t0, t1)
    other = [q for q in pts if abs(q[1] - d_main) >= TOL]
    if len(other) < 3:
        return None
    best = max(other, key=lambda q: sum(abs(o[1] - q[1]) < 0.02 for o in other))
    cluster = [o for o in other if abs(o[1] - best[1]) < 0.02]
    if len(cluster) < 3:
        return None
    d_edge = float(np.median([o[1] for o in cluster]))
    if abs(d_edge - d_main) < EDGE_MIN:  # no compensa recodificar por tan poco
        return None
    main = [q[0] for q in pts if abs(q[1] - d_main) < TOL]
    if side == "tail":
        tc = min(o[0] for o in cluster)
        if main and max(main) > tc:  # el tramo debe ser continuo hasta el final
            return None
        cut = _cut_point(ref, src, d_main, d_edge, tc - 14, tc + 4)
        a, b = cut, n
    else:
        tc = max(o[0] for o in cluster)
        if main and min(main) < tc:  # el tramo debe ser continuo desde el principio
            return None
        cut = _cut_point(ref, src, d_edge, d_main, tc - 4, tc + 14)
        a, b = 0.0, cut
    # el tramo tiene que encajar claramente mejor con el delay nuevo que con el principal
    ts = np.arange(a, b - 2, 1.0)
    if len(ts) < 3:
        return None
    e = np.array([_ncc(ref, src, t, d_edge) for t in ts])
    m = np.array([_ncc(ref, src, t, d_main) for t in ts])
    if (e > m + 0.05).mean() < 0.3 or np.median(e) < 1.5 * np.median(m):
        return None
    return cut, d_edge


def add_edges(ref, src, segments):
    """Añade a los tramos los saltos cortos del principio y del final, si los hay."""
    segs = [dict(x) for x in segments]
    head = edge_jump(ref, src, segs[0]["delay"], "head")
    if head and head[0] < (segs[0]["end"] or 1e9):
        segs[0]["start"] = round(float(head[0]), 3)
        segs.insert(0, {"start": 0.0, "end": segs[0]["start"], "delay": round(head[1], 3)})
    tail = edge_jump(ref, src, segs[-1]["delay"], "tail")
    if tail and tail[0] > segs[-1]["start"]:
        segs[-1]["end"] = round(float(tail[0]), 3)
        segs.append({"start": segs[-1]["end"], "end": None, "delay": round(tail[1], 3)})
    return segs


def find_segments(ref, src, g):
    """Tramos con delay distinto: [{"start", "end", "delay"}] en tiempo de src. [] si no hay cortes claros."""
    pts = scan(ref, src, g, 10, 10, 15, 10)
    groups, i = [], 0
    while i < len(pts):
        p = pts[i]
        if groups and abs(p[1] - np.median([q[1] for q in groups[-1]])) < TOL:
            groups[-1].append(p)
        elif i + 1 < len(pts) and abs(pts[i + 1][1] - p[1]) < TOL:
            groups.append([p])  # delay nuevo, confirmado por la ventana siguiente
        i += 1
    if len(groups) < 2:
        return []
    delays = [float(np.median([q[1] for q in grp])) for grp in groups]
    cuts = [_cut_point(ref, src, delays[k], delays[k + 1], groups[k][-1][0], groups[k + 1][0][0] + 8.0)
            for k in range(len(groups) - 1)]
    bounds = [0.0] + cuts + [None]
    return [{"start": round(float(bounds[k]), 3),
             "end": None if bounds[k + 1] is None else round(float(bounds[k + 1]), 3),
             "delay": round(delays[k], 3)} for k in range(len(groups))]


def dense_map(ref, src, g, win=4.0, step=2.0, search=3.0, min_conf=8.0):
    """Desfase medido cada `step` s en todo el episodio: [[t, ms, conf]].
    Los huecos son tramos sin nada común que medir (solo diálogo, silencio)."""
    return [[round(t, 1), round(d * 1000), round(c, 1)] for t, d, c in scan(ref, src, g, win, step, search, min_conf)]


def analyze_audio(ref, src, windows=20, window=30.0, search=5.0, min_conf=12.0):
    tempo0 = sync.detect_tempo(ref, src)
    if tempo0 != 1.0:
        src = sync.stretch(src, tempo0)
    g = global_offset(ref, src)
    L = int(window * SR)
    step = max((len(src) - L) / SR / (windows + 1), 1.0)
    pts = scan(ref, src, g, window, step, search, min_conf)
    res = {"points": [[round(t, 1), round(d * 1000), round(c, 1)] for t, d, c in pts],
           "dense": dense_map(ref, src, g),
           "delay_ms": None, "tempo": 1.0, "segments": [], "status": "unknown"}
    if len(pts) < 3:
        res["reason"] = "casi nada coincide con la referencia: ¿es el mismo episodio?"
        return res
    if not sync.coverage_ok(pts, len(src) / SR):
        res["reason"] = "solo coincide en un tramo (¿opening/ending compartido?): probablemente otro episodio"
        return res
    delay, tempo, kind, _ = sync.diagnose(pts)
    tempo *= tempo0
    res["delay_ms"] = round(delay * 1000)
    if kind == "cortes":
        if tempo0 == 1.0:
            segs = find_segments(ref, src, g)
            res["segments"] = add_edges(ref, src, segs) if segs else []
        res["status"] = "cuts" if res["segments"] else "unknown"
        if not res["segments"]:
            res["reason"] = "el desfase cambia a saltos y no se pudieron localizar los cortes"
    elif tempo != 1.0:
        res["tempo"], res["status"] = round(tempo, 7), "tempo"
    else:
        res["status"] = "ok" if abs(res["delay_ms"]) < THRESHOLD_MS else "offset"
        segs = add_edges(ref, src, [{"start": 0.0, "end": None, "delay": round(delay, 3)}])
        if len(segs) > 1:  # delay constante salvo un tramo corto al principio o al final
            res["segments"], res["status"] = segs, "cuts"
    return res


# ---------------------------------------------------------------- subtítulos

def _ass_time(s):
    h, m, sec = s.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def sub_events(path, n):
    """[(inicio, fin)] de las líneas de diálogo; None si no es un subtítulo de texto."""
    p = run(["ffmpeg", "-v", "error", "-i", path, "-map", f"0:s:{n}", "-f", "ass", "-"], capture_output=True)
    if p.returncode != 0:
        return None
    ev = []
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        f = line.split(",", 9)
        if len(f) < 10 or re.search(r"\\(pos|move)\(", f[9]):  # carteles, no diálogo
            continue
        try:
            a, b = _ass_time(f[1]), _ass_time(f[2])
        except ValueError:
            continue
        if 0.2 < b - a < 12:
            ev.append((round(a, 2), round(b, 2)))
    return sorted(ev)


FR = 50  # cuadros/s para comparar subtítulos con voz


def speech_activity(x):
    """Energía en la banda de voz por cuadro, normalizada."""
    y = sosfilt(butter(4, [300, 3000], "bandpass", fs=SR, output="sos"), x)
    hop = SR // FR
    n = len(y) // hop
    e = 10 * np.log10((y[:n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-10)
    return np.clip((e - np.median(e)) / (e.std() + 1e-9), -2, 2)


def voice_difference(ref, others):
    """Actividad de voz a partir de la diferencia espectral entre la referencia y
    los doblajes ya alineados: música y efectos coinciden, las voces no.
    others = [(audio, delay_s)]."""
    def mag(x):
        f, _, z = stft(x, fs=SR, nperseg=2 * SR // FR, noverlap=SR // FR, boundary=None)
        return np.abs(z[(f >= 300) & (f <= 3400)])
    a = mag(ref)
    acc = np.zeros(a.shape[1])
    for src, delay in others:
        k = int(round(delay * SR))
        y = np.concatenate([np.zeros(k, np.float32), src]) if k >= 0 else src[-k:]
        y = np.pad(y[:len(ref)], (0, max(0, len(ref) - len(y))))
        b = mag(y)
        acc += np.abs(a - b).sum(axis=0) / (a.sum(axis=0) + b.sum(axis=0) + 1e-3)
    acc /= len(others)
    return np.clip((acc - np.median(acc)) / (acc.std() + 1e-9), -2, 3)


def measure_subs(activity, events, max_lag=20.0):
    """-> (offset_s a sumar a los subtítulos, correlación 0..1 en ese offset)."""
    s = np.zeros(len(activity), dtype=np.float32)
    for a, b in events:
        s[int(a * FR):int(b * FR)] = 1.0
    a0, s0 = activity - activity.mean(), s - s.mean()
    if s.sum() < FR * 20 or a0.std() == 0:
        return 0.0, 0.0
    c = correlate(a0, s0, mode="full", method="fft")
    mid, m = len(s) - 1, int(max_lag * FR)
    c = c[mid - m:mid + m + 1] / (len(s) * a0.std() * s0.std())
    k = int(np.argmax(c))
    return (k - m) / FR, float(c[k])


# ---------------------------------------------------------------- informe y plan

def analyze_file(path, ref_track=0, progress=lambda frac, msg: None, **opts):
    path = Path(path)
    info = probe(path)
    audios = [t for t in info["tracks"] if t["type"] == "audio"]
    subs = [t for t in info["tracks"] if t["type"] == "subtitles"]
    steps, done = len(audios) + len(subs) + 1, 0

    def tick(msg):
        nonlocal done
        progress(done / steps, msg)
        done += 1

    tick("leyendo audio de referencia")
    ref = sync.load_audio(path, ref_track)
    out_audio, aligned = [], []
    for t in audios:
        base = {k: t[k] for k in ("n", "id", "lang", "name", "codec", "channels")}
        if t["n"] == ref_track:
            out_audio.append({**base, "status": "ref", "delay_ms": 0, "tempo": 1.0, "points": [], "segments": []})
            continue
        tick(f"audio {t['lang']}")
        src = sync.load_audio(path, t["n"])
        out_audio.append({**base, **analyze_audio(ref, src, **opts)})
        if out_audio[-1]["status"] in ("ok", "offset"):
            aligned.append((src, out_audio[-1]["delay_ms"] / 1000))

    # con doblajes: la voz es lo que difiere entre pistas; sin ellos: energía en banda de voz
    activity = voice_difference(ref, aligned) if aligned else speech_activity(ref)
    out_subs = []
    for t in subs:
        tick(f"subtítulos {t['lang']}")
        base = {k: t[k] for k in ("n", "id", "lang", "name", "codec")}
        ev = sub_events(path, t["n"]) if t["codec"].startswith(TEXT_SUBS) else None
        if not ev:
            out_subs.append({**base, "status": "unknown", "offset_ms": None, "conf": 0, "events": [],
                             "twin_delay_ms": None})
            continue
        off, conf = measure_subs(activity, ev)
        ms = round(off * 1000)
        status = "unknown" if conf < SUB_MIN_CONF else "ok" if abs(ms) < SUB_THRESHOLD_MS else "offset"
        # subtítulos hechos para un doblaje desfasado (p. ej. "English CC"): heredan su mismo desfase
        twin = [a["delay_ms"] for a in out_audio if a["lang"] == t["lang"] and a["status"] == "offset"
                and abs(a["delay_ms"]) >= 300 and abs(ms - a["delay_ms"]) < abs(ms)]
        if twin and status != "unknown":
            status = "offset"
        out_subs.append({**base, "status": status, "offset_ms": ms, "conf": round(conf, 2), "events": ev,
                         "twin_delay_ms": twin[0] if twin else None})

    order = ["ok", "ref", "unknown", "offset", "tempo", "cuts"]
    worst = max([a["status"] for a in out_audio] + [s["status"] for s in out_subs if s["status"] == "offset"],
                key=order.index)
    progress(1.0, "listo")
    return {"file": path.name, "duration": info["duration"], "ref_track": ref_track,
            "status": "ok" if worst == "ref" else worst,
            "suggest_start": round(max(0.0, transient(ref) - 1.6), 2),
            "audio": out_audio, "subs": out_subs}


def transient(x):
    """Segundo con el mayor salto de energía tras un tramo silencioso."""
    hop = SR // 100
    n = len(x) // hop
    e = 10 * np.log10((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12)
    quiet = np.convolve(e, np.ones(30) / 30, mode="full")[:len(e)]
    return (int(np.argmax(e[31:] - quiet[30:-1])) + 31) * hop / SR


def propose(report):
    """Plan de corrección por defecto a partir del informe."""
    plan = {"audio": {}, "subs": {}}
    for a in report["audio"]:
        if a["status"] == "offset":
            plan["audio"][str(a["n"])] = {"mode": "delay", "delay_ms": a["delay_ms"]}
        elif a["status"] == "cuts":
            plan["audio"][str(a["n"])] = {"mode": "segments", "segments": a["segments"], "codec": "aac"}
        elif a["status"] == "tempo":
            plan["audio"][str(a["n"])] = {"mode": "tempo", "tempo": a["tempo"], "delay_ms": a["delay_ms"], "codec": "aac"}
    for s in report["subs"]:
        if s["status"] == "offset":
            plan["subs"][str(s["n"])] = {"delay_ms": s.get("twin_delay_ms") or s["offset_ms"]}
    return plan


def plan_is_empty(plan):
    a = [p for p in plan.get("audio", {}).values() if p.get("mode", "none") != "none"
         and not (p["mode"] == "delay" and not p.get("delay_ms"))]
    s = [p for p in plan.get("subs", {}).values() if p.get("delay_ms")]
    return not a and not s


def shift_at(entry, t):
    """Segundos que el plan desplaza esa pista en el instante t (de salida)."""
    if not entry or entry.get("mode", "none") == "none":
        return 0.0
    if entry["mode"] == "segments":
        for s in entry["segments"]:
            end = float("inf") if s["end"] is None else s["end"]
            if s["start"] + s["delay"] <= t < end + s["delay"]:
                return s["delay"]
        return entry["segments"][0]["delay"]
    return entry.get("delay_ms", 0) / 1000.0


# ---------------------------------------------------------------- aplicar

def _encode_args(codec, channels):
    if codec == "flac":
        return ["-c:a", "flac"]
    return ["-c:a", "aac", "-b:a", "192k" if (channels or 2) <= 2 else "384k"]


def render_segments(path, track, segments, out, codec):
    """Recoloca cada tramo en su sitio (silencio en los huecos) y recodifica la pista."""
    sr, ch = int(track["sample_rate"] or 48000), int(track["channels"] or 2)
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix=".raw") as tmp:
        r = run(["ffmpeg", "-v", "error", "-y", "-i", path, "-map", f"0:a:{track['n']}",
                 "-af", "aresample=async=1:first_pts=0", "-ar", sr, "-ac", ch, "-f", "s16le", tmp.name])
        if r.returncode != 0:
            raise RuntimeError("ffmpeg no pudo decodificar la pista")
        src = np.memmap(tmp.name, dtype=np.int16, mode="r").reshape(-1, ch)
        spans = []
        for s in segments:
            a = int(s["start"] * sr)
            b = len(src) if s["end"] is None else min(int(s["end"] * sr), len(src))
            pos = a + int(round(s["delay"] * sr))
            if pos < 0:
                a, pos = a - pos, 0
            if b > a:
                spans.append((a, b, pos))
        dst = np.zeros((max(pos + b - a for a, b, pos in spans), ch), dtype=np.int16)
        fade = np.linspace(0, 1, sr // 200)[:, None]  # 5 ms para que el empalme no haga clic
        for i, (a, b, pos) in enumerate(spans):
            seg = src[a:b].astype(np.float32)
            if len(seg) > 2 * len(fade):
                if i > 0:
                    seg[:len(fade)] *= fade
                if i < len(spans) - 1:
                    seg[-len(fade):] *= fade[::-1]
            dst[pos:pos + len(seg)] = seg.astype(np.int16)
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(sr), "-ac", str(ch),
                              "-i", "-", *_encode_args(codec, ch), str(out)], stdin=subprocess.PIPE)
        p.stdin.write(dst.tobytes())
        p.stdin.close()
        if p.wait() != 0:
            raise RuntimeError("ffmpeg no pudo codificar la pista corregida")


def render_tempo(path, track, tempo, out, codec):
    r = run(["ffmpeg", "-v", "error", "-y", "-i", path, "-map", f"0:a:{track['n']}",
             "-af", f"atempo={tempo:.8f}", *_encode_args(codec, track["channels"]), out])
    if r.returncode != 0:
        raise RuntimeError("ffmpeg no pudo corregir la velocidad")


def apply(path, plan, out, work=None):
    """Escribe en `out` una copia de `path` con el plan aplicado. Nunca toca el original."""
    path, out = Path(path), Path(out)
    if out.resolve() == path.resolve():
        raise ValueError("la salida no puede ser el archivo original")
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(work or out.parent)
    work.mkdir(parents=True, exist_ok=True)
    info = probe(path)
    main, extra, order, keep, temps = [], [], [], [], []
    try:
        for t in info["tracks"]:
            if t["type"] == "audio":
                p = plan.get("audio", {}).get(str(t["n"]), {})
                mode = p.get("mode", "none")
                if mode in ("segments", "tempo"):
                    ext = ".flac" if p.get("codec") == "flac" else ".mka"
                    f = work / f".{out.stem}.a{t['n']}{ext}"
                    temps.append(f)
                    if mode == "segments":
                        render_segments(path, t, p["segments"], f, p.get("codec", "aac"))
                        delay = 0
                    else:
                        render_tempo(path, t, p["tempo"], f, p.get("codec", "aac"))
                        delay = int(p.get("delay_ms", 0))
                    extra += ["--language", f"0:{t['lang']}", "--track-name", f"0:{t['name']}",
                              "--default-track-flag", f"0:{int(t['default'])}", "--sync", f"0:{delay}", f]
                    order.append(f"{len(temps)}:0")
                    continue
                keep.append(t["id"])
                if mode == "delay" and p.get("delay_ms"):
                    main += ["--sync", f"{t['id']}:{int(p['delay_ms'])}"]
            elif t["type"] == "subtitles":
                ms = plan.get("subs", {}).get(str(t["n"]), {}).get("delay_ms")
                if ms:
                    main += ["--sync", f"{t['id']}:{int(ms)}"]
            order.append(f"0:{t['id']}")
        if temps:
            main += ["--audio-tracks", ",".join(map(str, keep))] if keep else ["--no-audio"]
        tmp_out = out.with_name(f".{out.stem}.tmp.mkv")
        r = run(["mkvmerge", "-q", "-o", tmp_out, "--track-order", ",".join(order), *main, path, *extra],
                capture_output=True, text=True)
        if r.returncode > 1:
            tmp_out.unlink(missing_ok=True)
            raise RuntimeError(f"mkvmerge falló: {r.stdout.strip() or r.stderr.strip()}")
        tmp_out.replace(out)
    finally:
        for f in temps:
            f.unlink(missing_ok=True)
    return out
