"""Datos para el visor de episodio completo: espectrograma por pista (en teselas PNG),
envolventes de volumen y audio continuo para escuchar todo el episodio.

Todo se calcula una vez por archivo y se guarda en WORK/timeline/<clave>/.
"""
import base64
import io
import subprocess
import threading
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image

import spectro

SR = 16000
NFFT = 512
SPEC_FPS = 50                    # columnas de espectrograma por segundo (20 ms)
SPEC_BINS = 128                  # 0-8 kHz, 62.5 Hz por fila
ENV_FPS = 200                    # envolvente: un valor cada 5 ms
LEVELS = [(50, 40.0), (5, 400.0), (0.5, 4000.0)]   # (columnas/s, segundos por tesela)
DB_LO, DB_HI = -100.0, -25.0
_CMAP = (colormaps["magma"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
_locks, _locks_lock = {}, threading.Lock()


def _lock(key):
    with _locks_lock:
        return _locks.setdefault(key, threading.Lock())


def _features(x):
    """-> (espectrograma uint8 [SPEC_BINS, columnas], envolvente uint8 a ENV_FPS)."""
    hop = SR // SPEC_FPS
    win = np.hanning(NFFT).astype(np.float32)
    n = max(0, (len(x) - NFFT) // hop + 1)
    spec = np.zeros((SPEC_BINS, n), np.uint8)
    for a in range(0, n, 3000):  # por bloques de 60 s para no gastar memoria
        b = min(a + 3000, n)
        idx = np.arange(a, b)[:, None] * hop + np.arange(NFFT)[None, :]
        mag = np.abs(np.fft.rfft(x[idx] * win, axis=1))[:, :2 * SPEC_BINS] / win.sum() * 2
        mag = mag.reshape(b - a, SPEC_BINS, 2).max(axis=2)
        db = 20 * np.log10(mag + 1e-9)
        spec[:, a:b] = np.clip((db - DB_LO) / (DB_HI - DB_LO) * 255, 0, 255).T.astype(np.uint8)
    eh = SR // ENV_FPS
    m = len(x) // eh
    rms = np.sqrt((x[:m * eh].reshape(m, eh) ** 2).mean(axis=1) + 1e-12)
    env = np.clip((20 * np.log10(rms) + 90) / 90 * 255, 0, 255).astype(np.uint8)  # -90..0 dBFS
    return spec, env


def ensure(path, key, n_tracks, root):
    """Calcula (si falta) los datos de todas las pistas de audio del archivo."""
    d = Path(root) / key
    if (d / "done").is_file():
        return d
    with _lock(key):
        if (d / "done").is_file():
            return d
        d.mkdir(parents=True, exist_ok=True)
        for n in range(n_tracks):
            spec, env = _features(spectro.load(path, n, sr=SR))
            np.save(d / f"spec{n}.npy", spec)
            np.save(d / f"env{n}.npy", env)
        (d / "done").write_text("ok")
    return d


def tile_png(d, track, level, index):
    """Tesela del espectrograma de una pista. Nivel 0 = más detalle."""
    fps, span = LEVELS[level]
    spec = np.load(d / f"spec{track}.npy", mmap_mode="r")
    step = int(round(SPEC_FPS / fps))
    a = int(index * span * SPEC_FPS)
    b = min(int((index + 1) * span * SPEC_FPS), spec.shape[1])
    part = np.asarray(spec[:, a:b]) if b > a else np.zeros((SPEC_BINS, 1), np.uint8)
    if step > 1:  # reducir en tiempo quedándonos con el máximo (no se pierden golpes cortos)
        m = part.shape[1] // step * step
        part = part[:, :m].reshape(SPEC_BINS, -1, step).max(axis=2) if m else part[:, :1]
    img = Image.fromarray(_CMAP[part[::-1]])
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=3)
    return buf.getvalue()


def envelopes(d, n_tracks):
    out = []
    for n in range(n_tracks):
        e = np.load(d / f"env{n}.npy")
        out.append(base64.b64encode(e.tobytes()).decode())
    return {"fps": ENV_FPS, "db_lo": -90, "tracks": out}


def audio_file(path, key, root, ref, track, mode):
    """Audio continuo de todo el episodio (AAC en .m4a, se puede buscar con Range).
    mode: split = referencia en el oído izquierdo y la pista en el derecho; solo = solo la pista."""
    out = Path(root) / key / f"audio_{mode}_{ref}_{track}.m4a"
    if out.is_file():
        return out
    with _lock(f"{key}/audio/{mode}/{ref}/{track}"):
        if out.is_file():
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        norm = "aresample=async=1:first_pts=0"
        if mode == "split":
            fc = (f"[0:a:{ref}]{norm},aformat=channel_layouts=mono[l];"
                  f"[0:a:{track}]{norm},aformat=channel_layouts=mono[r];"
                  f"[l][r]join=inputs=2:channel_layout=stereo[a]")
            args = ["-filter_complex", fc, "-map", "[a]"]
        else:
            args = ["-map", f"0:a:{track}", "-af", norm, "-ac", "2"]
        tmp = out.with_suffix(".tmp.m4a")
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), *args, "-c:a", "aac", "-b:a", "160k",
                            "-movflags", "+faststart", str(tmp)], capture_output=True, text=True)
        if r.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(r.stderr.strip() or "ffmpeg falló")
        tmp.replace(out)
    return out
