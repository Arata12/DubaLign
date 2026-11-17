#!/usr/bin/env python3
"""Sincroniza el audio de un archivo (src) contra el video de referencia (ref).

Calcula el desfase por correlación cruzada del audio, detecta drift
(p. ej. 25 fps vs 23.976) y cortes distintos, y muxea el resultado con mkvmerge.

  ./run.sh ref.mkv src.mkv                 # un episodio
  ./run.sh input/ref/ input/src/           # carpetas (se emparejan por orden)
  ./run.sh ref.mkv src.mkv --dry-run       # solo analizar
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from fractions import Fraction

from scipy.signal import correlate, resample_poly

SR = 8000  # Hz para el análisis
MEDIA_EXT = {".mkv", ".mp4", ".m4v", ".avi", ".ts", ".m2ts", ".webm",
             ".mka", ".flac", ".ac3", ".eac3", ".dts", ".aac", ".m4a", ".opus", ".mp3", ".wav"}


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], **kw)


def load_audio(path, track):
    """Decodifica la pista de audio nº `track` a mono float32 a SR Hz."""
    p = run(["ffmpeg", "-v", "error", "-i", path, "-map", f"0:a:{track}",
             "-af", "aresample=async=1:first_pts=0",  # respeta el delay del contenedor
             "-ac", "1", "-ar", SR, "-f", "s16le", "-"], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        sys.exit(f"ffmpeg no pudo leer el audio {track} de {path}:\n{p.stderr.decode()}")
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def peak(ref_seg, src_seg):
    """Devuelve (posición de src dentro de ref en muestras, confianza z-score)."""
    c = np.abs(correlate(ref_seg, src_seg, mode="valid", method="fft"))
    k = int(np.argmax(c))
    return k, float((c[k] - c.mean()) / (c.std() + 1e-12))


FILM = 24000 / 1001
# cambios de velocidad típicos entre releases (PAL <-> NTSC film, 24 <-> 23.976)
TEMPOS = [1.0, FILM / 25, 25 / FILM, 24 / 25, 25 / 24, 1000 / 1001, 1001 / 1000]


def stretch(x, tempo):
    """Cambia la velocidad de x por `tempo` (>1 = más rápido/corto)."""
    if tempo == 1.0:
        return x
    f = Fraction(1 / tempo).limit_denominator(5000)
    return resample_poly(x, f.numerator, f.denominator).astype(np.float32)


def detect_tempo(ref, src):
    """Prueba velocidades conocidas con un trozo central de src; devuelve la mejor."""
    n = min(60 * SR, len(src) // 3)
    chunk = src[len(src) // 2 - n // 2: len(src) // 2 + n // 2]
    scores = {}
    for tp in TEMPOS:
        seg = stretch(chunk, tp)
        if len(seg) < len(ref):
            scores[tp] = peak(ref, seg)[1]
    best = max(scores, key=scores.get)
    return best if scores[best] > 1.5 * scores.get(1.0, 0) else 1.0


def analyze(ref, src, windows, win_s, search_s, min_conf):
    """Lista de (t_src, delay, conf): delay = segundos a sumar a src para calzar con ref."""
    # 1) desfase global con todo el audio
    c = np.abs(correlate(ref, src, mode="full", method="fft"))
    g = int(np.argmax(c)) - (len(src) - 1)
    print(f"  desfase global aprox.: {g / SR * 1000:+.0f} ms")

    # 2) refinado por ventanas repartidas en el episodio
    L, S = int(win_s * SR), int(search_s * SR)
    out = []
    for t in np.linspace(0, len(src) - L, windows + 2)[1:-1].astype(int):
        seg = src[t:t + L]
        a = t + g - S
        if seg.std() < 1e-3 or a < 0 or a + L + 2 * S > len(ref):
            continue
        k, conf = peak(ref[a:a + L + 2 * S], seg)
        if conf >= min_conf:
            out.append((t / SR, (g - S + k) / SR, conf))
    return out


def robust(resid, tol):
    """True si >=80% de ventanas cuadran y los descartes son sueltos (no un tramo
    seguido que coincide entre sí, que indicaría un corte real)."""
    bad = np.abs(resid) >= tol
    if bad.mean() > 0.2:
        return False, bad
    i = np.flatnonzero(bad)
    step = any(j - k == 1 and abs(resid[j] - resid[k]) < tol for k, j in zip(i, i[1:]))
    return not step, bad


def coverage_ok(points, duration):
    """Las coincidencias tienen que repartirse por el episodio (al menos una por tercio).
    Si solo encajan en un tramo (p. ej. opening/ending, que se repiten entre episodios),
    probablemente no es el mismo episodio."""
    return len({min(int(3 * p[0] / duration), 2) for p in points}) == 3


def diagnose(points):
    """-> (delay_s, tempo, tipo, nº ventanas descartadas). tempo != 1 implica drift lineal."""
    t = np.array([p[0] for p in points])
    d = np.array([p[1] for p in points])
    med = float(np.median(d))
    ok, bad = robust(d - med, 0.040)
    if ok:
        return float(np.median(d[~bad])), 1.0, "constante", int(bad.sum())
    b, a = np.polyfit(t, d, 1)
    ok, bad = robust(d - (a + b * t), 0.060)
    if ok:
        b, a = np.polyfit(t[~bad], d[~bad], 1)
        # ref_t = (1+b)*src_t + a  ->  estirar src por (1+b), luego retrasar a
        return float(a), 1.0 / (1.0 + float(b)), "drift", int(bad.sum())
    return med, 1.0, "cortes", 0


def audio_ids(path):
    """IDs de mkvmerge de las pistas de audio, en orden."""
    j = json.loads(run(["mkvmerge", "-J", path], capture_output=True, text=True).stdout)
    return [t["id"] for t in j.get("tracks", []) if t["type"] == "audio"]


def process(ref_path, src_path, args):
    print(f"\n== {ref_path.name}  <-  {src_path.name}")
    ref = load_audio(ref_path, args.ref_track)
    src = load_audio(src_path, args.src_track)
    tempo0 = detect_tempo(ref, src)
    if tempo0 != 1.0:
        print(f"  velocidad distinta detectada: tempo x{tempo0:.5f}")
        src = stretch(src, tempo0)
    pts = analyze(ref, src, args.windows, args.window, args.search, args.min_conf)
    if len(pts) < 3:
        print("  ERROR: muy pocas ventanas fiables; ¿son el mismo episodio?")
        return False
    for t, d, conf in pts:
        print(f"    t={t / 60:5.1f} min  delay={d * 1000:+8.0f} ms  conf={conf:5.1f}")
    if not coverage_ok(pts, len(src) / SR):
        print("  ERROR: solo encaja en un tramo del episodio (¿opening/ending compartido?); "
              "probablemente no es el mismo episodio.")
        return False
    delay, tempo, kind, dropped = diagnose(pts)
    if dropped:
        print(f"  ({dropped} ventana(s) suelta(s) descartada(s) por no cuadrar con el resto)")
    tempo *= tempo0
    if kind == "constante" and tempo != 1.0:
        kind = "drift"
    ms = round(delay * 1000)

    if kind == "cortes":
        print("  ERROR: el desfase cambia a saltos (cortes/ediciones distintas). "
              "No se puede arreglar con un solo delay; revisa los tiempos de arriba.")
        return False
    print(f"  resultado: {kind}, delay {ms:+d} ms" + (f", tempo x{tempo:.6f}" if tempo != 1 else ""))
    if args.dry_run:
        return True

    audio_file, tid = src_path, audio_ids(src_path)[args.src_track]
    if tempo != 1.0:  # corregir drift recodificando a FLAC
        args.work.mkdir(parents=True, exist_ok=True)
        audio_file, tid = args.work / f"{src_path.stem}.stretched.flac", 0
        r = run(["ffmpeg", "-v", "error", "-y", "-i", src_path, "-map", f"0:a:{args.src_track}",
                 "-af", f"atempo={tempo:.8f}", "-c:a", "flac", audio_file])
        if r.returncode != 0:
            return False

    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"{ref_path.stem}.mkv"
    cmd = ["mkvmerge", "-q", "-o", out, ref_path,
           "--no-video", "--no-chapters", "--no-attachments", "--no-global-tags",
           "--audio-tracks", tid, "--sync", f"{tid}:{ms}",
           "--language", f"{tid}:{args.lang}", "--default-track-flag", f"{tid}:{int(args.default)}"]
    if args.name:
        cmd += ["--track-name", f"{tid}:{args.name}"]
    if not args.src_subs:
        cmd += ["--no-subtitles"]
    r = run(cmd + [audio_file])
    if r.returncode > 1:  # 1 = solo advertencias
        return False
    print(f"  -> {out}")
    return True


def media_files(p):
    return sorted(f for f in p.iterdir() if f.suffix.lower() in MEDIA_EXT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", type=Path, help="video de referencia (archivo o carpeta)")
    ap.add_argument("src", type=Path, help="archivo/carpeta con el audio a sincronizar")
    ap.add_argument("-o", "--output", type=Path, default=Path("output"))
    ap.add_argument("--work", type=Path, default=Path("work"), help="temporales")
    ap.add_argument("--ref-track", type=int, default=0, help="nº de pista de audio en ref (0 = primera)")
    ap.add_argument("--src-track", type=int, default=0, help="nº de pista de audio en src")
    ap.add_argument("--lang", default="spa", help="idioma de la pista añadida")
    ap.add_argument("--name", default="", help="nombre de la pista añadida")
    ap.add_argument("--default", action="store_true", help="marcar la pista añadida como predeterminada")
    ap.add_argument("--src-subs", action="store_true", help="copiar también los subtítulos de src (con el mismo delay NO aplicado)")
    ap.add_argument("--dry-run", action="store_true", help="solo analizar, no muxear")
    ap.add_argument("--windows", type=int, default=20, help="nº de ventanas de análisis")
    ap.add_argument("--window", type=float, default=30, help="duración de cada ventana (s)")
    ap.add_argument("--search", type=float, default=5, help="búsqueda ± alrededor del desfase global (s)")
    ap.add_argument("--min-conf", type=float, default=12, help="confianza mínima por ventana")
    args = ap.parse_args()

    if args.ref.is_dir() != args.src.is_dir():
        sys.exit("ref y src deben ser ambos archivos o ambos carpetas")
    if args.ref.is_dir():
        refs, srcs = media_files(args.ref), media_files(args.src)
        if len(refs) != len(srcs) or not refs:
            sys.exit(f"las carpetas no cuadran: {len(refs)} ref vs {len(srcs)} src")
        pairs = list(zip(refs, srcs))
    else:
        pairs = [(args.ref, args.src)]

    failed = [r.name for r, s in pairs if not process(r, s, args)]
    if failed:
        sys.exit("\nFallaron: " + ", ".join(failed))


if __name__ == "__main__":
    main()
