#!/usr/bin/env python3
"""Espectrogramas apilados de todas las pistas de audio de un archivo, para
comprobar la sincronía a ojo.

  .venv/bin/python spectro.py video.mkv                 # zoom automático en un golpe fuerte
  .venv/bin/python spectro.py video.mkv --start 530 --dur 3
  .venv/bin/python spectro.py video.mkv --full          # episodio completo
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SR = 16000


def tracks(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                        "stream_tags=language,title", "-of", "json", str(path)],
                       capture_output=True, text=True)
    out = []
    for i, s in enumerate(json.loads(p.stdout)["streams"]):
        tg = s.get("tags", {})
        out.append(f"a:{i} {tg.get('language', '?')} {tg.get('title', '')}".strip())
    return out


def load(path, track, start=None, dur=None, sr=SR):
    """Mono float32. Con start/dur devuelve exactamente dur*sr muestras (ceros fuera del archivo)."""
    cmd = ["ffmpeg", "-v", "error"]
    pad = 0.0
    if start is not None:
        pad = min(max(-start, 0.0), dur)
        if dur - pad <= 0:
            return np.zeros(int(dur * sr), np.float32)
        cmd += ["-ss", f"{max(start, 0.0):.4f}", "-t", f"{dur - pad:.4f}"]
    p = subprocess.run(cmd + ["-i", str(path), "-map", f"0:a:{track}", "-af", "aresample=async=1:first_pts=0",
                              "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode())
    x = np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if start is None:
        return x
    x = np.concatenate([np.zeros(int(pad * sr), np.float32), x])
    n = int(dur * sr)
    return np.pad(x, (0, max(0, n - len(x))))[:n]


def envelope(x, hop):
    n = len(x) // hop
    return np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12)


def find_transient(x, sr=SR):
    """Segundo donde hay el mayor salto de energía tras un tramo silencioso."""
    hop = sr // 100
    e = 20 * np.log10(envelope(x, hop))
    quiet = np.convolve(e, np.ones(30) / 30, mode="full")[:len(e)]  # media de los 300 ms previos
    jump = e[31:] - quiet[30:-1]
    return (int(np.argmax(jump)) + 31) * hop / sr


def render(path, out, start=None, dur=2.5, full=False, shifts=None, subs=None, title=None):
    """Dibuja el PNG. shifts = {nº audio: s de retraso a simular}; subs = [(etiqueta, eventos, shift_s)]."""
    names = tracks(path)
    shifts = shifts or {}
    if full:
        start, dur = 0.0, None
        audio = []
        for i in range(len(names)):
            x, k = load(path, i), int(shifts.get(i, 0) * SR)
            audio.append(np.concatenate([np.zeros(k, np.float32), x]) if k >= 0 else x[-k:])
    else:
        if start is None:
            start = max(0.0, find_transient(load(path, 0)) - dur * 0.4)
        audio = [load(path, i, start - shifts.get(i, 0), dur) for i in range(len(names))]
    end = start + (dur if dur else max(len(x) for x in audio) / SR)

    n = len(audio)
    rows = n + 1 + bool(subs)
    ratios = [1] * (n + 1) + ([0.12 * len(subs) + 0.25] if subs else [])
    fig, axes = plt.subplots(rows, 1, figsize=(16, 2.6 * sum(ratios)), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": ratios})
    nfft, hop = (4096, 2048) if dur is None or dur > 120 else (1024, 256) if dur > 12 else (512, 32)
    with np.errstate(divide="ignore"):
        for ax, x, name in zip(axes, audio, names):
            ax.specgram(x, NFFT=nfft, Fs=SR, noverlap=nfft - hop, xextent=(start, start + len(x) / SR),
                        cmap="magma", vmin=-110, vmax=-30)
            ax.set_facecolor("black")  # silencio digital (-inf dB) también en negro
            ax.set_ylabel("Hz")
            ax.set_title(name, loc="left", fontsize=10)
            ax.grid(axis="x", color="white", alpha=0.35, lw=0.5)

    # envolventes superpuestas: si están en sync, los golpes caen en el mismo instante
    ax = axes[n]
    ehop = max(SR // 500, int(SR * (end - start) / 3000))
    for x, name in zip(audio, names):
        e = envelope(x, ehop)
        ax.plot(start + np.arange(len(e)) * ehop / SR, 20 * np.log10(e), lw=0.8, label=name)
    ax.set_ylabel("dBFS")
    ax.set_title("Envolvente de volumen (todas superpuestas)", loc="left", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, ncol=n)
    ax.grid(alpha=0.4, lw=0.5)

    if subs:
        ax = axes[-1]
        for i, (label, events, shift) in enumerate(subs):
            bars = [(a + shift, b - a) for a, b in events if a + shift < end and b + shift > start]
            ax.broken_barh(bars, (i - 0.35, 0.7), color=f"C{i}")
        ax.set_yticks(range(len(subs)), [s[0] for s in subs], fontsize=8)
        ax.set_ylim(len(subs) - 0.5, -0.5)
        ax.set_title("Subtítulos (cada barra es una línea en pantalla)", loc="left", fontsize=10)
        ax.grid(axis="x", alpha=0.4, lw=0.5)
    axes[-1].set_xlim(start, end)
    axes[-1].set_xlabel("tiempo (s)")
    if dur and dur <= 6:
        axes[-1].xaxis.set_major_locator(plt.MultipleLocator(0.1))
        axes[-1].tick_params(axis="x", labelsize=7, rotation=90)
    fig.suptitle(title or f"{Path(path).name}  —  " + ("completo" if full else f"{start:.2f}s a {end:.2f}s"), fontsize=11)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return start


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("--start", type=float, help="inicio del zoom (s)")
    ap.add_argument("--dur", type=float, default=2.5, help="duración del zoom (s)")
    ap.add_argument("--full", action="store_true", help="episodio completo en vez de zoom")
    ap.add_argument("-o", "--output", type=Path)
    a = ap.parse_args()
    out = a.output or Path("output") / f"{a.file.stem}.{'full' if a.full else 'zoom'}.png"
    try:
        render(a.file, out, a.start, a.dur, a.full)
    except RuntimeError as e:
        sys.exit(str(e))
    print(out)


if __name__ == "__main__":
    main()
