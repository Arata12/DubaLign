#!/usr/bin/env python3
"""Revisa archivos con varios audios y subtítulos, y corrige los que estén
desfasados respecto a la pista de referencia (la primera, normalmente japonés).

  .venv/bin/python fix.py media/*.mkv              # revisa y escribe los corregidos en output/
  .venv/bin/python fix.py media/*.mkv --dry-run    # solo informe

Solo se reescriben los archivos que lo necesitan; los originales no se tocan.
Es lo mismo que hace la web (docker compose up), pero sin interfaz.
"""
import argparse
import contextlib
import io
from pathlib import Path

import engine

LABEL = {"ref": "referencia", "ok": "ok", "offset": "DESFASADO", "cuts": "CORTES", "tempo": "VELOCIDAD DISTINTA",
         "unknown": "no medible"}


def show(report):
    for a in report["audio"]:
        ms = "" if a["delay_ms"] is None else f"{a['delay_ms']:+d} ms"
        extra = f"  tramos: {[(s['start'], round(s['delay'] * 1000)) for s in a['segments']]}" if a["segments"] else ""
        print(f"  audio a:{a['n']} {a['lang']:<4} {ms:>9}  {LABEL[a['status']]}{extra}" + (f"  ({a['reason']})" if a.get("reason") else ""))
    for s in report["subs"]:
        ms = "" if s["offset_ms"] is None else f"{s['offset_ms']:+d} ms"
        print(f"  subs  s:{s['n']} {s['lang']:<4} {ms:>9}  {LABEL[s['status']]}  (corr {s['conf']})  {s['name']}")


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("output"))
    ap.add_argument("--ref-track", type=int, default=0, help="nº de pista de audio de referencia")
    ap.add_argument("--codec", choices=["aac", "flac"], default="aac", help="códec si hay que recodificar una pista")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    for f in a.files:
        print(f"\n== {f.name}")
        report = quiet(engine.analyze_file, f, a.ref_track)
        show(report)
        plan = engine.propose(report)
        for p in plan["audio"].values():
            if "codec" in p:
                p["codec"] = a.codec
        if engine.plan_is_empty(plan) or a.dry_run:
            continue
        out = engine.apply(f, plan, a.output / f.name)
        print(f"  -> {out}\n  verificación del corregido:")
        show(quiet(engine.analyze_file, out, a.ref_track))


if __name__ == "__main__":
    main()
