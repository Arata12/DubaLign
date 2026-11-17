#!/usr/bin/env bash
# Crea el venv local si falta y ejecuta sync.py con él.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
    .venv/bin/pip install -q -r requirements.txt
fi
exec .venv/bin/python sync.py "$@"
