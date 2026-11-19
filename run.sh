#!/usr/bin/env bash
# Ejecuta dubalign-sync dentro del contenedor Docker sin dependencias locales.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p output work
exec docker compose run --rm -T --no-deps --entrypoint dubalign-sync \
  -v "$PWD:/host:ro" -v "$PWD/output:/host/output" -v "$PWD/work:/host/work" \
  -w /host web "$@"
