#!/usr/bin/env bash
# Alternativa a `docker compose up --build`: Redis y Phoenix en Docker, la API local con uv
# (útil para desarrollar: los cambios de código se ven con solo reiniciar).
#
#   ./run.sh            # API en http://localhost:8000
#   PORT=8080 ./run.sh  # otro puerto
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  echo "Falta el archivo .env. Copiá la plantilla y completá las keys:  cp .env.example .env" >&2
  exit 1
fi
for comando in docker uv; do
  command -v "$comando" >/dev/null || { echo "Falta '$comando' en el PATH (ver README, sección Requisitos)." >&2; exit 1; }
done

echo "==> Redis y Phoenix (docker compose, espera a que estén sanos)"
docker compose up -d --wait redis phoenix

echo "==> Dependencias (uv sync, Python 3.12)"
uv sync --frozen

echo "==> API en http://localhost:${PORT:-8000}  (Swagger: /docs · Phoenix: http://localhost:6006)"
exec uv run uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
