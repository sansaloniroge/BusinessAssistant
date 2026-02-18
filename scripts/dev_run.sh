#!/usr/bin/env bash
set -euo pipefail

# Script rápido para:
# 1) Levantar Postgres/pgvector con docker-compose
# 2) Esperar a que la DB responda
# 3) Arrancar la API (FastAPI) en modo dev

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

echo "==> Levantando Postgres (docker-compose)"
docker compose up -d postgres

echo "==> Esperando a que Postgres esté listo"
python3 scripts/db_check.py

echo "==> Arrancando API (uvicorn)"
# Nota: ajusta el módulo si tu entrypoint real difiere.
# Alternativas comunes en este repo:
# - apps.api.asgi:app
# - apps.api.main:app
# - apps.api.app:app
exec uvicorn apps.api.asgi:app --host 0.0.0.0 --port 8000 --reload

