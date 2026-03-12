#!/usr/bin/env sh
set -eu

if [ "${RUN_MIGRATIONS_ON_START:-0}" = "1" ] || [ "${RUN_MIGRATIONS_ON_START:-0}" = "true" ]; then
  echo "Running migrations..."
  alembic upgrade head
fi

exec "$@"

