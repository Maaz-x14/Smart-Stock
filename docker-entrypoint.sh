#!/bin/sh
# Applies pending migrations before the app starts serving. Fails
# loudly (set -e) rather than starting an app pointed at a stale
# schema - a crash on boot is easier to notice than a 500 at 2am.
set -e

echo "Running migrations..."
alembic upgrade head

echo "Starting app..."
exec "$@"
