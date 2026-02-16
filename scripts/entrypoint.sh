#!/bin/bash
set -e

DB_PATH="${DPS_DB_PATH:-/app/data/image_store.db}"
CSV_PATH="${DPS_DATA_DIR:-/app/data}/${DPS_CSV_FILENAME:-Challenge2.csv}"

echo "=== DepthFrame Processing Service ==="
echo "DB path:  $DB_PATH"
echo "CSV path: $CSV_PATH"

# ── Step 1: Idempotent ingestion ──
# Only run if the database doesn't exist yet.
# To force re-ingestion, delete the DB file and restart the container.

if [ ! -f "$DB_PATH" ]; then
    if [ ! -f "$CSV_PATH" ]; then
        echo "ERROR: CSV file not found at $CSV_PATH"
        echo "Mount the data directory with the CSV file:"
        echo "  docker run -v /path/to/data:/app/data ..."
        exit 1
    fi

    echo "Database not found. Running ingestion..."
    python -m depthframe_processing_service.ingest
    echo "Ingestion complete."
else
    echo "Database exists. Skipping ingestion."
fi

# ── Step 2: Start the API server ──
echo "Starting API server on ${DPS_HOST:-0.0.0.0}:${DPS_PORT:-8000}"
exec uvicorn depthframe_processing_service.api.app:app \
    --host "${DPS_HOST:-0.0.0.0}" \
    --port "${DPS_PORT:-8000}" \
    --workers 1 \
    --log-level "${DPS_LOG_LEVEL:-info}"
