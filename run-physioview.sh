#!/usr/bin/env bash
#
# run-physioview.sh
# Launch the PhysioView Dashboard and Beat Editor together in a single
# terminal. Press Ctrl+C to stop all three.
#
# Usage:
#   ./run-physioview.sh

set -u

# Resolve the repo root (the directory this script lives in) so it works
# no matter where it is invoked from.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_PY="$ROOT/venv/bin/python"
FRONTEND_DIR="$ROOT/beat-editor/frontend"
BACKEND_DIR="$ROOT/beat-editor/backend"

# --- Sanity checks ----------------------------------------------------------
if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: virtualenv Python not found at $VENV_PY"
    echo "       Run ./setup.sh first to create the venv and install dependencies."
    exit 1
fi
if [ ! -d "$BACKEND_DIR/node_modules" ] || [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo "WARN: Beat Editor dependencies are missing — run ./setup.sh first."
fi

# --- Teardown: kill every child process in this process group on exit -------
trap 'echo; echo "Shutting down PhysioView..."; kill 0' EXIT INT TERM

# --- Helper: run a command in a directory with a labeled log prefix ---------
# Args: <label> <working-dir> <command> [args...]
run() {
    local label="$1"; shift
    local dir="$1"; shift
    (
        cd "$dir" || exit 1
        "$@" 2>&1 | while IFS= read -r line; do
            printf '[%s] %s\n' "$label" "$line"
        done
    ) &
}

echo "Starting PhysioView Dashboard + Beat Editor (Ctrl+C to stop all)..."

# Dashboard (runs from the repo root so its relative paths resolve correctly)
run "Dashboard"   "$ROOT"          "$VENV_PY" app.py

# Beat Editor backend
run "Beat Editor Backend"  "$BACKEND_DIR"   npm start

# Beat Editor frontend
run "Beat Editor Frontend" "$FRONTEND_DIR"  npm run dev

# Wait for all three; Ctrl+C triggers the trap above to stop everything.
wait
