#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -----------------------------------------------------------
# CONFIGURE PYTHON AND NODE.JS
# -----------------------------------------------------------
MIN_PY=3.9
MAX_PY=3.13
VENV_NAME='venv'

# CHECK NODE VERSION ----------------------------------------
echo 'Looking for a supported Node.js version...'
if ! command -v node &>/dev/null; then
  echo "Node.js is not installed."
  echo "Install Node.js 22 via:"
  echo "  https://nodejs.org/"
  exit 1
fi

REQUIRED_NODE_MAJOR=22
NODE_VERSION=$(node -v | sed 's/^v//')
NODE_MAJOR=$(echo "$NODE_VERSION" | cut -d. -f1)

if [ "$NODE_MAJOR" -ne "$REQUIRED_NODE_MAJOR" ]; then
  echo "Node.js $REQUIRED_NODE_MAJOR required."
  echo " Found Node.js $NODE_VERSION"
  echo "Install Node.js 22 via:"
  echo "  https://nodejs.org/"
  exit 1
fi
echo "Node.js $NODE_VERSION found."

# CHECK PYTHON VERSION (3.9–3.13) ---------------------------
echo 'Looking for a supported Python version...'
FOUND_PY=""
for ver in 3.13 3.12 3.11 3.10 3.9; do
  if command -v python$ver &>/dev/null; then
    FOUND_PY="python$ver"
    break
  fi
done

if [ -z "$FOUND_PY" ]; then
  echo "Python 3.9 to 3.13 is required."
  echo "Install via:"
  echo "  https://www.python.org/downloads/"
  echo "or (macOS with Homebrew):"
  echo "  brew install python@3.13"
  exit 1
fi

echo "Using $($FOUND_PY --version)."

# -----------------------------------------------------------
# SETUP VIRTUAL ENVIRONMENT
# -----------------------------------------------------------
$FOUND_PY -m pip install --upgrade pip virtualenv
if [ ! -d "$VENV_NAME" ]; then
  echo 'Creating virtual environment...'
  $FOUND_PY -m virtualenv "$VENV_NAME"
else
  echo 'Virtual environment already exists.'
fi

source "$VENV_NAME/bin/activate"

# INSTALL PHYSIOVIEW PACKAGE REQUIREMENTS -------------------
pip install -r requirements.txt
deactivate

# -----------------------------------------------------------
# SETUP PHYSIOVIEW BEAT EDITOR
# -----------------------------------------------------------
echo 'Installing the PhysioView Beat Editor...'
# Install frontend dependencies
cd beat-editor/frontend
npm install
cd "$SCRIPT_DIR"

# Install backend dependencies
cd beat-editor/backend
npm install
cd "$SCRIPT_DIR"

echo ''
echo 'Setup complete!'




