#!/bin/bash
# Resolve symlink to find the real script location
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
  SOURCE="$(readlink "$SOURCE")"
done
SCRIPT_DIR="$(cd "$(dirname "$SOURCE")" && pwd)"

VENV="$SCRIPT_DIR/venv/bin/python3"

if [ ! -f "$VENV" ]; then
  echo "ERROR: venv not found at $VENV"
  echo "Run: python3 -m venv $SCRIPT_DIR/venv && $SCRIPT_DIR/venv/bin/pip install openai"
  exit 1
fi

if [ "$1" = "--batch" ]; then
  shift
  exec "$VENV" "$SCRIPT_DIR/swarm/orchestrator.py" "$@"
fi

exec "$VENV" "$SCRIPT_DIR/swarm/main.py" "$@"
