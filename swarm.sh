#!/bin/bash
# SwarmCoder v2 launcher
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv/bin/python3"

if [ ! -f "$VENV" ]; then
  echo "ERROR: venv not found."
  echo "Run: python3 -m venv $SCRIPT_DIR/venv && $SCRIPT_DIR/venv/bin/pip install openai"
  exit 1
fi

if [ "$1" = "--batch" ]; then
  shift
  exec "$VENV" "$SCRIPT_DIR/swarm/orchestrator.py" "$@"
fi

exec "$VENV" "$SCRIPT_DIR/swarm/main.py" "$@"
