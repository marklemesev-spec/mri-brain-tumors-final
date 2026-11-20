#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f ".venv/bin/python" ]; then
  echo "[ERROR] Виртуальное окружение .venv не найдено. Сначала выполните установку по README.md."
  exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
exec ".venv/bin/python" main.py "$@"