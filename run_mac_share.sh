#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f ".venv/bin/python" ]; then
  echo "[ERROR] Виртуальное окружение .venv не найдено. Выполните установку по инструкции."
  exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
echo "Запуск публичной ссылки Neuro-CDSS. Для отключения в любой момент нажмите Ctrl + C."
exec caffeinate -i ".venv/bin/python" main.py --share --no-demo20 "$@"