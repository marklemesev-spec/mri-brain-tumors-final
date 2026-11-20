@echo off
chcp 65001 >nul
setlocal
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Виртуальное окружение .venv не найдено. Сначала выполните установку по README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" main.py %*
pause