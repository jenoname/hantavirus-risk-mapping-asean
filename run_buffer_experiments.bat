@echo off
setlocal

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\run_buffer_experiments.py
) else (
    python scripts\run_buffer_experiments.py
)

if errorlevel 1 exit /b 1
endlocal
