@echo off
REM run_grid_experiments.bat
REM Jalankan dari root project (folder yang ada config.py & scripts\)
REM
REM Usage: run_grid_experiments.bat

setlocal enabledelayedexpansion

for %%G in (25 50 75) do (
    echo   RUNNING PIPELINE WITH GRID = %%G km

    set GRID_RESOLUTION_KM=%%G

    python scripts\run_pipeline.py
    if errorlevel 1 (
        echo ERROR: pipeline gagal pada grid %%G km
        exit /b 1
    )

    set OUT_DIR=outputs_%%Gkm
    set PROC_DIR=data_processed_%%Gkm
    set MODEL_DIR=models_%%Gkm

    if exist "!OUT_DIR!" rmdir /s /q "!OUT_DIR!"
    if exist "!PROC_DIR!" rmdir /s /q "!PROC_DIR!"
    if exist "!MODEL_DIR!" rmdir /s /q "!MODEL_DIR!"

    xcopy outputs "!OUT_DIR!" /E /I /Q
    xcopy data\processed "!PROC_DIR!" /E /I /Q
    xcopy models "!MODEL_DIR!" /E /I /Q

    echo Hasil grid %%G km disimpan di: !OUT_DIR!, !PROC_DIR!, !MODEL_DIR!
    echo.
)

echo   SEMUA EKSPERIMEN SELESAI
echo Bandingkan isi:
echo   - outputs_25km\evaluation_metrics.json
echo   - outputs_50km\evaluation_metrics.json
echo   - outputs_75km\evaluation_metrics.json

endlocal
