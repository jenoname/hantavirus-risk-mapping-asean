# Hantavirus Risk Mapping Project

This repository implements the project brief from `Tahap Pemaparan Project_Kelompok 1.pdf` for a May 2025-May 2026 analysis window, using the provided `data raw light.zip` dataset.

## Scope

- Topic: spatial hantavirus risk modelling
- Grid: 0.5 degree cells
- Time range: `2025-05-01` to `2026-05-31`
- Data sources: GBIF reservoir-host occurrence records plus raw-light alternatives: ERA5 monthly climate, MODIS MOD11C3 monthly LST, GMTED2010 elevation, and MCD12C1 land cover
- Models: Random Forest, ResNet-50 patch model when TensorFlow is installed, explicit non-ResNet patch baseline otherwise, late-fusion ensemble
- Outputs: processed dataset, evaluation summary, SHAP/feature-importance outputs, Grad-CAM only for a TensorFlow ResNet-50 run, interactive risk map

The provided raw-light zip contains downloaded climate/elevation/land-cover files and a manifest. The local environment does not currently include heavy NetCDF/HDF geospatial readers, so the pipeline inventories the real raw zip, records dataset substitutions, and creates a runnable ML-ready table using the same schema and stage logic.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/run_pipeline.py
```

Expected generated files:

- `data/processed/dataset_ml.parquet` or `dataset_ml.csv`
- `models/model_rf.pkl`
- `models/model_cnn.pkl`
- `models/ensemble_weights.json`
- `outputs/evaluation_metrics.json`
- `outputs/maps/risk_map.html`
- `reports/final_report_outline.md`
- `data/raw_light/raw_light_summary.json`

## Member Responsibilities

- Anggota 1: `config.py`, `scripts/01_download_gbif.py`, `scripts/02_download_climate.py`, `scripts/04_preprocess.py`
- Anggota 2: `scripts/05_eda.py`, `scripts/06_model_rf.py`, `scripts/06_model_cnn.py`, `scripts/06_ensemble.py`
- Anggota 3: `scripts/07_evaluate.py`, `scripts/07_shap_gradcam.py`, `scripts/08_risk_map.py`

## Raw-Light Dataset Notes

Use the date range in `config.py` for every download/query:

- Start: May 1, 2025
- End: May 31, 2026

Substitutions and limits in this dataset:

- CHIRPS/IMERG rainfall: unavailable/blocked; ERA5 total precipitation is used.
- MODIS MOD11A2: replaced by monthly MOD11C3 CMG LST.
- SRTM: replaced by GMTED2010 mean elevation.
- ESA WorldCover: replaced by MCD12C1 2024 because the 2025 product is not published.
- The current checked output used the non-ResNet patch fallback because TensorFlow was not installed locally. This is reported explicitly in `outputs/evaluation_metrics.json`.

The large `data raw light.zip` is kept outside the final project zip by default to avoid duplicating a 1.21 GB file. Place it in the project root before running the pipeline.
