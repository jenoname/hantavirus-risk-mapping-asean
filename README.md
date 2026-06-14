# Hantavirus Reservoir-Occurrence Risk Mapping

This repository implements a spatial machine-learning pipeline for mapping
proximity-based reservoir-occurrence risk across an ASEAN-centred geographic
bounding box.

The target is a binary grid label derived from the distance between each grid
centroid and validated occurrence records for configured rodent reservoir
species. It is a spatial screening proxy and does not represent confirmed
Hantavirus infection, human incidence, or transmission probability.

## Study Configuration

- Geographic bounds: 92.0 to 141.0 degrees east and -11.0 to 28.5 degrees north
- Grid resolution: 0.5 degrees
- Analysis period: `2025-05-01` to `2026-05-31`
- Label buffers evaluated: 25, 50, and 75 km
- Dataset split: 70% training, 15% validation, and 15% test
- Tabular model: Random Forest
- Patch model: CNN ResNet-50, with an explicit sklearn MLP fallback
- Ensemble: validation-optimised late fusion

## Data Sources

The pipeline supports the following inputs:

- GBIF-derived reservoir-species occurrence records
- ERA5 monthly climate data in NetCDF format
- MODIS MOD11C3 daytime and nighttime land-surface temperature in HDF format
- MCD12C1 land-cover data in HDF format
- GMTED2010 elevation data in GeoTIFF format

Environmental readers are implemented in `scripts/04_preprocess.py`. If the
required raw environmental files or reader libraries are unavailable or
incomplete, the pipeline uses a deterministic synthetic fallback while
preserving the feature schema defined by `FEATURE_COLUMNS` in `config.py`.

The generated grid covers the complete rectangular bounding box. It does not
apply a national-boundary or land mask.

## Models

### Model A: Random Forest

Random Forest uses the 19 tabular features defined in `config.py`. Training
uses a pipeline containing `StandardScaler`, optional SMOTE, and
`RandomForestClassifier` with balanced class weights.

`RandomizedSearchCV` evaluates 20 parameter combinations using five-fold
cross-validation and F1-macro. A controlled search then compares 100, 200,
300, and 500 trees while holding the other selected parameters constant.

### Model B: CNN ResNet-50

The CNN receives a `32x32x3` patch for each grid cell:

- Channel 0: MODIS daytime land-surface temperature
- Channel 1: weighted land-cover index
- Channel 2: precipitation

Each channel is constructed from per-cell values with deterministic spatial
jitter and per-patch min-max normalisation. If TensorFlow cannot run, the
pipeline records and uses an explicit sklearn MLP patch baseline.

### Model C: Late Fusion

The ensemble combines the two model probabilities:

```text
P(C) = w * P(A) + (1 - w) * P(B)
```

The weight `w` is optimised on the validation split using F1-macro. The test
split is reserved for final evaluation.

## Pipeline

The main stages are:

1. Prepare and validate reservoir-species occurrence records.
2. Prepare climate and environmental source metadata.
3. Prepare raw-light data.
4. Build the grid, features, and distance-based labels.
5. Generate exploratory data analysis outputs.
6. Train Random Forest and the patch model.
7. Optimise late-fusion weights.
8. Evaluate the models and LORO spatial transfer.
9. Generate SHAP, Grad-CAM, or patch-sensitivity outputs.
10. Generate the interactive risk map.

The corresponding scripts are stored in `scripts/`.

## Installation

From PowerShell:

```powershell
Set-Location "D:\hantavirus-risk-mapping-asean"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Adjust the project path if the repository is stored elsewhere.

## Running the Pipeline

Run all stages:

```powershell
python scripts/run_pipeline.py
```

Run an individual stage:

```powershell
python scripts/04_preprocess.py
python scripts/06_model_rf.py
python scripts/06_model_cnn.py
python scripts/06_ensemble.py
python scripts/07_evaluate.py
python scripts/07_shap_gradcam.py
python scripts/08_risk_map.py
```

## Buffer Sensitivity Analysis

The controlled experiment evaluates 25, 50, and 75 km label buffers while
retaining the same 0.5-degree grid and split indices:

```powershell
.\run_buffer_experiments.bat
```

Experiment artefacts are stored under:

```text
experiments/buffer_25km/
experiments/buffer_50km/
experiments/buffer_75km/
```

The consolidated results are written to:

```text
experiments/buffer_comparison.csv
experiments/buffer_comparison.json
```

The preferred buffer is selected using validation F1-macro, with validation
Average Precision as the tie-breaker. Test metrics are not used for buffer
selection.

## Main Outputs

```text
data/raw/gbif_rodent_occurrences_2025_05_to_2026_05.csv
data/processed/dataset_ml.parquet
data/processed/dataset_metadata.json
models/model_rf.pkl
models/model_cnn.pkl
models/model_cnn_resnet50.keras
models/ensemble_weights.json
models/split_indices.json
outputs/evaluation_metrics.json
outputs/rf_n_estimators_comparison.csv
outputs/ensemble_predictions.csv
outputs/figures/shap_feature_importance.png
outputs/figures/shap_beeswarm_plot.png
outputs/figures/gradcam_resnet50_patch.png
outputs/figures/patch_sensitivity_not_gradcam.png
outputs/maps/risk_map.html
```

`model_cnn_resnet50.keras` and the Grad-CAM output are produced only when the
TensorFlow ResNet-50 backend runs successfully.

## Important Interpretation Limits

- GBIF occurrence records indicate species observations, not confirmed
  Hantavirus infection.
- Risk labels represent proximity to retained occurrence records.
- Environmental features may come from the deterministic fallback when the
  required raw files are incomplete.
- CNN patches are generated representations, not local satellite-image crops.
- Occurrence-derived features are structurally related to the distance-based
  target and should be interpreted cautiously.
- The risk map is a research visualisation and requires external
  epidemiological validation before operational use.
