from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_LIGHT_DIR = DATA_DIR / "raw_light"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MAP_DIR = OUTPUT_DIR / "maps"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

START_DATE = "2025-05-01"
END_DATE = "2026-05-31"
RAW_LIGHT_ZIP_CANDIDATES = [
    PROJECT_ROOT / "data raw light.zip",
    Path.home() / "Downloads" / "hantavirus_risk_mapping_raw_light_may2025_may2026.zip",
]
RAW_LIGHT_ZIP = next((path for path in RAW_LIGHT_ZIP_CANDIDATES if path.exists()), RAW_LIGHT_ZIP_CANDIDATES[0])
DATASET_VARIANT = "raw_light_asean_may2025_may2026"
REGION_NAME = "ASEAN"
REGION_BOUNDS = {
    "west": 92.0,
    "north": 28.5,
    "south": -11.0,
    "east": 141.0,
}
GRID_RESOLUTION_DEGREES = 0.5
RISK_BUFFER_KM = 50
RANDOM_SEED = 42

RESERVOIR_SPECIES = [
    "Rattus norvegicus",
    "Rattus rattus",
    "Apodemus agrarius",
    "Apodemus flavicollis",
    "Peromyscus maniculatus",
    "Oligoryzomys longicaudatus",
]

FEATURE_COLUMNS = [
    "presence_count",
    "species_richness",
    "era5_temp_c",
    "era5_precip_mm",
    "era5_dewpoint_c",
    "era5_humidity",
    "chirps_precip_mm",
    "modis_lst_day_c",
    "modis_lst_night_c",
    "srtm_elevation_m",
    "srtm_slope_deg",
    "frac_tree",
    "frac_shrub",
    "frac_grass",
    "frac_cropland",
    "frac_built",
    "frac_bare",
    "frac_water",
    "frac_wetland",
]

def ensure_directories() -> None:
    for path in [RAW_DIR, RAW_LIGHT_DIR, PROCESSED_DIR, OUTPUT_DIR, FIGURE_DIR, MAP_DIR, MODEL_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
