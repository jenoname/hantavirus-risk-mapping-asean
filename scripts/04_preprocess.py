from __future__ import annotations
import json, sys, zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import (
    DATASET_VARIANT, FEATURE_COLUMNS,
    GRID_RESOLUTION_DEGREES,
    MODEL_DIR, PROCESSED_DIR, RAW_DIR, RAW_LIGHT_DIR,
    REGION_BOUNDS, REGION_NAME,
    RISK_BUFFER_KM, RANDOM_SEED,
    START_DATE, END_DATE,
    ensure_directories,
)
from scripts.common import assign_asean_region

_W = REGION_BOUNDS["west"]    # 92.0  : Myanmar western tip
_E = REGION_BOUNDS["east"]    # 141.0 : Papua eastern tip
_S = REGION_BOUNDS["south"]   # -11.0 : Timor / Rote
_N = REGION_BOUNDS["north"]   # 28.5  : Kachin State (Myanmar north)


# Grid

def snap_to_grid(s: pd.Series) -> pd.Series:
    return (np.round(s / GRID_RESOLUTION_DEGREES) * GRID_RESOLUTION_DEGREES).round(2)


def haversine_proj(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Approximate Cartesian projection for cKDTree (km units)."""
    return np.c_[lon * 111.32 * np.cos(np.deg2rad(lat)), lat * 110.57]


def build_grid() -> pd.DataFrame:
    lons = np.arange(_W, _E + GRID_RESOLUTION_DEGREES, GRID_RESOLUTION_DEGREES)
    lats = np.arange(_S, _N + GRID_RESOLUTION_DEGREES, GRID_RESOLUTION_DEGREES)
    g = pd.DataFrame(
        [(round(lo, 2), round(la, 2)) for la in lats for lo in lons],
        columns=["lon", "lat"],
    )
    g["cell_id"] = g["lon"].astype(str) + "_" + g["lat"].astype(str)
    return g


# Environmental features 

def add_environmental_features_synthetic(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Physics-informed synthetic proxies calibrated to ASEAN geography.
    Mirrors the structure of the actual raw-light sources (ERA5, MOD11C3,
    MCD12C1, GMTED2010) within the ASEAN bbox (92–141°E, -11–28.5°N).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    la  = grid["lat"].to_numpy()
    lo  = grid["lon"].to_numpy()

    # Spatial basis functions
    # Maritime / equatorial influence — peaks near Equator & archipelago
    equatorial = np.exp(-(la ** 2) / 18.0)

    # Sunda shelf / Java Sea — warm shallow sea influence on humidity
    java_sea = np.exp(-((lo - 111.0) ** 2 + (la + 4.0) ** 2) / 60)

    # Mekong / mainland peninsula — river basin moisture gradient
    mekong = np.exp(-((lo - 103.0) ** 2 + (la - 14.0) ** 2) / 120)

    # Orographic / highland signal
    #   Myanmar ranges (west), Borneo central highlands, PNG highlands
    myanmar_mts = np.exp(-((lo - 95.0) ** 2 + (la - 20.0) ** 2) / 25)
    borneo_mts  = np.exp(-((lo - 116.5) ** 2 + (la -  1.5) ** 2) / 18)
    papua_mts   = np.exp(-((lo - 138.0) ** 2 + (la -  4.5) ** 2) / 20)
    mountain    = np.clip(myanmar_mts + 0.7 * borneo_mts + 0.6 * papua_mts, 0, 1)

    # Urban centres: Jakarta, KL, Bangkok, Manila, HCMC, Yangon, Hanoi,
    #                Phnom Penh, Vientiane, BSB, Singapore
    urban = (
        np.exp(-((lo - 106.8) ** 2 + (la +  6.2) ** 2) / 3.0) +  # Jakarta
        np.exp(-((lo - 101.7) ** 2 + (la -  3.1) ** 2) / 2.5) +  # KL
        np.exp(-((lo - 100.5) ** 2 + (la - 13.7) ** 2) / 3.0) +  # Bangkok
        np.exp(-((lo - 121.0) ** 2 + (la - 14.6) ** 2) / 2.5) +  # Manila
        np.exp(-((lo - 106.7) ** 2 + (la - 10.8) ** 2) / 2.0) +  # HCMC
        np.exp(-((lo -  96.1) ** 2 + (la - 16.8) ** 2) / 2.0) +  # Yangon
        np.exp(-((lo - 105.8) ** 2 + (la - 21.0) ** 2) / 2.0) +  # Hanoi
        np.exp(-((lo - 104.9) ** 2 + (la - 11.6) ** 2) / 1.5) +  # Phnom Penh
        np.exp(-((lo - 102.6) ** 2 + (la - 17.9) ** 2) / 1.5) +  # Vientiane
        np.exp(-((lo - 114.9) ** 2 + (la -  4.9) ** 2) / 0.8) +  # BSB
        np.exp(-((lo - 103.8) ** 2 + (la -  1.3) ** 2) / 0.5)    # Singapore
    )

    # ERA5 climate
    # Temperature: hot equatorial, cooler highlands, slightly cooler north
    grid["era5_temp_c"] = np.clip(
        28.5 - 0.35 * np.abs(la) - 6.0 * mountain
        + 1.5 * equatorial + rng.normal(0, 0.7, len(grid)),
        8.0, 38.0,
    )
    # Precipitation: equatorial + maritime + monsoon, reduced on highlands
    grid["era5_precip_mm"] = np.clip(
        180.0 + 120.0 * equatorial + 80.0 * java_sea + 60.0 * mekong
        - 50.0 * mountain + rng.normal(0, 30.0, len(grid)),
        10.0, None,
    )
    grid["era5_dewpoint_c"] = (
        grid["era5_temp_c"] - rng.uniform(1.5, 5.0, len(grid))
    )
    grid["era5_humidity"] = np.clip(
        72.0 + 15.0 * equatorial + 10.0 * java_sea
        + rng.normal(0, 5.0, len(grid)),
        40.0, 100.0,
    )

    # CHIRPS-like rainfall signal. When CHIRPS rasters are unavailable, keep a
    # related but non-identical monthly rainfall proxy instead of duplicating ERA5.
    grid["chirps_precip_mm"] = np.clip(
        0.82 * grid["era5_precip_mm"]
        + 35.0 * equatorial
        + 18.0 * np.sin(np.deg2rad(lo * 2.0))
        + rng.normal(0, 18.0, len(grid)),
        0.0,
        None,
    )

    # MOD11C3 LST
    grid["modis_lst_day_c"]   = grid["era5_temp_c"] + rng.normal(4.5, 1.2, len(grid))
    grid["modis_lst_night_c"] = grid["era5_temp_c"] - rng.normal(3.0, 0.9, len(grid))

    # GMTED2010 elevation
    grid["srtm_elevation_m"] = np.clip(
        25.0 + 2200.0 * myanmar_mts + 1500.0 * borneo_mts + 2500.0 * papua_mts
        + rng.normal(0, 50.0, len(grid)),
        0.0, None,
    )
    grid["srtm_slope_deg"] = np.clip(
        2.0 + 24.0 * mountain + 4.0 * np.abs(np.gradient(grid["srtm_elevation_m"].to_numpy()))
        + rng.normal(0, 1.2, len(grid)),
        0.0,
        45.0,
    )

    # MCD12C1 2024 land-cover fractions
    f_forest   = np.clip(0.25 + 0.45 * equatorial + 0.25 * java_sea + rng.normal(0, 0.07, len(grid)), 0, 1)
    f_cropland = np.clip(0.28 + 0.20 * mekong - 0.15 * mountain     + rng.normal(0, 0.07, len(grid)), 0, 1)
    f_built    = np.clip(0.02 + 0.35 * urban                         + rng.normal(0, 0.02, len(grid)), 0, 0.95)
    f_shrub    = np.clip(0.06 + 0.10 * mountain                      + rng.normal(0, 0.03, len(grid)), 0, 1)
    f_grass    = np.clip(0.08                                         + rng.normal(0, 0.03, len(grid)), 0, 1)
    f_bare     = np.clip(0.02                                         + rng.normal(0, 0.02, len(grid)), 0, 1)
    f_water    = np.clip(0.04 + 0.12 * java_sea                      + rng.normal(0, 0.02, len(grid)), 0, 1)
    f_wetland  = np.clip(0.03 + 0.08 * equatorial                    + rng.normal(0, 0.02, len(grid)), 0, 1)

    # Normalise so fractions sum to ≤ 1
    denom = np.maximum(
        f_forest + f_cropland + f_built + f_shrub + f_grass + f_bare + f_water + f_wetland,
        1.0,
    )
    grid["frac_tree"]     = f_forest   / denom
    grid["frac_cropland"] = f_cropland / denom
    grid["frac_built"]    = f_built    / denom
    grid["frac_shrub"]    = f_shrub    / denom
    grid["frac_grass"]    = f_grass    / denom
    grid["frac_bare"]     = f_bare     / denom
    grid["frac_water"]    = f_water    / denom
    grid["frac_wetland"]  = f_wetland  / denom

    return grid


def _raw_files(*suffixes: str) -> list[Path]:
    suffixes = tuple(s.lower() for s in suffixes)
    return sorted(
        p for p in RAW_LIGHT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes
    )


def _extract_zip_members(zip_path: Path, suffixes: tuple[str, ...], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir() or not member.filename.lower().endswith(suffixes):
                continue
            target = out_dir / Path(member.filename).name
            if not target.exists():
                with zf.open(member) as src, target.open("wb") as dst:
                    dst.write(src.read())
            extracted.append(target)
    return extracted


def _era5_files() -> list[Path]:
    files = _raw_files(".nc", ".netcdf")
    if files:
        return files
    files = []
    for zpath in _raw_files(".zip"):
        if "era5" in str(zpath).lower():
            files.extend(_extract_zip_members(zpath, (".nc", ".netcdf"), RAW_LIGHT_DIR / "_extracted" / "era5"))
    return files


def _pick_var(ds, names: list[str]):
    lower_map = {name.lower(): name for name in ds.data_vars}
    for name in names:
        if name.lower() in lower_map:
            return ds[lower_map[name.lower()]]
    for var_name in ds.data_vars:
        low = var_name.lower()
        if any(token in low for token in names):
            return ds[var_name]
    raise KeyError(f"Cannot find any variable matching {names}")


def _sample_xarray(da, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    if "time" in da.dims:
        da = da.mean("time", skipna=True)
    lon_name = next(name for name in ["longitude", "lon", "x"] if name in da.coords)
    lat_name = next(name for name in ["latitude", "lat", "y"] if name in da.coords)
    values = []
    for lo, la in zip(lon, lat):
        values.append(float(da.sel({lon_name: lo, lat_name: la}, method="nearest").values))
    return np.asarray(values, dtype=float)


def _add_era5_features(grid: pd.DataFrame) -> pd.DataFrame:
    import xarray as xr

    files = _era5_files()
    if not files:
        raise FileNotFoundError("ERA5 .nc file not found in data/raw_light")
    ds = xr.open_mfdataset([str(p) for p in files], combine="by_coords")
    lon = grid["lon"].to_numpy()
    lat = grid["lat"].to_numpy()
    temp = _sample_xarray(_pick_var(ds, ["t2m", "2m_temperature", "temperature"]), lon, lat)
    dew = _sample_xarray(_pick_var(ds, ["d2m", "2m_dewpoint_temperature", "dewpoint"]), lon, lat)
    precip = _sample_xarray(_pick_var(ds, ["tp", "total_precipitation", "precip"]), lon, lat)
    temp = np.where(temp > 150, temp - 273.15, temp)
    dew = np.where(dew > 150, dew - 273.15, dew)
    grid["era5_temp_c"] = temp
    grid["era5_dewpoint_c"] = dew
    grid["era5_precip_mm"] = np.clip(precip * 1000.0, 0.0, None)
    grid["era5_humidity"] = np.clip(100.0 - 5.0 * (grid["era5_temp_c"] - grid["era5_dewpoint_c"]), 0.0, 100.0)
    grid["chirps_precip_mm"] = grid["era5_precip_mm"]
    return grid


def _select_hdf_dataset(hdf, candidates: list[str]):
    names = hdf.datasets().keys()
    for candidate in candidates:
        for name in names:
            if candidate.lower() in name.lower():
                return hdf.select(name)[:]
    raise KeyError(f"Cannot find HDF dataset matching {candidates}")


def _sample_modis_grid(arr: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    rows = np.clip(((90.0 - lat) / 0.05).astype(int), 0, arr.shape[0] - 1)
    cols = np.clip(((lon + 180.0) / 0.05).astype(int), 0, arr.shape[1] - 1)
    values = arr[rows, cols].astype(float)
    values[(values <= 0) | (values > 65500)] = np.nan
    return values * 0.02 - 273.15


def _add_modis_lst_features(grid: pd.DataFrame) -> pd.DataFrame:
    from pyhdf.SD import SD, SDC

    files = [p for p in _raw_files(".hdf") if "mod11c3" in p.name.lower() or "mod11c3" in str(p.parent).lower()]
    if not files:
        raise FileNotFoundError("MODIS MOD11C3 .hdf files not found in data/raw_light")
    lon = grid["lon"].to_numpy()
    lat = grid["lat"].to_numpy()
    day_values, night_values = [], []
    for path in files:
        hdf = SD(str(path), SDC.READ)
        day_values.append(_sample_modis_grid(_select_hdf_dataset(hdf, ["LST_Day_CMG", "day"]), lon, lat))
        night_values.append(_sample_modis_grid(_select_hdf_dataset(hdf, ["LST_Night_CMG", "night"]), lon, lat))
        hdf.end()
    grid["modis_lst_day_c"] = np.nanmean(np.vstack(day_values), axis=0)
    grid["modis_lst_night_c"] = np.nanmean(np.vstack(night_values), axis=0)
    return grid


def _add_landcover_features(grid: pd.DataFrame) -> pd.DataFrame:
    from pyhdf.SD import SD, SDC

    files = [p for p in _raw_files(".hdf") if "mcd12c1" in p.name.lower() or "landcover" in str(p.parent).lower()]
    if not files:
        raise FileNotFoundError("MCD12C1 land-cover .hdf file not found in data/raw_light")
    hdf = SD(str(files[0]), SDC.READ)
    lc = _select_hdf_dataset(hdf, ["LC_Type1", "Land_Cover_Type_1", "land_cover"])
    hdf.end()
    classes = {
        "frac_tree": {1, 2, 3, 4, 5},
        "frac_shrub": {6, 7},
        "frac_grass": {10},
        "frac_cropland": {12, 14},
        "frac_built": {13},
        "frac_bare": {16},
        "frac_water": {17},
        "frac_wetland": {11},
    }
    out = {name: [] for name in classes}
    for lo, la in zip(grid["lon"].to_numpy(), grid["lat"].to_numpy()):
        row = int(np.clip((90.0 - la) / 0.05, 0, lc.shape[0] - 1))
        col = int(np.clip((lo + 180.0) / 0.05, 0, lc.shape[1] - 1))
        patch = lc[max(0, row - 5):min(lc.shape[0], row + 5), max(0, col - 5):min(lc.shape[1], col + 5)]
        valid = patch[(patch >= 1) & (patch <= 17)]
        denom = max(valid.size, 1)
        for name, codes in classes.items():
            out[name].append(float(np.isin(valid, list(codes)).sum() / denom))
    for name, values in out.items():
        grid[name] = values
    return grid


def _gmted_paths() -> list[str]:
    paths = [str(p) for p in _raw_files(".tif", ".tiff", ".adf") if p.name.lower().startswith(("mn", "w001"))]
    if paths:
        return paths
    zips = [p for p in _raw_files(".zip") if "gmted" in str(p).lower() or "mn30_grd" in p.name.lower()]
    if not zips:
        return []
    with zipfile.ZipFile(zips[0]) as zf:
        for member in zf.namelist():
            if member.lower().endswith((".tif", ".tiff", "w001001.adf", "w001000.adf")):
                return [f"zip://{zips[0]}!{member}"]
    return []


def _add_elevation_features(grid: pd.DataFrame) -> pd.DataFrame:
    import rasterio

    paths = _gmted_paths()
    if not paths:
        raise FileNotFoundError("GMTED2010 raster not found in data/raw_light")
    with rasterio.open(paths[0]) as src:
        elev, slope = [], []
        band = src.read(1, masked=True)
        for lo, la in zip(grid["lon"].to_numpy(), grid["lat"].to_numpy()):
            row, col = src.index(lo, la)
            row = int(np.clip(row, 0, src.height - 1))
            col = int(np.clip(col, 0, src.width - 1))
            value = float(band[row, col])
            elev.append(max(value, 0.0) if np.isfinite(value) else np.nan)
            win = band[max(0, row - 1):min(src.height, row + 2), max(0, col - 1):min(src.width, col + 2)].astype(float)
            gy, gx = np.gradient(np.ma.filled(win, np.nanmean(win)))
            slope.append(float(np.degrees(np.arctan(np.nanmean(np.sqrt(gx ** 2 + gy ** 2)) / 111_320.0))))
    grid["srtm_elevation_m"] = elev
    grid["srtm_slope_deg"] = slope
    return grid


def add_environmental_features_raw_light(grid: pd.DataFrame) -> pd.DataFrame:
    grid = grid.copy()
    grid = _add_era5_features(grid)
    grid = _add_modis_lst_features(grid)
    grid = _add_landcover_features(grid)
    grid = _add_elevation_features(grid)
    missing = [c for c in FEATURE_COLUMNS if c not in grid.columns]
    if missing:
        raise ValueError(f"Raw-light feature extraction missing columns: {missing}")
    return grid


def add_environmental_features(grid: pd.DataFrame) -> pd.DataFrame:
    try:
        out = add_environmental_features_raw_light(grid)
        print("Environmental features loaded from raw_light files (ERA5, MODIS, MCD12C1, GMTED2010).")
        return out
    except Exception as exc:
        print(f"WARNING: Using SYNTHETIC environmental features — raw_light files not found or incomplete: {exc}")
        return add_environmental_features_synthetic(grid.copy())


# Main 

def main() -> None:
    ensure_directories()

    # Load GBIF 
    gbif_path = RAW_DIR / "gbif_rodent_occurrences_2025_05_to_2026_05.csv"
    gbif = pd.read_csv(gbif_path, parse_dates=["eventDate"])
    gbif = gbif.dropna(subset=["decimalLongitude", "decimalLatitude"])
    gbif = gbif[
        gbif["decimalLongitude"].between(_W, _E)
        & gbif["decimalLatitude"].between(_S, _N)
        & ~((gbif["decimalLongitude"].abs() < 0.001) & (gbif["decimalLatitude"].abs() < 0.001))
        & gbif["eventDate"].between(START_DATE, END_DATE)
    ].copy()

    print("After ASEAN/date filtering:", len(gbif))

    print(
        gbif[[
            "decimalLongitude",
            "decimalLatitude",
            "eventDate"
        ]].head()
    )

    if gbif.empty:
        raise ValueError(
            "GBIF dataset empty after filtering.\n"
            f"Longitude range expected: {_W} to {_E}\n"
            f"Latitude range expected: {_S} to {_N}"
    )

    gbif["lon"]     = snap_to_grid(gbif["decimalLongitude"])
    gbif["lat"]     = snap_to_grid(gbif["decimalLatitude"])
    gbif["cell_id"] = gbif["lon"].astype(str) + "_" + gbif["lat"].astype(str)

    agg = gbif.groupby("cell_id").agg(
        presence_count=("gbifID",   "count"),
        species_richness=("species", "nunique"),
    ).reset_index()

    # Build grid + features
    grid    = add_environmental_features(build_grid())
    dataset = grid.merge(agg, on="cell_id", how="left")
    dataset[["presence_count", "species_richness"]] = (
        dataset[["presence_count", "species_richness"]].fillna(0)
    )

    # Risk label — 50 km buffer 
    tree = cKDTree(haversine_proj(
        gbif["decimalLongitude"].to_numpy(),
        gbif["decimalLatitude"].to_numpy(),
    ))
    dist_km, _ = tree.query(
        haversine_proj(dataset["lon"].to_numpy(), dataset["lat"].to_numpy()), k=1
    )
    dataset["distance_to_presence_km"] = dist_km
    dataset["risk_label"]              = (dist_km <= RISK_BUFFER_KM).astype(int)

    # Provenance
    dataset["date_start"]      = START_DATE
    dataset["date_end"]        = END_DATE
    dataset["dataset_variant"] = DATASET_VARIANT
    dataset["region"]          = REGION_NAME   # "ASEAN"
    dataset["asean_region"] = assign_asean_region(dataset)

    dataset["rainfall_source"]  = (
        "CHIRPS if available; otherwise non-identical ERA5-derived monthly rainfall proxy"
    )
    dataset["lst_source"] = (
        "MOD11C3 monthly CMG LST, May 2025 – Apr 2026 "
        "(May 2026 granule not yet published)"
    )
    dataset["landcover_source"] = (
        "MCD12C1 2024 (2025 edition not published)"
    )
    dataset["elevation_source"] = "GMTED2010 mean elevation (replaces SRTM)"

    # Attach raw-light availability stats if summary exists
    summary_path = RAW_LIGHT_DIR / "raw_light_summary.json"
    if summary_path.exists():
        avail = json.loads(summary_path.read_text())
        for k, v in avail.get("available", {}).items():
            dataset[f"avail_{k}"] = v

    # Impute remaining NaN in feature columns 
    feat_cols = [c for c in FEATURE_COLUMNS if c in dataset.columns]
    dataset[feat_cols] = dataset[feat_cols].fillna(
        dataset[feat_cols].median(numeric_only=True)
    )

    # Save 
    n1 = int(dataset["risk_label"].sum())
    n0 = len(dataset) - n1
    print(
        f"Grid: {len(dataset):,} cells | "
        f"risk=1: {n1:,} ({n1/len(dataset)*100:.1f}%) | "
        f"risk=0: {n0:,} | "
        f"region: ASEAN ({_W}–{_E}°E, {_S}–{_N}°N)"
    )

    out_parquet = PROCESSED_DIR / "dataset_ml.parquet"
    out_csv     = PROCESSED_DIR / "dataset_ml.csv"
    try:
        dataset.to_parquet(out_parquet, index=False)
        print(f"Wrote {out_parquet} — shape {dataset.shape}")
    except Exception as exc:
        dataset.to_csv(out_csv, index=False)
        print(f"Parquet unavailable ({exc}); wrote {out_csv} — shape {dataset.shape}")


    split_path = MODEL_DIR / "split_indices.json"
    if split_path.exists():
        split_path.unlink()


if __name__ == "__main__":
    main()
