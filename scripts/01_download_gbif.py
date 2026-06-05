"""Prepare ASEAN GBIF reservoir-host occurrence records for May 2025-May 2026."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import requests

from config import (
    RAW_DIR, RESERVOIR_SPECIES,
    START_DATE, END_DATE,
    REGION_BOUNDS, RANDOM_SEED,
    ensure_directories,
)

_W = REGION_BOUNDS["west"]    # 92.0
_E = REGION_BOUNDS["east"]    # 141.0
_S = REGION_BOUNDS["south"]   # -11.0
_N = REGION_BOUNDS["north"]   # 28.5

# ── Cluster definitions (lon, lat, country, ISO-3) ───────────────────────────
# 2–9 clusters per country; all coordinates verified inside ASEAN bbox.
CLUSTERS = [
    # Indonesia — largest archipelago, 9 clusters
    [106.8,  -6.2, "Indonesia",          "IDN"],  # Jakarta / West Java
    [112.7,  -7.3, "Indonesia",          "IDN"],  # Surabaya / East Java
    [110.4,  -7.8, "Indonesia",          "IDN"],  # Yogyakarta / Central Java
    [100.4,  -0.9, "Indonesia",          "IDN"],  # Padang / Sumatra
    [109.3,   0.0, "Indonesia",          "IDN"],  # Pontianak / Kalimantan
    [120.0,  -5.2, "Indonesia",          "IDN"],  # Makassar / Sulawesi
    [115.2,  -8.7, "Indonesia",          "IDN"],  # Bali
    [131.5,  -0.9, "Indonesia",          "IDN"],  # Maluku / Ambon
    [140.7,  -3.7, "Indonesia",          "IDN"],  # Papua / Jayapura
    # Malaysia — 3 clusters (Peninsula + Borneo)
    [101.7,   3.1, "Malaysia",           "MYS"],  # Kuala Lumpur
    [110.3,   1.5, "Malaysia",           "MYS"],  # Kuching / Sarawak
    [116.1,   5.9, "Malaysia",           "MYS"],  # Kota Kinabalu / Sabah
    # Thailand — 2 clusters
    [100.5,  13.7, "Thailand",           "THA"],  # Bangkok
    [ 98.9,  18.8, "Thailand",           "THA"],  # Chiang Mai
    # Filipina — 2 clusters
    [121.0,  14.6, "Filipina",           "PHL"],  # Metro Manila / Luzon
    [124.0,   8.5, "Filipina",           "PHL"],  # Davao / Mindanao
    # Viet Nam — 3 clusters (North / Central / South)
    [105.8,  21.0, "Viet Nam",           "VNM"],  # Hanoi
    [108.2,  16.1, "Viet Nam",           "VNM"],  # Da Nang
    [106.7,  10.8, "Viet Nam",           "VNM"],  # Ho Chi Minh City
    # Myanmar — 2 clusters
    [ 96.1,  16.8, "Myanmar",            "MMR"],  # Yangon
    [ 96.0,  21.9, "Myanmar",            "MMR"],  # Mandalay
    # Kamboja — 2 clusters
    [104.9,  11.6, "Kamboja",            "KHM"],  # Phnom Penh
    [103.9,  13.4, "Kamboja",            "KHM"],  # Siem Reap
    # Laos — 2 clusters
    [102.6,  17.9, "Laos",               "LAO"],  # Vientiane
    [102.1,  19.9, "Laos",               "LAO"],  # Luang Prabang
    # Brunei Darussalam — 2 clusters (small country, tight jitter)
    [114.9,   4.9, "Brunei Darussalam",  "BRN"],  # Bandar Seri Begawan
    [114.6,   4.6, "Brunei Darussalam",  "BRN"],  # Tutong
    # Singapura — 2 clusters (city-state, very tight jitter)
    [103.8,   1.3, "Singapura",          "SGP"],  # Central
    [103.7,   1.4, "Singapura",          "SGP"],  # West
]

_ARR     = np.array([[c[0], c[1]] for c in CLUSTERS])
_COUNTRY = [c[2] for c in CLUSTERS]
_ISO     = [c[3] for c in CLUSTERS]

# Sampling weight — proportional to country area / rodent habitat
_WEIGHTS = np.array([
    100, 90, 85, 80, 65, 70, 60, 45, 40,   # Indonesia (9)
     60, 45, 40,                             # Malaysia  (3)
     65, 50,                                 # Thailand  (2)
     60, 50,                                 # Filipina  (2)
     55, 55, 60,                             # Viet Nam  (3)
     50, 45,                                 # Myanmar   (2)
     45, 40,                                 # Kamboja   (2)
     40, 35,                                 # Laos      (2)
     20, 18,                                 # Brunei    (2)
     22, 20,                                 # Singapura (2)
], dtype=float)
_WEIGHTS /= _WEIGHTS.sum()

# Clusters that need very tight jitter (small territories)
_TIGHT_IDX = set(range(25, 29))  # Brunei (25,26) + Singapura (27,28)
ASEAN_COUNTRY_CODES = {
    "BN", "KH", "ID", "LA", "MY", "MM", "PH", "SG", "TH", "VN",  # ASEAN core
    "TW", "HK", "MO",  # Taiwan, Hong Kong, Macau — within ASEAN bbox
    "CX", "CC",        # Christmas Island, Cocos Islands — Indian Ocean territories
    "CN",              # China southern border overlaps ASEAN bbox
    "IN",              # India Andaman & Nicobar within bbox
}
ASEAN_CODE_TO_NAME = {
    "BN": "Brunei Darussalam",
    "KH": "Cambodia",
    "ID": "Indonesia",
    "LA": "Laos",
    "MY": "Malaysia",
    "MM": "Myanmar",
    "PH": "Philippines",
    "SG": "Singapore",
    "TH": "Thailand",
    "VN": "Viet Nam",
    "TW": "Taiwan",
    "HK": "Hong Kong",
    "MO": "Macau",
    "CX": "Christmas Island",
    "CC": "Cocos Islands",
    "CN": "China",
    "IN": "India",
}
RESERVOIR_PREFIXES = ("Apodemus ", "Peromyscus ", "Oligoryzomys ")
GBIF_API_URL = "https://api.gbif.org/v1/occurrence/search"
OUTPUT_COLUMNS = [
    "gbifID",
    "decimalLongitude",
    "decimalLatitude",
    "species",
    "eventDate",
    "countryCode",
    "country",
    "basisOfRecord",
    "source",
]


def create_asean_gbif_records(n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    idx     = rng.choice(len(CLUSTERS), size=n, p=_WEIGHTS)
    centers = _ARR[idx]

    sigma = np.where(np.isin(idx, list(_TIGHT_IDX)), 0.12, 0.9)
    lon   = np.clip(centers[:, 0] + rng.normal(0, 1, n) * sigma, _W, _E)
    lat   = np.clip(centers[:, 1] + rng.normal(0, 1, n) * sigma, _S, _N)

    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    return pd.DataFrame({
        "gbifID":           np.arange(1, n + 1),
        "species":          rng.choice(RESERVOIR_SPECIES, size=n),
        "decimalLongitude": lon,
        "decimalLatitude":  lat,
        "countryCode":      [_ISO[i]     for i in idx],
        "country":          [_COUNTRY[i] for i in idx],
        "eventDate":        rng.choice(dates, size=n).astype("datetime64[ns]"),
        "basisOfRecord":    "HUMAN_OBSERVATION",
        "source":           "demo_synthetic_gbif_schema",
    })


def validate_occurrences(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["decimalLongitude"] = pd.to_numeric(df["decimalLongitude"], errors="coerce")
    df["decimalLatitude"] = pd.to_numeric(df["decimalLatitude"], errors="coerce")
    df["eventDate"] = pd.to_datetime(df["eventDate"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["decimalLongitude", "decimalLatitude", "eventDate"])
    df = df[
        df["decimalLongitude"].between(_W, _E)
        & df["decimalLatitude"].between(_S, _N)
        & df["countryCode"].isin(ASEAN_COUNTRY_CODES)
        & ~((df["decimalLongitude"].abs() < 0.001) & (df["decimalLatitude"].abs() < 0.001))
        & df["eventDate"].between(START_DATE, END_DATE)
    ].copy()
    df["gbifID"] = df["gbifID"].astype(str)
    df["species"] = df["species"].fillna("Rodentia sp.")
    df["country"] = df["country"].fillna(df["countryCode"])
    df["basisOfRecord"] = df["basisOfRecord"].fillna("GBIF_OCCURRENCE")
    return df[OUTPUT_COLUMNS]


def _records_from_gbif_results(results: list[dict], source: str) -> list[dict]:
    rows = []
    for item in results:
        rows.append({
            "gbifID": item.get("gbifID") or item.get("key"),
            "decimalLongitude": item.get("decimalLongitude"),
            "decimalLatitude": item.get("decimalLatitude"),
            "species": item.get("species") or item.get("scientificName"),
            "eventDate": item.get("eventDate") or item.get("dateIdentified"),
            "countryCode": item.get("countryCode"),
            "country": item.get("country"),
            "basisOfRecord": item.get("basisOfRecord"),
            "source": source,
        })
    return rows


def _download_with_requests() -> pd.DataFrame:
    rows = []
    for species in RESERVOIR_SPECIES:
        offset = 0
        while True:
            params = {
                "hasCoordinate": "true",
                "hasGeospatialIssue": "false",
                "scientificName": species,
                "decimalLatitude": f"{_S},{_N}",
                "decimalLongitude": f"{_W},{_E}",
                "year": "2025,2026",
                "limit": 300,
                "offset": offset,
            }
            resp = requests.get(GBIF_API_URL, params=params, timeout=45)
            resp.raise_for_status()
            payload = resp.json()
            results = payload.get("results", [])
            rows.extend(_records_from_gbif_results(results, "gbif_occurrence_api"))
            if payload.get("endOfRecords", False) or not results:
                break
            offset += len(results)
    return validate_occurrences(pd.DataFrame(rows))


def _download_with_pygbif() -> pd.DataFrame:
    from pygbif import occurrences

    rows = []
    for species in RESERVOIR_SPECIES:
        offset = 0
        while True:
            payload = occurrences.search(
                hasCoordinate=True,
                hasGeospatialIssue=False,
                scientificName=species,
                decimalLatitude=f"{_S},{_N}",
                decimalLongitude=f"{_W},{_E}",
                year="2025,2026",
                limit=300,
                offset=offset,
            )
            results = payload.get("results", [])
            rows.extend(_records_from_gbif_results(results, "pygbif_occurrence_api"))
            if payload.get("endOfRecords", False) or not results:
                break
            offset += len(results)
    return validate_occurrences(pd.DataFrame(rows))


def download_from_gbif_api() -> pd.DataFrame:
    """Download real reservoir-host occurrences from GBIF with pygbif fallback."""
    try:
        df = _download_with_requests()
        if not df.empty:
            return df
        print("GBIF REST API returned no valid ASEAN/date-filtered records.")
    except Exception as exc:
        print(f"GBIF REST API failed: {exc}")
    try:
        df = _download_with_pygbif()
        if not df.empty:
            return df
        print("pygbif returned no valid ASEAN/date-filtered records.")
    except Exception as exc:
        print(f"pygbif fallback failed: {exc}")
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def load_real_asean_gbif_records() -> pd.DataFrame:
    from config import RAW_DIR, RESERVOIR_SPECIES, REGION_BOUNDS

    # Try the large local file first
    local_path = RAW_DIR / "rodent_occurrence.csv"
    api_path   = RAW_DIR / "gbif_rodent_occurrences_2025_05_to_2026_05.csv"

    df = None
    for path in [local_path, api_path]:
        if path.exists():
            try:
                df = pd.read_csv(path)
                print(f"Loaded {len(df)} records from {path.name}")
                break
            except Exception as exc:
                print(f"Could not read {path.name}: {exc}")

    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Keep only hantavirus reservoir species
    reservoir = [s.lower() for s in RESERVOIR_SPECIES]
    if "species" in df.columns:
        df = df[df["species"].str.lower().isin(reservoir)]
    elif "scientificName" in df.columns:
        df = df[df["scientificName"].str.lower().str.contains(
            "|".join([s.split()[1] for s in RESERVOIR_SPECIES]), na=False
        )]
        df["species"] = df["scientificName"]

    print(f"After species filter: {len(df)} reservoir records")

    # Standardize coordinate column names
    if "decimalLongitude" not in df.columns:
        df = df.rename(columns={"longitude": "decimalLongitude", "lon": "decimalLongitude"})
    if "decimalLatitude" not in df.columns:
        df = df.rename(columns={"latitude": "decimalLatitude", "lat": "decimalLatitude"})

    # Filter to ASEAN bbox
    b = REGION_BOUNDS
    df = df[
        (df["decimalLongitude"] >= b["west"])  &
        (df["decimalLongitude"] <= b["east"])  &
        (df["decimalLatitude"]  >= b["south"]) &
        (df["decimalLatitude"]  <= b["north"])
    ]

    # Drop nulls and Null Island
    df = df.dropna(subset=["decimalLongitude", "decimalLatitude"])
    df = df[~((df["decimalLongitude"].abs() < 0.1) &
              (df["decimalLatitude"].abs() < 0.1))]

    print(f"After ASEAN bbox filter: {len(df)} records")
    return df


def main() -> None:
    ensure_directories()
    df = load_real_asean_gbif_records()
    if not df.empty:
        # Ensure all OUTPUT_COLUMNS exist before validate_occurrences
        if "country" not in df.columns and "countryCode" in df.columns:
            df["country"] = df["countryCode"]
        if "gbifID" not in df.columns:
            df["gbifID"] = range(1, len(df) + 1)
        if "basisOfRecord" not in df.columns:
            df["basisOfRecord"] = "GBIF_OCCURRENCE"
        if "source" not in df.columns:
            df["source"] = "local_csv"
        if "eventDate" not in df.columns:
            df["eventDate"] = pd.Timestamp(START_DATE)
        df = validate_occurrences(df)
    if df.empty:
        df = download_from_gbif_api()
    if df.empty:
        print("WARNING: Using SYNTHETIC data — GBIF API unavailable. Install requests/pygbif and retry.")
        df = create_asean_gbif_records()
        df = validate_occurrences(df)
    out = RAW_DIR / "gbif_rodent_occurrences_2025_05_to_2026_05.csv"
    df.to_csv(out, index=False)

    summary = (
        df.groupby("country")
          .agg(n=("gbifID", "count"), species=("species", "nunique"))
          .sort_values("n", ascending=False)
    )
    print(f"\nWrote {out} — {len(df):,} records | {df['country'].nunique()} countries")
    print(f"lon [{df['decimalLongitude'].min():.2f}, {df['decimalLongitude'].max():.2f}] "
          f"lat [{df['decimalLatitude'].min():.2f}, {df['decimalLatitude'].max():.2f}]")
    print("\nRecords per country:\n" + summary.to_string())
    print(df[["decimalLongitude", "decimalLatitude"]].describe())
    print(df[["decimalLongitude", "decimalLatitude"]].head(20))


if __name__ == "__main__":
    main()
