"""Download raw Asia datasets for the May 2025-May 2026 project window.

This script downloads public sources directly when possible and writes a
manifest for sources that require a separate portal/API setup.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import sys
import time
from datetime import date
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import RAW_DIR, START_DATE, END_DATE, ensure_directories


ASIA_BBOX = {
    "west": 25.0,
    "south": -11.0,
    "east": 180.0,
    "north": 82.0,
}

GBIF_FIELDS = [
    "key",
    "scientificName",
    "species",
    "decimalLatitude",
    "decimalLongitude",
    "eventDate",
    "year",
    "month",
    "countryCode",
    "basisOfRecord",
    "institutionCode",
    "collectionCode",
    "datasetKey",
    "gbifID",
]


def month_range(start: str, end: str) -> Iterable[tuple[int, int]]:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    year, month = start_date.year, start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        yield year, month
        month += 1
        if month == 13:
            year += 1
            month = 1


def stream_download(url: str, out_path: Path, session: requests.Session) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if out_path.exists() and out_path.stat().st_size > 0:
        return {"url": url, "path": str(out_path), "status": "already_exists", "bytes": out_path.stat().st_size}

    with session.get(url, stream=True, timeout=120) as response:
        if response.status_code == 404:
            return {"url": url, "path": str(out_path), "status": "not_found"}
        response.raise_for_status()
        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    for attempt in range(10):
        try:
            tmp_path.replace(out_path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(1)
    return {"url": url, "path": str(out_path), "status": "downloaded", "bytes": out_path.stat().st_size}


def gunzip_file(gz_path: Path) -> Path:
    out_path = gz_path.with_suffix("")
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with gzip.open(gz_path, "rb") as src, tmp_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    tmp_path.replace(out_path)
    return out_path


def download_chirps(session: requests.Session) -> list[dict]:
    records = []
    chirps_dir = RAW_DIR / "chirps"
    base = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs"
    for year, month in month_range(START_DATE, END_DATE):
        name = f"chirps-v2.0.{year}.{month:02d}.tif.gz"
        gz_path = chirps_dir / name
        record = stream_download(f"{base}/{name}", gz_path, session)
        try:
            if record["status"] == "not_found":
                records.append(record)
                continue
            tif_path = gunzip_file(gz_path)
            record["uncompressed_path"] = str(tif_path)
            record["uncompressed_bytes"] = tif_path.stat().st_size
        except OSError as exc:
            record["uncompress_error"] = str(exc)
        records.append(record)
    return records


def get_rodentia_taxon_key(session: requests.Session) -> int:
    response = session.get(
        "https://api.gbif.org/v1/species/match",
        params={"name": "Rodentia", "rank": "ORDER"},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return int(data.get("usageKey") or 1459)


def download_gbif_occurrences(session: requests.Session, max_records: int | None = None) -> dict:
    gbif_dir = RAW_DIR / "gbif"
    gbif_dir.mkdir(parents=True, exist_ok=True)
    out_path = gbif_dir / "rodent_occurrence.csv"
    taxon_key = get_rodentia_taxon_key(session)
    base_params = {
        "taxonKey": taxon_key,
        "continent": "ASIA",
        "hasCoordinate": "true",
        "hasGeospatialIssue": "false",
        "eventDate": f"{START_DATE},{END_DATE}",
        "limit": 300,
    }

    count_params = dict(base_params)
    count_params["limit"] = 0
    count_response = session.get("https://api.gbif.org/v1/occurrence/search", params=count_params, timeout=120)
    count_response.raise_for_status()
    total = int(count_response.json().get("count", 0))

    existing_rows = 0
    if out_path.exists():
        with out_path.open("r", newline="", encoding="utf-8") as fh:
            existing_rows = max(sum(1 for _ in fh) - 1, 0)

    if max_records is None and existing_rows >= total:
        return {
            "path": str(out_path),
            "bytes": out_path.stat().st_size,
            "taxon_key": taxon_key,
            "query_total": total,
            "written": existing_rows,
            "status": "already_complete",
        }

    written = existing_rows
    offset = existing_rows
    mode = "a" if existing_rows else "w"
    with out_path.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=GBIF_FIELDS)
        if not existing_rows:
            writer.writeheader()
        while True:
            params = dict(base_params)
            params["offset"] = offset
            response = session.get("https://api.gbif.org/v1/occurrence/search", params=params, timeout=120)
            response.raise_for_status()
            data = response.json()
            rows = data.get("results", [])
            if not rows:
                break
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in GBIF_FIELDS})
                written += 1
                if max_records is not None and written >= max_records:
                    break
            if max_records is not None and written >= max_records:
                break
            offset += len(rows)
            if offset >= total or len(rows) < base_params["limit"]:
                break
            if offset >= 100000:
                break
            time.sleep(0.2)

    return {
        "path": str(out_path),
        "bytes": out_path.stat().st_size,
        "taxon_key": taxon_key,
        "query_total": total,
        "written": written,
        "note": "GBIF public occurrence search pages up to the API offset limit; use a GBIF account for a complete async download if query_total exceeds written.",
    }


def days_for_month(year: int, month: int) -> list[str]:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last_day = (next_month - date.resolution).day
    return [f"{day:02d}" for day in range(1, last_day + 1)]


def download_era5_land() -> list[dict]:
    try:
        import cdsapi
    except ImportError:
        return [{"status": "missing_dependency", "package": "cdsapi"}]

    url = os.getenv("CDSAPI_URL", "").strip()
    key = os.getenv("CDSAPI_KEY", "").strip()
    if not url or not key:
        return [{"status": "missing_credentials", "required_env": ["CDSAPI_URL", "CDSAPI_KEY"]}]

    era5_dir = RAW_DIR / "era5"
    era5_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(url=url, key=key, quiet=False, progress=True, sleep_max=60)
    dataset = "derived-era5-land-daily-statistics"
    jobs = [
        ("temperature_mean", "2m_temperature", "daily_mean"),
        ("dewpoint_mean", "2m_dewpoint_temperature", "daily_mean"),
        ("precipitation_sum", "total_precipitation", "daily_sum"),
    ]
    selected_jobs = {
        item.strip()
        for item in os.getenv("ERA5_JOBS", "").split(",")
        if item.strip()
    }
    if selected_jobs:
        jobs = [job for job in jobs if job[0] in selected_jobs]
    records = []

    for year, month in month_range(START_DATE, END_DATE):
        for label, variable, statistic in jobs:
            out_path = era5_dir / f"era5_land_{label}_{year}_{month:02d}_asia.zip"
            if out_path.exists() and out_path.stat().st_size > 0:
                records.append({
                    "path": str(out_path),
                    "status": "already_exists",
                    "bytes": out_path.stat().st_size,
                    "variable": variable,
                    "daily_statistic": statistic,
                    "year": year,
                    "month": month,
                })
                continue

            request = {
                "variable": [variable],
                "year": f"{year}",
                "month": f"{month:02d}",
                "day": days_for_month(year, month),
                "daily_statistic": statistic,
                "time_zone": "UTC+00:00",
                "frequency": "1_hourly",
                "area": [ASIA_BBOX["north"], ASIA_BBOX["west"], ASIA_BBOX["south"], ASIA_BBOX["east"]],
            }
            try:
                client.retrieve(dataset, request, str(out_path))
                records.append({
                    "path": str(out_path),
                    "status": "downloaded",
                    "bytes": out_path.stat().st_size,
                    "variable": variable,
                    "daily_statistic": statistic,
                    "year": year,
                    "month": month,
                })
            except Exception as exc:
                records.append({
                    "path": str(out_path),
                    "status": "failed",
                    "variable": variable,
                    "daily_statistic": statistic,
                    "year": year,
                    "month": month,
                    "error": str(exc),
                })
                if out_path.exists() and out_path.stat().st_size == 0:
                    out_path.unlink()
    return records


def scan_era5_land() -> list[dict]:
    era5_dir = RAW_DIR / "era5"
    era5_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(era5_dir.glob("era5_land_*_asia.zip")):
        parts = path.stem.replace("era5_land_", "").replace("_asia", "").split("_")
        records.append({
            "path": str(path),
            "status": "present",
            "bytes": path.stat().st_size,
            "name": path.name,
            "label": "_".join(parts[:-2]),
            "year": parts[-2] if len(parts) >= 2 else "",
            "month": parts[-1] if len(parts) >= 1 else "",
        })
    expected = [
        f"{label}_{year}_{month:02d}"
        for year, month in month_range(START_DATE, END_DATE)
        for label in ["temperature_mean", "dewpoint_mean", "precipitation_sum"]
    ]
    present = {
        f"{record['label']}_{record['year']}_{record['month']}"
        for record in records
    }
    for item in expected:
        if item not in present:
            records.append({"status": "missing", "name": f"era5_land_{item}_asia.zip"})
    return records


def write_portal_manifests() -> list[dict]:
    records = [
        {
            "dataset": "MODIS MOD11A2",
            "target_dir": str(RAW_DIR / "modis"),
            "status": "requires_large_nasa_cmr_granule_download",
            "source": "https://cmr.earthdata.nasa.gov/search/",
            "reason": "All Asia MOD11A2 8-day tiles for 13 months is a large multi-tile NASA download; Earthdata credentials can authenticate once granules are selected.",
        },
        {
            "dataset": "SRTM DEM",
            "target_dir": str(RAW_DIR / "srtm"),
            "status": "requires_large_nasa_cmr_granule_download",
            "source": "https://cmr.earthdata.nasa.gov/search/",
            "reason": "All Asia 30 m DEM coverage is many static granules and can be very large.",
        },
        {
            "dataset": "Dynamic World V1",
            "target_dir": str(RAW_DIR / "dynamic_world"),
            "status": "requires_google_earth_engine",
            "source": "https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1",
            "reason": "Bulk export for all Asia requires Earth Engine authentication and export tasks.",
        },
    ]
    for record in records:
        Path(record["target_dir"]).mkdir(parents=True, exist_ok=True)
    return records


def main() -> None:
    ensure_directories()
    session = requests.Session()
    session.headers.update({"User-Agent": "hantavirus-risk-raw-data-downloader/1.0"})

    max_records_env = os.getenv("GBIF_MAX_RECORDS", "").strip()
    max_records = int(max_records_env) if max_records_env else None

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "date_range": {"start": START_DATE, "end": END_DATE},
        "region": {"name": "Asia", "bbox": ASIA_BBOX},
        "downloads": {},
        "portal_required": write_portal_manifests(),
    }

    if os.getenv("SKIP_GBIF", "").strip() != "1":
        manifest["downloads"]["gbif"] = download_gbif_occurrences(session, max_records=max_records)
    if os.getenv("SKIP_CHIRPS", "").strip() != "1":
        manifest["downloads"]["chirps"] = download_chirps(session)
    if os.getenv("SKIP_ERA5", "").strip() != "1":
        if os.getenv("ERA5_SCAN_ONLY", "").strip() == "1":
            manifest["downloads"]["era5_land"] = scan_era5_land()
        else:
            manifest["downloads"]["era5_land"] = download_era5_land()

    out_path = RAW_DIR / "download_manifest_2025_05_to_2026_05.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
