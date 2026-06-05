"""Download lighter alternative raw datasets for Asia, May 2025-May 2026."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.auth import AuthBase, _basic_auth_str

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import PROJECT_ROOT, START_DATE, END_DATE


RAW_LIGHT_DIR = PROJECT_ROOT / "data" / "raw_light"
ASIA_AREA = [82.0, 25.0, -11.0, 180.0]  # north, west, south, east
CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"


class EarthdataRedirectAuth(AuthBase):
    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password

    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        host = urlparse(request.url).hostname or ""
        if host.endswith("earthdata.nasa.gov"):
            request.headers["Authorization"] = _basic_auth_str(self.username, self.password)
        return request


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    months = []
    year, month = start_date.year, start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        months.append((year, month))
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "hantavirus-risk-light-alternative-downloader/1.0"})
    username = os.getenv("EARTHDATA_USERNAME", "").strip()
    password = os.getenv("EARTHDATA_PASSWORD", "").strip()
    if username and password:
        session.auth = EarthdataRedirectAuth(username, password)
    return session


def stream_download(url: str, out_path: Path, session: requests.Session, auth: bool = False) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return {"status": "already_exists", "url": url, "path": str(out_path), "bytes": out_path.stat().st_size}
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    headers = {}
    existing = tmp_path.stat().st_size if tmp_path.exists() else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"
    with session.get(url, stream=True, timeout=180, allow_redirects=True, headers=headers) as response:
        if response.status_code == 404:
            return {"status": "not_found", "url": url, "path": str(out_path)}
        if existing and response.status_code == 200:
            existing = 0
        response.raise_for_status()
        mode = "ab" if existing and response.status_code == 206 else "wb"
        with tmp_path.open(mode) as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    tmp_path.replace(out_path)
    return {"status": "downloaded", "url": url, "path": str(out_path), "bytes": out_path.stat().st_size}


def download_era5_monthly() -> list[dict]:
    try:
        import cdsapi
    except ImportError:
        return [{"status": "missing_dependency", "package": "cdsapi"}]

    url = os.getenv("CDSAPI_URL", "").strip()
    key = os.getenv("CDSAPI_KEY", "").strip()
    if not url or not key:
        return [{"status": "missing_credentials", "required_env": ["CDSAPI_URL", "CDSAPI_KEY"]}]

    out_dir = RAW_LIGHT_DIR / "era5_monthly"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "era5_monthly_single_levels_temp_dewpoint_precip_asia_2025_05_to_2026_05.zip"
    if out_path.exists() and out_path.stat().st_size > 0:
        return [{"status": "already_exists", "path": str(out_path), "bytes": out_path.stat().st_size}]

    months_by_year: dict[int, list[str]] = {}
    for year, month in month_range(START_DATE, END_DATE):
        months_by_year.setdefault(year, []).append(f"{month:02d}")

    client = cdsapi.Client(url=url, key=key, quiet=False, progress=True, sleep_max=60)
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "total_precipitation",
        ],
        "year": [str(year) for year in sorted(months_by_year)],
        "month": sorted({month for months in months_by_year.values() for month in months}),
        "time": ["00:00"],
        "area": ASIA_AREA,
        "data_format": "netcdf",
        "download_format": "zip",
    }
    try:
        client.retrieve("reanalysis-era5-single-levels-monthly-means", request, str(out_path))
        return [{"status": "downloaded", "path": str(out_path), "bytes": out_path.stat().st_size, "request": request}]
    except Exception as exc:
        if out_path.exists() and out_path.stat().st_size == 0:
            out_path.unlink()
        return [{"status": "failed", "path": str(out_path), "error": str(exc), "request": request}]


def cmr_temporal(year: int, month: int) -> str:
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)
    end_dt = datetime.combine(end, datetime.min.time()) - timedelta(seconds=1)
    return f"{start.isoformat()}T00:00:00Z,{end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}"


def cmr_download_links(session: requests.Session, short_name: str, version: str, temporal: str, limit: int = 10) -> list[str]:
    params = {
        "short_name": short_name,
        "version": version,
        "temporal": temporal,
        "page_size": limit,
        "sort_key": "-start_date",
    }
    response = requests.get(CMR_GRANULES_URL, params=params, timeout=120)
    response.raise_for_status()
    links = []
    for entry in response.json().get("feed", {}).get("entry", []):
        for link in entry.get("links", []):
            href = link.get("href", "")
            if not href or not href.lower().startswith("http"):
                continue
            title = (link.get("title") or "").lower()
            rel = link.get("rel", "")
            if "inherited" in title or "opendap" in title or "browse" in title:
                continue
            if "data#" in rel or "download" in title or href.lower().endswith((".hdf", ".hdf5", ".nc4", ".he5")):
                links.append(href)
    return list(dict.fromkeys(links))


def filename_from_url(url: str) -> str:
    return Path(urlparse(url).path).name or "download.bin"


def download_cmr_monthly_product(session: requests.Session, short_name: str, version: str, out_subdir: str) -> list[dict]:
    records = []
    out_dir = RAW_LIGHT_DIR / out_subdir
    try:
        import earthaccess
        earthaccess.login(strategy="environment", persist=False)
    except Exception as exc:
        return [{"status": "earthaccess_login_failed", "short_name": short_name, "error": str(exc)}]

    for year, month in month_range(START_DATE, END_DATE):
        start, end = cmr_temporal(year, month).split(",")
        results = earthaccess.search_data(
            short_name=short_name,
            version=version,
            temporal=(start.replace("T00:00:00Z", ""), end.replace("T23:59:59Z", "")),
            count=1,
        )
        if not results:
            records.append({"status": "no_granule", "short_name": short_name, "year": year, "month": month})
            continue
        try:
            paths = earthaccess.download(results, local_path=str(out_dir), threads=1)
            for path in paths:
                path = Path(path)
                record = {"status": "downloaded", "path": str(path), "bytes": path.stat().st_size}
                record.update({"short_name": short_name, "version": version, "year": year, "month": month})
                records.append(record)
        except Exception as exc:
            records.append({"status": "failed", "short_name": short_name, "version": version, "year": year, "month": month, "error": str(exc)})
    return records


def download_mcd12c1_landcover(session: requests.Session) -> list[dict]:
    records = []
    out_dir = RAW_LIGHT_DIR / "mcd12c1_landcover"
    try:
        import earthaccess
        earthaccess.login(strategy="environment", persist=False)
    except Exception as exc:
        return [{"status": "earthaccess_login_failed", "short_name": "MCD12C1", "error": str(exc)}]

    for year in [2024, 2025]:
        results = earthaccess.search_data(
            short_name="MCD12C1",
            version="061",
            temporal=(f"{year}-01-01", f"{year}-12-31"),
            count=1,
        )
        if not results:
            records.append({"status": "no_granule", "short_name": "MCD12C1", "year": year})
            continue
        try:
            paths = earthaccess.download(results, local_path=str(out_dir), threads=1)
            for path in paths:
                path = Path(path)
                record = {"status": "downloaded", "path": str(path), "bytes": path.stat().st_size}
                record.update({"short_name": "MCD12C1", "version": "061", "year": year})
                records.append(record)
        except Exception as exc:
            records.append({"status": "failed", "short_name": "MCD12C1", "version": "061", "year": year, "error": str(exc)})
    return records


def download_gmted2010(session: requests.Session) -> list[dict]:
    urls = [
        "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/topo/downloads/GMTED/Grid_ZipFiles/mn30_grd.zip",
    ]
    records = []
    for url in urls:
        out_path = RAW_LIGHT_DIR / "gmted2010" / filename_from_url(url)
        try:
            records.append(stream_download(url, out_path, session))
        except Exception as exc:
            records.append({"status": "failed", "url": url, "path": str(out_path), "error": str(exc)})
    return records


def main() -> None:
    RAW_LIGHT_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "date_range": {"start": START_DATE, "end": END_DATE},
        "region": {"name": "Asia", "area_nwse": ASIA_AREA},
        "downloads": {},
    }

    if os.getenv("SKIP_ERA5_MONTHLY", "").strip() != "1":
        manifest["downloads"]["era5_monthly"] = download_era5_monthly()
    if os.getenv("SKIP_IMERG", "").strip() != "1":
        manifest["downloads"]["imerg_monthly"] = download_cmr_monthly_product(session, "GPM_3IMERGM", "07", "imerg_monthly")
    if os.getenv("SKIP_MOD11C3", "").strip() != "1":
        manifest["downloads"]["mod11c3_lst"] = download_cmr_monthly_product(session, "MOD11C3", "061", "mod11c3_lst")
    if os.getenv("SKIP_MCD12C1", "").strip() != "1":
        manifest["downloads"]["mcd12c1_landcover"] = download_mcd12c1_landcover(session)
    if os.getenv("SKIP_GMTED", "").strip() != "1":
        manifest["downloads"]["gmted2010"] = download_gmted2010(session)

    out_path = RAW_LIGHT_DIR / "download_manifest_light_alternatives_2025_05_to_2026_05.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
