from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import RAW_DIR, RAW_LIGHT_ZIP, START_DATE, END_DATE, ensure_directories


def main() -> None:
    ensure_directories()
    manifest = {
        "date_range": {"start": START_DATE, "end": END_DATE},
        "datasets": {
            "ERA5": "monthly temperature, precipitation, dewpoint, humidity",
            "CHIRPS": "daily precipitation GeoTIFF sampled/aggregated to monthly",
            "MODIS_MOD11A2": "8-day LST HDF tiles converted from DN to Celsius",
            "SRTM": "30 m elevation and derived slope",
            "ESA_WorldCover": "10 m land-cover fractions per grid cell",
        },
        "raw_light_zip": str(RAW_LIGHT_ZIP),
        "note": "The new raw_light dataset zip is inventoried in 03_prepare_raw_light.py. Heavy NetCDF/HDF raster parsing is optional and requires xarray/netCDF4/h5py/pyhdf/rasterio.",
    }
    out = RAW_DIR / "download_manifest_2025_05_to_2026_05.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
