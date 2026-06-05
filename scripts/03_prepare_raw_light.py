from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import RAW_LIGHT_DIR, RAW_LIGHT_ZIP, ensure_directories


def main() -> None:
    ensure_directories()
    if not RAW_LIGHT_ZIP.exists():
        summary_path = RAW_LIGHT_DIR / "raw_light_summary.json"
        if summary_path.exists():
            print(f"Raw-light zip not found at {RAW_LIGHT_ZIP}; using existing {summary_path}")
            return
        raise FileNotFoundError(
            f"Expected raw dataset zip at {RAW_LIGHT_ZIP}. "
            "Place hantavirus_risk_mapping_raw_light_may2025_may2026.zip in Downloads "
            "or data raw light.zip in the project root."
        )

    rows = []
    try:
        with zipfile.ZipFile(RAW_LIGHT_ZIP) as zf:
            for info in zf.infolist():
                rows.append({
                    "path": info.filename,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "is_dir": info.is_dir(),
                })
                normalized = Path(info.filename).as_posix()
                if normalized in {
                    "data/raw_light/README.md",
                    "data/raw_light/download_manifest_light_alternatives_2025_05_to_2026_05.json",
                    "raw_light/README.md",
                    "raw_light/download_manifest_light_alternatives_2025_05_to_2026_05.json",
                }:
                    target = RAW_LIGHT_DIR / Path(info.filename).name
                    target.write_bytes(zf.read(info))
    except zipfile.BadZipFile:
        summary_path = RAW_LIGHT_DIR / "raw_light_summary.json"
        if summary_path.exists():
            print(f"Raw-light zip central directory is damaged; using recovered {summary_path}")
            return
        raise

    inventory = pd.DataFrame(rows)
    inventory.to_csv(RAW_LIGHT_DIR / "raw_light_inventory.csv", index=False)

    manifest_path = RAW_LIGHT_DIR / "download_manifest_light_alternatives_2025_05_to_2026_05.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = {
        "source_zip": str(RAW_LIGHT_ZIP),
        "entry_count": len(rows),
        "date_range": manifest["date_range"],
        "region": manifest["region"],
        "available": {
            "era5_monthly": len([x for x in manifest["downloads"].get("era5_monthly", []) if x["status"] in {"downloaded", "already_exists"}]),
            "mod11c3_lst_months": len([x for x in manifest["downloads"].get("mod11c3_lst", []) if x["status"] == "downloaded"]),
            "mcd12c1_landcover": len([x for x in manifest["downloads"].get("mcd12c1_landcover", []) if x["status"] == "downloaded"]),
            "gmted2010": len([x for x in manifest["downloads"].get("gmted2010", []) if x["status"] in {"downloaded", "already_exists"}]),
        },
        "substitutions": {
            "CHIRPS_or_IMERG": "Unavailable/blocked; ERA5 total precipitation is used as rainfall feature.",
            "MODIS_MOD11A2": "Replaced by monthly MOD11C3 CMG LST.",
            "SRTM": "Replaced by GMTED2010 mean elevation.",
            "ESA_WorldCover": "Replaced by MCD12C1 2024 land cover because 2025 product is not published.",
        },
    }
    (RAW_LIGHT_DIR / "raw_light_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
