from __future__ import annotations

import runpy
import os
from pathlib import Path


PIPELINE = [
    "01_download_gbif.py",
    "02_download_climate.py",
    "download_raw_asia_2025_2026.py",
    "download_light_alternatives_asia_2025_2026.py",
    "03_prepare_raw_light.py",
    "04_preprocess.py",
    "05_eda.py",
    "06_model_rf.py",
    "06_model_cnn.py",
    "06_ensemble.py",
    "07_evaluate.py",
    "07_shap_gradcam.py",
    "08_risk_map.py",
]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    run_downloads = os.getenv("RUN_REAL_DOWNLOADS", "").strip() == "1"
    for script in PIPELINE:
        if script.startswith("download_") and not run_downloads:
            print(f"\nSkipping {script} (set RUN_REAL_DOWNLOADS=1 to fetch/update raw sources)")
            continue
        print(f"\nRunning {script}")
        runpy.run_path(str(script_dir / script), run_name="__main__")


if __name__ == "__main__":
    main()
