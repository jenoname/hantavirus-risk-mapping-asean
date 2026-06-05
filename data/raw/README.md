# Raw Data

For this raw-light version, the large dataset is expected at the project root:

- `data raw light.zip`

The pipeline inventories it into `data/raw_light/` using `scripts/03_prepare_raw_light.py`.

Original full-data filenames from the proposal:

- `gbif_rodent_occurrences_2025_05_to_2026_05.csv`
- `era5_climate_2025_05_to_2026_05.nc`
- `chirps_*.tif`
- `MOD11A2_*.hdf`
- `srtm_*.tif`
- `worldcover.tif`

In the provided raw-light bundle, ERA5 precipitation replaces CHIRPS/IMERG, MOD11C3 replaces MOD11A2, GMTED2010 replaces SRTM, and MCD12C1 replaces WorldCover.
