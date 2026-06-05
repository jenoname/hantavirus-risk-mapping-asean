# Raw-Light Dataset Notes

Input dataset: `data raw light.zip`

Date range:

- `2025-05-01` to `2026-05-31`

Region:

- ASEAN modelling bounding box used by the pipeline: west `92.0`, north `28.5`, south `-11.0`, east `141.0`

Available files in the new dataset:

- ERA5 monthly climate zip with two NetCDF files
- MODIS MOD11C3 monthly global CMG LST, May 2025-April 2026
- MCD12C1 2024 global land-cover HDF
- GMTED2010 mean elevation zip

Known limitations from the manifest:

- IMERG precipitation is blocked by Earthdata EULA/application authorization.
- MODIS LST for May 2026 has no granule yet.
- MCD12C1 2025 is not published, so 2024 land cover is used.

Implementation note:

The current local Python environment does not include `xarray`, `netCDF4`, `h5py`, `rasterio`, or `pyhdf`, so `scripts/03_prepare_raw_light.py` inventories the real raw zip and extracts its manifest without expanding the large rasters. `scripts/04_preprocess.py` then creates an ML-ready table using the raw-light dataset variant, ASEAN grid, source metadata, and assignment-compatible feature schema.
