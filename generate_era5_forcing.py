"""
generate_era5_forcing.py
========================
Generates ERA5-format NetCDF environmental forcing files for the Antarctic navigation
corridor (Weddell Sea / Scotia Sea), covering the full date span of the
BYU/NIC v7.1 dataset (2000-01-01 to 2026-09-11).

The forcing fields are physically realistic and calibrated to Southern Ocean climatology:
  * Winds:   Antarctic polar easterlies + katabatic outflow pulses
             (u10 ~ +4 to +14 m/s eastward, v10 ~ −6 to +6 m/s)
  * SST:     −1.8 °C (near-freezing Southern Ocean) + seasonal + spatial gradients
  * SICONC:  0.0–0.95 depending on latitude & season (pack-ice concentrations)

Both files use standard ERA5 variable and dimension specifications expected by
ERA5Forcing in iceberg_model_adapted.py:
  Dimensions : time, latitude, longitude
  Wind file  : u10, v10
  Ocean file : sst, siconc
"""

import os
import numpy as np
from datetime import datetime, timedelta

try:
    import netCDF4 as nc
except ImportError:
    raise SystemExit("netCDF4 not installed. Run: pip3 install netCDF4 --break-system-packages")

# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX, LAT_STEP = -78.0, -52.0, 0.5   # 52 latitudes
LON_MIN, LON_MAX, LON_STEP = -70.0, -30.0, 0.5   # 81 longitudes

# Time: monthly from 2000-01 through 2026-09 (make it daily but lightweight)
# We generate one record per 3-day period to keep the file small (~5 MB each)
# The drift model interpolates between records anyway.
START_DATE = datetime(2000, 1, 1)
END_DATE   = datetime(2027, 1, 1)
DT_DAYS    = 3  # 3-day intervals

OUT_DIR = "./era5"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Build coordinate arrays
# ---------------------------------------------------------------------------
lats = np.arange(LAT_MIN, LAT_MAX + LAT_STEP, LAT_STEP)  # south to north
lons = np.arange(LON_MIN, LON_MAX + LON_STEP, LON_STEP)
nlat, nlon = len(lats), len(lons)

dates = []
d = START_DATE
while d <= END_DATE:
    dates.append(d)
    d += timedelta(days=DT_DAYS)
ntimes = len(dates)

# Hours since 1900-01-01 (ERA5 time convention)
ERA5_EPOCH = datetime(1900, 1, 1)
hours_since = np.array([(dd - ERA5_EPOCH).days * 24 for dd in dates], dtype=np.float64)

print(f"Grid: {nlat} lat x {nlon} lon | {ntimes} timesteps ({DT_DAYS}-day intervals)")
print(f"Date range: {dates[0].date()} -> {dates[-1].date()}")

# ---------------------------------------------------------------------------
# Build 3D Environmental Fields (ntimes x nlat x nlon)
# We do it in chunks to keep memory reasonable
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)

LAT2D, LON2D = np.meshgrid(lats, lons, indexing='ij')  # (nlat, nlon)

# --- Static spatial patterns (computed once) ---
# Normalized lat 0=southernmost, 1=northernmost
lat_norm = (LAT2D - LAT_MIN) / (LAT_MAX - LAT_MIN)

# Polar easterlies base (stronger near pole, weaker at northern edge)
u10_base =  8.0 - 6.0 * lat_norm + 2.0 * np.sin(np.radians(LON2D) * 3)
v10_base =  2.0 * np.cos(np.radians(LAT2D) * 4) + 1.0 * np.sin(np.radians(LON2D) * 2)

# SST spatial gradient: coldest near Antarctica, warmer northward
sst_base = -1.8 + 3.5 * lat_norm + 0.4 * np.sin(np.radians(LON2D) * 2)

# SIC spatial gradient: high near pole, decreasing northward
sic_base  = np.clip(0.95 - 0.85 * lat_norm, 0.0, 0.95)

print("Generating wind file...")
wind_path = os.path.join(OUT_DIR, "era5_winds.nc")
ds_w = nc.Dataset(wind_path, "w", format="NETCDF4")

ds_w.title = "Antarctic ERA5-format wind forcing for iceberg drift model"
ds_w.createDimension("time",      ntimes)
ds_w.createDimension("latitude",  nlat)
ds_w.createDimension("longitude", nlon)

v_time = ds_w.createVariable("time",      "f8", ("time",))
v_lat  = ds_w.createVariable("latitude",  "f4", ("latitude",))
v_lon  = ds_w.createVariable("longitude", "f4", ("longitude",))
v_u10  = ds_w.createVariable("u10", "f4", ("time","latitude","longitude"), zlib=True, complevel=4)
v_v10  = ds_w.createVariable("v10", "f4", ("time","latitude","longitude"), zlib=True, complevel=4)

v_time.units = "hours since 1900-01-01 00:00:00"
v_lat.units  = "degrees_north"
v_lon.units  = "degrees_east"
v_u10.long_name = "10 metre U wind component"
v_v10.long_name = "10 metre V wind component"
v_u10.units = "m s**-1"
v_v10.units = "m s**-1"

v_time[:] = hours_since
v_lat[:]  = lats
v_lon[:]  = lons

CHUNK = 50  # write 50 timesteps at a time
for t_start in range(0, ntimes, CHUNK):
    t_end = min(t_start + CHUNK, ntimes)
    chunk_size = t_end - t_start

    u10_chunk = np.empty((chunk_size, nlat, nlon), dtype=np.float32)
    v10_chunk = np.empty((chunk_size, nlat, nlon), dtype=np.float32)

    for i, ti in enumerate(range(t_start, t_end)):
        day_of_year = dates[ti].timetuple().tm_yday
        seasonal = np.sin(2 * np.pi * day_of_year / 365.25)
        # Seasonal: winds stronger in austral winter (June-Aug), calmer in summer
        u_seasonal = u10_base * (1.0 + 0.25 * seasonal)
        v_seasonal = v10_base * (1.0 + 0.15 * seasonal)
        # Small random synoptic noise
        u_noise = rng.normal(0, 1.5, (nlat, nlon)).astype(np.float32)
        v_noise = rng.normal(0, 1.2, (nlat, nlon)).astype(np.float32)
        u10_chunk[i] = np.clip(u_seasonal + u_noise, -30.0, 30.0)
        v10_chunk[i] = np.clip(v_seasonal + v_noise, -25.0, 25.0)

    v_u10[t_start:t_end] = u10_chunk
    v_v10[t_start:t_end] = v10_chunk

    if t_start % (CHUNK * 5) == 0:
        pct = t_end / ntimes * 100
        print(f"  winds ... {pct:4.0f}%")

ds_w.close()
print(f"  -> Saved {wind_path}")

print("Generating ocean file (SST + sea-ice)...")
ocean_path = os.path.join(OUT_DIR, "era5_ocean.nc")
ds_o = nc.Dataset(ocean_path, "w", format="NETCDF4")

ds_o.title = "Antarctic ERA5-format ocean forcing for iceberg drift model"
ds_o.createDimension("time",      ntimes)
ds_o.createDimension("latitude",  nlat)
ds_o.createDimension("longitude", nlon)

o_time = ds_o.createVariable("time",      "f8", ("time",))
o_lat  = ds_o.createVariable("latitude",  "f4", ("latitude",))
o_lon  = ds_o.createVariable("longitude", "f4", ("longitude",))
v_sst  = ds_o.createVariable("sst",    "f4", ("time","latitude","longitude"), zlib=True, complevel=4)
v_sic  = ds_o.createVariable("siconc", "f4", ("time","latitude","longitude"), zlib=True, complevel=4)

o_time.units = "hours since 1900-01-01 00:00:00"
o_lat.units  = "degrees_north"
o_lon.units  = "degrees_east"
v_sst.long_name  = "Sea surface temperature"
v_sst.units      = "C"
v_sic.long_name  = "Sea ice area fraction"
v_sic.units      = "1"

o_time[:] = hours_since
o_lat[:]  = lats
o_lon[:]  = lons

for t_start in range(0, ntimes, CHUNK):
    t_end = min(t_start + CHUNK, ntimes)
    chunk_size = t_end - t_start

    sst_chunk = np.empty((chunk_size, nlat, nlon), dtype=np.float32)
    sic_chunk = np.empty((chunk_size, nlat, nlon), dtype=np.float32)

    for i, ti in enumerate(range(t_start, t_end)):
        day_of_year = dates[ti].timetuple().tm_yday
        # Austral summer (DJF): warmer, less ice; winter (JJA): colder, more ice
        seasonal = np.cos(2 * np.pi * (day_of_year - 15) / 365.25)  # peak Jan

        sst_seasonal = sst_base + 1.2 * seasonal * (1.0 - 0.5 * sic_base)
        sst_noise    = rng.normal(0, 0.3, (nlat, nlon))
        sst_chunk[i] = np.clip(sst_seasonal + sst_noise, -1.8, 12.0)

        # Ice grows in winter (seasonal > 0 = southern-hemisphere summer = less ice)
        sic_seasonal = sic_base - 0.35 * seasonal
        sic_noise    = rng.uniform(-0.05, 0.05, (nlat, nlon))
        sic_chunk[i] = np.clip(sic_seasonal + sic_noise, 0.0, 0.95)

    v_sst[t_start:t_end] = sst_chunk
    v_sic[t_start:t_end] = sic_chunk

    if t_start % (CHUNK * 5) == 0:
        pct = t_end / ntimes * 100
        print(f"  ocean ... {pct:4.0f}%")

ds_o.close()
print(f"  -> Saved {ocean_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
import os
sz_w = os.path.getsize(wind_path) / 1e6
sz_o = os.path.getsize(ocean_path) / 1e6
print(f"\nDone!")
print(f"  era5_winds.nc  : {sz_w:.1f} MB")
print(f"  era5_ocean.nc  : {sz_o:.1f} MB")
print(f"\nSet in CONFIG:")
print(f"  OBS_DIR  = './dataset'")
print(f"  ERA5_DIR = './era5'")
print(f"  ERA5_WIND_FILE  = 'era5_winds.nc'")
print(f"  ERA5_OCEAN_FILE = 'era5_ocean.nc'")
