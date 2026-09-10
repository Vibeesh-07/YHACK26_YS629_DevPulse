"""
iceberg_model_adapted.py
=========================================================================
Adapted from: England, Wagner & Eisenman (2020, Science Advances)
              "Modeling the Breakup of Tabular Icebergs"
              (itself building on Wagner, Dell & Eisenman 2017, WDE17)

WHAT CHANGED FROM THE ORIGINAL SCRIPT AND WHY
-------------------------------------------------------------------------
The original script assumed two data sources that are no longer available
here:
  1. ECCO2 ocean-current / ERA5 atmosphere / SST / sea-ice NetCDF files,
     read with the discontinued `PyNIO` ("Nio") library, from a hardcoded
     local Mac path.
  2. A MATLAB `.mat` file of synthetic, evenly-spaced seeding locations
     ("Group4_Seed") used to initialise 1000 identical parent icebergs.

What we actually have is:
  A. A real observational dataset of ~190 named Antarctic icebergs
     (`stats_database_v7.1/*.csv`, e.g. A-68, A-68A, B-15Z, C-18B, D-26,
     UK323...), each a daily-ish time series of
     `date, date_gap, disp, flags, lat, lon, mask, size, vel_angle`
     (date = Julian YYYYDDD, disp = displacement in km over date_gap
     days, size = plan-view area in km^2, mask = 0 open ocean /
     1-2 land-fast or grounded, vel_angle = drift-heading in radians).
  B. ERA5 reanalysis (the user supplies the NetCDF files: 10 m winds
     u10/v10, sea-surface temperature, sea-ice concentration).

This script is rebuilt so it uses BOTH at once, instead of ECCO2 + a
synthetic seed file:

  * ERA5 (via `netCDF4`, not the unmaintained `Nio`) supplies the
    atmospheric forcing (winds) and SST / sea-ice concentration exactly
    as WDE17/England et al. require.
  * There is no ocean-current reanalysis product supplied, so instead
    of leaving `u_ocn, v_ocn = 0` (which would break the drift physics,
    since the ocean term dominates equation 6 of WDE17), we *estimate*
    a monthly ocean-current climatology empirically, straight from the
    real dataset: for every daily observation of a real iceberg we
    already know its observed drift velocity (`disp`, `vel_angle`); we
    subtract the ERA5-wind-driven component predicted by the WDE17
    slab equations for an iceberg of that size at that place/time, and
    what's left over is treated as the local ocean-driven residual.
    Averaging those residuals onto a lat/lon/month grid gives a
    data-derived ocean-current field, gridded consistently with ERA5.
    This is the same logic used in the iceberg-drift literature to
    infer near-surface currents from iceberg trajectories (e.g. Silva
    et al. 2006), just applied here to build the forcing field the
    model needs.
  * Parent icebergs are seeded from the REAL starting position, start
    date and (area-derived) starting dimensions of each named iceberg
    in the dataset, instead of 1000 synthetic identical seeds.
  * A validation routine reruns the model from each real iceberg's
    observed starting point and compares the simulated trajectory and
    area decay against what was actually observed, which the original
    script had no way to do (it never saw real iceberg data at all).

Everything else — the drift equations (WDE17 eqs 6-9), the melt terms
(England et al. 2020 eqs 2-3 / appendix), the Poisson breakup scheme
(eq. 4), and the recursive child-iceberg bookkeeping — is preserved
from the original script, just re-plumbed onto the new data sources and
cleaned of hardcoded absolute paths / the removed `Nio`/`scipy.io`
dependencies.

REQUIREMENTS FOR THE USER TO ACTUALLY RUN THIS
-------------------------------------------------------------------------
1. The observed-iceberg CSV folder (already have it: stats_database_v7.1)
2. ERA5 NetCDF files with (at minimum) these variables, on a regular
   lat/lon grid, covering the years you want to simulate:
       u10, v10        - 10 m wind components               [m/s]
       sst             - sea surface temperature             [deg C or K]
       siconc          - sea ice area fraction                [0-1]
   (ERA5 single-levels products provide all four.) Point EDIT_ME_ERA5_DIR
   below at that folder, and confirm the variable names below match
   your download (ERA5 sometimes ships as t2m/sst rather than sst; use
   `ncdump -h file.nc` to check once and adjust CONFIG accordingly).
3. `pip install netCDF4 numpy pandas scipy --break-system-packages`
=========================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

try:
    import netCDF4 as nc
except ImportError:
    nc = None  # allow the observation-processing / climatology parts to
               # be exercised even where netCDF4 isn't installed yet


# ============================================================================================
# CONFIG  -- edit these for your machine
# ============================================================================================

CONFIG = {
    # Folder holding the real iceberg-tracking CSVs (A-68.csv, B-15Z.csv, ...)
    "OBS_DIR": "./dataset",

    # Folder holding your ERA5 NetCDF downloads
    "ERA5_DIR": "./era5",
    "ERA5_WIND_FILE": "era5_winds.nc",     # variables: u10, v10, time, latitude, longitude
    "ERA5_OCEAN_FILE": "era5_ocean.nc",    # variables: sst, siconc, time, latitude, longitude

    # ERA5 variable names (adjust if your download uses different names)
    "VAR_U10": "u10",
    "VAR_V10": "v10",
    "VAR_SST": "sst",
    "VAR_SIC": "siconc",
    "VAR_LAT": "latitude",
    "VAR_LON": "longitude",
    "VAR_TIME": "time",

    "OUTPUT_DIR": "./Output_adapted",

    # Grid resolution (degrees) of the empirical ocean-current climatology
    # we build from the observed dataset. 2 deg is a reasonable compromise
    # between coverage (few icebergs) and resolving mesoscale currents.
    "OCEAN_CLIM_RES_DEG": 2.0,

    # Quality control on the observed dataset
    # flags is a BITMASK (not a quality score). Set to 255 to accept all flag combinations.
    # Specific bad bits can be excluded via EXCLUDE_FLAG_BITS (see _load_one_iceberg).
    # Bit meanings: 0=low-res, 1=interpolated, 2=near-land, 6=SAR-derived, 7=manual-edit
    "MAX_QC_FLAG": 255,     # accept all flag values (flags field is a bitmask, not a score)
    "EXCLUDE_FLAG_BITS": 0, # bitmask of flag bits to REJECT (0 = reject nothing extra)
    "VALID_MASK_VALUES": (0,),  # 0 = open ocean / free-drifting in this dataset

    "MIN_VOLUME": 6e6,      # m^3, stop tracking below this (as in the original script)
    "NDAYS": 5000,          # max simulated days per iceberg
}

# ============================================================================================
# PHYSICAL CONSTANTS (unchanged from England, Wagner & Eisenman 2020)
# ============================================================================================

R = 6378e3            # Radius of earth [m]
Om = 7.2921e-5         # Rotation rate of earth [rad/s]
g = 9.81               # Acceleration due to gravity [m/s^2]

rhow = 1027            # Density of water [kg/m^3]
rhoa = 1.2             # Density of air [kg/m^3]
rhoi = 850             # Density of shelf ice [kg/m^3], Silva et al. (2006)
drho = rhow - rhoi
nu = 0.33              # Poisson ratio of ice

Cw = 0.9               # Bulk coefficient of water, Bigg et al. (1997)
Ca = 1.3               # Bulk coefficient of air, Bigg et al. (1997)

Ti = -4
a1 = 8.7e-6
a2 = 5.8e-7
b1 = 8.8e-8
b2 = 1.5e-8
c_melt = 6.7e-6

E = 0.1e9              # Young's modulus
epsilon = 3            # child iceberg aspect ratio (W = epsilon * L)

ff = lambda lat: 2 * Om * (np.sin(lat * np.pi / 180))                       # Coriolis
ga = np.sqrt(rhoa * drho / rhow / rhoi * Ca / Cw)                           # gamma, eq 7
S = lambda l, w: l * w / (l + w)                                           # harmonic mean length
La = lambda u, lat, S_: Cw * ga / ff(lat) * u / S_ / np.pi                  # Lambda, eq 9
alpha = lambda La_: 1 / (2 * La_ ** 3) * (np.sqrt(1 + 4 * La_ ** 4) - 1)    # alpha, eq 8
beta = lambda La_, lat: ff(lat) / abs(ff(lat)) * 1 / (np.sqrt(2) * La_ ** 3) \
    * np.sqrt(np.abs((1 + La_ ** 4) * np.sqrt(1 + 4 * La_ ** 4) - 3 * La_ ** 4 - 1))
beta_approx = lambda La_, lat: ff(lat) / abs(ff(lat)) * (La_ ** 3 - 1.5 * La_ ** 7)
bend_B = lambda h0: (E * h0 ** 3) / 12 / (1 - nu ** 2)                      # bending stiffness
buoy_len = lambda Bb: (Bb / g / rhow) ** 0.25                               # buoyancy length (Wagner 2014)


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    return int(np.argmin((nodes - node) ** 2))


# ============================================================================================
# PART A -- LOAD THE REAL OBSERVED ICEBERG DATASET
# ============================================================================================

def _julian_to_datetime(yyyyddd):
    """Convert the dataset's YYYYDDD Julian-day date format to a real date."""
    yyyyddd = int(yyyyddd)
    year = yyyyddd // 1000
    doy = yyyyddd % 1000
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)


def _load_one_iceberg(path, max_qc_flag, valid_mask_values, exclude_flag_bits=0):
    """Read and clean a single iceberg CSV from the observed dataset."""
    berg_id = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path)

    df["berg_id"] = berg_id
    df["datetime"] = df["date"].apply(_julian_to_datetime)

    # Fix 1: Negative size values are instrument/interpolation artifacts — drop them
    #         explicitly with a warning so they are visible.
    neg_size = (df["size"] < 0).sum()
    if neg_size > 0:
        print(f"  [preprocess] {berg_id}: dropping {neg_size} rows with negative size values")
        df = df[df["size"] >= 0].copy()

    # Fix 2: Quality control
    #  - size == 0 means no area measurement — not usable for physics
    #  - mask != 0 means land-fast / grounded / otherwise not freely drifting
    #  - flags: MAX_QC_FLAG=255 accepts all flag values (flags is a bitmask, not a score).
    #           EXCLUDE_FLAG_BITS allows rejecting specific bitmask bits if needed.
    flag_mask_ok = (df["flags"] & exclude_flag_bits) == 0 if exclude_flag_bits else True
    df = df[
        (df["size"] > 0)
        & (df["mask"].isin(valid_mask_values))
        & (df["flags"] <= max_qc_flag)
        & flag_mask_ok
    ].copy()

    df = df.sort_values("datetime").reset_index(drop=True)

    # Observed drift velocity implied by displacement + heading.
    # disp is displacement (km) accumulated over date_gap days.
    with np.errstate(divide="ignore", invalid="ignore"):
        speed_kmpd = np.where(df["date_gap"] > 0, df["disp"] / df["date_gap"], np.nan)
    speed_mps = speed_kmpd * 1000.0 / 86400.0
    df["u_obs"] = speed_mps * np.cos(df["vel_angle"])
    df["v_obs"] = speed_mps * np.sin(df["vel_angle"])

    return df


def load_observed_icebergs(obs_dir, max_qc_flag=255, valid_mask_values=(0,),
                            exclude_flag_bits=0):
    """
    Load every iceberg CSV in `obs_dir`.

    Parameters
    ----------
    obs_dir          : folder containing the iceberg tracking CSVs
    max_qc_flag      : upper bound on the flags field. Default 255 = accept all
                       (flags is a bitmask, not a quality score)
    valid_mask_values: tuple of mask values to keep. 0 = open ocean / freely drifting.
    exclude_flag_bits: bitmask of flag bits to REJECT regardless of max_qc_flag.
                       E.g. exclude_flag_bits=2 drops interpolated-position rows.

    Returns
    -------
    obs_all : concatenated, cleaned DataFrame of every observation
              (used to build the empirical ocean-current climatology)
    seeds   : one row per iceberg = its first valid observation (size > 0),
              used to seed the parent-iceberg simulations with REAL
              starting positions/dates/sizes instead of synthetic seeds.
    """
    files = sorted(glob.glob(os.path.join(obs_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No iceberg CSVs found in {obs_dir}")

    frames = []
    skipped_no_data = []
    for f in files:
        try:
            d = _load_one_iceberg(f, max_qc_flag, valid_mask_values, exclude_flag_bits)
            if len(d):
                frames.append(d)
            else:
                # Fix 3: Berg has no usable rows after cleaning — log and skip gracefully
                berg_id = os.path.splitext(os.path.basename(f))[0]
                skipped_no_data.append(berg_id)
        except Exception as e:
            print(f"  [skip] {f}: {e}")

    if skipped_no_data:
        print(f"  [preprocess] {len(skipped_no_data)} icebergs skipped (no valid open-ocean "
              f"rows with size > 0): {', '.join(skipped_no_data[:10])}"
              + (" ..." if len(skipped_no_data) > 10 else ""))

    obs_all = pd.concat(frames, ignore_index=True)

    # Fix 4: Build seeds only from rows that have a measurable size.
    #         size=0 rows are retained in obs_all for track continuity but
    #         cannot be used as simulation starting points.
    obs_with_size = obs_all[obs_all["size"] > 0]
    seeds = (obs_with_size.sort_values("datetime")
                          .groupby("berg_id", as_index=False)
                          .first())
    print(f"  [preprocess] {seeds['berg_id'].nunique()} icebergs have valid seeds "
          f"({len(obs_all)} total clean rows, "
          f"{len(obs_with_size)} with measured size > 0)")
    return obs_all, seeds


# Empirical length/width/height scaling used only to turn an observed
# plan-view AREA (km^2) into the (L, W, H) triple the physical model
# needs. Aspect ratio (L:W) is fixed at 3:2, consistent with the
# child-iceberg aspect ratio elsewhere in this model family. Thickness
# is looked up from the same tabular-iceberg size classes used in the
# original script (bigger icebergs -> thicker), since area alone can't
# tell you draft/freeboard.
_THICKNESS_TABLE_L_M = np.array([690, 1220, 2180, 3870, 6890, 12250, 21780, 38730])
_THICKNESS_TABLE_H_M = np.array([175, 200, 225, 250, 275, 300, 325, 350])


def area_to_dimensions(area_km2, aspect=1.5):
    """area (km^2) -> (L, W, H) in metres."""
    area_m2 = area_km2 * 1.0e6
    W = np.sqrt(area_m2 / aspect)
    L = aspect * W
    H = float(np.interp(L, _THICKNESS_TABLE_L_M, _THICKNESS_TABLE_H_M))
    return float(L), float(W), H


# ============================================================================================
# PART B -- LOAD ERA5 (replaces PyNIO/Nio + ECCO2)
# ============================================================================================

class ERA5Forcing:
    """
    Thin wrapper around the ERA5 NetCDF files, read with netCDF4 (an
    actively maintained, pip-installable library) instead of the
    discontinued `Nio`. Exposes the same style of lookup the original
    model used: nearest-neighbour in space, linear in time.
    """

    def __init__(self, cfg):
        if nc is None:
            raise ImportError("netCDF4 is required to load ERA5 forcing: "
                               "pip install netCDF4 --break-system-packages")

        wind_path = os.path.join(cfg["ERA5_DIR"], cfg["ERA5_WIND_FILE"])
        ocean_path = os.path.join(cfg["ERA5_DIR"], cfg["ERA5_OCEAN_FILE"])

        fw = nc.Dataset(wind_path)
        fo = nc.Dataset(ocean_path)

        self.lat = fw.variables[cfg["VAR_LAT"]][:]
        self.lon = fw.variables[cfg["VAR_LON"]][:]
        self.time = fw.variables[cfg["VAR_TIME"]][:]   # hours (or days) since epoch, per file's units

        self.u_atm = fw.variables[cfg["VAR_U10"]][:, :, :]
        self.v_atm = fw.variables[cfg["VAR_V10"]][:, :, :]
        self.sst = fo.variables[cfg["VAR_SST"]][:, :, :]
        self.sic = fo.variables[cfg["VAR_SIC"]][:, :, :]

        # ERA5 SST is often in Kelvin; the melt equations assume degC.
        if np.nanmean(self.sst) > 100:
            self.sst = self.sst - 273.15
        self.sst = np.where(self.sst < -4.0, -4.0, self.sst)

        # Land mask: ERA5 marks land as masked/NaN in ocean variables.
        self.land_mask = np.isnan(np.asarray(self.sic[0])).astype(int)

        self.minlat, self.maxlat = float(np.min(self.lat)), float(np.max(self.lat))
        self.minlon, self.maxlon = float(np.min(self.lon)), float(np.max(self.lon))

        fw.close()
        fo.close()

    def sample(self, lat_pt, lon_pt, t_index_float):
        """Nearest-neighbour in space, linear interpolation in time (as original script)."""
        yi = closest_node(lat_pt, self.lat)
        xi = closest_node(lon_pt, self.lon)

        ti1 = max(0, min(int(np.floor(t_index_float)), len(self.time) - 1))
        ti2 = min(ti1 + 1, len(self.time) - 1)
        if ti1 == ti2:
            dt1, dt2 = 0.0, 1.0
        else:
            dt1 = t_index_float - ti1
            dt2 = ti2 - t_index_float
            if dt1 + dt2 == 0:
                dt1, dt2 = 1.0, 0.0

        ua = self.u_atm[ti1, yi, xi] * dt2 + self.u_atm[ti2, yi, xi] * dt1
        va = self.v_atm[ti1, yi, xi] * dt2 + self.v_atm[ti2, yi, xi] * dt1
        sst = self.sst[ti1, yi, xi] * dt2 + self.sst[ti2, yi, xi] * dt1
        sic = self.sic[ti1, yi, xi] * dt2 + self.sic[ti2, yi, xi] * dt1
        return float(ua), float(va), float(sst), float(sic)


# ============================================================================================
# PART C -- BUILD AN EMPIRICAL OCEAN-CURRENT CLIMATOLOGY FROM THE OBSERVED DATASET + ERA5
# ============================================================================================

def build_empirical_ocean_climatology(obs_all, era5, cfg):
    """
    For every high-quality daily observation of a real iceberg's drift
    velocity, predict the ERA5-wind-driven component using the WDE17
    slab equations for an iceberg of that observed size, then take the
    residual (observed minus wind-driven) as a sample of the local
    ocean current. Bin residuals onto a (lon, lat, month) grid and
    average -> a monthly ocean-current climatology that stands in for
    the missing ECCO2 product, built directly from your data + ERA5.

    Returns three 3-D arrays (month, lat_bin, lon_bin): u_ocn_clim,
    v_ocn_clim, and the lat/lon bin edges, plus a lookup function.
    """
    res = cfg["OCEAN_CLIM_RES_DEG"]
    lon_bins = np.arange(-180, 180 + res, res)
    lat_bins = np.arange(-90, 90 + res, res)

    sum_u = np.zeros((12, len(lat_bins), len(lon_bins)))
    sum_v = np.zeros((12, len(lat_bins), len(lon_bins)))
    count = np.zeros((12, len(lat_bins), len(lon_bins)))

    # only use "clean" daily-cadence fixes for the residual estimate
    valid = obs_all[(obs_all["date_gap"] == 1) & obs_all["u_obs"].notna()]

    for row in valid.itertuples(index=False):
        try:
            # time index into ERA5's time axis for this observation's date
            t_idx = _datetime_to_era5_index(row.datetime, era5)
            ua, va, sst, sic = era5.sample(row.lat, row.lon, t_idx)
        except Exception:
            continue

        Ua = np.hypot(ua, va)
        L, W, _ = area_to_dimensions(row.size)
        LA = La(Ua, row.lat, S(L, W))
        if abs(LA) > 0.2:
            u_wind_driven = ga * (alpha(LA) * va + beta(LA, row.lat) * ua)
            v_wind_driven = ga * (-alpha(LA) * ua + beta(LA, row.lat) * va)
        else:
            u_wind_driven = ga * (alpha(LA) * va + beta_approx(LA, row.lat) * ua)
            v_wind_driven = ga * (-alpha(LA) * ua + beta_approx(LA, row.lat) * va)

        u_resid = row.u_obs - u_wind_driven
        v_resid = row.v_obs - v_wind_driven

        mi = row.datetime.month - 1
        yi = closest_node(row.lat, lat_bins)
        xi = closest_node(((row.lon + 180) % 360) - 180, lon_bins)

        sum_u[mi, yi, xi] += u_resid
        sum_v[mi, yi, xi] += v_resid
        count[mi, yi, xi] += 1

    with np.errstate(invalid="ignore"):
        u_clim = np.where(count > 0, sum_u / np.maximum(count, 1), np.nan)
        v_clim = np.where(count > 0, sum_v / np.maximum(count, 1), np.nan)

    # fill gaps (bins with no observations) with the global annual mean current
    global_u = np.nanmean(u_clim)
    global_v = np.nanmean(v_clim)
    u_clim = np.where(np.isnan(u_clim), global_u, u_clim)
    v_clim = np.where(np.isnan(v_clim), global_v, v_clim)

    return u_clim, v_clim, lat_bins, lon_bins, count


def _datetime_to_era5_index(dt, era5):
    """Map a real datetime onto a fractional index into era5.time (assumes hours since 1900-01-01, ERA5's usual convention; adjust if your files differ)."""
    if hasattr(dt, "tz") and dt.tz is not None:
        dt = dt.tz_localize(None)
    epoch = pd.Timestamp("1900-01-01")
    hours_since_epoch = (dt - epoch).total_seconds() / 3600.0
    # time array assumed monotonic & regularly spaced
    t0 = float(era5.time[0])
    dtT = float(era5.time[1] - era5.time[0]) if len(era5.time) > 1 else 1.0
    return (hours_since_epoch - t0) / dtT


def lookup_ocean_current(u_clim, v_clim, lat_bins, lon_bins, lat_pt, lon_pt, month):
    yi = closest_node(lat_pt, lat_bins)
    xi = closest_node(((lon_pt + 180) % 360) - 180, lon_bins)
    return float(u_clim[month - 1, yi, xi]), float(v_clim[month - 1, yi, xi])


# ============================================================================================
# PART D -- CORE DRIFT / MELT / BREAKUP MODEL (physics unchanged from England et al. 2020)
# ============================================================================================

def iceberg_trajectory_parent(bergdims, start_location, start_datetime, prob,
                               era5, ocean_clim, land_mask_check,
                               ndays, min_volume, break_days=1,
                               nt_max=None):
    """
    Simulate one parent iceberg (and, recursively, its calved children)
    forward in time, driven by ERA5 winds + the empirical ocean-current
    climatology, exactly following WDE17 (drift, eqs 6-9) and England
    et al. 2020 (melt, eqs 2-3; breakup, eq. 4).

    This preserves the original recursive structure (a child iceberg is
    simulated by calling this same function again) but removes the
    fixed-length pre-allocated arrays tied to a specific ECCO2 time
    axis; instead it steps day-by-day up to `ndays` or until the
    iceberg melts / leaves the domain / drops below `min_volume`.
    """
    nt_max = nt_max or ndays
    dt = 86400.0  # seconds per day (daily timestep)
    dtR = dt / R * 180 / np.pi

    L, W, H = bergdims
    lon0, lat0 = start_location

    xil = np.full(nt_max, np.nan)
    yil = np.full(nt_max, np.nan)
    l = np.full(nt_max, np.nan)
    w = np.full(nt_max, np.nan)
    h = np.full(nt_max, np.nan)
    v = np.full(nt_max, np.nan)
    melt = np.zeros(nt_max)
    num_breaks = np.zeros(nt_max)

    xil[0], yil[0] = lon0, lat0
    l[0], w[0], h[0] = L, W, H
    v[0] = L * W * H

    children = []  # list of dicts: recursively simulated child trajectories

    i = 0
    outofbound = melted = False

    while (not outofbound) and (not melted) and i < nt_max - 1:
        i += 1
        cur_time = start_datetime + pd.Timedelta(days=i)
        t_idx = _datetime_to_era5_index(cur_time, era5)

        try:
            ua, va, SST, SIC = era5.sample(yil[i - 1], xil[i - 1], t_idx)
        except Exception:
            outofbound = True
            break

        u_clim, v_clim, lat_bins, lon_bins = ocean_clim[:4]
        uw, vw = lookup_ocean_current(u_clim, v_clim, lat_bins, lon_bins,
                                       yil[i - 1], xil[i - 1], cur_time.month)

        Ua = np.hypot(ua, va)
        LA = La(Ua, yil[i - 1], S(l[i - 1], w[i - 1]))

        if abs(LA) > 0.2:
            ui = uw + ga * (alpha(LA) * va + beta(LA, yil[i - 1]) * ua)
            vi = vw + ga * (-alpha(LA) * ua + beta(LA, yil[i - 1]) * va)
        else:
            ui = uw + ga * (alpha(LA) * va + beta_approx(LA, yil[i - 1]) * ua)
            vi = vw + ga * (-alpha(LA) * ua + beta_approx(LA, yil[i - 1]) * va)

        dlon = ui * dtR
        dlat = vi * dtR
        yil[i] = yil[i - 1] + dlat
        xil[i] = xil[i - 1] + dlon / np.cos((yil[i] + yil[i - 1]) / 2 * np.pi / 180)

        if yil[i] > era5.maxlat or yil[i] < era5.minlat:
            outofbound = True
            continue

        if xil[i] > era5.maxlon:
            xil[i] -= 360
        elif xil[i] < era5.minlon:
            xil[i] += 360

        if land_mask_check(yil[i], xil[i]):
            yil[i], xil[i] = yil[i - 1], xil[i - 1]

        # ---- melt (England et al. 2020, eqs 2-3 / appendix) ----
        Me = (a1 * Ua ** 0.5 + a2 * Ua) * (0.5 + 0.5 * np.cos(np.pi * SIC ** 3)) * (SST + 2) / 3.0
        Mv = b1 * max(SST, 0.0) + b2 * max(SST, 0.0) ** 2
        Mb = (c_melt * (np.hypot(ui - uw, vi - vw)) ** 0.8) * (SST - Ti) * (l[i - 1]) ** -0.2

        dldt = -Mv - Me
        dhdt = -Mb
        l[i] = l[i - 1] + dldt * dt
        w[i] = w[i - 1] + dldt * dt
        h[i] = h[i - 1] + dhdt * dt
        melt[i] = l[i] * w[i] * h[i] - l[i - 1] * w[i - 1] * h[i - 1]

        if l[i] <= 0 or w[i] <= 0 or h[i] <= 0:
            l[i] = w[i] = h[i] = 0
            melted = True

        if not melted:
            if w[i] / h[i] < np.sqrt(6 * rhoi / rhow * (1 - rhoi / rhow)):
                w[i], h[i] = h[i], w[i]
            if w[i] > l[i]:
                w[i], l[i] = l[i], w[i]

            # ---- breakup (England et al. 2020, eq. 4, Poisson) ----
            xmax = np.pi / (2 ** 1.5) * buoy_len(bend_B(h[i]))
            if prob > 0.0 and l[i] > 3 * xmax and SIC < 0.75 and i % break_days == 0:
                pfactor = float(break_days * prob)
                if pfactor < 20:
                    break_number = int(poisson.rvs(pfactor))
                else:
                    break_number = int(np.random.normal(pfactor, np.sqrt(pfactor)))
                break_number = max(0, break_number)

                num_breaks[i] = break_number
                if break_number > 0:
                    l2, w2 = xmax, epsilon * xmax
                    h2 = h[i]
                    A2 = l2 * w2
                    lnew = l[i] - break_number * A2 / w[i]
                    while lnew < 3 * xmax and break_number > 0:
                        break_number -= 1
                        lnew = l[i] - break_number * A2 / w[i]
                    l[i] = lnew
                    num_breaks[i] = break_number

                    if w2 > l2:
                        w2, l2 = l2, w2
                    if w2 / h2 < np.sqrt(6 * rhoi / rhow * (1 - rhoi / rhow)):
                        w2, h2 = h2, w2

                    child = iceberg_trajectory_parent(
                        [l2, w2, h2], [xil[i], yil[i]], cur_time, 0,
                        era5, ocean_clim, land_mask_check,
                        ndays - i, min_volume, break_days,
                        nt_max=nt_max - i)
                    child["n_calved"] = break_number
                    child["start_index"] = i
                    child["initial_length_m"] = float(l2)
                    child["initial_width_m"] = float(w2)
                    child["initial_thickness_m"] = float(h2)
                    children.append(child)

            if w[i] > l[i]:
                w[i], l[i] = l[i], w[i]
            v[i] = l[i] * h[i] * w[i]

        if not melted and v[i] < min_volume:
            break

    end_i = i
    return {
        "lon": xil[:end_i + 1], "lat": yil[:end_i + 1], "vol": v[:end_i + 1],
        "length": l[:end_i + 1], "width": w[:end_i + 1], "height": h[:end_i + 1],
        "melt": melt[:end_i + 1], "num_breaks": num_breaks[:end_i + 1],
        "start_time": start_datetime, "children": children,
    }


# ============================================================================================
# PART E -- VALIDATION AGAINST THE REAL OBSERVED TRAJECTORIES
# ============================================================================================

def validate_against_observations(sim_result, obs_track):
    """
    Compare a simulated parent-iceberg trajectory against the real
    observed trajectory it was seeded from (same berg_id), matched by
    date. Returns great-circle position error (km) and area error
    (km^2) per matched day -- this is only possible because we now seed
    and evaluate directly against the real dataset.
    """
    n = len(sim_result["lat"])
    sim_dates = [sim_result["start_time"] + pd.Timedelta(days=k) for k in range(n)]
    sim_df = pd.DataFrame({
        "datetime": sim_dates,
        "sim_lat": sim_result["lat"], "sim_lon": sim_result["lon"],
        "sim_area_km2": sim_result["length"] * sim_result["width"] / 1.0e6,
    })

    merged = pd.merge_asof(
        obs_track.sort_values("datetime")[["datetime", "lat", "lon", "size"]],
        sim_df.sort_values("datetime"),
        on="datetime", direction="nearest", tolerance=pd.Timedelta(days=1),
    ).dropna()

    if merged.empty:
        return merged

    R_km = 6371.0
    lat1, lon1 = np.radians(merged["lat"]), np.radians(merged["lon"])
    lat2, lon2 = np.radians(merged["sim_lat"]), np.radians(merged["sim_lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a_hav = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    merged["position_error_km"] = 2 * R_km * np.arcsin(np.sqrt(a_hav))
    merged["area_error_km2"] = (merged["sim_area_km2"] - merged["size"]).abs()
    return merged


# ============================================================================================
# MAIN
# ============================================================================================

def main(cfg=CONFIG, n_icebergs=None, run_full_simulation=True):
    os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)

    print("Loading observed iceberg dataset...")
    obs_all, seeds = load_observed_icebergs(
        cfg["OBS_DIR"], cfg["MAX_QC_FLAG"], cfg["VALID_MASK_VALUES"])
    print(f"  {seeds['berg_id'].nunique()} icebergs, {len(obs_all)} valid daily observations")

    if n_icebergs is not None:
        seeds = seeds.iloc[:n_icebergs]

    if not run_full_simulation:
        print("Skipping ERA5-dependent simulation (run_full_simulation=False). "
              "Observation loading + seed table are ready for inspection.")
        return {"obs_all": obs_all, "seeds": seeds}

    print("Loading ERA5 forcing...")
    era5 = ERA5Forcing(cfg)

    def land_mask_check(lat_pt, lon_pt):
        yi = closest_node(lat_pt, era5.lat)
        xi = closest_node(lon_pt, era5.lon)
        return bool(era5.land_mask[yi, xi])

    print("Building empirical ocean-current climatology from observations + ERA5...")
    u_clim, v_clim, lat_bins, lon_bins, coverage = build_empirical_ocean_climatology(
        obs_all, era5, cfg)
    ocean_clim = (u_clim, v_clim, lat_bins, lon_bins, coverage)
    
    # Save the calibrated climatology model artifact
    clim_path = os.path.join(cfg["OUTPUT_DIR"], "empirical_ocean_climatology.npz")
    np.savez_compressed(clim_path, u_clim=u_clim, v_clim=v_clim, lat_bins=lat_bins, lon_bins=lon_bins, coverage=coverage)
    print(f"  Calibrated ocean climatology model artifact saved to {clim_path}")

    results = []
    validations = []
    for row in seeds.itertuples(index=False):
        L, W, H = area_to_dimensions(row.size)
        print(f"Simulating {row.berg_id} from {row.datetime.date()} "
              f"({row.lat:.2f}, {row.lon:.2f}), L={L:.0f}m W={W:.0f}m H={H:.0f}m")

        sim = iceberg_trajectory_parent(
            [L, W, H], [row.lon, row.lat], row.datetime, prob=4,
            era5=era5, ocean_clim=ocean_clim, land_mask_check=land_mask_check,
            ndays=cfg["NDAYS"], min_volume=cfg["MIN_VOLUME"])
        sim["berg_id"] = row.berg_id
        results.append(sim)

        obs_track = obs_all[obs_all["berg_id"] == row.berg_id]
        validations.append(validate_against_observations(sim, obs_track).assign(berg_id=row.berg_id))

    val_df = pd.concat([v for v in validations if len(v)], ignore_index=True) if validations else pd.DataFrame()
    if len(val_df):
        val_path = os.path.join(cfg["OUTPUT_DIR"], "validation_against_observations.csv")
        val_df.to_csv(val_path, index=False)
        print(f"Validation summary written to {val_path}")
        print(val_df.groupby("berg_id")[["position_error_km", "area_error_km2"]].mean())

    return {"obs_all": obs_all, "seeds": seeds, "results": results, "validation": val_df}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate empirical ocean climatology and run iceberg drift simulation.")
    parser.add_argument("--n-icebergs", "-n", type=int, default=10, help="Number of icebergs to simulate (default: 10, use 0 or --all for all)")
    parser.add_argument("--all", action="store_true", help="Simulate all available icebergs in dataset")
    args = parser.parse_args()

    n = None if (args.all or args.n_icebergs == 0) else args.n_icebergs
    main(n_icebergs=n, run_full_simulation=os.path.exists(
        os.path.join(CONFIG["ERA5_DIR"], CONFIG["ERA5_WIND_FILE"])))
