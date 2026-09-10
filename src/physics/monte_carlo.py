"""
src/physics/monte_carlo.py
==========================
Stage 4 & 5: Monte Carlo Uncertainty Ensemble & 95% Confidence Hazard Radii.

Executes N (50-100) stochastic forward drift trajectories per tabular iceberg
by perturbing:
  1. Wind forcing: u10, v10 ~ N(0, sigma_w^2) (atmospheric gusts & forecast shear)
  2. Ocean currents: u_ocn, v_ocn ~ N(0, sigma_o^2) (mesoscale eddies)
  3. Physical dimensions: L, W ~ N(mu, 0.05*mu) (satellite SAR measurement error)
  4. Calving / breakup threshold perturbations

For each forecast day t (1..7):
  - Calculates the ensemble centroid (mean lat/lon).
  - Computes the empirical 95% confidence dispersion radius (R_MC_95%).
  - Produces the total rigorous hazard boundary:
      R_hazard(t) = R_iceberg + R_MC_95%(t) + R_safety
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import iceberg_model_adapted as ima

EARTH_RADIUS_NM = 3440.065  # Nautical miles
METERS_PER_NM = 1852.0


def haversine_nm(lat1, lon1, lat2, lon2):
    """
    Great-circle distance in nautical miles between two points or arrays of points.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_NM * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


class PerturbedERA5Wrapper:
    """
    Wraps the ERA5 environmental forcing class to inject stochastic red noise
    (correlated across forecast days) representing wind forecast uncertainty.
    """

    def __init__(self, base_era5, ndays=7, wind_std=1.5):
        self.base = base_era5
        self.lat = base_era5.lat
        self.lon = base_era5.lon
        self.time = base_era5.time
        self.land_mask = base_era5.land_mask
        self.minlat, self.maxlat = base_era5.minlat, base_era5.maxlat
        self.minlon, self.maxlon = base_era5.minlon, base_era5.maxlon

        # First-order autoregressive AR(1) noise for temporal consistency across days
        alpha = 0.6  # temporal persistence
        noise_u = np.zeros(ndays + 2)
        noise_v = np.zeros(ndays + 2)
        u_prev, v_prev = 0.0, 0.0
        for i in range(ndays + 2):
            u_prev = alpha * u_prev + np.sqrt(1.0 - alpha**2) * np.random.normal(0, wind_std)
            v_prev = alpha * v_prev + np.sqrt(1.0 - alpha**2) * np.random.normal(0, wind_std)
            noise_u[i] = u_prev
            noise_v[i] = v_prev

        self.noise_u = noise_u
        self.noise_v = noise_v

    def sample(self, lat_pt, lon_pt, t_index_float):
        ua, va, sst, sic = self.base.sample(lat_pt, lon_pt, t_index_float)
        day_idx = min(int(t_index_float) % len(self.noise_u), len(self.noise_u) - 1)
        return ua + self.noise_u[day_idx], va + self.noise_v[day_idx], sst, sic


def run_monte_carlo_ensemble(berg_info, start_datetime, era5, ocean_clim, land_mask_check,
                             n_runs=50, ndays=7, wind_std=1.5, ocn_std=0.04, safety_buffer_nm=3.0):
    """
    Simulates an ensemble of n_runs stochastic drift trajectories for a single iceberg
    and computes the expanding 95% confidence hazard radius per forecast day.
    """
    base_L = berg_info.get("length_km", 10.0) * 1000.0
    base_W = berg_info.get("width_km", 5.0) * 1000.0
    base_H = berg_info.get("thickness_m", 220.0)
    pos0 = [float(berg_info["lon"]), float(berg_info["lat"])]
    start_ts = pd.Timestamp(start_datetime)

    # Base physical equivalent radius in nautical miles
    r_iceberg_nm = round(np.sqrt(base_L * base_W) / (2.0 * METERS_PER_NM), 2)

    all_lats = np.full((n_runs, ndays), np.nan)
    all_lons = np.full((n_runs, ndays), np.nan)
    all_vols = np.full((n_runs, ndays), np.nan)

    u_clim_base, v_clim_base = ocean_clim[0], ocean_clim[1]
    lat_bins, lon_bins, coverage = ocean_clim[2], ocean_clim[3], ocean_clim[4]

    for k in range(n_runs):
        # 1. Stochastic wind forcing wrapper
        pert_era5 = PerturbedERA5Wrapper(era5, ndays=ndays, wind_std=wind_std)

        # 2. Stochastic ocean current anomaly (mesoscale eddy perturbation)
        delta_u = np.random.normal(0, ocn_std)
        delta_v = np.random.normal(0, ocn_std)
        pert_ocean = (u_clim_base + delta_u, v_clim_base + delta_v, lat_bins, lon_bins, coverage)

        # 3. Satellite SAR dimension measurement uncertainty (+/- 5%)
        scale_L = np.random.normal(1.0, 0.05)
        scale_W = np.random.normal(1.0, 0.05)
        d_pert = [max(base_L * scale_L, 500.0), max(base_W * scale_W, 300.0), base_H]

        # 4. Forward simulation
        res = ima.iceberg_trajectory_parent(
            d_pert, pos0, start_ts, prob=4,
            era5=pert_era5, ocean_clim=pert_ocean,
            land_mask_check=land_mask_check,
            ndays=ndays, min_volume=1e6
        )

        n_pts = min(len(res["lat"]), ndays)
        for i in range(n_pts):
            if not np.isnan(res["lat"][i]):
                all_lats[k, i] = res["lat"][i]
                all_lons[k, i] = res["lon"][i]
                all_vols[k, i] = res["vol"][i]

    # Calculate day-by-day statistics
    daily_stats = []
    for d in range(ndays):
        day_lats = all_lats[:, d]
        day_lons = all_lons[:, d]
        valid = ~np.isnan(day_lats) & ~np.isnan(day_lons)

        if not np.any(valid):
            break

        valid_lats = day_lats[valid]
        valid_lons = day_lons[valid]

        centroid_lat = float(np.mean(valid_lats))
        centroid_lon = float(np.mean(valid_lons))

        # Great circle dispersion distances from centroid (nm)
        dispersions_nm = haversine_nm(centroid_lat, centroid_lon, valid_lats, valid_lons)

        r50_dispersion_nm = float(np.percentile(dispersions_nm, 50))
        r95_dispersion_nm = float(np.percentile(dispersions_nm, 95))

        # Total dynamic hazard radius: Iceberg Core + Monte Carlo 95% dispersion + Vessel Safety Margin
        total_hazard_nm = round(r_iceberg_nm + r95_dispersion_nm + safety_buffer_nm, 2)

        daily_stats.append({
            "day": d + 1,
            "date": (start_ts + pd.Timedelta(days=d)).strftime("%Y-%m-%d"),
            "centroid": [round(centroid_lat, 4), round(centroid_lon, 4)],
            "iceberg_radius_nm": r_iceberg_nm,
            "mc_50_dispersion_nm": round(r50_dispersion_nm, 2),
            "mc_95_dispersion_nm": round(r95_dispersion_nm, 2),
            "safety_buffer_nm": safety_buffer_nm,
            "total_hazard_radius_nm": total_hazard_nm,
            "ensemble_members_alive": int(np.sum(valid))
        })

    return {
        "id": berg_info["id"],
        "ensemble_runs": n_runs,
        "daily_stats": daily_stats
    }


MAX_BUFFER_EXTRA_NM = 15.0  # Max nm added on top of iceberg physical radius for hazard ring cap

def compute_corridor_monte_carlo_hazards(icebergs, start_datetime, era5, ocean_clim, land_mask_check,
                                         n_runs=50, ndays=7, safety_buffer_nm=3.0):
    """
    Runs Monte Carlo ensemble simulations for all icebergs in the corridor
    and organizes dynamic hazard obstacles for each day (1..ndays).
    """
    mc_results = []
    for berg in icebergs:
        res = run_monte_carlo_ensemble(
            berg_info=berg,
            start_datetime=start_datetime,
            era5=era5,
            ocean_clim=ocean_clim,
            land_mask_check=land_mask_check,
            n_runs=n_runs,
            ndays=ndays,
            safety_buffer_nm=safety_buffer_nm
        )
        mc_results.append(res)

    # Reorganize by day for routing & dashboard contracts
    hazards_by_day = {}
    for d in range(ndays):
        day_num = d + 1
        hazards_by_day[day_num] = []

        for res in mc_results:
            if d < len(res["daily_stats"]):
                st = res["daily_stats"][d]
                r_iceberg = st["iceberg_radius_nm"]
                # Cap the hazard buffer: never exceeds iceberg_radius + MAX_BUFFER_EXTRA_NM.
                # This prevents the ring from growing arbitrarily large on later forecast days
                # while still giving a physics-informed standoff that scales with iceberg size.
                capped_buffer_nm = round(
                    min(st["total_hazard_radius_nm"], r_iceberg + MAX_BUFFER_EXTRA_NM), 2
                )
                hazards_by_day[day_num].append({
                    "id": res["id"],
                    "center": st["centroid"],
                    "iceberg_radius_nm": r_iceberg,
                    "mc_95_dispersion_nm": st["mc_95_dispersion_nm"],
                    "buffer_radius_nm": capped_buffer_nm,
                    "total_hazard_radius_nm_uncapped": st["total_hazard_radius_nm"]  # kept for diagnostics
                })

    return mc_results, hazards_by_day


if __name__ == "__main__":
    import json
    with open("src/contracts/initial_state.json", "r") as f:
        state = json.load(f)

    print("Testing Monte Carlo Analysis on corridor icebergs (N=50 runs/berg)...")
    era5 = ima.ERA5Forcing(ima.CONFIG)
    npz = np.load("./Output_adapted/empirical_ocean_climatology.npz")
    ocean_clim = (npz["u_clim"], npz["v_clim"], npz["lat_bins"], npz["lon_bins"], npz["coverage"])

    def land_mask_check(lat_pt, lon_pt):
        yi = ima.closest_node(lat_pt, era5.lat)
        xi = ima.closest_node(lon_pt, era5.lon)
        return bool(era5.land_mask[yi, xi])

    mc_results, hazards_by_day = compute_corridor_monte_carlo_hazards(
        icebergs=state["icebergs"][:2],
        start_datetime=state["forecast_start_date"],
        era5=era5,
        ocean_clim=ocean_clim,
        land_mask_check=land_mask_check,
        n_runs=50,
        ndays=7
    )

    for berg_mc in mc_results:
        print(f"\n🧊 {berg_mc['id']} (N={berg_mc['ensemble_runs']} runs):")
        print(f"   {'Day':<5} | {'Centroid Lat':<12} | {'Centroid Lon':<12} | {'R_berg':<8} | {'MC 95% Disp':<12} | {'Total Hazard Radius'}")
        print(f"   {'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*12}-+-{'-'*19}")
        for st in berg_mc["daily_stats"]:
            print(f"   Day {st['day']:<2}| {st['centroid'][0]:>10.4f}° | {st['centroid'][1]:>10.4f}° | {st['iceberg_radius_nm']:>6.2f}nm | {st['mc_95_dispersion_nm']:>9.2f}nm | {st['total_hazard_radius_nm']:>10.2f} nm")
