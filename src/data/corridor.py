"""
src/data/corridor.py
====================
Step 1: Navigation Corridor (AOI) Definition & Iceberg Hazard Filtering.

Takes ship Start [lat, lon] and Destination [lat, lon] coordinates:
1. Calculates dynamic geographic bounding corridor (AOI) with configurable safety buffer.
2. Filters tracking records from the catalog to isolate icebergs threatening this corridor.
3. Computes geometric dimensions (length, width, thickness), mass (kg), and initial velocities.
4. Produces the standard Step 1 contract (initial_state.json).
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Thickness lookup table from England, Wagner & Eisenman (2020)
_THICKNESS_TABLE_L_M = np.array([690, 1220, 2180, 3870, 6890, 12250, 21780, 38730])
_THICKNESS_TABLE_H_M = np.array([175, 200, 225, 250, 275, 300, 325, 350])
RHO_ICE = 900.0  # kg / m^3


def compute_corridor_bbox(start_coords, dest_coords, buffer_lat=1.5, buffer_lon=2.5):
    """
    Compute geographic bounding box for the ship navigation corridor.
    Intelligently inspects any intersecting landmasses (e.g. South America, Antarctic Peninsula)
    to ensure the corridor bounding box includes open-water passages (like Drake Passage around
    Cape Horn or Bransfield Strait around Prime Head) so routes never get clipped by grid boundaries.
    """
    min_lat = min(start_coords[0], dest_coords[0]) - buffer_lat
    max_lat = max(start_coords[0], dest_coords[0]) + buffer_lat
    min_lon = min(start_coords[1], dest_coords[1]) - buffer_lon
    max_lon = max(start_coords[1], dest_coords[1]) + buffer_lon

    try:
        from src.data.land_mask import _build_merged_geometry
        from shapely.geometry import LineString
        geom = _build_merged_geometry()
        if geom is not None:
            line = LineString([(start_coords[1], start_coords[0]), (dest_coords[1], dest_coords[0])])
            if line.intersects(geom):
                for g in (geom.geoms if hasattr(geom, 'geoms') else [geom]):
                    if g.intersects(line):
                        g_min_lon, g_min_lat, g_max_lon, g_max_lat = g.bounds
                        # Continental South America (extends north, but Cape Horn/Drake Passage is at -56° to -60°)
                        if g_max_lat > -20.0 and g_min_lat < -40.0:
                            min_lat = min(min_lat, -58.5)
                        # Antarctica / Antarctic Peninsula (extends south, but Prime Head is at -63.23°)
                        elif g_min_lat <= -80.0:
                            max_lat = max(max_lat, -61.5)
                        else:
                            # Specific island or local headland: expand bounds so ship can route around
                            min_lat = min(min_lat, g_min_lat - 1.5)
                            max_lat = max(max_lat, g_max_lat + 1.5)
                            min_lon = min(min_lon, g_min_lon - 2.0)
                            max_lon = max(max_lon, g_max_lon + 2.0)
    except Exception:
        pass

    # Ensure bounds stay within valid geographic coordinates
    min_lat = max(-80.0, min_lat)
    max_lat = min(80.0, max_lat)
    min_lon = max(-180.0, min_lon)
    max_lon = min(180.0, max_lon)

    return {
        "min_lat": round(float(min_lat), 2),
        "max_lat": round(float(max_lat), 2),
        "min_lon": round(float(min_lon), 2),
        "max_lon": round(float(max_lon), 2),
    }


def area_to_dimensions(area_km2, aspect=1.5):
    """Convert plan-view area (km^2) to Length, Width, Height in meters."""
    area_m2 = area_km2 * 1.0e6
    W = np.sqrt(area_m2 / aspect)
    L = aspect * W
    H = float(np.interp(L, _THICKNESS_TABLE_L_M, _THICKNESS_TABLE_H_M))
    return float(L), float(W), float(H)


def estimate_iceberg_physics(area_km2):
    """Calculate dimensions (m), volume (m^3), and mass (kg)."""
    L, W, H = area_to_dimensions(area_km2)
    vol_m3 = L * W * H
    mass_kg = RHO_ICE * vol_m3
    return {
        "length_km": round(L / 1000.0, 2),
        "width_km": round(W / 1000.0, 2),
        "thickness_m": round(H, 1),
        "length_m": L,
        "width_m": W,
        "mass_kg": float(f"{mass_kg:.3e}"),
        "vol_m3": vol_m3
    }


def filter_icebergs_in_corridor(corridor_bbox, dataset_dir="./dataset", initial_contract="src/contracts/initial_state.json", max_icebergs=5):
    """
    Find icebergs within the navigation corridor.
    Checks existing contract icebergs first, then scans observation dataset.
    """
    b = corridor_bbox
    icebergs = []
    seen_ids = set()

    # 1. Check existing initial_state.json if available
    if os.path.exists(initial_contract):
        try:
            with open(initial_contract, "r") as f:
                contract_data = json.load(f)
                for berg in contract_data.get("icebergs", []):
                    lat, lon = berg["lat"], berg["lon"]
                    if b["min_lat"] <= lat <= b["max_lat"] and b["min_lon"] <= lon <= b["max_lon"]:
                        icebergs.append(berg)
                        seen_ids.add(berg["id"].lower())
        except Exception:
            pass

    # 2. Scan catalog for any additional active tabular icebergs in the bounding box
    if len(icebergs) < max_icebergs and os.path.isdir(dataset_dir):
        catalog_files = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")))
        for fpath in catalog_files:
            if len(icebergs) >= max_icebergs:
                break
            berg_id = os.path.splitext(os.path.basename(fpath))[0].upper()
            if berg_id.lower() in seen_ids:
                continue

            try:
                df = pd.read_csv(fpath)
                in_corridor = df[
                    (df["lat"] >= b["min_lat"]) & (df["lat"] <= b["max_lat"]) &
                    (df["lon"] >= b["min_lon"]) & (df["lon"] <= b["max_lon"]) &
                    (df["size"] > 0)
                ]
                if len(in_corridor) > 0:
                    last_row = in_corridor.iloc[-1]
                    area = float(last_row["size"])
                    phys = estimate_iceberg_physics(area)
                    icebergs.append({
                        "id": berg_id,
                        "lat": round(float(last_row["lat"]), 3),
                        "lon": round(float(last_row["lon"]), 3),
                        "length_km": phys["length_km"],
                        "width_km": phys["width_km"],
                        "thickness_m": phys["thickness_m"],
                        "mass_kg": phys["mass_kg"],
                        "initial_u_vel": 0.25,
                        "initial_v_vel": -0.10
                    })
                    seen_ids.add(berg_id.lower())
            except Exception:
                continue

    return icebergs


def execute_step1(start_coords, dest_coords, forecast_start_date=None, buffer_lat=1.5, buffer_lon=2.5, save_path="src/contracts/initial_state.json"):
    """
    STEP 1: Main execution function.
    Given ship start and destination, returns the corridor bounding box and filtered icebergs.
    """
    if forecast_start_date is None:
        forecast_start_date = datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z")

    from src.data.land_mask import build_land_mask_fn, find_nearest_water_coord
    land_fn = build_land_mask_fn()
    orig_start = [round(float(start_coords[0]), 4), round(float(start_coords[1]), 4)]
    orig_dest = [round(float(dest_coords[0]), 4), round(float(dest_coords[1]), 4)]
    start_coords = find_nearest_water_coord(orig_start, land_mask_fn=land_fn)
    dest_coords = find_nearest_water_coord(orig_dest, land_mask_fn=land_fn)
    if start_coords != orig_start:
        print(f"⚠️ [Land Avoidance] Starting point {orig_start} is on land. Marked down to nearby coast: {start_coords}")
    if dest_coords != orig_dest:
        print(f"⚠️ [Land Avoidance] Destination point {orig_dest} is on land. Marked down to nearby coast: {dest_coords}")

    bbox = compute_corridor_bbox(start_coords, dest_coords, buffer_lat, buffer_lon)
    corridor_icebergs = filter_icebergs_in_corridor(bbox)

    step1_output = {
        "forecast_start_date": forecast_start_date,
        "start_coords": [float(start_coords[0]), float(start_coords[1])],
        "dest_coords": [float(dest_coords[0]), float(dest_coords[1])],
        "corridor_bbox": bbox,
        "icebergs": corridor_icebergs
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(step1_output, f, indent=2)

    return step1_output


if __name__ == "__main__":
    sample_start = [-63.5, -58.2]
    sample_dest = [-60.8, -52.4]
    res = execute_step1(sample_start, sample_dest, "2026-09-10T00:00:00Z")
    print(f"Step 1 completed: Bounding corridor {res['corridor_bbox']}")
    print(f"Found {len(res['icebergs'])} icebergs in corridor:")
    for b in res["icebergs"]:
        print(f"  * {b['id']} at ({b['lat']}, {b['lon']}), Size: {b['length_km']}km x {b['width_km']}km, Mass: {b['mass_kg']} kg")
