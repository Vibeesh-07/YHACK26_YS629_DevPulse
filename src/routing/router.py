"""
src/routing/router.py
=====================
Step 3 Orchestrator: Dynamic Risk-Aware A* Routing & KPI Calculator.

Takes the Step 1 corridor and Step 2 Monte Carlo hazard states, then:
  1. Builds the 2D navigation cost grid.
  2. Runs Risk-Aware A* pathfinding for each day (1 to 7) around moving hazards.
  3. Applies Catmull-Rom spline smoothing to produce continuous shipping curves.
  4. Calculates real-time navigational KPIs (Route Distance, Closest Hazard, Confidence).
  5. Produces the complete multi-day state (src/contracts/multi_day_route_state.json)
     and updates the active daily contract (src/contracts/route_day_state.json).
"""

import os
import sys
import json
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.routing.cost_grid import CostGrid
from src.routing.astar import find_risk_aware_route
from src.routing.smoother import smooth_route_polyline
from src.physics.monte_carlo import haversine_nm
import iceberg_model_adapted as ima


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculates initial compass bearing (0-360 degrees) from point 1 to point 2.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)
    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    theta = np.arctan2(y, x)
    bearing = (np.degrees(theta) + 360) % 360
    return round(float(bearing), 1)


def interpolate_ship_position(route_polyline, day_num, total_days, total_distance_nm):
    """
    Calculates the vessel's projected position, progress, and heading on day_num (1..total_days).
    """
    if not route_polyline:
        return {
            "coords": [0.0, 0.0],
            "heading_deg": 0.0,
            "distance_traveled_nm": 0.0,
            "distance_remaining_nm": total_distance_nm,
            "progress_pct": 0.0,
            "average_speed_knots": 0.0,
            "daily_distance_nm": 0.0
        }

    # Total duration in hours
    total_hours = max(1.0, float(total_days) * 24.0)
    avg_speed_knots = round(total_distance_nm / total_hours, 1)
    daily_dist_nm = round(total_distance_nm / max(1, total_days), 1)

    # Progress fraction for day_num: Day 1 is departure (0%), Day total_days is arrival (100%)
    if total_days <= 1:
        frac = 1.0
    else:
        frac = min(1.0, max(0.0, (day_num - 1) / float(total_days - 1)))

    target_dist_nm = frac * total_distance_nm
    dist_remaining_nm = round(max(0.0, total_distance_nm - target_dist_nm), 1)
    progress_pct = round(frac * 100.0, 1)

    if len(route_polyline) == 1 or target_dist_nm <= 0.0:
        p0 = route_polyline[0]
        heading = 0.0
        if len(route_polyline) > 1:
            p1 = route_polyline[1]
            heading = calculate_bearing(p0[0], p0[1], p1[0], p1[1])
        return {
            "coords": [round(float(p0[0]), 4), round(float(p0[1]), 4)],
            "heading_deg": heading,
            "distance_traveled_nm": 0.0,
            "distance_remaining_nm": total_distance_nm,
            "progress_pct": 0.0,
            "average_speed_knots": avg_speed_knots,
            "daily_distance_nm": daily_dist_nm
        }

    accum_dist = 0.0
    current_pt = route_polyline[0]
    heading = 0.0

    for i in range(len(route_polyline) - 1):
        pt_a = route_polyline[i]
        pt_b = route_polyline[i + 1]
        seg_dist = haversine_nm(pt_a[0], pt_a[1], pt_b[0], pt_b[1])

        if accum_dist + seg_dist >= target_dist_nm:
            remain = target_dist_nm - accum_dist
            seg_frac = remain / seg_dist if seg_dist > 1e-6 else 0.0
            lat = pt_a[0] + seg_frac * (pt_b[0] - pt_a[0])
            lon = pt_a[1] + seg_frac * (pt_b[1] - pt_a[1])
            heading = calculate_bearing(pt_a[0], pt_a[1], pt_b[0], pt_b[1])
            current_pt = [round(float(lat), 4), round(float(lon), 4)]
            break
        accum_dist += seg_dist
    else:
        current_pt = route_polyline[-1]
        if len(route_polyline) >= 2:
            p_prev = route_polyline[-2]
            heading = calculate_bearing(p_prev[0], p_prev[1], current_pt[0], current_pt[1])

    return {
        "coords": [round(float(current_pt[0]), 4), round(float(current_pt[1]), 4)],
        "heading_deg": heading,
        "distance_traveled_nm": round(target_dist_nm, 1),
        "distance_remaining_nm": dist_remaining_nm,
        "progress_pct": progress_pct,
        "average_speed_knots": avg_speed_knots,
        "daily_distance_nm": daily_dist_nm
    }


def calculate_closest_hazard_nm(route_polyline, hazards):
    """
    Computes minimum clearance distance (nm) between ship route and any iceberg hazard.
    """
    if not hazards or not route_polyline:
        return 999.0

    min_dist = float("inf")
    for pt in route_polyline:
        for h in hazards:
            c_lat, c_lon = h["center"]
            d = haversine_nm(pt[0], pt[1], c_lat, c_lon)
            clearance = max(d - h["iceberg_radius_nm"], 0.0)
            if clearance < min_dist:
                min_dist = clearance
    return round(float(min_dist), 2)


def solve_multiday_routes(step1_state, step2_output, res_deg=0.06):
    """
    Solves optimal collision-free routes for each forecast day (Days 1 through N).
    """
    start_coords = step1_state["start_coords"]
    dest_coords = step1_state["dest_coords"]
    bbox = step1_state["corridor_bbox"]
    forecast_days = step2_output.get("forecast_days", 7)
    daily_hazard_states = step2_output["daily_hazard_states"]

    # ── Fast Pre-Check: Is the direct geodesic line unobstructed? ────────────
    from src.data.land_mask import build_land_mask_fn
    land_mask_fn = build_land_mask_fn()

    num_samples = 60
    sample_lats = np.linspace(start_coords[0], dest_coords[0], num_samples)
    sample_lons = np.linspace(start_coords[1], dest_coords[1], num_samples)
    direct_waypoints = [[round(float(la), 4), round(float(lo), 4)] for la, lo in zip(sample_lats, sample_lons)]
    direct_dist_nm = round(haversine_nm(start_coords[0], start_coords[1], dest_coords[0], dest_coords[1]), 1)

    # Check 1: Does the direct path intersect land?
    direct_hits_land = any(land_mask_fn(p[0], p[1]) for p in direct_waypoints)

    # Check 2: Do any icebergs project onto the direct path on each day?
    day_obstructed = []
    day_closest_hazards = []
    for day_idx in range(forecast_days):
        hazards = daily_hazard_states[day_idx]["hazards"] if day_idx < len(daily_hazard_states) else []
        obstructed = direct_hits_land
        min_h_dist = 999.0
        for h in hazards:
            center = h.get("center", [h.get("lat"), h.get("lon")])
            if center[0] is None or center[1] is None:
                continue
            r_buffer = float(h.get("buffer_radius_nm", 5.0))
            dists = [haversine_nm(p[0], p[1], center[0], center[1]) for p in direct_waypoints]
            min_d = min(dists) if dists else 999.0
            if min_d < min_h_dist:
                min_h_dist = min_d
            if min_d <= (r_buffer + 1.0):
                obstructed = True
        day_obstructed.append(obstructed)
        day_closest_hazards.append(min_h_dist)

    all_days_data = []

    # ── Instant Fast-Path: If NO day has land or iceberg obstructions, bypass heavy grid! ─
    if not any(day_obstructed):
        print(f"\n[Step 3] ⚡ Direct path is completely unobstructed! (No land, 0 icebergs in path)")
        print(f"         Bypassing heavy 2D grid generation & A* search. Instant direct route ({direct_dist_nm} nm).")
        for day_idx in range(forecast_days):
            day_num = day_idx + 1
            hazards = daily_hazard_states[day_idx]["hazards"] if day_idx < len(daily_hazard_states) else []
            closest_hazard_nm = round(day_closest_hazards[day_idx], 1)
            confidence_pct = max(90, 99 - day_idx)

            ship_info = interpolate_ship_position(direct_waypoints, day_num, forecast_days, direct_dist_nm)

            print(f"  * Day {day_num}: Direct straight path | Distance: {direct_dist_nm} nm | Closest Hazard: {closest_hazard_nm} nm | Ship: {ship_info['progress_pct']}% at {ship_info['coords']} ({ship_info['average_speed_knots']} kn)")
            day_payload = {
                "day": day_num,
                "total_days": forecast_days,
                "status": {
                    "hazards_nearby": len(hazards),
                    "route_confidence_pct": confidence_pct,
                    "is_direct_path": True
                },
                "kpis": {
                    "route_distance_nm": direct_dist_nm,
                    "closest_hazard_nm": closest_hazard_nm,
                    "icebergs_tracked": len(hazards),
                    "nodes_evaluated": 0,
                    "average_speed_knots": ship_info["average_speed_knots"],
                    "daily_distance_nm": ship_info["daily_distance_nm"],
                    "distance_traveled_nm": ship_info["distance_traveled_nm"],
                    "distance_remaining_nm": ship_info["distance_remaining_nm"],
                    "progress_pct": ship_info["progress_pct"]
                },
                "navigation": {
                    "start": {"name": "Start", "coords": start_coords},
                    "destination": {"name": "Destination", "coords": dest_coords},
                    "route_polyline": direct_waypoints,
                    "raw_waypoints_count": len(direct_waypoints),
                    "smoothed_points_count": len(direct_waypoints),
                    "is_direct_path": True
                },
                "ship": ship_info,
                "hazards": hazards
            }
            all_days_data.append(day_payload)

        # Save multi-day contract
        multi_day_path = "src/contracts/multi_day_route_state.json"
        os.makedirs(os.path.dirname(multi_day_path), exist_ok=True)
        with open(multi_day_path, "w") as f:
            json.dump(all_days_data, f, indent=2)
        print(f"\n💾 Full {forecast_days}-Day Route State saved to: {multi_day_path}")

        active_contract_path = "src/contracts/route_day_state.json"
        active_day = all_days_data[2] if len(all_days_data) >= 3 else all_days_data[0]
        with open(active_contract_path, "w") as f:
            json.dump(active_day, f, indent=2)
        print(f"💾 Active Contract B updated: {active_contract_path} (Day {active_day['day']})")

        return all_days_data

    # ── Fallback: Construct 2D Navigation Cost Grid for obstructed corridors ──────────
    print(f"\n[Step 3] Initializing 2D Navigation Cost Grid (res = {res_deg}°)...")

    # ── Land Mask: Natural Earth polygons via shapely ─────────────────────────
    from src.data.land_mask import build_land_mask_grid
    import numpy as _np_lm

    _tmp_lats = _np_lm.arange(bbox["min_lat"], bbox["max_lat"] + res_deg, res_deg)
    _tmp_lons = _np_lm.arange(bbox["min_lon"], bbox["max_lon"] + res_deg, res_deg)
    print(f"  Building land mask for {len(_tmp_lats)}x{len(_tmp_lons)} grid…")
    land_grid, land_mask_fn = build_land_mask_grid(_tmp_lats, _tmp_lons)
    print(f"  Land mask: {land_grid.sum()} / {land_grid.size} cells blocked as land.")

    # ── Sea-ice mask from ERA5 (optional) ────────────────────────────────────
    sic_fn = None
    try:
        era5 = ima.ERA5Forcing(ima.CONFIG)
        def sic_fn(lat, lon):
            yi = ima.closest_node(lat, era5.lat)
            xi = ima.closest_node(lon, era5.lon)
            val = era5.sic[0, yi, xi]
            return float(val) if not np.isnan(val) else 0.0
    except Exception:
        pass

    cost_grid = CostGrid(bbox, res_deg=res_deg, land_mask_fn=land_mask_fn,
                         sic_fn=sic_fn, land_grid=land_grid)
    print(f"  Grid size: {cost_grid.nrows} lat x {cost_grid.ncols} lon ({cost_grid.nrows * cost_grid.ncols} cells)")

    print(f"\n[Step 3] Solving Risk-Aware A* routes for Days 1 through {forecast_days}...")
    for day_idx in range(forecast_days):
        day_num = day_idx + 1
        hazards = daily_hazard_states[day_idx]["hazards"] if day_idx < len(daily_hazard_states) else []

        # A* Pathfinding (Fast-paths to direct geodesic line if path is unobstructed)
        astar_res = find_risk_aware_route(cost_grid, start_coords, dest_coords, hazards)
        is_direct = astar_res.get("is_direct", False)

        if is_direct:
            # Direct straight path: use clean geodesic points without warping
            smoothed_polyline = astar_res["waypoints"]
            route_dist_nm = round(float(astar_res["total_distance_nm"]), 1)
            closest_hazard_nm = float(astar_res.get("closest_hazard_nm", 999.0))
            confidence_pct = max(90, 99 - day_idx)
            ship_info = interpolate_ship_position(smoothed_polyline, day_num, forecast_days, route_dist_nm)
            print(f"  * Day {day_num}: Direct straight path (unobstructed) | Distance: {route_dist_nm} nm | Closest Hazard: {closest_hazard_nm:.1f} nm | Ship: {ship_info['progress_pct']}% at {ship_info['coords']} ({ship_info['average_speed_knots']} kn) | Nodes: 0 (Fast LOS)")
        else:
            # Spline Smoothing with Land Avoidance Validation
            smoothed_polyline = smooth_route_polyline(astar_res["waypoints"], land_mask_fn=cost_grid.land_mask_fn)

            # Calculate actual smoothed route distance
            route_dist_nm = 0.0
            for i in range(len(smoothed_polyline) - 1):
                p1, p2 = smoothed_polyline[i], smoothed_polyline[i + 1]
                route_dist_nm += haversine_nm(p1[0], p1[1], p2[0], p2[1])
            route_dist_nm = round(route_dist_nm, 1)

            # Calculate closest hazard clearance
            closest_hazard_nm = calculate_closest_hazard_nm(smoothed_polyline, hazards)

            # Dynamic Route Confidence (decreases with forecast horizon and hazard proximity)
            base_conf = max(65, 95 - day_idx * 2)
            if closest_hazard_nm < 3.0:
                base_conf -= 10
            elif closest_hazard_nm < 5.0:
                base_conf -= 4
            confidence_pct = max(50, min(99, base_conf))
            ship_info = interpolate_ship_position(smoothed_polyline, day_num, forecast_days, route_dist_nm)
            print(f"  * Day {day_num}: Route {route_dist_nm} nm | Closest Hazard: {closest_hazard_nm} nm | Confidence: {confidence_pct}% | Ship: {ship_info['progress_pct']}% at {ship_info['coords']} ({ship_info['average_speed_knots']} kn) | Nodes: {astar_res['nodes_evaluated']}")

        day_payload = {
            "day": day_num,
            "total_days": forecast_days,
            "status": {
                "hazards_nearby": len(hazards),
                "route_confidence_pct": confidence_pct,
                "is_direct_path": is_direct
            },
            "kpis": {
                "route_distance_nm": route_dist_nm,
                "closest_hazard_nm": closest_hazard_nm,
                "icebergs_tracked": len(hazards),
                "nodes_evaluated": astar_res["nodes_evaluated"],
                "average_speed_knots": ship_info["average_speed_knots"],
                "daily_distance_nm": ship_info["daily_distance_nm"],
                "distance_traveled_nm": ship_info["distance_traveled_nm"],
                "distance_remaining_nm": ship_info["distance_remaining_nm"],
                "progress_pct": ship_info["progress_pct"]
            },
            "navigation": {
                "start": {"name": "Start", "coords": start_coords},
                "destination": {"name": "Destination", "coords": dest_coords},
                "route_polyline": smoothed_polyline,
                "raw_waypoints_count": len(astar_res["waypoints"]),
                "smoothed_points_count": len(smoothed_polyline),
                "is_direct_path": is_direct
            },
            "ship": ship_info,
            "hazards": hazards
        }
        all_days_data.append(day_payload)

    # Save multi-day contract
    multi_day_path = "src/contracts/multi_day_route_state.json"
    os.makedirs(os.path.dirname(multi_day_path), exist_ok=True)
    with open(multi_day_path, "w") as f:
        json.dump(all_days_data, f, indent=2)
    print(f"\n💾 Full {forecast_days}-Day Route State saved to: {multi_day_path}")

    # Update active daily contract with Day 3 (standard reference) or Day 1
    active_contract_path = "src/contracts/route_day_state.json"
    active_day = all_days_data[2] if len(all_days_data) >= 3 else all_days_data[0]
    with open(active_contract_path, "w") as f:
        json.dump(active_day, f, indent=2)
    print(f"💾 Active Contract B updated: {active_contract_path} (Day {active_day['day']})")

    return all_days_data


if __name__ == "__main__":
    from src.data.corridor import execute_step1
    from src.physics.drift_engine import execute_step2

    sample_start = [-63.5, -58.2]
    sample_dest = [-60.8, -52.4]

    s1 = execute_step1(sample_start, sample_dest, "2026-09-10T00:00:00Z")
    s2 = execute_step2(s1, forecast_days=7)
    multiday = solve_multiday_routes(s1, s2)
    print("\nA* Multi-Day Routing complete!")
