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
    Solves optimal collision-free routes for each forecast day (Days 1 through 7).
    """
    start_coords = step1_state["start_coords"]
    dest_coords = step1_state["dest_coords"]
    bbox = step1_state["corridor_bbox"]
    forecast_days = step2_output.get("forecast_days", 7)
    daily_hazard_states = step2_output["daily_hazard_states"]

    print(f"\n[Step 3] Initializing 2D Navigation Cost Grid (res = {res_deg}°)...")

    # Load environmental forcing for land & sea-ice masks
    try:
        era5 = ima.ERA5Forcing(ima.CONFIG)
        def land_mask_fn(lat, lon):
            yi = ima.closest_node(lat, era5.lat)
            xi = ima.closest_node(lon, era5.lon)
            return bool(era5.land_mask[yi, xi])

        def sic_fn(lat, lon):
            yi = ima.closest_node(lat, era5.lat)
            xi = ima.closest_node(lon, era5.lon)
            # Latest available SIC slice
            val = era5.sic[0, yi, xi]
            return float(val) if not np.isnan(val) else 0.0
    except Exception:
        land_mask_fn = None
        sic_fn = None

    cost_grid = CostGrid(bbox, res_deg=res_deg, land_mask_fn=land_mask_fn, sic_fn=sic_fn)
    print(f"  Grid size: {cost_grid.nrows} lat x {cost_grid.ncols} lon ({cost_grid.nrows * cost_grid.ncols} cells)")

    all_days_data = []

    print(f"\n[Step 3] Solving Risk-Aware A* routes for Days 1 through {forecast_days}...")
    for day_idx in range(forecast_days):
        day_num = day_idx + 1
        hazards = daily_hazard_states[day_idx]["hazards"] if day_idx < len(daily_hazard_states) else []

        # A* Pathfinding
        astar_res = find_risk_aware_route(cost_grid, start_coords, dest_coords, hazards)

        # Spline Smoothing
        smoothed_polyline = smooth_route_polyline(astar_res["waypoints"])

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

        day_payload = {
            "day": day_num,
            "total_days": forecast_days,
            "status": {
                "hazards_nearby": len(hazards),
                "route_confidence_pct": confidence_pct
            },
            "kpis": {
                "route_distance_nm": route_dist_nm,
                "closest_hazard_nm": closest_hazard_nm,
                "icebergs_tracked": len(hazards),
                "nodes_evaluated": astar_res["nodes_evaluated"]
            },
            "navigation": {
                "start": {"name": "Start", "coords": start_coords},
                "destination": {"name": "Destination", "coords": dest_coords},
                "route_polyline": smoothed_polyline,
                "raw_waypoints_count": len(astar_res["waypoints"]),
                "smoothed_points_count": len(smoothed_polyline)
            },
            "hazards": hazards
        }
        all_days_data.append(day_payload)
        print(f"  * Day {day_num}: Route {route_dist_nm} nm | Closest Hazard: {closest_hazard_nm} nm | Confidence: {confidence_pct}% | Nodes: {astar_res['nodes_evaluated']}")

    # Save multi-day contract
    multi_day_path = "src/contracts/multi_day_route_state.json"
    os.makedirs(os.path.dirname(multi_day_path), exist_ok=True)
    with open(multi_day_path, "w") as f:
        json.dump(all_days_data, f, indent=2)
    print(f"\n💾 Full 7-Day Route State saved to: {multi_day_path}")

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
