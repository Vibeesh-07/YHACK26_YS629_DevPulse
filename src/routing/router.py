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


def slice_polyline_by_distance(polyline, target_dist_nm):
    """
    Slices a polyline from index 0 up to target_dist_nm.
    Returns:
      (sliced_points, end_coords, heading_at_end)
    """
    if not polyline:
        return [], [0.0, 0.0], 0.0

    if len(polyline) == 1 or target_dist_nm <= 1e-4:
        p0 = polyline[0]
        heading = 0.0
        if len(polyline) > 1:
            heading = calculate_bearing(p0[0], p0[1], polyline[1][0], polyline[1][1])
        return [[round(float(p0[0]), 4), round(float(p0[1]), 4)]], [round(float(p0[0]), 4), round(float(p0[1]), 4)], heading

    accum = 0.0
    pts = [[round(float(polyline[0][0]), 4), round(float(polyline[0][1]), 4)]]
    heading = 0.0

    for i in range(len(polyline) - 1):
        p1 = polyline[i]
        p2 = polyline[i + 1]
        seg_dist = haversine_nm(p1[0], p1[1], p2[0], p2[1])

        if accum + seg_dist >= target_dist_nm:
            remain = target_dist_nm - accum
            frac = remain / seg_dist if seg_dist > 1e-6 else 0.0
            lat = p1[0] + frac * (p2[0] - p1[0])
            lon = p1[1] + frac * (p2[1] - p1[1])
            heading = calculate_bearing(p1[0], p1[1], p2[0], p2[1])
            end_pt = [round(float(lat), 4), round(float(lon), 4)]
            pts.append(end_pt)
            return pts, end_pt, heading

        accum += seg_dist
        pts.append([round(float(p2[0]), 4), round(float(p2[1]), 4)])

    end_pt = polyline[-1]
    if len(polyline) >= 2:
        heading = calculate_bearing(polyline[-2][0], polyline[-2][1], end_pt[0], end_pt[1])
    return pts, [round(float(end_pt[0]), 4), round(float(end_pt[1]), 4)], heading


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
    Receding-Horizon Dynamic Routing:
      On each day d, the route is dynamically planned from the vessel's CURRENT POSITION (P_d)
      to the destination, avoiding day d's drifting icebergs and hazards.
      The historical sailed track from Start up to P_d remains fixed as the sailed voyage history.
    """
    start_coords = [round(float(step1_state["start_coords"][0]), 4), round(float(step1_state["start_coords"][1]), 4)]
    dest_coords = [round(float(step1_state["dest_coords"][0]), 4), round(float(step1_state["dest_coords"][1]), 4)]
    bbox = step1_state["corridor_bbox"]
    forecast_days = step2_output.get("forecast_days", 7)
    daily_hazard_states = step2_output["daily_hazard_states"]

    from src.data.land_mask import build_land_mask_fn
    land_mask_fn = build_land_mask_fn()

    # Cost grid initialized lazily if obstacle avoidance is required
    cost_grid = None

    def get_or_build_cost_grid():
        nonlocal cost_grid
        if cost_grid is not None:
            return cost_grid
        from src.data.land_mask import build_land_mask_grid
        import numpy as _np_lm
        _tmp_lats = _np_lm.arange(bbox["min_lat"], bbox["max_lat"] + res_deg, res_deg)
        _tmp_lons = _np_lm.arange(bbox["min_lon"], bbox["max_lon"] + res_deg, res_deg)
        print(f"  Building land mask for {len(_tmp_lats)}x{len(_tmp_lons)} grid…")
        land_grid, _ = build_land_mask_grid(_tmp_lats, _tmp_lons)
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
        return cost_grid

    all_days_data = []

    # Ship progression state across the voyage
    current_pos = [start_coords[0], start_coords[1]]
    history_polyline = [current_pos]
    dist_traveled_so_far = 0.0

    # Fixed daily advance: ship travels a constant fraction of the baseline straight-line distance each day.
    # This is independent of how the A* route shape changes due to iceberg drift.
    # Using straight-line distance as baseline keeps the ship position deterministic.
    baseline_total_nm = haversine_nm(start_coords[0], start_coords[1], dest_coords[0], dest_coords[1])
    baseline_daily_nm = baseline_total_nm / float(forecast_days) if forecast_days > 1 else baseline_total_nm

    print(f"\n[Step 3] Solving Dynamic Multi-Day Routes from Current Vessel Positions (Days 1 to {forecast_days})...")
    print(f"         Baseline voyage: {round(baseline_total_nm, 1)} nm | Daily advance: {round(baseline_daily_nm, 1)} nm/day")

    for day_idx in range(forecast_days):
        day_num = day_idx + 1
        hazards = daily_hazard_states[day_idx]["hazards"] if day_idx < len(daily_hazard_states) else []

        rem_straight_nm = haversine_nm(current_pos[0], current_pos[1], dest_coords[0], dest_coords[1])

        # If vessel has arrived at destination or final arrival day
        if rem_straight_nm < 0.5 or (day_num == forecast_days and day_num > 1 and rem_straight_nm < 3.0):
            forward_polyline = [current_pos, dest_coords] if current_pos != dest_coords else [dest_coords]
            forward_dist_nm = round(rem_straight_nm, 1)
            is_direct = True
            nodes_eval = 0
            closest_h = calculate_closest_hazard_nm(forward_polyline, hazards)
            heading = 0.0
            if len(history_polyline) >= 2:
                heading = calculate_bearing(history_polyline[-2][0], history_polyline[-2][1], current_pos[0], current_pos[1])
        else:
            # 1. Fast Line-of-Sight Check from CURRENT POSITION to destination
            num_samples = max(20, min(80, int(rem_straight_nm / 3.0)))
            sample_lats = np.linspace(current_pos[0], dest_coords[0], num_samples)
            sample_lons = np.linspace(current_pos[1], dest_coords[1], num_samples)
            direct_waypoints = [[round(float(la), 4), round(float(lo), 4)] for la, lo in zip(sample_lats, sample_lons)]

            direct_hits_land = any(land_mask_fn(p[0], p[1]) for p in direct_waypoints)
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

            if not obstructed:
                # Fast path directly from current_pos to destination
                forward_polyline = direct_waypoints
                forward_dist_nm = round(rem_straight_nm, 1)
                is_direct = True
                nodes_eval = 0
                closest_h = round(min_h_dist, 1)
                heading = calculate_bearing(direct_waypoints[0][0], direct_waypoints[0][1],
                                            direct_waypoints[1][0], direct_waypoints[1][1]) if len(direct_waypoints) > 1 else 0.0
            else:
                # Standard 2D Risk-Aware A* pathfinding from CURRENT POSITION to destination
                cgrid = get_or_build_cost_grid()
                astar_res = find_risk_aware_route(cgrid, current_pos, dest_coords, hazards)
                smoothed_polyline = smooth_route_polyline(astar_res["waypoints"], land_mask_fn=land_mask_fn)

                fwd_d = 0.0
                for i in range(len(smoothed_polyline) - 1):
                    fwd_d += haversine_nm(smoothed_polyline[i][0], smoothed_polyline[i][1],
                                          smoothed_polyline[i + 1][0], smoothed_polyline[i + 1][1])
                forward_polyline = smoothed_polyline
                forward_dist_nm = round(fwd_d, 1)
                is_direct = astar_res.get("is_direct", False)
                nodes_eval = astar_res.get("nodes_evaluated", 0)
                closest_h = calculate_closest_hazard_nm(forward_polyline, hazards)
                heading = calculate_bearing(forward_polyline[0][0], forward_polyline[0][1],
                                            forward_polyline[1][0], forward_polyline[1][1]) if len(forward_polyline) > 1 else 0.0

        # Full combined route for Day d: sailed history + dynamic forward route
        full_route = history_polyline[:-1] + forward_polyline

        total_voyage_dist_nm = round(dist_traveled_so_far + forward_dist_nm, 1)
        total_hours = max(1.0, float(forecast_days) * 24.0)
        avg_speed_knots = round(total_voyage_dist_nm / total_hours, 1)
        daily_run_nm = round(total_voyage_dist_nm / max(1, forecast_days), 1)
        progress_pct = round(min(100.0, (dist_traveled_so_far / max(1.0, total_voyage_dist_nm)) * 100.0), 1)

        confidence_pct = max(65, 99 - day_idx * 2)
        if closest_h < 3.0:
            confidence_pct -= 10
        elif closest_h < 5.0:
            confidence_pct -= 4
        confidence_pct = max(50, min(99, confidence_pct))

        ship_info = {
            "coords": [round(float(current_pos[0]), 4), round(float(current_pos[1]), 4)],
            "heading_deg": heading,
            "distance_traveled_nm": round(dist_traveled_so_far, 1),
            "distance_remaining_nm": round(forward_dist_nm, 1),
            "progress_pct": progress_pct,
            "average_speed_knots": avg_speed_knots,
            "daily_distance_nm": daily_run_nm
        }

        print(f"  * Day {day_num}: Ship at {ship_info['coords']} ({ship_info['progress_pct']}%) | Dynamic route from current pos: {forward_dist_nm} nm ahead (Traveled: {ship_info['distance_traveled_nm']} nm) | Speed: {avg_speed_knots} kn | Clearance: {closest_h} nm | Nodes: {nodes_eval}")

        day_payload = {
            "day": day_num,
            "total_days": forecast_days,
            "status": {
                "hazards_nearby": len(hazards),
                "route_confidence_pct": confidence_pct,
                "is_direct_path": is_direct
            },
            "kpis": {
                "route_distance_nm": total_voyage_dist_nm,
                "closest_hazard_nm": closest_h,
                "icebergs_tracked": len(hazards),
                "nodes_evaluated": nodes_eval,
                "average_speed_knots": avg_speed_knots,
                "daily_distance_nm": daily_run_nm,
                "distance_traveled_nm": round(dist_traveled_so_far, 1),
                "distance_remaining_nm": round(forward_dist_nm, 1),
                "progress_pct": progress_pct
            },
            "navigation": {
                "start": {"name": "Start", "coords": start_coords},
                "current_position": {"name": "Vessel Position", "coords": current_pos},
                "destination": {"name": "Destination", "coords": dest_coords},
                "history_polyline": history_polyline,
                "forward_polyline": forward_polyline,
                "route_polyline": full_route,
                "raw_waypoints_count": len(forward_polyline),
                "smoothed_points_count": len(full_route),
                "is_direct_path": is_direct
            },
            "ship": ship_info,
            "hazards": hazards
        }
        all_days_data.append(day_payload)

        # ── Advance the vessel for the next day along the CURRENT DAY'S forward route ──
        # We advance by exactly baseline_daily_nm (fixed per-day travel, 1/N of total voyage).
        # This keeps ship progression independent of each day's A* route shape (which changes
        # as icebergs drift). The forward route is re-planned fresh each day from the new position.
        if day_num < forecast_days:
            sail_step_nm = baseline_daily_nm
            sliced_pts, next_pos, seg_heading = slice_polyline_by_distance(forward_polyline, sail_step_nm)

            dist_traveled_so_far += sail_step_nm
            history_polyline = history_polyline[:-1] + sliced_pts
            current_pos = next_pos

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


if __name__ == "__main__":
    from src.data.corridor import execute_step1
    from src.physics.drift_engine import execute_step2

    sample_start = [-63.5, -58.2]
    sample_dest = [-60.8, -52.4]

    s1 = execute_step1(sample_start, sample_dest, "2026-09-10T00:00:00Z")
    s2 = execute_step2(s1, forecast_days=7)
    multiday = solve_multiday_routes(s1, s2)
    print("\nA* Multi-Day Routing complete!")
