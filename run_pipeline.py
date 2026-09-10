"""
run_pipeline.py
===============
End-to-End Orchestrator: Step 1 (Corridor & Iceberg Filter) -> Step 2 (Iceberg Drift Simulation)

Usage:
  python run_pipeline.py
  python run_pipeline.py --start-lat -63.5 --start-lon -58.2 --dest-lat -60.8 --dest-lon -52.4 --date 2026-09-10 --days 7
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Import modules from src
from src.data.corridor import execute_step1
from src.physics.drift_engine import execute_step2


def run_pipeline(start_coords, dest_coords, forecast_start_date=None, forecast_days=7, buffer_lat=1.5, buffer_lon=2.5):
    print("=" * 80)
    print(" 🧭 ANTARCTIC SEA-ICE & DRIFT PIPELINE: STEP 1 -> STEP 2")
    print("=" * 80)

    if forecast_start_date is None:
        forecast_start_date = "2026-09-10T00:00:00Z"

    print(f"\n[INPUT] Ship Start Coordinates       : {start_coords}")
    print(f"[INPUT] Ship Destination Coordinates : {dest_coords}")
    print(f"[INPUT] Departure / Forecast Date    : {forecast_start_date}")
    print(f"[INPUT] Forecast Horizon             : {forecast_days} days\n")

    # -------------------------------------------------------------
    # STEP 1: Geographic Corridor (AOI) & Iceberg Identification
    # -------------------------------------------------------------
    print("-" * 80)
    print("📍 STEP 1: Establishing Navigation Corridor & Filtering Iceberg Hazards")
    print("-" * 80)
    step1_output = execute_step1(
        start_coords=start_coords,
        dest_coords=dest_coords,
        forecast_start_date=forecast_start_date,
        buffer_lat=buffer_lat,
        buffer_lon=buffer_lon,
        save_path="src/contracts/initial_state.json"
    )

    bbox = step1_output["corridor_bbox"]
    icebergs = step1_output["icebergs"]

    print(f"✅ Bounding Box Defined (AOI) : Lat [{bbox['min_lat']}, {bbox['max_lat']}] | Lon [{bbox['min_lon']}, {bbox['max_lon']}]")
    print(f"✅ Active Icebergs Located     : {len(icebergs)} icebergs within hazard range")
    print(f"💾 Contract A Generated       : src/contracts/initial_state.json\n")

    for i, b in enumerate(icebergs, 1):
        print(f"   [{i}] {b['id']:<10} Position: ({b['lat']:>7.3f}, {b['lon']:>7.3f}) | Size: {b['length_km']} x {b['width_km']} km | Mass: {b['mass_kg']:.2e} kg")

    # -------------------------------------------------------------
    # STEP 2: Drift Model Integration (Forward Simulation)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("🌊 STEP 2: Running Physics Drift & Thermal Melt Integration (EWE20 Model)")
    print("-" * 80)

    # Directly pass Step 1 output into Step 2
    step2_output = execute_step2(
        step1_state=step1_output,
        forecast_days=forecast_days,
        output_contract="src/contracts/route_day_state.json"
    )

    print(f"💾 Contract B Generated       : src/contracts/route_day_state.json (Ready for A* routing)")

    print("\n" + "=" * 80)
    print(" 📊 7-DAY FORECAST SUMMARY FOR CORRIDOR ICEBERGS")
    print("=" * 80)

    for sim in step2_output["simulations"]:
        print(f"\n🧊 Iceberg {sim['id']}:")
        traj = sim["trajectory"]
        print(f"   {'Day':<5} | {'Date':<11} | {'Latitude':<10} | {'Longitude':<11} | {'Remaining Vol':<16} | {'Daily Melt Loss'}")
        print(f"   {'-'*4}-+-{'-'*11}-+-{'-'*10}-+-{'-'*11}-+-{'-'*16}-+-{'-'*15}")
        for pt in traj:
            vol_km3 = pt['volume_m3'] / 1.0e9
            melt_k_m3 = abs(pt['melt_loss_m3']) / 1.0e3
            print(f"   Day {pt['day']:<2}| {pt['date']:<11} | {pt['lat']:>8.4f}° | {pt['lon']:>9.4f}° | {vol_km3:>10.4f} km³   | {melt_k_m3:>10.2f} k m³")

    # -------------------------------------------------------------
    # STEP 3: Risk-Aware A* Routing & Spline Smoothing
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("🧭 STEP 3: Solving Multi-Day Risk-Aware A* Navigation Routes")
    print("-" * 80)
    from src.routing.router import solve_multiday_routes
    step3_output = solve_multiday_routes(step1_output, step2_output)

    print("\n" + "=" * 80)
    print(" 📊 7-DAY RECOMMENDED NAVIGATION ROUTES")
    print("=" * 80)
    print(f"{'Day':<6} | {'Route Distance':<16} | {'Closest Hazard':<16} | {'Confidence':<12} | {'A* Waypoints'}")
    print(f"{'-'*6}-+-{'-'*16}-+-{'-'*16}-+-{'-'*12}-+-{'-'*14}")
    for d_state in step3_output:
        kpis = d_state["kpis"]
        nav = d_state["navigation"]
        print(f"Day {d_state['day']:<2} | {kpis['route_distance_nm']:>10.1f} nm     | {kpis['closest_hazard_nm']:>10.2f} nm     | {d_state['status']['route_confidence_pct']:>8}%     | {nav['smoothed_points_count']} pts")

    print("\n" + "=" * 80)
    print(" 🚀 PIPELINE COMPLETE: STEPS 1 -> 2 -> 3 FULLY INTEGRATED")
    print("=" * 80)

    return {
        "step1": step1_output,
        "step2": step2_output,
        "step3": step3_output
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Step 1 -> Step 2 Antarctic Iceberg Drift Pipeline")
    parser.add_argument("--start-lat", type=float, default=-63.5, help="Ship start latitude (default: -63.5)")
    parser.add_argument("--start-lon", type=float, default=-58.2, help="Ship start longitude (default: -58.2)")
    parser.add_argument("--dest-lat", type=float, default=-60.8, help="Ship dest latitude (default: -60.8)")
    parser.add_argument("--dest-lon", type=float, default=-52.4, help="Ship dest longitude (default: -52.4)")
    parser.add_argument("--date", type=str, default="2026-09-10T00:00:00Z", help="Forecast start date (ISO string)")
    parser.add_argument("--days", type=int, default=7, help="Forecast horizon in days (default: 7)")

    args = parser.parse_args()
    run_pipeline(
        start_coords=[args.start_lat, args.start_lon],
        dest_coords=[args.dest_lat, args.dest_lon],
        forecast_start_date=args.date,
        forecast_days=args.days
    )
