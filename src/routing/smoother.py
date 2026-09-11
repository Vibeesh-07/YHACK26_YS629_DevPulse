"""
src/routing/smoother.py
=======================
Stage 6: Nautical Spline Path Smoother.

Converts discrete, jagged grid waypoints produced by A* into a smooth,
continuous shipping polyline suitable for navigation displays and auto-pilot routing.
Uses Catmull-Rom cubic spline interpolation with line-of-sight waypoint pruning.
"""

import numpy as np
from src.physics.monte_carlo import haversine_nm


def catmull_rom_spline(P0, P1, P2, P3, num_points=10):
    """
    Evaluates Catmull-Rom cubic spline between P1 and P2 given control points P0 and P3.
    """
    t = np.linspace(0, 1, num_points)
    t2 = t * t
    t3 = t2 * t

    # Catmull-Rom basis matrix coefficients
    q0 = -t3 + 2.0 * t2 - t
    q1 = 3.0 * t3 - 5.0 * t2 + 2.0
    q2 = -3.0 * t3 + 4.0 * t2 + t
    q3 = t3 - t2

    pts = []
    for i in range(num_points):
        lat = 0.5 * (P0[0] * q0[i] + P1[0] * q1[i] + P2[0] * q2[i] + P3[0] * q3[i])
        lon = 0.5 * (P0[1] * q0[i] + P1[1] * q1[i] + P2[1] * q2[i] + P3[1] * q3[i])
        pts.append([round(float(lat), 4), round(float(lon), 4)])
    return pts


def prune_waypoints(waypoints, min_distance_nm=4.0, land_mask_fn=None):
    """
    Prune excessively dense collinear waypoints before spline generation.
    Preserves intermediate waypoints if skipping them would cause the line segment
    to intersect any landmass.
    """
    if len(waypoints) <= 3:
        return waypoints

    pruned = [waypoints[0]]
    for i in range(1, len(waypoints) - 1):
        prev = pruned[-1]
        curr = waypoints[i]
        next_pt = waypoints[i + 1]
        d = haversine_nm(prev[0], prev[1], curr[0], curr[1])
        if d >= min_distance_nm:
            # If land_mask_fn is provided, ensure shortcut from prev to curr doesn't cut land
            if land_mask_fn is not None:
                from src.data.land_mask import check_line_intersects_land
                if check_line_intersects_land(prev, curr, land_mask_fn=land_mask_fn):
                    # Direct jump intersects land, keep the preceding waypoint
                    if waypoints[i - 1] != prev:
                        pruned.append(waypoints[i - 1])
            pruned.append(curr)
        elif land_mask_fn is not None:
            # If skipping curr causes direct line from prev to next_pt to clip land, keep curr!
            sample_lats = np.linspace(prev[0], next_pt[0], 6)
            sample_lons = np.linspace(prev[1], next_pt[1], 6)
            if any(land_mask_fn(la, lo) for la, lo in zip(sample_lats, sample_lons)):
                pruned.append(curr)

    # Check connection to final destination
    if land_mask_fn is not None and len(pruned) > 0:
        from src.data.land_mask import check_line_intersects_land
        if check_line_intersects_land(pruned[-1], waypoints[-1], land_mask_fn=land_mask_fn):
            pruned.append(waypoints[-2])

    pruned.append(waypoints[-1])
    return pruned


def smooth_route_polyline(waypoints, density_per_segment=6, land_mask_fn=None):
    """
    Takes discrete A* waypoints and generates a smooth, curved nautical route polyline.
    Strictly verifies each spline segment and ensures 100% of waypoints remain in open water.
    """
    if len(waypoints) <= 2:
        return waypoints

    clean_pts = prune_waypoints(waypoints, land_mask_fn=land_mask_fn)
    if len(clean_pts) <= 2:
        return waypoints

    # Extend endpoints to act as phantom control points
    p_start = [
        2.0 * clean_pts[0][0] - clean_pts[1][0],
        2.0 * clean_pts[0][1] - clean_pts[1][1]
    ]
    p_end = [
        2.0 * clean_pts[-1][0] - clean_pts[-2][0],
        2.0 * clean_pts[-1][1] - clean_pts[-2][1]
    ]
    extended = [p_start] + clean_pts + [p_end]

    smoothed = [clean_pts[0]]
    for i in range(1, len(extended) - 2):
        p0, p1, p2, p3 = extended[i - 1], extended[i], extended[i + 1], extended[i + 2]
        segment_pts = catmull_rom_spline(p0, p1, p2, p3, num_points=density_per_segment)

        # Check if spline segment clipped land
        clipped_land = False
        if land_mask_fn is not None:
            for pt in segment_pts:
                if land_mask_fn(pt[0], pt[1]):
                    clipped_land = True
                    break

        if clipped_land:
            # Fallback to linear segment between p1 and p2
            lats = np.linspace(p1[0], p2[0], density_per_segment)
            lons = np.linspace(p1[1], p2[1], density_per_segment)
            linear_pts = [[round(float(la), 4), round(float(lo), 4)] for la, lo in zip(lats, lons)]
            # If linear fallback also touches land, strictly use endpoints
            if land_mask_fn is not None and any(land_mask_fn(pt[0], pt[1]) for pt in linear_pts):
                segment_pts = [p1, p2]
            else:
                segment_pts = linear_pts

        smoothed.extend(segment_pts[1:])

    # Guarantee destination matches exactly
    smoothed[-1] = clean_pts[-1]

    # Airtight final guarantee: snap any stray points touching land to nearest navigable water
    if land_mask_fn is not None:
        from src.data.land_mask import find_nearest_water_coord
        smoothed = [
            find_nearest_water_coord(pt, land_mask_fn) if land_mask_fn(pt[0], pt[1]) else pt
            for pt in smoothed
        ]

    return smoothed
