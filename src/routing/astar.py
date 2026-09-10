"""
src/routing/astar.py
====================
Stage 6: Risk-Aware A* Pathfinding Router.

Executes an 8-direction A* search across the 2D spherical navigation mesh:
  * f(n) = g(n) + h(n)
  * g(n): Cumulative transit cost along the path, heavily penalizing cells
          inside or near Monte Carlo 95% hazard buffer zones.
  * h(n): Admissible great-circle Haversine distance from node n to destination.
  * Guarantees optimal, collision-free waypoint routing around drifting icebergs.
"""

import heapq
import numpy as np
from src.physics.monte_carlo import haversine_nm

BASE_COST = 1.0


def check_direct_path_clear(cost_grid, start_coords, dest_coords, hazards, num_samples=60, safety_margin_nm=1.0):
    """
    Fast Line-of-Sight (LOS) Obstacle Check.
    Determines if the direct geodesic straight line between start and destination
    is completely free of land obstacles, sea-ice, and iceberg hazard zones.

    Returns:
      (is_clear, reason, closest_hazard_nm, direct_waypoints)
    """
    lats = np.linspace(start_coords[0], dest_coords[0], num_samples)
    lons = np.linspace(start_coords[1], dest_coords[1], num_samples)
    direct_waypoints = [[round(float(la), 4), round(float(lo), 4)] for la, lo in zip(lats, lons)]

    # 1. Continuous geometry line intersection check
    try:
        from src.data.land_mask import _build_merged_geometry
        from shapely.geometry import LineString
        geom = _build_merged_geometry()
        if geom is not None:
            line = LineString([(start_coords[1], start_coords[0]), (dest_coords[1], dest_coords[0])])
            if line.intersects(geom):
                return False, "Direct path intersects coastline/island geometry", 0.0, []
    except Exception:
        pass

    # 1b. Check land collisions along direct path sample points
    if cost_grid.land_mask_fn is not None:
        for p in direct_waypoints:
            if cost_grid.land_mask_fn(p[0], p[1]):
                return False, f"Land intersects direct path at ({p[0]}, {p[1]})", 0.0, []

    # 2. Check cost grid impassable cells (impassable mask)
    for p in direct_waypoints:
        r, c = cost_grid.coords_to_node(p[0], p[1])
        if cost_grid.impassable[r, c]:
            return False, f"Grid obstacle intersects direct path at node ({r}, {c})", 0.0, []

    # 3. Check if any iceberg projects onto the path
    min_dist_to_hazard = 999.0
    for h in hazards:
        center = h.get("center", [h.get("lat"), h.get("lon")])
        if center[0] is None or center[1] is None:
            continue
        r_buffer = float(h.get("buffer_radius_nm", 5.0))
        # Distance from all sample points on the direct line to this hazard
        dists = [haversine_nm(p[0], p[1], center[0], center[1]) for p in direct_waypoints]
        min_d = min(dists)
        if min_d < min_dist_to_hazard:
            min_dist_to_hazard = min_d
        # If the hazard's buffer zone intersects or passes too close to the direct line:
        if min_d <= (r_buffer + safety_margin_nm):
            return False, f"Iceberg {h.get('id', 'hazard')} projects onto path (dist {min_d:.2f} nm <= buffer {r_buffer:.2f} nm)", min_dist_to_hazard, []

    return True, "Direct path is clear (no land, no icebergs project onto path)", min_dist_to_hazard, direct_waypoints


def find_risk_aware_route(cost_grid, start_coords, dest_coords, hazards):
    """
    Finds the optimal collision-free waypoint path from start to destination.

    Fast Path: If no icebergs project onto the path and no land blocks it,
    immediately returns the straight geodesic path with zero graph-search overhead.
    Otherwise: Executes risk-aware A* pathfinding.

    Returns:
      dict with waypoints, total_distance_nm, path_cost, and nodes_evaluated.
    """
    # Ensure start and dest coordinates are in open water
    if cost_grid.land_mask_fn:
        from src.data.land_mask import find_nearest_water_coord
        if cost_grid.land_mask_fn(start_coords[0], start_coords[1]):
            start_coords = find_nearest_water_coord(start_coords, cost_grid.land_mask_fn)
        if cost_grid.land_mask_fn(dest_coords[0], dest_coords[1]):
            dest_coords = find_nearest_water_coord(dest_coords, cost_grid.land_mask_fn)

    # ── Fast Path: Check if direct line is unobstructed ──────────────────────
    is_clear, reason, closest_h_dist, direct_waypoints = check_direct_path_clear(
        cost_grid, start_coords, dest_coords, hazards
    )
    if is_clear:
        direct_dist = haversine_nm(start_coords[0], start_coords[1], dest_coords[0], dest_coords[1])
        return {
            "success": True,
            "is_direct": True,
            "waypoints": direct_waypoints,
            "total_distance_nm": round(direct_dist, 2),
            "closest_hazard_nm": round(closest_h_dist, 2),
            "path_cost": round(direct_dist, 2),
            "nodes_evaluated": 0
        }

    # ── Standard A*: Overlay dynamic hazards onto grid ────────────────────────
    cost_matrix, impassable_mask = cost_grid.build_daily_cost_matrix(hazards)

    start_node = cost_grid.coords_to_node(start_coords[0], start_coords[1])
    goal_node = cost_grid.coords_to_node(dest_coords[0], dest_coords[1])

    # Ensure start and goal are navigable
    if impassable_mask[start_node]:
        start_node = _find_nearest_passable(cost_grid, start_node, impassable_mask)
    if impassable_mask[goal_node]:
        goal_node = _find_nearest_passable(cost_grid, goal_node, impassable_mask)

    dest_lat, dest_lon = cost_grid.node_to_coords(goal_node[0], goal_node[1])

    def heuristic(r, c):
        lat, lon = cost_grid.node_to_coords(r, c)
        return haversine_nm(lat, lon, dest_lat, dest_lon) * BASE_COST

    # Priority queue: (f_score, h_score, (r, c))
    open_heap = []
    h_start = heuristic(start_node[0], start_node[1])
    heapq.heappush(open_heap, (h_start, h_start, start_node))

    g_score = np.full((cost_grid.nrows, cost_grid.ncols), np.inf, dtype=np.float64)
    g_score[start_node] = 0.0

    came_from = {}
    visited = set()
    nodes_evaluated = 0

    while open_heap:
        f, h, current = heapq.heappop(open_heap)

        if current in visited:
            continue
        visited.add(current)
        nodes_evaluated += 1

        if current == goal_node:
            # Reconstruct path
            path = [goal_node]
            curr = goal_node
            while curr in came_from:
                curr = came_from[curr]
                path.append(curr)
            path.reverse()

            waypoints = []
            waypoints.append([float(start_coords[0]), float(start_coords[1])])
            for r, c in path[1:-1]:
                lat, lon = cost_grid.node_to_coords(r, c)
                waypoints.append([round(lat, 4), round(lon, 4)])
            waypoints.append([float(dest_coords[0]), float(dest_coords[1])])

            # Calculate actual nautical distance along waypoints
            total_dist_nm = 0.0
            for i in range(len(waypoints) - 1):
                p1, p2 = waypoints[i], waypoints[i + 1]
                total_dist_nm += haversine_nm(p1[0], p1[1], p2[0], p2[1])

            return {
                "success": True,
                "waypoints": waypoints,
                "total_distance_nm": round(total_dist_nm, 2),
                "path_cost": round(float(g_score[goal_node]), 2),
                "nodes_evaluated": nodes_evaluated
            }

        r, c = current
        cur_cost = cost_matrix[r, c]

        for nr, nc, step_dist in cost_grid.get_neighbors(r, c, impassable_mask):
            if (nr, nc) in visited:
                continue

            # Edge cost = distance * average cell penalty
            neighbor_cost = cost_matrix[nr, nc]
            edge_penalty = (cur_cost + neighbor_cost) / 2.0
            tentative_g = g_score[r, c] + step_dist * edge_penalty

            if tentative_g < g_score[nr, nc]:
                came_from[(nr, nc)] = current
                g_score[nr, nc] = tentative_g
                h_n = heuristic(nr, nc)
                f_n = tentative_g + h_n
                heapq.heappush(open_heap, (f_n, h_n, (nr, nc)))

    # Fallback if path blocked completely: navigate to the closest reachable water position
    if visited:
        best_visited = min(visited, key=lambda n: heuristic(n[0], n[1]))
        path = [best_visited]
        curr = best_visited
        while curr in came_from:
            curr = came_from[curr]
            path.append(curr)
        path.reverse()
        fallback_wps = [[float(start_coords[0]), float(start_coords[1])]]
        for r, c in path[1:]:
            lat, lon = cost_grid.node_to_coords(r, c)
            fallback_wps.append([round(lat, 4), round(lon, 4)])
        return {
            "success": False,
            "waypoints": fallback_wps,
            "total_distance_nm": round(haversine_nm(start_coords[0], start_coords[1], dest_coords[0], dest_coords[1]), 2),
            "path_cost": float("inf"),
            "nodes_evaluated": nodes_evaluated
        }

    return {
        "success": False,
        "waypoints": [[start_coords[0], start_coords[1]], [dest_coords[0], dest_coords[1]]],
        "total_distance_nm": round(haversine_nm(start_coords[0], start_coords[1], dest_coords[0], dest_coords[1]), 2),
        "path_cost": float("inf"),
        "nodes_evaluated": nodes_evaluated
    }


def _find_nearest_passable(cost_grid, node, impassable_mask, max_radius=8):
    """Find the closest navigable cell if start or destination falls on an obstacle."""
    r0, c0 = node
    for radius in range(1, max_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < cost_grid.nrows and 0 <= c < cost_grid.ncols:
                    if not impassable_mask[r, c]:
                        return (r, c)
    return node
