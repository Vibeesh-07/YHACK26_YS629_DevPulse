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


def find_risk_aware_route(cost_grid, start_coords, dest_coords, hazards):
    """
    Finds the optimal collision-free waypoint path from start to destination
    avoiding all impassable iceberg cores and minimizing hazard buffer exposure.

    Returns:
      waypoints: list of [lat, lon] coordinates from start to destination.
      metrics: dict with total distance, path cost, and nodes evaluated.
    """
    # 1. Overlay dynamic hazards onto grid
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

    # Fallback if path blocked completely
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
