"""
src/routing/cost_grid.py
========================
Stage 5: Dynamic 2D Navigation Cost Grid.

Constructs a spatial navigation mesh across the ship corridor bounding box:
  * Open Ocean (SIC < 15%): Base transit cost (1.0).
  * Sea Ice Caution (15% <= SIC <= 80%): Progressive penalty based on concentration.
  * Pack Ice (SIC > 80%): Impassable (infinity).
  * Land / Coastlines: Impassable (infinity).
  * Iceberg Core (d <= R_iceberg): Impassable collision zone (infinity).
  * Hazard Buffer Zone (R_iceberg < d <= R_buffer): Non-linear repulsion penalty
    derived from the 95% Monte Carlo uncertainty dispersion.
"""

import os
import sys
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.physics.monte_carlo import haversine_nm

INFINITY = 1e9
BASE_COST = 1.0
HAZARD_PENALTY_WEIGHT = 45.0


class CostGrid:
    """
    2D Discrete Navigation Cost Mesh for Antarctic polar ship routing.
    """

    def __init__(self, bbox, res_deg=0.06, land_mask_fn=None, sic_fn=None):
        self.min_lat = bbox["min_lat"]
        self.max_lat = bbox["max_lat"]
        self.min_lon = bbox["min_lon"]
        self.max_lon = bbox["max_lon"]
        self.res = res_deg

        self.lats = np.arange(self.min_lat, self.max_lat + self.res, self.res)
        self.lons = np.arange(self.min_lon, self.max_lon + self.res, self.res)
        self.nrows = len(self.lats)
        self.ncols = len(self.lons)

        self.land_mask_fn = land_mask_fn
        self.sic_fn = sic_fn

        # Precompute static base costs (land & sea-ice)
        self.base_costs = np.full((self.nrows, self.ncols), BASE_COST, dtype=np.float32)
        self.impassable = np.zeros((self.nrows, self.ncols), dtype=bool)

        for r, lat in enumerate(self.lats):
            for c, lon in enumerate(self.lons):
                # Check land
                if self.land_mask_fn and self.land_mask_fn(lat, lon):
                    self.impassable[r, c] = True
                    self.base_costs[r, c] = INFINITY
                    continue

                # Check sea ice
                if self.sic_fn:
                    sic = self.sic_fn(lat, lon)
                    if sic > 0.80:
                        self.impassable[r, c] = True
                        self.base_costs[r, c] = INFINITY
                    elif sic >= 0.15:
                        self.base_costs[r, c] = BASE_COST + 3.5 * float(sic)

    def coords_to_node(self, lat, lon):
        """Map (lat, lon) onto nearest grid (row, col)."""
        r = int(np.clip(np.round((lat - self.min_lat) / self.res), 0, self.nrows - 1))
        c = int(np.clip(np.round((lon - self.min_lon) / self.res), 0, self.ncols - 1))
        return (r, c)

    def node_to_coords(self, r, c):
        """Map grid (row, col) to (lat, lon)."""
        return float(self.lats[r]), float(self.lons[c])

    def build_daily_cost_matrix(self, hazards):
        """
        Takes dynamic hazards for a given day and overlays:
          - Impassable iceberg cores (infinity)
          - Repulsive hazard buffer zones (exponential/quadratic penalty)
        Returns a 2D numpy array of cell transit costs.
        """
        cost_matrix = self.base_costs.copy()
        impassable_mask = self.impassable.copy()

        LAT_2D, LON_2D = np.meshgrid(self.lats, self.lons, indexing="ij")

        for h in hazards:
            h_lat, h_lon = h["center"]
            r_core = h["iceberg_radius_nm"]
            r_buffer = h["buffer_radius_nm"]

            # Compute great-circle distance from all cells to this hazard center
            d_matrix = haversine_nm(LAT_2D, LON_2D, h_lat, h_lon)

            # 1. Impassable Iceberg Core
            core_mask = d_matrix <= r_core
            cost_matrix[core_mask] = INFINITY
            impassable_mask[core_mask] = True

            # 2. Dynamic Repulsion Zone (Buffer zone)
            buffer_mask = (d_matrix > r_core) & (d_matrix <= r_buffer)
            if np.any(buffer_mask):
                # Normalized distance into buffer (0 at outer edge, 1 at iceberg core)
                span = max(r_buffer - r_core, 0.1)
                severity = (r_buffer - d_matrix[buffer_mask]) / span
                penalty = HAZARD_PENALTY_WEIGHT * (severity ** 2)
                cost_matrix[buffer_mask] += penalty.astype(np.float32)

        return cost_matrix, impassable_mask

    def get_neighbors(self, r, c, impassable_mask):
        """
        Return 8-connected navigable neighbors: (nr, nc, step_distance_nm).
        """
        neighbors = []
        cur_lat = self.lats[r]
        cur_lon = self.lons[c]

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.nrows and 0 <= nc < self.ncols:
                    if not impassable_mask[nr, nc]:
                        n_lat = self.lats[nr]
                        n_lon = self.lons[nc]
                        dist_nm = haversine_nm(cur_lat, cur_lon, n_lat, n_lon)
                        neighbors.append((nr, nc, dist_nm))
        return neighbors


if __name__ == "__main__":
    sample_bbox = {"min_lat": -65.0, "max_lat": -59.0, "min_lon": -60.0, "max_lon": -50.0}
    grid = CostGrid(sample_bbox, res_deg=0.08)
    print(f"CostGrid initialized: {grid.nrows} rows x {grid.ncols} cols ({grid.nrows * grid.ncols} cells)")
    print(f"Cell size: ~{grid.res * 60:.1f} nautical miles")
