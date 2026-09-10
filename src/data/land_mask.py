"""
src/data/land_mask.py
=====================
Robust land-avoidance mask for Antarctic ship routing.

Uses Natural Earth 50m land + glaciated-areas GeoJSON (via shapely 2.x) to
build a precomputed boolean grid.  Covers Antarctica, South Georgia, Falklands,
South Shetland Islands, and all continental coastlines.

Falls back to a hard-coded geographic heuristic if the network is unavailable.
"""

import os
import json
import logging
import urllib.request
import numpy as np

log = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "../../.cache")

# Natural Earth 50m land + glaciated areas (for Antarctica ice sheet)
_SOURCES = {
    "land_50m": (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector"
        "/master/geojson/ne_50m_land.geojson",
        os.path.join(_CACHE_DIR, "ne_50m_land.geojson"),
    ),
    "glaciated_50m": (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector"
        "/master/geojson/ne_50m_glaciated_areas.geojson",
        os.path.join(_CACHE_DIR, "ne_50m_glaciated.geojson"),
    ),
}


# ─── GeoJSON Loader ──────────────────────────────────────────────────────────

def _fetch_geojson(url, cache_path):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    if not os.path.exists(cache_path):
        log.info(f"Downloading {url} …")
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.load(resp)
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            log.warning(f"Download failed ({e}). Falling back.")
            return None
    with open(cache_path) as f:
        return json.load(f)


# ─── Build shapely geometry ──────────────────────────────────────────────────

def _build_merged_geometry():
    """Return a single merged shapely geometry of all land + ice, or None."""
    try:
        from shapely.geometry import shape
        from shapely.ops import unary_union

        polys = []
        for key, (url, cache) in _SOURCES.items():
            data = _fetch_geojson(url, cache)
            if data is None:
                continue
            for feat in data.get("features", []):
                try:
                    polys.append(shape(feat["geometry"]))
                except Exception:
                    pass

        if not polys:
            return None

        merged = unary_union(polys)
        log.info(f"Land geometry built from {len(polys)} polygons.")
        return merged

    except Exception as e:
        log.warning(f"shapely geometry build failed: {e}")
        return None


# ─── Public API ──────────────────────────────────────────────────────────────

def build_land_mask_fn():
    """
    Returns callable  land_mask_fn(lat, lon) -> bool.
    True  = land / ice = impassable.

    Combines shapely polygon check (Natural Earth 50m) with a geographic
    heuristic so that Antarctica, South Shetlands, and small islands are
    always correctly identified even where polygon precision is insufficient.
    """
    geom = _build_merged_geometry()

    if geom is not None:
        try:
            from shapely.prepared import prep
            from shapely.geometry import Point
            prepared = prep(geom)

            def land_mask_fn(lat, lon):
                # OR: land if shapely polygon OR heuristic says so
                return prepared.contains(Point(lon, lat)) or _fallback_land_mask(lat, lon)

            return land_mask_fn
        except Exception as e:
            log.warning(f"PreparedGeometry failed: {e}")

    log.warning("Using hard-coded land heuristic only.")
    return _fallback_land_mask


def build_land_mask_grid(lats, lons):
    """
    Precomputes a 2D boolean numpy array (nrows × ncols) where True = land.
    Combines shapely polygon result with geographic heuristic (OR logic).
    Returns: (land_grid, land_mask_fn)
    """
    nrows, ncols = len(lats), len(lons)
    land_grid = np.zeros((nrows, ncols), dtype=bool)

    # Always build the heuristic grid first (fast, catches Antarctica etc.)
    LAT_2D, LON_2D = np.meshgrid(lats, lons, indexing="ij")
    heuristic_grid = np.vectorize(_fallback_land_mask)(LAT_2D, LON_2D)
    land_grid = np.logical_or(land_grid, heuristic_grid)

    # Overlay shapely polygons (more accurate for coastlines)
    geom = _build_merged_geometry()
    if geom is not None:
        try:
            from shapely import contains_xy
            flat = contains_xy(geom, LON_2D.ravel(), LAT_2D.ravel())
            shapely_grid = flat.reshape(nrows, ncols)
            land_grid = np.logical_or(land_grid, shapely_grid)
            log.info(f"Combined land grid: {land_grid.sum()}/{land_grid.size} cells blocked.")
        except AttributeError:
            # Older shapely — row-by-row
            from shapely.prepared import prep
            from shapely.geometry import Point
            prepared = prep(geom)
            for r, lat in enumerate(lats):
                for c, lon in enumerate(lons):
                    if prepared.contains(Point(lon, lat)):
                        land_grid[r, c] = True
            log.info(f"Land grid (iterative): {land_grid.sum()}/{land_grid.size} cells blocked.")
        except Exception as e:
            log.warning(f"Shapely grid scan error: {e}. Using heuristic-only grid.")

    land_fn = build_land_mask_fn()
    return land_grid, land_fn


# ─── Hard-coded Fallback ──────────────────────────────────────────────────────

def _fallback_land_mask(lat, lon):
    """
    Geographic heuristic for the Weddell / Scotia corridor.
    Covers the main features at routing resolution.
    """
    # Antarctic continent / ice sheet
    if lat < -73.0:
        return True
    # Antarctic Peninsula
    if lat < -62.0 and -68.0 <= lon <= -55.0:
        return True
    # South Georgia Island
    if -55.5 <= lat <= -53.5 and -38.5 <= lon <= -35.5:
        return True
    # Falkland / Malvinas
    if -53.0 <= lat <= -51.0 and -61.5 <= lon <= -57.0:
        return True
    # South Shetland Islands
    if -63.0 <= lat <= -61.5 and -62.0 <= lon <= -54.0:
        return True
    return False


# ─── CLI test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    fn = build_land_mask_fn()
    tests = [
        (-54.5,  -36.5, "South Georgia", True),
        (-51.7,  -57.8, "Falklands",     True),
        (-75.0,  -26.0, "Antarctica",    True),
        (-63.5,  -62.0, "South Shetlands", True),
        (-60.0,  -45.0, "Scotia Sea",    False),
        (-62.0,  -55.0, "Drake Pass.",   False),
        (-60.8,  -52.4, "Route dest",    False),
    ]
    print("\nLand mask accuracy test:")
    for lat, lon, desc, expected in tests:
        got = fn(lat, lon)
        status = "OK" if got == expected else "FAIL"
        print(f"  [{status}] {desc} ({lat},{lon}): expected={expected}, got={got}")
