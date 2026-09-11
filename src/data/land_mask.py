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

_CACHED_MERGED_GEOM = None


def _build_merged_geometry():
    """Return a single merged shapely geometry of all land + ice, or None. Cached in memory."""
    global _CACHED_MERGED_GEOM
    if _CACHED_MERGED_GEOM is not None:
        return _CACHED_MERGED_GEOM
    try:
        from shapely.geometry import shape
        from shapely.ops import unary_union
        import shapely

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
        if hasattr(shapely, "prepare"):
            shapely.prepare(merged)
        log.info(f"Land geometry built from {len(polys)} polygons and prepared.")
        _CACHED_MERGED_GEOM = merged
        return _CACHED_MERGED_GEOM

    except Exception as e:
        log.warning(f"shapely geometry build failed: {e}")
        return None


# ─── Public API ──────────────────────────────────────────────────────────────

def build_land_mask_fn():
    """
    Returns callable land_mask_fn(lat, lon) -> bool.
    True = land / ice = impassable.
    Ultra-fast C evaluation using shapely.contains_xy against prepared geometry.
    """
    geom = _build_merged_geometry()

    if geom is not None:
        try:
            import shapely
            from shapely import contains_xy
            if hasattr(shapely, "prepare"):
                shapely.prepare(geom)

            def land_mask_fn(lat, lon):
                if lat < -72.0:
                    return True
                return bool(contains_xy(geom, float(lon), float(lat)))

            return land_mask_fn
        except Exception as e:
            try:
                from shapely.prepared import prep
                from shapely.geometry import Point
                prepared = prep(geom)

                def land_mask_fn(lat, lon):
                    if lat < -72.0:
                        return True
                    return prepared.contains(Point(lon, lat))

                return land_mask_fn
            except Exception as e2:
                log.warning(f"PreparedGeometry failed: {e2}")

    log.warning("Using hard-coded land heuristic only.")
    return _fallback_land_mask


def build_land_mask_grid(lats, lons):
    """
    Precomputes a 2D boolean numpy array (nrows × ncols) where True = land.
    Uses Natural Earth 50m vector polygons clipped to the corridor bbox for sub-millisecond execution.
    Returns: (land_grid, land_mask_fn)
    """
    nrows, ncols = len(lats), len(lons)
    land_grid = np.zeros((nrows, ncols), dtype=bool)

    LAT_2D, LON_2D = np.meshgrid(lats, lons, indexing="ij")
    # Polar ice sheet interior
    land_grid[LAT_2D < -72.0] = True

    # Overlay shapely polygons (authoritative for all coastlines & islands)
    geom = _build_merged_geometry()
    if geom is not None:
        try:
            import shapely
            from shapely import contains_xy
            from shapely.geometry import box

            min_lat, max_lat = float(np.min(lats)), float(np.max(lats))
            min_lon, max_lon = float(np.min(lons)), float(np.max(lons))
            corridor_box = box(min_lon - 0.5, min_lat - 0.5, max_lon + 0.5, max_lat + 0.5)

            local_geom = geom.intersection(corridor_box)
            if hasattr(shapely, "prepare"):
                shapely.prepare(local_geom)

            flat = contains_xy(local_geom, LON_2D.ravel(), LAT_2D.ravel())
            shapely_grid = flat.reshape(nrows, ncols)
            land_grid = np.logical_or(land_grid, shapely_grid)
            log.info(f"Combined land grid: {land_grid.sum()}/{land_grid.size} cells blocked.")
        except Exception as e:
            try:
                from shapely.prepared import prep
                from shapely.geometry import Point
                prepared = prep(geom)
                for r, lat in enumerate(lats):
                    for c, lon in enumerate(lons):
                        if prepared.contains(Point(lon, lat)):
                            land_grid[r, c] = True
                log.info(f"Land grid (iterative): {land_grid.sum()}/{land_grid.size} cells blocked.")
            except Exception as e2:
                log.warning(f"Shapely grid scan error: {e2}. Using heuristic fallback.")
                heuristic_grid = np.vectorize(_fallback_land_mask)(LAT_2D, LON_2D)
                land_grid = np.logical_or(land_grid, heuristic_grid)
    else:
        heuristic_grid = np.vectorize(_fallback_land_mask)(LAT_2D, LON_2D)
        land_grid = np.logical_or(land_grid, heuristic_grid)

    land_fn = build_land_mask_fn()
    return land_grid, land_fn


def _fallback_land_mask(lat, lon):
    """
    Geographic heuristic for the Weddell / Scotia corridor.
    Covers the core features with accurate geographic bounds.
    """
    # 1. Antarctic continent interior & deep shelf
    if lat < -72.0:
        return True
    # 2. Falkland Islands (East & West Falkland)
    if -52.4 <= lat <= -51.2 and -61.2 <= lon <= -57.6:
        return True
    # 3. South Georgia Island core (Shapely handles detailed coastline)
    if -54.75 <= lat <= -54.10 and -37.8 <= lon <= -35.9:
        return True
    return False


def find_nearest_water_coord(coord, land_mask_fn=None, max_search_nm=120.0, standoff_nm=0.6):
    """
    If coord is on land according to land_mask_fn, radially searches outward
    to find the nearest navigable open-water coordinate (coastal snap).
    Adds a small standoff margin (standoff_nm) away from the coastline so
    vessels and routes do not sit right on coastal rocks or beaches.

    Returns:
      [snapped_lat, snapped_lon] (floats rounded to 4 decimal places)
      If coord is already in water or no water is found within max_search_nm,
      returns the original coord.
    """
    if coord is None or len(coord) < 2:
        return coord

    if land_mask_fn is None:
        land_mask_fn = build_land_mask_fn()

    lat0, lon0 = float(coord[0]), float(coord[1])
    if not land_mask_fn(lat0, lon0):
        return [round(lat0, 4), round(lon0, 4)]

    cos_lat = max(0.1, np.cos(np.radians(lat0)))
    # Start fine (0.5nm up to 30nm), then coarser (1.0nm up to max_search_nm)
    dists = np.concatenate([
        np.arange(0.5, 30.0, 0.5),
        np.arange(30.0, max_search_nm + 1.0, 1.0)
    ])

    for dist_nm in dists:
        deg_lat = dist_nm / 60.0
        deg_lon = dist_nm / (60.0 * cos_lat)
        for angle in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            test_lat = lat0 + deg_lat * np.sin(angle)
            test_lon = lon0 + deg_lon * np.cos(angle)
            if not land_mask_fn(test_lat, test_lon):
                # Found open water: apply coastal standoff in the outward direction
                standoff_lat = test_lat + (standoff_nm / 60.0) * np.sin(angle)
                standoff_lon = test_lon + (standoff_nm / (60.0 * cos_lat)) * np.cos(angle)
                if not land_mask_fn(standoff_lat, standoff_lon):
                    return [round(float(standoff_lat), 4), round(float(standoff_lon), 4)]
                return [round(float(test_lat), 4), round(float(test_lon), 4)]

    log.warning(f"Could not find open water within {max_search_nm} nm of ({lat0}, {lon0})")
    return [round(lat0, 4), round(lon0, 4)]


def check_line_intersects_land(p1, p2, land_mask_fn=None):
    """
    Rigorously checks whether the direct line between p1 and p2 intersects any landmass,
    island, coast, or continental shelf.

    Checks performed in order:
      1. Endpoints check: if p1 or p2 sits on land, returns True.
      2. Polar interior check: if path dips south of -72.0, returns True.
      3. Continuous Shapely LineString intersection against prepared 50m vector polygons.
      4. Dense sub-nautical-mile sampling (every 0.25 nm) against land_mask_fn.

    Returns:
      bool: True if ANY landmass is between p1 and p2, False if completely unobstructed water.
    """
    if p1 is None or p2 is None or len(p1) < 2 or len(p2) < 2:
        return True

    lat1, lon1 = float(p1[0]), float(p1[1])
    lat2, lon2 = float(p2[0]), float(p2[1])

    if land_mask_fn is None:
        land_mask_fn = build_land_mask_fn()

    # 1. Endpoints check
    if land_mask_fn(lat1, lon1) or land_mask_fn(lat2, lon2):
        return True

    # 2. Polar interior check
    if min(lat1, lat2) < -72.0:
        return True

    # 3. Continuous Shapely vector LineString intersection
    geom = _build_merged_geometry()
    if geom is not None:
        try:
            from shapely.geometry import LineString
            line = LineString([(lon1, lat1), (lon2, lat2)])
            if line.intersects(geom):
                return True
        except Exception as e:
            log.warning(f"LineString.intersects failed: {e}")

    # 4. Dense sub-nautical-mile sampling (0.25 nm intervals)
    cos_lat = max(0.1, np.cos(np.radians((lat1 + lat2) / 2.0)))
    dist_nm = np.hypot(lat2 - lat1, (lon2 - lon1) * cos_lat) * 60.0
    n_samples = max(30, int(dist_nm / 0.25))
    sample_lats = np.linspace(lat1, lat2, n_samples)
    sample_lons = np.linspace(lon1, lon2, n_samples)

    for la, lo in zip(sample_lats, sample_lons):
        if land_mask_fn(la, lo):
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
