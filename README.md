# Antarctic Iceberg Tracking, Hazard Prediction & Safe Route Planning
**Challenge 9: AI-Enabled Antarctic Sea-Ice & Navigation Decision Support**  
*Repository: `Vibeesh-07/YHACK26_YS629_DevPulse`*

---

## 🧭 Executive Overview
This decision-support system takes a vessel's **Start (Lat/Lon)** and **Destination (Lat/Lon)** coordinates, establishes a dynamic navigation corridor (Area of Interest / AOI), tracks relevant Antarctic tabular icebergs, predicts their 7-day drift and calving breakup using the **England, Wagner & Eisenman (EWE20)** physics model, calculates Monte Carlo 95% uncertainty zones, and generates daily collision-free optimal routes using a risk-aware **A* Pathfinding Algorithm**.

---

## 🗺️ System UI Reference
The interactive dashboard interface is designed with clean, minimalist aesthetics, live metric cards, dynamic hazard rings, and an interactive 7-day timeline scrubber:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  Route status — day 3 of 7                               [ 3 hazards nearby ]  [ 87% route confidence ]  •••   │
├──────────────────────────────────────────────────────────┬─────────────────────────────────────────────┤
│                                                          │  ┌───────────────────────────────────────┐  │
│                                              Destination │  │ Route distance                        │  │
│                                                   ●      │  │ 142 nm                                │  │
│                   ╭ - - - ╮                              │  └───────────────────────────────────────┘  │
│                  '    ●    '                             │  ┌───────────────────────────────────────┐  │
│                   ╰ - - - ╯                              │  │ Closest hazard                        │  │
│                                                          │  │ 4.2 nm                                │  │
│        ╭ - - - ╮                                         │  └───────────────────────────────────────┘  │
│       '    ●    '                                        │  ┌───────────────────────────────────────┐  │
│        ╰ - - - ╯              ╭ - - - - - ╮              │  │ Icebergs tracked                      │  │
│                              '      ●      '             │  │ 3                                     │  │
│                               ╰ - - - - - ╯              │  └───────────────────────────────────────┘  │
│                                                          │                                             │
│       ●──────────────────────────────────────────────────╯                                             │
│     Start                                                                                              │
│                                                                                                        │
│   ● Iceberg (predicted)   ○ Hazard buffer (dashed)   ─ Recommended route                               │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│   [ ▶ ]  ═════════════════════════════●═══════════════════════════════════════                 Day 3 / 7 │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Parallel Team Work Segregation

To develop this platform concurrently without blocking, the system is split across **3 distinct developer streams** with mock data contracts defined upfront.

```
                                  PARALLEL WORKFLOW & DATA FLOW
                                  
  PERSON 1: DATA & AOI             PERSON 2: DRIFT & UNCERTAINTY        PERSON 3: ROUTING & DASHBOARD
 ┌───────────────────────────┐     ┌──────────────────────────────┐     ┌─────────────────────────────┐
 │ • Route Corridor (AOI)    │     │ • EWE20 Drift Physics        │     │ • Dynamic Cost Grid         │
 │ • BYU/NIC v7.1 Ingestion  │     │ • Pack-Ice Damping           │     │ • A* Path + Spline Smooth   │
 │ • ERA5 & Ocean Currents   │ ──► │ • Calving / Breakup Logic    │ ──► │ • Exact Dashboard UI        │
 │ • Sea-Ice Grid (SIC)      │Cont.│ • Monte Carlo 95% Ellipses   │Cont.│ • KPI Cards & Badges        │
 │ • Mass/Volume Estimation  │  A  │ • 7-Day Hazard Radii         │  B  │ • 1–7 Day Playable Scrubber │
 └───────────────────────────┘     └──────────────────────────────┘     └─────────────────────────────┘
```

---

### 👤 Person 1: Data Ingestion, Corridor AOI & Iceberg State
- **Role**: Data Engineer / Geospatial Specialist
- **Stages Owned**: **Stage 0, Stage 1, Stage 2**
- **Core Responsibilities**:
  1. **Corridor Bounding Box (Stage 0)**: Calculate the bounding navigation corridor around `(start_lat, start_lon)` and `(dest_lat, dest_lon)` with a buffer ($\pm 200\text{--}300\text{ km}$) to strictly filter out irrelevant continental data.
  2. **BYU/NIC v7.1 Parser (Stage 1)**: Ingest tracking records, extracting the latest observation date, position, and dimensions (`major_axis_km`, `minor_axis_km`) for all icebergs within the AOI.
  3. **Environmental Forcing & Sea Ice (Stage 0/1)**: Load and interpolate ERA5 10m wind ($u_{10}, v_{10}$), ocean surface currents ($u_{ocn}, v_{ocn}$), and Sea-Ice Concentration (SIC %). Includes a robust synthetic/fallback generator for offline demos.
  4. **Mass & Volume Estimator (Stage 2)**: Calculate tabular iceberg volume ($V \approx L \times W \times H$ where $H \approx 200\text{--}250\text{ m}$) and mass ($M = \rho_{ice} \times V$, with $\rho_{ice} \approx 900\text{ kg/m}^3$).
- **Assigned Files**:
  ```text
  src/data/
  ├── __init__.py
  ├── corridor.py          # Bounding corridor & geographic filter
  ├── byu_parser.py        # BYU/NIC v7.1 dataset parser
  ├── env_forcing.py       # ERA5, ocean current & SIC interpolation (+ fallback generator)
  └── mass_estimator.py    # Dimension & mass estimation
  ```

---

### 👤 Person 2: Drift Physics, Calving & Monte Carlo Uncertainty
- **Role**: Scientific Computing & Simulation Engineer
- **Stages Owned**: **Stage 3, Stage 4, Stage 5 (Hazard Boundary Calculations)**
- **Core Responsibilities**:
  1. **EWE20 Physics Engine (Stage 3)**: Modularize `iceberg_model_EWE20.py` into a vectorized daily drift step balancing air drag, water drag, Coriolis force, and sea surface slope.
  2. **Pack-Ice Damping (Stage 3)**: Apply resistance damping when local $\text{SIC} > 80\%$, locking iceberg velocity to surrounding sea ice.
  3. **Probabilistic Calving (Stage 3)**: Implement Poisson daughter-calving where parent icebergs shed smaller fragments with independent trajectory tracking.
  4. **Monte Carlo 95% Uncertainty (Stage 4 & 5)**: Execute 50–100 perturbations per iceberg across wind, current, and calving probability to compute expanding 95% confidence hazard radii:
     $$R_{\text{hazard}}(t) = R_{\text{iceberg}} + R_{\text{MonteCarlo\_95\%}}(t) + R_{\text{safety}}$$
- **Assigned Files**:
  ```text
  src/physics/
  ├── __init__.py
  ├── drift_engine.py      # EWE20 momentum & drift step equations
  ├── breakup.py           # Probabilistic daughter iceberg calving
  └── monte_carlo.py       # Ensemble simulation & 95% confidence buffer radii
  ```

---

### 👤 Person 3: Dynamic Cost Grid, A* Pathfinding & Interactive Dashboard
- **Role**: Optimization & Frontend/UI Lead
- **Stages Owned**: **Stage 5 (Grid Construction), Stage 6, Stage 7, Stage 8**
- **Core Responsibilities**:
  1. **Dynamic Cost Grid (Stage 5)**: Discretize the corridor into a 2D mesh applying cost rules:
     - Iceberg core: $\infty$ (impassable)
     - Hazard buffer zone: High penalty
     - $\text{SIC} > 80\%$: $\infty$ (impassable)
     - $\text{SIC } 15\text{--}80\%$: Caution cost
     - Open water ($\text{SIC} < 15\%$): Base distance cost
  2. **Risk-Aware A* Router & Spline Smoothing (Stage 6)**:
     - Calculate optimal daily routes from Start to Destination for Days 1 through 7.
     - Smooth discrete A* waypoints using a **Cubic B-Spline / Catmull-Rom** algorithm to produce the continuous green navigation path shown in the design.
  3. **Interactive Dashboard (Stage 7)**:
     - Implement the reference UI:
       - Header with status badges (`3 hazards nearby`, `87% route confidence`).
       - Canvas map with green Start/Destination pins, soft blue iceberg markers, dashed hazard circles, and the smoothed green route.
       - Live KPI cards: **Route distance** (`142 nm`), **Closest hazard** (`4.2 nm`), **Icebergs tracked** (`3`).
       - Bottom timeline scrubber (Days 1 to 7) with Play/Pause animation.
- **Assigned Files**:
  ```text
  src/routing/
  ├── __init__.py
  ├── cost_grid.py         # 2D hazard & sea-ice cost matrix
  ├── astar.py             # Risk-aware A* pathfinding
  └── smoother.py          # Spline curve interpolation for ship paths

  src/ui/
  ├── __init__.py
  └── dashboard.py         # Interactive UI matching the visual design
  ```

---

## 📋 Data Contracts Between Streams

### Contract A: Person 1 ➔ Person 2 (`src/contracts/initial_state.json`)
```json
{
  "start_coords": [-63.5, -58.2],
  "dest_coords": [-60.8, -52.4],
  "corridor_bbox": {
    "min_lat": -65.0, "max_lat": -59.0,
    "min_lon": -60.0, "max_lon": -50.0
  },
  "icebergs": [
    {
      "id": "A-68A",
      "lat": -62.5,
      "lon": -55.2,
      "length_km": 18.0,
      "width_km": 8.5,
      "thickness_m": 220.0,
      "mass_kg": 3.02e13,
      "initial_u_vel": 0.45,
      "initial_v_vel": -0.22
    }
  ]
}
```

### Contract B: Person 2 ➔ Person 3 (`src/contracts/route_day_state.json`)
```json
{
  "day": 3,
  "total_days": 7,
  "status": {
    "hazards_nearby": 3,
    "route_confidence_pct": 87
  },
  "kpis": {
    "route_distance_nm": 142.0,
    "closest_hazard_nm": 4.2,
    "icebergs_tracked": 3
  },
  "navigation": {
    "start": {"name": "Start", "coords": [-63.5, -58.2]},
    "destination": {"name": "Destination", "coords": [-60.8, -52.4]}
  },
  "hazards": [
    {
      "id": "A-68A-1",
      "center": [-61.9, -56.5],
      "iceberg_radius_nm": 1.2,
      "buffer_radius_nm": 4.5
    },
    {
      "id": "A-68A-2",
      "center": [-62.4, -54.8],
      "iceberg_radius_nm": 1.5,
      "buffer_radius_nm": 5.8
    },
    {
      "id": "B-15-frag",
      "center": [-61.2, -53.6],
      "iceberg_radius_nm": 1.0,
      "buffer_radius_nm": 3.8
    }
  ]
}
```

---

## ⏱️ Parallel Execution Timeline

| Milestone | Person 1 (Data & AOI) | Person 2 (Physics & Drift) | Person 3 (Routing & UI) |
|---|---|---|---|
| **Phase 1: Minute 1** *(No Blocking)* | Build corridor bounding box; generate mock `initial_state.json`. | Load mock `initial_state.json`; extract equations from `iceberg_model_adapted.py`. | Load mock `route_day_state.json`; scaffold UI layout matching the reference mockup. |
| **Phase 2: Core Logic** | Implement BYU/NIC parser & ERA5/SIC fallback generator. | Vectorize drift step with wind, ocean drag, and pack-ice damping. | Implement cost grid, A* router, and cubic spline path smoother. |
| **Phase 3: Ensemble & Polish** | Finalize mass estimation; test AOI filtering on historical data. | Implement Monte Carlo 50-run loop and 95% buffer radius expansion. | Wire up Day 1–7 scrubber slider, ▶ play animation loop, and dynamic KPI cards. |
| **Phase 4: Full Integration** | Stream live environmental states to Person 2. | Run drift on real data; output 7-day forecast coordinates to Person 3. | Test end-to-end: changing Start/End coordinates dynamically recalculates routes. |

---

## 📚 References & Scientific Foundations
- **England, M., Wagner, T., & Eisenman, I. (2020)**: *Modeling the Breakup of Tabular Icebergs*, Science Advances.
- **Wagner, T., Dell, R., & Eisenman, I. (2017)**: *An Analytical Model of Iceberg Drift*, Journal of Physical Oceanography.
- **BYU / National Ice Center (NIC)**: *Antarctic Iceberg Tracking Database v7.1*.
- **Copernicus Marine / ECMWF ERA5**: *Reanalysis Atmospheric & Ocean Surface Forcing*.
