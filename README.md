# YHACK26_YS629_DevPulse
## Challenge 9: AI-Enabled Antarctic Sea-Ice & Navigation Decision Support

### Overview
This repository contains the physics-based models, drift prediction algorithms, and dynamic routing decision-support system for Antarctic maritime navigation.

### Models Included
- **`iceberg_model_EWE20.py`**: Lagrangian iceberg drift and decay model with probabilistic breakup scheme for tabular icebergs (*England, Wagner, and Eisenman, 2020, Science Advances*).
- **`iceberg_model_WDE17.m`**: Analytical model of iceberg drift solving steady-state equations with wind, ocean current, and Coriolis forcing (*Wagner, Dell, and Eisenman, 2017, J. Phys. Oceanogr.*).

### System Pipeline
1. **Historical & Satellite Data**: BYU/NIC v7 Antarctic Iceberg Database + ERA5 atmospheric and ocean current forcing.
2. **Physics Drift Engine**: Eisenman analytical momentum equations.
3. **Uncertainty Modeling**: Monte-Carlo ensemble simulations generating spatiotemporal probability cones.
4. **Hazard Mapping**: Time-dependent 3D hazard field $H(x, y, t)$.
5. **Dynamic Routing**: Spatiotemporal A* / D* Lite pathfinding for safe Antarctic maritime navigation.
