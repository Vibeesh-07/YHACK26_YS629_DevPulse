/**
 * app.js
 * Dynamic Frontend Controller — Antarctic Navigation Decision Support System
 * Supports: real-time coord input, click-to-place waypoints, live map updates
 */

const METERS_PER_NM = 1852.0;

let map;
let multiDayData = [];
let currentDay = 1;
let isPlaying = false;
let playInterval = null;

// Map Layer Groups
let routeLayerGroup;
let wakeLayerGroup;
let shipLayerGroup;
let hazardsLayerGroup;
let waypointsLayerGroup;
let draftMarkersGroup;

// Click-to-place state
let pickMode = null; // null | 'start' | 'dest'

// ─── Math & Geodesic Helpers ──────────────────────────────────────────────────

function calculateBearingJs(lat1, lon1, lat2, lon2) {
  const phi1 = (lat1 * Math.PI) / 180;
  const phi2 = (lat2 * Math.PI) / 180;
  const deltaLambda = ((lon2 - lon1) * Math.PI) / 180;
  const y = Math.sin(deltaLambda) * Math.cos(phi2);
  const x = Math.cos(phi1) * Math.sin(phi2) - Math.sin(phi1) * Math.cos(phi2) * Math.cos(deltaLambda);
  const theta = Math.atan2(y, x);
  return Math.round((((theta * 180) / Math.PI) + 360) % 360);
}

function haversineNmJs(lat1, lon1, lat2, lon2) {
  const R_nm = 3440.065;
  const phi1 = (lat1 * Math.PI) / 180;
  const phi2 = (lat2 * Math.PI) / 180;
  const deltaPhi = ((lat2 - lat1) * Math.PI) / 180;
  const deltaLambda = ((lon2 - lon1) * Math.PI) / 180;
  const a = Math.sin(deltaPhi / 2) * Math.sin(deltaPhi / 2) +
            Math.cos(phi1) * Math.cos(phi2) *
            Math.sin(deltaLambda / 2) * Math.sin(deltaLambda / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R_nm * c;
}

function getShipDataForDay(dayData) {
  if (dayData.ship && dayData.ship.coords) {
    return dayData.ship;
  }
  const nav = dayData.navigation || {};
  const polyline = nav.route_polyline || [];
  const totalDist = (dayData.kpis && dayData.kpis.route_distance_nm) || 200;
  const totalDays = dayData.total_days || 7;
  const dayNum = dayData.day || 1;
  const frac = totalDays <= 1 ? 1.0 : Math.min(1.0, Math.max(0.0, (dayNum - 1) / (totalDays - 1)));
  const targetDist = frac * totalDist;
  const avgSpeed = +(totalDist / Math.max(1, totalDays * 24)).toFixed(1);
  const dailyDist = +(totalDist / Math.max(1, totalDays)).toFixed(1);

  if (polyline.length === 0) {
    return {
      coords: [-63.5, -58.2],
      heading_deg: 0,
      distance_traveled_nm: +(targetDist.toFixed(1)),
      distance_remaining_nm: +(Math.max(0, totalDist - targetDist).toFixed(1)),
      progress_pct: +(frac * 100).toFixed(1),
      average_speed_knots: avgSpeed,
      daily_distance_nm: dailyDist
    };
  }

  const idx = Math.min(polyline.length - 1, Math.floor(frac * (polyline.length - 1)));
  const nextIdx = Math.min(polyline.length - 1, idx + 1);
  const p1 = polyline[idx];
  const p2 = polyline[nextIdx];
  const heading = calculateBearingJs(p1[0], p1[1], p2[0], p2[1]);

  return {
    coords: p1,
    heading_deg: heading,
    distance_traveled_nm: +(targetDist.toFixed(1)),
    distance_remaining_nm: +(Math.max(0, totalDist - targetDist).toFixed(1)),
    progress_pct: +(frac * 100).toFixed(1),
    average_speed_knots: avgSpeed,
    daily_distance_nm: dailyDist
  };
}

function makeVesselIcon(ship) {
  const heading = (ship && ship.heading_deg !== undefined) ? ship.heading_deg : 0;
  return L.divIcon({
    className: "vessel-div-icon",
    html: `
      <div class="vessel-icon-container" title="Research Vessel Explorer">
        <div class="radar-pulse-ring"></div>
        <svg class="vessel-marker-svg" viewBox="0 0 36 36" style="transform: rotate(${heading}deg);">
          <!-- Ship Hull pointing 0 deg (North) -->
          <path d="M18,2 L27,24 C27,28 22,32 18,34 C14,32 9,28 9,24 Z" 
                fill="#0284c7" stroke="#38bdf8" stroke-width="2" />
          <!-- Superstructure -->
          <polygon points="18,8 23,22 18,19 13,22" fill="#e0f2fe" opacity="0.95" />
          <!-- Navigation mast light -->
          <circle cx="18" cy="12" r="2.5" fill="#f59e0b" />
          <!-- Bow guide line -->
          <line x1="18" y1="2" x2="18" y2="7" stroke="#ffffff" stroke-width="2" stroke-linecap="round" />
        </svg>
      </div>
    `,
    iconSize: [44, 44],
    iconAnchor: [22, 22]
  });
}

function makeVesselPopup(dayData, ship) {
  return `
    <div style="font-family:'Inter',sans-serif;font-size:12px;color:#111;min-width:190px">
      <strong style="color:#0284c7;font-size:13px">&#x1F6A2; Active Research Vessel</strong><br>
      ${(ship.progress_pct >= 100 || ship.distance_remaining_nm === 0)
        ? `<div style="margin:5px 0;padding:4px 6px;background:#f0fdf4;border-radius:4px;border-left:3px solid #10b981">
             <strong style="color:#15803d">&#x2714; Voyage Completed (100%)</strong><br>
             <span style="font-size:10.5px;color:#166534">Arrived at destination</span>
           </div>`
        : `<div style="margin:5px 0;padding:4px 6px;background:#f0f9ff;border-radius:4px;border-left:3px solid #0284c7">
             <strong>Day ${dayData.day} of ${dayData.total_days}</strong> (${ship.progress_pct || 0}% completed)<br>
             <span style="font-size:10.5px;color:#0369a1">&#x27A4; Route re-planned from current vessel position</span>
           </div>`
      }
      <span style="color:#555">Average Speed:</span> <strong>${ship.average_speed_knots || "—"} knots</strong><br>
      <span style="color:#555">Daily Run:</span> <strong>${ship.daily_distance_nm || "—"} nm/day</strong><br>
      <span style="color:#555">Traveled (Wake):</span> <strong>${ship.distance_traveled_nm || 0} nm</strong><br>
      <span style="color:#555">Remaining:</span> <strong>${ship.distance_remaining_nm || 0} nm</strong><br>
      <span style="color:#555">Current Heading:</span> <strong>${ship.heading_deg || 0}&deg;</strong><br>
      <span style="color:#888;font-size:11px">[${(ship.coords ? ship.coords[0] : 0).toFixed(4)}, ${(ship.coords ? ship.coords[1] : 0).toFixed(4)}]</span>
    </div>
  `;
}

// ─── Boot ───────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  initMap();
  setupEventListeners();
  await loadState();
});

// ─── Map Init ────────────────────────────────────────────────────────────────

function initMap() {
  map = L.map("polar-map", {
    center: [-62.2, -55.3],
    zoom: 6,
    zoomControl: false,
    attributionControl: false
  });

  L.control.zoom({ position: "topright" }).addTo(map);

  // Dark nautical base tiles
  L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    { maxZoom: 16, attribution: "Esri" }
  ).addTo(map);

  routeLayerGroup     = L.layerGroup().addTo(map);
  wakeLayerGroup      = L.layerGroup().addTo(map);
  shipLayerGroup      = L.layerGroup().addTo(map);
  hazardsLayerGroup   = L.layerGroup().addTo(map);
  waypointsLayerGroup = L.layerGroup().addTo(map);
  draftMarkersGroup   = L.layerGroup().addTo(map);

  // Map click handler for waypoint picking
  map.on("click", onMapClick);
}

// ─── State Loading ────────────────────────────────────────────────────────────

async function loadState() {
  try {
    const res = await fetch("/api/state");
    if (!res.ok) throw new Error("Failed to fetch state");
    const data = await res.json();
    if (data && data.length > 0) {
      multiDayData = data;
      const totalDays = data[0].total_days || (data.length - 1);
      updateTimelineControls(totalDays);
      renderDay(0);
      // Pre-fill form from loaded state
      const nav = data[0].navigation;
      if (nav) {
        document.getElementById("input-start-lat").value = nav.start.coords[0];
        document.getElementById("input-start-lon").value = nav.start.coords[1];
        document.getElementById("input-dest-lat").value  = nav.destination.coords[0];
        document.getElementById("input-dest-lon").value  = nav.destination.coords[1];
      }
      const daysInput = document.getElementById("input-days");
      if (daysInput) daysInput.value = totalDays;
      updateHeaderStatus("Route loaded from cache", false);
    }
  } catch (e) {
    console.warn("No cached state:", e.message);
    updateHeaderStatus("Enter coordinates and recalculate", false);
  }
}

// ─── Render Day ──────────────────────────────────────────────────────────────

function renderDay(dayNum) {
  currentDay = dayNum;
  const dayData = multiDayData.find(d => d.day === dayNum) || multiDayData[0];
  if (!dayData) return;

  const totalDays = dayData.total_days || (multiDayData.length > 0 ? multiDayData[0].total_days || (multiDayData.length - 1) : 7);
  const ship = getShipDataForDay(dayData);

  // Header
  const dayTitle = dayData.day === 0
    ? `Route status — Day 0 of ${totalDays} (Departure)`
    : (dayData.day === totalDays
        ? `Route status — Day ${totalDays} of ${totalDays} (Arrival)`
        : `Route status — Day ${dayData.day} of ${totalDays}`);
  document.getElementById("header-status-title").textContent = dayTitle;
  document.getElementById("badge-hazards-text").textContent =
    `${dayData.status.hazards_nearby} hazards nearby`;
  document.getElementById("badge-confidence-text").textContent =
    `${dayData.status.route_confidence_pct}% confidence`;

  // KPIs
  document.getElementById("kpi-distance").textContent = dayData.kpis.route_distance_nm;
  document.getElementById("kpi-hazard").textContent   = dayData.kpis.closest_hazard_nm;
  document.getElementById("kpi-icebergs").textContent = dayData.kpis.icebergs_tracked;

  // Vessel Speed & Motion KPIs
  const speedEl = document.getElementById("kpi-speed");
  if (speedEl) speedEl.textContent = ship.average_speed_knots || "—";
  const speedHintEl = document.getElementById("kpi-speed-hint");
  if (speedHintEl) {
    speedHintEl.textContent = `Daily run: ~${ship.daily_distance_nm || "—"} nm/day (${totalDays} travel days)`;
  }

  // Vessel Progress KPIs
  const progEl = document.getElementById("kpi-progress");
  if (progEl) progEl.textContent = `${ship.progress_pct || 0}`;
  const progBarEl = document.getElementById("vessel-progress-bar");
  if (progBarEl) progBarEl.style.width = `${ship.progress_pct || 0}%`;
  const progHintEl = document.getElementById("kpi-progress-hint");
  if (progHintEl) {
    progHintEl.textContent = `Traveled: ${ship.distance_traveled_nm || 0} nm | Remaining: ${ship.distance_remaining_nm || 0} nm`;
  }
  const vesselBadgeEl = document.getElementById("vessel-badge");
  if (vesselBadgeEl) {
    vesselBadgeEl.textContent = dayNum === 0
      ? "Day 0 (Start)"
      : (dayNum === totalDays ? `Day ${totalDays} (Dest)` : `Day ${dayNum} / ${totalDays}`);
  }

  const hazardBadge = document.getElementById("kpi-hazard-status");
  if (dayData.kpis.closest_hazard_nm >= 5.0) {
    hazardBadge.className = "kpi-status-badge status-safe";
    hazardBadge.textContent = "Safe clearance (> 5 nm)";
  } else {
    hazardBadge.className = "kpi-status-badge status-warn";
    hazardBadge.textContent = "Caution clearance (< 5 nm)";
  }

  // Timeline
  const slider = document.getElementById("timeline-slider");
  if (slider) slider.value = dayNum;
  document.getElementById("current-day-label").textContent =
    `Day ${dayNum} / ${totalDays}`;

  const base = new Date("2026-09-10T00:00:00Z");
  base.setDate(base.getDate() + dayNum);
  document.getElementById("current-date-label").textContent =
    dayData.date || base.toISOString().split("T")[0];

  document.querySelectorAll(".timeline-ticks .tick").forEach(t => {
    t.classList.toggle("active", parseInt(t.dataset.day) === dayNum);
  });

  renderMapLayers(dayData, ship);
  renderIcebergList(dayData.hazards);
}

// ─── Map Layers ───────────────────────────────────────────────────────────────

function renderMapLayers(dayData, ship) {
  routeLayerGroup.clearLayers();
  wakeLayerGroup.clearLayers();
  shipLayerGroup.clearLayers();
  hazardsLayerGroup.clearLayers();
  waypointsLayerGroup.clearLayers();

  if (!ship) ship = getShipDataForDay(dayData);

  const nav = dayData.navigation;
  const start = nav.start.coords;
  const dest  = nav.destination.coords;

  // Start marker
  L.marker(start, { icon: makeWaypointIcon("#10b981", "S") })
    .bindTooltip(`Start: [${start[0].toFixed(3)}, ${start[1].toFixed(3)}]`,
      { permanent: false, direction: "left" })
    .addTo(waypointsLayerGroup);

  // Destination marker
  L.marker(dest, { icon: makeWaypointIcon("#f59e0b", "D") })
    .bindTooltip(`Dest: [${dest[0].toFixed(3)}, ${dest[1].toFixed(3)}]`,
      { permanent: false, direction: "right" })
    .addTo(waypointsLayerGroup);

  // ── 1. Historical Sailed Wake (Start -> Current Vessel Position) ──
  const wakePoints = (nav.history_polyline && nav.history_polyline.length > 1)
    ? nav.history_polyline
    : (ship && ship.progress_pct > 0 && ship.coords ? [start, ship.coords] : null);

  if (wakePoints && wakePoints.length > 1) {
    L.polyline(wakePoints, {
      color: "#38bdf8",
      weight: 3.5,
      dashArray: "5,7",
      opacity: 0.85,
      lineCap: "round"
    })
      .bindTooltip(`Sailed wake: ${ship.distance_traveled_nm || 0} nm completed`, {
        permanent: false,
        direction: "center"
      })
      .addTo(wakeLayerGroup);
  }

  // ── 2. Dynamic Forward Route (Current Vessel Position -> Destination) ──
  const isArrived = (ship && ship.progress_pct >= 100) || (ship && ship.distance_remaining_nm === 0);
  const forwardPoints = (!isArrived && nav.forward_polyline && nav.forward_polyline.length > 1)
    ? nav.forward_polyline
    : (!isArrived && nav.route_polyline && nav.route_polyline.length > 1 ? nav.route_polyline : null);

  if (forwardPoints && forwardPoints.length > 1) {
    // Outer emerald glow
    L.polyline(forwardPoints, {
      color: "#10b981",
      weight: 10,
      opacity: 0.22,
      lineCap: "round",
      lineJoin: "round"
    }).addTo(routeLayerGroup);

    // Primary dynamic forward route line
    L.polyline(forwardPoints, {
      color: "#10b981",
      weight: 3.5,
      opacity: 0.95,
      lineCap: "round",
      lineJoin: "round"
    })
      .bindTooltip(`Dynamic route from current position (${ship.distance_remaining_nm || 0} nm remaining)`, {
        permanent: false,
        direction: "center"
      })
      .addTo(routeLayerGroup);
  }

  // Moving Ship Marker (positioned at current position P_d)
  if (ship && ship.coords) {
    const vesselTooltipText = isArrived
      ? `🚢 MV Explorer — Day ${dayData.day} (100% — Arrived at Destination)`
      : `🚢 MV Explorer — Day ${dayData.day} (${ship.progress_pct}% — ${ship.average_speed_knots} kn) [Origin of Day ${dayData.day} Route]`;

    L.marker(ship.coords, {
      icon: makeVesselIcon(ship),
      zIndexOffset: 1000
    })
      .bindTooltip(vesselTooltipText,
        { permanent: false, direction: "top", offset: [0, -18] })
      .bindPopup(makeVesselPopup(dayData, ship))
      .addTo(shipLayerGroup);
  }

  // Icebergs + MC hazard rings (day-specific positions)
  (dayData.hazards || []).forEach(h => {
    const center = h.center;
    const rCoreM   = h.iceberg_radius_nm * METERS_PER_NM;
    const rBufferM = h.buffer_radius_nm  * METERS_PER_NM;

    L.circle(center, {
      radius: rBufferM,
      color: "#f59e0b", dashArray: "5,5",
      weight: 1.5, fillColor: "#f59e0b", fillOpacity: 0.07
    }).addTo(hazardsLayerGroup);

    L.circle(center, {
      radius: rCoreM,
      color: "#38bdf8", weight: 2,
      fillColor: "#38bdf8", fillOpacity: 0.8
    }).bindPopup(makeIcebergPopup(h)).addTo(hazardsLayerGroup);

    L.marker(center, {
      icon: L.divIcon({
        className: "",
        html: `<span style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;color:#38bdf8;text-shadow:0 1px 4px #000">${h.id}</span>`,
        iconAnchor: [-8, 10]
      })
    }).addTo(hazardsLayerGroup);
  });

  // Pan map to fit route
  if (nav.route_polyline && nav.route_polyline.length > 1) {
    try {
      const bounds = L.latLngBounds(nav.route_polyline);
      map.fitBounds(bounds, { padding: [60, 80], animate: true, duration: 0.8 });
    } catch (_) {}
  }
}

function makeWaypointIcon(color, label) {
  return L.divIcon({
    className: "",
    html: `<div style="
      width:28px;height:28px;border-radius:50%;
      background:${color};border:2.5px solid #fff;
      box-shadow:0 0 14px ${color};
      display:flex;align-items:center;justify-content:center;
      font-family:'JetBrains Mono',monospace;font-size:11px;
      font-weight:700;color:#fff;
    ">${label}</div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14]
  });
}

function makeIcebergPopup(h) {
  const rawDisp = h.mc_95_dispersion_nm ? `${h.mc_95_dispersion_nm} nm` : "—";
  const uncapped = h.total_hazard_radius_nm_uncapped ? `${h.total_hazard_radius_nm_uncapped} nm` : null;
  const cappedLabel = uncapped && parseFloat(h.total_hazard_radius_nm_uncapped) > parseFloat(h.buffer_radius_nm)
    ? `<span style="color:#f59e0b;font-weight:700">${h.buffer_radius_nm} nm</span> <span style="color:#888;font-size:10px">(capped)</span>`
    : `<strong>${h.buffer_radius_nm} nm</strong>`;
  return `<div style="font-family:'Inter',sans-serif;font-size:12px;color:#111;min-width:180px;line-height:1.6">
    <strong style="color:#0284c7;font-size:13px">${h.id}</strong><br>
    <span style="color:#555">Physical radius:</span> <strong>${h.iceberg_radius_nm} nm</strong><br>
    <span style="color:#555">95% MC drift spread:</span> <strong style="color:#888">${rawDisp}</strong><br>
    <hr style="border:none;border-top:1px solid #ddd;margin:4px 0">
    <span style="color:#555">Active avoidance zone:</span> ${cappedLabel}<br>
    <span style="color:#888;font-size:10px">A* router keeps ship outside this radius</span><br>
    <span style="color:#888;font-size:11px">[${h.center[0].toFixed(3)}, ${h.center[1].toFixed(3)}]</span>
  </div>`;
}

// ─── Iceberg List ─────────────────────────────────────────────────────────────

function renderIcebergList(hazards) {
  const container = document.getElementById("iceberg-list");
  if (!hazards || hazards.length === 0) {
    container.innerHTML = `<div class="iceberg-empty-state">No hazards in corridor</div>`;
    return;
  }
  container.innerHTML = "";
  hazards.forEach(h => {
    const row = document.createElement("div");
    row.className = "iceberg-row";
    row.innerHTML = `
      <span class="iceberg-name">&#x25CF; ${h.id}</span>
      <div class="iceberg-meta">
        <div>Core: ${h.iceberg_radius_nm} nm</div>
        <div>Zone: ${h.buffer_radius_nm} nm</div>
      </div>
    `;
    row.addEventListener("click", () => {
      map.setView(h.center, 8, { animate: true });
    });
    container.appendChild(row);
  });
}

// ─── Click-to-Place Waypoints ────────────────────────────────────────────────

function onMapClick(e) {
  if (!pickMode) return;

  const lat = parseFloat(e.latlng.lat.toFixed(4));
  const lon = parseFloat(e.latlng.lng.toFixed(4));

  if (pickMode === "start") {
    document.getElementById("input-start-lat").value = lat;
    document.getElementById("input-start-lon").value = lon;
    setPickMode("dest");  // auto-advance to dest
  } else if (pickMode === "dest") {
    document.getElementById("input-dest-lat").value = lat;
    document.getElementById("input-dest-lon").value = lon;
    setPickMode(null);
    // Show draft line
    updateDraftLine();
  }
}

function setPickMode(mode) {
  pickMode = mode;
  const hint = document.getElementById("map-click-hint");
  const hintPt = document.getElementById("hint-next-point");
  const cancelBtn = document.getElementById("hint-cancel");
  const badgeModeText = document.getElementById("badge-mode-text");

  if (mode === null) {
    hint.classList.add("hidden");
    map.getContainer().style.cursor = "";
    badgeModeText.textContent = "Enter coords or pick on map";
  } else {
    hint.classList.remove("hidden");
    hintPt.textContent = mode === "start" ? "Start" : "Destination";
    cancelBtn.style.display = "inline-block";
    map.getContainer().style.cursor = "crosshair";
    badgeModeText.textContent = mode === "start"
      ? "Click map to set Start point"
      : "Click map to set Destination";
  }
}

function updateDraftLine() {
  draftMarkersGroup.clearLayers();
  const sLat = parseFloat(document.getElementById("input-start-lat").value);
  const sLon = parseFloat(document.getElementById("input-start-lon").value);
  const dLat = parseFloat(document.getElementById("input-dest-lat").value);
  const dLon = parseFloat(document.getElementById("input-dest-lon").value);

  if (isNaN(sLat) || isNaN(sLon) || isNaN(dLat) || isNaN(dLon)) return;

  // Draft markers
  L.marker([sLat, sLon], { icon: makeWaypointIcon("#10b981", "S") })
    .addTo(draftMarkersGroup);
  L.marker([dLat, dLon], { icon: makeWaypointIcon("#f59e0b", "D") })
    .addTo(draftMarkersGroup);

  // Dashed draft line
  L.polyline([[sLat, sLon], [dLat, dLon]], {
    color: "#6b7280", dashArray: "6,6", weight: 2, opacity: 0.6
  }).addTo(draftMarkersGroup);

  // Fit to draft bounds
  try {
    map.fitBounds(L.latLngBounds([[sLat, sLon], [dLat, dLon]]),
      { padding: [80, 100], animate: true });
  } catch (_) {}
}

function updateTimelineControls(totalDays) {
  const slider = document.getElementById("timeline-slider");
  if (slider) {
    slider.min = 0;
    slider.max = totalDays;
    if (parseInt(slider.value) > totalDays) {
      slider.value = 0;
    }
  }

  const ticksContainer = document.getElementById("timeline-ticks");
  if (ticksContainer) {
    ticksContainer.innerHTML = "";
    for (let d = 0; d <= totalDays; d++) {
      const span = document.createElement("span");
      span.className = "tick" + (d === currentDay ? " active" : "");
      span.dataset.day = d;
      span.textContent = `Day ${d}`;
      span.addEventListener("click", () => renderDay(d));
      ticksContainer.appendChild(span);
    }
  }
}

// ─── Event Listeners ──────────────────────────────────────────────────────────

function setupEventListeners() {
  // Timeline slider
  document.getElementById("timeline-slider").addEventListener("input", e => {
    renderDay(parseInt(e.target.value));
  });

  // Playback
  document.getElementById("btn-play").addEventListener("click", togglePlayback);
  document.getElementById("btn-prev").addEventListener("click", () => {
    if (currentDay > 0) renderDay(currentDay - 1);
  });
  document.getElementById("btn-next").addEventListener("click", () => {
    const maxDays = multiDayData.length > 0 ? (multiDayData[0].total_days || (multiDayData.length - 1)) : 7;
    if (currentDay < maxDays) renderDay(currentDay + 1);
  });

  // Preset selector
  const presetSelect = document.getElementById("select-preset");
  const presetCard = document.getElementById("preset-card");
  const presetDesc = document.getElementById("preset-desc");
  const presetMeta = document.getElementById("preset-meta");

  const BENCHMARK_PRESETS = {
    "1": {
      start: [-56.5, -36.5],
      dest: [-52.0, -36.5],
      days: 3,
      desc: "Direct path intersects South Georgia Island. Verifies A* routes around into open ocean with zero land collision.",
      meta: "Obstacle: South Georgia | Straight: 270 nm | Duration: 3 days"
    },
    "2": {
      start: [-62.2, -58.0],
      dest: [-60.5, -45.0],
      days: 4,
      desc: "Long-range Scotia Sea transit passing north of Elephant Island and clearing tabular icebergs A68B & A68C.",
      meta: "Obstacle: Scotia Icebergs | Straight: 387 nm | Duration: 4 days"
    },
    "3": {
      start: [-62.2, -58.8],
      dest: [-63.2, -54.8],
      days: 2,
      desc: "Navigates Bransfield Strait channel between South Shetland Islands and Antarctic Peninsula without touching land.",
      meta: "Obstacle: Strait / Archipelago | Straight: 125 nm | Duration: 2 days"
    },
    "4": {
      start: [-64.5, -52.0],
      dest: [-60.0, -50.0],
      days: 4,
      desc: "Northbound Weddell Sea 'Iceberg Alley' escape corridor with high density of active drifting icebergs.",
      meta: "Obstacle: Weddell Gyre Icebergs | Straight: 276 nm | Duration: 4 days"
    },
    "5": {
      start: [-57.0, -65.0],
      dest: [-60.5, -60.0],
      days: 3,
      desc: "Open deep-ocean baseline across Drake Passage. Verifies instant straight geodesic routing without processing delay.",
      meta: "Obstacle: None (Open Deep Ocean) | Straight: 261 nm | Duration: 3 days"
    }
  };

  if (presetSelect) {
    presetSelect.addEventListener("change", (e) => {
      const val = e.target.value;
      const p = BENCHMARK_PRESETS[val];
      if (p) {
        document.getElementById("input-start-lat").value = p.start[0];
        document.getElementById("input-start-lon").value = p.start[1];
        document.getElementById("input-dest-lat").value = p.dest[0];
        document.getElementById("input-dest-lon").value = p.dest[1];
        if (p.days) {
          const daysInput = document.getElementById("input-days");
          if (daysInput) daysInput.value = p.days;
        }

        if (presetCard) {
          presetDesc.textContent = p.desc;
          presetMeta.textContent = p.meta;
          presetCard.classList.remove("hidden");
        }

        draftMarkersGroup.clearLayers();
        updateDraftLine();
        const bounds = L.latLngBounds([p.start, p.dest]);
        map.fitBounds(bounds, { padding: [60, 60], maxZoom: 7 });
      }
    });
  }

  // Pick-on-map button
  document.getElementById("btn-pick-on-map").addEventListener("click", () => {
    setPickMode("start");
  });

  document.getElementById("hint-cancel").addEventListener("click", () => {
    setPickMode(null);
  });

  // Update draft line when coords change manually
  ["input-start-lat","input-start-lon","input-dest-lat","input-dest-lon"].forEach(id => {
    document.getElementById(id).addEventListener("change", () => {
      draftMarkersGroup.clearLayers();
      updateDraftLine();
    });
  });

  // Form submission
  document.getElementById("coords-form").addEventListener("submit", async e => {
    e.preventDefault();
    await runRecalculate();
  });
}

// ─── Recalculate ─────────────────────────────────────────────────────────────

async function runRecalculate() {
  const startLat = parseFloat(document.getElementById("input-start-lat").value);
  const startLon = parseFloat(document.getElementById("input-start-lon").value);
  const destLat  = parseFloat(document.getElementById("input-dest-lat").value);
  const destLon  = parseFloat(document.getElementById("input-dest-lon").value);
  const dateStr  = document.getElementById("input-date").value + "T00:00:00Z";
  const days     = parseInt(document.getElementById("input-days").value) || 5;

  // Validate
  if (isNaN(startLat) || isNaN(startLon) || isNaN(destLat) || isNaN(destLon)) {
    showError("Please enter valid latitude/longitude values.");
    return;
  }
  if (startLat === destLat && startLon === destLon) {
    showError("Start and destination cannot be the same point.");
    return;
  }

  hideError();
  showLoading();
  animateLoadingSteps();

  const btn = document.getElementById("btn-recalculate");
  const btnLabel = document.getElementById("btn-recalc-label");
  btn.disabled = true;
  btnLabel.textContent = "Calculating...";

  try {
    const res = await fetch("/api/recalculate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start_coords: [startLat, startLon],
        dest_coords:  [destLat, destLon],
        date:         dateStr,
        days:         days
      })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ error: "Unknown error" }));
      throw new Error(errData.error || `Server error ${res.status}`);
    }

    multiDayData = await res.json();
    draftMarkersGroup.clearLayers();
    const totalDays = multiDayData.length > 0 ? (multiDayData[0].total_days || (multiDayData.length - 1)) : days;
    updateTimelineControls(totalDays);
    renderDay(0);
    updateHeaderStatus(`Route recalculated (${totalDays} travel days) — ${new Date().toLocaleTimeString()}`, false);
  } catch (err) {
    showError("Calculation failed: " + err.message);
    updateHeaderStatus("Route calculation failed", true);
  } finally {
    hideLoading();
    btn.disabled = false;
    btnLabel.textContent = "Recalculate A* Route";
  }
}

// ─── Loading Overlay ──────────────────────────────────────────────────────────

let loadingStepTimers = [];

function showLoading() {
  document.getElementById("loading-overlay").classList.remove("hidden");
  [1, 2, 3, 4].forEach(i => {
    const el = document.getElementById(`lstep-${i}`);
    if (el) {
      el.className = "lstep";
      el.innerHTML = "&#x2B21; " + el.textContent.replace(/^[✓⬡\u2B21\s]+/, "");
    }
  });
}

function hideLoading() {
  loadingStepTimers.forEach(t => clearTimeout(t));
  loadingStepTimers = [];
  // Ensure all steps show as completed before closing overlay
  [1, 2, 3, 4].forEach(i => {
    const el = document.getElementById(`lstep-${i}`);
    if (el) {
      el.classList.remove("active");
      el.classList.add("done");
      el.innerHTML = "✓ " + el.textContent.replace(/^[✓⬡\u2B21\s]+/, "");
    }
  });
  setTimeout(() => {
    document.getElementById("loading-overlay").classList.add("hidden");
  }, 350);
}

function animateLoadingSteps() {
  loadingStepTimers.forEach(t => clearTimeout(t));
  loadingStepTimers = [];
  const delays = [0, 300, 600, 950];
  delays.forEach((delay, i) => {
    const timer = setTimeout(() => {
      if (i > 0) {
        const prev = document.getElementById(`lstep-${i}`);
        if (prev) {
          prev.classList.remove("active");
          prev.classList.add("done");
          prev.innerHTML = "✓ " + prev.textContent.replace(/^[✓⬡\u2B21\s]+/, "");
        }
      }
      const cur = document.getElementById(`lstep-${i + 1}`);
      if (cur) cur.classList.add("active");
    }, delay);
    loadingStepTimers.push(timer);
  });
}

// ─── UI Helpers ───────────────────────────────────────────────────────────────

function showError(msg) {
  const banner = document.getElementById("error-banner");
  document.getElementById("error-text").textContent = msg;
  banner.classList.remove("hidden");
}

function hideError() {
  document.getElementById("error-banner").classList.add("hidden");
}

function updateHeaderStatus(msg, isError) {
  const title = document.getElementById("header-status-title");
  title.textContent = msg;
  const dot = document.getElementById("pulse-dot");
  dot.style.background = isError ? "#ef4444" : "#10b981";
  dot.style.boxShadow  = isError ? "0 0 10px #ef4444" : "0 0 10px #10b981";
}

// ─── Playback ─────────────────────────────────────────────────────────────────

function togglePlayback() {
  isPlaying = !isPlaying;
  const icon = document.getElementById("play-icon");

  if (isPlaying) {
    if (multiDayData.length === 0) { isPlaying = false; return; }
    icon.textContent = "❚❚";
    playInterval = setInterval(() => {
      const maxDays = multiDayData.length > 0 ? (multiDayData[0].total_days || (multiDayData.length - 1)) : 7;
      let next = currentDay + 1;
      if (next > maxDays) next = 0;
      renderDay(next);
    }, 1500);
  } else {
    icon.textContent = "▶";
    clearInterval(playInterval);
  }
}

