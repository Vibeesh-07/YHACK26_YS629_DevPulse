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
let hazardsLayerGroup;
let waypointsLayerGroup;
let draftMarkersGroup;

// Click-to-place state
let pickMode = null; // null | 'start' | 'dest'

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

  routeLayerGroup   = L.layerGroup().addTo(map);
  hazardsLayerGroup = L.layerGroup().addTo(map);
  waypointsLayerGroup = L.layerGroup().addTo(map);
  draftMarkersGroup = L.layerGroup().addTo(map);

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
      renderDay(1);
      // Pre-fill form from loaded state
      const nav = data[0].navigation;
      if (nav) {
        document.getElementById("input-start-lat").value = nav.start.coords[0];
        document.getElementById("input-start-lon").value = nav.start.coords[1];
        document.getElementById("input-dest-lat").value  = nav.destination.coords[0];
        document.getElementById("input-dest-lon").value  = nav.destination.coords[1];
      }
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

  // Header
  document.getElementById("header-status-title").textContent =
    `Route status — day ${dayData.day} of ${dayData.total_days}`;
  document.getElementById("badge-hazards-text").textContent =
    `${dayData.status.hazards_nearby} hazards nearby`;
  document.getElementById("badge-confidence-text").textContent =
    `${dayData.status.route_confidence_pct}% confidence`;

  // KPIs
  document.getElementById("kpi-distance").textContent = dayData.kpis.route_distance_nm;
  document.getElementById("kpi-hazard").textContent   = dayData.kpis.closest_hazard_nm;
  document.getElementById("kpi-icebergs").textContent = dayData.kpis.icebergs_tracked;

  const hazardBadge = document.getElementById("kpi-hazard-status");
  if (dayData.kpis.closest_hazard_nm >= 5.0) {
    hazardBadge.className = "kpi-status-badge status-safe";
    hazardBadge.textContent = "Safe clearance (> 5 nm)";
  } else {
    hazardBadge.className = "kpi-status-badge status-warn";
    hazardBadge.textContent = "Caution clearance (< 5 nm)";
  }

  // Timeline
  document.getElementById("timeline-slider").value = dayNum;
  document.getElementById("current-day-label").textContent =
    `Day ${dayNum} / ${dayData.total_days}`;

  const base = new Date("2026-09-10T00:00:00Z");
  base.setDate(base.getDate() + (dayNum - 1));
  document.getElementById("current-date-label").textContent =
    base.toISOString().split("T")[0];

  document.querySelectorAll(".timeline-ticks .tick").forEach(t => {
    t.classList.toggle("active", parseInt(t.dataset.day) === dayNum);
  });

  renderMapLayers(dayData);
  renderIcebergList(dayData.hazards);
}

// ─── Map Layers ───────────────────────────────────────────────────────────────

function renderMapLayers(dayData) {
  routeLayerGroup.clearLayers();
  hazardsLayerGroup.clearLayers();
  waypointsLayerGroup.clearLayers();

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

  // Route glow + main line
  if (nav.route_polyline && nav.route_polyline.length > 1) {
    L.polyline(nav.route_polyline, {
      color: "#10b981", weight: 10, opacity: 0.18
    }).addTo(routeLayerGroup);

    L.polyline(nav.route_polyline, {
      color: "#10b981", weight: 3, opacity: 0.95,
      lineCap: "round", lineJoin: "round"
    }).addTo(routeLayerGroup);
  }

  // Icebergs + MC hazard rings
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
  return `<div style="font-family:'Inter',sans-serif;font-size:12px;color:#111;min-width:160px">
    <strong style="color:#0284c7;font-size:13px">${h.id}</strong><br>
    <span style="color:#555">Physical radius:</span> <strong>${h.iceberg_radius_nm} nm</strong><br>
    <span style="color:#555">95% MC uncertainty:</span> <strong>${h.mc_95_dispersion_nm || "—"} nm</strong><br>
    <span style="color:#555">Total hazard zone:</span> <strong>${h.buffer_radius_nm} nm</strong><br>
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

// ─── Event Listeners ──────────────────────────────────────────────────────────

function setupEventListeners() {
  // Timeline slider
  document.getElementById("timeline-slider").addEventListener("input", e => {
    renderDay(parseInt(e.target.value));
  });

  // Tick clicks
  document.querySelectorAll(".timeline-ticks .tick").forEach(t => {
    t.addEventListener("click", () => renderDay(parseInt(t.dataset.day)));
  });

  // Playback
  document.getElementById("btn-play").addEventListener("click", togglePlayback);
  document.getElementById("btn-prev").addEventListener("click", () => {
    if (currentDay > 1) renderDay(currentDay - 1);
  });
  document.getElementById("btn-next").addEventListener("click", () => {
    if (currentDay < 7) renderDay(currentDay + 1);
  });

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
        days:         7
      })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ error: "Unknown error" }));
      throw new Error(errData.error || `Server error ${res.status}`);
    }

    multiDayData = await res.json();
    draftMarkersGroup.clearLayers();
    renderDay(1);
    updateHeaderStatus(`Route recalculated — ${new Date().toLocaleTimeString()}`, false);
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

let loadingStepTimer = null;

function showLoading() {
  document.getElementById("loading-overlay").classList.remove("hidden");
  // Reset step states
  [1,2,3,4].forEach(i => {
    const el = document.getElementById(`lstep-${i}`);
    el.className = "lstep";
    el.classList.remove("done", "active");
  });
}

function hideLoading() {
  if (loadingStepTimer) { clearTimeout(loadingStepTimer); loadingStepTimer = null; }
  document.getElementById("loading-overlay").classList.add("hidden");
}

function animateLoadingSteps() {
  const delays = [0, 2500, 5000, 8000];
  delays.forEach((delay, i) => {
    loadingStepTimer = setTimeout(() => {
      // Complete previous step
      if (i > 0) {
        const prev = document.getElementById(`lstep-${i}`);
        prev.classList.remove("active");
        prev.classList.add("done");
        prev.textContent = prev.textContent.replace("&#x2B21;", "✓").replace("⬡", "✓");
      }
      // Activate current
      const cur = document.getElementById(`lstep-${i + 1}`);
      if (cur) cur.classList.add("active");
    }, delay);
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
      let next = currentDay + 1;
      if (next > 7) next = 1;
      renderDay(next);
    }, 1500);
  } else {
    icon.textContent = "▶";
    clearInterval(playInterval);
  }
}
