/** CityLab frontend: streams a study from /api/ask-stream and shows live progress,
 * building cards with detection boxes, the evidence ledger, a chart and the brief.
 * Plain JS (no build step): Leaflet, Chart.js and marked are loaded from CDNs and are
 * all optional; the page degrades gracefully if one fails to load. */
"use strict";

const $ = (id) => document.getElementById(id);
const PAGE_SIZE = 12;
const LABEL_COLORS = {
  "parking space": "#ff5a1f", car: "#7b3fe4", tree: "#1f8a70", pool: "#0a8fd6",
  "solar panel": "#f0b429", roof: "#1c62d6",
};
const state = {
  map: null, markerLayer: null, markers: new Map(), chart: null, previewRequest: null,
  controller: null, cards: [], cardsShown: 0, cardObserver: null, selected: null,
  evidence: [], cachedIds: new Set(), progress: new Map(), toolCalls: 0, failedCalls: 0,
  history: [], highlight: null, lastQuestion: "", correcting: false,
  details: { property: {}, interior: {}, upgrades: {} }, visionSource: {}, calibration: null, chart2: null, replay: null,
};
// Editable measurement fields: label, unit, input limits (validated before sending).
const EDITABLE = {
  parking_space_count: { label: "Parking spaces", unit: "", min: 0, max: null, step: 1, integer: true },
  green_cover_percent: { label: "Green cover", unit: "%", min: 0, max: 100, step: 0.1, integer: false },
  roof_area_sq_ft: { label: "Roof area", unit: " sq ft", min: 0, max: null, step: 1, integer: false },
};
const SHOWCASE_MISSING = "Demo mode: this question is not in the offline showcase";

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function esc(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}

function fmt(value, digits = 2) {
  if (value == null || Number.isNaN(value)) return "-";
  return typeof value === "number" ? Number(value.toFixed(digits)).toLocaleString() : esc(value);
}

/** Fallback markdown (paragraphs, **bold**, "- " bullets) when marked is unavailable. */
function simpleMarkdown(md) {
  return String(md || "").split(/\n{2,}/).map((block) => {
    const lines = block.split("\n");
    const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    if (lines.every((l) => /^\s*[-*] /.test(l))) {
      return "<ul>" + lines.map((l) => `<li>${inline(l.replace(/^\s*[-*] /, ""))}</li>`).join("") + "</ul>";
    }
    const heading = /^(#{1,6})\s+(.*)$/.exec(lines[0]);
    if (heading && lines.length === 1) return `<h4>${inline(heading[2])}</h4>`;
    return `<p>${inline(lines.join(" "))}</p>`;
  }).join("");
}

let markedReady = false;
/** Render untrusted markdown safely: raw HTML is escaped, links must be http(s), no images. */
function renderMarkdown(md) {
  if (typeof marked === "undefined") return simpleMarkdown(md);
  if (!markedReady) {
    marked.use({
      gfm: true,
      renderer: {
        html(arg) { return esc(typeof arg === "string" ? arg : arg && arg.text); },
        image(href, title, text) { return esc(typeof href === "object" ? href.text : text); },
        link(href, title, text) {
          const url = typeof href === "object" ? href.href : href;
          const label = typeof href === "object" ? esc(href.text) : text;
          return /^https?:\/\//i.test(url || "") ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${label}</a>` : label;
        },
      },
    });
    markedReady = true;
  }
  return marked.parse(String(md || ""));
}

function showFormError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}

/** Error banner: a message, optional hint, and Retry / saved-demo buttons. */
function showBanner(message, { offerDemo = false, retry = false, title = "Something went wrong.", hint = "" } = {}) {
  $("error-title").textContent = title;
  $("error-text").textContent = message || "";
  $("error-hint").textContent = hint;
  $("error-hint").hidden = !hint;
  $("show-demo").hidden = !offerDemo;
  $("retry").hidden = !retry || !state.lastQuestion;
  $("error-banner").hidden = !message;
  $("error-banner").classList.toggle("showcase", title.startsWith("Not in the offline"));
  if (message) $("error-banner").scrollIntoView({ behavior: "smooth", block: "center" });
}

/** A friendly message for an HTTP error response from the API. */
async function httpError(res, fallback) {
  const error = await res.json().catch(() => ({}));
  const detail = typeof error.detail === "string" ? error.detail : "";
  if (res.status === 422) return Object.assign(new Error(Array.isArray(error.detail) || !detail
    ? "Please check the values you entered: counts must be 0 or more and percentages between 0 and 100."
    : detail), { kind: "http", status: 422 });
  if (res.status === 503 && detail.startsWith(SHOWCASE_MISSING)) {
    return Object.assign(new Error(detail), { kind: "showcase", status: 503 });
  }
  if (res.status === 404) return Object.assign(new Error(detail || "Not found."), { kind: "http", status: 404 });
  return Object.assign(new Error(detail || fallback || `The server answered ${res.status}.`), { kind: "http", status: res.status });
}

const NAMES = {
  green_cover_percent: "green cover (%)", paved_surface_percent: "paved surface (%)",
  parking_space_count: "parking spaces (aerial)", surface_parking_spaces: "surface parking spaces (CoStar)",
  roof_area_sq_ft: "roof area (sq ft)", max_rent: "asking rent ($/month)", vacancy_percent: "vacancy (%)",
  total_num_units: "number of units", building_size_sq_ft: "building size (sq ft)", year_built: "year built",
};
const nice = (k) => NAMES[k] || String(k || "this factor").replace(/_/g, " ");

function address(b) {
  const a = b.primary_address || {};
  const street = [a.streetNumber, a.streetDirection, a.streetName, a.streetType, a.streetSuffix].filter(Boolean).join(" ");
  const place = [a.city, a.state].filter(Boolean).join(", ");
  return [street, place, a.postcode].filter(Boolean).join(" · ");
}

// ---------------------------------------------------------------------------
// Server-sent events over fetch (EventSource cannot POST)
// ---------------------------------------------------------------------------

/** Incremental SSE parser: feed() text chunks, get back complete {event, data} frames. */
function sseParser() {
  let buffer = "";
  return {
    feed(chunk) {
      // Normalise CRLF / CR line endings on the whole buffer; a trailing "\r" may be the
      // first half of a "\r\n" split across chunks, so it waits for the next chunk.
      buffer += chunk;
      const held = buffer.endsWith("\r") ? "\r" : "";
      buffer = (held ? buffer.slice(0, -1) : buffer).replace(/\r\n?/g, "\n");
      const frames = [];
      let cut;
      while ((cut = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        let event = "message";
        const data = [];
        for (const line of block.split("\n")) {
          if (!line || line.startsWith(":")) continue; // comments such as ": keepalive"
          const colon = line.indexOf(":");
          const field = colon === -1 ? line : line.slice(0, colon);
          const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
          if (field === "event") event = value;
          else if (field === "data") data.push(value);
        }
        if (data.length) frames.push({ event, data: data.join("\n") });
      }
      buffer += held;
      return frames;
    },
  };
}

async function streamStudy(body, onFrame, signal) {
  const res = await fetch("/api/ask-stream", {
    method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body), signal,
  });
  if (!res.ok) {
    const err = await httpError(res);
    if (err.status === 422) err.message = "Check the question and the number of buildings (1 to 50).";
    throw err;
  }
  if (!res.body || !res.body.getReader) throw new Error("This browser cannot read a streaming response.");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const parser = sseParser();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (const frame of parser.feed(decoder.decode(value, { stream: true }))) {
      let data = null;
      try { data = JSON.parse(frame.data); } catch { continue; }
      onFrame(frame.event, data);
    }
  }
}

// ---------------------------------------------------------------------------
// Live progress
// ---------------------------------------------------------------------------

function resetProgress() {
  state.progress = new Map();
  state.toolCalls = 0;
  state.failedCalls = 0;
  state.cachedIds = new Set();
  $("progress").innerHTML = "";
  $("tool-count").textContent = "0 tool calls";
  $("citation").hidden = true;
  $("citation-problems").hidden = true;
  $("spinner").hidden = false;
  $("cancel").hidden = false;
  $("progress-card").hidden = false;
  $("dashboard").hidden = false;
}

/** One line per subagent or per tool+source; repeats increase the line's count. */
function addProgress(kind, data) {
  const p = data.payload || {};
  const key = kind === "handoff" ? `handoff:${p.subagent}` : `tool:${p.tool}:${p.source}`;
  let item = state.progress.get(key);
  if (!item) {
    const li = document.createElement("li");
    li.className = kind === "handoff" ? "handoff" : "tool";
    $("progress").appendChild(li);
    item = { li, count: 0, failed: 0, cached: 0 };
    state.progress.set(key, item);
  }
  item.count += 1;
  if (p.failed) item.failed += 1;
  if (p.cached) item.cached += 1;
  if (kind === "handoff") {
    item.li.innerHTML = `<b>${esc(p.subagent)}</b>${item.count > 1 ? ` <span class="times">×${item.count}</span>` : ""}: ` +
      `<span class="task">${esc(p.task || "")}</span>`;
  } else {
    state.toolCalls += 1;
    noteVisionSource(p);
    if (p.failed) state.failedCalls += 1;
    if (p.cached) state.cachedIds.add(p.evidence_id);
    const notes = [item.cached ? `${item.cached} cached` : "", item.failed ? `${item.failed} failed` : ""].filter(Boolean).join(", ");
    item.li.innerHTML = `<code>${esc(p.tool)} [${esc(p.source)}]</code>` +
      `${item.count > 1 ? ` <span class="times">×${item.count}</span>` : ""}${notes ? ` <span class="notes">(${esc(notes)})</span>` : ""}`;
    item.li.classList.toggle("has-failed", item.failed > 0);
    $("tool-count").textContent = `${state.toolCalls} tool call${state.toolCalls === 1 ? "" : "s"}` +
      (state.failedCalls ? ` · ${state.failedCalls} failed` : "");
  }
  for (const other of state.progress.values()) other.li.classList.remove("active");
  item.li.classList.add("active");
}

function showCitation(data) {
  const passed = data.message === "PASS";
  const problems = (data.payload && data.payload.problems) || [];
  $("citation").hidden = false;
  const badge = $("citation-badge");
  badge.className = `citation-badge ${passed ? "pass" : "fail"}`;
  badge.textContent = passed ? "✓ Numbers verified" : `✗ Citation check failed${data.message.includes("incomplete") ? " (study incomplete)" : ""}` +
    (problems.length ? ` · ${problems.length} problem${problems.length === 1 ? "" : "s"} (click)` : "");
  badge.disabled = passed && !problems.length;
  $("citation-problems").innerHTML = problems.map((p) => `<li>${esc(p)}</li>`).join("");
  badge.onclick = () => {
    const list = $("citation-problems");
    list.hidden = !list.hidden || !problems.length;
    badge.setAttribute("aria-expanded", String(!list.hidden));
  };
}

function finishProgress() {
  $("spinner").hidden = true;
  $("cancel").hidden = true;
  for (const item of state.progress.values()) item.li.classList.remove("active");
}

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

/** Green (short payback) to red (long payback) for retrofit pins. */
function paybackColor(years, lo, hi) {
  if (years == null) return "#8a939c";
  const t = hi > lo ? Math.min(1, Math.max(0, (years - lo) / (hi - lo))) : 0;
  const mix = (a, b) => Math.round(a + (b - a) * t);
  return `rgb(${mix(31, 198)}, ${mix(138, 40)}, ${mix(112, 40)})`;
}

function renderMap(study) {
  const { buildings, measurements } = study;
  state.markers = new Map();
  if (typeof L === "undefined") {
    $("map").textContent = "The map could not load. Check your connection; the building cards are below.";
    return;
  }
  const byProp = Object.fromEntries(measurements.map((m) => [m.property_id, m]));
  const solar = Object.fromEntries((study.solar_results || []).map((r) => [r.property_id, r]));
  const paybacks = (study.solar_results || []).map((r) => r.payback_years).filter((v) => v != null);
  const lo = Math.min(...paybacks), hi = Math.max(...paybacks);
  if (!state.map) {
    $("map").innerHTML = ""; // drop the "appears here" placeholder before Leaflet takes over
    state.map = L.map("map");
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
    }).addTo(state.map);
    state.markerLayer = L.layerGroup().addTo(state.map);
  }
  try { state.markerLayer.clearLayers(); } catch { state.markerLayer = L.layerGroup().addTo(state.map); }
  const points = [];
  const key = colorKey(study);
  const valueOf = (b) => key === "payback_years" ? (solar[b.id] || {}).payback_years : (byProp[b.id] || {})[key];
  const values = buildings.map(valueOf).filter((v) => v != null);
  const vlo = Math.min(...values), vhi = Math.max(...values);
  const filter = $("map-filter").value;
  const colorOf = (v) => v == null ? "#b3bac1" : key === "payback_years" ? paybackColor(v, vlo, vhi)
    : seqColor(vhi > vlo ? (v - vlo) / (vhi - vlo) : 0.5);
  $("map-legend").textContent = values.length
    ? `Colour: ${nice(key)} from ${fmt(vlo, 1)} (${key === "payback_years" ? "green, shortest" : "light"}) to ${fmt(vhi, 1)} ` +
      `(${key === "payback_years" ? "red, longest" : "dark"}); grey = unavailable. Numbered pins are the solar top 10.`
    : `No building has a value for ${nice(key)}.`;
  for (const b of buildings) {
    if (b.latitude == null || b.longitude == null) continue;
    const m = byProp[b.id] || {};
    if (filter === "measured" && !byProp[b.id]) continue;
    if (filter === "corrected" && !m.is_corrected) continue;
    const r = solar[b.id];
    let marker;
    if (r && r.rank != null && r.rank <= 10) {
      // Retrofit: numbered pin for the top 10, coloured by payback.
      marker = L.marker([b.latitude, b.longitude], { icon: L.divIcon({
        className: "pin", iconSize: [36, 36], iconAnchor: [18, 18],
        html: `<span style="background:${paybackColor(r.payback_years, lo, hi)}">${esc(r.rank)}</span>`,
      }), title: `#${r.rank} ${b.name || b.id}`, keyboard: true }).bindPopup(
        `<strong>#${esc(r.rank)} ${esc(b.name || `Property ${b.id}`)}</strong><br>` +
        `Payback: ${fmt(r.payback_years, 1)} years · ${fmt(r.capacity_kw, 1)} kW`);
    } else {
      const base = { radius: r ? 7 : 9, fillColor: colorOf(valueOf(b)) };
      marker = L.circleMarker([b.latitude, b.longitude], {
        ...base, color: "#1c2733", weight: 2, fillOpacity: 0.85,
      });
      marker.base = base;
      marker.bindPopup(
        `<strong>${esc(b.name || `Property ${b.id}`)}</strong><br>` +
        (r ? `Rank ${fmt(r.rank, 0)} · payback ${fmt(r.payback_years, 1)} years`
          : `Rent: $${fmt(b.max_rent, 0)} · Parking: ${fmt(m.parking_space_count, 0)}<br>Green cover: ${fmt(m.green_cover_percent)}%`));
    }
    marker.addTo(state.markerLayer);
    marker.on("click", () => selectBuilding(b.id, "map"));
    state.markers.set(b.id, marker);
    points.push([b.latitude, b.longitude]);
  }
  if (points.length && !state.keepView) state.map.fitBounds(points, { padding: [30, 30], maxZoom: 13 });
  state.keepView = false;
  setTimeout(() => state.map && state.map.invalidateSize(), 50);
}

/** Highlight one building's card, map marker, solar row and email together. */
function selectBuilding(id, from) {
  state.selected = id;
  for (const [mid, marker] of state.markers) {
    const on = mid === id;
    if (marker.setStyle) {
      const base = marker.base || { radius: 9, fillColor: "#ff5a1f" };
      marker.setStyle({ radius: on ? 14 : base.radius, fillColor: on ? "#1f8a70" : base.fillColor, weight: on ? 4 : 2 });
      if (on) marker.bringToFront();
    } else if (marker.getElement && marker.getElement()) {
      marker.getElement().classList.toggle("selected", on);
      if (on) marker.setZIndexOffset(1000); else marker.setZIndexOffset(0);
    }
  }
  document.querySelectorAll("#solar-ranking tr.selected").forEach((r) => r.classList.remove("selected"));
  const row = document.querySelector(`#solar-ranking tr[data-id="${CSS.escape(String(id))}"]`);
  if (row) {
    row.classList.add("selected");
    if (from === "map") row.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  if (state.study && state.study.solar_results) renderEmail(state.study, id);
  try { renderInspector(id); } catch (err) { console.warn("CityLab: could not draw the inspector", err); }
  try { renderUpgrades(id); } catch (err) { console.warn("CityLab: could not draw the upgrades", err); }
  document.querySelectorAll(".bcard.selected").forEach((el) => el.classList.remove("selected"));
  if (from === "map") {
    const index = state.cards.findIndex((c) => c.building.id === id);
    while (index >= state.cardsShown && state.cardsShown < state.cards.length) showMoreCards();
  }
  const card = document.querySelector(`.bcard[data-id="${CSS.escape(String(id))}"]`);
  if (card) {
    card.classList.add("selected");
    if (from === "map") card.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  const marker = state.markers.get(id);
  if (marker && (from === "card" || from === "table") && state.map) {
    state.map.panTo(marker.getLatLng());
    marker.openPopup();
  }
}

// ---------------------------------------------------------------------------
// Building cards with detection overlays (lazy, paged)
// ---------------------------------------------------------------------------

function keyMeasurement(study, m) {
  const predictor = study.stats && study.stats.predictor;
  if (predictor && m[predictor] != null) return [nice(predictor), fmt(m[predictor], 1)];
  if (m.parking_space_count != null) return ["parking spaces", fmt(m.parking_space_count, 0)];
  if (m.green_cover_percent != null) return ["green cover", `${fmt(m.green_cover_percent, 1)}%`];
  if (m.roof_area_sq_ft != null) return ["roof area", `${fmt(m.roof_area_sq_ft, 0)} sq ft`];
  return ["measurement", "not measured"];
}

function renderCards(study) {
  const byProp = Object.fromEntries(study.measurements.map((m) => [m.property_id, m]));
  state.cards = study.buildings.map((building) => ({ building, measurement: byProp[building.id] || null }));
  state.cardsShown = 0;
  $("cards").innerHTML = "";
  if (state.cardObserver) state.cardObserver.disconnect();
  showMoreCards(study);
  if ("IntersectionObserver" in window) {
    state.cardObserver = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) showMoreCards(study);
    }, { rootMargin: "300px" });
    state.cardObserver.observe($("cards-sentinel"));
  }
}

function showMoreCards(study = state.study) {
  const next = state.cards.slice(state.cardsShown, state.cardsShown + PAGE_SIZE);
  for (const { building: b, measurement: m } of next) $("cards").appendChild(buildCard(study, b, m));
  state.cardsShown += next.length;
  $("more-cards").hidden = state.cardsShown >= state.cards.length;
  $("more-cards").textContent = `Show more buildings (${state.cards.length - state.cardsShown} left)`;
}

/** Short labels for where a solar estimate's roof area and height actually came from. */
const SOURCE_LABEL = {
  "Vexcel detected structure footprint": "Vexcel aerial footprint",
  "Matterport room scan (median room height)": "Matterport scan",
};
function sourceLabel(text) {
  if (!text) return null;
  return SOURCE_LABEL[text] || (text.startsWith("CoStar") ? "CoStar record" : text.split(",")[0].split("(")[0].trim());
}

/** The building's solar assumptions, if the study produced one (retrofit mode only). */
function solarFor(study, propertyId) {
  return (study.solar_results || []).find((r) => r.property_id === propertyId) || null;
}

/** A compact provenance strip: where the roof, height and facing numbers came from. */
function provenanceRow(solar) {
  if (!solar || !solar.assumptions) return "";
  const a = solar.assumptions;
  const chips = [];
  const roof = sourceLabel(a.roof_area_source);
  if (roof) chips.push(`<span class="prov-chip" title="${esc(a.roof_area_source)}">Roof · ${esc(roof)}</span>`);
  const height = sourceLabel(a.building_height_source);
  if (height != null && a.estimated_building_height_ft != null) {
    chips.push(`<span class="prov-chip" title="${esc(a.building_height_source)}">Height · ${fmt(a.estimated_building_height_ft, 1)} ft (${esc(height)})</span>`);
  }
  if (a.azimuth_deg != null) {
    chips.push(`<span class="prov-chip" title="${esc(a.azimuth_source || "")}">Facing · ${fmt(a.azimuth_deg, 0)}°</span>`);
  }
  if (a.weather_station && a.weather_station.distance != null) {
    chips.push(`<span class="prov-chip" title="NREL PVWatts weather station">Weather · ${fmt(a.weather_station.distance / 1000, 1)} km away</span>`);
  }
  return chips.length ? `<div class="provenance">${chips.join("")}</div>` : "";
}

function buildCard(study, b, m) {
  const card = document.createElement("article");
  card.className = "bcard";
  card.dataset.id = String(b.id);
  card.tabIndex = 0;
  const [label, value] = keyMeasurement(study, m || {});
  const where = address(b);
  const canEdit = canEditStudy(study, m);
  const editable = canEdit ? editableFields(study).slice(0, 2).map((field) => [field, EDITABLE[field]]).map(([field, spec]) =>
    `<button type="button" class="edit-value" data-field="${field}" title="Correct ${esc(spec.label.toLowerCase())}">` +
    `<span>${esc(spec.label)}</span><strong>${m[field] == null ? "unknown" : esc(fmt(m[field], spec.integer ? 0 : 1)) + spec.unit}</strong>` +
    `<i aria-hidden="true">✎</i></button>`).join("") : "";
  const solar = solarFor(study, b.id);
  const hasTour = !!b.matterport_model_id;
  // The Sun & roof tab is always offered; it checks for its own report image on click.
  const tabs = `<div class="bcard-tabs" role="tablist">` +
    (hasTour ? `<button type="button" class="bcard-tab" data-tab="tour" role="tab">3D tour</button>` : "") +
    `<button type="button" class="bcard-tab" data-tab="roof" role="tab">Sun &amp; roof</button></div>` +
    (hasTour ? `<div class="bcard-panel" data-panel="tour" hidden></div>` : "") +
    `<div class="bcard-panel" data-panel="roof" hidden></div>`;
  card.innerHTML =
    `<div class="bimg"><img alt="Aerial image of property ${esc(b.id)}" loading="lazy" decoding="async" src="/api/image/${encodeURIComponent(b.id)}">` +
    `<canvas class="boxes" aria-hidden="true"></canvas><span class="noimg" hidden>No aerial image</span>` +
    `${study.is_mock ? '<span class="demo-tag">Demo data</span>' : ""}` +
    `${m && m.is_corrected ? '<span class="corrected-tag">Corrected</span>' : ""}</div>` +
    `<div class="bbody"><h4>${esc(b.name || `Property ${b.id}`)}</h4>` +
    `${where ? `<p class="addr">${esc(where)}</p>` : ""}` +
    (canEdit ? `<div class="edit-values">${editable}</div>`
      : `<p class="key"><span>${esc(label)}</span><strong>${esc(value)}</strong></p>`) +
    `<div class="editor-slot"></div>` +
    provenanceRow(solar) + tabs +
    `<dl><dt>Rent</dt><dd>${b.max_rent != null ? "$" + fmt(b.max_rent, 0) : "-"}</dd>` +
    `<dt>Units</dt><dd>${fmt(b.total_num_units, 0)}</dd><dt>Built</dt><dd>${b.year_built != null ? esc(b.year_built) : "-"}</dd></dl>` +
    `<p class="pid">Property ${esc(b.id)}</p></div>`;
  const img = card.querySelector("img");
  const canvas = card.querySelector("canvas");
  const obs = (m && m.observations) || {};
  const draw = () => drawDetections(img, canvas, (m && m.detections) || [], obs.image_width_px, obs.image_height_px);
  img.addEventListener("load", draw);
  img.addEventListener("error", () => { img.hidden = true; canvas.hidden = true; card.querySelector(".noimg").hidden = false; });
  if ("ResizeObserver" in window) new ResizeObserver(draw).observe(img);
  if (canEdit) {
    card.querySelectorAll(".edit-value").forEach((btn) => btn.addEventListener("click", (e) => {
      e.stopPropagation();
      openEditor(card, m, btn.dataset.field);
    }));
    // Clicking a detection box corrects the parking count (boxes are parking detections).
    card.querySelector(".bimg").addEventListener("click", (e) => {
      const rect = img.getBoundingClientRect();
      const x = e.clientX - rect.left, y = e.clientY - rect.top;
      const hit = (canvas.boxes || []).some((bx) => x >= bx.x && x <= bx.x + bx.w && y >= bx.y && y <= bx.y + bx.h);
      if (hit) { e.stopPropagation(); openEditor(card, m, "parking_space_count"); }
    });
  }
  card.querySelectorAll(".bcard-tab").forEach((tab) => tab.addEventListener("click", (e) => {
    e.stopPropagation();
    openBuildingTab(card, tab.dataset.tab, b);
  }));
  card.addEventListener("click", () => selectBuilding(b.id, "card"));
  card.addEventListener("keydown", (e) => {
    if (e.target !== card) return; // typing in the editor must not select the card
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectBuilding(b.id, "card"); }
  });
  if (state.selected === b.id) card.classList.add("selected");
  if (state.highlight && state.highlight.propertyId === b.id) card.classList.add("flash-card");
  return card;
}

/** Load the clicked tab's panel the first time (3D tour iframe, or the roof-view report), and toggle it. */
async function openBuildingTab(card, name, building) {
  const tabs = card.querySelectorAll(".bcard-tab");
  const panels = card.querySelectorAll(".bcard-panel");
  const target = card.querySelector(`.bcard-panel[data-panel="${name}"]`);
  const wasOpen = target && !target.hidden;
  panels.forEach((p) => (p.hidden = true));
  tabs.forEach((t) => t.classList.remove("active"));
  if (wasOpen) return; // clicking an open tab just closes it
  card.querySelector(`.bcard-tab[data-tab="${name}"]`).classList.add("active");
  target.hidden = false;
  if (target.dataset.loaded) return;
  target.dataset.loaded = "1";
  if (name === "tour") {
    try {
      const config = await getConfig();
      if (!config.matterport_sdk_key) throw new Error("no key");
      const url = `https://my.matterport.com/show/?m=${encodeURIComponent(building.matterport_model_id)}` +
        `&play=1&qs=1&applicationKey=${encodeURIComponent(config.matterport_sdk_key)}`;
      target.innerHTML = `<iframe src="${esc(url)}" title="3D tour of property ${esc(building.id)}" ` +
        `allow="xr-spatial-tracking" allowfullscreen loading="lazy"></iframe>`;
    } catch {
      target.innerHTML = `<p class="panel-empty">The 3D tour is not available right now.</p>`;
    }
  } else {
    const img = new Image();
    img.alt = `Shadow and roof-sun report for property ${building.id}`;
    img.loading = "lazy";
    img.onerror = () => { target.innerHTML = `<p class="panel-empty">No roof report has been generated for this building yet.</p>`; };
    img.onload = () => target.replaceChildren(img);
    img.src = `/api/roof-view/${encodeURIComponent(building.id)}`;
  }
}

let configPromise = null;
/** GET /api/config once per page load (the Matterport SDK key, never a secret token). */
function getConfig() {
  if (!configPromise) configPromise = fetch("/api/config").then((r) => r.ok ? r.json() : {});
  return configPromise;
}

/** Inline editor: number input with the current value, a note, Save / Cancel. */
function openEditor(card, m, field) {
  const spec = EDITABLE[field];
  const slot = card.querySelector(".editor-slot");
  if (!spec || !slot) return;
  const current = m[field];
  slot.innerHTML =
    `<form class="editor" novalidate><label>${esc(spec.label)}${spec.unit ? ` (${esc(spec.unit)})` : ""}` +
    `<input type="number" name="value" inputmode="decimal" min="${spec.min}"${spec.max != null ? ` max="${spec.max}"` : ""} ` +
    `step="${spec.step}" value="${current == null ? "" : esc(current)}" required></label>` +
    `<label>Note <input type="text" name="note" maxlength="300" placeholder="Why is this value wrong?"></label>` +
    `<p class="editor-msg" role="status"></p>` +
    `<div class="editor-actions"><button type="submit" class="save-btn">Save</button>` +
    `<button type="button" class="cancel-edit">Cancel</button></div></form>`;
  const form = slot.querySelector("form");
  const valueInput = form.querySelector('input[name="value"]');
  const noteInput = form.querySelector('input[name="note"]');
  form.addEventListener("click", (e) => e.stopPropagation());
  form.querySelector(".cancel-edit").addEventListener("click", () => { slot.innerHTML = ""; });
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const raw = valueInput.value.trim();
    const value = Number(raw);
    const msg = form.querySelector(".editor-msg");
    const problem = raw === "" || !Number.isFinite(value) ? "Enter a number."
      : value < spec.min ? `${spec.label} cannot be below ${spec.min}.`
      : spec.max != null && value > spec.max ? `${spec.label} must be between ${spec.min} and ${spec.max}.`
      : spec.integer && !Number.isInteger(value) ? `${spec.label} must be a whole number.` : "";
    if (problem) { msg.textContent = problem; msg.className = "editor-msg error"; valueInput.focus(); return; }
    submitCorrection(card, m, field, value, noteInput.value.trim(), form);
  });
  valueInput.focus();
  valueInput.select();
}

/** POST /api/correct, then show the corrected study with what changed. */
async function submitCorrection(card, m, field, value, note, form) {
  if (state.correcting) return;
  state.correcting = true;
  const msg = form.querySelector(".editor-msg");
  msg.innerHTML = `<span class="spinner small" aria-hidden="true"></span> ${studyKind(state.study) === "research"
    ? "Re-running statistics..." : "Re-estimating solar, ranking and upgrades..."}`;
  msg.className = "editor-msg busy";
  form.querySelectorAll("button, input").forEach((el) => { el.disabled = true; });
  const corrections = { [field]: value };
  if (note) corrections.note = note;
  try {
    const res = await fetch("/api/correct", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ study_id: state.study.id, measurement_id: m.id, corrections }),
    });
    if (!res.ok) throw await httpError(res, "The correction could not be applied.");
    const data = await res.json();
    const event = (data.events || []).find((e) => e.event_type === "correction") || { payload: {} };
    state.history.push({ study: state.study, evidence: state.evidence, cachedIds: new Set(state.cachedIds),
                         panel: state.lastPanel || null });
    const evidence = [...state.evidence, ...correctionEvidence(data, event.payload)];
    state.highlight = { propertyId: m.property_id, statsBefore: state.study.stats };
    state.history[state.history.length - 1].details = JSON.parse(JSON.stringify(state.details));
    detailsFromEvents(data.events);
    await loadCalibration();
    state.selected = m.property_id;
    state.keepView = true;
    renderStudy(data.study, evidence);
    selectBuilding(m.property_id, "correction");
    showWhatChanged(event.payload);
    state.highlight = null;
  } catch (err) {
    msg.textContent = err.kind === "http" || err.kind === "showcase" ? err.message
      : "Could not reach the CityLab API to apply the correction.";
    msg.className = "editor-msg error";
    form.querySelectorAll("button, input").forEach((el) => { el.disabled = false; });
  } finally {
    state.correcting = false;
  }
}

/** Ledger rows for the correction and the re-run, so new [E#] tags in the brief resolve. */
function correctionEvidence(data, payload) {
  const rows = [];
  const changes = payload.changes || [];
  const field = changes[0];
  if (payload.evidence_id && field) {
    rows.push({ id: payload.evidence_id, tool: "correct_measurement", source: data.study.is_mock ? "mock" : "real",
                summary: `${field.label}: ${field.before ?? "unknown"} → ${field.after ?? "unknown"} (user correction)` });
  }
  const tool = (data.events || []).find((e) => e.event_type === "tool_call");
  const stats = data.study.stats || {};
  const p = stats.predictor;
  if (tool && tool.payload && tool.payload.tool === "rank_buildings") {
    rows.push({ id: tool.payload.evidence_id, tool: "rank_buildings", source: tool.payload.source,
                summary: `${(data.study.solar_results || []).length} buildings re-ranked after correction` });
  } else if (tool && tool.payload) {
    const coef = p && stats.coefficients ? stats.coefficients[p] : null;
    rows.push({ id: tool.payload.evidence_id, tool: "run_regression", source: tool.payload.source,
                summary: coef == null ? "regression re-run after correction: no result"
                  : `${p} coefficient ${fmt(coef, 4)}, n=${stats.sample_size} (re-run after correction)` });
  }
  return rows;
}

/** The "What changed" panel: before → after with arrows, plus the conclusion sentence. */
function showWhatChanged(payload) {
  state.lastPanel = payload;
  const changes = (payload && payload.changes) || [];
  $("what-changed").hidden = !changes.length;
  $("undo").hidden = !state.history.length;
  $("wc-conclusion").textContent = (payload && payload.conclusion) || "";
  const cal = (payload && payload.calibration) || {};
  $("wc-calibration").textContent = Object.keys(cal).map((f) => `${(EDITABLE[f] || { label: f }).label}: ${calibrationLine(f)}`).join(" ");
  const arrow = { up: "▲", down: "▼", unchanged: "＝", "n/a": "→" };
  const show = (v) => v == null ? "unknown" : typeof v === "number" ? fmt(v, Math.abs(v) < 1 ? 4 : 2) : esc(v);
  $("wc-list").innerHTML = changes.map((c) =>
    `<li class="dir-${esc(c.direction)}"><span class="wc-label">${esc(c.label)}</span>` +
    `<span class="wc-values">${show(c.before)} <b class="wc-arrow" aria-label="${esc(c.direction)}">${arrow[c.direction] || "→"}</b> ${show(c.after)}</span></li>`).join("");
  if (changes.length) $("what-changed").scrollIntoView({ behavior: "smooth", block: "center" });
}

function undoCorrection() {
  const previous = state.history.pop();
  if (!previous) return;
  state.cachedIds = previous.cachedIds;
  if (previous.details) state.details = previous.details;
  state.keepView = true;
  renderStudy(previous.study, previous.evidence);
  if (previous.panel && state.history.length) showWhatChanged(previous.panel);
  else { state.lastPanel = null; $("what-changed").hidden = true; }
}

/** Draw [left, top, right, bottom] source-pixel boxes scaled to the displayed image size.
 * sourceW/H are the original image's size (Measurement.observations.image_width_px/height_px);
 * the served file may be smaller (the showcase resizes to 1024 px), so they win over naturalWidth. */
function drawDetections(img, canvas, detections, sourceW, sourceH) {
  if (!img.complete || !img.naturalWidth) return;
  const w = img.clientWidth, h = img.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * ratio);
  canvas.height = Math.round(h * ratio);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const sx = w / (sourceW > 0 ? sourceW : img.naturalWidth);
  const sy = h / (sourceH > 0 ? sourceH : img.naturalHeight);
  ctx.font = "600 12px Inter, system-ui, sans-serif";
  ctx.lineWidth = 2.5;
  canvas.boxes = []; // displayed-pixel boxes, for click hit-testing
  for (const d of detections) {
    if (!Array.isArray(d.bbox) || d.bbox.length !== 4) continue;
    const [left, top, right, bottom] = d.bbox;
    const x = left * sx, y = top * sy, bw = (right - left) * sx, bh = (bottom - top) * sy;
    canvas.boxes.push({ x, y, w: bw, h: bh, label: d.label });
    const color = LABEL_COLORS[String(d.label).toLowerCase()] || "#e8281c";
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, bw, bh);
    if (bw > 26 && bh > 14) {
      const text = d.label + (d.confidence != null ? ` ${Math.round(d.confidence * 100)}%` : "");
      const tw = ctx.measureText(text).width + 8;
      ctx.fillStyle = color;
      ctx.fillRect(x, Math.max(0, y - 17), tw, 17);
      ctx.fillStyle = "#fff";
      ctx.fillText(text, x + 4, Math.max(12, y - 4));
    }
  }
}

// ---------------------------------------------------------------------------
// Evidence ledger and [E#] chips
// ---------------------------------------------------------------------------

function renderLedger(evidence) {
  state.evidence = evidence || [];
  const table = $("ledger");
  if (!state.evidence.length) {
    table.innerHTML = "";
    $("ledger-empty").hidden = false;
    $("ledger-empty").textContent = "No evidence ledger for this study (the saved demo study was not produced by a live run).";
    $("ledger-filter").hidden = true;
    return;
  }
  $("ledger-empty").hidden = true;
  $("ledger-filter").hidden = false;
  table.innerHTML = "<thead><tr><th>Id</th><th>Tool</th><th>Source</th><th>Summary</th></tr></thead><tbody>" +
    state.evidence.map((e) => {
      const source = state.cachedIds.has(e.id) ? "cached" : e.source;
      return `<tr id="ev-${esc(e.id)}" data-text="${esc(`${e.id} ${e.tool} ${source} ${e.summary}`.toLowerCase())}">` +
        `<td><code>${esc(e.id)}</code></td><td>${esc(e.tool)}</td><td><span class="src src-${esc(source)}">${esc(source)}</span></td>` +
        `<td class="summary">${esc(e.summary)}</td></tr>`;
    }).join("") + "</tbody>";
  filterLedger();
}

function filterLedger() {
  const q = $("ledger-filter").value.trim().toLowerCase();
  for (const row of $("ledger").querySelectorAll("tbody tr")) row.hidden = q && !row.dataset.text.includes(q);
}

/** Replace [E12] / [E1, E2] text in the rendered brief with clickable chips (DOM-safe). */
function linkEvidenceTags(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) if (/\[E\d+/.test(walker.currentNode.nodeValue)) nodes.push(walker.currentNode);
  const pattern = /\[(E\d+(?:\s*[,;]\s*E\d+)*)\]/g;
  for (const node of nodes) {
    const text = node.nodeValue;
    const frag = document.createDocumentFragment();
    let last = 0;
    for (const match of text.matchAll(pattern)) {
      frag.append(text.slice(last, match.index));
      for (const id of match[1].split(/\s*[,;]\s*/)) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "ev-chip";
        chip.textContent = id;
        chip.title = `Show evidence ${id}`;
        chip.addEventListener("click", () => jumpToEvidence(id));
        frag.append(chip);
      }
      last = match.index + match[0].length;
    }
    frag.append(text.slice(last));
    node.replaceWith(frag);
  }
}

function jumpToEvidence(id) {
  const row = $(`ev-${id}`);
  if (!row) { showBanner(`${id} is not in this study's evidence ledger.`); return; }
  $("ledger-filter").value = "";
  filterLedger();
  document.querySelectorAll("#ledger tr.flash").forEach((r) => r.classList.remove("flash"));
  row.scrollIntoView({ behavior: "smooth", block: "center" });
  row.classList.add("flash");
  setTimeout(() => row.classList.remove("flash"), 2600);
}

// ---------------------------------------------------------------------------
// Chart
// ---------------------------------------------------------------------------

function renderChart(study) {
  const demo = study.is_mock ? " — Demo data" : "";
  $("chart-summary").textContent = "";
  if (typeof Chart === "undefined") {
    $("chart-message").textContent = "The chart could not load. The statistics and building cards are available below.";
    $("chart-message").hidden = false;
    return;
  }
  $("chart-message").hidden = true;
  if (state.chart) state.chart.destroy();
  const common = { maintainAspectRatio: false, animation: false,
    plugins: { title: { display: !!demo, text: demo ? "Demo data (synthetic)" : "", color: "#7a5a05", font: { size: 16, weight: "700" } } } };
  Chart.defaults.font.size = 15;
  Chart.defaults.color = "#1c2733";

  if (study.solar_results && study.solar_results.length) {
    $("chart-title").textContent = "Estimated payback by building" + demo;
    const names = Object.fromEntries(study.buildings.map((b) => [b.id, b.name || `Property ${b.id}`]));
    const ranked = study.solar_results.filter((r) => r.payback_years != null).slice(0, 15);
    state.chart = new Chart($("chart"), {
      type: "bar",
      data: { labels: ranked.map((r) => `#${r.rank} ${names[r.property_id] || r.property_id}`),
        datasets: [{ label: "Simple payback (years)", data: ranked.map((r) => r.payback_years), backgroundColor: "#1f8a70", borderColor: "#1c2733", borderWidth: 1.5 }] },
      options: { ...common, indexAxis: "y",
        plugins: { ...common.plugins, legend: { display: false } },
        scales: { x: { beginAtZero: true, title: { display: true, text: "Simple payback (years)" } } } },
    });
    $("chart-summary").textContent = ranked.length
      ? `${ranked.length} ranked buildings; shortest payback ${fmt(ranked[0].payback_years, 1)} years (${names[ranked[0].property_id] || ranked[0].property_id}). Payback ignores incentives and financing.${demo}`
      : "No building has an estimated payback.";
    return;
  }

  const stats = study.stats || {};
  const xKey = stats.predictor, yKey = stats.outcome;
  $("chart-title").textContent = xKey && yKey ? `${nice(yKey)} vs ${nice(xKey)}${demo}` : "How it lines up" + demo;
  const byProp = Object.fromEntries(study.measurements.map((m) => [m.property_id, m]));
  const points = study.buildings.map((b) => {
    const m = byProp[b.id] || {};
    return { x: m[xKey] != null ? m[xKey] : b[xKey], y: b[yKey], name: b.name || `Property ${b.id}` };
  }).filter((p) => p.x != null && p.y != null);
  const coef = xKey && stats.coefficients ? stats.coefficients[xKey] : null;
  const datasets = [{ type: "scatter", label: "Buildings", data: points, backgroundColor: "rgba(255,90,31,.85)", borderColor: "#1c2733", borderWidth: 1.5, pointRadius: 7 }];
  if (coef != null && points.length > 1) {
    // Slope = reported coefficient (controls held fixed), drawn through the sample mean.
    const mx = points.reduce((s, p) => s + p.x, 0) / points.length;
    const my = points.reduce((s, p) => s + p.y, 0) / points.length;
    const xs = points.map((p) => p.x);
    const lo = Math.min(...xs), hi = Math.max(...xs);
    datasets.push({ type: "line", label: "Trend (reported coefficient)", data: [
      { x: lo, y: my + coef * (lo - mx) }, { x: hi, y: my + coef * (hi - mx) }],
      borderColor: "#1c2733", borderWidth: 3, borderDash: [8, 5], pointRadius: 0, fill: false });
  }
  state.chart = new Chart($("chart"), {
    type: "scatter", data: { datasets },
    options: { ...common,
      plugins: { ...common.plugins, legend: { display: datasets.length > 1, labels: { boxWidth: 18 } },
        tooltip: { callbacks: { label: (c) => c.raw.name ? `${c.raw.name}: (${fmt(c.raw.x, 1)}, ${fmt(c.raw.y, 1)})` : c.dataset.label } } },
      scales: { x: { title: { display: true, text: nice(xKey) } }, y: { title: { display: true, text: nice(yKey) } } } },
  });
  if (coef == null) {
    $("chart-summary").textContent = "The regression did not produce a result, so no trend line is drawn.";
    return;
  }
  const ci = (stats.confidence_intervals || {})[xKey];
  const p = (stats.p_values || {})[xKey];
  $("chart-summary").textContent = `Coefficient ${fmt(coef, 3)} (95% CI ${ci ? `${fmt(ci[0], 3)} to ${fmt(ci[1], 3)}` : "-"}), ` +
    `p = ${fmt(p, 3)}, n = ${fmt(stats.sample_size, 0)}. The trend line uses the reported coefficient with controls held fixed; an association, not a cause.${demo}`;
}

// ---------------------------------------------------------------------------
// Verdict, KPIs, statistics, solar and outreach (kept from the earlier page)
// ---------------------------------------------------------------------------

function verdict(p) {
  if (p == null) return ["no", "No result"];
  if (p < 0.05) return ["ok", "Clear signal"];
  if (p < 0.1) return ["weak", "Weak signal"];
  return ["no", "Not clear"];
}

function renderKpis(study) {
  const card = (c) => `<div class="kpi"><div class="label">${esc(c.label)}</div><div class="value">${esc(c.value)}</div><div class="hint">${esc(c.hint)}</div></div>`;
  if (study.solar_results) {
    const ranked = study.solar_results.filter((r) => r.payback_years != null);
    const best = ranked[0];
    $("verdict").innerHTML = `<div><div class="small">${study.is_mock ? "Synthetic solar demonstration" : "Solar screening"}</div>` +
      `<div class="big">${ranked.length ? `${ranked.length} candidates with estimated payback` : "No candidate has a known payback"}</div>` +
      `<div class="sub">Ranked by simple payback from measured roof area. Owner emails are drafts for review and were not sent.</div></div>`;
    $("kpis").innerHTML = [
      { label: "Buildings", value: fmt(study.buildings.length, 0), hint: study.is_mock ? "synthetic sample" : "CoStar sample" },
      { label: "Best payback", value: best ? fmt(best.payback_years, 1) + " years" : "-", hint: "screening estimate" },
      { label: "Capacity", value: best ? fmt(best.capacity_kw, 1) + " kW" : "-", hint: "top candidate" },
      { label: "Annual energy", value: best ? fmt(best.annual_energy_kwh, 0) + " kWh" : "-", hint: "top candidate" },
    ].map(card).join("");
    return;
  }
  const s = study.stats || {};
  const key = s.predictor;
  const coef = key && s.coefficients ? s.coefficients[key] : null;
  const p = key && s.p_values ? s.p_values[key] : null;
  const [cls, label] = verdict(p);
  const level = p == null ? 0 : p < 0.01 ? 5 : p < 0.05 ? 4 : p < 0.1 ? 3 : p < 0.25 ? 2 : 1;
  const dir = coef == null ? "" : coef >= 0 ? "higher" : "lower";
  const lead = cls === "ok" ? "Yes:" : cls === "weak" ? "Maybe:" : "Not clear:";
  const sentence = coef == null ? `We couldn't measure a link for ${esc(nice(key))}.`
    : `${lead} buildings with more <em>${esc(nice(key))}</em> tend to have <em>${dir} ${esc(nice(s.outcome))}</em>.`;
  const sub = study.is_mock ? "Demo data: fixed synthetic numbers illustrate the interface; no real association was measured." : p == null ? "" :
    cls === "ok" ? "The model reports evidence of an association, subject to its assumptions. This does not establish causation." :
    cls === "weak" ? "There is a hint of a pattern, but treat it as a lead, not a fact." :
    "The sample doesn't show a dependable pattern. More buildings might change that.";
  $("verdict").innerHTML =
    `<div><div class="small">${study.is_mock ? "Demo data · illustrative result" : "What the evidence says"}</div><div class="big">${sentence}</div><div class="sub">${esc(sub)}</div></div>` +
    `<div class="gauge"><div class="bars">${[1, 2, 3, 4, 5].map((i) => `<i class="${i <= level ? "on" : ""}"></i>`).join("")}</div>` +
    `<div class="glabel">${esc(label)}</div></div>`;
  $("kpis").innerHTML = [
    { label: "Analyzed buildings", value: fmt(s.sample_size, 0), hint: `of ${study.buildings.length} sampled${study.is_mock ? " (demo data)" : ""}` },
    { label: "Effect", value: coef == null ? "-" : (coef > 0 ? "+" : "") + fmt(coef), hint: `change in ${nice(s.outcome)} per unit of ${nice(key)}` },
    { label: "p-value", value: p == null ? "-" : fmt(p, 3), hint: "test of no association under the model assumptions" },
    { label: "Model fit", value: fmt(s.r_squared), hint: "share of variation explained (R²)" },
  ].map(card).join("");
}

/** CSS class for a stats cell that changed after a correction: green if up, red if down. */
function changeClass(before, after) {
  if (before == null || after == null || Math.abs(before - after) < 1e-12) return "";
  return after > before ? "changed-up" : "changed-down";
}

function renderStats(stats) {
  if (!stats || !stats.coefficients) { $("stats").innerHTML = "<p>No statistics for this study.</p>"; return; }
  const old = (state.highlight && state.highlight.statsBefore) || null;
  const rows = Object.keys(stats.coefficients).map((name) => {
    const ci = (stats.confidence_intervals || {})[name];
    const p = (stats.p_values || {})[name];
    const [cls, text] = verdict(p);
    const was = old ? { coef: (old.coefficients || {})[name], ci: (old.confidence_intervals || {})[name], p: (old.p_values || {})[name] } : {};
    const cCoef = old ? changeClass(was.coef, stats.coefficients[name]) : "";
    const cCi = old && ci && was.ci ? changeClass(was.ci[0] + was.ci[1], ci[0] + ci[1]) : "";
    const cP = old ? changeClass(was.p, p) : "";
    return `<tr class="${name === stats.predictor ? "predictor" : ""}"><td>${esc(nice(name))}</td><td class="${cCoef}">${fmt(stats.coefficients[name])}</td>` +
      `<td class="${cCi}">${ci ? `${fmt(ci[0])} to ${fmt(ci[1])}` : "-"}</td><td class="${cP}">${fmt(p, 3)}</td><td><span class="pill ${cls}">${text}</span></td></tr>`;
  }).join("");
  $("stats").innerHTML =
    `<div class="table-wrap"><table><thead><tr><th>Factor</th><th>Effect</th><th>95% CI</th><th>p-value</th><th>Strength</th></tr></thead><tbody>${rows}</tbody></table></div>` +
    `<p class="stat-note">${esc(stats.interpretation || "")}</p>` +
    `<p class="stat-note">Controls used: ${esc((stats.controls || []).join(", ") || "none")}.</p>`;
}

function renderBuildingTable(study) {
  const byProp = Object.fromEntries(study.measurements.map((m) => [m.property_id, m]));
  const head = ["Building", "Units", "Year built", "Max rent ($)", "Vacancy %", "Green cover %", "Paved %", "Parking (aerial)", "Parking (CoStar)"];
  $("evidence").innerHTML = `<thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>` +
    study.buildings.map((b) => {
      const m = byProp[b.id] || {};
      return `<tr><td>${esc(b.name || b.id)}</td><td>${fmt(b.total_num_units, 0)}</td><td>${b.year_built != null ? esc(b.year_built) : "-"}</td>` +
        `<td>${fmt(b.max_rent, 0)}</td><td>${fmt(b.vacancy_percent)}</td><td>${fmt(m.green_cover_percent)}</td>` +
        `<td>${fmt(m.paved_surface_percent)}</td><td>${fmt(m.parking_space_count, 0)}</td><td>${fmt(b.surface_parking_spaces, 0)}</td></tr>`;
    }).join("") + "</tbody>";
}

function renderSolarAndOutreach(study) {
  $("stats-card").hidden = !study.stats;
  const solar = study.solar_results;
  $("solar-card").hidden = !solar;
  $("outreach-card").hidden = !solar || !(study.outreach_drafts || []).length;
  if (!solar) return;
  const byId = Object.fromEntries(study.buildings.map((b) => [b.id, b]));
  const ranked = solar.filter((r) => r.rank != null).slice(0, 10);
  const paybacks = ranked.map((r) => r.payback_years).filter((v) => v != null);
  const lo = Math.min(...paybacks), hi = Math.max(...paybacks);
  $("solar-ranking").innerHTML = "<thead><tr><th>Rank</th><th>Building</th><th>Roof area (sq ft)</th><th>Roof from</th>" +
    "<th>Facing</th><th>kW</th><th>kWh / yr</th><th>Payback (years)</th><th>Savings / unit / mo</th></tr></thead><tbody>" +
    ranked.map((r) => {
      const b = byId[r.property_id] || {};
      const a = r.assumptions || {};
      const roofFrom = sourceLabel(a.roof_area_source) || "-";
      const facing = a.azimuth_deg != null ? `${fmt(a.azimuth_deg, 0)}°` : "-";
      const perUnit = a.savings_per_unit_per_month_usd != null ? `$${fmt(a.savings_per_unit_per_month_usd, 2)}` : "-";
      return `<tr data-id="${esc(r.property_id)}" tabindex="0"><td><span class="rank-dot" style="background:${paybackColor(r.payback_years, lo, hi)}">${esc(r.rank)}</span></td>` +
        `<td>${esc(b.name || `Property ${r.property_id}`)}</td><td>${fmt(r.usable_roof_area_sq_ft, 0)}</td>` +
        `<td>${esc(roofFrom)}</td><td>${esc(facing)}</td>` +
        `<td>${fmt(r.capacity_kw, 1)}</td><td>${fmt(r.annual_energy_kwh, 0)}</td><td><strong>${fmt(r.payback_years, 1)}</strong></td>` +
        `<td>${esc(perUnit)}</td></tr>`;
    }).join("") + "</tbody>";
  $("solar-ranking").querySelectorAll("tbody tr").forEach((row) => {
    const id = Number(row.dataset.id);
    row.addEventListener("click", () => selectBuilding(id, "table"));
    row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectBuilding(id, "table"); } });
  });
  const first = ranked[0] && ranked[0].property_id;
  if (first != null && state.selected == null) state.selected = first;
  renderEmail(study, state.selected != null ? state.selected : first);
}

/** The selected building's owner email draft (subject, body, caveats), never sent. */
function renderEmail(study, id) {
  const drafts = study.outreach_drafts || [];
  const draft = drafts.find((d) => d.property_id === id);
  const building = study.buildings.find((b) => b.id === id);
  const name = building ? building.name || `Property ${building.id}` : "this building";
  state.emailText = draft && draft.body_markdown ? `${draft.subject || ""}\n\n${draft.body_markdown}` : "";
  $("copy-email").disabled = !state.emailText;
  $("copy-status").textContent = "";
  $("outreach").innerHTML = !draft || !draft.body_markdown
    ? `<p class="stat-note">No email draft for ${esc(name)}. Drafts are written for the top ${drafts.length || 10} ranked buildings; pick one of them.</p>`
    : `<p class="email-to">For: <strong>${esc(name)}</strong></p>` +
      `<h4 class="email-subject">${esc(draft.subject || "(no subject)")}</h4>` +
      `<div class="email-body">${simpleMarkdown(draft.body_markdown)}</div>` +
      ((draft.caveats || []).length ? `<ul class="email-caveats">${draft.caveats.map((c) => `<li>${esc(c)}</li>`).join("")}</ul>` : "");
}

async function copyEmail() {
  if (!state.emailText) return;
  try {
    await navigator.clipboard.writeText(state.emailText);
    $("copy-status").textContent = "Copied to clipboard.";
  } catch {
    // Clipboard API unavailable (e.g. not a secure context): select the text for manual copy.
    const range = document.createRange();
    range.selectNodeContents($("outreach"));
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    $("copy-status").textContent = "Text selected: press Ctrl/Cmd+C to copy.";
  }
}

// ---------------------------------------------------------------------------
// Structured agent results: plan, property details, interior scans, upgrades, provenance
// ---------------------------------------------------------------------------

const PROPERTY_TYPES = { 11: "Multifamily", 5: "Office", 2: "Industrial", 6: "Retail" };
const CLASSES = { 1: "A", 2: "B", 3: "C", 4: "F" };
const UNAVAILABLE = '<span class="unavail" title="Not recorded in the data source">unavailable</span>';

function resetDetails() {
  state.details = { property: {}, interior: {}, upgrades: {} };
  state.visionSource = {};
  state.replay = null;
  $("plan-box").hidden = true;
  $("plan").innerHTML = "";
  $("run-status").textContent = "";
}

/** Keep the structured payloads (property details, interior scan, upgrade screening) by property. */
function takeDetail(kind, payload) {
  const key = { property_details: "property", interior: "interior", upgrade_screening: "upgrades" }[kind];
  if (key && payload && payload.property_id != null) state.details[key][payload.property_id] = payload;
}

function detailsFromEvents(events) {
  for (const e of events || []) {
    if (e.event_type === "plan") renderPlan(e.payload);
    else if (e.event_type === "tool_call") noteVisionSource(e.payload || {});
    else takeDetail(e.event_type, e.payload || {});
  }
}

/** Per-building measurement provenance from tool_call events (live, cached or demo). */
function noteVisionSource(p) {
  if (p.tool !== "vision_inspect" || p.property_id == null || p.failed) return;
  state.visionSource[p.property_id] = p.source === "mock" ? "demo" : p.cached ? "cached" : "live";
}

function provenance(pid) {
  const study = state.study || {};
  if (study.is_mock) return ["demo", "Demo data (synthetic)"];
  let kind = state.visionSource[pid];
  if (!kind) {  // older saved runs: read it from the evidence summary instead
    const row = (state.evidence || []).find((e) => e.tool === "vision_inspect" && new RegExp(`for property ${pid}\\b`).test(e.summary || ""));
    kind = row ? (/\(cached\)/.test(row.summary) ? "cached" : "live") : "unavailable";
  }
  const label = { live: "Live measurement (this run)", cached: "Cached real measurement (earlier run)",
                  unavailable: "No measurement" }[kind] || kind;
  return [kind, state.replay ? `${label} · recorded showcase run` : label];
}

/** The planner's to-do list from DeepAgents write_todos. */
function renderPlan(payload) {
  const todos = (payload && payload.todos) || [];
  $("plan-box").hidden = !todos.length;
  const icon = { completed: "✓", in_progress: "◐", pending: "○" };
  $("plan").innerHTML = todos.map((t) => `<li class="todo ${esc(t.status)}"><span class="todo-icon" aria-hidden="true">` +
    `${icon[t.status] || "○"}</span><span>${esc(t.content)}</span><small>${esc(String(t.status).replace("_", " "))}</small></li>`).join("");
}

function addNote(data) {
  const li = document.createElement("li");
  li.className = "note";
  li.innerHTML = `<b>note</b> ${esc((data.payload || {}).path || "")} <span class="notes">(${esc((data.payload || {}).chars || 0)} characters, DeepAgents scratch file)</span>`;
  $("progress").appendChild(li);
}

function studyKind(study) {
  if (!study.solar_results) return "research";
  return study.buildings.length === 1 ? "property" : "retrofit";
}

const canEditStudy = (study, m) => !!(m && m.id && (study.stats || study.solar_results));
const editableFields = (study) => studyKind(study) === "research"
  ? ["parking_space_count", "green_cover_percent", "roof_area_sq_ft"] : ["roof_area_sq_ft", "parking_space_count"];

function calibrationLine(field) {
  const c = state.calibration && state.calibration.fields ? state.calibration.fields[field] : null;
  if (!c) return "No corrections recorded for this measure yet.";
  if (c.confidence_multiplier == null) return `Corrected ${c.corrections} time${c.corrections === 1 ? "" : "s"} so far (need ${state.calibration.min_corrections} to judge reliability).`;
  return `Corrected ${c.corrections} times; median correction ${fmt(c.median_relative_error * 100, 0)}% → reliability ${esc(c.reliability)}, ` +
    `confidence × ${fmt(c.confidence_multiplier, 2)} (from correction history; the model is not retrained).`;
}

async function loadCalibration() {
  try {
    const res = await fetch("/api/calibration");
    state.calibration = res.ok ? await res.json() : null;
  } catch { state.calibration = null; }
}

/** Right panel: everything known about one building, with editable measurements. */
function renderInspector(id) {
  const study = state.study;
  const body = $("inspector-body");
  if (!study) return;
  const b = study.buildings.find((x) => x.id === id);
  if (!b) { body.innerHTML = '<p class="stat-note">Select a building on the map, in a table or on a card.</p>'; return; }
  const m = study.measurements.find((x) => x.property_id === id) || null;
  const r = (study.solar_results || []).find((x) => x.property_id === id);
  const det = state.details.property[id], interior = state.details.interior[id];
  const val = (v, f) => (v == null ? UNAVAILABLE : f(v));
  const facts = [
    ["Property id", esc(b.id)], ["Type", val(b.property_type, (v) => esc(PROPERTY_TYPES[v] || `type ${v}`))],
    ["Building size", val(b.building_size_sq_ft, (v) => `${fmt(v, 0)} sq ft (gross)`)], ["Year built", val(b.year_built, esc)],
    ["Class", val(b.building_class, (v) => esc(CLASSES[v] || "unknown"))], ["Units", val(b.total_num_units, (v) => fmt(v, 0))],
    ["Asking rent (max)", val(b.max_rent, (v) => `$${fmt(v, 0)} / month`)], ["Vacancy", val(b.vacancy_percent, (v) => `${fmt(v, 1)}%`)],
    ["Surface parking (CoStar)", val(b.surface_parking_spaces, (v) => fmt(v, 0))],
    ["Coordinates", b.latitude != null ? `${fmt(b.latitude, 5)}, ${fmt(b.longitude, 5)}` : UNAVAILABLE],
  ];
  if (det) {
    const names = (list) => (list && list.length ? esc(list.join(", ")) : UNAVAILABLE);
    facts.push(["True owner", names((det.owners || {}).true_owner)], ["Property manager", names((det.owners || {}).property_manager)],
      ["Current tenants", det.tenants ? `${esc(det.tenants.current_tenant_count)}${det.tenants.tenants ? ": " + esc(det.tenants.tenants.slice(0, 5).join(", ")) : ""}` : UNAVAILABLE]);
  }
  const [kind, label] = provenance(id);
  const editable = canEditStudy(study, m);
  const measures = m ? editableFields(study).map((field) => {
    const spec = EDITABLE[field];
    const shown = m[field] == null ? "unknown" : esc(fmt(m[field], spec.integer ? 0 : 1)) + spec.unit;
    return editable ? `<button type="button" class="edit-value" data-field="${field}"><span>${esc(spec.label)}</span><strong>${shown}</strong><i aria-hidden="true">✎</i></button>`
      : `<div class="edit-value static"><span>${esc(spec.label)}</span><strong>${shown}</strong></div>`;
  }).join("") : '<p class="stat-note">This building has no image measurement.</p>';
  const obs = (m && m.observations) || {};
  const cal = m ? editableFields(study).map((f) => `<li><b>${esc(EDITABLE[f].label)}:</b> ${calibrationLine(f)}</li>`).join("") : "";
  const interiorHtml = !interior ? '<p class="stat-note">Interior scan not checked for this building.</p>'
    : !interior.available ? `<p class="stat-note">No interior scan: ${esc(interior.reason || "")}.</p>`
    : `<p>Matterport scan: ${esc(interior.summary.floor_count)} floor(s), ${fmt(interior.summary.scanned_floor_area, 0)} ${esc(interior.summary.units || "")} scanned, ` +
      `${esc(interior.summary.room_count)} rooms${interior.summary.labelled_rooms.length ? ` (${esc(interior.summary.labelled_rooms.slice(0, 6).join(", "))})` : ""}. ` +
      `Measured ceiling heights: ${interior.summary.measured_ceiling_heights.length ? esc(interior.summary.measured_ceiling_heights.join(", ")) : "none recorded"}.</p>` +
      `<p class="stat-note">Not observed: ${esc((interior.not_observed || []).join(", "))} (${esc(interior.not_observed_reason || "")}).</p>`;
  const solarHtml = r ? `<p>Rank <b>${fmt(r.rank, 0)}</b> · ${fmt(r.capacity_kw, 1)} kW · ${fmt(r.annual_energy_kwh, 0)} kWh/yr · simple payback <b>${fmt(r.payback_years, 1)}</b> years</p>` : "";
  body.innerHTML =
    `<h4 class="insp-name">${esc(b.name || `Property ${b.id}`)}</h4>${address(b) ? `<p class="addr">${esc(address(b))}</p>` : ""}` +
    `<p><span class="prov prov-${esc(kind)}">${esc(label)}</span></p>` +
    `<div class="bimg insp-img"><img alt="Aerial image of property ${esc(b.id)}" src="/api/image/${encodeURIComponent(b.id)}">` +
    `<canvas class="boxes" aria-hidden="true"></canvas><span class="noimg" hidden>No aerial image available</span></div>` +
    `<p class="stat-note">Boxes: detections the vision model confirmed (${esc((m && m.detections || []).length)}). ` +
    `Model confidence: ${m && m.confidence != null ? fmt(m.confidence * 100, 0) + "%" : "unavailable"}. ` +
    `${obs.basis ? esc(obs.basis) + "." : ""}</p>` +
    `<h5>Measurements${editable ? " <small>(click to correct)</small>" : ""}</h5><div class="edit-values">${measures}</div><div class="editor-slot"></div>` +
    (cal ? `<ul class="calibration">${cal}</ul>` : "") +
    `<h5>CoStar record</h5><dl class="facts">${facts.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>` +
    `<h5>Interior</h5>${interiorHtml}` + (solarHtml ? `<h5>Rooftop solar</h5>${solarHtml}` : "");
  const img = body.querySelector("img"), canvas = body.querySelector("canvas");
  const draw = () => drawDetections(img, canvas, (m && m.detections) || [], obs.image_width_px, obs.image_height_px);
  img.addEventListener("load", draw);
  img.addEventListener("error", () => { img.hidden = true; canvas.hidden = true; body.querySelector(".noimg").hidden = false; });
  if (editable) body.querySelectorAll("button.edit-value").forEach((btn) => btn.addEventListener("click", () => openEditor(body, m, btn.dataset.field)));
}

/** Bottom section: the five upgrades for the selected (or top-ranked) building. */
function renderUpgrades(id) {
  const study = state.study;
  const ids = Object.keys(state.details.upgrades).map(Number);
  const pick = state.details.upgrades[id] ? id : (study.solar_results || []).map((r) => r.property_id).find((x) => state.details.upgrades[x]) ?? ids[0];
  const data = state.details.upgrades[pick];
  $("upgrades-card").hidden = !data;
  if (!data) return;
  const b = study.buildings.find((x) => x.id === pick);
  $("upgrades-for").textContent = `for ${b ? b.name || `property ${b.id}` : `property ${pick}`}${pick !== id && id != null ? " (selected building has no screening)" : ""}`;
  const money = (v) => (v == null ? "-" : "$" + fmt(v, 0));
  $("upgrades").innerHTML = `<div class="table-wrap"><table><thead><tr><th>Rank</th><th>Upgrade</th><th>Status</th><th>kW</th><th>kWh / yr</th>` +
    `<th>Cost</th><th>Savings / yr</th><th>Simple payback</th></tr></thead><tbody>` +
    data.upgrades.map((u) => u.status === "estimated"
      ? `<tr><td>${u.rank != null ? esc(u.rank) : "-"}</td><td><b>${esc(u.name)}</b></td><td><span class="pill ok">estimated</span></td><td>${fmt(u.capacity_kw, 1)}</td>` +
        `<td>${fmt(u.annual_kwh, 0)}</td><td>${money(u.cost_usd)}</td><td>${money(u.annual_savings_usd)}</td><td>${u.simple_payback_years != null ? fmt(u.simple_payback_years, 1) + " years" : "-"}</td></tr>`
      : `<tr><td>-</td><td><b>${esc(u.name)}</b></td><td><span class="pill no">not estimated</span></td><td colspan="5" class="reason">${esc(u.reason || "")}</td></tr>`).join("") +
    `</tbody></table></div><p class="stat-note">${esc(data.ranking_basis)}</p>` +
    data.upgrades.filter((u) => u.assumptions).map((u) => `<details class="assumptions"><summary>${esc(u.name)}: assumptions and formula</summary>` +
      `<p><code>${esc(u.formula || "")}</code></p><ul>${Object.entries(u.assumptions).map(([k, a]) =>
        `<li><b>${esc(k.replace(/_/g, " "))}</b>: ${esc(a && a.value != null ? a.value : "unavailable")} <small>(${esc(a && a.source || "")})</small></li>`).join("")}</ul></details>`).join("") +
    `<ul class="email-caveats">${(data.caveats || []).map((c) => `<li>${esc(c)}</li>`).join("")}</ul>`;
}

/** A second, descriptive chart. */
function renderChart2(study) {
  if (typeof Chart === "undefined") return;
  if (state.chart2) state.chart2.destroy();
  state.chart2 = null;
  $("chart2-summary").textContent = "";
  const demo = study.is_mock ? " — Demo data" : "";
  if (study.solar_results) {
    const pts = study.solar_results.filter((r) => r.capacity_kw != null && r.payback_years != null)
      .map((r) => ({ x: r.capacity_kw, y: r.payback_years, rank: r.rank }));
    $("chart2-title").textContent = "System size vs payback" + demo;
    state.chart2 = new Chart($("chart2"), { type: "scatter",
      data: { datasets: [{ label: "Buildings", data: pts, backgroundColor: "rgba(31,138,112,.85)", borderColor: "#1c2733", borderWidth: 1.5, pointRadius: 7 }] },
      options: { maintainAspectRatio: false, animation: false, plugins: { legend: { display: false },
        tooltip: { callbacks: { label: (c) => `rank ${c.raw.rank}: ${fmt(c.raw.x, 1)} kW, ${fmt(c.raw.y, 1)} years` } } },
        scales: { x: { title: { display: true, text: "Estimated capacity (kW)" } }, y: { title: { display: true, text: "Simple payback (years)" } } } } });
    $("chart2-summary").textContent = `${pts.length} buildings with an estimate. Bigger systems do not always pay back faster: yield and roof obstructions matter.`;
    return;
  }
  const stats = study.stats || {};
  const xKey = stats.predictor, yKey = stats.outcome;
  const byProp = Object.fromEntries(study.measurements.map((m) => [m.property_id, m]));
  const pts = study.buildings.map((b) => ({ x: (byProp[b.id] || {})[xKey] ?? b[xKey], y: b[yKey] }))
    .filter((p) => p.x != null && p.y != null).sort((a, b) => a.x - b.x);
  $("chart2-title").textContent = `Average ${nice(yKey)} by ${nice(xKey)} group${demo}`;
  if (pts.length < 6) { $("chart2-summary").textContent = "Too few buildings with both values to group."; return; }
  const third = Math.ceil(pts.length / 3);
  const groups = [pts.slice(0, third), pts.slice(third, 2 * third), pts.slice(2 * third)].filter((g) => g.length);
  const labels = groups.map((g, i) => `${["Low", "Middle", "High"][i]} (${fmt(g[0].x, 1)}–${fmt(g[g.length - 1].x, 1)}, n=${g.length})`);
  const means = groups.map((g) => g.reduce((s, p) => s + p.y, 0) / g.length);
  state.chart2 = new Chart($("chart2"), { type: "bar",
    data: { labels, datasets: [{ label: `Average ${nice(yKey)}`, data: means, backgroundColor: "#ff5a1f", borderColor: "#1c2733", borderWidth: 1.5 }] },
    options: { maintainAspectRatio: false, animation: false, plugins: { legend: { display: false } },
      scales: { y: { title: { display: true, text: `Average ${nice(yKey)}` } }, x: { title: { display: true, text: `${nice(xKey)} (thirds of the sample)` } } } } });
  $("chart2-summary").textContent = `Descriptive only: thirds of the ${pts.length} buildings by ${nice(xKey)}, with no controls. The regression on the left holds size and age fixed.`;
}

// ---- report actions -----------------------------------------------------------------

function reportText() {
  const study = state.study;
  return `${study.brief_markdown.trim()}\n\n## Caveats from the tools\n${(study.caveats || []).map((c) => `- ${c}`).join("\n")}\n`;
}

function downloadText(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function copyText(text, statusEl, fallbackEl) {
  try { await navigator.clipboard.writeText(text); statusEl.textContent = "Copied to clipboard."; }
  catch {
    const range = document.createRange(); range.selectNodeContents(fallbackEl);
    const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
    statusEl.textContent = "Text selected: press Ctrl/Cmd+C to copy.";
  }
}

// ---- map colouring --------------------------------------------------------------------

function seqColor(t) {
  const mix = (a, b) => Math.round(a + (b - a) * t);
  return `rgb(${mix(255, 168)}, ${mix(214, 44)}, ${mix(170, 8)})`;
}

function colorKey(study) {
  const chosen = $("color-by").value;
  if (chosen !== "auto") return chosen;
  return (study.stats && study.stats.predictor) || (study.solar_results ? "payback_years" : "parking_space_count");
}

// ---------------------------------------------------------------------------
// Whole study
// ---------------------------------------------------------------------------

function renderStudy(study, evidence) {
  state.study = study;
  $("results").hidden = false;
  $("study-question").textContent = study.question;
  $("mock-badge").hidden = false;
  $("mock-badge").textContent = study.is_mock ? "Demo data (synthetic)" : "Real CoStar + aerial data";
  $("mock-badge").classList.toggle("live", !study.is_mock);
  $("replay-badge").hidden = !state.replay;
  if (state.replay) $("replay-badge").textContent = `Offline showcase replay · recorded ${String(state.replay.generated_at || "").slice(0, 10)}`;
  $("report-title").textContent = { research: "Research brief", retrofit: "Retrofit proposal", property: "Property report" }[studyKind(study)];
  $("dashboard").hidden = false;
  $("brief").innerHTML = renderMarkdown(study.brief_markdown);
  linkEvidenceTags($("brief"));
  $("caveats").innerHTML = (study.caveats || []).map((c) => `<li>${esc(c)}</li>`).join("");
  $("caveat-box").hidden = !(study.caveats || []).length;
  if (!study.buildings.some((b) => b.id === state.selected)) state.selected = null;
  if (state.selected == null && study.buildings.length) {
    const top = (study.solar_results || []).find((r) => r.rank === 1);
    state.selected = top ? top.property_id : study.buildings[0].id;
  }
  // Each section is drawn on its own: if one fails (for example the map library or its
  // tiles), the rest of the report still appears and the failure is named, not hidden.
  const sections = [
    ["evidence ledger", () => renderLedger(evidence)], ["map", () => renderMap(study)],
    ["building cards", () => renderCards(study)], ["chart", () => renderChart(study)],
    ["second chart", () => renderChart2(study)],
    ["summary", () => renderKpis(study)], ["statistics", () => renderStats(study.stats)],
    ["building table", () => renderBuildingTable(study)], ["solar view", () => renderSolarAndOutreach(study)],
    ["selection", () => { if (state.selected != null) selectBuilding(state.selected, "init"); }],
  ];
  const failed = [];
  for (const [name, draw] of sections) {
    try { draw(); } catch (err) { failed.push(name); console.warn(`CityLab: could not draw the ${name}`, err); }
  }
  if (failed.length) {
    showBanner(`The results are below, but the ${failed.join(", ")} could not be drawn in this browser.`,
      { title: "Part of the page could not be drawn." });
  }
}

function setRunning(running) {
  $("run").disabled = running;
  $("run").textContent = running ? "Working..." : "Find the answer";
}

async function runStudy(question) {
  if (state.controller) return;
  state.lastQuestion = question;
  state.history = [];
  state.lastPanel = null;
  state.selected = null;
  $("what-changed").hidden = true;
  showFormError("");
  showBanner("");
  $("results").hidden = true;
  $("skeleton").hidden = false;
  resetDetails();
  resetProgress();
  loadCalibration();
  $("progress-card").scrollIntoView({ behavior: "smooth", block: "center" });
  setRunning(true);
  const controller = new AbortController();
  state.controller = controller;
  let gotResult = false, evidence = [];
  const body = {
    question, market_id: Number($("market").value), property_type: 11,
    limit: Number($("sample-limit").value), mode: $("mode").value, use_mock_tools: $("demo-mode").checked,
  };
  try {
    await streamStudy(body, (event, data) => {
      if (event === "handoff" || event === "tool_call") addProgress(event, data);
      else if (event === "plan") renderPlan(data.payload);
      else if (event === "note") addNote(data);
      else if (event === "replay") { state.replay = data; $("run-status").textContent = data.message; }
      else if (["property_details", "interior", "upgrade_screening"].includes(event)) takeDetail(event, data.payload || {});
      else if (event === "citation_check") showCitation(data);
      else if (event === "evidence") evidence = data.evidence || [];
      else if (event === "result") {
        gotResult = true;
        finishProgress();
        $("skeleton").hidden = true;
        detailsFromEvents(data.events);
        renderStudy(data.study, evidence); // draws section by section; never mistaken for a network error
      }
      else if (event === "error") throw Object.assign(new Error(data.message || "The study failed."), { kind: "server" });
    }, controller.signal);
    if (!gotResult) throw Object.assign(new Error("The connection closed before the study finished."), { kind: "network" });
  } catch (err) {
    finishProgress();
    $("skeleton").hidden = true;
    if (err.name === "AbortError") {
      showBanner("Cancelled. The server may finish the run in the background; cached results will be reused next time.",
        { title: "Cancelled.", retry: true });
    } else if (err.kind === "showcase") {
      showBanner(err.message, { title: "Not in the offline showcase.", retry: false,
        hint: "This server is running the offline demo. Click one of the sample questions above (Q1-Q3) to replay it." });
    } else if (err.status === 422) {
      showBanner(err.message, { title: "Please check your request.", retry: true });
    } else if (err.kind === "server" || err.kind === "http") {
      showBanner(err.message, { title: err.status === 503 ? "The study could not run right now." : "Something went wrong.",
        retry: true, offerDemo: true });
    } else {
      showBanner("Could not reach the CityLab API. Check that the server is running and your connection is up.",
        { title: "Network problem.", retry: true, offerDemo: true });
    }
  } finally {
    state.controller = null;
    setRunning(false);
  }
}

/** Offline fallback: the saved mock study (GET /api/mock-study), rendered without a ledger. */
async function showSavedDemo() {
  try {
    const res = await fetch("/api/mock-study");
    if (!res.ok) throw new Error();
    showBanner("");
    $("progress-card").hidden = true;
    renderStudy(await res.json(), []);
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch {
    showBanner("The saved demo study is not reachable either. Start the server with: uvicorn backend.main:app",
      { title: "Network problem.", retry: true });
  }
}

// ---------------------------------------------------------------------------
// Examples and sample preview (unchanged behaviour; Demo mode replaces the Data select)
// ---------------------------------------------------------------------------

// The judging scenarios (also recorded in the offline showcase as S1-S3).
const SCENARIO_EXAMPLES = [
  { id: "S1", mode: "research", market_id: 14, question: "Do office buildings with more parking spaces have higher vacancy rates in Houston?" },
  { id: "S2", mode: "retrofit", market_id: 20, question: "Which commercial buildings in Los Angeles have the best rooftop solar retrofit potential?" },
  { id: "S3", mode: "retrofit", market_id: 14, question: "Evaluate property 23453 for building upgrades" },
];

async function loadExamples() {
  let questions = [];
  try {
    const res = await fetch("/api/questions");
    if (res.ok) questions = (await res.json()).questions || [];
  } catch { /* examples are optional */ }
  try {
    for (const q of [...questions, ...SCENARIO_EXAMPLES]) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip";
      b.innerHTML = `<b>${esc(q.id)}</b><span>${esc(q.question)}</span>`;
      b.addEventListener("click", () => {
        $("question").value = q.question;
        $("market").value = String(q.market_id);
        $("mode").value = q.mode;
        $("question").focus();
        loadPreview();
      });
      $("chips").appendChild(b);
    }
  } catch { /* examples are optional */ }
}

async function loadPreview() {
  if (state.previewRequest) state.previewRequest.abort();
  const controller = new AbortController();
  state.previewRequest = controller;
  const demo = $("demo-mode").checked;
  const params = new URLSearchParams({ market_id: $("market").value, property_type: "11",
    limit: $("sample-limit").value, use_mock_tools: String(demo) });
  $("preview-count").textContent = "Loading the selected sample...";
  try {
    const response = await fetch("/api/buildings?" + params, { signal: controller.signal });
    if (!response.ok) throw new Error("The selected sample could not be loaded.");
    const buildings = await response.json();
    if (controller.signal.aborted) return;
    $("preview-count").textContent = `${buildings.length} ${demo ? "synthetic" : "CoStar"} buildings. Previewing records does not call imagery or model APIs.`;
    $("sample-preview").innerHTML = "<thead><tr><th>Building</th><th>Units</th><th>Year</th><th>Class</th><th>Max rent ($)</th><th>Vacancy %</th></tr></thead><tbody>" +
      buildings.map((b) => `<tr><td>${esc(b.name || b.id)}</td><td>${fmt(b.total_num_units, 0)}</td>` +
        `<td>${b.year_built != null ? esc(b.year_built) : "-"}</td><td>${esc(({ 1: "A", 2: "B", 3: "C", 4: "F" })[b.building_class] || "Unknown")}</td>` +
        `<td>${fmt(b.max_rent, 0)}</td><td>${fmt(b.vacancy_percent)}</td></tr>`).join("") + "</tbody>";
  } catch (error) {
    if (error.name !== "AbortError") {
      $("preview-count").textContent = error.message === "Failed to fetch" ? "The API is not reachable; the preview needs the server." : error.message;
      $("sample-preview").innerHTML = "";
    }
  }
}

$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("question").value.trim();
  if (!q) { showFormError("Type a question first."); return; }
  runStudy(q);
});
$("cancel").addEventListener("click", () => { if (state.controller) state.controller.abort(); });
$("show-demo").addEventListener("click", showSavedDemo);
$("retry").addEventListener("click", () => { if (state.lastQuestion) runStudy(state.lastQuestion); });
$("undo").addEventListener("click", undoCorrection);
$("copy-email").addEventListener("click", copyEmail);
$("download-email").addEventListener("click", () => {
  if (state.emailText) downloadText(`citylab-owner-email-${state.selected}.txt`, state.emailText, "text/plain");
});
$("copy-report").addEventListener("click", () => state.study && copyText(reportText(), $("report-status"), $("brief")));
$("download-report").addEventListener("click", () => state.study && downloadText(`citylab-${state.study.id}.md`, reportText(), "text/markdown"));
$("print-report").addEventListener("click", () => {
  document.body.classList.add("print-report");
  window.print();
  setTimeout(() => document.body.classList.remove("print-report"), 500);
});
for (const id of ["color-by", "map-filter"]) $(id).addEventListener("change", () => {
  if (!state.study) return;
  state.keepView = true;
  try { renderMap(state.study); if (state.selected != null) selectBuilding(state.selected, "init"); }
  catch (err) { console.warn("CityLab: could not redraw the map", err); }
});
$("more-cards").addEventListener("click", () => showMoreCards());
$("ledger-filter").addEventListener("input", filterLedger);
for (const id of ["market", "sample-limit", "demo-mode"]) $(id).addEventListener("change", loadPreview);
/** Fills the five landing-page frames with one real building's actual data, then cycles them. */
async function initShowcaseStrip() {
  const strip = $("strip");
  if (!strip) return;
  const frames = [...strip.querySelectorAll(".strip-frame")];
  const dotsBox = strip.querySelector(".strip-dots");
  dotsBox.innerHTML = frames.map((_, i) => `<button type="button" role="tab" aria-label="Step ${i + 1}"></button>`).join("");
  const dots = [...dotsBox.querySelectorAll("button")];
  let step = 0, timer = null;
  const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function show(i) {
    step = i;
    frames.forEach((f, idx) => f.classList.toggle("active", idx === i));
    dots.forEach((d, idx) => d.classList.toggle("active", idx === i));
  }
  function advance() { show((step + 1) % frames.length); }
  dots.forEach((dot, i) => dot.addEventListener("click", () => { show(i); restart(); }));
  function restart() {
    if (timer) clearInterval(timer);
    if (!reduced) timer = setInterval(advance, 3200);
  }

  let data;
  try {
    const res = await fetch("/api/showcase-strip");
    if (!res.ok) throw new Error("not ready");
    data = await res.json();
  } catch {
    strip.querySelectorAll(".strip-placeholder span").forEach((s) => { s.textContent = "No fully-measured building is cached yet."; });
    show(0);
    return;
  }
  const media = [
    [`/api/image/${data.property_id}`, `Aerial photo of ${data.name || "property " + data.property_id}`],
    [`/api/image/${data.property_id}/boxes`, "Vexcel's detections boxed on the aerial photo"],
    [`/api/roof-view/${data.property_id}`, "Shadow-measured roof and per-side sun report"],
  ];
  frames[0].querySelector(".strip-placeholder span").textContent =
    `${data.name || `Property ${data.property_id}`}, built ${data.year_built || "?"}, ${fmt(data.building_size_sq_ft, 0)} sq ft`;
  frames[0].querySelector(".strip-placeholder").classList.add("strip-text");
  [1, 2, 3].forEach((frameIndex, mediaIndex) => {
    const [src, alt] = media[mediaIndex];
    const holder = frames[frameIndex].querySelector(".strip-placeholder");
    const img = new Image();
    img.alt = alt;
    img.loading = "lazy";
    img.onload = () => holder.replaceWith(img);
    img.onerror = () => { holder.querySelector("span").textContent = "Not available for this building."; };
    img.src = src;
  });
  $("strip-payback").textContent = data.payback_years != null ? fmt(data.payback_years, 1) : "?";
  show(0);
  restart();
}
initShowcaseStrip();

loadExamples();
loadPreview();
