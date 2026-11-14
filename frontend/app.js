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
};

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

function showBanner(message, offerDemo = false) {
  $("error-text").textContent = message || "";
  $("show-demo").hidden = !offerDemo;
  $("error-banner").hidden = !message;
  if (message) $("error-banner").scrollIntoView({ behavior: "smooth", block: "center" });
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
    const error = await res.json().catch(() => ({}));
    const detail = Array.isArray(error.detail) ? "Check the question and the number of buildings."
      : typeof error.detail === "string" ? error.detail : `The server answered ${res.status}.`;
    throw Object.assign(new Error(detail), { kind: "http" });
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

function renderMap(buildings, measurements) {
  state.markers = new Map();
  if (typeof L === "undefined") {
    $("map").textContent = "The map could not load. Check your connection; the building cards are below.";
    return;
  }
  const byProp = Object.fromEntries(measurements.map((m) => [m.property_id, m]));
  if (!state.map) {
    state.map = L.map("map");
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
    }).addTo(state.map);
    state.markerLayer = L.layerGroup().addTo(state.map);
  }
  state.markerLayer.clearLayers();
  const points = [];
  for (const b of buildings) {
    if (b.latitude == null || b.longitude == null) continue;
    const m = byProp[b.id] || {};
    const marker = L.circleMarker([b.latitude, b.longitude], {
      radius: 9, color: "#1c2733", weight: 2, fillColor: "#ff5a1f", fillOpacity: 0.85,
    }).bindPopup(
      `<strong>${esc(b.name || `Property ${b.id}`)}</strong><br>` +
      `Rent: $${fmt(b.max_rent, 0)} · Parking: ${fmt(m.parking_space_count, 0)}<br>` +
      `Green cover: ${fmt(m.green_cover_percent)}%`,
    ).addTo(state.markerLayer);
    marker.on("click", () => selectBuilding(b.id, "map"));
    state.markers.set(b.id, marker);
    points.push([b.latitude, b.longitude]);
  }
  if (points.length) state.map.fitBounds(points, { padding: [30, 30], maxZoom: 13 });
  setTimeout(() => state.map.invalidateSize(), 50);
}

/** Highlight one building's card and map marker together. */
function selectBuilding(id, from) {
  state.selected = id;
  for (const [mid, marker] of state.markers) {
    const on = mid === id;
    marker.setStyle({ radius: on ? 14 : 9, fillColor: on ? "#1f8a70" : "#ff5a1f", weight: on ? 4 : 2 });
    if (on) marker.bringToFront();
  }
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
  if (marker && from === "card" && state.map) {
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

function buildCard(study, b, m) {
  const card = document.createElement("article");
  card.className = "bcard";
  card.dataset.id = String(b.id);
  card.tabIndex = 0;
  const [label, value] = keyMeasurement(study, m || {});
  const where = address(b);
  card.innerHTML =
    `<div class="bimg"><img alt="Aerial image of property ${esc(b.id)}" loading="lazy" decoding="async" src="/api/image/${encodeURIComponent(b.id)}">` +
    `<canvas class="boxes" aria-hidden="true"></canvas><span class="noimg" hidden>No aerial image</span>` +
    `${study.is_mock ? '<span class="demo-tag">Demo data</span>' : ""}</div>` +
    `<div class="bbody"><h4>${esc(b.name || `Property ${b.id}`)}</h4>` +
    `${where ? `<p class="addr">${esc(where)}</p>` : ""}` +
    `<p class="key"><span>${esc(label)}</span><strong>${esc(value)}</strong></p>` +
    `<dl><dt>Rent</dt><dd>${b.max_rent != null ? "$" + fmt(b.max_rent, 0) : "-"}</dd>` +
    `<dt>Units</dt><dd>${fmt(b.total_num_units, 0)}</dd><dt>Built</dt><dd>${b.year_built != null ? esc(b.year_built) : "-"}</dd></dl>` +
    `<p class="pid">Property ${esc(b.id)}</p></div>`;
  const img = card.querySelector("img");
  const canvas = card.querySelector("canvas");
  const draw = () => drawDetections(img, canvas, (m && m.detections) || []);
  img.addEventListener("load", draw);
  img.addEventListener("error", () => { img.hidden = true; canvas.hidden = true; card.querySelector(".noimg").hidden = false; });
  if ("ResizeObserver" in window) new ResizeObserver(draw).observe(img);
  card.addEventListener("click", () => selectBuilding(b.id, "card"));
  card.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectBuilding(b.id, "card"); } });
  if (state.selected === b.id) card.classList.add("selected");
  return card;
}

/** Draw [left, top, right, bottom] source-pixel boxes scaled to the displayed image size. */
function drawDetections(img, canvas, detections) {
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
  const sx = w / img.naturalWidth, sy = h / img.naturalHeight;
  ctx.font = "600 12px Inter, system-ui, sans-serif";
  ctx.lineWidth = 2.5;
  for (const d of detections) {
    if (!Array.isArray(d.bbox) || d.bbox.length !== 4) continue;
    const [left, top, right, bottom] = d.bbox;
    const x = left * sx, y = top * sy, bw = (right - left) * sx, bh = (bottom - top) * sy;
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

function renderStats(stats) {
  if (!stats || !stats.coefficients) { $("stats").innerHTML = "<p>No statistics for this study.</p>"; return; }
  const rows = Object.keys(stats.coefficients).map((name) => {
    const ci = (stats.confidence_intervals || {})[name];
    const p = (stats.p_values || {})[name];
    const [cls, text] = verdict(p);
    return `<tr class="${name === stats.predictor ? "predictor" : ""}"><td>${esc(nice(name))}</td><td>${fmt(stats.coefficients[name])}</td>` +
      `<td>${ci ? `${fmt(ci[0])} to ${fmt(ci[1])}` : "-"}</td><td>${fmt(p, 3)}</td><td><span class="pill ${cls}">${text}</span></td></tr>`;
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
  $("solar-card").hidden = !study.solar_results;
  const top = (study.solar_results || []).filter((r) => r.payback_years != null).slice(0, 10);
  const names = Object.fromEntries(study.buildings.map((b) => [b.id, b.name || b.id]));
  $("solar-ranking").innerHTML = "<thead><tr><th>Rank</th><th>Building</th><th>Roof (sq ft)</th><th>Capacity (kW)</th><th>Energy (kWh/yr)</th><th>Payback (years)</th></tr></thead><tbody>" +
    top.map((r) => `<tr><td>${fmt(r.rank, 0)}</td><td>${esc(names[r.property_id])}</td><td>${fmt(r.usable_roof_area_sq_ft, 0)}</td>` +
      `<td>${fmt(r.capacity_kw, 1)}</td><td>${fmt(r.annual_energy_kwh, 0)}</td><td>${fmt(r.payback_years, 1)}</td></tr>`).join("") + "</tbody>";
  const drafts = study.outreach_drafts || [];
  $("outreach-card").hidden = !drafts.length;
  $("outreach").innerHTML = drafts.slice(0, 1).map((d) => `<h4>${esc(d.subject || "Draft")}</h4>` +
    simpleMarkdown(d.body_markdown || "No draft text available.")).join("") +
    (drafts.length > 1 ? `<p class="stat-note">${drafts.length - 1} more drafts are in the result for review. None were sent.</p>` : "");
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
  $("brief").innerHTML = renderMarkdown(study.brief_markdown);
  linkEvidenceTags($("brief"));
  $("caveats").innerHTML = (study.caveats || []).map((c) => `<li>${esc(c)}</li>`).join("");
  $("caveat-box").hidden = !(study.caveats || []).length;
  renderLedger(evidence);
  renderMap(study.buildings, study.measurements);
  renderCards(study);
  renderChart(study);
  renderKpis(study);
  renderStats(study.stats);
  renderBuildingTable(study);
  renderSolarAndOutreach(study);
}

function setRunning(running) {
  $("run").disabled = running;
  $("run").textContent = running ? "Working..." : "Find the answer";
}

async function runStudy(question) {
  if (state.controller) return;
  showFormError("");
  showBanner("");
  $("results").hidden = true;
  resetProgress();
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
      else if (event === "citation_check") showCitation(data);
      else if (event === "evidence") evidence = data.evidence || [];
      else if (event === "result") { gotResult = true; finishProgress(); renderStudy(data.study, evidence); }
      else if (event === "error") throw Object.assign(new Error(data.message || "The study failed."), { kind: "server" });
    }, controller.signal);
    if (!gotResult) throw Object.assign(new Error("The connection closed before the study finished."), { kind: "network" });
  } catch (err) {
    finishProgress();
    if (err.name === "AbortError") {
      showBanner("Cancelled. The server may finish the run in the background; cached results will be reused next time.");
    } else if (err.kind === "server" || err.kind === "http") {
      showBanner(err.message, true);
    } else {
      showBanner("Could not reach the CityLab API. Check that the server is running.", true);
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
    showBanner("The saved demo study is not reachable either. Start the server with: uvicorn backend.main:app");
    $("show-demo").hidden = true;
  }
}

// ---------------------------------------------------------------------------
// Examples and sample preview (unchanged behaviour; Demo mode replaces the Data select)
// ---------------------------------------------------------------------------

async function loadExamples() {
  try {
    const res = await fetch("/api/questions");
    if (!res.ok) return;
    const data = await res.json();
    for (const q of data.questions || []) {
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
$("more-cards").addEventListener("click", () => showMoreCards());
$("ledger-filter").addEventListener("input", filterLedger);
for (const id of ["market", "sample-limit", "demo-mode"]) $(id).addEventListener("change", loadPreview);
loadExamples();
loadPreview();
