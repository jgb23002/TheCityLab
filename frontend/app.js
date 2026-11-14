/** CityLab frontend: asks the API, shows progress, brief, map, chart and evidence. */
"use strict";

const $ = (id) => document.getElementById(id);
let map = null;
let markerLayer = null;
let chart = null;
let previewRequest = null;

function esc(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}

function fmt(value, digits = 2) {
  if (value == null || Number.isNaN(value)) return "-";
  return typeof value === "number" ? Number(value.toFixed(digits)).toLocaleString() : esc(value);
}

/** Tiny markdown: paragraphs, **bold**, and "- " bullets. Input is escaped first. */
function renderMarkdown(md) {
  const blocks = String(md || "").split(/\n{2,}/);
  return blocks.map((block) => {
    const lines = block.split("\n");
    const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    if (lines.every((l) => /^\s*[-*] /.test(l))) {
      return "<ul>" + lines.map((l) => `<li>${inline(l.replace(/^\s*[-*] /, ""))}</li>`).join("") + "</ul>";
    }
    return `<p>${inline(lines.join(" "))}</p>`;
  }).join("");
}

function showError(message) {
  const box = $("error");
  box.textContent = message;
  box.hidden = !message;
}

function showProgress(events, completed = true) {
  const list = $("progress");
  $("progress-card").hidden = false;
  $("progress-card").scrollIntoView({ behavior: "smooth", block: "center" });
  // Collapse per-building events to the last actual update for each stage.
  const stages = new Map(events.map((event) => [event.event_type || "run", event]));
  list.innerHTML = [...stages.values()].map((event) =>
    `<li class="${completed ? "done" : "active"}">${esc(event.message)}</li>`).join("");
}

function renderMap(buildings, measurements) {
  if (typeof L === "undefined") {
    $("map").textContent = "The map could not load. Check your connection; the building evidence is available below.";
    return;
  }
  const byProp = Object.fromEntries(measurements.map((m) => [m.property_id, m]));
  if (!map) {
    map = L.map("map");
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);
    markerLayer = L.layerGroup().addTo(map);
  }
  markerLayer.clearLayers();
  const points = [];
  for (const b of buildings) {
    if (b.latitude == null || b.longitude == null) continue;
    const m = byProp[b.id] || {};
    L.circleMarker([b.latitude, b.longitude], {
      radius: 8, color: "#ff5a1f", fillColor: "#ff5a1f", fillOpacity: 0.7,
    }).bindPopup(
      `<strong>${esc(b.name || b.id)}</strong><br>` +
      `Green cover: ${fmt(m.green_cover_percent)}%<br>` +
      `Paved surface: ${fmt(m.paved_surface_percent)}%<br>` +
      `Max rent: $${fmt(b.max_rent, 0)}<br>` +
      `Vacancy: ${fmt(b.vacancy_percent)}%`
    ).addTo(markerLayer);
    points.push([b.latitude, b.longitude]);
  }
  if (points.length) map.fitBounds(points, { padding: [30, 30], maxZoom: 13 });
  setTimeout(() => map.invalidateSize(), 50);
}

function renderChart(study) {
  if (typeof Chart === "undefined") {
    $("chart-message").textContent = "The chart could not load. The statistics and building evidence are available below.";
    $("chart-message").hidden = false;
    return;
  }
  $("chart-message").hidden = true;
  if (study.solar_results) {
    const top = study.solar_results.filter((result) => result.payback_years != null).slice(0, 10);
    if (chart) chart.destroy();
    chart = new Chart($("chart"), {
      type: "bar",
      data: { labels: top.map((result) => String(result.property_id)), datasets: [
        { label: "Estimated payback (years)", data: top.map((result) => result.payback_years), backgroundColor: "#1f8a70" },
      ] },
      options: { maintainAspectRatio: false, scales: { y: { beginAtZero: true, title: { display: true, text: "Estimated payback (years)" } } } },
    });
    return;
  }
  const byProp = Object.fromEntries(study.measurements.map((m) => [m.property_id, m]));
  const stats = study.stats || {};
  const xKey = stats.predictor || "green_cover_percent";
  const yKey = stats.outcome || "max_rent";
  const points = study.buildings
    .map((b) => ({ x: (byProp[b.id] || {})[xKey], y: b[yKey], name: b.name }))
    .filter((p) => p.x != null && p.y != null);
  if (chart) chart.destroy();
  chart = new Chart($("chart"), {
    type: "scatter",
    data: { datasets: [{ label: "Buildings", data: points, backgroundColor: "rgba(255,90,31,.85)", borderColor: "#1c2733", borderWidth: 1.5, pointRadius: 7 }] },
    options: {
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => `${c.raw.name || ""}: (${c.raw.x}, ${c.raw.y})` } },
      },
      scales: { x: { title: { display: true, text: xKey } }, y: { title: { display: true, text: yKey } } },
    },
  });
}

function verdict(p) {
  if (p == null) return ["no", "No result"];
  if (p < 0.05) return ["ok", "Clear signal"];
  if (p < 0.1) return ["weak", "Weak signal"];
  return ["no", "Not clear"];
}

const NAMES = {
  green_cover_percent: "green space", paved_surface_percent: "paved surface",
  max_rent: "asking rent", vacancy_percent: "vacancy", total_num_units: "number of units",
};
const nice = (k) => NAMES[k] || String(k || "this factor").replace(/_/g, " ");

function renderKpis(study) {
  if (study.solar_results) {
    const ranked = study.solar_results.filter((result) => result.payback_years != null);
    const best = ranked[0];
    $("verdict").innerHTML = `<div><div class="small">${study.is_mock ? "Synthetic solar demonstration" : "Solar screening"}</div>` +
      `<div class="big">${ranked.length ? `${ranked.length} candidates with estimated payback` : "No candidate has a known payback"}</div>` +
      `<div class="sub">Ranked using measured roof area and returned solar assumptions. Owner email is a draft for review.</div></div>`;
    const cards = [
      { label: "Buildings", value: fmt(study.buildings.length, 0), hint: study.is_mock ? "synthetic sample" : "CoStar sample" },
      { label: "Best payback", value: best ? fmt(best.payback_years, 1) + " years" : "-", hint: "screening estimate" },
      { label: "Capacity", value: best ? fmt(best.capacity_kw, 1) + " kW" : "-", hint: "top candidate" },
      { label: "Annual energy", value: best ? fmt(best.annual_energy_kwh, 0) + " kWh" : "-", hint: "top candidate" },
    ];
    $("kpis").innerHTML = cards.map((card) => `<div class="kpi"><div class="label">${esc(card.label)}</div>` +
      `<div class="value">${esc(card.value)}</div><div class="hint">${esc(card.hint)}</div></div>`).join("");
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
  const sentence = coef == null
    ? `We couldn't measure a link for ${esc(nice(key))}.`
    : `${lead} buildings with more <em>${esc(nice(key))}</em> tend to have <em>${dir} ${esc(nice(s.outcome))}</em>.`;
  const sub = study.is_mock ? "Fixed synthetic numbers illustrate the interface. No regression was fitted and no real association was measured." : p == null ? "" :
    cls === "ok" ? "The model reports evidence of an association, subject to its assumptions. This does not establish causation." :
    cls === "weak" ? "There is a hint of a pattern, but the sample is small, so treat it as a lead, not a fact." :
    "The sample doesn't show a dependable pattern. More buildings might change that.";
  $("verdict").innerHTML =
    `<div><div class="small">${study.is_mock ? "Illustrative mock result" : "What the evidence says"}</div><div class="big">${sentence}</div><div class="sub">${esc(sub)}</div></div>` +
    `<div class="gauge"><div class="bars">${[1,2,3,4,5].map((i) => `<i class="${i <= level ? "on" : ""}"></i>`).join("")}</div>` +
    `<div class="glabel">${esc(label)}</div></div>`;
  const cards = [
    { label: "Analyzed buildings", value: fmt(s.sample_size, 0), hint: `of ${study.buildings.length} sampled${study.is_mock ? " (synthetic)" : ""}` },
    { label: "Effect", value: coef == null ? "-" : (coef > 0 ? "+" : "") + fmt(coef), hint: `change in ${nice(s.outcome)} per 1 point of ${nice(key)}` },
    { label: "p-value", value: p == null ? "-" : fmt(p, 3), hint: "test of no association under the model assumptions" },
    { label: "Model fit", value: fmt(s.r_squared), hint: "share of variation explained (R\u00b2)" },
  ];
  $("kpis").innerHTML = cards.map((c) =>
    `<div class="kpi"><div class="label">${esc(c.label)}</div>` +
    `<div class="value">${esc(c.value)}</div><div class="hint">${esc(c.hint)}</div></div>`).join("");
}

function renderStats(stats) {
  if (!stats || !stats.coefficients) { $("stats").innerHTML = "<p>No statistics for this study.</p>"; return; }
  const rows = Object.keys(stats.coefficients).map((name) => {
    const ci = (stats.confidence_intervals || {})[name];
    const p = (stats.p_values || {})[name];
    const [cls, text] = verdict(p);
    const bold = name === stats.predictor ? "font-weight:700" : "";
    return `<tr style="${bold}"><td>${esc(nice(name))}</td><td>${fmt(stats.coefficients[name])}</td>` +
      `<td>${ci ? `${fmt(ci[0])} to ${fmt(ci[1])}` : "-"}</td><td>${fmt(p, 3)}</td><td><span class="pill ${cls}">${text}</span></td></tr>`;
  }).join("");
  $("stats").innerHTML =
    `<div class="table-wrap"><table><thead><tr><th>Factor</th><th>Effect</th><th>95% range</th><th>p-value</th><th>Strength</th></tr></thead><tbody>${rows}</tbody></table></div>` +
    `<p class="stat-note">${esc(stats.interpretation || "")}</p>`;
}

function renderEvidence(study) {
  const byProp = Object.fromEntries(study.measurements.map((m) => [m.property_id, m]));
  const head = ["Building", "Units", "Year built", "Max rent ($)", "Vacancy %", "Green cover %", "Paved %", "Parking (AI)", "Parking (CoStar)"];
  const rows = study.buildings.map((b) => {
    const m = byProp[b.id] || {};
    return `<tr><td>${esc(b.name || b.id)}</td><td>${fmt(b.total_num_units, 0)}</td><td>${fmt(b.year_built, 0)}</td>` +
      `<td>${fmt(b.max_rent, 0)}</td><td>${fmt(b.vacancy_percent)}</td><td>${fmt(m.green_cover_percent)}</td>` +
      `<td>${fmt(m.paved_surface_percent)}</td><td>${fmt(m.parking_space_count, 0)}</td><td>${fmt(b.surface_parking_spaces, 0)}</td></tr>`;
  }).join("");
  $("evidence").innerHTML = `<thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows}</tbody>`;
}

function renderStudy(study) {
  $("results").hidden = false;
  $("study-question").textContent = study.question;
  $("mock-badge").hidden = false;
  $("mock-badge").textContent = study.is_mock ? "Mock / synthetic data" : "Real CoStar + aerial data";
  $("mock-badge").classList.toggle("live", !study.is_mock);
  $("brief").innerHTML = renderMarkdown(study.brief_markdown);
  $("caveats").innerHTML = (study.caveats || []).map((c) => `<li>${esc(c)}</li>`).join("");
  $("caveat-box").hidden = !(study.caveats || []).length;
  renderMap(study.buildings, study.measurements);
  renderChart(study);
  renderKpis(study);
  renderStats(study.stats);
  renderEvidence(study);
  $("stats-card").hidden = !study.stats;
  $("solar-card").hidden = !study.solar_results;
  const top = (study.solar_results || []).filter((result) => result.payback_years != null).slice(0, 10);
  const names = Object.fromEntries(study.buildings.map((building) => [building.id, building.name || building.id]));
  $("solar-ranking").innerHTML = "<thead><tr><th>Rank</th><th>Building</th><th>Roof (sq ft)</th><th>Capacity (kW)</th><th>Energy (kWh/yr)</th><th>Payback (years)</th></tr></thead><tbody>" +
    top.map((result) => `<tr><td>${fmt(result.rank, 0)}</td><td>${esc(names[result.property_id])}</td>` +
      `<td>${fmt(result.usable_roof_area_sq_ft, 0)}</td><td>${fmt(result.capacity_kw, 1)}</td>` +
      `<td>${fmt(result.annual_energy_kwh, 0)}</td><td>${fmt(result.payback_years, 1)}</td></tr>`).join("") + "</tbody>";
  const drafts = study.outreach_drafts || [];
  $("outreach-card").hidden = !drafts.length;
  $("outreach").innerHTML = drafts.map((draft) => `<h4>${esc(draft.subject || "Draft")}</h4>` +
    renderMarkdown(draft.body_markdown || "No draft text available.")).join("");
}

async function runStudy(question) {
  if ($("run").disabled) return;
  showError("");
  $("run").disabled = true;
  $("run").textContent = "Working...";
  $("results").hidden = true;
  showProgress([{ message: "Running your study. Cached measurements are reused; fresh aerial analysis may take several minutes." }], false);
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, market_id: Number($("market").value), property_type: 11,
        limit: Number($("sample-limit").value), mode: $("mode").value,
        use_mock_tools: $("data-source").value === "mock" }),
    });
    if (!res.ok) {
      const error = await res.json().catch(() => ({}));
      throw new Error(typeof error.detail === "string" ? error.detail : `The server answered ${res.status}. Check the selected sample and settings.`);
    }
    const data = await res.json();
    showProgress(data.events || []);
    renderStudy(data.study);
  } catch (err) {
    $("progress-card").hidden = true;
    showError(err.message || "Something went wrong. Try again.");
  } finally {
    $("run").disabled = false;
    $("run").textContent = "Find the answer";
  }
}

async function loadExamples() {
  try {
    const res = await fetch("/api/questions");
    if (!res.ok) return;
    const data = await res.json();
    const box = $("chips");
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
      box.appendChild(b);
    }
  } catch (err) { /* examples are optional */ }
}

async function loadPreview() {
  if (previewRequest) previewRequest.abort();
  const controller = new AbortController();
  previewRequest = controller;
  const params = new URLSearchParams({ market_id: $("market").value, property_type: "11",
    limit: $("sample-limit").value, use_mock_tools: String($("data-source").value === "mock") });
  $("preview-count").textContent = "Loading the selected sample...";
  try {
    const response = await fetch("/api/buildings?" + params, { signal: controller.signal });
    if (!response.ok) throw new Error("The selected sample could not be loaded.");
    const buildings = await response.json();
    if (controller.signal.aborted) return;
    $("preview-count").textContent = `${buildings.length} ${$("data-source").value === "mock" ? "synthetic" : "CoStar"} buildings. Previewing records does not call imagery or model APIs.`;
    $("sample-preview").innerHTML = "<thead><tr><th>Building</th><th>Units</th><th>Year</th><th>Class</th><th>Max rent ($)</th><th>Vacancy %</th></tr></thead><tbody>" +
      buildings.map((building) => `<tr><td>${esc(building.name || building.id)}</td><td>${fmt(building.total_num_units, 0)}</td>` +
        `<td>${fmt(building.year_built, 0)}</td><td>${esc(({1: "A", 2: "B", 3: "C", 4: "F"})[building.building_class] || "Unknown")}</td>` +
        `<td>${fmt(building.max_rent, 0)}</td><td>${fmt(building.vacancy_percent)}</td></tr>`).join("") + "</tbody>";
  } catch (error) {
    if (error.name !== "AbortError") {
      $("preview-count").textContent = error.message;
      $("sample-preview").innerHTML = "";
    }
  }
}

$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("question").value.trim();
  if (!q) { showError("Type a question first."); return; }
  runStudy(q);
});

loadExamples();
loadPreview();
for (const id of ["market", "sample-limit", "data-source"]) $(id).addEventListener("change", loadPreview);
