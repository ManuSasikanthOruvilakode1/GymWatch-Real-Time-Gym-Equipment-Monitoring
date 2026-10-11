"use strict";

const STATE_TEXT = { in_use: "In use", free: "Free", unknown: "Warming up" };
const cards = new Map();
const histories = new Map();
let statsWindow = 3600;

const $ = (sel, root = document) => root.querySelector(sel);

function fmtDuration(s) {
  if (s == null || !isFinite(s)) return "–";
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : `${m}:${String(sec).padStart(2, "0")}`;
}
const fmtTime = (t) => (t ? new Date(t * 1000).toLocaleTimeString() : "–");

function ensureCard(cam) {
  if (cards.has(cam.id)) return cards.get(cam.id);
  const node = $("#camera-tpl").content.firstElementChild.cloneNode(true);
  $(".name", node).textContent = cam.name;
  const img = $(".feed img", node);
  img.src = `/api/cameras/${encodeURIComponent(cam.id)}/stream.mjpg`;
  img.alt = `Live view: ${cam.name}`;
  $(".chart", node).setAttribute("aria-label", `Occupancy probability, last 60 seconds, ${cam.name}`);
  attachHover(cam.id, $(".chart", node));
  $("#cameras").appendChild(node);
  cards.set(cam.id, node);
  return node;
}

function renderCameras(list) {
  for (const cam of list) {
    const node = ensureCard(cam);
    const pill = $(".pill", node);
    pill.className = `pill ${cam.state}`;
    $(".pill-text", pill).textContent = STATE_TEXT[cam.state] ?? cam.state;
    const pending = cam.pending ? ` · changing to ${STATE_TEXT[cam.pending].toLowerCase()}…` : "";
    $(".age", node).textContent =
      cam.state === "unknown" ? "Waiting for a stable reading" : `${STATE_TEXT[cam.state]} for ${fmtDuration(cam.state_age_s)}${pending}`;
    const warn = $(".warn", node);
    const problems = [];
    if (cam.source_error) problems.push(`Source error: ${cam.source_error}`);
    if (cam.frame_age_s != null && cam.frame_age_s > 5) problems.push(`No new frames for ${Math.round(cam.frame_age_s)} s`);
    if (cam.equipment_mismatch) problems.push(`Camera may be pointing at a ${cam.equipment_seen.replace("_", " ")}`);
    warn.hidden = problems.length === 0;
    warn.textContent = problems.join(" · ");
  }
  const busy = list.filter((c) => c.state === "in_use").length;
  const free = list.filter((c) => c.state === "free").length;
  renderSummary({ busy, free, total: list.length });
}

let lastStats = [];
function renderSummary({ busy, free, total }) {
  const util = lastStats.length ? lastStats.reduce((a, s) => a + s.utilisation, 0) / lastStats.length : 0;
  const sessions = lastStats.reduce((a, s) => a + s.sessions, 0);
  const tiles = [
    ["Machines in use", `${busy} / ${total}`],
    ["Machines free", `${free}`],
    ["Avg utilisation", `${Math.round(util * 100)}%`],
    ["Sessions", `${sessions}`],
  ];
  $("#summary").innerHTML = tiles
    .map(([l, v]) => `<div class="tile"><div class="label">${l}</div><div class="value">${v}</div></div>`)
    .join("");
}

async function refreshStats() {
  try {
    const [stats, sessions] = await Promise.all([
      fetch(`/api/stats?window_s=${statsWindow}`).then((r) => r.json()),
      fetch("/api/sessions?limit=20").then((r) => r.json()),
    ]);
    lastStats = stats;
    for (const s of stats) {
      const node = cards.get(s.camera_id);
      if (!node) continue;
      $(".k-util", node).textContent = `${Math.round(s.utilisation * 100)}%`;
      $(".k-sessions", node).textContent = s.sessions;
      $(".k-mean", node).textContent = s.sessions ? fmtDuration(s.mean_session_s) : "–";
    }
    const names = Object.fromEntries([...cards].map(([id, n]) => [id, $(".name", n).textContent]));
    const body = $("#sessions tbody");
    body.innerHTML = sessions.length
      ? sessions
          .map(
            (s) => `<tr><td>${names[s.camera_id] ?? s.camera_id}</td><td>${fmtTime(s.started_at)}</td>` +
              `<td>${s.ended_at ? fmtTime(s.ended_at) : "<em>ongoing</em>"}</td>` +
              `<td class="num">${fmtDuration(s.duration_s ?? Date.now() / 1000 - s.started_at)}</td></tr>`
          )
          .join("")
      : `<tr><td colspan="4" class="empty">No sessions yet</td></tr>`;
  } catch (e) {
    /* transient; next tick retries */
  }
}

// -- probability chart --------------------------------------------------------------------
const W = 300, H = 80;
function path(ts, vs, t0, span) {
  return vs.map((v, i) => `${i ? "L" : "M"}${(((ts[i] - t0) / span) * W).toFixed(1)},${(H - v * H).toFixed(1)}`).join("");
}

async function refreshCharts() {
  for (const [id, node] of cards) {
    try {
      const h = await fetch(`/api/cameras/${encodeURIComponent(id)}/history?seconds=60`).then((r) => r.json());
      histories.set(id, h);
      drawChart(node, h);
    } catch (e) { /* ignore */ }
  }
}

function drawChart(node, h, hoverIdx = null) {
  const svg = $(".chart svg", node);
  const now = Date.now() / 1000, t0 = now - 60, span = 60;
  let s = `<line class="grid" x1="0" x2="${W}" y1="${H / 2}" y2="${H / 2}"/>` +
          `<line class="grid" x1="0" x2="${W}" y1="${H}" y2="${H}"/>`;
  if (h.t.length > 1) {
    s += `<path class="raw" d="${path(h.t, h.p_raw, t0, span)}"/>`;
    s += `<path class="smooth" d="${path(h.t, h.p_smooth, t0, span)}"/>`;
  }
  if (hoverIdx != null) {
    const x = ((h.t[hoverIdx] - t0) / span) * W;
    s += `<line class="cross" x1="${x}" x2="${x}" y1="0" y2="${H}"/>`;
  }
  svg.innerHTML = s;
}

function attachHover(id, chart) {
  const tip = $(".tip", chart);
  chart.addEventListener("pointermove", (ev) => {
    const h = histories.get(id);
    if (!h || !h.t.length) return;
    const rect = chart.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
    const target = Date.now() / 1000 - 60 + frac * 60;
    let best = 0;
    for (let i = 1; i < h.t.length; i++) if (Math.abs(h.t[i] - target) < Math.abs(h.t[best] - target)) best = i;
    drawChart(cards.get(id), h, best);
    tip.hidden = false;
    tip.style.left = `${((h.t[best] - (Date.now() / 1000 - 60)) / 60) * 100}%`;
    tip.textContent = `${Math.round(Date.now() / 1000 - h.t[best])} s ago · smoothed ${(h.p_smooth[best] * 100).toFixed(0)}% · frame ${(h.p_raw[best] * 100).toFixed(0)}%`;
  });
  chart.addEventListener("pointerleave", () => { tip.hidden = true; });
}

// -- live updates ---------------------------------------------------------------------------
function connect() {
  const conn = $("#conn");
  const es = new EventSource("/api/events");
  es.onopen = () => { conn.textContent = "● live"; conn.classList.add("live"); };
  es.onmessage = (ev) => renderCameras(JSON.parse(ev.data));
  es.onerror = () => { conn.textContent = "reconnecting…"; conn.classList.remove("live"); };
}

$("#window").addEventListener("change", (e) => { statsWindow = Number(e.target.value); refreshStats(); });

fetch("/api/cameras").then((r) => r.json()).then((list) => {
  renderCameras(list);
  refreshStats();
  refreshCharts();
  connect();
  setInterval(refreshCharts, 1000);
  setInterval(refreshStats, 3000);
});
