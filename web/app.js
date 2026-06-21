"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const POSC = { 1: "GK", 2: "DEF", 3: "MID", 4: "FWD" };
const short = (n) => { const p = n.trim().split(" "); return p[p.length - 1]; };

let DATA, byId = {}, squad = null;
const locked = new Set(), excluded = new Set();
const filt = { q: "", pos: "ALL", team: "", availOnly: false, sortKey: "pred", sortDir: -1 };

init();

async function init() {
  try {
    DATA = await (await fetch("/api/data")).json();
  } catch (e) { $("#app").innerHTML = '<div class="loading">Could not reach the model server.<br>Run <code>python serve.py</code> and reload.</div>'; return; }
  DATA.players.forEach(p => byId[p.id] = p);
  squad = DATA.squad;
  renderHeader(); renderMetrics();
  $("#app").replaceWith($("#mainTpl").content.cloneNode(true));
  $("#gwk").textContent = DATA.meta.gw ?? "–";
  buildTeamSelect(); buildThead(); wire();
  renderSquad(); renderTable(); drawScatter();
}

function renderHeader() {
  const m = DATA.meta;
  const modeLabel = m.mode === "opener" ? "SEASON OPENER" : "MID-SEASON";
  $("#hstats").innerHTML = `
    <span class="chip live hideable">ENGINE <b>${m.engine.replace(/_/g, " ")}</b></span>
    <span class="chip hideable">MODE <b>${modeLabel}</b></span>
    <span class="chip">GW <b>${m.gw ?? "–"}</b></span>
    <span class="chip hideable">${m.season}</span>`;
}

function renderMetrics() {
  const m = DATA.meta, mt = m.metric;
  const cards = [];
  if (mt) {
    const beats = mt.mae < mt.baselineMae;
    cards.push({ v: mt.mae, k: "Model MAE", note: `vs ${mt.baselineName} ${mt.baselineMae} ${beats ? "✓ beats it" : ""}`, good: beats });
    cards.push({ v: mt.spearman, k: "Rank corr (ρ)", note: `walk-forward GW ${mt.rounds.join("–")}`, good: mt.spearman > 0.5 });
  } else {
    cards.push({ v: "—", k: "Model MAE", note: "no in-season data yet (opener)" });
    cards.push({ v: "—", k: "Rank corr", note: "seeded from last season" });
  }
  cards.push({ v: m.nPlayers, k: "Players analysed", note: "with a fixture this GW" });
  cards.push({ v: "£" + m.budget.toFixed(0) + "m", k: "Squad budget", note: "15 players · max 3/club" });
  $("#metrics").innerHTML = cards.map(c =>
    `<div class="metric ${c.good ? "good" : ""}"><div class="v">${c.v}</div><div class="k">${c.k}</div><div class="note">${c.note}</div></div>`).join("");
}

/* ---------------- squad / pitch ---------------- */
function renderSquad() {
  const picks = squad.picks.map(p => ({ ...p, ...byId[p.id] }));
  const xi = picks.filter(p => p.inXi);
  const lines = { 4: [], 3: [], 2: [], 1: [] };
  xi.forEach(p => lines[p.pos].push(p));
  const order = [4, 3, 2, 1]; // FWD top -> GK bottom
  $("#pitch").innerHTML = order.map(pos =>
    `<div class="line">${lines[pos].sort((a, b) => b.pred - a.pred).map(plCard).join("")}</div>`).join("");
  const f = `${lines[2].length}-${lines[3].length}-${lines[4].length}`;
  $("#formTag").textContent = "FORMATION " + f;

  const bench = picks.filter(p => !p.inXi).sort((a, b) => a.pos - b.pos);
  $("#bench").innerHTML = bench.map(p =>
    `<div class="bp"><div class="bn">${short(p.name)}</div><div class="bv">${POSC[p.pos]} · £${p.price}</div></div>`).join("");

  $("#budgetVal").innerHTML = `£${squad.budget.toFixed(1)}<small>m</small>`;
  $("#spent").textContent = "£" + squad.cost.toFixed(1) + "m";
  $("#xipts").textContent = squad.xiPoints.toFixed(1);
  $("#costBar").style.width = Math.min(100, squad.cost / squad.budget * 100) + "%";
  $$("#pitch .pl").forEach((el, i) => el.style.animationDelay = (i * 35) + "ms");
}
function plCard(p) {
  return `<div class="pl" data-pos="${p.pos}" data-id="${p.id}" title="${p.name} · ${p.pred} pts">
    <div class="jersey">${p.pred.toFixed(1)}${p.isCaptain ? '<div class="armband">C</div>' : ''}</div>
    <div class="nm">${short(p.name)}</div></div>`;
}

/* ---------------- explorer table ---------------- */
const COLS = [
  { k: "name", t: "Player", num: false }, { k: "posLabel", t: "Pos", num: false },
  { k: "teamShort", t: "Team", num: false }, { k: "price", t: "£", num: true },
  { k: "pred", t: "xPts", num: true }, { k: "form", t: "Form", num: true },
  { k: "value", t: "Value", num: true }, { k: "avail", t: "Avail", num: true },
];
function buildThead() {
  $("#thead").innerHTML = COLS.map(c =>
    `<th class="${c.num ? "num" : ""}" data-k="${c.k}">${c.t}<span class="ar" data-ar="${c.k}"></span></th>`).join("") + "<th>Lock/Ban</th>";
}
function buildTeamSelect() {
  const opts = Object.entries(DATA.teams).sort((a, b) => a[1].localeCompare(b[1]))
    .map(([id, n]) => `<option value="${id}">${n}</option>`).join("");
  $("#teamSel").insertAdjacentHTML("beforeend", opts);
}
function filteredPlayers() {
  let rows = DATA.players.filter(p => {
    if (filt.q && !p.name.toLowerCase().includes(filt.q)) return false;
    if (filt.pos !== "ALL" && p.posLabel !== filt.pos) return false;
    if (filt.team && String(p.team) !== filt.team) return false;
    if (filt.availOnly && p.avail < 1) return false;
    return true;
  });
  const k = filt.sortKey, d = filt.sortDir;
  rows.sort((a, b) => {
    let x = a[k], y = b[k];
    if (typeof x === "string") return x.localeCompare(y) * d;
    return ((x ?? -1) - (y ?? -1)) * d;
  });
  return rows;
}
function renderTable() {
  const rows = filteredPlayers();
  $("#cnt").textContent = `${rows.length} players`;
  $$("[data-ar]").forEach(s => s.textContent = "");
  const ar = $(`[data-ar="${filt.sortKey}"]`); if (ar) ar.textContent = filt.sortDir < 0 ? " ↓" : " ↑";
  const inSquad = new Set(squad.picks.map(p => p.id));
  $("#tbody").innerHTML = rows.slice(0, 400).map(p => {
    const dot = p.avail >= 1 ? "var(--green)" : p.avail > 0 ? "#e9d100" : "var(--magenta)";
    return `<tr>
      <td class="nm-cell">${p.name}${inSquad.has(p.id) ? ' <span style="color:var(--green)">●</span>' : ''}</td>
      <td><span class="ppos ${p.posLabel}">${p.posLabel}</span></td>
      <td class="team-cell">${p.teamShort}</td>
      <td class="num">£${p.price.toFixed(1)}</td>
      <td class="num pred-cell">${p.pred.toFixed(2)}</td>
      <td class="num">${p.form ?? "–"}</td>
      <td class="num">${p.value.toFixed(2)}</td>
      <td class="num"><span class="avail-dot" style="background:${dot}"></span>${(p.avail * 100).toFixed(0)}%</td>
      <td><div class="act">
        <button class="ib lock ${locked.has(p.id) ? "on" : ""}" data-lock="${p.id}" title="Lock into squad">▲</button>
        <button class="ib ban ${excluded.has(p.id) ? "on" : ""}" data-ban="${p.id}" title="Exclude">✕</button>
      </div></td></tr>`;
  }).join("");
}

/* ---------------- live re-optimise ---------------- */
let timer = null;
function reoptimize() {
  const budget = $("#budget").value;
  $("#status").textContent = "Optimising…";
  const qs = new URLSearchParams({ budget, lock: [...locked].join(","), exclude: [...excluded].join(",") });
  fetch("/api/optimize?" + qs).then(r => r.json()).then(s => {
    if (s.error) { $("#status").textContent = "⚠ " + s.error; return; }
    squad = { ...s };
    $("#status").textContent = `✓ XI re-solved · £${s.cost.toFixed(1)}m · ${s.xiPoints.toFixed(1)} xPts`;
    renderSquad(); renderTable(); drawScatter(); renderTags();
  }).catch(() => $("#status").textContent = "⚠ optimisation failed");
}
function renderTags() {
  const tag = (id, cls) => `<div class="tg ${cls}" data-untag="${id}" data-tcls="${cls}">${short(byId[id].name)} <span>✕</span></div>`;
  $("#lockTags").innerHTML = [...locked].map(id => tag(id, "lock")).join("");
  $("#banTags").innerHTML = [...excluded].map(id => tag(id, "ban")).join("");
}

/* ---------------- value scatter (canvas) ---------------- */
function drawScatter() {
  const cv = $("#scatter"), dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = 340; cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 46, r: 14, t: 14, b: 32 };
  const P = DATA.players.filter(p => p.pred > 0);
  const maxP = Math.max(...P.map(p => p.price)) + 0.5, maxY = Math.max(...P.map(p => p.pred)) + 0.5;
  const X = v => pad.l + (v / maxP) * (w - pad.l - pad.r);
  const Y = v => h - pad.b - (v / maxY) * (h - pad.t - pad.b);
  ctx.strokeStyle = "#2a1f47"; ctx.fillStyle = "#6b5d8c"; ctx.font = '11px "JetBrains Mono"';
  for (let g = 0; g <= 4; g++) { const gx = pad.l + g / 4 * (w - pad.l - pad.r); ctx.beginPath(); ctx.moveTo(gx, pad.t); ctx.lineTo(gx, h - pad.b); ctx.stroke(); ctx.fillText("£" + (maxP * g / 4).toFixed(0), gx - 6, h - 14); }
  for (let g = 0; g <= 4; g++) { const gy = Y(maxY * g / 4); ctx.beginPath(); ctx.moveTo(pad.l, gy); ctx.lineTo(w - pad.r, gy); ctx.stroke(); ctx.fillText((maxY * g / 4).toFixed(0), 8, gy + 3); }
  const COL = { 1: "#04f5ff", 2: "#00ff87", 3: "#e9d100", 4: "#e90052" };
  const inSquad = new Set(squad.picks.map(p => p.id));
  P.forEach(p => {
    const x = X(p.price), y = Y(p.pred), big = inSquad.has(p.id);
    ctx.beginPath(); ctx.arc(x, y, big ? 5.5 : 2.4, 0, 7);
    ctx.fillStyle = COL[p.pos] + (big ? "" : "88"); ctx.fill();
    if (big) { ctx.lineWidth = 1.5; ctx.strokeStyle = "#fff"; ctx.stroke(); }
  });
}

/* ---------------- events ---------------- */
function wire() {
  $("#q").addEventListener("input", e => { filt.q = e.target.value.toLowerCase(); renderTable(); });
  $("#posChips").addEventListener("click", e => {
    const c = e.target.closest(".pc"); if (!c) return;
    $$(".pc").forEach(x => x.classList.remove("on")); c.classList.add("on");
    filt.pos = c.dataset.p; renderTable();
  });
  $("#teamSel").addEventListener("change", e => { filt.team = e.target.value; renderTable(); });
  $("#availTog").addEventListener("click", e => { filt.availOnly = !filt.availOnly; e.target.classList.toggle("on", filt.availOnly); renderTable(); });
  $("#thead").addEventListener("click", e => {
    const th = e.target.closest("th[data-k]"); if (!th) return;
    const k = th.dataset.k;
    if (filt.sortKey === k) filt.sortDir *= -1; else { filt.sortKey = k; filt.sortDir = (k === "name" || k === "teamShort" || k === "posLabel") ? 1 : -1; }
    renderTable();
  });
  $("#tbody").addEventListener("click", e => {
    const lk = e.target.closest("[data-lock]"), bn = e.target.closest("[data-ban]");
    if (lk) { const id = +lk.dataset.lock; locked.has(id) ? locked.delete(id) : (locked.add(id), excluded.delete(id)); }
    if (bn) { const id = +bn.dataset.ban; excluded.has(id) ? excluded.delete(id) : (excluded.add(id), locked.delete(id)); }
    if (lk || bn) { renderTags(); reoptimize(); }
  });
  $("#pitch").addEventListener("click", e => {
    const pl = e.target.closest(".pl"); if (!pl) return;
    const id = +pl.dataset.id; excluded.add(id); locked.delete(id); renderTags(); reoptimize();
  });
  document.addEventListener("click", e => {
    const t = e.target.closest("[data-untag]"); if (!t) return;
    const id = +t.dataset.untag; (t.dataset.tcls === "lock" ? locked : excluded).delete(id); renderTags(); reoptimize(); renderTable();
  });
  const b = $("#budget");
  b.addEventListener("input", () => { $("#budgetVal").innerHTML = `£${(+b.value).toFixed(1)}<small>m</small>`; clearTimeout(timer); timer = setTimeout(reoptimize, 280); });
  window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(drawScatter, 150); });
}
