"use strict";

const themeSwitch = document.querySelector(".theme-switch");
function syncTheme() {
  applyTheme();
  const dark = document.documentElement.dataset.theme === "black-gold";
  themeSwitch.title = dark ? "Switch to light mode" : "Switch to dark mode";
  themeSwitch.setAttribute("aria-label", themeSwitch.title);
  const titleView = "assets/site/figures/FullTitle.svg#" + (dark ? "dark-" : "light-");
  document.querySelector("#title-print").setAttribute("href", titleView + "letters");
  document.querySelector("#title-robot").setAttribute("href", titleView + "robot");
}
syncTheme();
themeSwitch.hidden = false;
themeSwitch.addEventListener("click", () => {
  savedTheme = document.documentElement.dataset.theme === "black-gold" ? "light-gold" : "black-gold";
  try { localStorage.setItem(themeStorageKey, savedTheme); } catch { /* Keep the choice for this page. */ }
  const url = new URL(location.href);
  if (url.searchParams.has("theme")) {
    url.searchParams.set("theme", savedTheme);
    history.replaceState(history.state, "", url);
  }
  syncTheme();
});
systemDarkMode.addEventListener("change", syncTheme);
window.addEventListener("popstate", syncTheme);
window.addEventListener("storage", event => {
  if (event.key === themeStorageKey || event.key === null) {
    savedTheme = event.newValue;
    syncTheme();
  }
});

const dimensions = ["Navigation", "Grounding", "Geometry", "Physics", "Memory", "Constraints"];
const table = document.querySelector("#results tbody");
const profiles = Array.from(table.rows, row => {
  const cells = row.cells;
  return {
    row,
    name: cells[1].querySelector(".model-name").textContent,
    family: row.dataset.family,
    tasks: Number(row.dataset.tasks),
    overall: [Number(cells[2].textContent), parseFloat(cells[4].textContent)],
    dimensions: JSON.parse(row.dataset.dimensions)
  };
});

const dialog = document.querySelector("#model-profile");
function showProfile(profile) {
  document.querySelector("#profile-title").textContent = profile.name;
  document.querySelector("#profile-summary").textContent =
    profile.family + " · " + profile.tasks + " tasks";
  const body = document.querySelector("#profile-body");
  body.replaceChildren();
  [...dimensions, "Overall"].forEach((label, index) => {
    const [score, success] = profile.dimensions[index] || profile.overall;
    const tr = body.insertRow();
    const heading = document.createElement("th");
    heading.scope = "row";
    heading.textContent = label;
    tr.append(heading);
    for (const value of [score.toFixed(2), success.toFixed(2) + "%"]) {
      const cell = tr.insertCell();
      cell.className = "numeric";
      cell.textContent = value;
    }
  });
  dialog.showModal();
}

document.querySelector("#close-dialog").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", event => {
  if (event.target !== dialog) return;
  const bounds = dialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right ||
      event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
});

const search = document.querySelector("#search");
const dimension = document.querySelector("#dimension");
const grouped = document.querySelector("#grouped");
const sortButtons = document.querySelectorAll("[data-rank-by]");
let metric = 0;
const familyOrder = ["VLM", "VLA", "WAM"];
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const moveOptions = {duration: 400, easing: "cubic-bezier(.22,.75,.25,1)"};

function setupTitle() {
  const stage = document.querySelector(".wordmark-stage");
  const replay = stage.querySelector(".title-replay");
  const ring = stage.querySelector(".title-ring");
  const logoHoldMs = 500;
  let introTimer;
  let playing = false;

  function travel(expanded) {
    stage.dataset.expanded = String(expanded);
    return Promise.allSettled(ring.getAnimations().map(animation => animation.finished));
  }

  async function play() {
    if (playing || reducedMotion.matches) return;
    clearTimeout(introTimer);
    playing = true;
    if (stage.dataset.expanded === "true") {
      stage.classList.add("title-replaying");
      await travel(false);
      await new Promise(resolve => setTimeout(resolve, logoHoldMs));
    }
    await travel(true);
    stage.classList.remove("title-replaying");
    playing = false;
  }

  stage.dataset.expanded = String(reducedMotion.matches);
  replay.hidden = reducedMotion.matches;
  requestAnimationFrame(() => requestAnimationFrame(() => stage.classList.add("title-ready")));
  replay.addEventListener("pointerenter", event => {
    if (event.pointerType === "mouse") play();
  });
  replay.addEventListener("click", play);
  reducedMotion.addEventListener("change", () => {
    clearTimeout(introTimer);
    stage.dataset.expanded = "true";
    replay.hidden = reducedMotion.matches;
  });

  function scheduleIntro() {
    if (!playing && stage.dataset.expanded === "false") introTimer = setTimeout(play, 650);
  }
  if (document.readyState === "complete") scheduleIntro();
  else window.addEventListener("load", scheduleIntro, {once: true});
}
setupTitle();

// Capture the on-screen position, including any movement still in progress.
function rowPositions(body) {
  const positions = new Map();
  if (reducedMotion.matches) return positions;
  const top = body.getBoundingClientRect().top;
  for (const row of body.rows) {
    if (row.dataset.model) positions.set(row.dataset.model, row.getBoundingClientRect().top - top);
    for (const animation of row.getAnimations()) animation.cancel();
  }
  return positions;
}

function moveRows(body, positions) {
  if (reducedMotion.matches) return;
  const top = body.getBoundingClientRect().top;
  // Read final positions before starting any transforms.
  const rows = Array.from(body.rows, row => [row, row.getBoundingClientRect().top - top]);
  for (const [row, y] of rows) {
    const before = positions.get(row.dataset.model);
    if (before === undefined) {
      row.animate([{opacity: 0}, {opacity: 1}], {duration: 230});
    } else if (Math.abs(before - y) > 1) {
      row.animate([{transform: 'translateY(' + (before - y) + 'px)'}, {transform: "none"}], moveOptions);
    }
  }
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

// The strip plots and resource comparison use the leaderboard's Interactive Scores.
function renderDomainStrips() {
  const host = document.querySelector("#domain-strips");
  const winnerColors = {"Claude Fable 5.2": "fable", "GPT-6.1-Sol": "sol", "GPT-6-Astra": "astra"};
  const maxScore = Math.max(...profiles.flatMap(profile => profile.dimensions.map(pair => pair[0])));
  dimensions.forEach((dimension, index) => {
    const row = node("div", "domain-strip");
    const rail = node("span", "domain-rail");
    const groups = new Map();
    for (const profile of profiles) {
      const score = profile.dimensions[index][0];
      if (!groups.has(score)) groups.set(score, []);
      groups.get(score).push(profile.name);
    }
    const leaderScore = Math.max(...groups.keys());
    for (const [score, names] of groups) {
      const isLeader = score === leaderScore;
      const labels = isLeader ? names : [names.join(", ")];
      labels.forEach((label, position) => {
        const dot = node("i", "domain-dot");
        dot.style.left = (100 * score / maxScore) + "%";
        dot.title = dimension + " · " + score.toFixed(2) + " · " + label;
        if (isLeader) {
          dot.classList.add("winner", winnerColors[label]);
          if (labels.length > 1) dot.style.top = (position === 0 ? "26%" : "74%");
        }
        rail.append(dot);
      });
    }
    row.setAttribute("role", "img");
    row.setAttribute("aria-label", dimension + ": " + groups.get(leaderScore).join(" and ") + " lead at " + leaderScore.toFixed(2) + " Interactive Score.");
    row.append(node("span", "", dimension), rail);
    host.append(row);
  });
}
renderDomainStrips();

function renderQaComparison() {
  // Mean terminal QA over all 60 tasks, from the paper's paired QA outcomes table.
  const qaScores = {
    "Claude Fable 5.2": ["Fable", 73.61], "GPT-6.1-Sol": ["Sol", 84.44],
    "GPT-6-Astra": ["Astra", 78.33], "DeepSeek-V4.1-Flash": ["DeepSeek", 66.11],
    "Kimi-K3.1-Pro": ["Kimi", 75.83], "Qwen3.8-27B": ["Qwen", 71.33],
    "Gemini-Decision-1": ["Gemini", 63.33], "Jev-1.14.0": ["Jev", 73.22]
  };
  const body = document.querySelector("#qa-outcomes");
  for (const profile of profiles) {
    if (!qaScores[profile.name]) continue;
    const [name, qa] = qaScores[profile.name];
    const success = profile.overall[1];
    const row = body.insertRow();
    const heading = document.createElement("th");
    heading.scope = "row";
    heading.textContent = name;
    row.append(heading);
    row.title = `${profile.name} · Physical success ${success.toFixed(2)}% · QA ${qa.toFixed(2)}/100`;
    for (const [value, style] of [[success, "success"], [qa, "score"]]) {
      const reading = node("span", "qa-reading");
      const text = value.toFixed(1) + (style === "success" ? "%" : "");
      reading.append(gauge(value, 100, style), node("span", "", text));
      row.insertCell().append(reading);
    }
  }
}
renderQaComparison();

function setupResourcePlot() {
  // Mean elapsed minutes and millions of tokens per task, from Detailed Resource Accounting.
  const resources = {
    "Claude Fable 5.2": [51.12, 3.07], "GPT-6.1-Sol": [46.36, 2.44], "GPT-6-Astra": [40.80, 2.25],
    "DeepSeek-V4.1-Flash": [171.58, 7.22], "Kimi-K3.1-Pro": [90.70, 3.18], "Qwen3.8-27B": [51.05, 6.47],
    "Gemini-Decision-1": [11.66, .77], "Jev-1.14.0": [12.53, 1.26], "GR00T-N1.5": [23.83],
    "π0.5 (R1Pro SFT)": [30.41], "Xiaomi XR-1": [31.69], "Awomo-0.5I": [62.51], "OpenWAM-Alpha": [27.66]
  };
  const plot = document.querySelector("#efficiency-scatter");
  const readout = document.querySelector(".point-readout");
  const controls = document.querySelectorAll("[data-resource]");
  const labels = {"Claude Fable 5.2": "Fable", "Qwen3.8-27B": "Qwen", "Jev-1.14.0": "Jev"};
  function render(axis) {
    const byTime = axis === "minutes";
    const xMax = byTime ? 180 : 8;
    const yMax = 30;
    const samples = profiles.flatMap(profile => {
      const [minutes, tokens] = resources[profile.name] || [];
      const x = byTime ? minutes : tokens;
      return x === undefined ? [] : [{profile, minutes, tokens, x, score: profile.overall[0]}];
    });
    const frontier = samples.filter(p => !samples.some(q =>
      q.x <= p.x && q.score >= p.score && (q.x < p.x || q.score > p.score)
    )).sort((a, b) => a.x - b.x);
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${xMax} ${yMax}`);
    svg.setAttribute("preserveAspectRatio", "none");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("class", "pareto-frontier");
    const line = document.createElementNS(svg.namespaceURI, "polyline");
    line.setAttribute("points", frontier.map(p => `${p.x},${yMax - p.score}`).join(" "));
    svg.append(line);

    readout.hidden = true;
    plot.replaceChildren(svg, node("span", "scatter-tick tick-top", String(yMax)), node("span", "scatter-tick tick-zero", "0"),
      node("span", "scatter-tick tick-end", String(xMax)));
    if (byTime) {
      const fable = samples.find(sample => labels[sample.profile.name] === "Fable");
      const qwen = samples.find(sample => labels[sample.profile.name] === "Qwen");
      if (fable && qwen) {
        const minute = Math.round((fable.minutes + qwen.minutes) / 2);
        const guide = node("span", "shared-runtime");
        guide.style.left = (minute / xMax * 100) + "%";
        guide.style.height = (Math.max(fable.score, qwen.score) / yMax * 100) + "%";
        guide.append(node("span", "scatter-tick", `≈${minute} min`));
        plot.append(guide);
      }
    }
    for (const sample of samples) {
      const {profile, minutes, tokens, x, score} = sample;
      const point = node("button", "scatter-point");
      const efficient = frontier.includes(sample);
      point.classList.toggle("on-frontier", efficient);
      point.type = "button";
      point.style.left = (x / xMax * 100) + "%";
      point.style.bottom = (score / yMax * 100) + "%";
      const label = profile.name + " · Score " + score.toFixed(2) + " · " + minutes.toFixed(2) + " min/task" +
        (tokens === undefined ? "" : " · " + tokens.toFixed(2) + "M tokens/task") +
        (efficient ? " · Pareto frontier" : "");
      point.setAttribute("aria-label", label);
      const shortName = labels[profile.name];
      if (shortName) {
        const compared = shortName === "Fable" || shortName === "Qwen";
        if (compared) {
          point.classList.add("accent");
          point.style.setProperty("--mark", shortName === "Fable" ? "var(--fable)" : "var(--sol)");
        }
        point.append(node("span", "", compared ? `${shortName} ${score.toFixed(2)}` : shortName));
        point.classList.toggle("label-under", shortName === "Jev");
        point.classList.toggle("label-below", shortName === "Qwen" && byTime);
        point.classList.toggle("label-left", x > .7 * xMax);
      }
      plot.append(point);
    }
    for (const button of controls) button.setAttribute("aria-pressed", String(button.dataset.resource === axis));
  }
  function inspect(event) {
    const point = event.target.closest(".scatter-point");
    if (!point) return;
    readout.textContent = point.getAttribute("aria-label");
    readout.hidden = false;
  }
  plot.addEventListener("pointerover", inspect);
  plot.addEventListener("focusin", inspect);
  plot.addEventListener("click", inspect);
  plot.addEventListener("pointerleave", () => { readout.hidden = true; });
  plot.addEventListener("focusout", () => { readout.hidden = true; });
  for (const button of controls) button.addEventListener("click", () => render(button.dataset.resource));
  render("minutes");
}
setupResourcePlot();

const findings = document.querySelector("#findings");
document.addEventListener("click", event => {
  const open = findings.querySelector("details[open]");
  if (open && !open.contains(event.target)) open.open = false;
  if (!event.target.closest(".scatter-point")) findings.querySelector(".point-readout").hidden = true;
});
findings.addEventListener("keydown", event => {
  if (event.key !== "Escape") return;
  const open = findings.querySelector("details[open]");
  if (open) {
    open.open = false;
    open.querySelector("summary").focus();
  }
  findings.querySelector(".point-readout").hidden = true;
});

function gauge(value, ceiling, style) {
  const track = node("span", "gauge " + style);
  track.setAttribute("aria-hidden", "true");
  const fill = document.createElement("i");
  fill.style.width = (100 * value / ceiling) + "%";
  track.append(fill);
  return track;
}

function metricReading(value, ceiling, index) {
  const reading = node("span", "metric-reading");
  reading.append(gauge(value, ceiling, index ? "success" : "score"),
    node("b", "metric-value", value.toFixed(2) + (index ? "%" : "")));
  return reading;
}

function updateReading(cell, value, ceiling, index) {
  const fill = cell.querySelector(".gauge i");
  if (!fill) {
    cell.replaceChildren(metricReading(value, ceiling, index));
    return;
  }
  const width = (100 * value / ceiling) + "%";
  if (parseFloat(fill.style.width) !== parseFloat(width)) {
    const from = getComputedStyle(fill).width;
    for (const animation of fill.getAnimations()) animation.cancel();
    fill.style.width = width;
    if (!reducedMotion.matches) fill.animate([{width: from}, {width}], moveOptions);
  }
  cell.querySelector(".metric-value").textContent = value.toFixed(2) + (index ? "%" : "");
}

function familyIcon(family) {
  const ns = "http://www.w3.org/2000/svg";
  const icon = document.createElementNS(ns, "svg");
  icon.setAttribute("aria-hidden", "true");
  icon.setAttribute("class", "family-icon " + family.toLowerCase());
  const use = document.createElementNS(ns, "use");
  use.setAttribute("href", "#icon-" + family.toLowerCase());
  icon.append(use);
  return icon;
}

function modelLabel(profile) {
  const name = node("span", "model-label");
  const text = node("span", "model-text");
  text.append(node("b", "model-name", profile.name),
    node("small", "model-meta", profile.family + " · " + profile.tasks + " tasks"));
  name.append(familyIcon(profile.family), text);
  return name;
}

const domainTip = document.querySelector("#domain-tip");
const metricNames = ["Interactive Score", "Physical success"];
let tipAnchor;
function domainValue(profile, index) {
  return profile.dimensions[index][metric].toFixed(2) + (metric === 1 ? "%" : "");
}
function domainRank(profile, index) {
  const score = profile.dimensions[index][metric];
  return 1 + profiles.filter(other => other.dimensions[index][metric] > score).length;
}
function hideDomainTip() {
  domainTip.hidden = true;
  tipAnchor = null;
}
function showDomainTip(button, profile, selected) {
  domainTip.replaceChildren(node("strong", "", profile.name),
    node("small", "", metricNames[metric] + " · Rank among " + profiles.length));
  dimensions.forEach((name, i) => {
    const row = node("span", "tip-domain");
    if (i === selected) row.classList.add("current");
    row.append(node("span", "", name), node("b", "", domainValue(profile, i)),
      node("small", "", "#" + domainRank(profile, i)));
    domainTip.append(row);
  });
  tipAnchor = button;
  domainTip.hidden = false;
  positionDomainTip();
}
function positionDomainTip() {
  if (!tipAnchor) return;
  const anchor = tipAnchor.getBoundingClientRect();
  if (anchor.bottom < 0 || anchor.top > innerHeight) {
    hideDomainTip();
    return;
  }
  const tip = domainTip.getBoundingClientRect();
  const left = anchor.left + anchor.width / 2 - tip.width / 2;
  domainTip.style.left = Math.max(8, Math.min(left, innerWidth - tip.width - 8)) + "px";
  const top = anchor.bottom + tip.height + 8 <= innerHeight ? anchor.bottom + 6 : anchor.top - tip.height - 6;
  domainTip.style.top = Math.max(8, top) + "px";
}
function domainButton(profile, index, expanded) {
  const button = node("button", expanded ? "domain-reading" : "mini-column");
  button.type = "button";
  button.dataset.track = ["seek", "shape", "shift"][Math.floor(index / 2)];
  const value = profile.dimensions[index][metric];
  const action = expanded ? "Return to overall ranking" : "Rank by " + dimensions[index];
  button.setAttribute("aria-label", [dimensions[index], metricNames[metric], domainValue(profile, index),
    "rank " + domainRank(profile, index) + " of " + profiles.length, action].join(", "));
  if (expanded) {
    button.append(metricReading(value, 100, metric));
  } else {
    const fill = document.createElement("i");
    fill.style.height = value + "%";
    button.append(fill);
  }
  button.addEventListener("pointerenter", event => {
    if (event.pointerType !== "touch") showDomainTip(button, profile, index);
  });
  button.addEventListener("pointerleave", hideDomainTip);
  button.addEventListener("focus", () => showDomainTip(button, profile, index));
  button.addEventListener("blur", hideDomainTip);
  button.addEventListener("click", event => {
    event.stopPropagation();
    dimension.value = expanded ? "overall" : dimensions[index];
    renderRanking();
    if (event.detail === 0) dimension.focus({preventScroll: true});
  });
  return button;
}
function miniProfile(profile, dimensionIndex) {
  if (dimensionIndex !== -1) return domainButton(profile, dimensionIndex, true);
  const bars = node("span", "mini-profile");
  dimensions.forEach((_, i) => bars.append(domainButton(profile, i, false)));
  return bars;
}
window.addEventListener("scroll", positionDomainTip, true);
window.addEventListener("resize", positionDomainTip);
document.addEventListener("keydown", event => {
  if (event.key === "Escape") hideDomainTip();
});

for (const profile of profiles) {
  profile.row.dataset.model = profile.name;
  const button = node("button", "model-link");
  button.type = "button";
  button.setAttribute("aria-haspopup", "dialog");
  button.append(modelLabel(profile));
  profile.row.cells[1].replaceChildren(button);
  // The name is keyboard accessible; a pointer anywhere on the row opens the same profile.
  profile.row.addEventListener("click", () => showProfile(profile));
}

function renderLeaders(by = 0) {
  const body = document.querySelector("#leading-systems");
  const positions = rowPositions(body);
  const leaders = [...profiles].sort((a, b) => b.overall[by] - a.overall[by] || b.overall[0] - a.overall[0]).slice(0, 5);
  body.replaceChildren();
  for (const profile of leaders) {
    const row = body.insertRow();
    row.dataset.model = profile.name;
    row.insertCell().textContent = leaders.findIndex(p => p.overall[by] === profile.overall[by]) + 1;
    const heading = node("th", "leader-name");
    heading.scope = "row";
    const button = node("button", "model-link", profile.name);
    button.type = "button";
    button.setAttribute("aria-haspopup", "dialog");
    button.addEventListener("click", () => showProfile(profile));
    heading.append(button, node("small", "model-meta", profile.family + " · " + profile.tasks + " tasks"));
    row.append(heading, node("td", "numeric leader-score", profile.overall[0].toFixed(2)),
      node("td", "numeric", profile.overall[1].toFixed(2) + "%"));
  }
  for (const button of document.querySelectorAll("[data-leader-metric]")) {
    if (Number(button.dataset.leaderMetric) === by) button.parentElement.setAttribute("aria-sort", "descending");
    else button.parentElement.removeAttribute("aria-sort");
  }
  document.querySelector("#leaders-order").textContent = "Ranked by " + metricNames[by];
  moveRows(body, positions);
}
document.querySelectorAll("[data-leader-metric]").forEach(button => button.addEventListener("click", () => {
  renderLeaders(Number(button.dataset.leaderMetric));
}));
renderLeaders();

function renderRanking() {
  hideDomainTip();
  const positions = rowPositions(table);
  const dimensionIndex = dimensions.indexOf(dimension.value);
  const family = document.querySelector('input[name="family"]:checked').value;
  const query = search.value.trim().toLowerCase();
  const values = profile => dimensionIndex === -1 ? profile.overall : profile.dimensions[dimensionIndex];
  const ceiling = Math.max(10, 10 * Math.ceil(Math.max(...profiles.flatMap(values)) / 10));
  const ranked = profiles.filter(profile => !family || profile.family === family)
    .sort((a, b) => values(b)[metric] - values(a)[metric] || a.name.localeCompare(b.name));
  const groups = grouped.checked ? familyOrder.map(name => ranked.filter(p => p.family === name)) : [ranked];
  const fragment = document.createDocumentFragment();
  const visibleProfiles = [];
  for (const group of groups) {
    let previous, rank = 0, groupVisible = 0;
    group.forEach((profile, index) => {
      const scores = values(profile);
      if (scores[metric] !== previous) rank = index + 1;
      previous = scores[metric];
      if (!profile.name.toLowerCase().includes(query)) return;
      if (grouped.checked && groupVisible === 0) {
        const heading = node("tr", "family-heading");
        const cell = document.createElement("th");
        cell.colSpan = 5;
        cell.textContent = profile.family;
        heading.append(cell);
        fragment.append(heading);
      }
      const row = profile.row;
      row.cells[0].textContent = String(rank).padStart(2, "0");
      row.classList.toggle("first-place", rank === 1);
      fragment.append(row);
      visibleProfiles.push(profile);
      groupVisible++;
    });
  }
  table.replaceChildren(fragment);
  for (const profile of visibleProfiles) {
    const row = profile.row;
    const scores = values(profile);
    updateReading(row.cells[2], scores[0], ceiling, 0);
    updateReading(row.cells[4], scores[1], ceiling, 1);
    const domainView = dimensionIndex + "/" + metric;
    if (row.dataset.domainView !== domainView) {
      const changed = row.dataset.domainView !== undefined;
      const cell = row.cells[3];
      cell.replaceChildren(miniProfile(profile, dimensionIndex));
      row.dataset.domainView = domainView;
      if (changed && !reducedMotion.matches) {
        const start = dimensionIndex === -1 ? "scaleY(0)" : "scaleX(0)";
        cell.querySelectorAll("i").forEach(fill => fill.animate([{transform: start}, {transform: "none"}], moveOptions));
      }
    }
  }
  sortButtons.forEach((button, i) => {
    const header = button.closest("th");
    if (i === metric) header.setAttribute("aria-sort", "descending");
    else header.removeAttribute("aria-sort");
  });
  document.querySelector("#empty").hidden = visibleProfiles.length !== 0;
  document.querySelector("#view-label").textContent = grouped.checked ? "Grouped view" : "Ranked view";
  moveRows(table, positions);
}

sortButtons.forEach((button, index) => button.addEventListener("click", () => {
  metric = index;
  renderRanking();
}));
dimension.addEventListener("change", renderRanking);
for (const control of document.querySelectorAll('.controls input:not([type=search])')) {
  control.addEventListener("change", renderRanking);
}
search.addEventListener("input", renderRanking);
renderRanking();

// Reveal each section once when reached, including visits via a section link.
if ("IntersectionObserver" in window) {
  const arrivals = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add("in-view");
        if (entry.target.classList.contains("table-scroll") && !reducedMotion.matches) {
          entry.target.querySelectorAll(".gauge i, .mini-column i").forEach(fill => {
            const start = fill.parentElement.classList.contains("gauge") ? "scaleX(0)" : "scaleY(0)";
            fill.animate([{transform: start}, {transform: "none"}], moveOptions);
          });
        }
        arrivals.unobserve(entry.target);
      }
    }
  }, {threshold: .08});
  document.querySelectorAll(".section-head, .track-card, .harness-grid, .code-panel, .paper-figure, .table-scroll")
    .forEach(element => arrivals.observe(element));
}
reducedMotion.addEventListener("change", event => {
  if (event.matches) document.getAnimations().forEach(animation => animation.cancel());
});

for (const link of document.querySelectorAll(".tracks a")) {
  const track = document.querySelector(link.hash);
  const preview = node("div", "track-preview");
  preview.id = track.id + "-preview";
  preview.setAttribute("role", "tooltip");
  preview.append(node("strong", "", track.querySelector("small").textContent),
    track.querySelector("ul").cloneNode(true));
  link.setAttribute("aria-label", link.textContent);
  link.setAttribute("aria-describedby", preview.id);
  link.append(preview);
  link.addEventListener("keydown", event => {
    if (event.key === "Escape") link.classList.add("dismissed");
  });
  link.addEventListener("pointerleave", () => link.classList.remove("dismissed"));
  link.addEventListener("blur", () => link.classList.remove("dismissed"));
}

document.querySelector("#download").addEventListener("click", () => {
  function metrics([score, success]) {
    return {interactive_score: score, physical_success_percent: success};
  }
  const systems = profiles.map(profile => ({
    system: profile.name, family: profile.family, tasks: profile.tasks, overall: metrics(profile.overall),
    dimensions: Object.fromEntries(dimensions.map((label, index) => [label, metrics(profile.dimensions[index])]))
  }));
  const data = {source: "WorldLeague: Benchmarking Physical Agency, Physical Agency Profiles table", seed: 0, systems};
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: "application/json"}));
  const link = document.createElement("a");
  link.href = url;
  link.download = "worldleague-profiles-seed0.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

document.querySelectorAll("[data-copy]").forEach(button => {
  const code = document.getElementById(button.dataset.copy);
  button.setAttribute("aria-label", "Copy " + button.closest(".code-panel").querySelector("h3").textContent);
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(code.textContent);
      button.textContent = "Copied";
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = "Text selected";
    }
    setTimeout(() => { button.textContent = "Copy"; }, 1800);
  });
});

const siteHeader = document.querySelector(".site-header");
const navLinks = [...document.querySelectorAll("#section-nav a")];
const sections = navLinks.map(link => document.querySelector(link.hash));
function markCurrentSection() {
  siteHeader.classList.toggle("is-scrolled", window.scrollY > 1);
  let current = null;
  for (const section of sections) {
    if (section.getBoundingClientRect().top <= 115) current = section;
  }
  if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) {
    current = sections.at(-1);
  }
  for (const link of navLinks) {
    if (current && link.hash === "#" + current.id) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  }
}
window.addEventListener("scroll", markCurrentSection, {passive: true});
markCurrentSection();
