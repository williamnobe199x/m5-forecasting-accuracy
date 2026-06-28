
const data = JSON.parse(document.getElementById("dashboard-data").textContent);
const colors = {
  actual: "#1f2937", snaive: "#b7791f", base: "#2563eb", model: "#2563eb",
  trend: "#167a78", optimized: "#2f855a", coral: "#d55c45", indigo: "#4f46e5"
};
const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 });
const pct = (v) => `${(Number(v) * 100).toFixed(2)}%`;
const byId = (id) => document.getElementById(id);

function setText(id, value) {
  const node = byId(id);
  if (node) node.textContent = value;
}

function numericExtent(rows, keys) {
  const vals = [];
  rows.forEach((row) => keys.forEach((key) => {
    const value = Number(row[key]);
    if (Number.isFinite(value)) vals.push(value);
  }));
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const pad = (max - min || 1) * 0.06;
  return [Math.max(0, min - pad), max + pad];
}

function lineChart(target, rows, series, options = {}) {
  const el = byId(target);
  if (!el || rows.length === 0) return;
  const width = 900, height = options.height || 280;
  const pad = { left: 58, right: 18, top: 18, bottom: 38 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const keys = series.map((s) => s.key);
  const [yMin, yMax] = numericExtent(rows, keys);
  const x = (idx) => pad.left + (rows.length === 1 ? 0 : idx / (rows.length - 1)) * plotW;
  const y = (value) => pad.top + (yMax - value) / (yMax - yMin) * plotH;
  const grid = [0, .25, .5, .75, 1].map((frac) => {
    const yy = pad.top + frac * plotH;
    const val = yMax - frac * (yMax - yMin);
    return `<line x1="${pad.left}" y1="${yy}" x2="${width - pad.right}" y2="${yy}" class="grid-line"/>`
      + `<text x="8" y="${yy + 4}" class="axis">${fmt.format(val)}</text>`;
  }).join("");
  const lines = series.map((s) => {
    const points = rows.map((row, idx) => `${x(idx).toFixed(1)},${y(Number(row[s.key])).toFixed(1)}`).join(" ");
    return `<polyline points="${points}" class="line" stroke="${s.color}"><title>${s.label}</title></polyline>`;
  }).join("");
  const labels = [0, Math.floor((rows.length - 1) / 2), rows.length - 1].map((idx) => {
    const label = options.xLabel ? options.xLabel(rows[idx]) : rows[idx].day;
    return `<text x="${x(idx) - 16}" y="${height - 12}" class="axis">${label}</text>`;
  }).join("");
  el.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${options.label || "line chart"}">${grid}${lines}${labels}</svg>`;
}

function legend(target, series) {
  const el = byId(target);
  if (!el) return;
  el.innerHTML = series.map((s) => `<span><i style="background:${s.color}"></i>${s.label}</span>`).join("");
}

function horizontalBars(target, rows, valueKey, labelKey, options = {}) {
  const el = byId(target);
  if (!el) return;
  const values = rows.map((row) => Number(row[valueKey]));
  const max = Math.max(...values.map((v) => Math.abs(v)), 1);
  el.innerHTML = rows.map((row) => {
    const value = Number(row[valueKey]);
    const width = Math.max(2, Math.abs(value) / max * 100);
    const color = options.color || (value < 0 ? colors.coral : colors.teal);
    const label = row[labelKey];
    const display = options.format ? options.format(value, row) : fmt.format(value);
    return `<div class="bar-row"><div class="bar-label">${label}</div><div class="bar-track"><div class="bar-fill" style="width:${width}%;background:${color}"></div></div><div class="bar-value">${display}</div></div>`;
  }).join("");
}

function renderWrmsse() {
  const rows = data.levels.filter((row) => row.level !== "Average");
  const max = Math.max(...rows.flatMap((row) => [row.wrmsse_snaive, row.wrmsse_model, row.wrmsse_optimized].map(Number)));
  byId("wrmsse-bars").innerHTML = rows.map((row) => {
    const clean = String(row.level).replaceAll("_", " ");
    const entries = [
      ["sNaive", row.wrmsse_snaive, colors.snaive],
      ["LightGBM", row.wrmsse_model, colors.model],
      ["Optimized", row.wrmsse_optimized, colors.optimized],
    ].map(([label, value, color]) => {
      const width = Math.max(2, Number(value) / max * 100);
      return `<div class="mini-bar"><span>${label}</span><span class="mini-track"><span class="mini-fill" style="width:${width}%;background:${color}"></span></span><span>${Number(value).toFixed(3)}</span></div>`;
    }).join("");
    return `<div class="wrmsse-row"><strong>${clean}</strong><div class="mini-bars">${entries}</div></div>`;
  }).join("");
}

function renderHeatmap() {
  const rows = data.stateCategorySales;
  const states = [...new Set(rows.map((r) => r.state_id))];
  const cats = [...new Set(rows.map((r) => r.cat_id))];
  const max = Math.max(...rows.map((r) => Number(r.validation_sales)));
  byId("state-category-heatmap").innerHTML = [
    `<div class="heat-row"><strong></strong>${cats.map((c) => `<strong>${c}</strong>`).join("")}</div>`,
    ...states.map((state) => {
      const cells = cats.map((cat) => {
        const row = rows.find((r) => r.state_id === state && r.cat_id === cat);
        const value = row ? Number(row.validation_sales) : 0;
        const alpha = 0.16 + value / max * 0.72;
        return `<div class="heat-cell" style="background:rgba(22,122,120,${alpha})"><strong>${fmt.format(value)}</strong><small>${pct(row ? row.share : 0)}</small></div>`;
      }).join("");
      return `<div class="heat-row"><strong>${state}</strong>${cells}</div>`;
    })
  ].join("");
}

function renderExampleSelector() {
  const select = byId("series-select");
  const ids = [...new Set(data.examples.map((row) => row.id))];
  select.innerHTML = ids.map((id) => `<option value="${id}">${id}</option>`).join("");
  const draw = () => {
    const rows = data.examples.filter((row) => row.id === select.value);
    const series = [
      { key: "actual", label: "Actual", color: colors.actual },
      { key: "base", label: "Base", color: colors.base },
      { key: "optimized", label: "Optimized", color: colors.optimized },
    ];
    legend("example-legend", series);
    lineChart("example-chart", rows, series, { height: 250, label: "example series forecast" });
  };
  select.addEventListener("change", draw);
  draw();
}

function renderFeatureGallery() {
  const el = byId("feature-gallery");
  if (el.dataset.rendered === "1") return;
  const rows = data.featureImages.slice(0, 12);
  el.innerHTML = rows.map((img) => `<figure><img loading="lazy" src="assets/${img.file}" alt="${img.store_id} ${img.kind} feature importance"><figcaption>${img.store_id} · ${img.kind}</figcaption></figure>`).join("");
  el.dataset.rendered = "1";
}

function renderTables() {
  const topStores = [...data.storeSales].sort((a, b) => Number(b.validation_sales) - Number(a.validation_sales)).slice(0, 10);
  byId("store-table").innerHTML = `<tr><th>Store</th><th>State</th><th>Validation sales</th><th>Share</th></tr>`
    + topStores.map((r) => `<tr><td>${r.store_id}</td><td>${r.state_id}</td><td>${fmt.format(r.validation_sales)}</td><td>${pct(r.share)}</td></tr>`).join("");
  byId("event-table").innerHTML = `<tr><th>Event group</th><th>Days</th><th>Avg sales</th><th>Median sales</th></tr>`
    + data.eventSummary.map((r) => `<tr><td>${r.event_group}</td><td>${r.days}</td><td>${fmt.format(r.avg_sales)}</td><td>${fmt.format(r.median_sales)}</td></tr>`).join("");
}

function init() {
  setText("kpi-wrmsse", Number(data.summary.optimized_avg_wrmsse).toFixed(3));
  setText("kpi-gain", pct(data.summary.total_gain_pct));
  setText("kpi-bias", pct(data.summary.optimized_total_bias_pct));
  setText("kpi-series", fmt.format(data.visualSummary.bottom_series));
  setText("best-candidate", data.optimizationSummary.best_candidate || "trend_global");
  setText("best-improvement", pct(data.optimizationSummary.improvement_pct || 0));

  const forecastSeries = [
    { key: "actual", label: "Actual", color: colors.actual },
    { key: "model", label: "Base LightGBM", color: colors.base },
    { key: "optimized", label: "Optimized", color: colors.optimized },
  ];
  legend("forecast-legend", forecastSeries);
  lineChart("forecast-chart", data.daily, forecastSeries, { height: 300, label: "validation forecast totals" });

  const edaSeries = [
    { key: "sales", label: "Daily sales", color: colors.indigo },
    { key: "roll_28", label: "28-day rolling mean", color: colors.teal },
  ];
  legend("eda-legend", edaSeries);
  lineChart("eda-sales-chart", data.edaDaily, edaSeries, { height: 310, label: "M5 daily sales", xLabel: (row) => row.date });

  renderWrmsse();
  renderHeatmap();
  renderTables();
  renderExampleSelector();

  const candidateRows = [...data.candidates].slice(0, 10);
  horizontalBars("candidate-bars", candidateRows, "avg_wrmsse", "candidate", { color: colors.teal, format: (v) => Number(v).toFixed(3) });
  const storeBias = [...data.storeBias].sort((a, b) => Math.abs(Number(b.optimized_bias_pct)) - Math.abs(Number(a.optimized_bias_pct))).slice(0, 10)
    .map((r) => ({ ...r, label: `${r.store_id} (${r.state_id})` }));
  horizontalBars("store-bias-bars", storeBias, "optimized_bias_pct", "label", { format: pct });
  horizontalBars("price-bars", data.priceSummary, "avg_price", "store_id", { color: colors.amber, format: (v) => `$${Number(v).toFixed(2)}` });

  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
      button.classList.add("active");
      byId(button.dataset.view).classList.add("active");
      if (button.dataset.view === "model") renderFeatureGallery();
    });
  });
}
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
