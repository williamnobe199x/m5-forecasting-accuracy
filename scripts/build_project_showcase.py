"""Build an interactive HTML dashboard for the M5 project."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="output/project_summary.json", type=Path)
    parser.add_argument("--optimization-summary", default="output/optimization_summary.json", type=Path)
    parser.add_argument("--levels", default="output/level_contribution.csv", type=Path)
    parser.add_argument("--daily", default="output/daily_totals.csv", type=Path)
    parser.add_argument("--candidates", default="output/optimization_candidates.csv", type=Path)
    parser.add_argument("--visual-dir", default="output/visual", type=Path)
    parser.add_argument("--out", default="docs/m5_project_showcase.html", type=Path)
    parser.add_argument("--index-out", default="docs/index.html", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    frame = frame.where(pd.notna(frame), None)
    return frame.to_dict(orient="records")


def load_dashboard_data(args: argparse.Namespace) -> dict:
    visual = args.visual_dir
    data = {
        "summary": read_json(args.summary),
        "optimizationSummary": read_json(args.optimization_summary),
        "visualSummary": read_json(visual / "visual_summary.json"),
        "levels": records(args.levels),
        "daily": records(args.daily),
        "candidates": records(args.candidates),
        "edaDaily": records(visual / "eda_daily_sales.csv"),
        "storeSales": records(visual / "eda_store_sales.csv"),
        "categorySales": records(visual / "eda_category_sales.csv"),
        "stateCategorySales": records(visual / "eda_state_category_sales.csv"),
        "eventSummary": records(visual / "eda_event_summary.csv"),
        "priceSummary": records(visual / "eda_price_summary.csv"),
        "storeBias": records(visual / "forecast_store_bias.csv"),
        "categoryBias": records(visual / "forecast_category_bias.csv"),
        "examples": records(visual / "forecast_example_series.csv"),
        "featureImages": records(visual / "feature_importance_inventory.csv"),
    }
    return data


def compact_num(value: float) -> str:
    return f"{value:,.0f}"


def pct_text(value: float) -> str:
    return f"{value * 100:.2f}%"


def static_line_svg(rows: list[dict], series: list[dict], height: int = 280) -> str:
    if not rows:
        return ""
    width = 900
    pad_l, pad_r, pad_t, pad_b = 58, 18, 18, 38
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    values = [float(row[item["key"]]) for row in rows for item in series if row.get(item["key"]) is not None]
    ymin, ymax = min(values), max(values)
    pad = (ymax - ymin or 1) * 0.06
    ymin = max(0, ymin - pad)
    ymax += pad

    def x(idx: int) -> float:
        return pad_l + (idx / max(len(rows) - 1, 1)) * plot_w

    def y(value: float) -> float:
        return pad_t + (ymax - value) / (ymax - ymin) * plot_h

    grid = []
    for frac in [0, 0.25, 0.5, 0.75, 1]:
        yy = pad_t + frac * plot_h
        value = ymax - frac * (ymax - ymin)
        grid.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width-pad_r}" y2="{yy:.1f}" class="grid-line"/>')
        grid.append(f'<text x="8" y="{yy+4:.1f}" class="axis">{compact_num(value)}</text>')

    polylines = []
    for item in series:
        points = " ".join(
            f"{x(idx):.1f},{y(float(row[item['key']])):.1f}"
            for idx, row in enumerate(rows)
            if row.get(item["key"]) is not None
        )
        polylines.append(f'<polyline points="{points}" class="line" stroke="{item["color"]}"/>')

    labels = []
    for idx in [0, len(rows) // 2, len(rows) - 1]:
        label = rows[idx].get("date") or rows[idx].get("day") or idx + 1
        labels.append(f'<text x="{x(idx)-16:.1f}" y="{height-12}" class="axis">{html.escape(str(label))}</text>')

    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="static line chart">{"".join(grid + polylines + labels)}</svg>'


def write_html(data: dict, out: Path, index_out: Path) -> None:
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    summary = data["summary"]
    visual = data["visualSummary"]
    opt = data["optimizationSummary"]
    title = "M5 Forecasting Accuracy 项目复盘与优化"
    best_candidate = html.escape(str(opt.get("best_candidate", "trend_global")))
    kpi_wrmsse = f"{float(summary.get('optimized_avg_wrmsse', 0)):.3f}"
    kpi_gain = pct_text(float(summary.get("total_gain_pct", 0)))
    kpi_bias = pct_text(float(summary.get("optimized_total_bias_pct", 0)))
    kpi_series = compact_num(float(visual.get("bottom_series", 0)))
    best_improvement = pct_text(float(opt.get("improvement_pct", 0)))
    forecast_svg = static_line_svg(
        data["daily"],
        [
            {"key": "actual", "color": "#1f2937"},
            {"key": "model", "color": "#2563eb"},
            {"key": "optimized", "color": "#2f855a"},
        ],
        height=300,
    )
    eda_svg = static_line_svg(
        data["edaDaily"],
        [
            {"key": "sales", "color": "#4f46e5"},
            {"key": "roll_28", "color": "#167a78"},
        ],
        height=310,
    )

    css = """
:root {
  --ink: #1f2937;
  --muted: #667085;
  --paper: #f6f7f9;
  --panel: #ffffff;
  --line: #d8dee8;
  --teal: #167a78;
  --coral: #d55c45;
  --indigo: #4f46e5;
  --amber: #b7791f;
  --green: #2f855a;
  --blue: #2563eb;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Inter, "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
  color: var(--ink);
  background: var(--paper);
}
a { color: var(--indigo); }
.shell { max-width: 1240px; margin: 0 auto; padding: 0 24px; }
header {
  padding: 26px 0 18px;
  background: #ffffff;
  border-bottom: 1px solid var(--line);
  position: sticky;
  top: 0;
  z-index: 5;
}
.topbar { display: flex; align-items: center; justify-content: space-between; gap: 18px; }
h1 { font-size: 24px; line-height: 1.15; margin: 0; letter-spacing: 0; }
.subtitle { color: var(--muted); margin-top: 6px; font-size: 14px; }
.tabs { display: flex; gap: 6px; flex-wrap: wrap; }
.tab {
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink);
  border-radius: 6px;
  padding: 8px 11px;
  font-weight: 700;
  cursor: pointer;
}
.tab.active { background: var(--ink); color: #fff; border-color: var(--ink); }
main { padding: 24px 0 44px; }
.view { display: none; }
.view.active { display: block; }
.grid { display: grid; gap: 16px; }
.grid.kpi { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.grid.two { grid-template-columns: 1.25fr .75fr; }
.grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 18px;
}
.panel h2, .panel h3 { margin: 0 0 12px; letter-spacing: 0; }
.panel h2 { font-size: 22px; }
.panel h3 { font-size: 16px; }
.kpi-box strong { display: block; font-size: 30px; line-height: 1; margin-bottom: 7px; }
.kpi-box span { color: var(--muted); font-size: 13px; }
.claim {
  display: inline-block;
  font-size: 11px;
  font-weight: 800;
  color: #0f4f4d;
  background: #d9f0ec;
  padding: 3px 6px;
  border-radius: 4px;
  margin-right: 6px;
}
p { line-height: 1.62; color: #344054; margin: 0 0 12px; }
.chart { min-height: 280px; }
svg { width: 100%; height: auto; display: block; }
.axis { fill: var(--muted); font-size: 12px; }
.grid-line { stroke: #e8edf3; stroke-width: 1; }
.line { fill: none; stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; }
.legend { display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 12px; color: var(--muted); font-size: 13px; }
.legend i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.bars { display: grid; gap: 9px; }
.bar-row { display: grid; grid-template-columns: 170px 1fr 72px; gap: 10px; align-items: center; }
.bar-label { font-weight: 700; font-size: 13px; }
.bar-track { height: 13px; background: #eef2f6; border-radius: 4px; overflow: hidden; }
.bar-fill { height: 100%; background: var(--teal); border-radius: 4px; }
.bar-value { text-align: right; color: var(--muted); font-size: 12px; }
.wrmsse-row { display: grid; grid-template-columns: 170px 1fr; gap: 12px; align-items: center; padding: 8px 0; border-bottom: 1px solid #edf1f5; }
.mini-bars { display: grid; gap: 4px; }
.mini-bar { display: grid; grid-template-columns: 72px 1fr 54px; gap: 8px; align-items: center; font-size: 12px; color: var(--muted); }
.mini-track { height: 8px; background: #eef2f6; border-radius: 3px; overflow: hidden; }
.mini-fill { height: 100%; border-radius: 3px; }
.flow { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; }
.step { border-top: 5px solid var(--amber); background: #fff; border-radius: 8px; padding: 14px; min-height: 132px; }
.step:nth-child(2) { border-color: var(--teal); }
.step:nth-child(3) { border-color: var(--indigo); }
.step:nth-child(4) { border-color: var(--coral); }
.step:nth-child(5) { border-color: var(--green); }
.select {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px 10px;
  background: #fff;
  color: var(--ink);
  max-width: 100%;
}
.heatmap { display: grid; gap: 6px; }
.heat-row { display: grid; grid-template-columns: 72px repeat(3, 1fr); gap: 6px; align-items: stretch; }
.heat-cell { border-radius: 6px; padding: 10px; color: #111827; min-height: 58px; border: 1px solid rgba(31,41,55,.08); }
.heat-cell small { display: block; color: #344054; margin-top: 4px; }
.image-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
figure { margin: 0; border: 1px solid var(--line); border-radius: 8px; background: #fff; overflow: hidden; }
figure img { width: 100%; display: block; }
figcaption { padding: 8px 10px; color: var(--muted); font-size: 12px; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; border-bottom: 1px solid #edf1f5; padding: 9px 8px; font-size: 13px; }
th { color: #344054; background: #f2f5f8; }
.note { color: var(--muted); font-size: 13px; }
footer { border-top: 1px solid var(--line); padding: 22px 0; color: var(--muted); background: #fff; }
@media (max-width: 960px) {
  .topbar, .grid.kpi, .grid.two, .grid.three, .flow, .image-grid { grid-template-columns: 1fr; display: grid; }
  header { position: static; }
  .bar-row, .wrmsse-row { grid-template-columns: 1fr; }
  .heat-row { grid-template-columns: 1fr; }
}
"""

    js = """
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
  el.innerHTML = rows.map((img) => `<figure><img loading="lazy" src="../${img.file}" alt="${img.store_id} ${img.kind} feature importance"><figcaption>${img.store_id} · ${img.kind}</figcaption></figure>`).join("");
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
"""

    body = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
  <script type="application/json" id="dashboard-data">{data_json}</script>
  <header>
    <div class="shell topbar">
      <div>
        <h1>{html.escape(title)}</h1>
        <div class="subtitle">从数据探索、复现评分、候选优化到结果展示的一体化 M5 Accuracy 项目页</div>
      </div>
      <nav class="tabs" aria-label="dashboard views">
        <button class="tab active" data-view="overview">总览</button>
        <button class="tab" data-view="eda">EDA</button>
        <button class="tab" data-view="model">流程</button>
        <button class="tab" data-view="results">结果</button>
        <button class="tab" data-view="roadmap">优化路线</button>
      </nav>
    </div>
  </header>
  <main class="shell">
    <section id="overview" class="view active">
      <div class="grid kpi">
        <div class="panel kpi-box"><strong id="kpi-wrmsse">{kpi_wrmsse}</strong><span>优化后 Avg WRMSSE</span></div>
        <div class="panel kpi-box"><strong id="kpi-gain">{kpi_gain}</strong><span>相对 sNaive 提升</span></div>
        <div class="panel kpi-box"><strong id="kpi-bias">{kpi_bias}</strong><span>28 天总销量偏差</span></div>
        <div class="panel kpi-box"><strong id="kpi-series">{kpi_series}</strong><span>底层商品-门店序列</span></div>
      </div>
      <div class="grid two" style="margin-top:16px">
        <div class="panel">
          <h2>验证窗口总销量：实际 vs 预测</h2>
          <div class="legend" id="forecast-legend"></div>
          <div class="chart" id="forecast-chart">{forecast_svg}</div>
          <p><span class="claim">[COMPUTED, HIGH]</span>最佳候选为 <strong id="best-candidate">{best_candidate}</strong>，相对原始 LightGBM 的本地验证提升为 <strong id="best-improvement">{best_improvement}</strong>。</p>
        </div>
        <div class="panel">
          <h2>项目判断</h2>
          <p><span class="claim">[INFERRED, HIGH]</span>当前最稳的优化不是继续细分 multiplier，而是把训练体系升级到 rolling CV、direct horizon 和聚合 lag 特征。</p>
          <p><span class="claim">[COMPUTED, HIGH]</span>本地候选扫描显示，分组趋势与 horizon 趋势校准没有超过全局趋势版本。</p>
          <p class="note">本页参考 Kaggle Heads or Tails 的交互式 M5 EDA 组织方式：先让数据结构可见，再解释模型、误差和优化路径。</p>
        </div>
      </div>
    </section>

    <section id="eda" class="view">
      <div class="grid two">
        <div class="panel">
          <h2>全量日销量走势</h2>
          <div class="legend" id="eda-legend"></div>
          <div class="chart" id="eda-sales-chart">{eda_svg}</div>
        </div>
        <div class="panel">
          <h2>州 × 品类验证期销售热力</h2>
          <div class="heatmap" id="state-category-heatmap"></div>
        </div>
      </div>
      <div class="grid three" style="margin-top:16px">
        <div class="panel"><h3>Top Store Mix</h3><table id="store-table"></table></div>
        <div class="panel"><h3>Event Sales Summary</h3><table id="event-table"></table></div>
        <div class="panel"><h3>Store Price Level</h3><div class="bars" id="price-bars"></div></div>
      </div>
    </section>

    <section id="model" class="view">
      <div class="panel">
        <h2>全流程逻辑</h2>
        <div class="flow">
          <div class="step"><h3>1. 原始数据</h3><p>sales、calendar、sell prices 三类表构成商品、门店、日期和价格上下文。</p></div>
          <div class="step"><h3>2. 长表特征</h3><p>把宽表销量转成 id-day 训练样本，并接入价格、事件、SNAP 和日期特征。</p></div>
          <div class="step"><h3>3. LightGBM</h3><p>按门店训练 10 个模型，用 Tweedie 目标处理非负、稀疏销量。</p></div>
          <div class="step"><h3>4. 递归预测</h3><p>滚动生成 28 天预测，并用本地验证窗口复现结果。</p></div>
          <div class="step"><h3>5. 层级评分</h3><p>聚合到 12 层，使用 WRMSSE 评价业务层级误差。</p></div>
        </div>
      </div>
      <div class="panel" style="margin-top:16px">
        <h2>特征重要性图库</h2>
        <div class="image-grid" id="feature-gallery"></div>
      </div>
    </section>

    <section id="results" class="view">
      <div class="grid two">
        <div class="panel">
          <h2>12 层 WRMSSE 对比</h2>
          <div id="wrmsse-bars"></div>
        </div>
        <div class="panel">
          <h2>优化候选排名</h2>
          <div class="bars" id="candidate-bars"></div>
          <p class="note">数值越小越好；候选选择基于单个本地验证窗口，不能直接视为盲测表现。</p>
        </div>
      </div>
      <div class="grid two" style="margin-top:16px">
        <div class="panel">
          <h2>门店级预测偏差</h2>
          <div class="bars" id="store-bias-bars"></div>
        </div>
        <div class="panel">
          <h2>高销量序列示例</h2>
          <select id="series-select" class="select" aria-label="example series"></select>
          <div class="legend" id="example-legend" style="margin-top:12px"></div>
          <div class="chart" id="example-chart"></div>
        </div>
      </div>
    </section>

    <section id="roadmap" class="view">
      <div class="grid three">
        <div class="panel"><h2>短期 P0</h2><p><span class="claim">[INFERRED, HIGH]</span>建立 4 折 rolling CV，把所有 multiplier、blend 和模型选择放到同一验证框架里。</p></div>
        <div class="panel"><h2>中期 P1</h2><p><span class="claim">[INFERRED, HIGH]</span>补充 item-store、item、dept-store 聚合 lag/rolling 特征，降低底层噪声。</p></div>
        <div class="panel"><h2>中期 P2</h2><p><span class="claim">[INFERRED, HIGH]</span>增加 4 段 direct horizon LightGBM，与当前 recursive 输出做 CV ensemble。</p></div>
        <div class="panel"><h2>长期 P3</h2><p><span class="claim">[INFERRED, MED]</span>尝试层级 reconciliation，让 L1-L12 的预测更一致。</p></div>
        <div class="panel"><h2>长期 P4</h2><p><span class="claim">[INFERRED, MED]</span>抽样评估 Chronos、TimesFM、Moirai 等 foundation model，稳定增益后再入 ensemble。</p></div>
        <div class="panel"><h2>发布形态</h2><p><span class="claim">[COMPUTED, HIGH]</span>当前产物已整理为脚本、输出数据、报告文档和 GitHub Pages 入口。</p></div>
      </div>
    </section>
  </main>
  <footer>
    <div class="shell">Local validation: d_1914..d_1941. Reference: <a href="https://www.kaggle.com/code/headsortails/back-to-predict-the-future-interactive-m5-eda">Kaggle interactive M5 EDA</a>.</div>
  </footer>
  <script src="dashboard.js" defer></script>
</body>
</html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    js_out = out.with_name("dashboard.js")
    js_out.write_text(js, encoding="utf-8")
    out.write_text(body, encoding="utf-8")
    if index_out:
        index_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, index_out)
        if index_out.with_name("dashboard.js") != js_out:
            shutil.copyfile(js_out, index_out.with_name("dashboard.js"))


def main() -> None:
    args = parse_args()
    data = load_dashboard_data(args)
    write_html(data, args.out, args.index_out)
    print(f"wrote {args.out}")
    print(f"wrote {args.index_out}")
    print(f"wrote {args.out.with_name('dashboard.js')}")


if __name__ == "__main__":
    main()
