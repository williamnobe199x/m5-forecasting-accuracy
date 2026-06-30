"""Build a presentation-style report page and matching speaker script."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="output/project_summary.json", type=Path)
    parser.add_argument("--optimization-summary", default="output/optimization_summary.json", type=Path)
    parser.add_argument("--levels", default="output/level_contribution.csv", type=Path)
    parser.add_argument("--candidates", default="output/optimization_candidates.csv", type=Path)
    parser.add_argument("--daily", default="output/daily_totals.csv", type=Path)
    parser.add_argument("--visual-dir", default="output/visual", type=Path)
    parser.add_argument("--job-dir", default="output/job_alignment", type=Path)
    parser.add_argument("--direct-score", default="output/direct_segment_lightgbm_wrmsse_by_level.csv", type=Path)
    parser.add_argument("--html-out", default="docs/report_presentation.html", type=Path)
    parser.add_argument("--script-out", default="docs/presentation_script.md", type=Path)
    parser.add_argument("--index-out", default="docs/index.html", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_json_optional(path: Path) -> dict:
    if not path.exists():
        return {}
    return read_json(path)


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_csv(path)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def num(value: float, digits: int = 3) -> str:
    return f"{value:,.{digits}f}"


def int_num(value: float) -> str:
    return f"{value:,.0f}"


def line_svg(rows: pd.DataFrame, series: list[tuple[str, str, str]], height: int = 260) -> str:
    width = 900
    pad_l, pad_r, pad_t, pad_b = 64, 22, 24, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    values = []
    for col, _, _ in series:
        values.extend(rows[col].astype(float).tolist())
    ymin = max(0, min(values) * 0.94)
    ymax = max(values) * 1.06

    def x(idx: int) -> float:
        return pad_l + idx / max(len(rows) - 1, 1) * plot_w

    def y(value: float) -> float:
        return pad_t + (ymax - value) / max(ymax - ymin, 1) * plot_h

    grid = []
    for frac in [0, 0.25, 0.5, 0.75, 1]:
        yy = pad_t + frac * plot_h
        label = ymax - frac * (ymax - ymin)
        grid.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width-pad_r}" y2="{yy:.1f}" class="gridline"/>')
        grid.append(f'<text x="10" y="{yy+4:.1f}" class="axis">{int_num(label)}</text>')
    for idx in [0, len(rows) // 2, len(rows) - 1]:
        label = rows.iloc[idx].get("date", rows.iloc[idx].get("day", idx + 1))
        grid.append(f'<text x="{x(idx)-18:.1f}" y="{height-12}" class="axis">{html.escape(str(label))}</text>')

    lines = []
    for col, label, color in series:
        points = " ".join(f"{x(idx):.1f},{y(float(value)):.1f}" for idx, value in enumerate(rows[col]))
        lines.append(f'<polyline points="{points}" class="line" stroke="{color}"><title>{html.escape(label)}</title></polyline>')
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="line chart">{"".join(grid + lines)}</svg>'


def bars_svg(rows: pd.DataFrame, label_col: str, value_cols: list[tuple[str, str, str]], height_per_row: int = 54) -> str:
    rows = rows.copy().reset_index(drop=True)
    width = 920
    height = max(220, 42 + len(rows) * height_per_row)
    pad_l, pad_r, pad_t = 168, 84, 26
    plot_w = width - pad_l - pad_r
    max_value = max(float(rows[col].max()) for col, _, _ in value_cols)
    parts = []
    for row_idx, row in rows.iterrows():
        y0 = pad_t + row_idx * height_per_row
        parts.append(f'<text x="6" y="{y0+23}" class="barlabel">{html.escape(str(row[label_col]).replace("_", " "))}</text>')
        for offset, (col, label, color) in enumerate(value_cols):
            value = float(row[col])
            width_value = value / max_value * plot_w
            yy = y0 + 4 + offset * 15
            parts.append(f'<rect x="{pad_l}" y="{yy}" width="{width_value:.1f}" height="10" rx="3" fill="{color}"/>')
            parts.append(f'<text x="{pad_l + width_value + 8:.1f}" y="{yy+9}" class="axis">{value:.3f}</text>')
            if row_idx == 0:
                parts.append(f'<text x="{pad_l - 78}" y="{yy+9}" class="axis">{html.escape(label)}</text>')
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="bar chart">{"".join(parts)}</svg>'


def single_bars(rows: pd.DataFrame, label_col: str, value_col: str, color: str, limit: int = 8) -> str:
    frame = rows.head(limit).copy().reset_index(drop=True)
    width = 880
    height = 36 + len(frame) * 32
    pad_l, pad_r, pad_t = 230, 80, 22
    plot_w = width - pad_l - pad_r
    max_value = float(frame[value_col].max())
    parts = []
    for idx, row in frame.iterrows():
        yy = pad_t + idx * 32
        value = float(row[value_col])
        bar_w = value / max(max_value, 1e-9) * plot_w
        parts.append(f'<text x="6" y="{yy+14}" class="barlabel">{html.escape(str(row[label_col]))}</text>')
        parts.append(f'<rect x="{pad_l}" y="{yy+2}" width="{bar_w:.1f}" height="15" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{pad_l+bar_w+8:.1f}" y="{yy+15}" class="axis">{value:.3f}</text>')
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="candidate bars">{"".join(parts)}</svg>'


def code_block(text: str) -> str:
    return f"<pre><code>{html.escape(text.strip())}</code></pre>"


def copy_report_assets(docs_dir: Path) -> None:
    assets_dir = docs_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    asset_map = {
        "M5 forecasting.png": "m5-forecasting.png",
        "importance_CA_1.png": "importance_CA_1.png",
        "importance_split_CA_1.png": "importance_split_CA_1.png",
    }
    for source_name, target_name in asset_map.items():
        source = ROOT / source_name
        if source.exists():
            shutil.copyfile(source, assets_dir / target_name)


def script_code() -> dict[str, str]:
    return {
        "pipeline": """
python scripts/reproduce_archive_prediction.py
python scripts/score_m5_wrmsse.py --predictions output/archive_validation_predictions.csv
python scripts/optimize_forecast_candidates.py
python scripts/run_full_pipeline.py
""",
        "trend": """
raw = (sum(d_1886..d_1913) + smooth) / (sum(d_1858..d_1885) + smooth)
multiplier = clip(raw, 0.85, 1.15)
forecast[F1..F28] = forecast[F1..F28] * multiplier
""",
        "wrmsse": """
rmse = sqrt(mean((actual - pred) ** 2, axis=1))
rmsse = rmse / sqrt(mean(diff(active_history) ** 2))
wrmsse = sum(revenue_weight * rmsse)
final_score = mean(level_1_wrmsse ... level_12_wrmsse)
""",
        "candidate": """
candidates = {
    "base_recursive_lightgbm": base,
    "trend_global": apply_global_multiplier(base),
    "trend_store": apply_group_multiplier(base, ["store_id"]),
    "trend_horizon_global": apply_horizon_multiplier(base),
    "blend_base_global_75": blend(base, trend_global, 0.75),
}
best = min(candidates, key=local_wrmsse)
""",
        "visual": """
daily totals -> sales trend chart
level_contribution.csv -> 12-level WRMSSE chart
optimization_candidates.csv -> candidate ranking chart
forecast_store_bias.csv -> store-level bias chart
feature importance PNGs -> model explanation gallery
""",
        "rolling_cv": """
for origin in [1802, 1830, 1858, 1886, 1914]:
    train_history = sales[:origin-1]
    actual = sales[origin:origin+27]
    candidates = seasonal_naive + trend + direct_segment
    score = WAPE(actual, prediction), Bias(actual, prediction)
summary = mean score across windows
""",
        "aggregate_features": """
item_all_store_roll_28 = sum(sales by item_id over recent 28 days)
dept_store_roll_28 = sum(sales by dept_id, store_id over recent 28 days)
bottom_zero_rate_28 = count_zero_days / 28
trend_28_vs_prior = recent_28_sum / prior_28_sum
""",
        "direct_segment": """
targets = [F1-F7 sum, F8-F14 sum, F15-F21 sum, F22-F28 sum]
train LightGBM models on historical origins
predict segment totals at d_1914
allocate each segment to daily forecasts using previous 28-day shape
score with the same 12-level WRMSSE script
""",
    }


def build_sections(summary: dict, opt: dict, visual: dict, job: dict | None = None) -> list[dict]:
    job = job or {}
    job_summary = job.get("summary", {})
    cv_best = job_summary.get("best_cv_candidate", "seasonal_naive_28")
    cv_wape = float(job_summary.get("best_cv_mean_wape", 0))
    cv_bias = float(job_summary.get("best_cv_mean_bias_pct", 0))
    direct_wrmsse = job_summary.get("direct_segment_lightgbm_avg_wrmsse")
    direct_wrmsse_text = "未运行" if direct_wrmsse is None else num(float(direct_wrmsse))
    covered = int(job_summary.get("covered", 0))
    partial = int(job_summary.get("partial", 0))
    missing = int(job_summary.get("missing", 0))
    themes = int(job_summary.get("capability_themes", covered + partial + missing))
    feature_count = int(job_summary.get("aggregate_feature_count", 0))
    return [
        {
            "id": "opening",
            "eyebrow": "01 / 开场",
            "title": "从竞赛复现到可解释的零售预测项目",
            "visual": "hero",
            "bullets": [
                "本项目把 M5 Accuracy 的 notebook 和模型产物整理成可复现脚本、可评分流程和成果网页。",
                "当前本地验证窗口为 d_1914..d_1941，优化后 Avg WRMSSE 为 "
                + num(summary["optimized_avg_wrmsse"]) + "。",
                "汇报重点不是单个模型炫技，而是数据工程、层级评估、误差诊断和优化选择的完整闭环。",
            ],
            "speech": [
                "各位好，我今天汇报的是 M5 Forecasting Accuracy 项目。这个项目的目标，是把一个竞赛型销售预测任务，整理成一个可以复现、可以解释、也可以展示的完整数据分析作品。",
                "我会先介绍参考来源和项目背景，再讲数据处理、特征工程、模型复现、评分方法、优化实验，最后分析结果和下一步改进空间。",
                "需要先说明的是，我这里的核心数字来自本地验证窗口 d_1914 到 d_1941，不直接等同于 Kaggle 官方榜单分数；但它适合用来比较我自己方案里的不同版本。",
            ],
        },
        {
            "id": "references",
            "eyebrow": "02 / 参考",
            "title": "我参考了哪些方案，以及各自给我的启发",
            "visual": "references",
            "bullets": [
                "参考了 M5 官方数据结构、Heads or Tails 的交互式 EDA、Ryan Holbrook 的多步预测练习。",
                "参考了 Matthias Anderer、btrotta、devmofl 三类高质量 M5 方案。",
                "我最终选择先强化 LightGBM 复现、WRMSSE 评估、趋势校准和可视化表达，而不是直接迁移 GPU 成本更高的深度模型。",
            ],
            "speech": [
                "我的参考可以分成四类。第一类是 M5 赛题本身，包括 sales、calendar、sell prices 三张核心表和 12 层层级评价。",
                "第二类是 EDA 参考，尤其是 Heads or Tails 的交互式 M5 EDA。它给我的启发是：成果汇报不能只放模型分数，还要让听众先理解数据结构、时间趋势、事件和品类分布。",
                "第三类是机器学习预测方法，Ryan Holbrook 的练习强调了 direct、recursive、DirRec 等多步预测思路。",
                "第四类是高排名方案。Matthias 的方案强调模型池和高层校准，btrotta 的方案强调 direct horizon 与聚合 lag，devmofl 的方案强调 DeepAR 和动态价格、事件特征。结合我的项目成本，我先落地更稳、更可解释的 LightGBM 路线。",
            ],
        },
        {
            "id": "problem",
            "eyebrow": "03 / 项目介绍",
            "title": "我要解决的问题：30,490 条底层序列的 28 天销售预测",
            "visual": "dataset",
            "bullets": [
                f"数据包含 {int_num(visual['bottom_series'])} 条商品-门店底层序列、{visual['stores']} 个门店、{visual['items']} 个商品。",
                "M5 的难点在于不仅预测底层 SKU，还要在总量、州、门店、品类、部门等 12 个层级上表现稳定。",
                "因此项目必须同时解决时间序列特征、价格/事件协变量、层级聚合评价和结果解释。",
            ],
            "speech": [
                "M5 的业务问题很直观：给定历史销量、日历和价格，预测未来 28 天的销售。",
                "但它的复杂度不低。底层有 30490 条商品-门店序列，上面还要聚合成州、门店、品类、部门等多个业务层级。",
                "这意味着，如果只在底层误差上做得好，但总量预测偏差很大，业务上仍然不够好。所以我后面会把 12 层 WRMSSE 作为核心评价标准。",
            ],
        },
        {
            "id": "data",
            "eyebrow": "04 / 数据处理",
            "title": "从三张原始表到可训练样本",
            "visual": "dataflow",
            "bullets": [
                "`sales_train_evaluation.csv` 提供商品-门店-日期销量宽表。",
                "`calendar.csv` 提供日期、节日、SNAP 等时间上下文。",
                "`sell_prices.csv` 提供门店-商品-周粒度价格。",
            ],
            "speech": [
                "数据处理的第一步，是把 sales 宽表整理成以 id 和 day 为单位的长表样本。",
                "第二步，把 calendar 的 weekday、month、event、SNAP 信息拼进去，让模型知道日期结构。",
                "第三步，把 sell prices 按 store、item、week 合并进来，让模型看到价格变化。",
                "这一步的价值是把原始销量序列变成机器学习模型可以消费的表格特征。",
            ],
        },
        {
            "id": "features",
            "eyebrow": "05 / 特征工程",
            "title": "核心特征：lag、rolling、价格、日历和类别编码",
            "visual": "feature",
            "bullets": [
                "本地模型产物使用了 lag/rolling、价格、日历、事件/SNAP 和类别 ID 等特征。",
                "feature importance 图显示 lag 和 rolling 类特征是模型解释中的核心信号。",
                "后续最值得补的是 item、dept-store 等聚合粒度 lag/rolling。",
            ],
            "speech": [
                "这个项目的核心不是把时间序列直接丢给模型，而是先把历史行为转化为特征。",
                "lag 特征回答的是：这个商品前 7 天、14 天、28 天卖得怎么样。rolling 特征回答的是：近期平均水平和趋势如何。",
                "价格特征帮助模型捕捉促销或价格变化，日历和事件特征帮助模型捕捉节假日、周末和 SNAP 效应。",
                "从当前特征重要性看，lag 和 rolling 是非常关键的；这也解释了为什么下一步要做聚合 lag，而不是盲目换模型。",
            ],
        },
        {
            "id": "model",
            "eyebrow": "06 / 模型复现",
            "title": "按门店训练 LightGBM，并递归预测未来 28 天",
            "visual": "pipeline",
            "code": "pipeline",
            "bullets": [
                "当前复现使用 archived LightGBM 模型和测试特征，生成 28 天验证预测。",
                "原始 LightGBM 版本 Avg WRMSSE 为 " + num(summary["model_avg_wrmsse"]) + "。",
                "递归预测工程成本低、复现稳定，但可能存在 horizon 误差传播。",
            ],
            "speech": [
                "模型部分我复现的是按门店训练的 LightGBM 体系。简单说，就是每个门店有一个对应模型，合计 10 个门店模型。",
                "预测未来 28 天时，模型先预测第 1 天，然后把预测结果滚动回特征中，再预测第 2 天，以此类推。",
                "这样做的优点是工程上容易复现，和表格特征结合很好；缺点是如果前面预测偏了，后面的 lag 特征也会受到影响，所以后续 direct horizon 是一个重要改进方向。",
            ],
        },
        {
            "id": "metric",
            "eyebrow": "07 / 评价方法",
            "title": "为什么我使用 12 层 WRMSSE，而不是普通 RMSE",
            "visual": "wrmsse",
            "code": "wrmsse",
            "bullets": [
                "WRMSSE 会按历史销量变化尺度标准化误差，并按最近销售额赋权。",
                "本项目在 12 个层级上分别评分，再取平均作为本地对比指标。",
                "这个指标能同时约束底层 SKU 和高层业务总量。",
            ],
            "speech": [
                "如果只用 RMSE，销量大的商品和销量小的商品会混在一起，不同层级也很难公平比较。",
                "WRMSSE 做了两件事：第一，用历史序列波动来标准化误差；第二，用最近销售额作为权重，让更重要的商品和层级影响更大。",
                "我的评分脚本会把预测聚合到 12 个层级，每一层算 WRMSSE，最后取平均。这样能看到模型在总量、门店、品类和底层 SKU 上的综合表现。",
            ],
        },
        {
            "id": "optimization",
            "eyebrow": "08 / 优化方案",
            "title": "我没有只调一个参数，而是扫描了 20 个可解释候选",
            "visual": "candidates",
            "code": "candidate",
            "bullets": [
                f"一共评估了 {visual['optimization_candidates']} 个优化候选。",
                "最佳候选为 `" + opt["best_candidate"] + "`，相对原始 LightGBM 提升 " + pct(opt["improvement_pct"]) + "。",
                "分组趋势和 horizon 趋势候选在单窗口验证中没有超过全局趋势校准。",
            ],
            "speech": [
                "优化阶段我做了一个比较克制的选择：不重新训练大模型，而是先对已经复现出的预测做可解释的后处理扫描。",
                "候选包括全局趋势 multiplier、州/门店/品类/部门分组 multiplier、按 horizon 的 multiplier，以及和 sNaive 或原始模型的 blend。",
                "最后结果非常清楚：全局趋势校准最好，Avg WRMSSE 从 0.475241 降到 0.471374。更细的分组校准反而变差，说明单窗口下细分后处理容易过拟合。",
            ],
        },
        {
            "id": "trend",
            "eyebrow": "09 / 关键优化细节",
            "title": "全局趋势 multiplier：只用预测日前的历史窗口",
            "visual": "trend",
            "code": "trend",
            "bullets": [
                "multiplier 使用 `d_1886..d_1913` 与 `d_1858..d_1885` 的总销量比值。",
                "原始 28 天总销量偏差为 " + pct(summary["model_total_bias_pct"]) + "，优化后为 " + pct(summary["optimized_total_bias_pct"]) + "。",
                "这个优化主要修正高层总量低估，对 L12 底层几乎没有改善。",
            ],
            "speech": [
                "这个 multiplier 的设计原则是不能泄漏未来信息。它只看预测日前已经知道的历史窗口。",
                "具体做法是比较最近 28 天和再之前 28 天的总量变化，如果最近整体上升，就对未来预测做一个保守放大。",
                "它的效果主要体现在总量层和品类层，能缓解系统性低估；但底层 SKU 的噪声很大，所以 L12 层几乎没有改善。",
            ],
        },
        {
            "id": "visualization",
            "eyebrow": "10 / 可视化表达",
            "title": "我把值得展示的内容拆成五类图",
            "visual": "visual",
            "code": "visual",
            "bullets": [
                "生成了 EDA 日销量、州-品类销售热力、候选优化排名、12 层 WRMSSE、示例序列和门店偏差数据。",
                "汇报网页按“先理解数据，再理解模型，再解释结果”的顺序组织。",
                "这种结构比只展示最终分数更容易体现项目逻辑和边际贡献。",
            ],
            "speech": [
                "为了让汇报更清楚，我没有只做一个分数表，而是把可视化拆成五类。",
                "第一类是 EDA，展示总销量趋势和业务层级结构。第二类是模型解释，展示特征重要性。第三类是评分，展示 12 层 WRMSSE。",
                "第四类是优化候选排名，解释为什么选择全局趋势校准。第五类是偏差诊断，解释模型在哪些层级改善明显、哪些地方仍然不足。",
            ],
        },
        {
            "id": "results",
            "eyebrow": "11 / 结果分析",
            "title": "结果不是“分数变好”这么简单：高层改善明显，底层仍是难点",
            "visual": "results",
            "bullets": [
                "sNaive Avg WRMSSE: " + num(summary["snaive_avg_wrmsse"]) + "。",
                "LightGBM Avg WRMSSE: " + num(summary["model_avg_wrmsse"]) + "。",
                "Optimized Avg WRMSSE: " + num(summary["optimized_avg_wrmsse"]) + "。",
                "L1_Total 从 " + num(0.6288945410210011) + " 降到 " + num(0.199010472456473) + "；L12_Item_Store 优化后为 " + num(0.8165459640822168) + "。",
            ],
            "speech": [
                "从最终结果看，机器学习模型相比 sNaive 有明显提升，Avg WRMSSE 从 0.837726 降到 0.475241。",
                "加入趋势优化后，进一步降到 0.471374。这个幅度不算巨大，但它是可解释、可复现、没有未来泄漏的提升。",
                "更重要的是分层结果。高层总量和品类层改善更明显，而底层 SKU 层仍然困难。这说明下一步要提升底层，需要更强的多粒度特征和 direct horizon 模型。",
            ],
        },
        {
            "id": "contribution",
            "eyebrow": "12 / 边际贡献",
            "title": "我的边际贡献：把竞赛方案整理成可复现、可解释、可展示的项目",
            "visual": "contribution",
            "bullets": [
                "完成复现脚本、评分脚本、优化扫描脚本、可视化资产脚本和汇报网页。",
                "建立了 sNaive、LightGBM、Optimized 三层对比结果。",
                "新增岗位对齐、rolling CV、聚合特征样例和 direct horizon 试验，使项目更接近真实预测岗位交付。",
            ],
            "speech": [
                "我总结自己的边际贡献有三点。",
                "第一，复现层面：把原始 notebook 和模型产物整理成脚本化流程，可以重复生成预测、评分和报告。",
                "第二，分析层面：用 WRMSSE 做 12 层拆解，不只看一个平均分，还分析高层和底层的差异。",
                "第三，岗位化层面：我把 JD 里常见的 rolling CV、WAPE、Bias、聚合特征、direct horizon、复盘和缺口清单补进项目，让它不只是竞赛复现，也更像真实业务里的预测分析交付。",
            ],
        },
        {
            "id": "jd_alignment",
            "eyebrow": "13 / 岗位对齐",
            "title": "把 9 张 JD 要求转成 HR 看得懂的能力证据",
            "visual": "jd_matrix",
            "bullets": [
                f"9 张截图被整理为 {themes} 个能力主题：{covered} 个已覆盖、{partial} 个部分覆盖、{missing} 个未覆盖。",
                "已覆盖的核心证据包括：Python/LightGBM、销售预测建模、12 层评价、rolling CV、可视化汇报。",
                "未覆盖部分会单独形成 Word 清单，避免在汇报里把能力说满。",
            ],
            "speech": [
                "为了让这个项目更贴近招聘场景，我把 9 张 JD 里反复出现的能力拆成 12 个主题。",
                "这里我没有把所有要求都硬套到项目上。已经有证据的就标成已覆盖，比如预测建模、Python、LightGBM、分层评估和汇报展示。",
                "只有一部分证据的，比如聚合特征和 direct horizon，我标成部分覆盖。完全没有做的，比如 SQL 数据仓库、补货库存优化、A/B Test、业务系统上线和 RAG 智能体，我会放进缺口清单。",
            ],
        },
        {
            "id": "rolling_cv",
            "eyebrow": "14 / 稳定实验",
            "title": "新增 rolling CV：优化必须跨多个历史窗口站得住",
            "visual": "rolling_cv",
            "code": "rolling_cv",
            "bullets": [
                f"新增 {job_summary.get('rolling_windows', 5)} 个 28 天滚动窗口，用 WAPE 和 Bias 做业务口径验证。",
                f"纯历史基线下最稳候选为 `{cv_best}`，平均 WAPE 为 {pct(cv_wape)}，平均 Bias 为 {pct(cv_bias)}。",
                "这个结果说明：单窗口有效的趋势后处理不能无条件外推，后续优化应优先进入训练特征和多窗口验证。",
            ],
            "speech": [
                "这一页是我这轮最重要的补强。之前只有 d_1914 到 d_1941 一个验证窗口，容易把偶然有效的方法误认为稳定有效。",
                "我新增了 5 个历史滚动窗口，每个窗口都只使用预测日前的历史数据，然后比较 seasonal naive、趋势校准、horizon 校准和 direct segment 校准。",
                "结果很有提醒意义：在纯历史基线下，最稳的是 28 天 seasonal naive；更复杂的趋势后处理没有跨窗口稳定胜出。所以后续不能只继续调 multiplier，而要把多粒度特征接入模型训练，再用 rolling CV 重新判断。",
            ],
        },
        {
            "id": "aggregate_direct",
            "eyebrow": "15 / 新实验",
            "title": "补聚合特征，并验证 direct horizon 试点",
            "visual": "aggregate_direct",
            "code": "aggregate_features",
            "bullets": [
                f"新增 {feature_count} 个聚合 lag/rolling 特征定义，覆盖 item、dept-store、cat-store、store、state-category 等粒度。",
                f"训练了 4 段 direct segment LightGBM 试点；本地 Avg WRMSSE 为 {direct_wrmsse_text}，未替换当前最优方案。",
                "直接采用新模型不是目标；目标是把可解释实验、失败证据和下一步训练方向沉淀下来。",
            ],
            "speech": [
                "第二个补强，是把多粒度特征先做成样例和 schema。比如同商品跨门店近 28 天销量、同部门同门店近 28 天销量、门店总趋势、底层零销量比例等。",
                "这些特征的意义是降低底层 SKU 的随机噪声，让模型看到更稳定的群体趋势。它们已经生成了样例摘要，但还没有作为训练特征重新训练当前主模型，所以我把它标为部分覆盖。",
                "第三个补强，是 direct horizon。我训练了 4 个 7 天段的 LightGBM 模型，再按上一周期日分布拆回 28 天。结果没有超过当前 LightGBM 主方案，所以暂不采用。这个结论本身有价值：它说明 direct 方向需要更强特征和更细调参，而不是简单换个训练目标就会更好。",
            ],
        },
        {
            "id": "jd_gaps",
            "eyebrow": "16 / 缺口清单",
            "title": "哪些 JD 要求还没满足：我把边界讲清楚",
            "visual": "jd_gaps",
            "bullets": [
                "P0 缺口：聚合特征尚未接入主训练；direct horizon 还只是试点。",
                "P1 缺口：SQL/数据治理、库存补货仿真、服务化部署和监控闭环还没有完成。",
                "P2/P3 缺口：因果/A-B、运筹约束、RAG/智能体适合作为后续扩展，不应该在当前项目里夸大。",
            ],
            "speech": [
                "最后这页是面向面试和复盘的诚实边界。",
                "这个项目目前最强的是预测建模、评价体系、可解释优化和展示表达；但它还不是完整供应链计划系统。",
                "所以 Word 文档里我会列清楚未满足要求：比如 SQL 数据仓库、库存补货策略、生产计划运筹、A/B Test、业务系统上线、SAP IBP 等平台经验，以及大模型 RAG 智能体。这些不是失败，而是下一阶段可以继续补的作品方向。",
            ],
        },
        {
            "id": "next",
            "eyebrow": "17 / 下一步",
            "title": "下一轮优化：把试点变成主模型能力",
            "visual": "next",
            "bullets": [
                "P0: 把 item、dept-store、cat-store 聚合 lag/rolling 接进训练数据，并用 rolling CV 复验。",
                "P1: 重做 direct horizon LightGBM，加入更强特征、早停和 ensemble，而不是采用当前试点。",
                "P2: 补 DuckDB/SQL 口径层、库存补货仿真和红黄绿预警 Dashboard。",
                "P3: 基础稳定后，再评估层级 reconciliation、Chronos/TimesFM/Moirai 或 RAG 复盘助手。",
            ],
            "speech": [
                "最后讲下一步。rolling CV 已经补上，聚合特征和 direct horizon 也已经有试点，但还没有成为主模型能力。",
                "所以下一轮优先级很清楚：第一，把聚合特征真正接进训练；第二，用 rolling CV 重新训练和比较 direct horizon；第三，补 SQL、库存仿真、监控预警这些更贴近 JD 的业务层能力。",
                "在这些基础做好后，再考虑层级 reconciliation 或 Chronos、TimesFM、Moirai 这类 foundation model 作为 ensemble 候选。这样技术升级会建立在稳定实验体系上。",
            ],
        },
    ]


def reference_cards() -> str:
    refs = [
        ("Kaggle M5", "赛题数据、12 层层级评价、WRMSSE 口径", "https://www.kaggle.com/c/m5-forecasting-accuracy"),
        ("Heads or Tails EDA", "交互式 M5 EDA 的叙事结构参考", "https://www.kaggle.com/code/headsortails/back-to-predict-the-future-interactive-m5-eda"),
        ("Ryan Holbrook", "多步预测 direct / recursive / DirRec 思路", "https://www.kaggle.com/code/ryanholbrook/exercise-forecasting-with-machine-learning"),
        ("matthiasanderer", "模型池、loss multiplier、高层校准", "https://github.com/matthiasanderer/m5-accuracy-competition"),
        ("btrotta", "direct horizon、聚合 lag/rolling", "https://github.com/btrotta/kaggle-m5"),
        ("devmofl", "DeepAR、动态事件与价格特征", "https://github.com/devmofl/M5_Accuracy_3rd"),
    ]
    return "".join(
        f'<a class="ref-card" href="{html.escape(url)}"><strong>{html.escape(name)}</strong><span>{html.escape(desc)}</span></a>'
        for name, desc, url in refs
    )


def flow_diagram() -> str:
    items = [
        ("原始数据", "sales / calendar / prices"),
        ("特征工程", "lag / rolling / price / event"),
        ("LightGBM", "10 个门店模型"),
        ("28 天预测", "recursive forecast"),
        ("WRMSSE", "12 层聚合评分"),
        ("优化汇报", "candidate scan + webpage"),
    ]
    return '<div class="flow">' + "".join(
        f'<div class="flow-step"><b>{html.escape(title)}</b><span>{html.escape(desc)}</span></div>' for title, desc in items
    ) + "</div>"


def hierarchy_visual(visual: dict) -> str:
    items = [
        ("底层序列", int_num(visual["bottom_series"])),
        ("州", int_num(visual["states"])),
        ("门店", int_num(visual["stores"])),
        ("品类", int_num(visual["categories"])),
        ("部门", int_num(visual["departments"])),
        ("商品", int_num(visual["items"])),
    ]
    return '<div class="metric-grid">' + "".join(
        f'<div><strong>{value}</strong><span>{label}</span></div>' for label, value in items
    ) + "</div>"


def status_badge(status: str) -> str:
    return f'<span class="status status-{html.escape(status.lower())}">{html.escape(status)}</span>'


def dataframe_table(rows: pd.DataFrame, columns: list[tuple[str, str]], limit: int = 8) -> str:
    if rows.empty:
        return '<div class="empty-note">暂无数据，请先运行全流程脚本。</div>'
    frame = rows.head(limit).copy()
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body_rows = []
    for _, row in frame.iterrows():
        cells = []
        for col, _label in columns:
            raw_value = row.get(col, "")
            if pd.isna(raw_value):
                value = ""
            elif isinstance(raw_value, float):
                value = pct(raw_value) if ("wape" in col or "bias" in col or col.endswith("_pct")) else num(raw_value)
            else:
                value = str(raw_value)
            if col == "status":
                cells.append(f"<td>{status_badge(value)}</td>")
            else:
                cells.append(f"<td>{html.escape(value)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'


def jd_alignment_visual(data: dict) -> str:
    job_summary = data["job"].get("summary", {})
    jd = data["job"].get("jd", pd.DataFrame())
    items = [
        ("能力主题", int_num(job_summary.get("capability_themes", len(jd)))),
        ("已覆盖", int_num(job_summary.get("covered", 0))),
        ("部分覆盖", int_num(job_summary.get("partial", 0))),
        ("未覆盖", int_num(job_summary.get("missing", 0))),
    ]
    metrics = '<div class="metric-grid compact">' + "".join(
        f'<div><strong>{value}</strong><span>{label}</span></div>' for label, value in items
    ) + "</div>"
    table = dataframe_table(
        jd,
        [
            ("theme", "能力主题"),
            ("status", "状态"),
            ("hr_talking_point", "HR 关注点"),
            ("next_action", "下一步"),
        ],
        7,
    )
    return metrics + table


def rolling_cv_visual(data: dict, charts: dict, codes: dict) -> str:
    summary = data["job"].get("cv_summary", pd.DataFrame())
    detail = dataframe_table(
        summary,
        [
            ("candidate", "候选"),
            ("mean_wape", "Mean WAPE"),
            ("mean_bias_pct", "Mean Bias"),
            ("avg_rank", "Avg Rank"),
            ("wins", "Wins"),
        ],
        6,
    )
    return charts.get("rolling_cv", "") + detail + code_block(codes["rolling_cv"])


def aggregate_direct_visual(data: dict, codes: dict) -> str:
    schema = data["job"].get("feature_schema", pd.DataFrame())
    scorecard = data["job"].get("scorecard", pd.DataFrame())
    schema_table = dataframe_table(
        schema,
        [
            ("feature_name", "特征"),
            ("source_grain", "粒度"),
            ("window_days", "窗口"),
            ("business_meaning", "业务含义"),
        ],
        7,
    )
    score_table = dataframe_table(
        scorecard,
        [
            ("area", "模块"),
            ("result", "结果"),
            ("job_value", "岗位价值"),
        ],
        5,
    )
    return score_table + schema_table + code_block(codes["direct_segment"])


def jd_gap_visual(data: dict) -> str:
    gaps = data["job"].get("gaps", pd.DataFrame())
    return dataframe_table(
        gaps,
        [
            ("priority", "优先级"),
            ("gap_area", "缺口"),
            ("jd_requirement", "JD 要求"),
            ("next_action", "补齐动作"),
        ],
        8,
    )


def visual_for(section: dict, data: dict, charts: dict, codes: dict) -> str:
    name = section["visual"]
    if name == "hero":
        return f"""
        <div class="hero-art">
          <img src="assets/m5-forecasting.png" alt="M5 project visual">
          <div class="hero-kpi"><b>{num(data['summary']['optimized_avg_wrmsse'])}</b><span>Optimized Avg WRMSSE</span></div>
        </div>
        """
    if name == "references":
        return f'<div class="ref-grid">{reference_cards()}</div>'
    if name == "dataset":
        return hierarchy_visual(data["visual"])
    if name == "dataflow":
        return flow_diagram()
    if name == "feature":
        return """
        <div class="image-pair">
          <figure><img src="assets/importance_CA_1.png" alt="CA_1 feature importance"><figcaption>CA_1 gain importance</figcaption></figure>
          <figure><img src="assets/importance_split_CA_1.png" alt="CA_1 split importance"><figcaption>CA_1 split importance</figcaption></figure>
        </div>
        """
    if name == "pipeline":
        return flow_diagram() + code_block(codes["pipeline"])
    if name == "wrmsse":
        return code_block(codes["wrmsse"]) + charts["wrmsse"]
    if name == "candidates":
        return charts["candidates"] + code_block(codes["candidate"])
    if name == "trend":
        return charts["daily"] + code_block(codes["trend"])
    if name == "visual":
        return code_block(codes["visual"]) + """
        <div class="tile-list">
          <span>EDA 趋势</span><span>层级热力</span><span>WRMSSE 对比</span><span>候选排名</span><span>偏差诊断</span>
        </div>
        """
    if name == "results":
        return charts["wrmsse"]
    if name == "contribution":
        return """
        <div class="contribution">
          <div><b>复现</b><span>notebook -> scripts</span></div>
          <div><b>评价</b><span>WRMSSE 12 层</span></div>
          <div><b>优化</b><span>20 个候选扫描</span></div>
          <div><b>展示</b><span>网页 + 讲稿</span></div>
        </div>
        """
    if name == "jd_matrix":
        return jd_alignment_visual(data)
    if name == "rolling_cv":
        return rolling_cv_visual(data, charts, codes)
    if name == "aggregate_direct":
        return aggregate_direct_visual(data, codes)
    if name == "jd_gaps":
        return jd_gap_visual(data)
    if name == "next":
        return """
        <div class="roadmap">
          <div><b>P0</b><span>Rolling CV</span></div>
          <div><b>P1</b><span>Aggregate lag</span></div>
          <div><b>P2</b><span>Direct horizon</span></div>
          <div><b>P3</b><span>Reconciliation / foundation model</span></div>
        </div>
        """
    return ""


def build_markdown(sections: list[dict]) -> str:
    lines = [
        "# M5 Forecasting Accuracy 成果汇报讲稿",
        "",
        "> 使用方式：每个章节对应网页中的一个汇报段落。先讲“屏幕重点”，再按“讲稿”自然展开。",
        "",
    ]
    for idx, section in enumerate(sections, start=1):
        lines.extend(
            [
                f"## {idx}. {section['title']}",
                "",
                f"**屏幕重点：** {section['eyebrow']}",
                "",
                "**要点：**",
            ]
        )
        for bullet in section["bullets"]:
            lines.append(f"- {bullet}")
        if section.get("code"):
            lines.extend(["", f"**主要代码/伪代码：** `{section['code']}`"])
        lines.extend(["", "**讲稿：**"])
        for para in section["speech"]:
            lines.append("")
            lines.append(para)
        lines.append("")
    return "\n".join(lines)


def build_html(sections: list[dict], data: dict, charts: dict, codes: dict) -> str:
    css = """
:root{--ink:#17212b;--muted:#667085;--paper:#f6f7fb;--panel:#fff;--line:#d9e0ea;--teal:#167a78;--blue:#2563eb;--amber:#b7791f;--coral:#d55c45;--green:#2f855a}
*{box-sizing:border-box}body{margin:0;font-family:Inter,"Segoe UI","Microsoft YaHei",Arial,sans-serif;background:var(--paper);color:var(--ink)}a{color:inherit}
.layout{display:grid;grid-template-columns:260px 1fr;min-height:100vh}.side{position:sticky;top:0;height:100vh;padding:24px;background:#111827;color:#f9fafb;overflow:auto}.side h1{font-size:22px;line-height:1.15;margin:0 0 12px;letter-spacing:0}.side p{color:#cbd5e1;font-size:13px;line-height:1.55}.side a{display:block;text-decoration:none;color:#e5e7eb;padding:8px 0;border-bottom:1px solid rgba(255,255,255,.08);font-size:13px}
.main{padding:26px clamp(18px,4vw,58px) 54px}.section{min-height:88vh;display:grid;grid-template-columns:minmax(0,.92fr) minmax(360px,1.08fr);gap:28px;align-items:center;border-bottom:1px solid var(--line);padding:34px 0}.section:first-child{padding-top:0}
.eyebrow{color:var(--teal);font-weight:900;font-size:13px;letter-spacing:.06em;text-transform:uppercase}.text h2{font-size:clamp(30px,4vw,54px);line-height:1.04;margin:10px 0 18px;letter-spacing:0}.bullets{display:grid;gap:10px;margin:18px 0}.bullet{background:#fff;border-left:5px solid var(--teal);border-radius:8px;padding:12px 14px;box-shadow:0 8px 20px rgba(17,24,39,.05);line-height:1.55}.speaker{background:#eef5f4;border:1px solid #cfe4e0;border-radius:8px;padding:16px 18px;margin-top:16px}.speaker b{display:block;margin-bottom:8px}.speaker p{margin:0 0 10px;line-height:1.72;color:#334155}
.visual{background:#fff;border:1px solid var(--line);border-radius:8px;padding:18px;box-shadow:0 16px 36px rgba(17,24,39,.07);overflow:hidden}.visual svg{width:100%;display:block}.gridline{stroke:#e8edf3;stroke-width:1}.axis{fill:#667085;font-size:12px}.line{fill:none;stroke-width:3;stroke-linecap:round;stroke-linejoin:round}.barlabel{fill:#344054;font-size:12px;font-weight:700}
pre{margin:16px 0 0;background:#0f172a;color:#e2e8f0;border-radius:8px;padding:14px;overflow:auto;font-size:12px;line-height:1.55}code{font-family:"Cascadia Code","Consolas",monospace}
.hero-art{position:relative}.hero-art img{width:100%;display:block;border-radius:8px;border:1px solid var(--line)}.hero-kpi{position:absolute;right:18px;bottom:18px;background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px 16px;box-shadow:0 12px 28px rgba(17,24,39,.16)}.hero-kpi b{font-size:36px;display:block}.hero-kpi span{color:var(--muted);font-size:12px}
.ref-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.ref-card{text-decoration:none;border:1px solid var(--line);border-radius:8px;padding:14px;background:#f8fafc}.ref-card strong{display:block;margin-bottom:8px}.ref-card span{display:block;color:var(--muted);font-size:13px;line-height:1.5}
.metric-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.metric-grid div,.contribution div,.roadmap div{border:1px solid var(--line);border-radius:8px;padding:18px;background:#f8fafc}.metric-grid strong{display:block;font-size:34px}.metric-grid span,.contribution span,.roadmap span{color:var(--muted);display:block;margin-top:6px}
.flow{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.flow-step{min-height:112px;border-top:5px solid var(--amber);background:#f8fafc;border-radius:8px;padding:14px}.flow-step:nth-child(2){border-color:var(--teal)}.flow-step:nth-child(3){border-color:var(--blue)}.flow-step:nth-child(4){border-color:var(--coral)}.flow-step:nth-child(5){border-color:var(--green)}.flow-step span{display:block;color:var(--muted);margin-top:8px;font-size:13px}
.image-pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}figure img{width:100%;display:block;border:1px solid var(--line);border-radius:8px}figcaption{color:var(--muted);font-size:12px;margin-top:6px}
.tile-list{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-top:16px}.tile-list span{background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:16px 10px;text-align:center;font-weight:800}
.contribution,.roadmap{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.contribution b,.roadmap b{font-size:24px;color:var(--teal)}
.metric-grid.compact{grid-template-columns:repeat(4,minmax(0,1fr));margin-bottom:14px}.metric-grid.compact strong{font-size:26px}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px;margin-top:14px;background:#fff}table{width:100%;border-collapse:collapse;font-size:12px;line-height:1.45}th,td{padding:10px 12px;border-bottom:1px solid #e8edf3;text-align:left;vertical-align:top}th{background:#f8fafc;color:#334155;font-size:12px}tr:last-child td{border-bottom:0}.status{display:inline-block;border-radius:999px;padding:3px 8px;font-weight:800;font-size:11px}.status-covered{background:#e7f7ef;color:#166534}.status-partial{background:#fff7df;color:#8a5a00}.status-missing{background:#feecec;color:#9b1c1c}.empty-note{background:#f8fafc;border:1px dashed var(--line);border-radius:8px;padding:18px;color:var(--muted)}
.footer{padding:28px 0;color:var(--muted);font-size:13px}.toplinks{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}.toplinks a{background:#fff;border:1px solid var(--line);border-radius:6px;text-decoration:none;padding:8px 10px}
@media(max-width:980px){.layout{grid-template-columns:1fr}.side{position:static;height:auto}.section{grid-template-columns:1fr;min-height:auto}.flow,.ref-grid,.tile-list{grid-template-columns:1fr}.image-pair,.contribution,.roadmap,.metric-grid{grid-template-columns:1fr}}
"""
    nav = "\n".join(f'<a href="#{s["id"]}">{idx:02d}. {html.escape(s["title"])}</a>' for idx, s in enumerate(sections, start=1))
    body_sections = []
    for section in sections:
        bullets = "".join(f'<div class="bullet">{html.escape(item)}</div>' for item in section["bullets"])
        speech = "".join(f'<p>{html.escape(para)}</p>' for para in section["speech"])
        visual = visual_for(section, data, charts, codes)
        body_sections.append(
            f"""
            <section class="section" id="{section['id']}">
              <div class="text">
                <div class="eyebrow">{html.escape(section['eyebrow'])}</div>
                <h2>{html.escape(section['title'])}</h2>
                <div class="bullets">{bullets}</div>
                <div class="speaker"><b>讲稿</b>{speech}</div>
              </div>
              <div class="visual">{visual}</div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>M5 Forecasting Accuracy 成果汇报</title>
  <style>{css}</style>
</head>
<body>
  <div class="layout">
    <aside class="side">
      <h1>M5 成果汇报</h1>
      <p>讲稿与网页一一对应：参考、项目介绍、处理过程、代码关键点、结果分析。</p>
      <div class="toplinks">
        <a href="m5_project_showcase.html">交互 Dashboard</a>
        <a href="presentation_script.md">讲稿 Markdown</a>
      </div>
      <nav>{nav}</nav>
    </aside>
    <main class="main">
      {"".join(body_sections)}
      <div class="footer">
        页面由本地脚本生成，核心指标来自 output/project_summary.json、output/level_contribution.csv 和 output/optimization_candidates.csv。<br>
      </div>
    </main>
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    summary = read_json(args.summary)
    opt = read_json(args.optimization_summary)
    levels = read_csv(args.levels)
    candidates = read_csv(args.candidates)
    daily = read_csv(args.daily)
    visual = read_json(args.visual_dir / "visual_summary.json")
    eda_daily = read_csv(args.visual_dir / "eda_daily_sales.csv")
    job = {
        "summary": read_json_optional(args.job_dir / "job_alignment_summary.json"),
        "jd": read_csv_optional(args.job_dir / "jd_capability_matrix.csv"),
        "gaps": read_csv_optional(args.job_dir / "jd_gap_list.csv"),
        "cv_summary": read_csv_optional(args.job_dir / "rolling_cv_summary.csv"),
        "feature_schema": read_csv_optional(args.job_dir / "aggregate_feature_schema.csv"),
        "scorecard": read_csv_optional(args.job_dir / "forecast_governance_scorecard.csv"),
        "direct_score": read_csv_optional(args.direct_score),
    }
    codes = script_code()

    charts = {
        "daily": line_svg(daily, [("actual", "Actual", "#17212b"), ("model", "LightGBM", "#2563eb"), ("optimized", "Optimized", "#2f855a")], 260),
        "eda": line_svg(eda_daily, [("sales", "Daily sales", "#4f46e5"), ("roll_28", "28-day rolling mean", "#167a78")], 260),
        "wrmsse": bars_svg(levels[levels["level"] != "Average"], "level", [("wrmsse_snaive", "sNaive", "#b7791f"), ("wrmsse_model", "LightGBM", "#2563eb"), ("wrmsse_optimized", "Optimized", "#2f855a")]),
        "candidates": single_bars(candidates.sort_values("avg_wrmsse"), "candidate", "avg_wrmsse", "#167a78", 8),
        "rolling_cv": ""
        if job["cv_summary"].empty
        else single_bars(job["cv_summary"].sort_values("mean_wape"), "candidate", "mean_wape", "#167a78", 8),
    }
    data = {
        "summary": summary,
        "opt": opt,
        "levels": levels,
        "candidates": candidates,
        "daily": daily,
        "visual": visual,
        "job": job,
    }
    sections = build_sections(summary, opt, visual, job)

    args.html_out.parent.mkdir(parents=True, exist_ok=True)
    args.script_out.parent.mkdir(parents=True, exist_ok=True)
    args.html_out.write_text(build_html(sections, data, charts, codes), encoding="utf-8")
    args.script_out.write_text(build_markdown(sections), encoding="utf-8")
    if args.index_out:
        shutil.copyfile(args.html_out, args.index_out)
    copy_report_assets(args.html_out.parent)
    print(f"wrote {args.html_out}")
    print(f"wrote {args.script_out}")
    print(f"wrote {args.index_out}")


if __name__ == "__main__":
    main()
