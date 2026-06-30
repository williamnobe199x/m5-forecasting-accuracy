"""Build job-aligned forecast validation and capability assets.

The report uses these assets to connect the M5 project with practical demand
forecasting, supply-chain analytics, and algorithm-engineering job requirements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
DAY_COLS = [f"d_{day}" for day in range(1, 1942)]
WINDOWS = [
    ("cv_1802_1829", 1802),
    ("cv_1830_1857", 1830),
    ("cv_1858_1885", 1858),
    ("cv_1886_1913", 1886),
    ("validation_1914_1941", 1914),
]
SEGMENTS = [(1, 7), (8, 14), (15, 21), (22, 28)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales", default="sales_train_evaluation.csv", type=Path)
    parser.add_argument("--out-dir", default="output/job_alignment", type=Path)
    parser.add_argument("--direct-score", default="output/direct_segment_lightgbm_wrmsse_by_level.csv", type=Path)
    parser.add_argument("--smooth", default=100.0, type=float)
    parser.add_argument("--clip-low", default=0.85, type=float)
    parser.add_argument("--clip-high", default=1.15, type=float)
    return parser.parse_args()


def dcols(start: int, length: int = 28) -> list[str]:
    return [f"d_{day}" for day in range(start, start + length)]


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum(actual))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.abs(actual - pred)) / denom)


def bias_pct(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum(actual))
    if denom <= 0:
        return float("nan")
    return float((np.sum(pred) - np.sum(actual)) / denom)


def ratio_from_arrays(recent: np.ndarray, prior: np.ndarray, smooth: float, low: float, high: float) -> float:
    raw = (float(np.sum(recent)) + smooth) / (float(np.sum(prior)) + smooth)
    return float(np.clip(raw, low, high))


def group_ratio(
    sales: pd.DataFrame,
    origin: int,
    group_cols: list[str],
    smooth: float,
    low: float,
    high: float,
) -> np.ndarray:
    recent_cols = dcols(origin - 28)
    prior_cols = dcols(origin - 56)
    recent = sales[recent_cols].sum(axis=1)
    prior = sales[prior_cols].sum(axis=1)

    if not group_cols:
        ratio = ratio_from_arrays(recent.to_numpy(), prior.to_numpy(), smooth, low, high)
        return np.full(len(sales), ratio, dtype=np.float64)

    frame = sales[group_cols].copy()
    frame["recent"] = recent
    frame["prior"] = prior
    grouped = frame.groupby(group_cols, observed=True)[["recent", "prior"]].sum().reset_index()
    grouped["ratio"] = ((grouped["recent"] + smooth) / (grouped["prior"] + smooth)).clip(low, high)
    merged = sales[group_cols].merge(grouped[group_cols + ["ratio"]], on=group_cols, how="left")
    return merged["ratio"].fillna(1.0).to_numpy(dtype=np.float64)


def horizon_ratio(
    sales: pd.DataFrame,
    origin: int,
    smooth: float,
    low: float,
    high: float,
) -> np.ndarray:
    ratios = []
    for offset in range(28):
        recent = sales[f"d_{origin - 28 + offset}"].to_numpy(dtype=np.float64)
        prior = sales[f"d_{origin - 56 + offset}"].to_numpy(dtype=np.float64)
        ratios.append(ratio_from_arrays(recent, prior, smooth, low, high))
    return np.asarray(ratios, dtype=np.float64)


def segment_ratio(
    sales: pd.DataFrame,
    origin: int,
    smooth: float,
    low: float,
    high: float,
) -> np.ndarray:
    ratios = np.ones(28, dtype=np.float64)
    for seg_start, seg_end in SEGMENTS:
        offsets = list(range(seg_start - 1, seg_end))
        recent_cols = [f"d_{origin - 28 + offset}" for offset in offsets]
        prior_cols = [f"d_{origin - 56 + offset}" for offset in offsets]
        ratio = ratio_from_arrays(
            sales[recent_cols].to_numpy(dtype=np.float64),
            sales[prior_cols].to_numpy(dtype=np.float64),
            smooth,
            low,
            high,
        )
        ratios[offsets] = ratio
    return ratios


def make_candidates(
    sales: pd.DataFrame,
    origin: int,
    smooth: float,
    low: float,
    high: float,
) -> dict[str, np.ndarray]:
    base = sales[dcols(origin - 28)].to_numpy(dtype=np.float64)
    candidates: dict[str, np.ndarray] = {"seasonal_naive_28": base}

    for name, group_cols in {
        "trend_global": [],
        "trend_state": ["state_id"],
        "trend_store": ["store_id"],
        "trend_category": ["cat_id"],
        "trend_department": ["dept_id"],
        "trend_store_department": ["store_id", "dept_id"],
        "trend_store_category": ["store_id", "cat_id"],
    }.items():
        ratios = group_ratio(sales, origin, group_cols, smooth, low, high)
        candidates[name] = np.maximum(base * ratios[:, None], 0)

    candidates["trend_horizon_daily"] = np.maximum(base * horizon_ratio(sales, origin, smooth, low, high), 0)
    candidates["direct_segment_trend"] = np.maximum(base * segment_ratio(sales, origin, smooth, low, high), 0)

    for weight in [0.25, 0.50, 0.75]:
        candidates[f"blend_base_global_{int(weight * 100)}"] = (
            base * (1 - weight) + candidates["trend_global"] * weight
        )
    return candidates


def group_metrics(
    sales: pd.DataFrame,
    actual: np.ndarray,
    pred: np.ndarray,
    group_cols: list[str],
    window_name: str,
    candidate: str,
) -> pd.DataFrame:
    frame = sales[group_cols].copy()
    frame["actual_sum"] = actual.sum(axis=1)
    frame["pred_sum"] = pred.sum(axis=1)
    frame["abs_error"] = np.abs(actual - pred).sum(axis=1)
    grouped = frame.groupby(group_cols, observed=True)[["actual_sum", "pred_sum", "abs_error"]].sum().reset_index()
    grouped["wape"] = grouped["abs_error"] / grouped["actual_sum"].replace(0, np.nan)
    grouped["bias_pct"] = (grouped["pred_sum"] - grouped["actual_sum"]) / grouped["actual_sum"].replace(0, np.nan)
    grouped["window"] = window_name
    grouped["candidate"] = candidate
    grouped["grain"] = "+".join(group_cols)
    return grouped


def build_rolling_cv(
    sales: pd.DataFrame,
    smooth: float,
    low: float,
    high: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    segment_rows = []
    group_rows = []

    for window_name, origin in WINDOWS:
        actual = sales[dcols(origin)].to_numpy(dtype=np.float64)
        candidates = make_candidates(sales, origin, smooth, low, high)

        for candidate, pred in candidates.items():
            rows.append(
                {
                    "window": window_name,
                    "origin_day": origin,
                    "candidate": candidate,
                    "wape": wape(actual, pred),
                    "bias_pct": bias_pct(actual, pred),
                    "actual_units": float(actual.sum()),
                    "predicted_units": float(pred.sum()),
                }
            )

            for seg_start, seg_end in SEGMENTS:
                idx = slice(seg_start - 1, seg_end)
                segment_rows.append(
                    {
                        "window": window_name,
                        "candidate": candidate,
                        "horizon_segment": f"F{seg_start}_F{seg_end}",
                        "wape": wape(actual[:, idx], pred[:, idx]),
                        "bias_pct": bias_pct(actual[:, idx], pred[:, idx]),
                    }
                )

        summary = pd.DataFrame(rows)
        best_candidate = (
            summary.loc[summary["window"] == window_name].sort_values(["wape", "bias_pct"]).iloc[0]["candidate"]
        )
        best_pred = candidates[str(best_candidate)]
        for group_cols in [["state_id"], ["store_id"], ["cat_id"], ["store_id", "cat_id"]]:
            group_rows.append(group_metrics(sales, actual, best_pred, group_cols, window_name, str(best_candidate)))

    detail = pd.DataFrame(rows)
    rank = detail.copy()
    rank["rank"] = rank.groupby("window")["wape"].rank(method="dense")
    summary = (
        rank.groupby("candidate", observed=True)
        .agg(
            mean_wape=("wape", "mean"),
            std_wape=("wape", "std"),
            mean_bias_pct=("bias_pct", "mean"),
            max_abs_bias_pct=("bias_pct", lambda s: float(np.nanmax(np.abs(s)))),
            avg_rank=("rank", "mean"),
            wins=("rank", lambda s: int((s == 1).sum())),
            windows=("window", "nunique"),
        )
        .reset_index()
        .sort_values(["mean_wape", "avg_rank"], ascending=True)
    )
    segment = pd.DataFrame(segment_rows)
    business_group = pd.concat(group_rows, ignore_index=True)
    return detail, summary, segment, business_group


def feature_stats(values: pd.Series) -> dict[str, float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p90": float(clean.quantile(0.90)),
        "max": float(clean.max()),
    }


def add_group_feature(
    sales: pd.DataFrame,
    features: pd.DataFrame,
    group_cols: list[str],
    source_cols: list[str],
    feature_name: str,
) -> None:
    grouped = sales[group_cols].copy()
    grouped[feature_name] = sales[source_cols].sum(axis=1)
    values = grouped.groupby(group_cols, observed=True)[feature_name].sum().reset_index()
    features[feature_name] = sales[group_cols].merge(values, on=group_cols, how="left")[feature_name].to_numpy()


def build_aggregate_features(sales: pd.DataFrame, origin: int = 1914) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = sales[ID_COLS].copy()
    recent_7 = dcols(origin - 7, 7)
    recent_14 = dcols(origin - 14, 14)
    recent_28 = dcols(origin - 28, 28)
    prior_28 = dcols(origin - 56, 28)

    features["bottom_roll_7_sum"] = sales[recent_7].sum(axis=1)
    features["bottom_roll_14_sum"] = sales[recent_14].sum(axis=1)
    features["bottom_roll_28_sum"] = sales[recent_28].sum(axis=1)
    features["bottom_zero_rate_28"] = (sales[recent_28] == 0).mean(axis=1)
    features["bottom_trend_28_vs_prior"] = (sales[recent_28].sum(axis=1) + 1) / (sales[prior_28].sum(axis=1) + 1)

    feature_specs = [
        ("item_all_store_roll_28_sum", ["item_id"], recent_28, "同商品跨门店近 28 天总销量"),
        ("dept_store_roll_28_sum", ["dept_id", "store_id"], recent_28, "同部门同门店近 28 天总销量"),
        ("cat_store_roll_28_sum", ["cat_id", "store_id"], recent_28, "同品类同门店近 28 天总销量"),
        ("store_roll_28_sum", ["store_id"], recent_28, "门店近 28 天总销量"),
        ("state_cat_roll_28_sum", ["state_id", "cat_id"], recent_28, "州-品类近 28 天总销量"),
    ]
    for name, group_cols, cols, _meaning in feature_specs:
        add_group_feature(sales, features, group_cols, cols, name)

    add_group_feature(sales, features, ["item_id"], prior_28, "item_all_store_prior_28_sum")
    add_group_feature(sales, features, ["dept_id", "store_id"], prior_28, "dept_store_prior_28_sum")
    features["item_all_store_trend"] = (features["item_all_store_roll_28_sum"] + 1) / (
        features["item_all_store_prior_28_sum"] + 1
    )
    features["dept_store_trend"] = (features["dept_store_roll_28_sum"] + 1) / (
        features["dept_store_prior_28_sum"] + 1
    )

    schema_rows = [
        {
            "feature_name": "bottom_roll_7_sum",
            "source_grain": "item-store",
            "window_days": 7,
            "business_meaning": "底层 SKU-门店短期销售强度",
            "implementation_status": "computed_pilot",
        },
        {
            "feature_name": "bottom_roll_14_sum",
            "source_grain": "item-store",
            "window_days": 14,
            "business_meaning": "底层 SKU-门店双周销售强度",
            "implementation_status": "computed_pilot",
        },
        {
            "feature_name": "bottom_roll_28_sum",
            "source_grain": "item-store",
            "window_days": 28,
            "business_meaning": "底层 SKU-门店月度销售强度",
            "implementation_status": "computed_pilot",
        },
        {
            "feature_name": "bottom_zero_rate_28",
            "source_grain": "item-store",
            "window_days": 28,
            "business_meaning": "间歇性需求和断货/低动销风险信号",
            "implementation_status": "computed_pilot",
        },
        {
            "feature_name": "bottom_trend_28_vs_prior",
            "source_grain": "item-store",
            "window_days": 56,
            "business_meaning": "底层趋势变化",
            "implementation_status": "computed_pilot",
        },
    ]
    for name, group_cols, cols, meaning in feature_specs:
        schema_rows.append(
            {
                "feature_name": name,
                "source_grain": "+".join(group_cols),
                "window_days": len(cols),
                "business_meaning": meaning,
                "implementation_status": "computed_pilot",
            }
        )
    schema_rows.extend(
        [
            {
                "feature_name": "item_all_store_trend",
                "source_grain": "item",
                "window_days": 56,
                "business_meaning": "同商品跨门店趋势变化，降低单门店噪声",
                "implementation_status": "computed_pilot",
            },
            {
                "feature_name": "dept_store_trend",
                "source_grain": "dept-store",
                "window_days": 56,
                "business_meaning": "同部门同门店趋势变化，支撑门店补货和品类判断",
                "implementation_status": "computed_pilot",
            },
        ]
    )
    schema = pd.DataFrame(schema_rows)

    stat_rows = []
    for col in [c for c in features.columns if c not in ID_COLS]:
        stat_rows.append({"feature_name": col, **feature_stats(features[col])})
    summary = pd.DataFrame(stat_rows)
    sample = features.head(120)
    return schema, summary, sample


def build_jd_matrix() -> pd.DataFrame:
    rows = [
        {
            "theme": "销售预测建模",
            "jd_signal": "SKU/门店/渠道/区域多维度预测，支持短中长期需求判断",
            "project_evidence": "M5 底层商品-门店 28 天预测，按 12 层 WRMSSE 拆解",
            "status": "Covered",
            "hr_talking_point": "能够把竞赛预测任务翻译成零售销售预测场景",
            "next_action": "增加周/月粒度聚合输出，贴近 DP 周期",
        },
        {
            "theme": "稳定实验体系",
            "jd_signal": "预测准确率监控、偏差分析、定期复盘",
            "project_evidence": "新增 5 个 rolling CV 窗口、WAPE、Bias 和 horizon segment 评估",
            "status": "Covered",
            "hr_talking_point": "不是单窗口调参，而是跨窗口比较方案稳定性",
            "next_action": "把红黄绿预警接入交互 Dashboard",
        },
        {
            "theme": "多粒度特征工程",
            "jd_signal": "节假日、促销、库存、市场波动与聚合趋势特征",
            "project_evidence": "已有日历/价格/lag/rolling；新增 item、dept-store、cat-store 聚合特征样例",
            "status": "Partial",
            "hr_talking_point": "能解释为什么聚合 lag 能降低底层噪声",
            "next_action": "把聚合特征接入训练数据并重新训练",
        },
        {
            "theme": "Direct horizon 思路",
            "jd_signal": "降低递归预测误差传播，按未来周期输出可控预测",
            "project_evidence": "新增 4 段 direct segment trend 试点评估",
            "status": "Partial",
            "hr_talking_point": "已经把 direct/recursive 的差异转成可落地实验",
            "next_action": "训练 4 个 direct LightGBM horizon 模型并与 recursive ensemble",
        },
        {
            "theme": "模型工具栈",
            "jd_signal": "Python、Pandas、NumPy、sklearn、LightGBM/XGBoost/CatBoost",
            "project_evidence": "项目脚本化使用 Pandas/NumPy/LightGBM 产物与可复现评分",
            "status": "Covered",
            "hr_talking_point": "有端到端脚本和可审计输出，而不是只跑 notebook",
            "next_action": "补充 XGBoost/CatBoost 或 AutoML 对照实验",
        },
        {
            "theme": "可视化与汇报",
            "jd_signal": "Dashboard、定期报告、管理层汇报、业务洞察表达",
            "project_evidence": "docs/index.html、交互 Dashboard、讲稿 Markdown、结果图表",
            "status": "Covered",
            "hr_talking_point": "能把模型结果讲给非算法听众",
            "next_action": "增加 HR 能力矩阵和业务 RCA 页面",
        },
        {
            "theme": "供应链补货与库存",
            "jd_signal": "安全库存、补货频次、库存周转、订单满足率联动",
            "project_evidence": "当前有需求预测和偏差诊断，但没有库存成本/服务水平优化模型",
            "status": "Missing",
            "hr_talking_point": "可承认边界，并提出预测到补货的下一步",
            "next_action": "增加 safety stock 和 reorder point 仿真模块",
        },
        {
            "theme": "业务落地和系统化",
            "jd_signal": "模型上线、内部服务、审批流程、异常预警、持续优化",
            "project_evidence": "当前是本地脚本和静态网页，没有 API/定时任务/服务化部署",
            "status": "Missing",
            "hr_talking_point": "项目展示了研发雏形，还需补工程化闭环",
            "next_action": "封装 CLI/API，增加定时重跑和监控日志",
        },
        {
            "theme": "SQL 与数据治理",
            "jd_signal": "复杂 SQL、窗口函数、数据质量治理、口径标准",
            "project_evidence": "当前使用 CSV 文件，没有 SQL 仓库和数据质量规则集",
            "status": "Missing",
            "hr_talking_point": "可补一层 DuckDB/SQL 指标口径来展示能力",
            "next_action": "用 DuckDB 复刻核心宽表、聚合口径和质量校验",
        },
        {
            "theme": "运筹优化/生产计划",
            "jd_signal": "产能约束、交期约束、MOQ、生产批量、供应周期",
            "project_evidence": "当前只覆盖预测，不包含约束优化和生产计划",
            "status": "Missing",
            "hr_talking_point": "适合定位为需求预测项目，而不是完整计划优化系统",
            "next_action": "建立小型线性规划 demo：预测需求到补货/生产约束",
        },
        {
            "theme": "因果/A-B/What-if",
            "jd_signal": "因果推断、A/B Test、策略模拟、Bad Case 闭环",
            "project_evidence": "当前有候选对照和误差分析，但没有因果识别或真实实验设计",
            "status": "Missing",
            "hr_talking_point": "可把预测偏差复盘扩展成实验和策略评估",
            "next_action": "增加促销 uplift 的准实验设计或 what-if 仿真",
        },
        {
            "theme": "大模型/智能体/RAG",
            "jd_signal": "Prompt、RAG、重排序、供应链智能体与大模型协同决策",
            "project_evidence": "当前未引入 LLM 决策助手或知识库检索流程",
            "status": "Missing",
            "hr_talking_point": "先把预测系统做稳，再把 LLM 放在解释和复盘层",
            "next_action": "用项目文档构建一个预测复盘 RAG 助手",
        },
    ]
    return pd.DataFrame(rows)


def build_gap_list(jd: pd.DataFrame) -> pd.DataFrame:
    gaps = jd[jd["status"].isin(["Partial", "Missing"])].copy()
    priority_map = {
        "多粒度特征工程": "P0",
        "Direct horizon 思路": "P0",
        "SQL 与数据治理": "P1",
        "供应链补货与库存": "P1",
        "业务落地和系统化": "P1",
        "因果/A-B/What-if": "P2",
        "运筹优化/生产计划": "P2",
        "大模型/智能体/RAG": "P3",
    }
    gaps["priority"] = gaps["theme"].map(priority_map).fillna("P2")
    gaps = gaps.rename(
        columns={
            "theme": "gap_area",
            "jd_signal": "jd_requirement",
            "project_evidence": "current_project_boundary",
        }
    )
    return gaps[
        [
            "priority",
            "gap_area",
            "jd_requirement",
            "current_project_boundary",
            "next_action",
            "hr_talking_point",
        ]
    ].sort_values(["priority", "gap_area"])


def build_scorecard(
    jd: pd.DataFrame,
    cv_summary: pd.DataFrame,
    schema: pd.DataFrame,
    direct_avg_wrmsse: float | None,
) -> tuple[pd.DataFrame, dict]:
    best = cv_summary.iloc[0]
    base = cv_summary.loc[cv_summary["candidate"] == "seasonal_naive_28"].iloc[0]
    status_counts = jd["status"].value_counts().to_dict()
    summary = {
        "jd_screenshots": 9,
        "capability_themes": int(len(jd)),
        "covered": int(status_counts.get("Covered", 0)),
        "partial": int(status_counts.get("Partial", 0)),
        "missing": int(status_counts.get("Missing", 0)),
        "rolling_windows": int(WINDOWS.__len__()),
        "best_cv_candidate": str(best["candidate"]),
        "best_cv_mean_wape": float(best["mean_wape"]),
        "base_cv_mean_wape": float(base["mean_wape"]),
        "best_cv_mean_bias_pct": float(best["mean_bias_pct"]),
        "aggregate_feature_count": int(len(schema)),
        "direct_horizon_segments": int(len(SEGMENTS)),
        "direct_segment_lightgbm_avg_wrmsse": direct_avg_wrmsse,
    }
    rows = [
        {
            "area": "Rolling CV",
            "artifact": "rolling_cv_detail.csv / rolling_cv_summary.csv",
            "result": f"{summary['rolling_windows']} windows, best candidate {summary['best_cv_candidate']}",
            "job_value": "支撑预测准确率监控、复盘和持续迭代",
        },
        {
            "area": "Business WAPE and Bias",
            "artifact": "rolling_cv_business_groups.csv",
            "result": "state/store/category/store-category grains",
            "job_value": "贴近 SKU/区域/渠道粒度的预测考核",
        },
        {
            "area": "Aggregate lag features",
            "artifact": "aggregate_feature_schema.csv / aggregate_feature_pilot_summary.csv",
            "result": f"{summary['aggregate_feature_count']} feature definitions",
            "job_value": "把低噪声聚合趋势纳入模型设计",
        },
        {
            "area": "Direct horizon pilot",
            "artifact": "rolling_cv_horizon_segments.csv",
            "result": (
                f"{summary['direct_horizon_segments']} 7-day segments"
                if direct_avg_wrmsse is None
                else f"{summary['direct_horizon_segments']} 7-day segments, pilot Avg WRMSSE {direct_avg_wrmsse:.3f}"
            ),
            "job_value": "验证 direct horizon 方向，但暂不替换当前最佳结果",
        },
        {
            "area": "JD gap analysis",
            "artifact": "jd_capability_matrix.csv / jd_gap_list.csv",
            "result": f"{summary['covered']} covered, {summary['partial']} partial, {summary['missing']} missing",
            "job_value": "把成果汇报转成招聘方能理解的能力证据",
        },
    ]
    return pd.DataFrame(rows), summary


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sales = pd.read_csv(args.sales, usecols=ID_COLS + DAY_COLS)

    cv_detail, cv_summary, cv_segments, cv_groups = build_rolling_cv(
        sales,
        smooth=args.smooth,
        low=args.clip_low,
        high=args.clip_high,
    )
    schema, feature_summary, feature_sample = build_aggregate_features(sales)
    jd = build_jd_matrix()
    gaps = build_gap_list(jd)
    direct_avg_wrmsse = None
    if args.direct_score.exists():
        direct_score = pd.read_csv(args.direct_score)
        average = direct_score.loc[direct_score["level"] == "Average", "wrmsse"]
        if not average.empty:
            direct_avg_wrmsse = float(average.iloc[0])
    scorecard, summary = build_scorecard(jd, cv_summary, schema, direct_avg_wrmsse)

    cv_detail.to_csv(args.out_dir / "rolling_cv_detail.csv", index=False)
    cv_summary.to_csv(args.out_dir / "rolling_cv_summary.csv", index=False)
    cv_segments.to_csv(args.out_dir / "rolling_cv_horizon_segments.csv", index=False)
    cv_groups.to_csv(args.out_dir / "rolling_cv_business_groups.csv", index=False)
    schema.to_csv(args.out_dir / "aggregate_feature_schema.csv", index=False)
    feature_summary.to_csv(args.out_dir / "aggregate_feature_pilot_summary.csv", index=False)
    feature_sample.to_csv(args.out_dir / "aggregate_feature_sample.csv", index=False)
    jd.to_csv(args.out_dir / "jd_capability_matrix.csv", index=False)
    gaps.to_csv(args.out_dir / "jd_gap_list.csv", index=False)
    scorecard.to_csv(args.out_dir / "forecast_governance_scorecard.csv", index=False)
    (args.out_dir / "job_alignment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
