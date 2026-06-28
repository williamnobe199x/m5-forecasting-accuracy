# M5 Forecasting Accuracy 项目复盘与优化

这个项目复现并整理了 M5 Forecasting Accuracy 的本地验证流程：从 archived LightGBM 预测、M5-style WRMSSE 评分、后处理优化候选扫描，到交互式 HTML 项目展示页。

## 当前结果

- [COMPUTED, HIGH] 本地验证窗口为 `d_1914..d_1941`，与 Kaggle 官方 private/public leaderboard 不是同一个切片。
- [COMPUTED, HIGH] sNaive 基线 Avg WRMSSE: `0.837726`。
- [COMPUTED, HIGH] archived LightGBM 复现 Avg WRMSSE: `0.475241`。
- [COMPUTED, HIGH] 优化后 Avg WRMSSE: `0.471374`。
- [COMPUTED, HIGH] 优化后相对 sNaive 提升: `43.73%`。
- [INFERRED, HIGH] 候选扫描显示，当前最稳的可落地优化是全局趋势 multiplier；更细的分组 multiplier 和 horizon multiplier 在单窗口验证中没有提升。

## 主要产物

- `scripts/score_m5_wrmsse.py`: 本地 WRMSSE 评分。
- `scripts/optimize_forecast_candidates.py`: 候选后处理优化扫描。
- `scripts/analyze_project_contribution.py`: 项目贡献与误差诊断汇总。
- `scripts/build_visual_assets.py`: EDA 和结果可视化小表生成。
- `scripts/build_project_showcase.py`: 交互式 HTML 展示页生成。
- `scripts/run_full_pipeline.py`: 刷新优化、分析和展示页的一键脚本。
- `docs/index.html`: GitHub Pages 入口。
- `docs/m5_project_showcase.html`: 项目展示页。

## 复现

先把 M5 原始数据放在项目根目录：

- `sales_train_evaluation.csv`
- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`
- `sample_submission.csv`

然后运行：

```bash
pip install -r requirements.txt
python scripts/run_full_pipeline.py
```

[COMPUTED, HIGH] `run_full_pipeline.py` 会刷新优化候选、贡献分析、可视化资产和 HTML 展示页。它默认使用已经存在的 `output/archive_validation_predictions.csv` 与 `output/snaive_validation_predictions.csv`。

如果你要从 archived 模型重新生成 base prediction，需要本地保留 `archive/` 下的 LightGBM 模型和测试特征，再先运行：

```bash
python scripts/reproduce_archive_prediction.py
python scripts/score_m5_wrmsse.py --predictions output/archive_validation_predictions.csv --details-out output/wrmsse_by_level.csv
```

本地查看展示页：

```bash
python -m http.server 8765 --bind 127.0.0.1
```

打开 `http://127.0.0.1:8765/docs/index.html`。

## 下一步优化路线

- [INFERRED, HIGH] 建立 rolling CV，把 multiplier、blend、direct 模型和新特征放到同一验证框架里。
- [INFERRED, HIGH] 增加 btrotta-style 聚合 lag/rolling 特征：`item_id`、`item_id + store_id`、`dept_id + store_id`。
- [INFERRED, HIGH] 增加 4 段 direct horizon LightGBM，与当前 recursive 预测做 ensemble。
- [INFERRED, MED] 在 CV 稳定后再引入层级 reconciliation 和时间序列 foundation model 抽样实验。

## 数据与模型文件

原始 CSV、pickle 特征表和 LightGBM 二进制模型体积较大，不建议直接提交到 GitHub。详见 `DATA.md`。
