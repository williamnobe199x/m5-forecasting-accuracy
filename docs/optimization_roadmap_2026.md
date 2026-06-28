# M5 Accuracy 优化路线：2026 版

## 已落地优化

[COMPUTED, HIGH] 我新增了一个趋势 multiplier 后处理脚本：`scripts/apply_trend_multiplier.py`。

[COMPUTED, HIGH] 该脚本只使用 `d_1913` 之前的历史数据，计算 `sum(d_1886..d_1913) / sum(d_1858..d_1885)`，并对预测做全局保守缩放。

[COMPUTED, HIGH] 本地验证期结果如下：

| Version | Avg WRMSSE | 相对 sNaive 提升 | 说明 |
|---|---:|---:|---|
| sNaive | 0.837726 | 0.00% | 季节朴素基线 |
| 原 LightGBM 归档模型 | 0.475241 | 43.27% | 10 个门店模型，递归预测 |
| LightGBM + 全局趋势 multiplier | 0.471374 | 43.73% | 乘数约 1.0064 |

[INFERRED, HIGH] 这个优化主要改善高层级总量与品类层级，底层 `L12_Item_Store` 基本没有改善；它适合作为趋势校准思路的最小可行版本。

[INFERRED, HIGH] 该 multiplier 不是“最终最优策略”，而是一个可复现、无未来真实值泄漏的第一步。正式使用前应放入 rolling CV。

## 现在预测技术的更新

[KNOWN, HIGH] 2024-2026 年，时间序列 foundation model 明显增多。Amazon Chronos 仓库在 2025 年发布 Chronos-2，并描述其支持 univariate、multivariate 和 covariate-informed forecasting；Chronos-Bolt 相对原 Chronos 更快且更省内存。来源：<https://github.com/amazon-science/chronos-forecasting>

[KNOWN, HIGH] Google Research 的 TimesFM 是预训练时间序列基础模型，README 显示 TimesFM 2.5 支持更长 context、quantile forecast，并在 2025-2026 增加 XReg covariate、LoRA fine-tuning 等能力。来源：<https://github.com/google-research/timesfm>

[KNOWN, HIGH] Salesforce 的 Uni2TS/Moirai 项目在 2024-2025 发布 Moirai、Moirai-MoE 和 Moirai-2.0，并提供 zero-shot forecast、rolling evaluation 示例。来源：<https://github.com/SalesforceAIResearch/uni2ts>

[KNOWN, HIGH] Lag-Llama 项目将自己描述为开源时间序列 foundation model，支持 zero-shot 和初步 fine-tuning。来源：<https://github.com/time-series-foundation-models/lag-llama>

[KNOWN, HIGH] Nixtla `hierarchicalforecast` 提供 BottomUp、TopDown、MiddleOut、MinTrace、ERM 等层级预测 reconciliation 方法。来源：<https://github.com/Nixtla/hierarchicalforecast>

## 对 M5 的技术判断

[INFERRED, HIGH] 对 M5 这种 30,490 条底层零售序列、12 层聚合、价格/日历/事件/SNAP 协变量丰富的任务，最新 foundation model 不应直接替换 LightGBM，而应作为候选模型加入 ensemble。

[INFERRED, HIGH] 最优先的升级顺序是：

1. rolling CV 固定口径；
2. direct + recursive LightGBM 组合；
3. 多粒度模型：store、store-category、store-department；
4. trend multiplier 和层级 reconciliation；
5. 再加入 Chronos-2 / TimesFM / Moirai 的抽样实验；
6. 只保留在 rolling CV 上稳定增益的模型进入 ensemble。

[INFERRED, MED] Chronos-2 和 TimesFM 对 M5 的潜在价值主要是补充底层间歇序列以外的趋势/周期视角；但它们未必天然优于带强特征工程的 LightGBM，因为 M5 的价格、节日、SNAP、门店/商品 ID 等表格特征非常关键。

## 下一步可执行实验

### 实验 A：滚动 CV

[INFERRED, HIGH] 建立 4 折 rolling CV：每折 28 天 horizon，训练截止点依次向前移动。所有模型、乘数、blend 权重都只根据 CV 均值和稳定性选择。

### 实验 B：direct 模型

[INFERRED, HIGH] 在现有 recursive 预测之外，为 horizon 第 1-7、8-14、15-21、22-28 天分别训练 direct 模型，然后与 recursive 输出平均。

### 实验 C：多粒度 LightGBM

[INFERRED, HIGH] 参考 M5 第一名经验，增加 `store_id + cat_id`、`store_id + dept_id` 两组模型。每条底层序列获得多个模型预测后，先等权平均，再用 CV 决定权重。

### 实验 D：层级 reconciliation

[INFERRED, HIGH] 先用 `hierarchicalforecast` 的 BottomUp / TopDown / MinTrace 在验证窗口试验。目标不是让某一层极优，而是让 12 层平均 WRMSSE 更稳。

### 实验 E：foundation model 抽样试验

[INFERRED, MED] 先抽样 500-2,000 条代表性序列跑 Chronos-2 或 TimesFM，而不是一次性跑全量 30,490 条。用相同 WRMSSE 子集指标判断是否值得投入 GPU/云资源。

## 对你当前项目的优化结论

[INFERRED, HIGH] 你的当前项目已经完成了一个强 LightGBM 工程复现底座，边际优化不应继续只改单个参数。下一阶段真正能拉开差距的是“实验体系”：CV、模型池、层级一致性和可解释报告。

[COMPUTED, HIGH] 我已经补齐了：复现脚本、评分脚本、趋势优化脚本、贡献分析脚本、VSCode task 和 HTML 展示页生成能力。
