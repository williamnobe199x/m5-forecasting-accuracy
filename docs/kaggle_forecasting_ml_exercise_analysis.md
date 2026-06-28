# Ryan Holbrook Forecasting With Machine Learning 练习对 M5 项目的启发

## 资料来源

- Kaggle notebook：<https://www.kaggle.com/code/ryanholbrook/exercise-forecasting-with-machine-learning>
- 本地副本：`exercise-forecasting-with-machine-learning.ipynb`
- GitHub 镜像 notebook：<https://raw.githubusercontent.com/drakearch/kaggle-courses/master/time_series/06-forecasting-with-machine-learning.ipynb>

## 这个 notebook 讲的不是 M5 trick，而是多步预测框架

该 notebook 是 Kaggle Learn Time Series 课程的练习，主题是把时间序列转成监督学习问题，并练习 multi-step forecasting。

练习里的 Store Sales 任务是 16-step forecast、1-step lead time；它用 `make_lags` 生成 lag 特征，用 `make_multistep_target` 生成多步目标。

练习最后让用户用 `RegressorChain` 实现 DirRec strategy，并使用 `XGBRegressor` 作为基础模型。

它对你的 M5 项目最大的价值不是直接复制模型，而是提醒我们：M5 的 28 天预测不应该只依赖 recursive 一种策略，应该至少做 direct / DirRec / recursive 的对比与 ensemble。

## 和你当前 M5 项目的差距

你的当前主流程是：10 个门店 LightGBM 模型 + 递归预测 28 天。预测第 `d_1915` 以后，会使用前面预测出来的销量参与 rolling 特征。

递归预测的优点是模型少、能持续更新短期 lag；缺点是误差会沿 horizon 传播。这个缺点正是 Ryan Holbrook 练习里 multi-step strategy 要处理的问题。

你现在还缺两类实验：

1. **Direct**：为不同 horizon 或 horizon 分段训练独立模型，避免预测值递归进入后续特征。
2. **DirRec**：第 k 步模型允许使用前面 1..k-1 步的预测作为额外特征，介于 direct 和 recursive 之间。

## 能带来的优化空间

### 1. 先做分段 direct，而不是一次做 28 个模型

M5 全量训练成本较高。建议先做 4 段 direct pilot：

- F1-F7：近端一周；
- F8-F14：第二周；
- F15-F21：第三周；
- F22-F28：第四周。

每段模型的训练目标可以是该段销量均值、该段逐日多输出，或该段内 7 个独立 LightGBM。先用 rolling CV 比较，再决定是否扩展到 28 个 horizon-specific 模型。

### 2. 用 DirRec 思想做轻量增强

对第 2-4 周预测，可以把当前 recursive 模型对前一周的预测聚合值作为额外特征，例如 `pred_week_1_sum`、`pred_week_1_mean`、`pred_week_1_weekend_sum`。

这比完整 `RegressorChain` 更容易接入现有 LightGBM，因为它不需要把 28 个预测逐列塞回训练表，只需要把早期预测摘要化。

### 3. 用 multi-output 思想重构验证口径

当前评分已经按 28 天整体 WRMSSE 做。下一步应该在 rolling CV 中额外保存 per-horizon / per-week WRMSSE，判断模型到底是第 1 周强、第 4 周弱，还是全部 horizon 同步偏差。

### 4. 不建议直接照搬 XGBRegressor

练习里用 `XGBRegressor` 是教学方便。你的项目已经有完整 LightGBM 特征和模型产物，直接换成 XGBoost 不一定收益更高，且训练成本更大。更合理的是保留 LightGBM，先改多步预测策略。

## 优先级建议

1. 建立 rolling CV，并输出每周、每层级 WRMSSE。
2. 做 F1-F7 / F8-F14 / F15-F21 / F22-F28 四段 direct LightGBM pilot。
3. 将 recursive 预测的一周摘要作为 DirRec 特征，测试是否改善第 2-4 周。
4. 如果 direct/DirRec 与 recursive 的误差互补，再做简单平均或 CV 加权 ensemble。

## 结论

有进一步优化空间，而且方向非常明确：不是从这个教学 notebook 复制代码，而是把它的 multi-step forecasting 思想落到 M5 上。

当前最值得新增的实验是 **direct / DirRec 与现有 recursive 的对比**。这也和前面 btrotta 仓库的结论一致：你的下一步优化重点应从“特征工程 + 单递归模型”升级到“多步预测策略组合”。
