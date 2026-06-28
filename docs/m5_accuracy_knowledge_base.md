# M5 Accuracy 知识库

## 资料来源

- M5 官方方法仓库：<https://github.com/Mcompetitions/M5-methods>
- M5 Accuracy 结果论文 PDF：<https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf>
- 第一名方法论文摘要：<https://ideas.repec.org/a/eee/intfor/v38y2022i4p1386-1399.html>
- 第二名方法论文摘要：<https://ideas.repec.org/a/eee/intfor/v38y2022i4p1405-1414.html>
- 本项目 notebooks 中引用的 Kaggle kernels：`kyakovlev/m5-simple-fe`、`kyakovlev/m5-lags-features`、`kyakovlev/m5-custom-features`。

## 竞赛问题

M5 Accuracy 要预测 Walmart 层级销量数据的 28 天点预测，评价对象不是只有 30,490 条商品-门店底层序列，还包括聚合后的 12 个层级，共 42,840 条序列。

评价指标是 WRMSSE。它先把预测从底层聚合到 12 个层级，再对每条序列计算 RMSSE，最后按近期销售额权重加权平均。

这意味着单纯把每个商品-门店预测得好还不够；如果总量、州、门店、品类等高层级偏差明显，最终分数也会被拉低。

## 官方 Accuracy 排名坐标

下表来自官方 `Mcompetitions/M5-methods` 仓库的 `Scores and Ranks.xlsx`，我在本地读取了 `Accuracy-Top50 (AL)` 工作表。

| Rank | Team | Avg WRMSSE |
|---:|---|---:|
| 1 | YJ_STU | 0.520438 |
| 2 | Matthias | 0.528165 |
| 3 | mf | 0.535717 |
| 4 | monsaraida | 0.535832 |
| 5 | Alan Lahoud | 0.536046 |

官方基线中，`sNaive` 的 Avg WRMSSE 是 0.847017，`ES_bu` 的 Avg WRMSSE 是 0.670981。

## 前几名方法经验

### 1. YJ_STU

官方论文描述第一名使用多个 LightGBM 模型的等权平均。模型池包括按门店、门店-品类、门店-部门训练的模型，并包含 recursive 与 non-recursive 两类多步预测方式。

官方论文描述该方案共构建 220 个模型，每条底层序列最终平均 6 个模型预测。

官方论文描述其训练目标使用 Tweedie 分布负对数似然，CV 使用最后四个 28 天窗口，并同时看误差均值与标准差来选择稳健组合。

第一名的核心不是单个复杂模型，而是：多粒度 partial pooling、recursive/direct 互补、稳定 CV、简单平均。

### 2. Matthias

官方论文和第二名论文摘要描述该方案用 LightGBM 预测底层间歇序列，用 N-BEATS 预测高层连续序列，然后通过 top-down alignment 调整底层预测，使高层级更协调。

官方论文描述其 LightGBM 按门店训练，并用 5 个 multiplier 调整趋势，共 50 个模型；使用自定义非对称损失。

第二名的关键经验是：底层局部误差最小不等于整体 WRMSSE 最小，高层趋势校准可以显著改善整体分数。

### 3. mf

官方论文描述第三名使用 43 个深度学习 LSTM 模型等权平均，递归预测底层序列，并使用较长的历史 CV 窗口选择模型实例。

第三名证明神经网络可以竞争，但其成功依赖大规模模型集成、特征工程和严格 CV，不是简单套一个 LSTM。

## 可迁移最佳实践

M5 的强基线不是 ARIMA 类单序列模型，而是全局模型或分组全局模型，例如按门店、品类、部门共享信息的 LightGBM。

最重要的特征族是：历史销量 lag、rolling mean/std、价格、价格变化、日历、节假日、SNAP、商品/门店/品类标识、目标编码、缺货或真实零销量识别。

最值得优先补的工程能力是本地 WRMSSE 与 rolling CV。没有本地指标，就只能猜 leaderboard。

优化方向应从单模型调参升级到组合：direct + recursive、不同 pooling 粒度、不同损失函数、不同随机种子、不同训练窗口。

对业务可解释性来说，门店级、部门级、品类级误差拆解比只看总分更有价值，因为它能定位模型是在高层趋势、部门结构还是底层 SKU 上出问题。

## 其他 GitHub/Kaggle 项目观察

`Mcompetitions/M5-methods` 是最权威的 GitHub 仓库，因为它包含获胜方法代码、官方分数表、基线预测、评估代码和论文材料。链接：<https://github.com/Mcompetitions/M5-methods>

`kyakovlev/m5-three-shades-of-dark-darker-magic` 是本项目主 notebook 的直接来源之一，Kaggle 页面显示它依赖 `M5 - Simple FE`、`M5 - Custom features` 和 `M5 - Lags features`。链接：<https://www.kaggle.com/code/kyakovlev/m5-three-shades-of-dark-darker-magic>

`btrotta/kaggle-m5` 的 README 描述其为 top 4% 解法，特点是代码短、LightGBM、基础特征、不使用复杂调整因子。链接：<https://github.com/btrotta/kaggle-m5>

`cnicault/m5-forecasting-accuracy` 的 README 描述其公开分数 0.48734、私榜分数 0.62408、排名约 190/5558，属于 top 4%。链接：<https://github.com/cnicault/m5-forecasting-accuracy>

`NaquibAlam/M5_Forecasting_Accuracy_kaggle` 的 README 描述了按 10 个门店和 4 个预测周建 40 个模型，并使用基础、价格、日历、lag/rolling、target encoding 特征和 LightGBM Tweedie。链接：<https://github.com/NaquibAlam/M5_Forecasting_Accuracy_kaggle>

这些项目的共同经验是：M5 不需要一开始就上很重的深度学习，先把 LightGBM、特征流水线、本地指标、CV 和提交生成做稳，就能达到较强水平。

## 与本项目的关系

本项目主体 notebook 与 Konstantin Yakovlev 的 M5 系列 Kaggle kernels 高度一致：`m5-simple-fe` 做基础 grid、价格和日历特征；`m5-lags-features` 做 lag/rolling；`m5-custom-features` 做 mean encoding、PCA、permutation importance；`m5-three-shades-of-dark-darker-magic` 做分门店 LightGBM 和递归预测。

当前项目是一个很好的 M5 工程复现底座，但它还不是第一名方案，因为它主要是 10 个门店 LightGBM 模型，而不是 220 个多粒度 direct/recursive 模型组合，也没有 top-down alignment。

如果要向前几名靠近，下一阶段应先补 CV 与对比实验，而不是盲目加深模型。
