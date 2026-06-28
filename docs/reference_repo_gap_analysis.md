# 三个参考仓库对比与本项目优化缺口

## 参考仓库

- Matthias Anderer 第二名方案：<https://github.com/matthiasanderer/m5-accuracy-competition>
- btrotta top 4% LightGBM 方案：<https://github.com/btrotta/kaggle-m5>
- devmofl 第三名 DeepAR 方案：<https://github.com/devmofl/M5_Accuracy_3rd>

## 你的当前项目位置

当前项目已经能复现 10 个门店 LightGBM 模型的 28 天递归预测，并用本地 WRMSSE 评分。原模型本地验证期 Avg WRMSSE 为 `0.475241`；加入全局趋势 multiplier 后为 `0.471374`。

当前 `archive/test_*.pkl` 的模型特征主要包括商品/部门/品类 ID、价格特征、日历/事件/SNAP、目标编码、底层 `id` 的 lag/rolling 特征。

当前项目的强项是“工程复现 + 底层 LightGBM 强基线”；主要缺口是 direct 模型、多粒度聚合 lag、高层模型/层级校准、滚动 CV 模型池。

## Matthias 第二名方案

该仓库 README 明确说明它包含：特征预处理 notebook、bottom level train/predict notebook、top level N-BEATS notebook、bottom/top alignment notebook，以及修改过的 GluonTS 包。

bottom level 部分需要对多个 `LOSS_MULTIPLIER` 运行，README 给出的最终 ensemble multiplier 为 `0.9, 0.93, 0.95, 0.97, 0.99`。

top level 部分用 N-BEATS 预测 L1-L5，并要求 CUDA/GPU 环境。

我本地解析 `m5-final-XX.ipynb` 看到它定义了 `custom_asymmetric_train` / `custom_asymmetric_valid`，通过 `LOSS_MULTIPLIER` 对欠预测和过预测施加不同惩罚。

我本地解析 `m5-alignandsubmit.ipynb` 看到它先平均多个 bottom-level submission，再将底层预测聚合到 L1-L5，与两份 N-BEATS top-level 预测进行对比/校准。

### 对你的启发

最值得借鉴的是“模型池 + 误差方向控制 + 高层校准”。你现在只有单一 LightGBM 预测和一个全局趋势 multiplier，尚未形成多个 loss/multiplier 版本的 ensemble。

最短路径不是马上复刻 N-BEATS，而是先做一个自动化高层校准脚本：把你的底层预测聚合到 L1-L5，和一个简单高层统计/LightGBM/N-BEATS-lite 模型对齐，验证 WRMSSE 是否下降。

自定义 asymmetric loss 可能改善系统性低估/高估，但必须用 rolling CV 选 multiplier；否则很容易在单个验证窗口过拟合。

## btrotta LightGBM 方案

该 README 说明其使用短代码、基础特征和 LightGBM，不使用 magic adjustment factor，也不优化自定义指标。

该 README 明确说它不是 recursive，而是为预测 horizon 的每一天训练 separate models，并对第 `n` 天 ahead 重新计算 lagged features。

该 README 说明它排除商品-门店首次销售前的训练行，用 3 年数据算特征、1 年数据训练，并排除 12 月训练样本。

该 README 列出聚合特征层级：item-store、item 全门店聚合、dept-store；还包含商品级 holiday adjustment、长期均值/方差、按星期的长期均值/方差、7/14/28 天平均、1-7 天 lag。

我本地读 `predict.py` 看到它实现了 `days_since_first_sale`、价格相对 7/14 天均价差、event uplift、聚合层级 rolling/lag，以及训练时排除 `month == 12`。

### 对你的启发

这是当前最值得优先借的仓库，因为它和你的 LightGBM 路线一致、工程成本远低于 DeepAR/N-BEATS。

你的当前缺口包括：没有 direct horizon 模型；缺少 item 全门店、dept-store 等聚合 lag/rolling；缺少 event uplift；未系统处理 12 月/Christmas 特殊训练噪声。

优先级最高的优化是先实现“btrotta-style aggregate lag features”，再实现“direct horizon pilot”。这两项比上深度学习更可能以低成本提升。

## devmofl 第三名 DeepAR 方案

该 README 说明方案为第三名模型，基于 PyTorchTS；复现需要 Python 3.7.4、CUDA 10.1、CUDNN 7.6.5、PyTorch 1.4。

该 README 说明训练 8 个模型，每个约 6 小时；每个训练模型还会生成 14 个预测 epoch，每个约 1 小时；最终再做 ensemble。

我本地读 `training.py` 看到模型是 `DeepAREstimator`，使用 TweedieOutput(1.2)、静态类别、动态实值、动态类别、移动平均窗口 `[7, 28]`、`lags_seq=[1]`，训练 300 epoch。

我本地读 `load_dataset.py` 看到特征包括 SNAP、价格、按 item 标准化价格、按 department 标准化价格、event type/name 动态类别、item/dept/cat/store/state 静态类别，以及 zero-sale period 动态 past 特征。

我本地读 `ensemble.py` 看到它会在多个 CV 起点上计算 WRMSSE，选择每个 trial 的 top-K epoch，再对 trial/epoch/prediction period 求均值。

### 对你的启发

DeepAR 全量复刻不适合作为你下一步短期优化，因为环境老、GPU 成本高、训练时间长。

但它的两个特征思想很值得迁移到你的 LightGBM：`zero-sale period` 和价格标准化。前者能帮助区分间歇需求/长期断售，后者能让价格变化在不同商品和部门之间更可比。

如果你未来有 GPU，DeepAR/现代 Chronos/TimesFM/Moirai 应作为 ensemble diversity，而不是替代 LightGBM。

## 建议优化优先级

### P0：先补滚动 CV

三个仓库都在不同形式上依赖多窗口验证或多模型选择；你的当前项目仍主要依赖一个本地验证窗口。下一步必须建立 4 个以上 28 天 rolling CV，否则 multiplier、direct 模型、聚合特征都无法可靠比较。

### P1：实现聚合 lag/rolling 特征

参考 btrotta，新增三组聚合视角：`item_id + store_id`、`item_id`、`dept_id + store_id`。在每组上生成 `lag_1..7`、`rolling_mean_7/14/28`、长期 mean/var、星期季节性 mean/var。

这能补足你当前只强依赖底层 id lag/rolling 的信息缺口，让模型看到“同商品跨门店”和“同部门同门店”的更低噪声趋势。

### P2：做 direct horizon pilot

参考 btrotta，先不要一次训练 28 个完整模型；可以做 4 个分段 direct 模型：F1-F7、F8-F14、F15-F21、F22-F28。与当前 recursive 预测做 CV ensemble。

这能直接测试 btrotta 对 recursive 误差传播的担忧是否也存在于你的项目。

### P3：做 loss/multiplier ensemble

参考 Matthias，训练 3-5 个不同欠预测惩罚版本或后处理 multiplier 版本，然后用 CV 选择平均权重。你已经有一个全局趋势 multiplier，小幅改善了高层级，说明这个方向有信号。

### P4：做高层模型和 alignment

先做简单版：对 L1-L5 训练轻量模型或统计趋势模型，生成高层目标预测；再把底层预测按高层比例缩放。只有当 CV 稳定提升后，再考虑 N-BEATS/现代 foundation model。

### P5：迁移 DeepAR 特征，不急着迁移 DeepAR

优先迁移 `zero-sale period`、按 item/dept 标准化价格、动态事件编码；暂缓复刻 8-trial DeepAR。

## 结论

你的项目还有明显优化空间，但最优下一步不是“换最新模型”，而是把 btrotta 的 direct/aggregate-lag 思路和 Matthias 的高层校准思路接到你现有 LightGBM 体系里。

如果只能选一个方向，先做 `aggregate lag/rolling + rolling CV`；如果选两个，再加 `direct horizon pilot`。
