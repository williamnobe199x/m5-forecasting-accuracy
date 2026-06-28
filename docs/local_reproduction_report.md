# 本地复现报告与逐段讲解

## 本地项目状态

当前目录不是 Git 仓库，`git status` 返回 `fatal: not a git repository`。

当前目录包含 M5 原始数据 `calendar.csv`、`sell_prices.csv`、`sales_train_validation.csv`、`sales_train_evaluation.csv`、`sample_submission.csv`。

当前目录包含 3 个预生成特征 pickle：`grid_part_1.pkl`、`grid_part_2.pkl`、`grid_part_3.pkl`。其中 `grid_part_1.pkl` 在当前 Python 环境下需要 `pyarrow` 才能反序列化。

`archive` 目录包含 10 个门店测试特征 pickle 和 10 个 LightGBM 模型 pickle，合计约 1.32 GB。

当前环境已安装 `pandas`、`numpy`、`lightgbm`、`sklearn`、`matplotlib`、`seaborn`、`nbformat`、`nbconvert`；未安装 `pyarrow`、`xgboost`、`openpyxl`。

## VSCode 复现方式

我新增了 `requirements.txt` 和 `.vscode/tasks.json`。

在 VSCode 中打开本项目文件夹后，可以运行：

```powershell
python -m pip install -r requirements.txt
```

然后用命令面板运行：

- `Tasks: Run Task` -> `M5: reproduce archived prediction`
- `Tasks: Run Task` -> `M5: score archived prediction`

也可以在 VSCode 终端直接运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts\reproduce_archive_prediction.py --archive-dir archive --sample-submission sample_submission.csv --predictions-out output\archive_validation_predictions.csv --submission-out output\submission_v1_local.csv --clip-zero
python scripts\score_m5_wrmsse.py --predictions output\archive_validation_predictions.csv --details-out output\wrmsse_by_level.csv
```

## 我实际跑出的结果

归档预测脚本完成运行，用时约 23.78 分钟。

生成文件：

- `output/archive_validation_predictions.csv`，形状为 30,490 行 x 29 列。
- `output/submission_v1_local.csv`，形状为 60,980 行 x 29 列。

本地验证期 WRMSSE 结果：

| Level | n_series | WRMSSE |
|---|---:|---:|
| L1_Total | 1 | 0.213365 |
| L2_State | 3 | 0.298353 |
| L3_Store | 10 | 0.391592 |
| L4_Category | 3 | 0.259211 |
| L5_Department | 7 | 0.334777 |
| L6_State_Category | 9 | 0.357300 |
| L7_State_Department | 21 | 0.432905 |
| L8_Store_Category | 30 | 0.455692 |
| L9_Store_Department | 70 | 0.539849 |
| L10_Item | 3,049 | 0.794566 |
| L11_Item_State | 9,147 | 0.808754 |
| L12_Item_Store | 30,490 | 0.816533 |
| Average | 42,840 | 0.475241 |

我用同一个评分器算了季节朴素基线，即用 `d_1886..d_1913` 预测 `d_1914..d_1941`，本地 Avg WRMSSE 是 0.837726。

归档模型相对本地 sNaive 基线提升明显。

本地 0.475241 不能直接说超过官方第一名 0.520438，因为这里算的是可见的 `d_1914..d_1941` 验证期；官方排名表对应竞赛最终口径。

## 当前项目逐段讲解

### 1. `m5-simple-fe.ipynb`

这个 notebook 的核心是把 M5 的宽表销量数据转成长表 grid。原始 `sales_train_validation.csv` 是一行一个商品-门店，`d_1..d_1913` 横向展开；模型训练更适合长表，也就是一行代表一个 `id` 在某一天 `d` 的状态。

它随后加入商品上市周 `release`，并把上市前的结构性零销量过滤掉。这个步骤很重要，因为上市前的 0 不是需求为 0，而是商品还不存在。

它继续合并价格表并构造价格特征，例如最大价、最小价、均价、价格标准差、价格归一化、价格变化 momentum。

它最后合并日历特征，包括星期、月份、年份、周末、事件、SNAP 等，并保存为 `grid_part_1.pkl`、`grid_part_2.pkl`、`grid_part_3.pkl`。

### 2. `m5-lags-features.ipynb`

这个 notebook 解释并构造 lag 特征。M5 是 28 天预测，所以常见安全 lag 从 `sales_lag_28` 开始，避免把未来真实销量泄漏到预测日。

它还构造 rolling 特征，例如过去 7、14、30、60、180 天的 rolling mean/std。

这些特征是 LightGBM 在时间序列里能工作的关键，因为树模型本身不知道时间顺序，必须通过 lag 和 rolling 把时间结构显式喂进去。

### 3. `m5-custom-features.ipynb`

这个 notebook 测试额外特征，包括目标均值编码、PCA、permutation importance、距离上次非零销量的特征。

它的思想不是一次性堆特征，而是每加一类特征就验证是否有提升，这和 M5 前几名强调的稳定 CV 方向一致。

### 4. `m5-three-shades-of-dark-darker-magic.ipynb`

这个 notebook 是主训练和预测入口。它按 10 个门店分别训练 LightGBM，目标函数使用 Tweedie，特征包含商品/品类/部门 ID、价格、日历、mean encoding、lag、rolling。

`USE_AUX=True` 时，代码会使用 `archive` 里的预训练模型做预测；本地复现脚本就是把这一部分整理成了可重复执行的 Python 入口。

预测阶段是递归式的：先预测 `d_1914`，把预测值写回 `base_test['sales']`，再用它参与 `d_1915` 的 rolling 特征，以此滚动到 28 天。

递归预测的优点是模型少、逻辑自然；缺点是前几天误差会进入后续 lag/rolling，形成误差传播。

## 我发现的主要问题

notebook 中有多处 Mac 绝对路径，例如 `/Users/gaoguozheng/Downloads/...`，在当前 Windows 工作区不能直接运行。

主 notebook 的导出单元里有 `ORIGINAL+'sample_submission.csv'`，而 `ORIGINAL` 没有尾部斜杠；这个表达式会拼成错误路径。新增脚本已修正为明确传入 `sample_submission.csv`。

当前工作区没有 `lags_df_28.pkl` 和 `mean_encoding_df.pkl`，所以如果从头重跑主训练 notebook，需要先跑完对应特征生成并修正保存路径。

当前 Python 控制台默认编码无法输出路径里的 emoji 字符；我在 VSCode 任务里设置了 `PYTHONIOENCODING=utf-8`。

当前项目更像 Kaggle kernel 复现包，而不是完整工程项目；缺少 README、依赖文件、统一脚本入口、本地评分器和 CV 实验记录。现在已补上其中一部分。

## 与主流结果对比

本地归档模型 Avg WRMSSE：0.475241。

本地 sNaive Avg WRMSSE：0.837726。

官方最终榜前 5 名 Avg WRMSSE：YJ_STU 0.520438，Matthias 0.528165，mf 0.535717，monsaraida 0.535832，Alan Lahoud 0.536046。

你的项目在本地验证期表现强于简单基线很多，但它不应被表述为官方最终排名级别的成绩，除非提交到同一评测口径或用官方 evaluation 文件完全复刻最终私榜口径。

从层级拆解看，模型在高层级表现更强，底层商品-门店层级 WRMSSE 较高。优化不应只看总分，要重点看 L10、L11、L12 的底层误差，同时防止高层总量被破坏。

## 优化路线

### 第 1 步：把实验口径固定

先把本地评分器作为所有实验的统一入口。每次改模型，都输出 `wrmsse_by_level.csv`，比较 12 个层级，而不是只看一个总分。

建议建立 rolling CV：至少使用最后 4 个 28 天窗口，例如 `d_1802..1829`、`d_1830..1857`、`d_1858..1885`、`d_1886..1913` 作为验证窗口。

### 第 2 步：复现完整特征流水线

把 `m5-simple-fe`、`m5-lags-features`、`m5-custom-features` 改成相对路径脚本，确保从 CSV 能重新生成全部 pickle。

完整复现后再训练模型，才能判断当前 `archive` 模型是否和 notebook 代码完全一致。

### 第 3 步：补 direct 模型

当前预测是 recursive。参考第一名经验，应增加 non-recursive/direct 模型，即为不同预测日或预测周训练不同模型，然后与 recursive 简单平均。

预期收益来自误差互补：recursive 擅长利用近端动态，direct 避免误差滚动传播。

### 第 4 步：补多粒度 pooling

当前主要是按门店 10 个模型。参考第一名经验，可以增加门店-品类 30 个模型、门店-部门 70 个模型，再做等权或 CV 加权平均。

多粒度模型可以让不同品类共享模式，同时保留门店差异。

### 第 5 步：做趋势和层级校准

如果低层预测聚合后高层趋势偏低或偏高，可以参考第二名方法，单独建高层模型或简单统计趋势模型，然后对底层预测乘 multiplier 校准。

先从简单 multiplier 开始比直接上 N-BEATS 更稳，因为它更容易验证效果来源。

### 第 6 步：处理真实零销量和缺货

M5 的零销量有多种含义：真实无需求、未上市、断货、低频间歇需求。把这些混在一个回归目标里会增加噪声。

可以先训练一个二分类模型预测未来是否非零，再用回归模型预测非零销量大小，最后组合为期望销量。

## 学习路径

1. 先读 `m5-simple-fe.ipynb`，目标是理解宽表转长表、价格表、日历表如何合并。
2. 再读 `m5-lags-features.ipynb`，目标是理解为什么 lag 从 28 开始，以及 rolling 特征如何避免泄漏。
3. 然后跑 `scripts/reproduce_archive_prediction.py`，目标是理解递归预测如何把预测值写回未来特征。
4. 接着读 `scripts/score_m5_wrmsse.py`，目标是理解 WRMSSE 为什么要聚合 12 个层级。
5. 最后做一个小实验：只改一个特征或一个参数，重训一个门店模型，用本地评分器看它对 12 个层级的影响。
