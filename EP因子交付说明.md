# EP / PE 因子计算交付说明

## 1. 交付内容

交付目录：`D:\因子计算\市盈率\EP因子_交付版`

```text
EP因子_交付版
├─ 代码
│  ├─ step1_样本筛选.py
│  ├─ step2_利润表处理.py
│  ├─ step3_TTM计算.py
│  ├─ step4_点时间对齐_计算PE.py
│  ├─ step5_MAD去极值_Z标准化.py
│  ├─ pure_factor_streaming.py
│  ├─ bse_code_mapping.py
│  ├─ plot_ep_pe分布.py
│  ├─ plot_industry_pe.py
│  ├─ cross_section_rlm.py
│  ├─ cross_section_ols.py
│  ├─ EP因子交付验证.py
│  └─ factor_verify.py
├─ 因子结果
│  ├─ ep.parquet
│  └─ pe.parquet
├─ 中间结果
│  ├─ ep_raw.parquet
│  ├─ pe_raw.parquet
│  ├─ ep_mad.parquet / ep.parquet / ep_processing_stats.parquet
│  ├─ pe_mad.parquet / pe.parquet / pe_processing_stats.parquet
│  ├─ _mask.parquet
│  ├─ _mkt_clean.parquet
│  ├─ b_share_codes.parquet
│  ├─ bse_code_mapping.parquet / bse_code_mapping.csv
│  ├─ factor_audit.parquet
│  ├─ factor_daily_coverage.parquet / factor_daily_coverage.csv
│  ├─ factor_test_start.json
│  ├─ industry_pe_latest.csv
│  ├─ point_in_time_field_audit.parquet / point_in_time_field_audit.csv
│  ├─ reports.parquet
│  ├─ reports_effective.parquet
│  ├─ report_version_decisions.parquet
│  └─ reports_ttm.parquet
├─ 图形
│  ├─ ep_原始分布.png
│  ├─ pe_分布.png
│  ├─ industry_pe_latest.png
│  └─ daily_return_distribution_20251229.png
├─ 标准化统计
│  ├─ *.csv
│  └─ 图形
├─ 回归结果
│  ├─ ep                       ← 按因子分目录
│  │  ├─ RLM_H1_独立同方差
│  │  │  ├─ RLM统计结果.md
│  │  │  ├─ forward_monthly_returns.parquet
│  │  │  ├─ rlm_regression_sample.parquet
│  │  │  ├─ rlm_monthly_results.parquet / rlm_monthly_results.csv
│  │  │  └─ rlm_annual_summary.csv / rlm_overall_summary.csv
│  │  └─ OLS_月度
│  ├─ OLS_单日示例
│  └─ _历史口径存档
├─ EP_PE统计分析与因子测试计划.md
├─ EP_PE统计分析与因子测试计划.tex
├─ EP_PE统计分析与因子测试计划.pdf
├─ MAD去极值与标准化统计分析.md / .tex / .pdf
├─ 处理记录.md
├─ README.md
└─ EP因子交付说明.md
```

正式结果均使用：

```text
index = ['date', 'stock_code']
columns = ['signal']
```

- `因子结果\ep.parquet`：经过每日横截面 MAD 去极值和 Z 标准化的 EP。
- `因子结果\pe.parquet`：经过同样处理的 PE。PE 的原始值见 `中间结果\pe_raw.parquet`。
- `中间结果\ep_raw.parquet`：点时间对齐后的原始 EP，未做去极值或标准化。
- `中间结果\pe_raw.parquet`：点时间对齐后的原始 PE，未做去极值或标准化。
- `中间结果\factor_audit.parquet`：按 `date + stock_code` 保存行情代码、统一证券身份、财务连接代码、市值、所用财报版本、TTM、原始 EP/PE、MAD 值、Z 值及数据可用性标记的逐行追溯表。
- `中间结果\bse_code_mapping.*`：北交所 2025 年代码切换的 242 对新旧代码映射及价格、总股本匹配证据。
- `中间结果\factor_daily_coverage.*`：每日全股票池及非停牌股票池的 EP/PE 覆盖率。
- `中间结果\factor_test_start.json`：覆盖率稳定规则及正式测试起点。
- `中间结果\report_version_decisions.parquet`：每条财报版本的采用或忽略决策，旧公告回退记录明确标记为忽略。
- `中间结果\point_in_time_field_audit.*`：ST、上市日期和交易状态字段的来源、可用性及筛选状态。
- `EP_PE统计分析与因子测试计划.md`：便于实时预览和直接修改的 Markdown 主文档；公式采用兼容所有预览器的纯文本数学格式，行业图引用 `图形\industry_pe_latest.png`。
- `EP_PE统计分析与因子测试计划.tex/.pdf`：主文档的 XeLaTeX 源文件和编译版。
- `MAD去极值与标准化统计分析.md/.tex/.pdf`：MAD 去极值和 Z 标准化专项分析。
- `中间结果\industry_pe_latest.csv`：行业柱状图对应的统计明细。
- `图形\industry_pe_latest.png`：同口径的 PNG 预览图。
- `图形\行业PE柱状图.tex`：行业图脚本生成的 `pgfplots` 正文片段。
- `回归结果\ep\RLM_H1_独立同方差\RLM统计结果.md`：当前 EP 月度横截面 Huber RLM 的完整统计结果。

## 2. 输入数据

行情数据：

```text
D:\实习生学习项目\基础数据\chn_equ_mkt_quotation.parquet
```

使用字段：`date`、`stock_code`、`me_total`、`suspended`、`status`。

利润表数据：

```text
D:\实习生学习项目\基础数据\vw_fdmt_is_new.parquet
```

使用字段：`TICKER_SYMBOL`、`ACT_PUBTIME`、`UPDATE_TIME`、`END_DATE`、
`FISCAL_PERIOD`、`MERGED_FLAG`、`N_INCOME_ATTR_P`。

## 3. 样本和过滤口径

### 3.1 B 股过滤

Step 1 在行情表的原始 `stock_code` 上分别识别两类 B 股：

```text
上海 B 股：stock_code startswith '900' and endswith '.BJ'
深圳 B 股：stock_code startswith '200' or '201', and endswith '.SZ'
```

当前行情表识别出 89 只唯一 B 股，包括 50 只上海 B 股和 39 只深圳 B 股，
名单及分类保存于 `中间结果\b_share_codes.parquet`。这一步在代码统一为六位代码之前执行，
因此不会把交易所后缀信息丢失后再猜测股票类型。普通 `.BJ` 股票作为北交所样本保留。

除上述 B 股外，行情面板行全部保留。按全样本期原始行情代码去重，清洗后共有
5,927 个唯一代码标识，包括沪市 2,382 个、深市 3,006 个和北交所 539 个。
这里的 539 是代码标识数，不是北交所公司数，也不是被排除的股票数。

北交所 539 个代码标识由 251 个 `43/83/87xxxx.BJ` 历史代码和 288 个
`920xxx.BJ` 新代码组成。2025 年代码切换期间，同一证券的新旧代码会在全期去重统计中
重复出现：2025 年 9 月 30 日截面包含 242 个旧代码和 35 个 `920` 代码，
2025 年 10 月 9 日起当日存量证券全部使用 `920` 代码。2025 年 12 月 31 日行情截面
共有 288 只北交所股票，全部使用 `920xxx.BJ` 代码。

当前流程已生成 242 对一一映射：以 2025 年 9 月 30 日旧代码收盘价匹配 10 月 9 日
新代码前收盘价，222 对价格候选唯一，另外 20 对使用总股本消除同价歧义；原始价格和
复权价格均连续。该映射来自交付行情表内部证据，不是外部官方代码表。结果继续使用当日
行情代码，同时以 `security_id` 统一跨期证券身份，并以 `financial_code6` 将新代码行情
连接到旧代码财报历史。

### 3.2 ST、次新和停牌

- 不执行 ST/PT 股票筛选。行情表没有历史 ST 字段，`_mask.parquet` 和
  `factor_audit.parquet` 中以 `st_data_available = 0`、`st_filter_applied = 0` 留痕。
- 次新股只保留 `is_cixin` 标记用于审计，不用它把 EP 或 PE 置为 NaN。
- 只有行情表明确标记为停牌的日期会主动把 EP 和 PE 置为 NaN：
  `suspended == 1` 或 `status == '停牌'`。
- TTM 缺失、总市值缺失或非正等情况仍然会使 EP 自然为 NaN；这不是 ST 或次新筛选。

`_mkt_clean.parquet` 仍保留原始行情代码、统一证券身份、财务连接代码、交易所、市场类别、`status`、`suspended`、
`suspended_unknown`、`is_cixin`、`listing_date_known` 和 `first_seen_date`，便于下游复核。清洗后检查
`(date, code6)` 不重复。

## 4. 计算流程

### Step 1：行情清洗

1. 从原始行情表识别并排除 50 只 `900xxx.BJ` 上海 B 股和 39 只 `200/201xxx.SZ` 深圳 B 股。
2. 根据休市前后价格连续性和总股本生成 242 对北交所新旧代码映射。
3. 其余股票统一提取为六位 `code6`，检查日期和股票代码组合唯一。
4. 保留当日行情代码，并生成统一的 `security_id` 和用于连接财报的 `financial_code6`。
5. 根据同一交易日的 `suspended` 和 `status` 形成停牌标记。
6. 按统一证券身份计算 `first_seen_date` 和 `is_cixin`，代码切换不再被误记为新上市；该日期仍不是官方上市日期。
7. 输出 `point_in_time_field_audit.*`，记录 ST、上市日期和交易状态字段的可用性。

### Step 2：利润表处理

1. 在计算 TTM 前将新 `920` 代码财报映射到同一证券的旧财务代码，本次映射 972 条输入记录。
2. 只保留合并报表：`MERGED_FLAG == 1`。
3. 只保留累计口径：报告期月份等于 `FISCAL_PERIOD`。
4. 删除完全重复记录，保留不同修订版本。
5. 定义财报可用时间：

```text
VERSION_TIME = max(ACT_PUBTIME, UPDATE_TIME)
```

### Step 3：点时 TTM

按股票和 `VERSION_TIME` 顺序维护已经可获得的报告状态。同一报告期优先采用
`ACT_PUBTIME` 较新的公告；同一公告允许后续数据更新，但较旧公告即使具有更晚的
`UPDATE_TIME` 也不能覆盖已经生效的新公告。每条版本的决策保存在
`report_version_decisions.parquet`，实际采用的版本保存在 `reports_effective.parquet`。

年报：

```text
TTM = 当年累计归母净利润
```

非年报：

```text
TTM = 本期累计利润 + 上年年报利润 - 上年同期累计利润
```

报告组件缺失或在本期报告之后才可获得时，TTM 为 NaN，不向过去回填未来数据。

### Step 4：点时间 EP 和 PE

行情日 `T` 使用截至 `T` 日收盘后的可用财务信息，并用于 `T+1` 日交易。
切点为：

```text
signal_cutoff = T 日 + 1 天 - 1 纳秒
```

原始 EP：

```text
EP_T = T 日 TTM 归母净利润 / T 日总市值
```

原始 PE：

```text
PE_T = 1 / EP_T = T 日总市值 / T 日 TTM 归母净利润
```

处理约定：

- 负利润保留为负 EP，并对应保留负 PE；
- `EP == 0` 时 PE 没有定义，PE 置为 NaN；
- 总市值小于等于 0、TTM 缺失或停牌日，EP 和 PE 均为 NaN；
- `ep_raw.parquet` 和 `pe_raw.parquet` 是去极值之前的中间结果。
- `factor_audit.parquet` 将每个因子观测与实际采用的报告期、公告时间、版本时间、
  TTM、市值和当日交易状态直接关联，同时保留 `market_code`、`security_id`、
  `financial_code6` 和 `bse_code_mapped`。

### Step 5：MAD 去极值和 Z 标准化

EP 和 PE 分别按交易日横截面独立处理：

```text
MAD = median(abs(x - median(x)))
scale = 1.4826 * MAD
下限 = median(x) - 3 * scale
上限 = median(x) + 3 * scale
Z = (MAD处理后的值 - 当日均值) / 当日标准差
```

PE 在 EP 接近零时会出现很大的绝对值，原始 PE 应结合
`pe_processing_stats.parquet` 和 `pe_mad.parquet` 解读；正式 PE 结果是经过横截面处理的 `signal`。

Step 5 同时生成逐日覆盖率。正式测试起点采用固定规则：非停牌股票中的有效 EP
覆盖率连续 3 个自然月末不低于 95%，以第三个月末作为首次能够确认稳定的起点。
本次计算得到的起点为 `2020-07-31`。

## 5. 中间结果和掩码字段

| 文件 | 含义 |
|---|---|
| `ep_raw.parquet` | 原始 EP，`signal` 列 |
| `pe_raw.parquet` | 原始 PE，`signal` 列 |
| `ep_mad.parquet` / `pe_mad.parquet` | MAD 处理后的 EP / PE |
| `ep.parquet` / `pe.parquet` | Z 标准化后的 EP / PE |
| `_mask.parquet` | 停牌、次新、ST/上市日期可用性及未筛选留痕 |
| `b_share_codes.parquet` | 被排除的 89 只沪深 B 股名单及分类 |
| `bse_code_mapping.parquet/.csv` | 242 对北交所新旧代码映射及匹配证据 |
| `factor_audit.parquet` | 市值、所用财报、TTM、原始值、MAD 值和 Z 值的逐行追溯表 |
| `factor_daily_coverage.parquet/.csv` | 每日全样本和非停牌样本覆盖率及稳定状态 |
| `factor_test_start.json` | 覆盖率稳定参数和正式测试起点 |
| `point_in_time_field_audit.parquet/.csv` | ST、上市日期和交易状态的时点可用性说明 |
| `reports.parquet` | 点时利润表版本 |
| `reports_effective.parquet` | 阻止旧公告回退后实际采用的财报版本 |
| `report_version_decisions.parquet` | 所有财报版本的采用或忽略决策 |
| `reports_ttm.parquet` | 点时 TTM 事件 |

## 6. 验证

运行 `factor_verify.py` 会检查：

- EP、PE 各阶段索引唯一且一致；
- PE 与 EP 在非零 EP 观测上互为倒数；
- 停牌行的 EP/PE 均为 NaN；
- ST 数据可用性和筛选标记始终为 0，上市日期代理值不参与筛选；
- B 股名单为 89 只，清洗后的行情不再含 `900xxx.BJ` 或 `200/201xxx.SZ`；
- 北交所映射为 242 对且一一对应，原始及复权价格在代码切换处连续；
- 2025 年末北交所 288 只股票中至少 270 只有效 EP，防止财报连接再次断层；
- MAD 截断数量、逐日 Z 均值和标准差；
- 较旧公告没有进入有效版本状态；
- 财报版本、TTM 事件和点时间对齐没有未来数据；
- 逐行追溯表字段齐全且与各阶段因子值一致；
- 每日覆盖率和正式测试起点符合预设规则。

## 7. 本次运行结果

本次运行日期为 `2026-09-14`，数据范围为 `2020-01-02` 至 `2025-12-31`。本次运行统计如下：

| 项目 | 数值 |
|---|---:|
| 原始行情行数 | 7,155,034 |
| 原始行情全期唯一代码标识数 | 6,016 |
| 排除 B 股数量 | 89 只（73,234 行）：上海 50 只、深圳 39 只 |
| 清洗后全期唯一代码标识数 | 5,927 |
| 清洗后行情行数 | 7,081,800 |
| 2025-12-31 行情截面 | 5,470 只：沪市 2,299、深市 2,883、北交所 288 |
| 清洗后停牌标记 | 21,985 行 |
| 北交所新旧代码映射 | 242 对 |
| 次新标记（仅记录） | 451,442 行 / 1,925 个统一证券身份 |
| 原始有效 EP | 6,732,630 |
| 原始有效 PE | 6,732,630 |
| 负 EP / 负 PE | 1,437,888 |
| EP MAD 截断 | 745,075 |
| PE MAD 截断 | 953,354 |
| 最终有效 Z EP / PE | 各 6,732,629 |
| 旧公告回退记录（已忽略） | 3,881 |
| 逐行追溯记录 | 7,081,800 |
| 正式测试起点 | 2020-07-31 |
| 点时未来数据违规 | 0 |

本次数据中没有有效的零 EP；程序仍会将零 EP 的 PE 按定义置为 NaN。原始 EP
池化中位数约为 `0.0211`，原始 PE 池化中位数约为 `26.2719` 倍。

当前修改阶段以 `EP_PE统计分析与因子测试计划.md` 为主文档，其中统计数字及行业图已按
89 只 B 股过滤、北交所代码映射和公告版本防回退口径更新；对应 `.tex/.pdf` 已同步编译。

`回归结果\_历史口径存档\OLS_日复合收益` 和 `OLS_单日示例` 中的结果均早于本次口径修正，
仍不能作为新口径的正式回归结论。
本次已使用 `cross_section_rlm.py` 完成 64 个月度横截面 Huber RLM，正式结果位于
`回归结果\ep\RLM_H1_独立同方差`。该脚本与具体因子无关，换因子只需传 `--factor` 参数。
单月系数使用 H1 标准误计算大样本近似 `z` 值，
月度系数序列暂按独立同方差假设推断，不使用 Newey--West HAC；尚未运行
新口径 OLS/WLS 对照、IC 或分层回测。

| RLM 核心指标 | 结果 |
|---|---:|
| 完整持有期 | 64 个月 |
| EP 月均系数 | 0.1623% |
| EP IID `t` 值 | 1.4487 |
| EP IID `p` 值 | 0.1524 |
| EP 系数为正的月份 | 53.12% |
| 月度模型收敛 | 64 / 64 |

EP 平均系数方向符合预期，但全期尚未达到 5% 显著性标准。该系数是一个标准差 EP
暴露对应的月度因子溢价，不是可交易组合收益。

## 8. 运行顺序

在项目 Python 环境中执行：

```powershell
python step1_样本筛选.py
python step2_利润表处理.py
python step3_TTM计算.py
python step4_点时间对齐_计算PE.py
python step5_MAD去极值_Z标准化.py
python factor_verify.py
python EP因子交付验证.py
python cross_section_rlm.py
```

分布图脚本 `plot_ep_pe分布.py` 直接读取 `ep_raw.parquet` 和 `pe_raw.parquet`，
不修改管道结果。

行业图脚本 `plot_industry_pe.py` 读取最新原始 PE 截面和申万一级行业表，以行业内
原始 PE 中位数绘图，同时生成 `industry_pe_latest.csv`、PNG 预览和纯 LaTeX
`pgfplots` 片段；图形和片段统一写入 `图形` 目录。当前截面为 `2025-12-31`，
31 个行业共使用 5,438 条行业和 PE
均有效的观测；该脚本只做描述性统计，不运行回归检验。

## 9. 限制

1. 当前数据没有历史 ST/PT 标记，因此不做 ST 筛选。
2. `first_seen_date` 只能识别行情数据起点之后上市的股票，不能替代官方上市日期。
3. 当前没有正式上市、退市及北交所转板日期表；股票池按行情表当日记录和交易所后缀构造，无法独立核验每只股票的精确上市、退市或转板边界。
4. 北交所新旧代码映射由行情表休市前后的价格和总股本推导，已通过一一对应及价格连续性检查，但仍不是外部官方映射表。
5. 本交付只计算因子，不包含 T+1 实际成交、涨跌停、复牌执行、滑点或交易成本模拟。
6. 本次已运行 H1 协方差口径的 RLM，但没有运行新口径 OLS/WLS 对照、IC 或分层回测；两个 OLS 子目录中的文件仍为历史结果或学习示例。
7. PE 在亏损和接近零利润区间存在金融含义和数值稳定性差异，正式选股使用哪一个方向应结合研究口径确认。
