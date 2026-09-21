# EP/PE 因子项目

本目录保留一套可追溯的 EP/PE 因子计算、统计分析与横截面回归流程。样本期为
2020-01-02 至 2025-12-31，共 1,455 个交易日。

## 快速开始

数据源目录按以下优先级解析，换机器不需要改代码：

1. 环境变量 `EP_FACTOR_DATA_ROOT`
2. `config/paths.local.toml`（本机覆盖，不纳入版本管理）
3. `config/paths.toml`（仓库默认值）

```powershell
pip install -r requirements.txt

# 复制一份本机配置并改成自己的数据目录
Copy-Item config\paths.toml config\paths.local.toml

# 因子计算流水线
python "代码\step1_样本筛选.py"
python "代码\step2_利润表处理.py"
python "代码\step3_TTM计算.py"
python "代码\step4_点时间对齐_计算PE.py"
python "代码\step5_MAD去极值_Z标准化.py"

# 横截面回归：读取因子数据并输出回归统计结果（实测约 23 秒）
python "代码\cross_section_rlm.py"                       # 默认因子 ep
python "代码\cross_section_rlm.py" --factor pb            # 换成其它因子
python "代码\cross_section_rlm.py" --factor "D:/其它路径/my.parquet"   # 任意路径

# Rank IC 检验：读取因子数据并输出 IC / IR 统计结果（实测约 25 秒）
python "代码\rank_ic.py"
python "代码\rank_ic.py" --factor pb

# 交付验收
python "代码\factor_verify.py"
```

## 代码分层

| 层 | 位置 | 职责 |
|---|---|---|
| 共享底层 | `代码/common/` | 分片读取 parquet、行情快照清洗、B 股剔除、北交所新旧代码映射、月度日历、前瞻收益、回归面板 |
| 数据清洗 | `代码/step1`–`step4` | 样本筛选、利润表点时间版本、TTM 归母净利润、点时间对齐与原始 EP/PE |
| 因子计算 | `代码/step5_MAD去极值_Z标准化.py` | 逐日横截面 MAD 去极值与 Z 标准化 |
| 回归分析 | `代码/cross_section_rlm.py`、`代码/cross_section_ols.py` | 通用月度横截面 Huber RLM 与同口径 OLS，读取任意因子文件 |
| IC 检验 | `代码/rank_ic.py` | 通用月度 Rank IC：行业与市值中性化后计算 IC / IR，读取任意因子文件 |
| 分析与绘图 | `代码/plot_*`、`代码/analyze_mad_standardization.py` | 描述性统计、分布图、行业图 |
| 验收 | `代码/factor_verify.py`、`代码/EP因子交付验证.py` | 独立一致性检查 |

回归脚本与具体因子无关：给定一份因子文件就输出对应的回归统计结果，换因子只换
参数、不改代码。样本构造全部调用 `代码/common/`，因此 RLM 与 OLS 使用完全相同的
样本、前瞻收益、控制变量和月度日历，结果差异只来自估计方法。

```powershell
python "代码\cross_section_rlm.py" --factor ep    # → 回归结果/ep/RLM_H1_独立同方差/
python "代码\cross_section_rlm.py" --factor pb    # → 回归结果/pb/RLM_H1_独立同方差/
```

## 目录说明

| 目录 | 内容 | 是否纳入 Git |
|---|---|---|
| `代码/` | 全部分析脚本与共享模块 | 是 |
| `config/` | 数据源路径配置 | 是（`paths.local.toml` 除外） |
| `日志/` | 变更日志与运行输出 | 是 |
| `中间结果/` | 原始 EP/PE、财务版本、点时间审计、处理过程数据 | 否 |
| `因子结果/` | 最终标准化 EP、PE 因子 | 否 |
| `标准化统计/` | MAD、Z 标准化统计表与图形 | 否 |
| `回归结果/` | 按因子分目录：`回归结果/<因子名>/RLM_H1_独立同方差/`、`OLS_月度/`、`RankIC/`，另含 `_历史口径存档/` | 否 |
| `图形/` | 文档使用的公共图形 | 否 |

## 主要文档

- `脚本说明.md`：逐个脚本的作用、输入、输出、运行方式与依赖顺序。
- `处理记录.md`：历次口径变更和完整计算过程记录。
- `EP因子交付说明.md`：数据口径、运行步骤和交付字段。
- `EP_PE统计分析与因子测试计划.md` / `.tex`：统计结果、RLM 回归说明及后续测试计划。
- `MAD去极值与标准化统计分析.md` / `.tex`：MAD 去极值和 Z 标准化专项分析。
- `日志/20260921_共享模块抽取与口径统一.md`：共享模块抽取的完整过程与验证结果。

PDF 是对应 LaTeX 文档的可重新生成交付件。

## 统一口径

| 项目 | 口径 |
|---|---|
| 因子方向 | 盈利收益率 `EP = TTM 归母净利润 / 总市值`，`PE = 1 / EP` |
| 样本筛选 | 不筛选 ST/PT 与次新股，只对停牌日置缺失；排除 89 只沪深 B 股 |
| 北交所代码 | 2025 年 10 月新旧代码按价格与总股本精确映射，切换日从数据推导 |
| 调仓规则 | 每月最后一个交易日形成因子，次月第一个交易日建仓，再下月首个交易日退出 |
| 前瞻收益 | 建仓日与退出日复权收盘价相除；任一端点停牌则该期收益置缺失 |
| 规模控制 | `ln(me_total)` 在 EP 有效 universe 内做横截面 Z 标准化 |
| 行业控制 | 截距 + K−1 个申万一级行业哑变量 |
| RLM 设置 | HuberT(c=1.345)、MAD 残差尺度、H1 协方差、最多 100 次迭代、1e-8 收敛 |
| 时间序列推断 | 月度系数的独立同方差 t 检验，未启用 Newey-West HAC |
