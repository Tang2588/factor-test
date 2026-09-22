# -*- coding: utf-8 -*-
"""通用行业内分层回测。

与具体因子无关：读取一份因子文件，在行业内按中性化残差分成五组，做组合回测。

    python "代码\\layered_backtest.py"
    python "代码\\layered_backtest.py" --factor pb

口径：

1. 每月形成日取因子，按「因子对行业与市值回归的残差」在每个一级行业内部排序；
2. 行业内按升序分成五组，Q1 = 残差最低，Q5 = 残差最高；
3. 各行业同序号组合并成全市场组合，因此每个组合的行业构成与股票池一致；
4. 主结果市值加权，同时给出等权对照；
5. 建仓日涨停的股票买不进、退出日跌停的股票卖不出，两者均剔除；
6. 多空组合 = Q5 − Q1。

样本、前瞻收益、行业与市值全部来自 ``代码/common``，与 ``cross_section_rlm.py``、
``rank_ic.py`` 完全一致，三个环节可以直接对照。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    INDUSTRY_PATH,
    MARKET_PATH,
    MIN_OBSERVATIONS,
    REG_DIR,
    TEST_START_PATH,
    add_neutralized_residual,
    build_endpoint_forward_returns,
    build_monthly_calendar,
    build_regression_panel,
    derive_bse_mapping,
    prepare_formation_panel,
    prepare_industry,
    prepare_market_snapshot,
    read_selected_dates,
    resolve_factor_path,
    scan_market_index,
)

MODEL_DIR_NAME = "分层回测"
N_GROUPS = 5
GROUP_LABELS = [f"Q{i}" for i in range(1, N_GROUPS + 1)]


def build_holdings(
    panel: pd.DataFrame,
    groups: int = N_GROUPS,
) -> tuple[pd.DataFrame, dict]:
    """逐月逐股确定分组与权重。"""
    data = panel.loc[
        panel["neutral_residual"].notna()
        & panel["forward_return"].notna()
        & panel["me_total"].gt(0)
        & panel["industry_lv1"].notna()
    ].copy()

    # 涨跌停标记来自 merge，dtype 是 object，必须先转成真正的布尔值：
    # 直接对 object 列做 ~ 会得到整数（~False = -1），导致过滤条件全部通过。
    entry_up_limit = data["entry_at_up_limit"].astype("boolean").fillna(False).astype(bool)
    exit_down_limit = data["exit_at_down_limit"].astype("boolean").fillna(False).astype(bool)

    audit = {
        "n_eligible_before_tradability": int(len(data)),
        "n_excluded_entry_up_limit": int(entry_up_limit.sum()),
        "n_excluded_exit_down_limit": int(exit_down_limit.sum()),
        "n_excluded_both_limits": int((entry_up_limit & exit_down_limit).sum()),
    }
    keep = (~entry_up_limit) & (~exit_down_limit)
    data = data.loc[keep].copy()
    audit["n_eligible_after_tradability"] = int(len(data))

    data["industry_short"] = data["industry_lv1"].astype("string").str.split("_").str[-1]

    # 行业内按残差升序分组，用分位数保证组数稳定（小行业也不会出现空组）
    key = ["formation_date", "industry_lv1"]
    rank = data.groupby(key, sort=False)["neutral_residual"].rank(method="first")
    size = data.groupby(key, sort=False)["neutral_residual"].transform("size")
    percentile = (rank - 0.5) / size
    data["quintile"] = np.minimum((percentile * groups).astype(int), groups - 1) + 1
    data["quintile"] = data["quintile"].astype(int)

    group_key = ["formation_date", "quintile"]
    cap_sum = data.groupby(group_key, sort=False)["me_total"].transform("sum")
    data["weight_cap"] = data["me_total"] / cap_sum
    data["weight_eq"] = 1.0 / data.groupby(group_key, sort=False)["security_id"].transform("size")

    audit["n_ties_in_factor"] = int(
        data.groupby(key, sort=False)["neutral_residual"].transform(lambda s: s.duplicated().sum()).sum()
    )
    audit["min_industry_stocks"] = int(size.min())
    audit["median_industry_stocks"] = float(size.median())
    return data, audit


def compute_monthly_returns(
    holdings: pd.DataFrame,
    groups: int = N_GROUPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐月计算五组收益、多空收益与基准收益。"""
    rows = []
    for formation_date, month in holdings.groupby("formation_date", sort=True):
        # 基准必须在全股票池上按市值加权，不能用组内归一化的权重（每只股票的
        # weight_cap 是相对它所在组归一化的，直接求和会放大约 N_GROUPS 倍）
        total_mv = float(month["me_total"].sum())
        record = {
            "formation_date": formation_date,
            "entry_date": month["entry_date"].iloc[0],
            "exit_date": month["exit_date"].iloc[0],
            "benchmark_cap": float(
                (month["me_total"] * month["forward_return"]).sum() / total_mv
            )
            if total_mv > 0
            else np.nan,
            "benchmark_eq": float(month["forward_return"].mean()),
            "n_universe": int(len(month)),
        }
        for q in range(1, groups + 1):
            sub = month.loc[month["quintile"].eq(q)]
            record[f"n_Q{q}"] = int(len(sub))
            record[f"cap_Q{q}"] = (
                float((sub["weight_cap"] * sub["forward_return"]).sum()) if len(sub) else np.nan
            )
            record[f"eq_Q{q}"] = float(sub["forward_return"].mean()) if len(sub) else np.nan
        record["ls_cap"] = record[f"cap_Q{groups}"] - record["cap_Q1"]
        record["ls_eq"] = record[f"eq_Q{groups}"] - record["eq_Q1"]
        rows.append(record)
    monthly = pd.DataFrame(rows).sort_values("formation_date").reset_index(drop=True)
    return monthly


def compute_turnover(holdings: pd.DataFrame, weight_col: str, groups: int = N_GROUPS) -> pd.DataFrame:
    """逐月单边换手率：0.5 * sum|w_new - w_old|，按月对齐同一证券。"""
    rows = []
    for q in range(1, groups + 1):
        prev: pd.Series | None = None
        subset = holdings.loc[holdings["quintile"].eq(q)]
        for formation_date, month in subset.groupby("formation_date", sort=True):
            current = month.set_index("security_id")[weight_col]
            if prev is None:
                turnover = np.nan
            else:
                aligned = pd.concat([prev.rename("old"), current.rename("new")], axis=1).fillna(0.0)
                turnover = float(0.5 * (aligned["new"] - aligned["old"]).abs().sum())
            rows.append(
                {
                    "formation_date": formation_date,
                    "quintile": q,
                    "n": int(len(current)),
                    "turnover": turnover,
                }
            )
            prev = current
    return pd.DataFrame(rows)


def _nav(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def performance(returns: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    """组合绩效指标。返回按 formation_date 对齐的序列计算结果。"""
    clean = pd.to_numeric(returns, errors="coerce")
    bench = pd.to_numeric(benchmark, errors="coerce")
    valid = clean.notna()
    clean = clean.loc[valid]
    bench = bench.loc[valid]
    months = len(clean)
    if months < 2:
        return {}
    nav = _nav(clean)
    total_return = float(nav.iloc[-1] - 1.0)
    annual_return = (1.0 + total_return) ** (12.0 / months) - 1.0
    annual_vol = float(clean.std(ddof=1) * np.sqrt(12.0))
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
    drawdown = nav / nav.cummax() - 1.0
    excess = clean - bench
    tracking_error = float(excess.std(ddof=1) * np.sqrt(12.0))
    info_ratio = (
        float(excess.mean() * 12.0 / tracking_error) if tracking_error > 0 else np.nan
    )
    return {
        "months": months,
        "total_return": total_return,
        "annual_return": float(annual_return),
        "annual_vol": annual_vol,
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "win_ratio": float((clean > 0).mean()),
        "excess_annual_return": float(excess.mean() * 12.0),
        "tracking_error": tracking_error,
        "info_ratio": info_ratio,
        "monthly_mean": float(clean.mean()),
        "monthly_std": float(clean.std(ddof=1)),
    }


def build_performance_table(
    monthly: pd.DataFrame,
    turnover: pd.DataFrame,
    groups: int = N_GROUPS,
) -> pd.DataFrame:
    rows = []
    turnover_mean = turnover.groupby("quintile")["turnover"].mean().to_dict()
    for q in range(1, groups + 1):
        stats = performance(monthly[f"cap_Q{q}"], monthly["benchmark_cap"])
        rows.append(
            {
                "组合": f"Q{q}",
                **stats,
                "avg_turnover": float(turnover_mean.get(q, np.nan)),
                "avg_stocks": float(monthly[f"n_Q{q}"].mean()),
            }
        )
    stats = performance(monthly["ls_cap"], monthly["benchmark_cap"])
    rows.append(
        {
            "组合": f"Q{groups}-Q1",
            **stats,
            "avg_turnover": float(
                np.nanmean([turnover_mean.get(1, np.nan), turnover_mean.get(groups, np.nan)])
            ),
            "avg_stocks": float(monthly[f"n_Q{groups}"].mean()),
        }
    )
    table = pd.DataFrame(rows)
    table["annual_return_pct"] = table["annual_return"] * 100
    return table


def monotonicity(monthly: pd.DataFrame, groups: int = N_GROUPS) -> dict[str, float]:
    """组序号与组收益的 Spearman 相关系数（逐月与全期两种）。"""
    ranks = np.arange(1, groups + 1)
    per_month = []
    for _, row in monthly.iterrows():
        values = np.array([row[f"cap_Q{q}"] for q in range(1, groups + 1)], dtype=float)
        per_month.append(float(spearmanr(ranks, values).statistic))
    rank_corr = pd.Series(per_month, index=monthly["formation_date"])
    mean_returns = np.array(
        [monthly[f"cap_Q{q}"].mean() for q in range(1, groups + 1)], dtype=float
    )
    return {
        "monthly_rank_corr_mean": float(rank_corr.mean()),
        "monthly_rank_corr_positive_ratio": float((rank_corr > 0).mean()),
        "full_period_rank_corr": float(spearmanr(ranks, mean_returns).statistic),
    }


def build_annual_table(monthly: pd.DataFrame, groups: int = N_GROUPS) -> pd.DataFrame:
    working = monthly.copy()
    working["year"] = working["formation_date"].dt.year
    rows = []
    for year, group in working.groupby("year", sort=True):
        record = {"year": int(year), "months": len(group)}
        for q in range(1, groups + 1):
            series = group[f"cap_Q{q}"]
            record[f"Q{q}"] = float(_nav(series).iloc[-1] - 1.0)
        record["LS"] = float(_nav(group["ls_cap"]).iloc[-1] - 1.0)
        record["benchmark"] = float(_nav(group["benchmark_cap"]).iloc[-1] - 1.0)
        rows.append(record)
    return pd.DataFrame(rows)


def save_nav_plot(monthly: pd.DataFrame, path: Path, factor_label: str, groups: int = N_GROUPS) -> None:
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    dates = monthly["formation_date"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, groups))

    axes[0].axhline(1.0, color="#555555", linewidth=0.9)
    for index, q in enumerate(range(1, groups + 1)):
        axes[0].plot(dates, _nav(monthly[f"cap_Q{q}"]), linewidth=1.4, color=colors[index], label=f"Q{q}")
    axes[0].plot(
        dates, _nav(monthly["benchmark_cap"]), linewidth=1.6, color="#333333",
        linestyle="--", label="Universe (cap-weighted)",
    )
    axes[0].set_ylabel("Cumulative NAV")
    axes[0].set_title(f"Industry-neutral layered portfolios of {factor_label} (cap-weighted)")
    axes[0].legend(ncol=groups + 1, fontsize=9)
    axes[0].grid(alpha=0.25)

    axes[1].axhline(1.0, color="#555555", linewidth=0.9)
    axes[1].plot(dates, _nav(monthly["ls_cap"]), color="#B23A48", linewidth=1.6, label="Q5 - Q1 (cap)")
    axes[1].plot(
        dates, _nav(monthly["ls_eq"]), color="#2E7D5B", linewidth=1.3,
        linestyle="--", label="Q5 - Q1 (equal weight)",
    )
    axes[1].set_ylabel("Cumulative NAV")
    axes[1].set_xlabel("Formation date")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_annual_plot(annual: pd.DataFrame, path: Path, factor_label: str, groups: int = N_GROUPS) -> None:
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 1, figsize=(12, 9))
    years = annual["year"].to_numpy()
    width = 0.8 / (groups + 2)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, groups))

    for index, q in enumerate(range(1, groups + 1)):
        axes[0].bar(
            years + (index - (groups - 1) / 2) * width,
            annual[f"Q{q}"] * 100,
            width=width,
            color=colors[index],
            label=f"Q{q}",
        )
    axes[0].bar(
        years + ((groups + 0.5) - (groups - 1) / 2) * width,
        annual["benchmark"] * 100,
        width=width,
        color="#333333",
        label="Universe",
    )
    axes[0].axhline(0.0, color="#555555", linewidth=0.9)
    axes[0].set_ylabel("Annual return (%)")
    axes[0].set_title(f"Annual returns by quintile, {factor_label}")
    axes[0].legend(ncol=groups + 1, fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(years, annual["LS"] * 100, width=0.5, color="#B23A48", label="Q5 - Q1")
    axes[1].axhline(0.0, color="#555555", linewidth=0.9)
    axes[1].set_ylabel("Annual return (%)")
    axes[1].set_xlabel("Year")
    axes[1].set_title("Long-short annual return (cap-weighted)")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_turnover_plot(turnover: pd.DataFrame, path: Path, groups: int = N_GROUPS) -> None:
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, groups))
    for index, q in enumerate(range(1, groups + 1)):
        subset = turnover.loc[turnover["quintile"].eq(q)].dropna(subset=["turnover"])
        axes.plot(
            subset["formation_date"],
            subset["turnover"] * 100,
            linewidth=1.3,
            color=colors[index],
            label=f"Q{q}",
        )
    axes.set_ylabel("One-way turnover (%)")
    axes.set_xlabel("Formation date")
    axes.set_title("Monthly one-way turnover by quintile")
    axes.legend(ncol=groups, fontsize=9)
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def num(value: float, digits: int = 4) -> str:
    return "NaN" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def pct(value: float, digits: int = 2) -> str:
    return "NaN" if value is None or not np.isfinite(value) else f"{value * 100:.{digits}f}%"


def save_report(
    performance_table: pd.DataFrame,
    equal_table: pd.DataFrame,
    monotonic: dict[str, float],
    annual: pd.DataFrame,
    turnover: pd.DataFrame,
    audit: dict,
    monthly: pd.DataFrame,
    factor_label: str,
    factor_path: Path,
    path: Path,
) -> None:
    ls_row = performance_table.iloc[-1]
    ls_eq_row = equal_table.iloc[-1]
    q_returns = [performance_table.iloc[q - 1]["annual_return"] for q in range(1, N_GROUPS + 1)]
    monotone_text = "单调递增" if all(
        q_returns[i] < q_returns[i + 1] for i in range(len(q_returns) - 1)
    ) else ("单调递减" if all(
        q_returns[i] > q_returns[i + 1] for i in range(len(q_returns) - 1)
    ) else "非单调")

    lines = [
        f"# {factor_label} 因子行业内分层回测结果",
        "",
        f"**运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**",
        "",
        f"**因子数据：** `{factor_path}`",
        "",
        "> 本报告只包含分层回测，不包含截面回归、Rank IC 或交易成本敏感性分析。",
        "",
        "## 1. 回测口径",
        "",
        "- 分组依据：因子对「对数市值 + 行业」回归后的**中性化残差**，与 Rank IC 口径一致；",
        "- 每月在每个申万一级行业内部按残差升序分五组，Q1 最低、Q5 最高；",
        "- 各行业同序号组合并成全市场组合，因此每个组合的行业构成与股票池一致；",
        "- 主结果市值加权，等权作为对照；权重在形成日确定，持有期内不调仓；",
        "- 建仓日涨停（买不进）与退出日跌停（卖不出）的股票均剔除；",
        f"- 多空组合 = Q{N_GROUPS} − Q1；基准为股票池市值加权收益（缺少指数数据时的内部基准）。",
        "",
        "## 2. 样本与可交易性",
        "",
        f"- 通过因子与收益筛选的观测：{audit['n_eligible_before_tradability']:,} 条；",
        f"- 因建仓日涨停剔除：{audit['n_excluded_entry_up_limit']:,} 条；",
        f"- 因退出日跌停剔除：{audit['n_excluded_exit_down_limit']:,} 条；",
        f"- 其中同时命中两条的观测：{audit['n_excluded_both_limits']:,} 条；",
        f"- 最终进入回测的观测：{audit['n_eligible_after_tradability']:,} 条；",
        f"- 行业内股票数：最少 {audit['min_industry_stocks']} 只，中位 {audit['median_industry_stocks']:.0f} 只；",
        f"- 回测月份：{len(monthly)} 个月。",
        "",
        "## 3. 组合绩效（市值加权）",
        "",
        "| 组合 | 年化收益 | 累计收益 | 年化波动 | 夏普比率 | 最大回撤 | 胜率 | 年化超额 | 信息比率 | 平均换手率 | 平均持股数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in performance_table.itertuples(index=False):
        lines.append(
            f"| {row.组合} | {pct(row.annual_return)} | {pct(row.total_return)} | "
            f"{pct(row.annual_vol)} | {num(row.sharpe, 3)} | {pct(row.max_drawdown)} | "
            f"{pct(row.win_ratio)} | {pct(row.excess_annual_return)} | {num(row.info_ratio, 3)} | "
            f"{pct(row.avg_turnover)} | {row.avg_stocks:,.0f} |"
        )

    lines += [
        "",
        "> 全期与各组的「年化超额」「信息比率」是相对股票池市值加权基准计算的多头超额。",
        f"多空组合（Q{N_GROUPS}−Q1）是零成本对冲组合，与多头基准比较没有经济含义，",
        "该行只应看年化收益、夏普比率与最大回撤。",
        "",
        "## 4. 单调性检验",
        "",
        f"- 五组年化收益排列：{monotone_text}；",
        f"- 全期组序号与组年化收益的 Spearman 相关系数：{num(monotonic['full_period_rank_corr'], 4)}；",
        f"- 逐月组序号与组收益的 Spearman 相关系数均值：{num(monotonic['monthly_rank_corr_mean'], 4)}；",
        f"- 该相关系数为正的月份比例：{pct(monotonic['monthly_rank_corr_positive_ratio'])}。",
        "",
        "## 5. 分年度收益",
        "",
        "| 年份 | 月数 | " + " | ".join(GROUP_LABELS) + " | 多空 | 基准 |",
        "|---|---:|" + "---:|" * (N_GROUPS + 2),
    ]
    for row in annual.itertuples(index=False):
        values = " | ".join(pct(getattr(row, label)) for label in GROUP_LABELS)
        lines.append(
            f"| {row.year} | {row.months} | {values} | {pct(row.LS)} | {pct(row.benchmark)} |"
        )

    lines += [
        "",
        "## 6. 等权对照",
        "",
        "| 组合 | 年化收益 | 年化波动 | 夏普比率 | 最大回撤 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in equal_table.itertuples(index=False):
        lines.append(
            f"| {row.组合} | {pct(row.annual_return)} | {pct(row.annual_vol)} | "
            f"{num(row.sharpe, 3)} | {pct(row.max_drawdown)} |"
        )

    lines += [
        "",
        "## 7. 换手率",
        "",
        "| 组合 | 平均单边换手率 | 中位单边换手率 | 最大单边换手率 |",
        "|---|---:|---:|---:|",
    ]
    for q in range(1, N_GROUPS + 1):
        subset = turnover.loc[turnover["quintile"].eq(q), "turnover"].dropna()
        lines.append(
            f"| Q{q} | {pct(subset.mean())} | {pct(subset.median())} | {pct(subset.max())} |"
        )

    lines += [
        "",
        "## 8. 图表",
        "",
        "![五组与多空累计净值](./layered_nav.png)",
        "",
        "![分年度收益与多空收益](./layered_annual.png)",
        "",
        "![月度换手率](./layered_turnover.png)",
        "",
        "## 9. 输出文件",
        "",
        "- `layered_monthly_returns.csv`：逐月五组、多空与基准收益。",
        "- `layered_performance.csv`：市值加权组合绩效汇总。",
        "- `layered_performance_equal.csv`：等权组合绩效汇总。",
        "- `layered_annual_returns.csv`：分年度收益。",
        "- `layered_turnover.csv`：逐月单边换手率。",
        "- `layered_holdings.parquet`：逐月逐股分组与权重。",
        "- `layered_config.json`：本次运行的完整口径记录。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通用行业内分层回测：读取因子数据并输出组合绩效与回测图。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            '  python "代码\\layered_backtest.py"\n'
            '  python "代码\\layered_backtest.py" --factor pb\n'
            '  python "代码\\layered_backtest.py" --factor "D:/其它/my.parquet" --factor-name MY\n'
        ),
    )
    parser.add_argument(
        "--factor",
        default="ep",
        help="因子名或因子文件路径。因子名会解析为 因子结果/<名称>.parquet（默认 ep）",
    )
    parser.add_argument(
        "--factor-name",
        default=None,
        help="输出目录与报告使用的因子标签，默认取因子文件名",
    )
    parser.add_argument("--out-root", default=None, help="结果根目录，默认 回归结果/")
    parser.add_argument(
        "--min-observations",
        type=int,
        default=MIN_OBSERVATIONS,
        help=f"中性化回归的单月最小观测数（默认 {MIN_OBSERVATIONS}）",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=N_GROUPS,
        help=f"行业内分组数（默认 {N_GROUPS}）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global N_GROUPS, GROUP_LABELS  # noqa: PLW0603
    args = parse_args(argv)
    N_GROUPS = int(args.groups)
    GROUP_LABELS = [f"Q{i}" for i in range(1, N_GROUPS + 1)]

    factor_path = resolve_factor_path(args.factor)
    if not factor_path.exists():
        raise FileNotFoundError(f"找不到因子文件：{factor_path}")
    factor_label = (args.factor_name or factor_path.stem).upper()
    out_root = Path(args.out_root) if args.out_root else REG_DIR
    out_dir = out_root / factor_path.stem / MODEL_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    with TEST_START_PATH.open(encoding="utf-8") as file:
        test_config = json.load(file)
    test_start = pd.Timestamp(test_config["formal_test_start_date"])

    print(f"Factor: {factor_label} ({factor_path})", flush=True)
    print(f"Output: {out_dir}", flush=True)
    print("Scanning market index (calendar + BSE code switch)...", flush=True)
    index = scan_market_index(MARKET_PATH)
    transition_old_date = index.old_bse_last_date
    transition_new_date = index.new_bse_first_date
    if transition_old_date is None or transition_new_date is None:
        raise ValueError("无法从行情数据中识别北交所代码切换日")

    calendar = build_monthly_calendar(index.dates, test_start)
    signal_dates = set(pd.to_datetime(calendar["formation_date"]))
    return_dates = set(pd.to_datetime(calendar["entry_date"])) | set(
        pd.to_datetime(calendar["exit_date"])
    )
    selected_market_dates = signal_dates | return_dates | {
        transition_old_date,
        transition_new_date,
    }

    print(f"Reading {len(selected_market_dates)} selected market dates...", flush=True)
    market_raw = read_selected_dates(
        MARKET_PATH,
        [
            "date",
            "stock_code",
            "close",
            "pre_close",
            "close_adj",
            "pre_close_adj",
            "share_total",
            "me_total",
            "status",
            "suspended",
            "up_limit",
            "down_limit",
        ],
        selected_market_dates,
    )
    market = prepare_market_snapshot(market_raw)

    print("Deriving and validating BSE code mapping...", flush=True)
    mapping_frame = derive_bse_mapping(
        market,
        transition_old_date,
        transition_new_date,
        expected_pairs=index.expected_conversions,
    )
    mapping = dict(zip(mapping_frame["old_code"], mapping_frame["new_code"]))

    print("Reading monthly factor and industry cross-sections...", flush=True)
    factor = read_selected_dates(factor_path, ["date", "stock_code", "signal"], signal_dates)
    industry_raw = read_selected_dates(
        INDUSTRY_PATH,
        ["date", "stock_code", "indus_name_lv1"],
        signal_dates,
    )
    formation = prepare_formation_panel(market, factor, signal_dates, mapping)
    industry = prepare_industry(industry_raw, mapping)

    print("Building fixed-endpoint adjusted forward returns...", flush=True)
    forward_returns = build_endpoint_forward_returns(market, calendar, mapping)

    print("Building monthly regression samples...", flush=True)
    panel, sample_audit = build_regression_panel(formation, industry, forward_returns)

    print("Neutralizing factor and assigning quintiles...", flush=True)
    panel = add_neutralized_residual(panel, min_observations=args.min_observations)
    holdings, audit = build_holdings(panel, groups=N_GROUPS)
    print(
        f"  eligible={audit['n_eligible_after_tradability']:,} "
        f"(excluded up-limit {audit['n_excluded_entry_up_limit']:,}, "
        f"down-limit {audit['n_excluded_exit_down_limit']:,})",
        flush=True,
    )

    print("Computing portfolio returns...", flush=True)
    monthly = compute_monthly_returns(holdings, groups=N_GROUPS)
    turnover_cap = compute_turnover(holdings, "weight_cap", groups=N_GROUPS)
    performance_cap = build_performance_table(monthly, turnover_cap, groups=N_GROUPS)
    equal_monthly = monthly.copy()
    for q in range(1, N_GROUPS + 1):
        equal_monthly[f"cap_Q{q}"] = monthly[f"eq_Q{q}"]
    equal_monthly["ls_cap"] = monthly["ls_eq"]
    equal_monthly["benchmark_cap"] = monthly["benchmark_eq"]
    performance_eq = build_performance_table(equal_monthly, turnover_cap, groups=N_GROUPS)
    monotonic = monotonicity(monthly, groups=N_GROUPS)
    annual = build_annual_table(monthly, groups=N_GROUPS)

    monthly.to_parquet(out_dir / "layered_monthly_returns.parquet", index=False)
    monthly.to_csv(out_dir / "layered_monthly_returns.csv", index=False, encoding="utf-8-sig")
    performance_cap.to_csv(out_dir / "layered_performance.csv", index=False, encoding="utf-8-sig")
    performance_eq.to_csv(
        out_dir / "layered_performance_equal.csv", index=False, encoding="utf-8-sig"
    )
    annual.to_csv(out_dir / "layered_annual_returns.csv", index=False, encoding="utf-8-sig")
    turnover_cap.to_csv(out_dir / "layered_turnover.csv", index=False, encoding="utf-8-sig")
    holdings[
        [
            "formation_date",
            "security_id",
            "industry_short",
            "quintile",
            "neutral_residual",
            "forward_return",
            "me_total",
            "weight_cap",
            "weight_eq",
        ]
    ].to_parquet(out_dir / "layered_holdings.parquet", index=False)

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "factor": {
            "label": factor_label,
            "name": factor_path.stem,
            "path": str(factor_path),
        },
        "grouping": "neutralized residual of the factor on intercept + size_z + K-1 industry dummies",
        "groups": N_GROUPS,
        "within_industry": True,
        "weighting": "market-cap weighted (main), equal weighted (control)",
        "tradability_filter": "exclude entry-day up-limit buys and exit-day down-limit sells",
        "long_short": f"Q{N_GROUPS} - Q1",
        "benchmark": "market-cap weighted return of the eligible universe",
        "return_convention": {
            "formation": "last market trading day of month",
            "entry": "next month's first market trading day close_adj",
            "exit": "following month's first market trading day close_adj",
        },
        "min_observations": args.min_observations,
        "periods": len(monthly),
        "audit": audit,
    }
    (out_dir / "layered_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    save_nav_plot(monthly, out_dir / "layered_nav.png", factor_label, groups=N_GROUPS)
    save_annual_plot(annual, out_dir / "layered_annual.png", factor_label, groups=N_GROUPS)
    save_turnover_plot(turnover_cap, out_dir / "layered_turnover.png", groups=N_GROUPS)
    save_report(
        performance_cap,
        performance_eq,
        monotonic,
        annual,
        turnover_cap,
        audit,
        monthly,
        factor_label,
        factor_path,
        out_dir / "分层回测结果.md",
    )

    ls = performance_cap.iloc[-1]
    print("Layered backtest completed", flush=True)
    print(f"months={len(monthly)}", flush=True)
    print(f"ls_annual_return={ls['annual_return']:.10f}", flush=True)
    print(f"ls_sharpe={ls['sharpe']:.6f}", flush=True)
    print(f"ls_max_drawdown={ls['max_drawdown']:.6f}", flush=True)
    print(f"full_period_rank_corr={monotonic['full_period_rank_corr']:.6f}", flush=True)
    print(f"saved={out_dir}", flush=True)


if __name__ == "__main__":
    main()
