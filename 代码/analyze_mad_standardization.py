# -*- coding: utf-8 -*-
"""Describe the existing MAD winsorization and daily Z standardization outputs.

This script is read-only with respect to the factor pipeline. It consumes the
delivered audit/intermediate files and writes descriptive tables, figures, and
a Markdown report. It does not run regressions or backtests.
"""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "中间结果"
OUT = ROOT / "标准化统计"
FIG = OUT / "图形"
REPORT = ROOT / "MAD去极值与标准化统计分析.md"
INDUSTRY_PATH = Path(r"D:\实习生学习项目\基础数据\chn_equ_indus_sw.parquet")

FACTORS = {
    "EP": {"raw": "raw_ep", "mad": "mad_ep", "z": "z_ep"},
    "PE": {"raw": "raw_pe", "mad": "mad_pe", "z": "z_pe"},
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def finite_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", copy=False)
    return values[np.isfinite(values)]


def distribution_stats(series: pd.Series) -> dict[str, float | int]:
    values = finite_values(series)
    if not len(values):
        return {"n": 0}
    quantiles = np.quantile(values, [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p0_1": float(quantiles[0]),
        "p1": float(quantiles[1]),
        "p5": float(quantiles[2]),
        "p25": float(quantiles[3]),
        "median": float(quantiles[4]),
        "p75": float(quantiles[5]),
        "p95": float(quantiles[6]),
        "p99": float(quantiles[7]),
        "p99_9": float(quantiles[8]),
        "max": float(values.max()),
        "skew": float(skew(values, bias=False)),
        "excess_kurtosis": float(kurtosis(values, fisher=True, bias=False)),
        "negative_ratio": float((values < 0).mean()),
        "abs_gt_1_ratio": float((np.abs(values) > 1).mean()),
        "abs_gt_2_ratio": float((np.abs(values) > 2).mean()),
        "abs_gt_3_ratio": float((np.abs(values) > 3).mean()),
    }


def pct(value: float, digits: int = 2) -> str:
    return "NA" if not np.isfinite(value) else f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "NA"
    magnitude = abs(value)
    if magnitude and (magnitude >= 100_000 or magnitude < 0.0001):
        return f"{value:.3e}"
    return f"{value:,.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def load_processing_stats() -> pd.DataFrame:
    frames = []
    for factor in FACTORS:
        frame = pd.read_parquet(DATA / f"{factor.lower()}_processing_stats.parquet")
        frame["date"] = pd.to_datetime(frame["date"])
        frame["factor"] = factor
        frame["scale"] = frame["mad"] * 1.4826
        frame["clip_rate"] = frame["n_clipped"] / frame["n_valid"].replace(0, np.nan)
        frame["constant_cross_section"] = frame["n_valid"].gt(0) & frame["std_after_mad"].le(0)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def aggregate_daily(daily: pd.DataFrame, start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for factor, factor_data in daily.groupby("factor", sort=False):
        for scope, scoped in {
            "全样本": factor_data,
            "正式测试期": factor_data.loc[factor_data["date"].ge(start)],
        }.items():
            rates = scoped["clip_rate"].dropna()
            summary_rows.append(
                {
                    "factor": factor,
                    "scope": scope,
                    "days": len(scoped),
                    "valid_observations": int(scoped["n_valid"].sum()),
                    "clipped_observations": int(scoped["n_clipped"].sum()),
                    "weighted_clip_rate": float(scoped["n_clipped"].sum() / scoped["n_valid"].sum()),
                    "daily_clip_rate_mean": float(rates.mean()),
                    "daily_clip_rate_median": float(rates.median()),
                    "daily_clip_rate_p10": float(rates.quantile(0.10)),
                    "daily_clip_rate_p90": float(rates.quantile(0.90)),
                    "daily_clip_rate_min": float(rates.min()),
                    "daily_clip_rate_max": float(rates.max()),
                    "median_daily_median": float(scoped["median"].median()),
                    "median_daily_mad": float(scoped["mad"].median()),
                    "median_daily_scale": float(scoped["scale"].median()),
                    "constant_cross_sections": int(scoped["constant_cross_section"].sum()),
                }
            )
    summary = pd.DataFrame(summary_rows)

    annual = daily.copy()
    annual["year"] = annual["date"].dt.year
    annual = annual.loc[annual["date"].ge(start)]
    annual = (
        annual.groupby(["factor", "year"], as_index=False)
        .agg(
            trading_days=("date", "size"),
            n_valid=("n_valid", "sum"),
            n_clipped=("n_clipped", "sum"),
            median_daily_median=("median", "median"),
            median_daily_mad=("mad", "median"),
        )
    )
    annual["clip_rate"] = annual["n_clipped"] / annual["n_valid"]
    return summary, annual


def clipping_group_summary(
    data: pd.DataFrame,
    group: pd.Series,
    factor: str,
    scope_mask: pd.Series,
) -> pd.DataFrame:
    columns = FACTORS[factor]
    raw = data[columns["raw"]]
    mad = data[columns["mad"]]
    valid = scope_mask & raw.notna() & group.notna()
    temp = pd.DataFrame(
        {
            "group": group.loc[valid],
            "raw": raw.loc[valid],
            "lower_clipped": raw.loc[valid].lt(mad.loc[valid]),
            "upper_clipped": raw.loc[valid].gt(mad.loc[valid]),
        }
    )
    result = (
        temp.groupby("group", observed=True, sort=False)
        .agg(
            n_valid=("raw", "size"),
            raw_median=("raw", "median"),
            negative_ratio=("raw", lambda x: float(x.lt(0).mean())),
            lower_clipped=("lower_clipped", "sum"),
            upper_clipped=("upper_clipped", "sum"),
        )
        .reset_index()
    )
    result["n_clipped"] = result["lower_clipped"] + result["upper_clipped"]
    result["clip_rate"] = result["n_clipped"] / result["n_valid"]
    result.insert(0, "factor", factor)
    return result


def monthly_cross_section_diagnostics(
    data: pd.DataFrame, daily: pd.DataFrame, start: pd.Timestamp
) -> pd.DataFrame:
    dates = daily.loc[daily["factor"].eq("EP"), ["date"]].copy()
    dates["month"] = dates["date"].dt.to_period("M")
    month_ends = set(dates.groupby("month")["date"].max())
    month_ends.add(data["date"].max())
    sample = data.loc[data["date"].isin(month_ends) & data["date"].ge(start)]
    rows = []
    for date, day in sample.groupby("date", sort=True):
        for factor, columns in FACTORS.items():
            valid = day[columns["raw"]].notna() & day[columns["mad"]].notna() & day[columns["z"]].notna()
            raw = day.loc[valid, columns["raw"]].to_numpy(float)
            mad = day.loc[valid, columns["mad"]].to_numpy(float)
            z = day.loc[valid, columns["z"]].to_numpy(float)
            if len(raw) < 3:
                continue
            rows.append(
                {
                    "date": date,
                    "factor": factor,
                    "n": len(raw),
                    "raw_mad_pearson": float(np.corrcoef(raw, mad)[0, 1]),
                    "raw_mad_spearman": float(spearmanr(raw, mad).statistic),
                    "mad_z_pearson": float(np.corrcoef(mad, z)[0, 1]),
                    "raw_skew": float(skew(raw, bias=False)),
                    "mad_skew": float(skew(mad, bias=False)),
                    "raw_excess_kurtosis": float(kurtosis(raw, fisher=True, bias=False)),
                    "mad_excess_kurtosis": float(kurtosis(mad, fisher=True, bias=False)),
                }
            )
    return pd.DataFrame(rows)


def standardization_checks(data: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor, columns in FACTORS.items():
        grouped = data.loc[data[columns["z"]].notna()].groupby("date")[columns["z"]]
        means = grouped.mean()
        stds = grouped.std(ddof=0)
        expected_days = daily.loc[
            daily["factor"].eq(factor) & daily["std_after_mad"].gt(0), "date"
        ]
        means = means.reindex(expected_days)
        stds = stds.reindex(expected_days)
        z_values = finite_values(data[columns["z"]])
        rows.append(
            {
                "factor": factor,
                "standardized_days": int(len(stds)),
                "max_abs_daily_mean": float(means.abs().max()),
                "max_abs_daily_std_minus_one": float((stds - 1).abs().max()),
                "pooled_mean": float(z_values.mean()),
                "pooled_std": float(z_values.std(ddof=0)),
                "abs_z_gt_1_ratio": float((np.abs(z_values) > 1).mean()),
                "abs_z_gt_2_ratio": float((np.abs(z_values) > 2).mean()),
                "abs_z_gt_3_ratio": float((np.abs(z_values) > 3).mean()),
            }
        )
    return pd.DataFrame(rows)


def latest_industry_summary(latest_data: pd.DataFrame, latest: pd.Timestamp) -> pd.DataFrame:
    if not INDUSTRY_PATH.exists():
        return pd.DataFrame()
    industry = pd.read_parquet(
        INDUSTRY_PATH,
        columns=["date", "stock_code", "indus_name_lv1"],
        filters=[("date", "=", latest)],
    )
    industry["code6"] = industry["stock_code"].astype("string").str.extract(r"([0-9]{6})", expand=False)
    industry = industry.drop_duplicates("code6")
    frame = latest_data.copy()
    frame["code6"] = frame["stock_code"].astype("string").str.extract(r"([0-9]{6})", expand=False)
    frame = frame.merge(industry[["code6", "indus_name_lv1"]], on="code6", how="left")
    frame["industry"] = frame["indus_name_lv1"].str.rsplit("_", n=1).str[-1]
    rows = []
    for factor, columns in FACTORS.items():
        valid = frame[columns["raw"]].notna() & frame["industry"].notna()
        temp = frame.loc[valid, ["industry", columns["raw"], columns["mad"]]].copy()
        temp["clipped"] = temp[columns["raw"]].ne(temp[columns["mad"]])
        grouped = (
            temp.groupby("industry", as_index=False)
            .agg(n_valid=(columns["raw"], "size"), n_clipped=("clipped", "sum"))
        )
        grouped["clip_rate"] = grouped["n_clipped"] / grouped["n_valid"]
        grouped.insert(0, "factor", factor)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def save_figures(
    data: pd.DataFrame,
    daily: pd.DataFrame,
    market: pd.DataFrame,
    size: pd.DataFrame,
    latest: pd.Timestamp,
    start: pd.Timestamp,
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    monthly = daily.loc[daily["date"].ge(start)].copy()
    monthly["month"] = monthly["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        monthly.groupby(["factor", "month"], as_index=False)
        .agg(n_valid=("n_valid", "sum"), n_clipped=("n_clipped", "sum"))
    )
    monthly["clip_rate"] = monthly["n_clipped"] / monthly["n_valid"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = {"EP": "#2F6B9A", "PE": "#C14F3F"}
    for factor, group in monthly.groupby("factor", sort=False):
        ax.plot(group["month"], group["clip_rate"] * 100, label=factor, color=colors[factor], linewidth=1.7)
    ax.axhline(2 * norm.sf(3) * 100, color="#666666", linestyle="--", linewidth=1, label="正态分布 ±3σ 理论尾部")
    ax.set_ylabel("MAD 截断比例（%）")
    ax.set_xlabel("")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(ncol=3, frameon=False)
    ax.set_title("正式测试期月度 MAD 截断比例")
    fig.tight_layout()
    fig.savefig(FIG / "monthly_clipping_rate.png", dpi=150)
    plt.close(fig)

    latest_data = data.loc[data["date"].eq(latest)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, (factor, columns) in enumerate(FACTORS.items()):
        raw = finite_values(latest_data[columns["raw"]])
        mad = finite_values(latest_data[columns["mad"]])
        z = finite_values(latest_data[columns["z"]])
        lo, hi = np.quantile(raw, [0.01, 0.99])
        bins = np.linspace(lo, hi, 100)
        axes[row, 0].hist(raw[(raw >= lo) & (raw <= hi)], bins=bins, density=True, alpha=0.45, color="#C14F3F", label="原始值")
        axes[row, 0].hist(mad[(mad >= lo) & (mad <= hi)], bins=bins, density=True, alpha=0.45, color="#2F6B9A", label="MAD 后")
        axes[row, 0].set_title(f"{factor}：原始值与 MAD 后（显示 1%--99%）")
        axes[row, 0].legend(frameon=False)
        axes[row, 0].grid(axis="y", color="#E1E1E1", linewidth=0.5)
        axes[row, 1].hist(z, bins=np.linspace(-4.5, 4.5, 100), color="#4B8B62", alpha=0.85)
        axes[row, 1].axvline(0, color="#555555", linewidth=0.8)
        axes[row, 1].set_title(f"{factor}：Z 标准化后")
        axes[row, 1].grid(axis="y", color="#E1E1E1", linewidth=0.5)
    fig.suptitle(f"{latest.date()} 最新截面的处理前后分布", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "latest_distribution_transform.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    market_plot = market.pivot(index="group", columns="factor", values="clip_rate").sort_index()
    market_plot.mul(100).plot(kind="bar", ax=axes[0], color=["#2F6B9A", "#C14F3F"])
    axes[0].set_title("正式测试期：按市场的截断比例")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("截断比例（%）")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(frameon=False)
    size_plot = size.pivot(index="group", columns="factor", values="clip_rate").reindex([1, 2, 3, 4, 5])
    size_plot.index = ["最小20%", "次小", "中间", "次大", "最大20%"]
    size_plot.mul(100).plot(kind="bar", ax=axes[1], color=["#2F6B9A", "#C14F3F"])
    axes[1].set_title("正式测试期：按市值五组的截断比例")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("截断比例（%）")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", color="#E1E1E1", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "clipping_by_market_and_size.png", dpi=150)
    plt.close(fig)


def build_report(
    data: pd.DataFrame,
    daily: pd.DataFrame,
    daily_summary: pd.DataFrame,
    annual: pd.DataFrame,
    distributions: pd.DataFrame,
    checks: pd.DataFrame,
    diagnostics: pd.DataFrame,
    market: pd.DataFrame,
    size: pd.DataFrame,
    sign: pd.DataFrame,
    industry: pd.DataFrame,
    latest_extremes: pd.DataFrame,
    start: pd.Timestamp,
    latest: pd.Timestamp,
) -> None:
    ep_full = daily_summary.query("factor == 'EP' and scope == '全样本'").iloc[0]
    pe_full = daily_summary.query("factor == 'PE' and scope == '全样本'").iloc[0]
    ep_formal = daily_summary.query("factor == 'EP' and scope == '正式测试期'").iloc[0]
    pe_formal = daily_summary.query("factor == 'PE' and scope == '正式测试期'").iloc[0]
    latest_ep = daily.query("factor == 'EP' and date == @latest").iloc[0]
    latest_pe = daily.query("factor == 'PE' and date == @latest").iloc[0]
    total_rows = len(data)
    valid_ep = int(data["raw_ep"].notna().sum())
    missing_ep = total_rows - valid_ep
    suspended_missing = int((data["raw_ep"].isna() & data["suspended"].fillna(False)).sum())
    nonsuspended_missing = missing_ep - suspended_missing
    normal_tail = float(2 * norm.sf(3))

    diag_summary = diagnostics.groupby("factor").agg(
        pearson_median=("raw_mad_pearson", "median"),
        pearson_min=("raw_mad_pearson", "min"),
        spearman_median=("raw_mad_spearman", "median"),
        spearman_min=("raw_mad_spearman", "min"),
        raw_skew_median=("raw_skew", "median"),
        mad_skew_median=("mad_skew", "median"),
        raw_kurt_median=("raw_excess_kurtosis", "median"),
        mad_kurt_median=("mad_excess_kurtosis", "median"),
    )

    lines = [
        "# MAD 去极值与 Z 标准化统计分析",
        "",
        f"**样本期：** 2020-01-02 至 {latest.date()}  ",
        f"**正式测试起点：** {start.date()}  ",
        "**范围：** 仅分析因子清洗与标准化，不包含回归或回测。",
        "",
        "> 本文中的“异常值”是指超出当日横截面 MAD 阈值的观测。它表示相对于当日股票池非常极端，不等同于数据错误，也不等同于应当删除的公司。",
        "",
        "## 1. 先看结论",
        "",
        f"- 原始行情面板共有 **{total_rows:,}** 条股票日记录，EP/PE 各有 **{valid_ep:,}** 条原始有效观测；缺失 **{missing_ep:,}** 条，其中停牌日解释 **{suspended_missing:,}** 条，其他财务或市值不可用解释 **{nonsuspended_missing:,}** 条。MAD 不填充缺失值。",
        f"- 全样本 EP 被截断 **{int(ep_full['clipped_observations']):,}** 条，占有效值 **{pct(ep_full['weighted_clip_rate'])}**；PE 被截断 **{int(pe_full['clipped_observations']):,}** 条，占 **{pct(pe_full['weighted_clip_rate'])}**。正式测试期分别为 **{pct(ep_formal['weighted_clip_rate'])}** 和 **{pct(pe_formal['weighted_clip_rate'])}**。",
        f"- 若横截面严格接近正态分布，均值正负 3 个标准差以外理论上约有 **{pct(normal_tail)}**。实际截断率明显更高，说明估值因子具有厚尾、偏态和异质群体；不能把所有被截断值都解释成录入错误。",
        f"- 最新截面 {latest.date()}：EP 阈值为 **[{num(latest_ep['lower'], 5)}, {num(latest_ep['upper'], 5)}]**，截断 **{int(latest_ep['n_clipped']):,}/{int(latest_ep['n_valid']):,}（{pct(latest_ep['clip_rate'])}）**；PE 阈值为 **[{num(latest_pe['lower'], 2)}, {num(latest_pe['upper'], 2)}]**，截断 **{int(latest_pe['n_clipped']):,}/{int(latest_pe['n_valid']):,}（{pct(latest_pe['clip_rate'])}）**。",
        "- Z 标准化后，每个非恒定交易日的横截面均值在数值误差内等于 0、总体标准差等于 1。但标准化不会使分布自动变成正态分布，也不会修复缺失值、错误财报或证券映射。",
        "- PE 的截断率高于 EP，主要因为 `PE = 1 / EP` 在 EP 接近零时会放大成巨大正负值。因此主测试使用 EP 更稳定，原始 PE 更适合用中位数和分位数解释。",
        "",
        "## 2. 实际处理公式与为什么这样做",
        "",
        "对每个交易日、每个因子分别计算：",
        "",
        "```text",
        "median_t = median(x_i,t)",
        "MAD_t    = median(abs(x_i,t - median_t))",
        "scale_t  = 1.4826 * MAD_t",
        "lower_t  = median_t - 3 * scale_t",
        "upper_t  = median_t + 3 * scale_t",
        "x_mad    = clip(x, lower_t, upper_t)",
        "z_i,t    = (x_mad_i,t - mean_t(x_mad)) / std_t(x_mad, ddof=0)",
        "```",
        "",
        "### 2.1 为什么用中位数和 MAD",
        "",
        "均值和标准差会被少量极端 PE/EP 明显拉动；中位数与 MAD 对尾部观测更不敏感。`1.4826` 是正态一致性系数，使 MAD 尺度在正态分布下与标准差可比较。`3 × scale` 是透明、可复现的基准，不代表所有越界观测都是错误数据。",
        "",
        "### 2.2 为什么截断而不是删除",
        "",
        "截断保留股票和样本量，只限制极端值对均值、标准差和回归斜率的影响。代价是所有超过同一端阈值的股票会得到相同 MAD 值，尾部内部的精细排序变成并列。因此原始值必须保留，用于异常追溯和经济含义解释。",
        "",
        "### 2.3 为什么按日做 Z 标准化",
        "",
        "每日标准化消除不同交易日估值水平与离散程度的变化，使跨月回归系数具有统一单位：因子增加 1 个当日横截面标准差。它适合排序、回归和组合构造，但不再直接表示 EP 比率或 PE 倍数，因此报告经济水平时仍应使用原始值。",
        "",
        "## 3. 样本有效性与处理数量",
        "",
    ]
    lines += markdown_table(
        ["因子", "范围", "有效观测", "截断观测", "加权截断率", "每日截断率中位数", "每日截断率 10%--90%", "恒定截面"],
        [
            [
                row.factor,
                row.scope,
                f"{int(row.valid_observations):,}",
                f"{int(row.clipped_observations):,}",
                pct(row.weighted_clip_rate),
                pct(row.daily_clip_rate_median),
                f"{pct(row.daily_clip_rate_p10)} -- {pct(row.daily_clip_rate_p90)}",
                str(int(row.constant_cross_sections)),
            ]
            for row in daily_summary.itertuples(index=False)
        ],
    )
    lines += [
        "",
        "全样本最初只有 1 个有效观测的截面无法计算非零标准差，因此产生 1 条额外 Z 缺失；正式测试期没有恒定截面。MAD 本身没有改变有效观测数量。",
        "",
        "![正式测试期月度 MAD 截断比例](./标准化统计/图形/monthly_clipping_rate.png)",
        "",
        "图中的正态参考线只用于理解尾部厚度。MAD 阈值以中位数为中心，并不要求实际分布服从正态。",
        "",
        "## 4. 原始值、MAD 值和 Z 值分别告诉我们什么",
        "",
        "- **原始值**：保留绝对经济含义，可回答盈利收益率是多少、PE 是多少倍，也最适合排查财报和市值异常；缺点是均值、标准差和 Pearson 相关容易被极端值主导。",
        "- **MAD 值**：仍保留原因子单位，但限制尾部影响，可观察截断对均值、波动、偏度和峰度的改变；尾部内部排序会产生并列。",
        "- **Z 值**：表示股票相对当日横截面的标准差位置，适合跨期比较和回归；不再保留原始倍数含义。Z 是 MAD 值的正线性变换，所以当日排序、偏度和峰度不会再次改变。",
        "",
        "### 4.1 全样本池化分布",
        "",
    ]
    dist_rows = []
    for row in distributions.query("scope == '全样本'").itertuples(index=False):
        dist_rows.append([
            row.factor,
            row.stage,
            f"{int(row.n):,}",
            num(row.mean),
            num(row.std),
            num(row.median),
            f"{num(row.p1)} / {num(row.p99)}",
            num(row.skew, 2),
            num(row.excess_kurtosis, 2),
        ])
    lines += markdown_table(
        ["因子", "阶段", "观测数", "均值", "标准差", "中位数", "1% / 99%", "偏度", "超额峰度"],
        dist_rows,
    )
    lines += [
        "",
        "池化统计把多个交易日放在一起，既包含横截面形状，也包含时间变化，不能替代逐日诊断。尤其是 Z 值应先按日理解，再用池化统计观察总体尾部。",
        "",
        f"![最新截面处理前后分布](./标准化统计/图形/latest_distribution_transform.png)",
        "",
        "Z 分布两端出现的尖柱来自截断端点堆积：所有越过同一侧 MAD 阈值的观测都会被压到同一个数值，再经过同一组均值和标准差变换。它是缩尾处理的机械结果，不代表端点附近聚集了大量原始观测。",
        "",
        "### 4.2 月末截面的形状与排序变化",
        "",
    ]
    diag_rows = []
    for factor, row in diag_summary.iterrows():
        diag_rows.append([
            factor,
            num(row.pearson_median, 4),
            num(row.pearson_min, 4),
            num(row.spearman_median, 4),
            num(row.spearman_min, 4),
            num(row.raw_skew_median, 2),
            num(row.mad_skew_median, 2),
            num(row.raw_kurt_median, 2),
            num(row.mad_kurt_median, 2),
        ])
    lines += markdown_table(
        ["因子", "Pearson 中位", "Pearson 最低", "Spearman 中位", "Spearman 最低", "原始偏度", "MAD后偏度", "原始峰度", "MAD后峰度"],
        diag_rows,
    )
    lines += [
        "",
        "Spearman 相关高说明截断总体保留了排序；低于 1 的部分来自上下端点并列。Pearson 相关反映数值距离变化，PE 通常受近零 EP 产生的极端倒数影响更明显。MAD 后偏度和峰度下降，说明尾部对二阶及高阶统计量的支配被削弱。",
        "",
        "## 5. Z 标准化验收",
        "",
    ]
    lines += markdown_table(
        ["因子", "标准化交易日", "日均值绝对值最大值", "日标准差偏离 1 的最大值", "池化均值", "池化标准差", "Z 绝对值大于 1", "Z 绝对值大于 2", "Z 绝对值大于 3"],
        [
            [
                row.factor,
                str(int(row.standardized_days)),
                f"{row.max_abs_daily_mean:.2e}",
                f"{row.max_abs_daily_std_minus_one:.2e}",
                f"{row.pooled_mean:.2e}",
                num(row.pooled_std, 6),
                pct(row.abs_z_gt_1_ratio),
                pct(row.abs_z_gt_2_ratio),
                pct(row.abs_z_gt_3_ratio),
            ]
            for row in checks.itertuples(index=False)
        ],
    )
    lines += [
        "",
        "均值为 0、标准差为 1 是标准化的机械结果，不是因子有效性的证据。`|Z|>2` 也不等同于原始异常值：MAD 截断和随后标准化使用的是两套尺度，且处理后分布仍可偏态或厚尾。",
        "",
        "## 6. 异常值集中在哪里",
        "",
        "### 6.1 按年份",
        "",
    ]
    lines += markdown_table(
        ["因子", "年份", "交易日", "有效观测", "截断观测", "截断率", "日中位数的年度中位"],
        [
            [row.factor, str(row.year), str(int(row.trading_days)), f"{int(row.n_valid):,}", f"{int(row.n_clipped):,}", pct(row.clip_rate), num(row.median_daily_median)]
            for row in annual.itertuples(index=False)
        ],
    )
    lines += [
        "",
        "年度截断率变化可以识别因子分布是否发生结构漂移。它只能说明横截面尾部相对当年阈值的变化，不能单独解释为市场风险上升或数据质量下降。",
        "",
        "### 6.2 按市场、市值和正负值",
        "",
        "以下市场、市值和正负值分组均使用 2020-07-31 起的正式测试期样本。市值组在每个交易日分别按总市值五等分，因此反映的是相对市值层级，而不是固定市值门槛。",
        "",
        "![按市场和市值组的截断比例](./标准化统计/图形/clipping_by_market_and_size.png)",
        "",
    ]
    market_rows = []
    for row in market.itertuples(index=False):
        market_rows.append([row.factor, str(row.group), f"{int(row.n_valid):,}", pct(row.negative_ratio), f"{int(row.lower_clipped):,}", f"{int(row.upper_clipped):,}", pct(row.clip_rate)])
    lines += markdown_table(
        ["因子", "市场", "有效观测", "负值占比", "下尾截断", "上尾截断", "截断率"],
        market_rows,
    )
    lines += ["", "市值五组详细结果保存在 `标准化统计/clipping_by_size.csv`。分组结果用于定位极端值来源，不应据此在生产阶段删除某个市场或市值组。", ""]
    sign_rows = []
    for row in sign.itertuples(index=False):
        sign_rows.append([row.factor, str(row.group), f"{int(row.n_valid):,}", f"{int(row.lower_clipped):,}", f"{int(row.upper_clipped):,}", pct(row.clip_rate)])
    lines += markdown_table(
        ["因子", "原始值符号", "有效观测", "下尾截断", "上尾截断", "截断率"],
        sign_rows,
    )

    if not industry.empty:
        lines += ["", "### 6.3 最新截面的行业集中度", ""]
        top_industry = industry.loc[industry["n_valid"].ge(30)].sort_values(["factor", "clip_rate"], ascending=[True, False]).groupby("factor").head(5)
        lines += markdown_table(
            ["因子", "行业", "有效观测", "截断观测", "截断率"],
            [[row.factor, row.industry, str(int(row.n_valid)), str(int(row.n_clipped)), pct(row.clip_rate)] for row in top_industry.itertuples(index=False)],
        )
        lines += ["", "行业截断率可能来自真实商业模式差异、周期亏损或估值结构，不应直接作为数据错误清单。完整结果见 `标准化统计/latest_clipping_by_industry.csv`。", ""]

    ep_extreme = latest_extremes.query("factor == 'EP'")
    lines += [
        "## 7. 最新截面异常样本如何阅读",
        "",
        "下表列出最新截面 EP 最低和最高的各 3 个样本。被截断后，同一侧越过阈值的股票会落在相同端点，因此 Z 值不能继续区分尾部内部的原始差异。",
        "",
    ]
    lines += markdown_table(
        ["方向", "股票代码", "市场", "原始 EP", "MAD 后 EP", "Z EP", "总市值"],
        [[row.tail, str(row.stock_code), str(row.market_scope), num(row.raw_value, 6), num(row.mad_value, 6), num(row.z_value, 4), num(row.me_total, 0)] for row in ep_extreme.groupby("tail", sort=False).head(3).itertuples(index=False)],
    )
    lines += [
        "",
        "完整 EP/PE 最新极端样本见 `标准化统计/latest_extreme_observations.csv`。核查顺序应是：财报公告与版本、TTM 构造、市值单位、停牌状态、证券代码映射，最后才判断是否为真实经济极值。",
        "",
        "## 8. 可以得到的信息与不能得到的结论",
        "",
        "### 可以得到",
        "",
        "1. **数据质量线索**：极端值是否集中在特定日期、市场、行业或市值组；是否可能来自单位、代码映射或财报版本错误。",
        "2. **分布稳定性**：每日中位数、MAD、阈值和截断率是否随时间突变，从而决定测试起点及是否需要分阶段处理。",
        "3. **模型敏感性**：原始值与 MAD 值的 Pearson/Spearman 差异、偏度和峰度变化，可判断回归是否容易被少量极端值主导。",
        "4. **横截面可比性**：Z 值验收确认每日均值和尺度统一，使不同月份的因子暴露和回归系数可比较。",
        "5. **经济解释边界**：原始 EP/PE 用于解释绝对估值；Z 值只表示相对位置，两者用途不能互换。",
        "",
        "### 不能直接得到",
        "",
        "1. 截断比例高不能证明数据错误，也不能证明因子有效；",
        "2. Z 值近似均值 0、标准差 1 不能证明服从正态分布；",
        "3. 去极值不能解决未来数据、幸存者偏差、行业映射错误或横截面相关性；",
        "4. 描述统计不能替代 Rank IC、横截面回归和分层回测；",
        "5. 当前 3 倍 MAD 参数是研究选择，应在样本外测试前固定，并用 3.5 倍、5 倍或不截断作为稳健性对照，而不是根据回测结果反向调参。",
        "",
        "## 9. 建议的下游使用方式",
        "",
        "- 数据排查和经济解释：使用 `raw_ep/raw_pe`；",
        "- 观察去极值影响：使用 `mad_ep/mad_pe` 与原始值对照；",
        "- 横截面回归、Rank IC 和分层排序：主输入使用 `z_ep`，PE 仅作稳健性检验；",
        "- 每次重跑必须保存每日 `median/MAD/lower/upper/n_clipped`，并监控截断率突变；",
        "- 不使用行业中位数填充缺失因子；缺失本身可能包含经营、上市时间或数据可得性信息。",
        "",
        "## 10. 输出文件",
        "",
        "- `标准化统计/stage_distribution_stats.csv`：原始、MAD、Z 三阶段分布；",
        "- `标准化统计/daily_processing_summary.csv`：全样本与正式期处理汇总；",
        "- `标准化统计/annual_clipping_summary.csv`：年度截断统计；",
        "- `标准化统计/month_end_diagnostics.csv`：月末排序、偏度、峰度变化；",
        "- `标准化统计/standardization_checks.csv`：逐日 Z 验收汇总；",
        "- `标准化统计/clipping_by_market.csv`、`clipping_by_size.csv`、`clipping_by_sign.csv`：异常值结构；",
        "- `标准化统计/latest_clipping_by_industry.csv`：最新行业截断统计；",
        "- `标准化统计/latest_extreme_observations.csv`：最新截面极端观测。",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    with (DATA / "factor_test_start.json").open("r", encoding="utf-8") as file:
        config = json.load(file)
    start = pd.Timestamp(config["formal_test_start_date"])

    columns = [
        "date", "market_scope", "me_total", "suspended",
        "raw_ep", "mad_ep", "z_ep", "raw_pe", "mad_pe", "z_pe",
    ]
    data = pd.read_parquet(DATA / "factor_audit.parquet", columns=columns)
    data["date"] = pd.to_datetime(data["date"])
    latest = pd.Timestamp(data["date"].max())
    formal = data["date"].ge(start)

    daily = load_processing_stats()
    daily_summary, annual = aggregate_daily(daily, start)

    distribution_rows = []
    scopes = {
        "全样本": pd.Series(True, index=data.index),
        "正式测试期": formal,
        "最新截面": data["date"].eq(latest),
    }
    stage_labels = {"raw": "原始值", "mad": "MAD后", "z": "Z值"}
    for scope, mask in scopes.items():
        for factor, factor_columns in FACTORS.items():
            for stage, column in factor_columns.items():
                row = distribution_stats(data.loc[mask, column])
                row.update({"scope": scope, "factor": factor, "stage": stage_labels[stage]})
                distribution_rows.append(row)
    distributions = pd.DataFrame(distribution_rows)

    checks = standardization_checks(data, daily)
    diagnostics = monthly_cross_section_diagnostics(data, daily, start)

    market_frames = []
    size_frames = []
    sign_frames = []
    market_group = data["market_scope"].astype("string").fillna("未知")
    size_valid = formal & data["raw_ep"].notna() & data["me_total"].gt(0)
    size_pct = data.loc[size_valid].groupby("date")["me_total"].rank(method="first", pct=True)
    size_group = pd.Series(pd.NA, index=data.index, dtype="Int8")
    size_group.loc[size_valid] = np.ceil(size_pct * 5).clip(1, 5).astype("int8")
    for factor, factor_columns in FACTORS.items():
        market_frames.append(clipping_group_summary(data, market_group, factor, formal))
        size_frames.append(clipping_group_summary(data, size_group, factor, formal))
        raw = data[factor_columns["raw"]]
        sign_group = pd.Series(pd.NA, index=data.index, dtype="string")
        sign_group.loc[raw.lt(0)] = "负值"
        sign_group.loc[raw.ge(0)] = "非负值"
        sign_frames.append(clipping_group_summary(data, sign_group, factor, formal))
    market = pd.concat(market_frames, ignore_index=True)
    size = pd.concat(size_frames, ignore_index=True)
    sign = pd.concat(sign_frames, ignore_index=True)

    latest_columns = [
        "date", "stock_code", "market_scope", "me_total", "TTM_ni",
        "raw_ep", "mad_ep", "z_ep", "raw_pe", "mad_pe", "z_pe",
    ]
    latest_data = pd.read_parquet(
        DATA / "factor_audit.parquet",
        columns=latest_columns,
        filters=[("date", "=", latest)],
    )
    latest_data["date"] = pd.to_datetime(latest_data["date"])
    extreme_rows = []
    for factor, factor_columns in FACTORS.items():
        valid = latest_data.loc[latest_data[factor_columns["raw"]].notna()].copy()
        for tail, sample in {
            "最低": valid.nsmallest(10, factor_columns["raw"]),
            "最高": valid.nlargest(10, factor_columns["raw"]),
        }.items():
            for row in sample.itertuples(index=False):
                extreme_rows.append(
                    {
                        "factor": factor,
                        "tail": tail,
                        "date": row.date,
                        "stock_code": row.stock_code,
                        "market_scope": row.market_scope,
                        "me_total": row.me_total,
                        "TTM_ni": row.TTM_ni,
                        "raw_value": getattr(row, factor_columns["raw"]),
                        "mad_value": getattr(row, factor_columns["mad"]),
                        "z_value": getattr(row, factor_columns["z"]),
                    }
                )
    latest_extremes = pd.DataFrame(extreme_rows)
    industry = latest_industry_summary(latest_data, latest)

    daily_summary.to_csv(OUT / "daily_processing_summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUT / "annual_clipping_summary.csv", index=False, encoding="utf-8-sig")
    distributions.to_csv(OUT / "stage_distribution_stats.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(OUT / "standardization_checks.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(OUT / "month_end_diagnostics.csv", index=False, encoding="utf-8-sig")
    market.to_csv(OUT / "clipping_by_market.csv", index=False, encoding="utf-8-sig")
    size.to_csv(OUT / "clipping_by_size.csv", index=False, encoding="utf-8-sig")
    sign.to_csv(OUT / "clipping_by_sign.csv", index=False, encoding="utf-8-sig")
    industry.to_csv(OUT / "latest_clipping_by_industry.csv", index=False, encoding="utf-8-sig")
    latest_extremes.to_csv(OUT / "latest_extreme_observations.csv", index=False, encoding="utf-8-sig")

    save_figures(data, daily, market, size, latest, start)
    build_report(
        data, daily, daily_summary, annual, distributions, checks, diagnostics,
        market, size, sign, industry, latest_extremes, start, latest,
    )
    print(f"saved report: {REPORT}")
    print(f"saved tables: {OUT}")
    print(f"latest date: {latest.date()}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
