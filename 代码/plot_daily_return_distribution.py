# -*- coding: utf-8 -*-
"""Plot one trading day's cross-sectional adjusted-return distribution."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
MARKET_PATH = Path(r"D:\实习生学习项目\基础数据\chn_equ_mkt_quotation.parquet")
DEFAULT_DATE = pd.Timestamp("2025-12-29")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def normalize_code(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def b_share_mask(series: pd.Series) -> pd.Series:
    code = normalize_code(series)
    sh_b = code.str.startswith("900", na=False) & code.str.endswith(".BJ", na=False)
    sz_b = code.str.startswith(("200", "201"), na=False) & code.str.endswith(
        ".SZ", na=False
    )
    return sh_b | sz_b


def read_date(path: Path, target_date: pd.Timestamp) -> pd.DataFrame:
    columns = [
        "date",
        "stock_code",
        "close_adj",
        "pre_close_adj",
        "status",
        "suspended",
    ]
    pieces: list[pd.DataFrame] = []
    parquet = pq.ParquetFile(path)
    date_index = parquet.schema_arrow.names.index("date")

    for row_group_index in range(parquet.num_row_groups):
        metadata = parquet.metadata.row_group(row_group_index)
        date_stats = metadata.column(date_index).statistics
        if date_stats is not None:
            minimum = pd.Timestamp(date_stats.min).normalize()
            maximum = pd.Timestamp(date_stats.max).normalize()
            if target_date < minimum or target_date > maximum:
                continue
        frame = parquet.read_row_group(row_group_index, columns=columns).to_pandas()
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        selected = frame.loc[frame["date"].eq(target_date)].copy()
        if not selected.empty:
            pieces.append(selected)

    if not pieces:
        raise ValueError(f"行情表中没有交易日 {target_date.date()}")
    return pd.concat(pieces, ignore_index=True)


def calculate_returns(frame: pd.DataFrame) -> tuple[pd.Series, dict[str, float | int]]:
    b_share = b_share_mask(frame["stock_code"])
    suspended = pd.to_numeric(frame["suspended"], errors="coerce").eq(1)
    suspended |= frame["status"].astype("string").str.strip().eq("停牌")

    close = pd.to_numeric(frame["close_adj"], errors="coerce")
    pre_close = pd.to_numeric(frame["pre_close_adj"], errors="coerce")
    daily_return = close / pre_close - 1.0
    valid_price = np.isfinite(daily_return) & close.gt(0) & pre_close.gt(0)
    valid = ~b_share & ~suspended & valid_price
    values = daily_return.loc[valid].astype(float)

    mean = float(values.mean())
    std = float(values.std(ddof=1))
    centered = (values - mean).abs()
    central_half_sigma = float(centered.le(0.5 * std).mean())
    tail_three_sigma = float(centered.gt(3.0 * std).mean())
    jb = stats.jarque_bera(values)

    summary: dict[str, float | int] = {
        "raw_rows": int(len(frame)),
        "excluded_b_share_rows": int(b_share.sum()),
        "excluded_suspended_rows": int((~b_share & suspended).sum()),
        "invalid_price_rows": int((~b_share & ~suspended & ~valid_price).sum()),
        "n": int(len(values)),
        "mean": mean,
        "median": float(values.median()),
        "std": std,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "skewness": float(stats.skew(values, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(values, fisher=True, bias=False)),
        "central_half_sigma_observed": central_half_sigma,
        "central_half_sigma_normal": float(2 * stats.norm.cdf(0.5) - 1),
        "tail_three_sigma_count": int(centered.gt(3.0 * std).sum()),
        "tail_three_sigma_observed": tail_three_sigma,
        "tail_three_sigma_normal": float(2 * stats.norm.sf(3.0)),
        "jarque_bera": float(jb.statistic),
        "jarque_bera_p": float(jb.pvalue),
    }
    return values, summary


def draw_chart(
    values: pd.Series,
    summary: dict[str, float | int],
    target_date: pd.Timestamp,
    output_path: Path,
) -> None:
    returns_pct = values.to_numpy() * 100.0
    mean_pct = float(summary["mean"]) * 100.0
    std_pct = float(summary["std"]) * 100.0

    bin_width = 0.5
    lower = np.floor(returns_pct.min() / bin_width) * bin_width
    upper = np.ceil(returns_pct.max() / bin_width) * bin_width
    edges = np.arange(lower, upper + bin_width * 1.01, bin_width)
    centers = (edges[:-1] + edges[1:]) / 2.0
    x_grid = np.linspace(lower, upper, 1_500)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    color = "#3568a8"
    normal_color = "#c43b3b"

    axes[0].hist(
        returns_pct,
        bins=edges,
        density=True,
        color=color,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.9,
        label="实际横截面分布",
    )
    axes[0].plot(
        x_grid,
        stats.norm.pdf(x_grid, loc=mean_pct, scale=std_pct),
        color=normal_color,
        linewidth=2.0,
        label="同均值、同标准差正态分布",
    )
    zoom = max(10.0, min(12.0, 4.0 * std_pct))
    axes[0].set_xlim(mean_pct - zoom, mean_pct + zoom)
    axes[0].set_title("中央区域：实际分布峰值更尖")
    axes[0].set_xlabel("单日复权收益率（%）")
    axes[0].set_ylabel("概率密度")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.7)

    counts, _ = np.histogram(returns_pct, bins=edges)
    expected = len(returns_pct) * (
        stats.norm.cdf(edges[1:], loc=mean_pct, scale=std_pct)
        - stats.norm.cdf(edges[:-1], loc=mean_pct, scale=std_pct)
    )
    axes[1].bar(
        centers,
        counts,
        width=bin_width * 0.92,
        color=color,
        alpha=0.9,
        label="实际样本数",
    )
    expected_visible = np.where(expected >= 0.05, expected, np.nan)
    axes[1].plot(
        centers,
        expected_visible,
        color=normal_color,
        linewidth=2.0,
        label="正态分布预期样本数",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=0.5)
    axes[1].set_xlim(lower - 0.5, upper + 0.5)
    axes[1].set_title("对数纵轴：实际分布尾部更厚")
    axes[1].set_xlabel("单日复权收益率（%）")
    axes[1].set_ylabel("样本数（对数）")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(axis="y", which="both", color="#d9d9d9", linewidth=0.6, alpha=0.7)

    stats_text = (
        f"有效样本 {int(summary['n']):,}  均值 {mean_pct:.3f}%  "
        f"标准差 {std_pct:.3f}%\n"
        f"偏度 {float(summary['skewness']):.2f}  "
        f"超额峰度 {float(summary['excess_kurtosis']):.2f}\n"
        f"均值 +/-0.5 sigma 内：实际 "
        f"{100 * float(summary['central_half_sigma_observed']):.2f}% / "
        f"正态 {100 * float(summary['central_half_sigma_normal']):.2f}%\n"
        f"均值 +/-3 sigma 外：实际 "
        f"{100 * float(summary['tail_three_sigma_observed']):.2f}% / "
        f"正态 {100 * float(summary['tail_three_sigma_normal']):.2f}%"
    )
    figure.text(
        0.5,
        0.01,
        stats_text,
        ha="center",
        va="bottom",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f5f5f5", "edgecolor": "#b8b8b8"},
    )
    figure.suptitle(
        f"{target_date.date()} 股票横截面单日收益率：尖峰厚尾",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0.16, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=str(DEFAULT_DATE.date()))
    arguments = parser.parse_args()
    target_date = pd.Timestamp(arguments.date).normalize()

    frame = read_date(MARKET_PATH, target_date)
    values, summary = calculate_returns(frame)
    date_tag = target_date.strftime("%Y%m%d")
    output_path = ROOT / "图形" / f"daily_return_distribution_{date_tag}.png"
    stats_path = ROOT / "中间结果" / f"daily_return_stats_{date_tag}.csv"

    draw_chart(values, summary, target_date, output_path)
    pd.DataFrame([{**{"date": target_date.date()}, **summary}]).to_csv(
        stats_path, index=False, encoding="utf-8-sig"
    )

    print(f"日期: {target_date.date()}")
    print(f"有效样本: {int(summary['n']):,}")
    print(f"均值: {100 * float(summary['mean']):.4f}%")
    print(f"中位数: {100 * float(summary['median']):.4f}%")
    print(f"标准差: {100 * float(summary['std']):.4f}%")
    print(f"偏度: {float(summary['skewness']):.4f}")
    print(f"超额峰度: {float(summary['excess_kurtosis']):.4f}")
    print(
        "三倍标准差外: "
        f"{int(summary['tail_three_sigma_count'])} 条 "
        f"({100 * float(summary['tail_three_sigma_observed']):.2f}%)"
    )
    print(f"图形: {output_path}")
    print(f"统计: {stats_path}")


if __name__ == "__main__":
    main()
