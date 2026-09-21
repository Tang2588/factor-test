# -*- coding: utf-8 -*-
"""Plot latest raw PE medians by Shenwan level-1 industry.

This is descriptive analysis only. Raw signed PE observations are retained;
industry medians are used because raw PE means are unstable near zero EP.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


BASE = Path(r"D:\因子计算\市盈率\EP因子_交付版")
PE_PATH = BASE / "中间结果" / "pe_raw.parquet"
INDUSTRY_PATH = Path(r"D:\实习生学习项目\基础数据\chn_equ_indus_sw.parquet")
CSV_PATH = BASE / "中间结果" / "industry_pe_latest.csv"
PNG_PATH = BASE / "图形" / "industry_pe_latest.png"
TEX_PATH = BASE / "图形" / "行业PE柱状图.tex"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def latest_parquet_date(path: Path) -> pd.Timestamp:
    parquet = pq.ParquetFile(path)
    date_col = parquet.schema_arrow.names.index("date")
    maxima = []
    for i in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(i).column(date_col).statistics
        if stats is not None and stats.has_min_max:
            maxima.append(pd.Timestamp(stats.max))
    if maxima:
        return max(maxima).normalize()

    latest = None
    for batch in parquet.iter_batches(columns=["date"], batch_size=200_000):
        batch_max = pd.to_datetime(batch.column(0).to_pandas()).max()
        latest = batch_max if latest is None else max(latest, batch_max)
    if latest is None:
        raise ValueError(f"No dates found in {path}")
    return pd.Timestamp(latest).normalize()


def code6(series: pd.Series) -> pd.Series:
    return series.astype("string").str.extract(r"([0-9]{6})", expand=False)


def load_latest_industry_pe() -> tuple[pd.Timestamp, pd.DataFrame, dict[str, int | float]]:
    latest = latest_parquet_date(PE_PATH)
    pe = pd.read_parquet(PE_PATH, filters=[("date", "=", latest)]).reset_index()
    industry = pd.read_parquet(
        INDUSTRY_PATH,
        columns=["date", "stock_code", "indus_name_lv1"],
        filters=[("date", "=", latest)],
    )

    pe["code6"] = code6(pe["stock_code"])
    industry["code6"] = code6(industry["stock_code"])
    if pe["code6"].isna().any() or industry["code6"].isna().any():
        raise ValueError("Unable to normalize one or more stock codes")
    if pe.duplicated("code6").any():
        raise ValueError("Duplicate stock codes in latest raw PE cross-section")
    if industry.duplicated("code6").any():
        raise ValueError("Duplicate stock codes in latest industry cross-section")

    merged = pe.merge(
        industry[["code6", "indus_name_lv1"]],
        on="code6",
        how="left",
        validate="one_to_one",
    )
    merged["signal"] = pd.to_numeric(merged["signal"], errors="coerce")
    finite_pe = np.isfinite(merged["signal"])
    valid = merged.loc[finite_pe & merged["indus_name_lv1"].notna()].copy()
    valid["industry"] = valid["indus_name_lv1"].str.rsplit("_", n=1).str[-1]

    grouped = (
        valid.groupby(["indus_name_lv1", "industry"], as_index=False)
        .agg(
            observations=("signal", "size"),
            median_pe=("signal", "median"),
            mean_pe=("signal", "mean"),
            negative_ratio=("signal", lambda x: float((x < 0).mean())),
            positive_median_pe=("signal", lambda x: float(x[x > 0].median())),
        )
        .sort_values("median_pe", ascending=False)
        .reset_index(drop=True)
    )
    if grouped["industry"].duplicated().any():
        raise ValueError("Duplicate short industry names")

    diagnostics = {
        "cross_section_rows": len(merged),
        "valid_pe": int(finite_pe.sum()),
        "valid_industry_pe": len(valid),
        "valid_pe_missing_industry": int((finite_pe & merged["indus_name_lv1"].isna()).sum()),
        "industries": len(grouped),
        "market_median_pe": float(merged.loc[finite_pe, "signal"].median()),
    }
    return latest, grouped, diagnostics


def save_png(stats: pd.DataFrame, latest: pd.Timestamp, market_median: float) -> None:
    plot_data = stats.sort_values("median_pe", ascending=True)
    values = plot_data["median_pe"].to_numpy()
    colors = np.where(values < 0, "#B64949", "#3B6FB6")

    fig, ax = plt.subplots(figsize=(10.5, 12.5))
    bars = ax.barh(plot_data["industry"], values, color=colors, height=0.68)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.axvline(
        market_median,
        color="#C0392B",
        linestyle="--",
        linewidth=1.2,
        label=f"全市场原始 PE 中位数 {market_median:.2f} 倍",
    )
    ax.xaxis.grid(True, color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("行业内原始 PE 中位数（倍）")
    ax.set_title(f"申万一级行业原始 PE 中位数（{latest.date()}）", pad=14)

    span = values.max() - values.min()
    offset = max(span * 0.012, 0.5)
    for bar, value in zip(bars, values):
        x = value + offset if value >= 0 else value - offset
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.5,
        )
    ax.set_xlim(values.min() - 5, values.max() + 9)
    ax.legend(
        handles=[
            Patch(facecolor="#3B6FB6", label="行业中位 PE 为正"),
            Patch(facecolor="#B64949", label="行业中位 PE 为负"),
            plt.Line2D([], [], color="#C0392B", linestyle="--", label=f"全市场中位数 {market_median:.2f}"),
        ],
        loc="lower right",
        fontsize=9,
        frameon=False,
    )
    fig.text(
        0.5,
        0.015,
        "口径：原始 PE=1/EP；保留亏损公司的负 PE；按行业取中位数；未经 MAD 去极值或 Z 标准化。",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.035, 0.98, 0.98))
    fig.savefig(PNG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_latex_fragment(
    stats: pd.DataFrame,
    latest: pd.Timestamp,
    diagnostics: dict[str, int | float],
) -> None:
    ascending = stats.sort_values("median_pe", ascending=True)
    labels = ",\n    ".join(ascending["industry"])
    positive = "\n    ".join(
        f"({row.median_pe:.4f},{row.industry})"
        for row in ascending.itertuples()
        if row.median_pe >= 0
    )
    negative = "\n    ".join(
        f"({row.median_pe:.4f},{row.industry})"
        for row in ascending.itertuples()
        if row.median_pe < 0
    )
    top = stats.iloc[0]
    second = stats.iloc[1]
    third = stats.iloc[2]
    bottom = stats.iloc[-1]
    market_median = float(diagnostics["market_median_pe"])
    date_text = f"{latest.year} 年 {latest.month} 月 {latest.day} 日"

    content = rf"""% 正文片段。主文档导言区需加入：\usepackage{{pgfplots}}
% 并设置：\pgfplotsset{{compat=1.18}}
% 建议使用 XeLaTeX 编译中文标签。
\subsection{{分行业原始 PE}}

图~\ref{{fig:industry-raw-pe}} 使用 {date_text}最新截面，按申万一级行业汇总
未经 MAD 去极值和 Z 标准化的原始 PE。为避免接近零的 EP 使 PE 均值被极端值主导，
柱高采用行业内原始 PE 的中位数。亏损公司的负 PE 仍保留在统计中。

\begin{{figure}}[p]
  \centering
  \pgfplotsset{{compat=1.18}}
  \begin{{tikzpicture}}
    \begin{{axis}}[
      xbar,
      width=0.94\textwidth,
      height=0.80\textheight,
      xmin=-8,
      xmax=68,
      bar width=4.2pt,
      symbolic y coords={{
        {labels}
      }},
      ytick={{
        {labels}
      }},
      yticklabel style={{font=\scriptsize}},
      xlabel={{行业内原始 PE 中位数（倍）}},
      xlabel style={{font=\small}},
      xtick distance=10,
      xmajorgrids=true,
      grid style={{gray!25}},
      axis line style={{gray!65}},
      tick style={{gray!65}},
      enlarge y limits=0.012,
      point meta=x,
      nodes near coords={{\pgfmathprintnumber[fixed,precision=2]{{\pgfplotspointmeta}}}},
      every node near coord/.append style={{font=\tiny}},
    ]
      \addplot+[
        draw=none,
        fill={{rgb,255:red,59;green,111;blue,182}},
        bar shift=0pt
      ] coordinates {{
        {positive}
      }};
      \addplot+[
        draw=none,
        fill={{rgb,255:red,182;green,73;blue,73}},
        bar shift=0pt
      ] coordinates {{
        {negative}
      }};
      \draw[red!70!black,dashed,line width=0.8pt]
        (axis cs:{market_median:.4f},{bottom.industry}) --
        (axis cs:{market_median:.4f},{top.industry});
    \end{{axis}}
  \end{{tikzpicture}}
  \caption{{{date_text}申万一级行业原始 PE 中位数。样本包含
  {int(diagnostics['valid_industry_pe']):,} 条 PE 和行业分类均有效的股票记录；
  红色虚线为全市场原始 PE 中位数 {market_median:.2f} 倍，红色柱表示行业中位 PE 为负。}}
  \label{{fig:industry-raw-pe}}
\end{{figure}}

行业中位 PE 最高的三个行业分别为{top.industry}（{top.median_pe:.2f} 倍）、
{second.industry}（{second.median_pe:.2f} 倍）和{third.industry}（{third.median_pe:.2f} 倍）。
{bottom.industry}的行业中位 PE 为 {bottom.median_pe:.2f} 倍，其负 PE 比例为
{bottom.negative_ratio * 100:.2f}\%；负 PE 表示行业内亏损观测较多，不应解释为负估值倍数。
本图仅为描述性统计，不包含回归检验。
"""
    TEX_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    latest, stats, diagnostics = load_latest_industry_pe()
    stats.insert(0, "date", latest.date().isoformat())
    stats.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    save_png(stats, latest, float(diagnostics["market_median_pe"]))
    save_latex_fragment(stats, latest, diagnostics)

    print(f"latest date: {latest.date()}")
    for key, value in diagnostics.items():
        print(f"{key}: {value}")
    print(f"saved: {CSV_PATH}")
    print(f"saved: {PNG_PATH}")
    print(f"saved: {TEX_PATH}")


if __name__ == "__main__":
    main()
