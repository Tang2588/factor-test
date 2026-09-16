# -*- coding: utf-8 -*-
"""Monthly cross-sectional OLS for EP and standardized Size."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import t as student_t


BASE = Path(r"D:\实习生学习项目\基础数据")
FACTOR_ROOT = Path(r"D:\因子计算\市盈率\EP因子_交付版")
EP_PATH = FACTOR_ROOT / "因子结果" / "ep.parquet"
SIZE_PATH = Path(r"D:\因子计算\规模因子_交付版\因子结果\size.parquet")
MARKET_PATH = BASE / "chn_equ_mkt_quotation.parquet"
INDUSTRY_PATH = BASE / "chn_equ_indus_sw.parquet"
TEST_START_PATH = FACTOR_ROOT / "中间结果" / "factor_test_start.json"
OUT_DIR = FACTOR_ROOT / "回归结果"
LOG_PATH = OUT_DIR / "ep_monthly_ols_log.md"


def read_batches_for_dates(path: Path, columns: list[str], dates: set[pd.Timestamp]) -> pd.DataFrame:
    """Read only selected dates from a parquet panel."""
    pieces = []
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=100_000):
        frame = batch.to_pandas(ignore_metadata=True)
        frame["date"] = pd.to_datetime(frame["date"])
        piece = frame.loc[frame["date"].isin(dates)].copy()
        if not piece.empty:
            pieces.append(piece)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)


def read_all_dates(path: Path) -> pd.DatetimeIndex:
    dates = set()
    for batch in pq.ParquetFile(path).iter_batches(columns=["date"], batch_size=200_000):
        dates.update(pd.to_datetime(batch.to_pandas(ignore_metadata=True)["date"]).dt.normalize())
    return pd.DatetimeIndex(sorted(dates))


def normalize_code(series: pd.Series) -> pd.Series:
    return series.astype("string").str.extract(r"(\d{6})", expand=False)


def cross_sectional_ols(
    frame: pd.DataFrame,
    y_col: str,
    factor_cols: list[str],
    industry_col: str,
    min_observations: int = 100,
) -> dict:
    """Solve one cross-section using normal equations with all industry dummies."""
    data = frame.copy()
    for col in [y_col, *factor_cols]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    valid = np.isfinite(data[y_col])
    for col in factor_cols:
        valid &= np.isfinite(data[col])
    valid &= data[industry_col].notna()
    data = data.loc[valid].copy()
    if len(data) < min_observations:
        raise ValueError(
            f"valid observations {len(data)} below minimum {min_observations}"
        )
    industries = sorted(data[industry_col].astype(str).unique())
    if len(industries) == 0:
        raise ValueError("no industry observations in cross-section")
    dummies = pd.get_dummies(
        data[industry_col].astype(pd.CategoricalDtype(categories=industries)),
        dtype=float,
    )
    dummies.columns = [f"industry_{name}" for name in industries]
    columns = factor_cols + list(dummies.columns)
    design = np.column_stack(
        [data[col].to_numpy(float) for col in factor_cols] + [dummies.to_numpy(float)]
    )
    y = data[y_col].to_numpy(float)
    rank = int(np.linalg.matrix_rank(design))
    if rank < design.shape[1]:
        raise ValueError(f"singular design matrix: rank={rank}, columns={design.shape[1]}")
    normal = design.T @ design
    beta = np.linalg.solve(normal, design.T @ y)
    residual = y - design @ beta
    n, p = design.shape
    dof = n - p
    sse = float(residual @ residual)
    sst = float((y - y.mean()) @ (y - y.mean()))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    covariance = (sse / dof) * np.linalg.inv(normal)
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_values = beta / se
    p_values = 2.0 * student_t.sf(np.abs(t_values), dof)
    return {
        "n": n,
        "p": p,
        "k": len(industries),
        "dof": dof,
        "beta": dict(zip(columns, beta)),
        "se": dict(zip(columns, se)),
        "t": dict(zip(columns, t_values)),
        "pvalue": dict(zip(columns, p_values)),
        "sse": sse,
        "rmse": float(np.sqrt(sse / dof)),
        "r2": r2,
        "condition": float(np.linalg.cond(normal)),
        "residual_mean": float(residual.mean()),
        "valid_codes": data["code6"].tolist(),
    }


def build_forward_month_returns(
    path: Path, rebalance_dates: pd.DatetimeIndex, trading_dates: pd.DatetimeIndex
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Compound daily returns on (rebalance day, next rebalance day]."""
    dates_np = rebalance_dates.to_numpy(dtype="datetime64[ns]")
    expected_days = {}
    for i in range(len(rebalance_dates) - 1):
        expected_days[i] = int(
            ((trading_dates > rebalance_dates[i]) & (trading_dates <= rebalance_dates[i + 1])).sum()
        )
    partials = []
    for batch in pq.ParquetFile(path).iter_batches(
        columns=["date", "stock_code", "ret"], batch_size=100_000
    ):
        frame = batch.to_pandas(ignore_metadata=True)
        frame["date"] = pd.to_datetime(frame["date"])
        dates = frame["date"].to_numpy(dtype="datetime64[ns]")
        interval = np.searchsorted(dates_np, dates, side="left") - 1
        keep = (interval >= 0) & (interval < len(rebalance_dates) - 1)
        if not keep.any():
            continue
        frame = frame.loc[keep].copy()
        frame["interval"] = interval[keep]
        frame["code6"] = normalize_code(frame["stock_code"])
        frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
        frame["bad"] = (~np.isfinite(frame["ret"])).astype("int8")
        frame["gross"] = (1.0 + frame["ret"].where(np.isfinite(frame["ret"]), 0.0)).astype(float)
        grouped = frame.groupby(["interval", "code6"], sort=False).agg(
            gross=("gross", "prod"),
            n_rows=("gross", "size"),
            n_bad=("bad", "sum"),
        ).reset_index()
        partials.append(grouped)
    if not partials:
        return pd.DataFrame(columns=["interval", "code6", "monthly_return"]), expected_days
    summary = pd.concat(partials, ignore_index=True).groupby(
        ["interval", "code6"], sort=False
    ).agg(
        gross=("gross", "prod"),
        n_rows=("n_rows", "sum"),
        n_bad=("n_bad", "sum"),
    ).reset_index()
    summary["expected_days"] = summary["interval"].map(expected_days)
    complete = (summary["n_rows"] == summary["expected_days"]) & summary["n_bad"].eq(0)
    summary["monthly_return"] = np.where(complete, summary["gross"] - 1.0, np.nan)
    return summary[["interval", "code6", "monthly_return"]], expected_days


def fmt(value: float, digits: int = 8) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.{digits}g}"


def p_fmt(value: float) -> str:
    if value < 0.001:
        return f"{value:.3e}"
    return f"{value:.6f}"


def main() -> None:
    with TEST_START_PATH.open(encoding="utf-8") as file:
        test_config = json.load(file)
    test_start = pd.Timestamp(test_config["formal_test_start_date"])
    market_dates = read_all_dates(MARKET_PATH)
    first_by_month = pd.Series(market_dates, index=market_dates).groupby(
        [market_dates.year, market_dates.month]
    ).first()
    rebalance_dates = pd.DatetimeIndex(first_by_month.to_numpy())
    rebalance_dates = rebalance_dates.sort_values()
    rebalance_dates = rebalance_dates[rebalance_dates > test_start]
    if len(rebalance_dates) < 2:
        raise ValueError("覆盖率测试起点之后没有足够的月度调仓日")
    intervals = len(rebalance_dates) - 1
    selected_dates = set(rebalance_dates)

    ep = read_batches_for_dates(EP_PATH, ["date", "stock_code", "signal"], selected_dates)
    ep = ep.rename(columns={"signal": "EP"})
    size = read_batches_for_dates(SIZE_PATH, ["date", "stock_code", "signal"], selected_dates)
    size = size.rename(columns={"signal": "Size"})
    industry = read_batches_for_dates(
        INDUSTRY_PATH, ["date", "stock_code", "indus_name_lv1"], selected_dates
    )
    for frame in (ep, size, industry):
        frame["code6"] = normalize_code(frame["stock_code"])
    if industry.duplicated(["date", "code6"]).any():
        raise ValueError("duplicate industry classification in date + code6")
    monthly_returns, expected_days = build_forward_month_returns(MARKET_PATH, rebalance_dates, market_dates)

    monthly_rows = []
    skipped_rows = []
    for i in range(intervals):
        start = rebalance_dates[i]
        end = rebalance_dates[i + 1]
        frame = (
            ep.loc[ep["date"].eq(start), ["code6", "EP"]]
            .merge(size.loc[size["date"].eq(start), ["code6", "Size"]], on="code6", validate="one_to_one")
            .merge(industry.loc[industry["date"].eq(start), ["code6", "indus_name_lv1"]], on="code6", validate="one_to_one")
            .merge(monthly_returns.loc[monthly_returns["interval"].eq(i), ["code6", "monthly_return"]], on="code6", validate="one_to_one")
        )
        try:
            result = cross_sectional_ols(
                frame.rename(columns={"monthly_return": "y"}),
                "y",
                ["EP", "Size"],
                "indus_name_lv1",
                min_observations=100,
            )
        except ValueError as exc:
            skipped_rows.append(
                {"date": start, "next_rebalance": end, "reason": str(exc)}
            )
            continue
        monthly_rows.append(
            {
                "date": start,
                "next_rebalance": end,
                "n": result["n"],
                "k": result["k"],
                "dof": result["dof"],
                "ep_beta": result["beta"]["EP"],
                "ep_se": result["se"]["EP"],
                "ep_t": result["t"]["EP"],
                "ep_p": result["pvalue"]["EP"],
                "size_beta": result["beta"]["Size"],
                "size_t": result["t"]["Size"],
                "size_p": result["pvalue"]["Size"],
                "r2": result["r2"],
                "rmse": result["rmse"],
                "condition": result["condition"],
                "return_mean": float(frame["monthly_return"].mean()),
                "return_std": float(frame["monthly_return"].std(ddof=1)),
            }
        )
    monthly = pd.DataFrame(monthly_rows)
    monthly["date"] = pd.to_datetime(monthly["date"])

    def annual_summary(group: pd.DataFrame) -> pd.Series:
        def ts_t(values: pd.Series) -> float:
            values = values.dropna()
            return float(values.mean() / (values.std(ddof=1) / np.sqrt(len(values)))) if len(values) > 1 and values.std(ddof=1) > 0 else np.nan
        return pd.Series(
            {
                "months": len(group),
                "ep_mean": group["ep_beta"].mean(),
                "ep_std": group["ep_beta"].std(ddof=1),
                "ep_ts_t": ts_t(group["ep_beta"]),
                "ep_positive_ratio": (group["ep_beta"] > 0).mean(),
                "size_mean": group["size_beta"].mean(),
                "size_std": group["size_beta"].std(ddof=1),
                "size_ts_t": ts_t(group["size_beta"]),
                "r2_mean": group["r2"].mean(),
                "n_mean": group["n"].mean(),
            }
        )

    annual = monthly.groupby(monthly["date"].dt.year).apply(annual_summary, include_groups=False).reset_index(names="year")
    overall = annual_summary(monthly)
    skipped = pd.DataFrame(skipped_rows)
    timeseries = monthly[[
        "date", "next_rebalance", "n", "ep_beta", "ep_t", "ep_p",
        "size_beta", "size_t", "size_p", "r2"
    ]].rename(columns={
        "date": "rebalance_date",
        "ep_beta": "ep_premium",
        "ep_t": "ep_cross_sectional_t",
        "ep_p": "ep_cross_sectional_p",
        "size_beta": "size_premium",
        "size_t": "size_cross_sectional_t",
        "size_p": "size_cross_sectional_p",
    })
    timeseries["rebalance_date"] = pd.to_datetime(timeseries["rebalance_date"])
    timeseries["next_rebalance"] = pd.to_datetime(timeseries["next_rebalance"])
    sig_ep = monthly["ep_p"] < 0.05
    sig_size = monthly["size_p"] < 0.05
    key_positive = monthly.nlargest(3, "ep_beta")
    key_negative = monthly.nsmallest(3, "ep_beta")

    lines = [
        "# EP 月度调仓截面 OLS 回归日志",
        "",
        "## 1. 回归设定",
        "",
        "每月第一个交易日调仓，因子暴露取调仓日，收益取调仓日之后至下月调仓日（含下月调仓日）的每日收益复合。未筛选 ST/PT、次新股或涨跌停；EP 因子样本排除 50 只 900xxx.BJ 上海 B 股和 39 只 200/201xxx.SZ 深圳 B 股。",
        f"覆盖率稳定规则给出的正式测试起点为 {test_start.date()}；首次调仓使用其后的第一个月度交易日。",
        "",
        "模型：",
        "",
        "$$",
        "r_{i,t\\rightarrow t+1} = \\lambda_{EP,t}EP^Z_{i,t} + \\lambda_{Size,t}Size^Z_{i,t} + \\sum_{k=1}^{31}\\gamma_{k,t}Industry_{i,k,t} + \\varepsilon_{i,t+1}",
        "$$",
        "",
        "EP 和 Size 使用横截面 Z 标准化值；Size 的底层变量为对数总市值。加入全部 31 个行业哑变量，不单独加入截距，使用正规方程求解。",
        "",
        "## 2. 样本概况",
        "",
        f"数据覆盖：{rebalance_dates[0].date()} 至 {rebalance_dates[-1].date()}；理论持有区间 {intervals} 个，完成回归 {len(monthly)} 个，跳过 {len(skipped_rows)} 个。",
        "",
        "## 3. 统计结果",
        "",
        "| 样本 | 月数 | EP平均溢价 | EP标准差 | EP时间序列t值 | EP正溢价比例 | EP显著月份 | Size平均溢价 | Size时间序列t值 | 平均R² | 平均样本数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in annual.itertuples(index=False):
        year_data = monthly.loc[monthly["date"].dt.year.eq(int(row.year))]
        lines.append(f"| {int(row.year)} | {int(row.months)} | {row.ep_mean:.8f} | {row.ep_std:.8f} | {row.ep_ts_t:.4f} | {row.ep_positive_ratio:.2%} | {(year_data['ep_p'] < 0.05).sum()} | {row.size_mean:.8f} | {row.size_ts_t:.4f} | {row.r2_mean:.6f} | {row.n_mean:.1f} |")
    lines += [
        f"| 全期 | {len(monthly)} | {overall['ep_mean']:.8f} | {overall['ep_std']:.8f} | {overall['ep_ts_t']:.4f} | {overall['ep_positive_ratio']:.2%} | {int(sig_ep.sum())} | {overall['size_mean']:.8f} | {overall['size_ts_t']:.4f} | {overall['r2_mean']:.6f} | {overall['n_mean']:.1f} |",
        "",
        "显著性统计使用每个持有区间截面回归的双侧 p<0.05：",
        f"- EP：{int(sig_ep.sum())}/{len(monthly)} 个区间显著，其中正向 {int((sig_ep & monthly['ep_beta'].gt(0)).sum())} 个、负向 {int((sig_ep & monthly['ep_beta'].lt(0)).sum())} 个。",
        f"- Size：{int(sig_size.sum())}/{len(monthly)} 个区间显著，其中正向 {int((sig_size & monthly['size_beta'].gt(0)).sum())} 个、负向 {int((sig_size & monthly['size_beta'].lt(0)).sum())} 个。",
        "",
        "## 4. 典型区间",
        "",
        "以下列出 EP 溢价最大的 3 个和最小的 3 个区间；所有月份的时间序列值见 ep_monthly_ols_timeseries.csv。",
        "",
        "| 类型 | 调仓日 | 下次调仓日 | EP溢价 | EP t值 | EP p值 | Size溢价 | R² |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for label, selected in [("EP正向典型", key_positive), ("EP负向典型", key_negative)]:
        for row in selected.itertuples(index=False):
            lines.append(f"| {label} | {row.date.date()} | {row.next_rebalance.date()} | {row.ep_beta:.8f} | {row.ep_t:.4f} | {p_fmt(row.ep_p)} | {row.size_beta:.8f} | {row.r2:.6f} |")
    lines += [
        "",
        "## 5. 结果分析",
        "",
        f"全期 EP 平均溢价为 {overall['ep_mean']:.8f}，时间序列 t 值为 {overall['ep_ts_t']:.4f}，正溢价比例为 {overall['ep_positive_ratio']:.2%}。平均值接近零且未达到常用显著性标准，说明控制 Size 和行业后，EP 的月度条件收益没有稳定方向。",
        "EP 的年度方向发生变化：2021--2022 年平均为负，2023--2024 年平均为正，2025 年再次转负；年度 t 值均未达到 5% 显著性，说明结果具有明显时变性。",
        f"Size 全期平均溢价为 {overall['size_mean']:.8f}，时间序列 t 值为 {overall['size_ts_t']:.4f}，在当前模型中比 EP 呈现更稳定的统计关系，但这不等于 EP 无法在其他模型或样本区间中有效。",
        "时间序列表中的因子溢价是逐月截面回归系数；跨月均值和时间序列 t 值用于判断长期稳定性，单个月份显著不能直接推出长期选股结论。",
        "",
        "## 6. 输出文件",
        "",
        "- ep_monthly_ols_timeseries.csv：逐月 EP/Size 溢价及截面 t 值、p 值。",
        "- ep_monthly_ols_annual.csv：年度统计结果。",
        "- ep_monthly_ols_monthly.csv：完整逐月回归诊断结果。",
        "- 本日志仅保留关键统计结果与分析。",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monthly.to_csv(OUT_DIR / "ep_monthly_ols_monthly.csv", index=False, encoding="utf-8-sig")
    timeseries.to_csv(OUT_DIR / "ep_monthly_ols_timeseries.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUT_DIR / "ep_monthly_ols_annual.csv", index=False, encoding="utf-8-sig")
    print("saved", LOG_PATH)
    print("monthly_rows", len(monthly))
    print("annual_rows", len(annual))
    print("overall_ep_mean", overall["ep_mean"])
    print("overall_ep_ts_t", overall["ep_ts_t"])


if __name__ == "__main__":
    main()
