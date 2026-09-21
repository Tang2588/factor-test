# -*- coding: utf-8 -*-
"""EP 因子月度横截面 Huber RLM 回归。

本脚本只负责估计方法和结果呈现，数据准备全部调用 ``common`` 共享模块，
与 ``monthly_ols_ep_size.py`` 使用同一套样本、同一套前瞻收益、同一套控制变量。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    HUBER_T,
    INDUSTRY_PATH,
    MARKET_PATH,
    MAX_ITER,
    MIN_OBSERVATIONS,
    REG_DIR,
    TEST_START_PATH,
    TOL,
    build_endpoint_forward_returns,
    build_monthly_calendar,
    build_regression_panel,
    derive_bse_mapping,
    prepare_formation_panel,
    prepare_industry,
    prepare_market_snapshot,
    read_selected_dates,
    scan_market_index,
)
from common.paths import EP_PATH  # noqa: E402

OUT_DIR = REG_DIR / "RLM_H1_独立同方差"


def _coefficient_delta(history: dict) -> float:
    """最后一次迭代与上一次迭代的参数最大变化量，用于判断收敛裕度。"""
    params = history.get("params") or []
    if len(params) < 3:
        return np.nan
    last = np.asarray(params[-1], dtype=float)
    previous = np.asarray(params[-2], dtype=float)
    if last.shape != previous.shape:
        return np.nan
    return float(np.max(np.abs(last - previous)))


def fit_monthly_rlm(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_rows = []
    coefficient_rows = []
    sample_rows = []

    for formation_date, raw_group in panel.groupby("formation_date", sort=True):
        data = raw_group.loc[raw_group["regression_eligible"]].copy().reset_index(drop=True)
        industries = sorted(data["industry_lv1"].astype(str).unique())
        dummies = pd.get_dummies(
            data["industry_lv1"].astype(str),
            prefix="industry",
            drop_first=True,
            dtype=float,
        )
        design = pd.concat(
            [
                pd.Series(1.0, index=data.index, name="const"),
                data["ep_z"].astype(float).rename("ep_z"),
                data["size_z"].astype(float).rename("size_z"),
                dummies,
            ],
            axis=1,
        )
        y = data["forward_return"].astype(float)
        n, p = design.shape
        required_n = max(MIN_OBSERVATIONS, 10 * p)
        if n < required_n:
            raise ValueError(
                f"{formation_date.date()}: 观测数 {n} 低于要求的 {required_n}"
            )
        rank = int(np.linalg.matrix_rank(design.to_numpy(float)))
        if rank != p:
            raise ValueError(
                f"{formation_date.date()}: 设计矩阵不满秩 rank={rank}, columns={p}"
            )

        model = sm.RLM(y, design, M=sm.robust.norms.HuberT(t=HUBER_T), missing="raise")
        result = model.fit(
            maxiter=MAX_ITER,
            tol=TOL,
            scale_est="mad",
            cov="H1",
            update_scale=True,
            conv="coefs",
        )
        history = dict(result.fit_history)
        iterations = int(history.get("iteration", MAX_ITER))
        converged = iterations < MAX_ITER
        coef_delta = _coefficient_delta(history)
        residual = np.asarray(result.resid, dtype=float)
        weights = np.asarray(result.weights, dtype=float)
        y_values = y.to_numpy(float)
        sst = float(np.square(y_values - y_values.mean()).sum())
        sse = float(np.square(residual).sum())
        pseudo_r2 = 1.0 - sse / sst if sst > 0 else np.nan

        names = list(design.columns)
        params = pd.Series(np.asarray(result.params), index=names)
        bse = pd.Series(np.asarray(result.bse), index=names)
        zvalues = pd.Series(np.asarray(result.tvalues), index=names)
        pvalues = pd.Series(np.asarray(result.pvalues), index=names)

        monthly_rows.append(
            {
                "formation_date": formation_date,
                "entry_date": data["entry_date"].iloc[0],
                "exit_date": data["exit_date"].iloc[0],
                "n": n,
                "p": p,
                "industries": len(industries),
                "reference_industry": industries[0],
                "ep_beta": float(params["ep_z"]),
                "ep_bse": float(bse["ep_z"]),
                "ep_zvalue": float(zvalues["ep_z"]),
                "ep_pvalue": float(pvalues["ep_z"]),
                "size_beta": float(params["size_z"]),
                "size_bse": float(bse["size_z"]),
                "size_zvalue": float(zvalues["size_z"]),
                "size_pvalue": float(pvalues["size_z"]),
                "intercept": float(params["const"]),
                "scale": float(result.scale),
                "iterations": iterations,
                "converged": converged,
                "final_coef_delta": coef_delta,
                "design_rank": rank,
                "condition_number": float(np.linalg.cond(design.to_numpy(float))),
                "pseudo_r2": pseudo_r2,
                "return_mean": float(y.mean()),
                "return_std": float(y.std(ddof=1)),
                "return_median": float(y.median()),
                "return_p01": float(y.quantile(0.01)),
                "return_p99": float(y.quantile(0.99)),
                "weight_min": float(weights.min()),
                "weight_median": float(np.median(weights)),
                "downweighted_ratio": float((weights < 1.0 - 1e-12).mean()),
                "weight_below_half_ratio": float((weights < 0.5).mean()),
            }
        )

        for name in names:
            coefficient_rows.append(
                {
                    "formation_date": formation_date,
                    "parameter": name,
                    "coefficient": float(params[name]),
                    "bse": float(bse[name]),
                    "zvalue": float(zvalues[name]),
                    "pvalue": float(pvalues[name]),
                }
            )

        regression_sample = data[
            [
                "formation_date",
                "entry_date",
                "exit_date",
                "security_id",
                "code_at_formation",
                "code_at_entry",
                "code_at_exit",
                "ep_z",
                "size_z",
                "industry_lv1",
                "forward_return",
            ]
        ].copy()
        regression_sample["rlm_weight"] = weights
        regression_sample["rlm_residual"] = residual
        sample_rows.append(regression_sample)
        print(
            f"RLM {formation_date.date()} n={n:,} "
            f"beta={params['ep_z']:.8f} z={zvalues['ep_z']:.3f} "
            f"iterations={iterations} converged={converged}",
            flush=True,
        )

    return (
        pd.DataFrame(monthly_rows),
        pd.DataFrame(coefficient_rows),
        pd.concat(sample_rows, ignore_index=True),
    )


def iid_mean(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(clean) < 2:
        return {"mean": np.nan, "iid_se": np.nan, "iid_t": np.nan, "iid_p": np.nan}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    se = std / np.sqrt(len(clean))
    t_value = mean / se if se > 0 else np.nan
    p_value = (
        float(2.0 * student_t.sf(abs(t_value), df=len(clean) - 1))
        if np.isfinite(t_value)
        else np.nan
    )
    return {
        "mean": mean,
        "iid_se": float(se),
        "iid_t": float(t_value),
        "iid_p": p_value,
    }


def summarize_period(group: pd.DataFrame, label: str) -> dict[str, float | str | int]:
    ep_iid = iid_mean(group["ep_beta"])
    size_iid = iid_mean(group["size_beta"])
    return {
        "period": label,
        "months": len(group),
        "ep_mean": ep_iid["mean"],
        "ep_median": float(group["ep_beta"].median()),
        "ep_std": float(group["ep_beta"].std(ddof=1)),
        "ep_iid_se": ep_iid["iid_se"],
        "ep_iid_t": ep_iid["iid_t"],
        "ep_iid_p": ep_iid["iid_p"],
        "ep_positive_ratio": float(group["ep_beta"].gt(0).mean()),
        "ep_abs_monthly_z_ge_2_ratio": float(group["ep_zvalue"].abs().ge(2).mean()),
        "ep_positive_significant_ratio": float(
            (group["ep_beta"].gt(0) & group["ep_zvalue"].ge(2)).mean()
        ),
        "ep_negative_significant_ratio": float(
            (group["ep_beta"].lt(0) & group["ep_zvalue"].le(-2)).mean()
        ),
        "size_mean": size_iid["mean"],
        "size_iid_t": size_iid["iid_t"],
        "size_iid_p": size_iid["iid_p"],
        "n_mean": float(group["n"].mean()),
        "n_min": int(group["n"].min()),
        "n_max": int(group["n"].max()),
        "converged_ratio": float(group["converged"].mean()),
        "iterations_median": float(group["iterations"].median()),
        "downweighted_ratio_mean": float(group["downweighted_ratio"].mean()),
        "weight_below_half_ratio_mean": float(group["weight_below_half_ratio"].mean()),
        "pseudo_r2_mean": float(group["pseudo_r2"].mean()),
    }


def build_summaries(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_rows = []
    for year, group in monthly.groupby(monthly["formation_date"].dt.year, sort=True):
        annual_rows.append(summarize_period(group, str(year)))
    annual = pd.DataFrame(annual_rows)
    overall = pd.DataFrame([summarize_period(monthly, "全期")])
    return annual, overall


def save_plot(monthly: pd.DataFrame, path: Path) -> None:
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].axhline(0.0, color="#555555", linewidth=0.9)
    axes[0].plot(monthly["formation_date"], monthly["ep_beta"] * 100, color="#176B87", linewidth=1.2)
    axes[0].scatter(
        monthly["formation_date"], monthly["ep_beta"] * 100, color="#176B87", s=13
    )
    axes[0].set_ylabel("EP premium (%)")
    axes[0].set_title("Monthly Huber RLM EP premium")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].axhline(0.0, color="#555555", linewidth=0.9)
    axes[1].plot(
        monthly["formation_date"],
        monthly["cumulative_ep_premium"] * 100,
        color="#B23A48",
        linewidth=1.5,
    )
    axes[1].set_ylabel("Cumulative premium (pct. points)")
    axes[1].set_xlabel("Formation date")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 4) -> str:
    return "NaN" if not np.isfinite(value) else f"{value:.{digits}f}"


def save_report(
    monthly: pd.DataFrame,
    annual: pd.DataFrame,
    overall: pd.DataFrame,
    sample_audit: pd.DataFrame,
    mapping: pd.DataFrame,
    calendar: pd.DataFrame,
    path: Path,
) -> None:
    summary = overall.iloc[0]
    direction = "正" if summary["ep_mean"] > 0 else "负"
    significance = "达到" if summary["ep_iid_p"] < 0.05 else "未达到"
    strongest = monthly.nlargest(3, "ep_beta")
    weakest = monthly.nsmallest(3, "ep_beta")
    return_valid_total = int(sample_audit["n_valid_return_after_controls"].sum())
    control_valid_total = int(sample_audit["n_valid_ep_size_industry"].sum())

    lines = [
        "# EP 月度横截面 Huber RLM 统计结果",
        "",
        f"**运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**",
        "",
        "> 本报告只包含 RLM 横截面回归，不包含 IC、分层回测、OLS 或 WLS 对照。",
        "> 样本、前瞻收益与控制变量由 `代码/common` 共享模块统一构造，",
        "> 与 `monthly_ols_ep_size.py` 完全一致，两者差异只来自估计方法。",
        "",
        "## 1. 回归口径",
        "",
        "每月最后一个交易日形成标准化 EP，次月第一个交易日按复权收盘价建仓，",
        "再下一个月第一个交易日按复权收盘价退出。模型为：",
        "",
        "```text",
        "r_fwd[i,m+1] = alpha[m] + beta_EP[m] * EP_Z[i,m]",
        "                 + beta_Size[m] * Size_Z[i,m]",
        "                 + K-1 个申万一级行业哑变量 + epsilon[i,m+1]",
        "```",
        "",
        f"RLM 使用 HuberT(c={HUBER_T})、MAD 残差尺度和 H1 协方差。单月统计量为系数除以 H1 标准误；年度和全期均值按照月度系数独立同方差假设，使用普通标准误和 Student t 分布推断，不使用 Newey-West HAC。",
        "",
        "## 2. 样本与映射",
        "",
        f"- 完整持有期：{len(calendar)} 个，从 {calendar['formation_date'].min().date()} 至 {calendar['formation_date'].max().date()}。",
        f"- 首次建仓：{calendar['entry_date'].min().date()}；最后退出：{calendar['exit_date'].max().date()}。",
        f"- 北交所新旧代码映射：{len(mapping)} 对，均通过休市前后价格连续性校验，其中 {(mapping['match_method'] == 'exact_price_and_share_total').sum()} 对使用总股本消除同价歧义。",
        f"- 控制变量和行业均有效的观测：{control_valid_total:,}；具有有效固定端点收益的观测：{return_valid_total:,}。",
        f"- 单月最终样本数：最少 {int(sample_audit['n_final'].min()):,}，中位数 {int(sample_audit['n_final'].median()):,}，最多 {int(sample_audit['n_final'].max()):,}。",
        "",
        "## 3. 全期结果",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 回归月份 | {int(summary['months'])} |",
        f"| EP 月均系数 | {pct(summary['ep_mean'], 4)} |",
        f"| EP 系数中位数 | {pct(summary['ep_median'], 4)} |",
        f"| EP 系数标准差 | {pct(summary['ep_std'], 4)} |",
        f"| EP IID 标准误 | {pct(summary['ep_iid_se'], 4)} |",
        f"| EP IID t 值 | {num(summary['ep_iid_t'])} |",
        f"| EP IID p 值 | {num(summary['ep_iid_p'])} |",
        f"| EP 系数为正的月份 | {pct(summary['ep_positive_ratio'])} |",
        f"| 单月 abs(z) >= 2 的比例 | {pct(summary['ep_abs_monthly_z_ge_2_ratio'])} |",
        f"| 正向且 z >= 2 的比例 | {pct(summary['ep_positive_significant_ratio'])} |",
        f"| 负向且 z <= -2 的比例 | {pct(summary['ep_negative_significant_ratio'])} |",
        f"| Size 月均系数 | {pct(summary['size_mean'], 4)} |",
        f"| Size IID t 值 | {num(summary['size_iid_t'])} |",
        f"| 平均 RLM 降权比例 | {pct(summary['downweighted_ratio_mean'])} |",
        f"| 平均权重低于 0.5 比例 | {pct(summary['weight_below_half_ratio_mean'])} |",
        f"| 平均伪 R² | {pct(summary['pseudo_r2_mean'])} |",
        "",
        f"EP 月均系数方向为{direction}，独立同方差假设下的显著性{significance} 5% 标准。EP 已按形成日横截面标准化，",
        f"因此月均系数 {pct(summary['ep_mean'], 4)} 可解释为 EP 提高一个横截面标准差时，下一持有期收益平均变化约 {pct(summary['ep_mean'], 4)}。",
        f"简单乘以 12 得到的年化系数约为 {pct(summary['ep_mean'] * 12, 2)}，它是因子溢价尺度，不是可交易组合年化收益。",
        "",
        "## 4. 分年度结果",
        "",
        "| 年份 | 月数 | EP 月均系数 | IID t 值 | IID p 值 | 正系数比例 | abs(z) >= 2 比例 | 平均样本数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in annual.itertuples(index=False):
        lines.append(
            f"| {row.period} | {row.months} | {pct(row.ep_mean, 4)} | "
            f"{num(row.ep_iid_t)} | {num(row.ep_iid_p)} | {pct(row.ep_positive_ratio)} | "
            f"{pct(row.ep_abs_monthly_z_ge_2_ratio)} | {row.n_mean:,.0f} |"
        )

    lines += [
        "",
        "## 5. 极值月份",
        "",
        "| 类型 | 因子形成日 | 建仓日 | 退出日 | EP 系数 | 单月 z 值 | 样本数 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for label, selected in [("最高", strongest), ("最低", weakest)]:
        for row in selected.itertuples(index=False):
            lines.append(
                f"| {label} | {row.formation_date.date()} | {row.entry_date.date()} | "
                f"{row.exit_date.date()} | {pct(row.ep_beta, 4)} | {row.ep_zvalue:.3f} | {row.n:,} |"
            )

    lines += [
        "",
        "## 6. 拟合诊断",
        "",
        f"- 收敛月份：{int(monthly['converged'].sum())}/{len(monthly)}；迭代次数中位数为 {monthly['iterations'].median():.0f}，最大 {int(monthly['iterations'].max())}。",
        f"- 最后一次迭代参数最大变化量中位数：{monthly['final_coef_delta'].median():.3e}（收敛容差 {TOL:.0e}）。",
        f"- 条件数范围：{monthly['condition_number'].min():.2f} 至 {monthly['condition_number'].max():.2f}。",
        f"- 北交所最终样本在代码切换附近未发生由代码连接失败造成的断层，逐月数量见 `rlm_sample_audit.csv`。",
        "- 涨停买入和跌停卖出只做数量标记，未用于 RLM 样本筛选；实际可交易性留待分层回测处理。",
        "- `cumulative_ep_premium` 是月度系数的累计和，不是组合净值。",
        "",
        "![RLM 月度 EP 系数与累计因子溢价](./rlm_ep_premium.png)",
        "",
        "## 7. 输出文件",
        "",
        "- `bse_code_mapping.csv/.parquet`：北交所新旧代码映射及匹配证据。",
        "- `forward_monthly_returns.parquet`：固定调仓端点的复权前瞻收益。",
        "- `rlm_regression_sample.parquet`：进入回归的逐股样本、残差和稳健权重。",
        "- `rlm_monthly_results.csv/.parquet`：逐月 EP/Size 系数及拟合诊断。",
        "- `rlm_coefficients_long.parquet`：全部月份、全部回归参数。",
        "- `rlm_sample_audit.csv`：逐月样本流失及交易状态审计。",
        "- `rlm_annual_summary.csv`、`rlm_overall_summary.csv`：年度和全期汇总。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with TEST_START_PATH.open(encoding="utf-8") as file:
        test_config = json.load(file)
    test_start = pd.Timestamp(test_config["formal_test_start_date"])

    print("Scanning market index (calendar + BSE code switch)...", flush=True)
    index = scan_market_index(MARKET_PATH)
    market_dates = index.dates
    transition_old_date = index.old_bse_last_date
    transition_new_date = index.new_bse_first_date
    if transition_old_date is None or transition_new_date is None:
        raise ValueError("无法从行情数据中识别北交所代码切换日")
    print(
        f"  trading days={len(market_dates):,}; BSE switch "
        f"{transition_old_date.date()} -> {transition_new_date.date()} "
        f"({index.expected_conversions} conversions)",
        flush=True,
    )

    calendar = build_monthly_calendar(market_dates, test_start)
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
    mapping_frame.to_parquet(OUT_DIR / "bse_code_mapping.parquet", index=False)
    mapping_frame.to_csv(OUT_DIR / "bse_code_mapping.csv", index=False, encoding="utf-8-sig")

    print("Reading monthly EP and industry cross-sections...", flush=True)
    ep = read_selected_dates(EP_PATH, ["date", "stock_code", "signal"], signal_dates)
    industry_raw = read_selected_dates(
        INDUSTRY_PATH,
        ["date", "stock_code", "indus_name_lv1"],
        signal_dates,
    )
    formation = prepare_formation_panel(market, ep, signal_dates, mapping)
    industry = prepare_industry(industry_raw, mapping)

    print("Building fixed-endpoint adjusted forward returns...", flush=True)
    forward_returns = build_endpoint_forward_returns(market, calendar, mapping)
    forward_returns.to_parquet(OUT_DIR / "forward_monthly_returns.parquet", index=False)

    print("Building monthly regression samples...", flush=True)
    panel, sample_audit = build_regression_panel(formation, industry, forward_returns)
    sample_audit.to_csv(OUT_DIR / "rlm_sample_audit.csv", index=False, encoding="utf-8-sig")
    sample_audit.to_parquet(OUT_DIR / "rlm_sample_audit.parquet", index=False)

    print("Running monthly Huber RLM...", flush=True)
    monthly, coefficients, regression_sample = fit_monthly_rlm(panel)
    monthly["formation_date"] = pd.to_datetime(monthly["formation_date"])
    monthly = monthly.sort_values("formation_date").reset_index(drop=True)
    monthly["cumulative_ep_premium"] = monthly["ep_beta"].cumsum()
    annual, overall = build_summaries(monthly)

    if not monthly["converged"].all():
        failed = monthly.loc[~monthly["converged"], "formation_date"].dt.strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"以下月份 RLM 未收敛：{failed}")

    monthly.to_parquet(OUT_DIR / "rlm_monthly_results.parquet", index=False)
    monthly.to_csv(OUT_DIR / "rlm_monthly_results.csv", index=False, encoding="utf-8-sig")
    coefficients.to_parquet(OUT_DIR / "rlm_coefficients_long.parquet", index=False)
    regression_sample.to_parquet(OUT_DIR / "rlm_regression_sample.parquet", index=False)
    annual.to_csv(OUT_DIR / "rlm_annual_summary.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUT_DIR / "rlm_overall_summary.csv", index=False, encoding="utf-8-sig")
    calendar.to_csv(OUT_DIR / "rlm_calendar.csv", index=False, encoding="utf-8-sig")

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "ep": str(EP_PATH),
            "market": str(MARKET_PATH),
            "industry": str(INDUSTRY_PATH),
            "test_start": str(TEST_START_PATH),
        },
        "return_convention": {
            "formation": "last market trading day of month",
            "entry": "next month's first market trading day close_adj",
            "exit": "following month's first market trading day close_adj",
            "suspended_endpoint": "forward return set to NaN",
            "limit_flags": "audited but not filtered in RLM",
        },
        "factor": "daily MAD-winsorized and Z-standardized EP signal",
        "size_control": "cross-sectional Z-score of ln(me_total) within valid EP formation universe",
        "industry_control": "intercept plus K-1 Shenwan level-1 dummies",
        "rlm": {
            "norm": "HuberT",
            "tuning_constant": HUBER_T,
            "scale_estimator": "MAD",
            "covariance": "H1",
            "max_iterations": MAX_ITER,
            "tolerance": TOL,
        },
        "time_series_inference": {
            "method": "IID mean t-test",
            "standard_error": "sample standard deviation of monthly coefficients / sqrt(months)",
            "reference_distribution": "Student t with months - 1 degrees of freedom",
            "newey_west_hac_applied": False,
        },
        "periods": len(calendar),
        "first_formation_date": str(calendar["formation_date"].min().date()),
        "last_formation_date": str(calendar["formation_date"].max().date()),
        "bse_mapping": {
            "pairs": len(mapping_frame),
            "old_last_date": str(pd.Timestamp(transition_old_date).date()),
            "new_first_date": str(pd.Timestamp(transition_new_date).date()),
            "detected_from_data": True,
            "method": "exact close/pre_close; exact share_total resolves duplicate-price candidates",
        },
    }
    (OUT_DIR / "rlm_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_plot(monthly, OUT_DIR / "rlm_ep_premium.png")
    save_report(
        monthly,
        annual,
        overall,
        sample_audit,
        mapping_frame,
        calendar,
        OUT_DIR / "RLM统计结果.md",
    )

    row = overall.iloc[0]
    print("RLM completed", flush=True)
    print(f"months={len(monthly)}", flush=True)
    print(f"ep_mean={row['ep_mean']:.10f}", flush=True)
    print(f"ep_iid_t={row['ep_iid_t']:.6f}", flush=True)
    print(f"ep_iid_p={row['ep_iid_p']:.6f}", flush=True)
    print(f"ep_positive_ratio={row['ep_positive_ratio']:.6f}", flush=True)
    print(f"saved={OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
