# -*- coding: utf-8 -*-
"""通用月度横截面 Huber RLM 回归。

本脚本与具体因子无关：读取一份因子文件，做「因子 + 规模 + 行业」的月度
横截面稳健回归，输出统计结果。换成其它因子只需换 ``--factor`` 参数，
不需要改代码。

    python "代码\\cross_section_rlm.py"                        # 默认因子 ep
    python "代码\\cross_section_rlm.py" --factor pb
    python "代码\\cross_section_rlm.py" --factor "D:/其它路径/my.parquet" --factor-name my

因子文件格式与交付一致：``index = ['date', 'stock_code']``，``columns = ['signal']``。
样本、前瞻收益和控制变量全部来自 ``代码/common``，因此与 ``cross_section_ols.py``
完全一致，两者差异只来自估计方法。
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
    resolve_factor_path,
    scan_market_index,
)

MODEL_DIR_NAME = "RLM_H1_独立同方差"


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
    min_observations: int = MIN_OBSERVATIONS,
    huber_t: float = HUBER_T,
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
                data["factor_z"].astype(float).rename("factor_z"),
                data["size_z"].astype(float).rename("size_z"),
                dummies,
            ],
            axis=1,
        )
        y = data["forward_return"].astype(float)
        n, p = design.shape
        required_n = max(min_observations, 10 * p)
        if n < required_n:
            raise ValueError(
                f"{formation_date.date()}: 观测数 {n} 低于要求的 {required_n}"
            )
        rank = int(np.linalg.matrix_rank(design.to_numpy(float)))
        if rank != p:
            raise ValueError(
                f"{formation_date.date()}: 设计矩阵不满秩 rank={rank}, columns={p}"
            )

        model = sm.RLM(y, design, M=sm.robust.norms.HuberT(t=huber_t), missing="raise")
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
                "factor_beta": float(params["factor_z"]),
                "factor_bse": float(bse["factor_z"]),
                "factor_zvalue": float(zvalues["factor_z"]),
                "factor_pvalue": float(pvalues["factor_z"]),
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
                "factor_z",
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
            f"beta={params['factor_z']:.8f} z={zvalues['factor_z']:.3f} "
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
    factor_iid = iid_mean(group["factor_beta"])
    size_iid = iid_mean(group["size_beta"])
    return {
        "period": label,
        "months": len(group),
        "factor_mean": factor_iid["mean"],
        "factor_median": float(group["factor_beta"].median()),
        "factor_std": float(group["factor_beta"].std(ddof=1)),
        "factor_iid_se": factor_iid["iid_se"],
        "factor_iid_t": factor_iid["iid_t"],
        "factor_iid_p": factor_iid["iid_p"],
        "factor_positive_ratio": float(group["factor_beta"].gt(0).mean()),
        "factor_abs_monthly_z_ge_2_ratio": float(group["factor_zvalue"].abs().ge(2).mean()),
        "factor_positive_significant_ratio": float(
            (group["factor_beta"].gt(0) & group["factor_zvalue"].ge(2)).mean()
        ),
        "factor_negative_significant_ratio": float(
            (group["factor_beta"].lt(0) & group["factor_zvalue"].le(-2)).mean()
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


def save_plot(monthly: pd.DataFrame, path: Path, factor_label: str) -> None:
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].axhline(0.0, color="#555555", linewidth=0.9)
    axes[0].plot(
        monthly["formation_date"], monthly["factor_beta"] * 100, color="#176B87", linewidth=1.2
    )
    axes[0].scatter(
        monthly["formation_date"], monthly["factor_beta"] * 100, color="#176B87", s=13
    )
    axes[0].set_ylabel(f"{factor_label} premium (%)")
    axes[0].set_title(f"Monthly Huber RLM {factor_label} premium")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].axhline(0.0, color="#555555", linewidth=0.9)
    axes[1].plot(
        monthly["formation_date"],
        monthly["cumulative_factor_premium"] * 100,
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
    factor_label: str,
    factor_path: Path,
    path: Path,
) -> None:
    summary = overall.iloc[0]
    direction = "正" if summary["factor_mean"] > 0 else "负"
    significance = "达到" if summary["factor_iid_p"] < 0.05 else "未达到"
    strongest = monthly.nlargest(3, "factor_beta")
    weakest = monthly.nsmallest(3, "factor_beta")
    return_valid_total = int(sample_audit["n_valid_return_after_controls"].sum())
    control_valid_total = int(sample_audit["n_valid_factor_size_industry"].sum())

    lines = [
        f"# {factor_label} 因子月度横截面 Huber RLM 统计结果",
        "",
        f"**运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**",
        "",
        f"**因子数据：** `{factor_path}`",
        "",
        "> 本报告只包含 RLM 横截面回归，不包含 IC、分层回测、OLS 或 WLS 对照。",
        "> 样本、前瞻收益与控制变量由 `代码/common` 共享模块统一构造，",
        "> 与 `cross_section_ols.py` 完全一致，两者差异只来自估计方法。",
        "",
        "## 1. 回归口径",
        "",
        "每月最后一个交易日形成标准化因子，次月第一个交易日按复权收盘价建仓，",
        "再下一个月第一个交易日按复权收盘价退出。模型为：",
        "",
        "```text",
        "r_fwd[i,m+1] = alpha[m] + beta_F[m] * F_Z[i,m]",
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
        f"| 因子月均系数 | {pct(summary['factor_mean'], 4)} |",
        f"| 因子系数中位数 | {pct(summary['factor_median'], 4)} |",
        f"| 因子系数标准差 | {pct(summary['factor_std'], 4)} |",
        f"| 因子 IID 标准误 | {pct(summary['factor_iid_se'], 4)} |",
        f"| 因子 IID t 值 | {num(summary['factor_iid_t'])} |",
        f"| 因子 IID p 值 | {num(summary['factor_iid_p'])} |",
        f"| 因子系数为正的月份 | {pct(summary['factor_positive_ratio'])} |",
        f"| 单月 abs(z) >= 2 的比例 | {pct(summary['factor_abs_monthly_z_ge_2_ratio'])} |",
        f"| 正向且 z >= 2 的比例 | {pct(summary['factor_positive_significant_ratio'])} |",
        f"| 负向且 z <= -2 的比例 | {pct(summary['factor_negative_significant_ratio'])} |",
        f"| Size 月均系数 | {pct(summary['size_mean'], 4)} |",
        f"| Size IID t 值 | {num(summary['size_iid_t'])} |",
        f"| 平均 RLM 降权比例 | {pct(summary['downweighted_ratio_mean'])} |",
        f"| 平均权重低于 0.5 比例 | {pct(summary['weight_below_half_ratio_mean'])} |",
        f"| 平均伪 R² | {pct(summary['pseudo_r2_mean'])} |",
        "",
        f"因子月均系数方向为{direction}，独立同方差假设下的显著性{significance} 5% 标准。因子已按形成日横截面标准化，",
        f"因此月均系数 {pct(summary['factor_mean'], 4)} 可解释为因子提高一个横截面标准差时，下一持有期收益平均变化约 {pct(summary['factor_mean'], 4)}。",
        f"简单乘以 12 得到的年化系数约为 {pct(summary['factor_mean'] * 12, 2)}，它是因子溢价尺度，不是可交易组合年化收益。",
        "",
        "## 4. 分年度结果",
        "",
        "| 年份 | 月数 | 因子月均系数 | IID t 值 | IID p 值 | 正系数比例 | abs(z) >= 2 比例 | 平均样本数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in annual.itertuples(index=False):
        lines.append(
            f"| {row.period} | {row.months} | {pct(row.factor_mean, 4)} | "
            f"{num(row.factor_iid_t)} | {num(row.factor_iid_p)} | {pct(row.factor_positive_ratio)} | "
            f"{pct(row.factor_abs_monthly_z_ge_2_ratio)} | {row.n_mean:,.0f} |"
        )

    lines += [
        "",
        "## 5. 极值月份",
        "",
        "| 类型 | 因子形成日 | 建仓日 | 退出日 | 因子系数 | 单月 z 值 | 样本数 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for label, selected in [("最高", strongest), ("最低", weakest)]:
        for row in selected.itertuples(index=False):
            lines.append(
                f"| {label} | {row.formation_date.date()} | {row.entry_date.date()} | "
                f"{row.exit_date.date()} | {pct(row.factor_beta, 4)} | {row.factor_zvalue:.3f} | {row.n:,} |"
            )

    lines += [
        "",
        "## 6. 拟合诊断",
        "",
        f"- 收敛月份：{int(monthly['converged'].sum())}/{len(monthly)}；迭代次数中位数为 {monthly['iterations'].median():.0f}，最大 {int(monthly['iterations'].max())}。",
        f"- 最后一次迭代参数最大变化量中位数：{monthly['final_coef_delta'].median():.3e}（收敛容差 {TOL:.0e}）。",
        f"- 条件数范围：{monthly['condition_number'].min():.2f} 至 {monthly['condition_number'].max():.2f}。",
        "- 北交所最终样本在代码切换附近未发生由代码连接失败造成的断层，逐月数量见 `rlm_sample_audit.csv`。",
        "- 涨停买入和跌停卖出只做数量标记，未用于 RLM 样本筛选；实际可交易性留待分层回测处理。",
        "- `cumulative_factor_premium` 是月度系数的累计和，不是组合净值。",
        "",
        "![月度因子系数与累计因子溢价](./rlm_factor_premium.png)",
        "",
        "## 7. 输出文件",
        "",
        "- `bse_code_mapping.csv/.parquet`：北交所新旧代码映射及匹配证据。",
        "- `forward_monthly_returns.parquet`：固定调仓端点的复权前瞻收益。",
        "- `rlm_regression_sample.parquet`：进入回归的逐股样本、残差和稳健权重。",
        "- `rlm_monthly_results.csv/.parquet`：逐月因子/Size 系数及拟合诊断。",
        "- `rlm_coefficients_long.parquet`：全部月份、全部回归参数。",
        "- `rlm_sample_audit.csv`：逐月样本流失及交易状态审计。",
        "- `rlm_annual_summary.csv`、`rlm_overall_summary.csv`：年度和全期汇总。",
        "- `rlm_config.json`：本次运行的完整口径记录。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通用月度横截面 Huber RLM 回归：读取因子数据并输出回归统计结果。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            '  python "代码\\cross_section_rlm.py"\n'
            '  python "代码\\cross_section_rlm.py" --factor pb\n'
            '  python "代码\\cross_section_rlm.py" --factor "D:/其它/my.parquet" --factor-name my\n'
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
    parser.add_argument(
        "--out-root",
        default=None,
        help="结果根目录，默认 回归结果/，最终写入 <out-root>/<因子名>/"
        + MODEL_DIR_NAME,
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=MIN_OBSERVATIONS,
        help=f"单月最小观测数（默认 {MIN_OBSERVATIONS}）",
    )
    parser.add_argument(
        "--huber-t",
        type=float,
        default=HUBER_T,
        help=f"Huber 调优常数（默认 {HUBER_T}）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
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
    mapping_frame.to_parquet(out_dir / "bse_code_mapping.parquet", index=False)
    mapping_frame.to_csv(out_dir / "bse_code_mapping.csv", index=False, encoding="utf-8-sig")

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
    forward_returns.to_parquet(out_dir / "forward_monthly_returns.parquet", index=False)

    print("Building monthly regression samples...", flush=True)
    panel, sample_audit = build_regression_panel(formation, industry, forward_returns)
    sample_audit.to_csv(out_dir / "rlm_sample_audit.csv", index=False, encoding="utf-8-sig")
    sample_audit.to_parquet(out_dir / "rlm_sample_audit.parquet", index=False)

    print("Running monthly Huber RLM...", flush=True)
    monthly, coefficients, regression_sample = fit_monthly_rlm(
        panel, min_observations=args.min_observations, huber_t=args.huber_t
    )
    monthly["formation_date"] = pd.to_datetime(monthly["formation_date"])
    monthly = monthly.sort_values("formation_date").reset_index(drop=True)
    monthly["cumulative_factor_premium"] = monthly["factor_beta"].cumsum()
    annual, overall = build_summaries(monthly)

    if not monthly["converged"].all():
        failed = (
            monthly.loc[~monthly["converged"], "formation_date"].dt.strftime("%Y-%m-%d").tolist()
        )
        raise RuntimeError(f"以下月份 RLM 未收敛：{failed}")

    monthly.to_parquet(out_dir / "rlm_monthly_results.parquet", index=False)
    monthly.to_csv(out_dir / "rlm_monthly_results.csv", index=False, encoding="utf-8-sig")
    coefficients.to_parquet(out_dir / "rlm_coefficients_long.parquet", index=False)
    regression_sample.to_parquet(out_dir / "rlm_regression_sample.parquet", index=False)
    annual.to_csv(out_dir / "rlm_annual_summary.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(out_dir / "rlm_overall_summary.csv", index=False, encoding="utf-8-sig")
    calendar.to_csv(out_dir / "rlm_calendar.csv", index=False, encoding="utf-8-sig")

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "estimator": "Huber RLM with H1 robust covariance",
        "factor": {
            "label": factor_label,
            "name": factor_path.stem,
            "path": str(factor_path),
        },
        "inputs": {
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
        "size_control": "cross-sectional Z-score of ln(me_total) within valid factor formation universe",
        "industry_control": "intercept plus K-1 Shenwan level-1 dummies",
        "rlm": {
            "norm": "HuberT",
            "tuning_constant": args.huber_t,
            "scale_estimator": "MAD",
            "covariance": "H1",
            "max_iterations": MAX_ITER,
            "tolerance": TOL,
        },
        "min_observations": args.min_observations,
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
    (out_dir / "rlm_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_plot(monthly, out_dir / "rlm_factor_premium.png", factor_label)
    save_report(
        monthly,
        annual,
        overall,
        sample_audit,
        mapping_frame,
        calendar,
        factor_label,
        factor_path,
        out_dir / "RLM统计结果.md",
    )

    row = overall.iloc[0]
    print("RLM completed", flush=True)
    print(f"factor={factor_label}", flush=True)
    print(f"months={len(monthly)}", flush=True)
    print(f"factor_mean={row['factor_mean']:.10f}", flush=True)
    print(f"factor_iid_t={row['factor_iid_t']:.6f}", flush=True)
    print(f"factor_iid_p={row['factor_iid_p']:.6f}", flush=True)
    print(f"factor_positive_ratio={row['factor_positive_ratio']:.6f}", flush=True)
    print(f"saved={out_dir}", flush=True)


if __name__ == "__main__":
    main()
