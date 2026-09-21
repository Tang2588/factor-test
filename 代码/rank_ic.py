# -*- coding: utf-8 -*-
"""通用月度 Rank IC 检验。

与具体因子无关：读取一份因子文件，输出该因子的月度 Rank IC 统计结果。
换因子只需改 ``--factor`` 参数，不需要改代码。

    python "代码\\rank_ic.py"
    python "代码\\rank_ic.py" --factor pb
    python "代码\\rank_ic.py" --factor "D:/其它/my.parquet" --factor-name MY

口径（与《EP_PE统计分析与因子测试计划》1.5 节一致）：

1. 每月形成日把标准化因子对「标准化对数市值 + K-1 个行业哑变量」做截面回归；
2. 取残差 ``X_neutral`` 作为行业和市值中性化后的因子暴露；
3. Rank IC = Spearman(X_neutral, 前瞻收益)；
4. IR = IC 均值 / IC 标准差。

同时输出未中性化的原始 IC 作为对照。样本、前瞻收益、控制变量全部来自
``代码/common``，与 ``cross_section_rlm.py`` 完全一致，因此 IC 与 RLM 可直接对照。
时间序列推断使用 IC 序列的独立同方差 t 检验，不执行 Newey-West HAC。
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
from scipy.stats import spearmanr
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    INDUSTRY_PATH,
    MARKET_PATH,
    MIN_OBSERVATIONS,
    REG_DIR,
    TEST_START_PATH,
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

MODEL_DIR_NAME = "RankIC"
IC_STRONG_THRESHOLD = 0.02


def neutralized_ic(
    panel: pd.DataFrame,
    min_observations: int = MIN_OBSERVATIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows = []
    residual_rows = []

    for formation_date, raw_group in panel.groupby("formation_date", sort=True):
        data = raw_group.loc[raw_group["regression_eligible"]].copy().reset_index(drop=True)
        industries = sorted(data["industry_lv1"].astype(str).unique())
        dummies = pd.get_dummies(
            data["industry_lv1"].astype(str),
            prefix="industry",
            drop_first=True,
            dtype=float,
        )
        controls = pd.concat(
            [
                pd.Series(1.0, index=data.index, name="const"),
                data["size_z"].astype(float).rename("size_z"),
                dummies,
            ],
            axis=1,
        )
        factor = data["factor_z"].astype(float)
        forward = data["forward_return"].astype(float)
        n, p = controls.shape
        required_n = max(min_observations, 10 * p)
        if n < required_n:
            raise ValueError(
                f"{formation_date.date()}: 观测数 {n} 低于要求的 {required_n}"
            )
        if int(np.linalg.matrix_rank(controls.to_numpy(float))) != p:
            raise ValueError(f"{formation_date.date()}: 中性化回归设计矩阵不满秩")

        fitted = sm.OLS(factor, controls, missing="raise").fit()
        residual = pd.Series(np.asarray(fitted.resid, dtype=float), index=data.index)

        ic_neutral = float(
            spearmanr(residual.to_numpy(float), forward.to_numpy(float)).statistic
        )
        ic_raw = float(
            spearmanr(factor.to_numpy(float), forward.to_numpy(float)).statistic
        )

        monthly_rows.append(
            {
                "formation_date": formation_date,
                "entry_date": data["entry_date"].iloc[0],
                "exit_date": data["exit_date"].iloc[0],
                "n": n,
                "p": p,
                "industries": len(industries),
                "ic_neutral": ic_neutral,
                "ic_raw": ic_raw,
                "ic_neutral_positive": ic_neutral > 0,
                "ic_neutral_strong": abs(ic_neutral) > IC_STRONG_THRESHOLD,
                "residual_std": float(residual.std(ddof=1)),
                "neutral_r2": float(fitted.rsquared),
                "return_mean": float(forward.mean()),
            }
        )

        enriched = data[
            [
                "formation_date",
                "security_id",
                "factor_z",
                "size_z",
                "industry_lv1",
                "forward_return",
            ]
        ].copy()
        enriched["neutral_residual"] = residual.to_numpy(float)
        residual_rows.append(enriched)
        print(
            f"IC {formation_date.date()} n={n:,} "
            f"ic={ic_neutral:+.4f} raw={ic_raw:+.4f}",
            flush=True,
        )

    return pd.DataFrame(monthly_rows), pd.concat(residual_rows, ignore_index=True)


def iid_mean(values: pd.Series) -> dict[str, float]:
    """IC 序列的均值、标准差、IR 和独立同方差 t 检验。"""
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(clean) < 2:
        return {"mean": np.nan, "std": np.nan, "ir": np.nan, "t": np.nan, "p": np.nan}
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
        "std": std,
        "ir": mean / std if std > 0 else np.nan,
        "t": t_value,
        "p": p_value,
    }


def summarize_period(group: pd.DataFrame, label: str) -> dict[str, float | str | int]:
    neutral = iid_mean(group["ic_neutral"])
    raw = iid_mean(group["ic_raw"])
    return {
        "period": label,
        "months": len(group),
        "ic_mean": neutral["mean"],
        "ic_std": neutral["std"],
        "ir": neutral["ir"],
        "ic_t": neutral["t"],
        "ic_p": neutral["p"],
        "ic_median": float(group["ic_neutral"].median()),
        "ic_positive_ratio": float(group["ic_neutral"].gt(0).mean()),
        "ic_strong_ratio": float(group["ic_neutral"].abs().gt(IC_STRONG_THRESHOLD).mean()),
        "ic_raw_mean": raw["mean"],
        "ic_raw_std": raw["std"],
        "ic_raw_median": float(group["ic_raw"].median()),
        "ic_raw_t": raw["t"],
        "ic_raw_p": raw["p"],
        "ic_raw_ir": raw["ir"],
        "ic_raw_positive_ratio": float(group["ic_raw"].gt(0).mean()),
        "ic_raw_strong_ratio": float(group["ic_raw"].abs().gt(IC_STRONG_THRESHOLD).mean()),
        "n_mean": float(group["n"].mean()),
        "neutral_r2_mean": float(group["neutral_r2"].mean()),
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
    axes[0].bar(
        monthly["formation_date"],
        monthly["ic_neutral"],
        width=20,
        color=np.where(monthly["ic_neutral"] >= 0, "#176B87", "#B23A48"),
    )
    axes[0].set_ylabel("Monthly Rank IC (neutralized)")
    axes[0].set_title(f"Monthly Rank IC of {factor_label}")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].axhline(0.0, color="#555555", linewidth=0.9)
    axes[1].plot(
        monthly["formation_date"],
        monthly["cumulative_ic"],
        color="#8A5A2B",
        linewidth=1.5,
    )
    axes[1].set_ylabel("Cumulative IC")
    axes[1].set_xlabel("Formation date")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def num(value: float, digits: int = 4) -> str:
    return "NaN" if not np.isfinite(value) else f"{value:.{digits}f}"


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def save_report(
    monthly: pd.DataFrame,
    annual: pd.DataFrame,
    overall: pd.DataFrame,
    factor_label: str,
    factor_path: Path,
    path: Path,
) -> None:
    summary = overall.iloc[0]
    strongest = monthly.nlargest(3, "ic_neutral")
    weakest = monthly.nsmallest(3, "ic_neutral")
    direction = "正" if summary["ic_mean"] > 0 else "负"
    significance = "达到" if summary["ic_p"] < 0.05 else "未达到"

    lines = [
        f"# {factor_label} 因子月度 Rank IC 统计结果",
        "",
        f"**运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**",
        "",
        f"**因子数据：** `{factor_path}`",
        "",
        "> 本报告只包含 Rank IC 检验，不包含截面回归、分层回测或交易成本检验。",
        "> 样本、前瞻收益与控制变量与 `cross_section_rlm.py` 完全一致，两者可直接对照。",
        "",
        "## 1. 检验口径",
        "",
        "每月形成日把标准化因子对「标准化对数市值 + K-1 个行业哑变量」做横截面 OLS，",
        "取残差作为行业和市值中性化后的因子暴露，再计算该暴露与下一持有期收益的",
        "Spearman 秩相关：",
        "",
        "```text",
        "IC_t = Spearman(X_neutral[i,t], r_fwd[i,t+1])",
        "IR   = mean(IC) / std(IC)",
        "```",
        "",
        "同时报告未中性化的原始 IC 作为对照。显著性使用 IC 序列的独立同方差 t 检验，",
        "不执行 Newey-West HAC。",
        "",
        "## 2. 全期结果",
        "",
        "| 指标 | 中性化 IC | 原始 IC |",
        "|---|---:|---:|",
        f"| 检验月份 | {int(summary['months'])} | {int(summary['months'])} |",
        f"| IC 均值 | {num(summary['ic_mean'])} | {num(summary['ic_raw_mean'])} |",
        f"| IC 中位数 | {num(summary['ic_median'])} | {num(summary['ic_raw_median'])} |",
        f"| IC 标准差 | {num(summary['ic_std'])} | {num(summary['ic_raw_std'])} |",
        f"| IR | {num(summary['ir'])} | {num(summary['ic_raw_ir'])} |",
        f"| IC 均值的 IID t 值 | {num(summary['ic_t'])} | {num(summary['ic_raw_t'])} |",
        f"| IC 均值的 IID p 值 | {num(summary['ic_p'])} | {num(summary['ic_raw_p'])} |",
        f"| IC > 0 的月份比例 | {pct(summary['ic_positive_ratio'])} | {pct(summary['ic_raw_positive_ratio'])} |",
        f"| abs(IC) > {IC_STRONG_THRESHOLD} 的月份比例 | {pct(summary['ic_strong_ratio'])} | {pct(summary['ic_raw_strong_ratio'])} |",
        f"| 平均样本数 | {summary['n_mean']:,.0f} | {summary['n_mean']:,.0f} |",
        "",
        f"中性化 IC 均值为 {num(summary['ic_mean'])}，方向为{direction}，独立同方差检验下",
        f"显著性{significance} 5% 标准；IR 为 {num(summary['ir'])}。IR 是月频指标，未做年化；",
        "IC 均值乘以 12 只是粗略的年化尺度，不代表可交易收益。",
        "",
        "## 3. 分年度结果",
        "",
        "**表 1：中性化 IC（主口径）**",
        "",
        "| 年份 | 月数 | IC 均值 | IC 标准差 | IR | IC > 0 比例 | abs(IC) > 0.02 比例 | 平均样本数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in annual.itertuples(index=False):
        lines.append(
            f"| {row.period} | {row.months} | {num(row.ic_mean)} | {num(row.ic_std)} | "
            f"{num(row.ir)} | {pct(row.ic_positive_ratio)} | {pct(row.ic_strong_ratio)} | "
            f"{row.n_mean:,.0f} |"
        )
    lines += [
        "",
        "**表 2：原始 IC（对照口径）**",
        "",
        "| 年份 | 月数 | IC 均值 | IC 标准差 | IR | IC > 0 比例 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in annual.itertuples(index=False):
        lines.append(
            f"| {row.period} | {row.months} | {num(row.ic_raw_mean)} | {num(row.ic_raw_std)} | "
            f"{num(row.ic_raw_ir)} | {pct(row.ic_raw_positive_ratio)} |"
        )

    lines += [
        "",
        "## 4. 极值月份",
        "",
        "| 类型 | 因子形成日 | 建仓日 | 退出日 | 中性化 IC | 原始 IC | 样本数 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for label, selected in [("最高", strongest), ("最低", weakest)]:
        for row in selected.itertuples(index=False):
            lines.append(
                f"| {label} | {row.formation_date.date()} | {row.entry_date.date()} | "
                f"{row.exit_date.date()} | {num(row.ic_neutral)} | {num(row.ic_raw)} | {row.n:,} |"
            )

    lines += [
        "",
        "## 5. 诊断",
        "",
        f"- 中性化回归平均 R² 为 {pct(summary['neutral_r2_mean'])}，即市值和行业平均解释了因子",
        "  横截面差异的这一比例，其余部分才进入 IC 计算。",
        f"- 单月样本数：最少 {int(monthly['n'].min()):,}，中位数 {int(monthly['n'].median()):,}，最多 {int(monthly['n'].max()):,}。",
        "",
        "![月度 Rank IC 与累计 IC](./ic_timeseries.png)",
        "",
        "## 6. 输出文件",
        "",
        "- `ic_monthly.csv/.parquet`：逐月中性化与原始 IC 及诊断。",
        "- `ic_residual_sample.parquet`：逐股中性化残差与前瞻收益。",
        "- `ic_annual_summary.csv`、`ic_overall_summary.csv`：年度和全期汇总。",
        "- `ic_config.json`：本次运行的完整口径记录。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通用月度 Rank IC 检验：读取因子数据并输出 IC / IR 统计结果。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            '  python "代码\\rank_ic.py"\n'
            '  python "代码\\rank_ic.py" --factor pb\n'
            '  python "代码\\rank_ic.py" --factor "D:/其它/my.parquet" --factor-name MY\n'
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
        help="结果根目录，默认 回归结果/",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=MIN_OBSERVATIONS,
        help=f"单月最小观测数（默认 {MIN_OBSERVATIONS}）",
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

    print("Running monthly Rank IC...", flush=True)
    monthly, residual_sample = neutralized_ic(panel, min_observations=args.min_observations)
    monthly["formation_date"] = pd.to_datetime(monthly["formation_date"])
    monthly = monthly.sort_values("formation_date").reset_index(drop=True)
    monthly["cumulative_ic"] = monthly["ic_neutral"].cumsum()
    annual, overall = build_summaries(monthly)

    monthly.to_parquet(out_dir / "ic_monthly.parquet", index=False)
    monthly.to_csv(out_dir / "ic_monthly.csv", index=False, encoding="utf-8-sig")
    residual_sample.to_parquet(out_dir / "ic_residual_sample.parquet", index=False)
    annual.to_csv(out_dir / "ic_annual_summary.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(out_dir / "ic_overall_summary.csv", index=False, encoding="utf-8-sig")
    calendar.to_csv(out_dir / "ic_calendar.csv", index=False, encoding="utf-8-sig")
    sample_audit.to_csv(out_dir / "ic_sample_audit.csv", index=False, encoding="utf-8-sig")

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "estimator": "cross-sectional OLS neutralization, then Spearman rank IC",
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
        "neutralization": "regress factor on intercept + size_z + K-1 Shenwan level-1 dummies",
        "ic_definition": "Spearman(neutralized residual, forward return)",
        "strong_ic_threshold": IC_STRONG_THRESHOLD,
        "return_convention": {
            "formation": "last market trading day of month",
            "entry": "next month's first market trading day close_adj",
            "exit": "following month's first market trading day close_adj",
            "suspended_endpoint": "forward return set to NaN",
        },
        "min_observations": args.min_observations,
        "time_series_inference": {
            "method": "IID mean t-test on IC series",
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
        },
    }
    (out_dir / "ic_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_plot(monthly, out_dir / "ic_timeseries.png", factor_label)
    save_report(
        monthly,
        annual,
        overall,
        factor_label,
        factor_path,
        out_dir / "IC统计结果.md",
    )

    row = overall.iloc[0]
    print("Rank IC completed", flush=True)
    print(f"factor={factor_label}", flush=True)
    print(f"months={len(monthly)}", flush=True)
    print(f"ic_mean={row['ic_mean']:.10f}", flush=True)
    print(f"ir={row['ir']:.6f}", flush=True)
    print(f"ic_t={row['ic_t']:.6f}", flush=True)
    print(f"ic_p={row['ic_p']:.6f}", flush=True)
    print(f"ic_positive_ratio={row['ic_positive_ratio']:.6f}", flush=True)
    print(f"saved={out_dir}", flush=True)


if __name__ == "__main__":
    main()
