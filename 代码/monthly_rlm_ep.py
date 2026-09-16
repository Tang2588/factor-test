# -*- coding: utf-8 -*-
"""Run monthly cross-sectional Huber RLM for the standardized EP factor."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
BASE = Path(r"D:\实习生学习项目\基础数据")
EP_PATH = ROOT / "因子结果" / "ep.parquet"
MARKET_PATH = BASE / "chn_equ_mkt_quotation.parquet"
INDUSTRY_PATH = BASE / "chn_equ_indus_sw.parquet"
TEST_START_PATH = ROOT / "中间结果" / "factor_test_start.json"
OUT_DIR = ROOT / "回归结果" / "RLM_新口径"

HUBER_T = 1.345
MAX_ITER = 100
TOL = 1e-8
HAC_LAGS = 3
MIN_OBSERVATIONS = 500
EXPECTED_BSE_CONVERSIONS = 242
TRANSITION_OLD_DATE = pd.Timestamp("2025-09-30")
TRANSITION_NEW_DATE = pd.Timestamp("2025-10-09")


def read_selected_dates(
    path: Path,
    columns: list[str],
    dates: set[pd.Timestamp],
    batch_size: int = 200_000,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=batch_size):
        frame = batch.to_pandas(ignore_metadata=True)
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        selected = frame.loc[frame["date"].isin(dates)].copy()
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        return pd.DataFrame(columns=columns)
    return pd.concat(pieces, ignore_index=True)


def read_market_dates(path: Path) -> pd.DatetimeIndex:
    dates: set[pd.Timestamp] = set()
    for batch in pq.ParquetFile(path).iter_batches(columns=["date"], batch_size=300_000):
        values = pd.to_datetime(batch.to_pandas(ignore_metadata=True)["date"]).dt.normalize()
        dates.update(values)
    return pd.DatetimeIndex(sorted(dates))


def normalize_market_code(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def extract_code6(series: pd.Series) -> pd.Series:
    return normalize_market_code(series).str.extract(r"(\d{6})", expand=False)


def b_share_mask(series: pd.Series) -> pd.Series:
    code = normalize_market_code(series)
    sh_b = code.str.startswith("900", na=False) & code.str.endswith(".BJ", na=False)
    sz_b = code.str.startswith(("200", "201"), na=False) & code.str.endswith(".SZ", na=False)
    return sh_b | sz_b


def build_monthly_calendar(
    market_dates: pd.DatetimeIndex,
    test_start: pd.Timestamp,
) -> pd.DataFrame:
    date_frame = pd.DataFrame({"date": market_dates})
    periods = date_frame["date"].dt.to_period("M")
    first_by_month = date_frame.groupby(periods)["date"].min().to_dict()
    last_by_month = date_frame.groupby(periods)["date"].max().to_dict()

    rows = []
    for period in sorted(last_by_month):
        formation_date = pd.Timestamp(last_by_month[period])
        entry_period = period + 1
        exit_period = period + 2
        if formation_date < test_start:
            continue
        if entry_period not in first_by_month or exit_period not in first_by_month:
            continue
        rows.append(
            {
                "formation_month": str(period),
                "formation_date": formation_date,
                "entry_date": pd.Timestamp(first_by_month[entry_period]),
                "exit_date": pd.Timestamp(first_by_month[exit_period]),
            }
        )
    calendar = pd.DataFrame(rows)
    if calendar.empty:
        raise ValueError("No complete monthly forward-return intervals are available")
    return calendar


def prepare_market_snapshot(raw: pd.DataFrame) -> pd.DataFrame:
    market = raw.copy()
    market["market_code"] = normalize_market_code(market["stock_code"])
    market = market.loc[~b_share_mask(market["market_code"])].copy()
    market["code6"] = extract_code6(market["market_code"])
    if market["code6"].isna().any():
        raise ValueError("Market data contains an unparseable stock code")

    suspended_numeric = pd.to_numeric(market["suspended"], errors="coerce")
    status = market["status"].astype("string").str.strip()
    market["suspended_flag"] = (
        suspended_numeric.eq(1).fillna(False) | status.eq("停牌").fillna(False)
    )
    for column in [
        "close",
        "pre_close",
        "close_adj",
        "pre_close_adj",
        "share_total",
        "me_total",
        "up_limit",
        "down_limit",
    ]:
        market[column] = pd.to_numeric(market[column], errors="coerce")
    market["at_up_limit"] = np.isclose(
        market["close"], market["up_limit"], rtol=0.0, atol=1e-8, equal_nan=False
    )
    market["at_down_limit"] = np.isclose(
        market["close"], market["down_limit"], rtol=0.0, atol=1e-8, equal_nan=False
    )
    if market.duplicated(["date", "market_code"]).any():
        raise ValueError("Market data contains duplicate date + market_code rows")
    return market


def derive_bse_mapping(market: pd.DataFrame) -> pd.DataFrame:
    old = market.loc[
        market["date"].eq(TRANSITION_OLD_DATE)
        & market["market_code"].str.match(r"^(43|83|87)\d{4}\.BJ$", na=False),
        ["market_code", "close", "close_adj", "share_total"],
    ].rename(
        columns={
            "market_code": "old_code",
            "close": "old_close",
            "close_adj": "old_close_adj",
            "share_total": "old_share_total",
        }
    )
    new = market.loc[
        market["date"].eq(TRANSITION_NEW_DATE)
        & market["market_code"].str.match(r"^920\d{3}\.BJ$", na=False),
        ["market_code", "pre_close", "pre_close_adj", "share_total"],
    ].rename(
        columns={
            "market_code": "new_code",
            "pre_close": "new_pre_close",
            "pre_close_adj": "new_pre_close_adj",
            "share_total": "new_share_total",
        }
    )
    preexisting_new_codes = set(
        market.loc[
            market["date"].eq(TRANSITION_OLD_DATE)
            & market["market_code"].str.match(r"^920\d{3}\.BJ$", na=False),
            "market_code",
        ]
    )

    candidates = old.merge(
        new,
        left_on="old_close",
        right_on="new_pre_close",
        how="left",
        validate="many_to_many",
    )
    candidates["share_total_equal"] = np.isclose(
        candidates["old_share_total"],
        candidates["new_share_total"],
        rtol=1e-12,
        atol=1e-6,
        equal_nan=False,
    )

    selected_rows = []
    for old_code, group in candidates.groupby("old_code", sort=True):
        available = group.loc[group["new_code"].notna()].copy()
        if len(available) == 1:
            selected = available.iloc[0].copy()
            selected["match_method"] = "exact_price"
        else:
            exact_share = available.loc[available["share_total_equal"]]
            if len(exact_share) != 1:
                raise ValueError(
                    f"BSE mapping is ambiguous for {old_code}: "
                    f"price candidates={len(available)}, share matches={len(exact_share)}"
                )
            selected = exact_share.iloc[0].copy()
            selected["match_method"] = "exact_price_and_share_total"
        selected_rows.append(selected)

    mapping = pd.DataFrame(selected_rows).reset_index(drop=True)
    newly_converted = set(new["new_code"]) - preexisting_new_codes
    if len(mapping) != EXPECTED_BSE_CONVERSIONS:
        raise ValueError(f"Expected {EXPECTED_BSE_CONVERSIONS} BSE mappings, got {len(mapping)}")
    if mapping["old_code"].nunique() != len(mapping) or mapping["new_code"].nunique() != len(mapping):
        raise ValueError("BSE old/new code mapping is not one-to-one")
    if set(mapping["new_code"]) != newly_converted:
        raise ValueError("BSE mapping targets do not equal the newly converted 920 codes")
    if not np.allclose(
        mapping["old_close"], mapping["new_pre_close"], rtol=0.0, atol=1e-10
    ):
        raise ValueError("BSE raw prices are discontinuous across the code conversion")
    if not np.allclose(
        mapping["old_close_adj"], mapping["new_pre_close_adj"], rtol=0.0, atol=1e-10
    ):
        raise ValueError("BSE adjusted prices are discontinuous across the code conversion")

    mapping.insert(0, "old_last_date", TRANSITION_OLD_DATE)
    mapping.insert(1, "new_first_date", TRANSITION_NEW_DATE)
    return mapping[
        [
            "old_last_date",
            "new_first_date",
            "old_code",
            "new_code",
            "match_method",
            "old_close",
            "new_pre_close",
            "old_close_adj",
            "new_pre_close_adj",
            "old_share_total",
            "new_share_total",
        ]
    ]


def apply_security_id(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    result = frame.copy()
    mapped = result["market_code"].map(mapping)
    result["security_id"] = mapped.fillna(result["market_code"]).astype("string")
    return result


def prepare_formation_panel(
    market: pd.DataFrame,
    ep: pd.DataFrame,
    signal_dates: set[pd.Timestamp],
    mapping: dict[str, str],
) -> pd.DataFrame:
    formation = market.loc[market["date"].isin(signal_dates)].copy()
    formation = apply_security_id(formation, mapping)
    if formation.duplicated(["date", "security_id"]).any():
        raise ValueError("Formation market panel is not unique after security mapping")

    factor = ep.copy()
    factor["code6"] = extract_code6(factor["stock_code"])
    factor["ep_z"] = pd.to_numeric(factor["signal"], errors="coerce")
    factor = factor[["date", "code6", "ep_z"]]
    if factor.duplicated(["date", "code6"]).any():
        raise ValueError("EP factor is not unique by date + code6")

    formation = formation.merge(
        factor,
        on=["date", "code6"],
        how="left",
        validate="one_to_one",
        indicator="factor_join",
    )
    if not formation["factor_join"].eq("both").all():
        missing = int(formation["factor_join"].ne("both").sum())
        raise ValueError(f"Formation market rows missing from EP output: {missing}")
    formation = formation.drop(columns="factor_join")

    formation["log_mv"] = np.where(
        np.isfinite(formation["me_total"]) & formation["me_total"].gt(0),
        np.log(formation["me_total"]),
        np.nan,
    )
    formation["size_z"] = np.nan
    for _, group in formation.groupby("date", sort=False):
        valid = np.isfinite(group["ep_z"]) & np.isfinite(group["log_mv"])
        values = group.loc[valid, "log_mv"]
        std = float(values.std(ddof=0))
        if len(values) and std > 0:
            formation.loc[values.index, "size_z"] = (values - values.mean()) / std

    return formation.rename(
        columns={"date": "formation_date", "market_code": "code_at_formation"}
    )


def prepare_industry(
    raw: pd.DataFrame,
    mapping: dict[str, str],
) -> pd.DataFrame:
    industry = raw.copy()
    industry["market_code"] = normalize_market_code(industry["stock_code"])
    industry = industry.loc[~b_share_mask(industry["market_code"])].copy()
    industry = apply_security_id(industry, mapping)
    industry = industry.rename(
        columns={"date": "formation_date", "indus_name_lv1": "industry_lv1"}
    )[["formation_date", "security_id", "industry_lv1"]]
    if industry.duplicated(["formation_date", "security_id"]).any():
        duplicates = industry.loc[
            industry.duplicated(["formation_date", "security_id"], keep=False)
        ]
        raise ValueError(f"Industry panel has mapped duplicates: {len(duplicates)}")
    return industry


def build_forward_returns(
    market: pd.DataFrame,
    calendar: pd.DataFrame,
    mapping: dict[str, str],
) -> pd.DataFrame:
    canonical = apply_security_id(market, mapping)
    if canonical.duplicated(["date", "security_id"]).any():
        raise ValueError("Market panel is not unique after security mapping")

    pieces = []
    for row in calendar.itertuples(index=False):
        entry = canonical.loc[
            canonical["date"].eq(row.entry_date),
            [
                "security_id",
                "market_code",
                "close_adj",
                "suspended_flag",
                "at_up_limit",
            ],
        ].rename(
            columns={
                "market_code": "code_at_entry",
                "close_adj": "entry_close_adj",
                "suspended_flag": "entry_suspended",
                "at_up_limit": "entry_at_up_limit",
            }
        )
        exit_frame = canonical.loc[
            canonical["date"].eq(row.exit_date),
            [
                "security_id",
                "market_code",
                "close_adj",
                "suspended_flag",
                "at_down_limit",
            ],
        ].rename(
            columns={
                "market_code": "code_at_exit",
                "close_adj": "exit_close_adj",
                "suspended_flag": "exit_suspended",
                "at_down_limit": "exit_at_down_limit",
            }
        )
        merged = entry.merge(exit_frame, on="security_id", how="outer", validate="one_to_one")
        entry_present = merged["code_at_entry"].notna()
        exit_present = merged["code_at_exit"].notna()
        valid_entry_price = np.isfinite(merged["entry_close_adj"]) & merged["entry_close_adj"].gt(0)
        valid_exit_price = np.isfinite(merged["exit_close_adj"]) & merged["exit_close_adj"].gt(0)
        entry_not_suspended = merged["entry_suspended"].fillna(True).eq(False)
        exit_not_suspended = merged["exit_suspended"].fillna(True).eq(False)
        available = (
            entry_present
            & exit_present
            & valid_entry_price
            & valid_exit_price
            & entry_not_suspended
            & exit_not_suspended
        )
        merged["forward_return"] = np.where(
            available,
            merged["exit_close_adj"] / merged["entry_close_adj"] - 1.0,
            np.nan,
        )
        merged["return_available"] = available
        merged["missing_reason"] = np.select(
            [
                ~entry_present,
                entry_present & ~entry_not_suspended,
                entry_present & entry_not_suspended & ~valid_entry_price,
                ~exit_present,
                exit_present & ~exit_not_suspended,
                exit_present & exit_not_suspended & ~valid_exit_price,
            ],
            [
                "missing_entry_row",
                "entry_suspended",
                "invalid_entry_price",
                "missing_exit_row",
                "exit_suspended",
                "invalid_exit_price",
            ],
            default="",
        )
        merged.insert(0, "formation_date", row.formation_date)
        merged.insert(1, "entry_date", row.entry_date)
        merged.insert(2, "exit_date", row.exit_date)
        pieces.append(merged)

    returns = pd.concat(pieces, ignore_index=True)
    if returns.duplicated(["formation_date", "security_id"]).any():
        raise ValueError("Forward returns are not unique by formation date + security")
    valid = returns["return_available"]
    if (returns.loc[valid, "forward_return"] < -1.0 - 1e-12).any():
        raise ValueError("A forward return is below -100%")
    return returns


def build_regression_panel(
    formation: pd.DataFrame,
    industry: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = formation.merge(
        industry,
        on=["formation_date", "security_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        returns,
        on=["formation_date", "security_id"],
        how="left",
        validate="one_to_one",
    )
    panel["ep_valid"] = np.isfinite(panel["ep_z"])
    panel["size_valid"] = np.isfinite(panel["size_z"])
    panel["industry_valid"] = panel["industry_lv1"].notna()
    panel["return_valid"] = panel["return_available"].fillna(False) & np.isfinite(
        panel["forward_return"]
    )
    panel["regression_eligible"] = (
        panel["ep_valid"]
        & panel["size_valid"]
        & panel["industry_valid"]
        & panel["return_valid"]
    )
    panel["missing_reason"] = panel["missing_reason"].fillna("missing_entry_row")

    rows = []
    for formation_date, group in panel.groupby("formation_date", sort=True):
        after_ep = group["ep_valid"]
        after_size = after_ep & group["size_valid"]
        after_industry = after_size & group["industry_valid"]
        after_return = after_industry & group["return_valid"]
        eligible_before_return = group.loc[after_industry]
        rows.append(
            {
                "formation_date": formation_date,
                "n_formation_market": len(group),
                "n_valid_ep": int(after_ep.sum()),
                "n_valid_ep_size": int(after_size.sum()),
                "n_valid_ep_size_industry": int(after_industry.sum()),
                "n_valid_return_after_controls": int(after_return.sum()),
                "n_final": int(group["regression_eligible"].sum()),
                "n_missing_entry_row": int(
                    eligible_before_return["missing_reason"].eq("missing_entry_row").sum()
                ),
                "n_entry_suspended": int(
                    eligible_before_return["missing_reason"].eq("entry_suspended").sum()
                ),
                "n_missing_exit_row": int(
                    eligible_before_return["missing_reason"].eq("missing_exit_row").sum()
                ),
                "n_exit_suspended": int(
                    eligible_before_return["missing_reason"].eq("exit_suspended").sum()
                ),
                "n_invalid_endpoint_price": int(
                    eligible_before_return["missing_reason"].isin(
                        ["invalid_entry_price", "invalid_exit_price"]
                    ).sum()
                ),
                "n_entry_at_up_limit": int(
                    eligible_before_return["entry_at_up_limit"].fillna(False).sum()
                ),
                "n_exit_at_down_limit": int(
                    eligible_before_return["exit_at_down_limit"].fillna(False).sum()
                ),
                "n_bse_final": int(
                    group.loc[group["regression_eligible"], "security_id"]
                    .astype("string")
                    .str.endswith(".BJ", na=False)
                    .sum()
                ),
            }
        )
    return panel, pd.DataFrame(rows)


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
                f"{formation_date.date()}: observations {n} below required {required_n}"
            )
        rank = int(np.linalg.matrix_rank(design.to_numpy(float)))
        if rank != p:
            raise ValueError(
                f"{formation_date.date()}: singular design rank={rank}, columns={p}"
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
        iterations = int(result.fit_history.get("iteration", MAX_ITER))
        converged = iterations < MAX_ITER
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


def hac_mean(values: pd.Series, maxlags: int = HAC_LAGS) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(clean) < 2:
        return {"mean": np.nan, "hac_se": np.nan, "hac_t": np.nan, "hac_p": np.nan}
    lags = min(maxlags, len(clean) - 1)
    fit = sm.OLS(clean, np.ones((len(clean), 1))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lags, "use_correction": True},
    )
    return {
        "mean": float(fit.params[0]),
        "hac_se": float(fit.bse[0]),
        "hac_t": float(fit.tvalues[0]),
        "hac_p": float(fit.pvalues[0]),
    }


def summarize_period(group: pd.DataFrame, label: str) -> dict[str, float | str | int]:
    ep_hac = hac_mean(group["ep_beta"])
    size_hac = hac_mean(group["size_beta"])
    return {
        "period": label,
        "months": len(group),
        "ep_mean": ep_hac["mean"],
        "ep_median": float(group["ep_beta"].median()),
        "ep_std": float(group["ep_beta"].std(ddof=1)),
        "ep_hac_se": ep_hac["hac_se"],
        "ep_hac_t": ep_hac["hac_t"],
        "ep_hac_p": ep_hac["hac_p"],
        "ep_positive_ratio": float(group["ep_beta"].gt(0).mean()),
        "ep_abs_monthly_z_ge_2_ratio": float(group["ep_zvalue"].abs().ge(2).mean()),
        "ep_positive_significant_ratio": float(
            (group["ep_beta"].gt(0) & group["ep_zvalue"].ge(2)).mean()
        ),
        "ep_negative_significant_ratio": float(
            (group["ep_beta"].lt(0) & group["ep_zvalue"].le(-2)).mean()
        ),
        "size_mean": size_hac["mean"],
        "size_hac_t": size_hac["hac_t"],
        "size_hac_p": size_hac["hac_p"],
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
    significance = "达到" if summary["ep_hac_p"] < 0.05 else "未达到"
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
        f"RLM 使用 HuberT(t={HUBER_T})、MAD 残差尺度、H1 协方差；全期系数均值使用 Newey-West HAC(maxlags={HAC_LAGS}) 推断。",
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
        f"| EP HAC 标准误 | {pct(summary['ep_hac_se'], 4)} |",
        f"| EP HAC t 值 | {num(summary['ep_hac_t'])} |",
        f"| EP HAC p 值 | {num(summary['ep_hac_p'])} |",
        f"| EP 系数为正的月份 | {pct(summary['ep_positive_ratio'])} |",
        f"| 单月 abs(z) >= 2 的比例 | {pct(summary['ep_abs_monthly_z_ge_2_ratio'])} |",
        f"| 正向且 z >= 2 的比例 | {pct(summary['ep_positive_significant_ratio'])} |",
        f"| 负向且 z <= -2 的比例 | {pct(summary['ep_negative_significant_ratio'])} |",
        f"| Size 月均系数 | {pct(summary['size_mean'], 4)} |",
        f"| Size HAC t 值 | {num(summary['size_hac_t'])} |",
        f"| 平均 RLM 降权比例 | {pct(summary['downweighted_ratio_mean'])} |",
        f"| 平均权重低于 0.5 比例 | {pct(summary['weight_below_half_ratio_mean'])} |",
        f"| 平均伪 R² | {pct(summary['pseudo_r2_mean'])} |",
        "",
        f"EP 月均系数方向为{direction}，HAC 显著性{significance} 5% 标准。EP 已按形成日横截面标准化，",
        f"因此月均系数 {pct(summary['ep_mean'], 4)} 可解释为 EP 提高一个横截面标准差时，下一持有期收益平均变化约 {pct(summary['ep_mean'], 4)}。",
        f"简单乘以 12 得到的年化系数约为 {pct(summary['ep_mean'] * 12, 2)}，它是因子溢价尺度，不是可交易组合年化收益。",
        "",
        "## 4. 分年度结果",
        "",
        "| 年份 | 月数 | EP 月均系数 | HAC t 值 | HAC p 值 | 正系数比例 | abs(z) >= 2 比例 | 平均样本数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in annual.itertuples(index=False):
        lines.append(
            f"| {row.period} | {row.months} | {pct(row.ep_mean, 4)} | "
            f"{num(row.ep_hac_t)} | {num(row.ep_hac_p)} | {pct(row.ep_positive_ratio)} | "
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
        f"- 收敛月份：{int(monthly['converged'].sum())}/{len(monthly)}；迭代次数中位数为 {monthly['iterations'].median():.0f}。",
        f"- 条件数范围：{monthly['condition_number'].min():.2f} 至 {monthly['condition_number'].max():.2f}。",
        f"- 北交所最终样本在代码切换附近未发生由代码连接失败造成的断层，逐月数量见 `rlm_sample_audit.csv`。",
        "- 涨停买入和跌停卖出只做数量标记，未用于 RLM 样本筛选；实际可交易性留待分层回测处理。",
        "- `cumulative_ep_premium` 是月度系数的累计和，不是组合净值。",
        "",
        "![RLM 月度 EP 系数与累计因子溢价](./rlm_ep_premium.png)",
        "",
        "## 7. 输出文件",
        "",
        "- `bse_code_mapping.csv/.parquet`：北交所 242 对新旧代码映射及匹配证据。",
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

    print("Reading market calendar...", flush=True)
    market_dates = read_market_dates(MARKET_PATH)
    calendar = build_monthly_calendar(market_dates, test_start)
    signal_dates = set(pd.to_datetime(calendar["formation_date"]))
    return_dates = set(pd.to_datetime(calendar["entry_date"])) | set(
        pd.to_datetime(calendar["exit_date"])
    )
    selected_market_dates = signal_dates | return_dates | {
        TRANSITION_OLD_DATE,
        TRANSITION_NEW_DATE,
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
    mapping_frame = derive_bse_mapping(market)
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
    forward_returns = build_forward_returns(market, calendar, mapping)
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
        raise RuntimeError(f"RLM did not converge for: {failed}")

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
        "time_series_inference": {"method": "Newey-West HAC", "maxlags": HAC_LAGS},
        "periods": len(calendar),
        "first_formation_date": str(calendar["formation_date"].min().date()),
        "last_formation_date": str(calendar["formation_date"].max().date()),
        "bse_mapping": {
            "pairs": len(mapping_frame),
            "old_last_date": str(TRANSITION_OLD_DATE.date()),
            "new_first_date": str(TRANSITION_NEW_DATE.date()),
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
    print(f"ep_hac_t={row['ep_hac_t']:.6f}", flush=True)
    print(f"ep_hac_p={row['ep_hac_p']:.6f}", flush=True)
    print(f"ep_positive_ratio={row['ep_positive_ratio']:.6f}", flush=True)
    print(f"saved={OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
