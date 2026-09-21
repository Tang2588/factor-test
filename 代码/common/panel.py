# -*- coding: utf-8 -*-
"""形成日面板、行业面板与回归面板。

三类回归（OLS / RLM / WLS）必须使用完全相同的样本构造规则，否则系数差异
里会混入口径差异。因此样本构造全部放在这里，回归脚本只负责估计方法。

规模控制变量统一在「EP 有效且总市值有效」的形成日universe内做横截面
Z 标准化，不使用外部规模因子文件，保证 OLS 与 RLM 的设计矩阵一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .market import apply_security_id, extract_code6


def prepare_formation_panel(
    market: pd.DataFrame,
    ep: pd.DataFrame,
    signal_dates: set[pd.Timestamp],
    mapping: dict[str, str],
) -> pd.DataFrame:
    formation = market.loc[market["date"].isin(signal_dates)].copy()
    formation = apply_security_id(formation, mapping)
    if formation.duplicated(["date", "security_id"]).any():
        raise ValueError("形成日行情面板在代码归并后不唯一")

    factor = ep.copy()
    factor["code6"] = extract_code6(factor["stock_code"])
    factor["ep_z"] = pd.to_numeric(factor["signal"], errors="coerce")
    factor = factor[["date", "code6", "ep_z"]]
    if factor.duplicated(["date", "code6"]).any():
        raise ValueError("EP 因子在 date + code6 上不唯一")

    formation = formation.merge(
        factor,
        on=["date", "code6"],
        how="left",
        validate="one_to_one",
        indicator="factor_join",
    )
    if not formation["factor_join"].eq("both").all():
        missing = int(formation["factor_join"].ne("both").sum())
        raise ValueError(f"形成日行情有 {missing} 行在 EP 输出中找不到对应因子")
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


def prepare_industry(raw: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    industry = raw.copy()
    industry["market_code"] = industry["stock_code"].astype("string").str.strip().str.upper()
    industry = apply_security_id(industry, mapping)
    industry = industry.rename(
        columns={"date": "formation_date", "indus_name_lv1": "industry_lv1"}
    )[["formation_date", "security_id", "industry_lv1"]]
    if industry.duplicated(["formation_date", "security_id"]).any():
        duplicates = industry.loc[
            industry.duplicated(["formation_date", "security_id"], keep=False)
        ]
        raise ValueError(f"行业面板在代码归并后存在 {len(duplicates)} 行重复")
    return industry


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
    # 不在前瞻收益面板中的证券单独标记，不再与「缺建仓行」混为一类
    panel["missing_reason"] = panel["missing_reason"].fillna("not_in_return_panel")

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
                "n_not_in_return_panel": int(
                    eligible_before_return["missing_reason"].eq("not_in_return_panel").sum()
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
