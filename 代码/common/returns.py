# -*- coding: utf-8 -*-
"""前瞻收益（端点口径）。

本模块只保留一种前瞻收益口径，即「建仓日复权收盘价 → 退出日复权收盘价」，
理由是它与实际调仓规则自洽：只在两个端点上交易，期间是否停牌不影响能否成交。

端点规则：

- 建仓日或退出日停牌 → 收益置为缺失；
- 建仓日或退出日复权收盘价缺失或非正 → 收益置为缺失；
- 建仓日或退出日没有行情记录 → 收益置为缺失。

每条缺失记录都写入 ``missing_reason``，便于逐月审计样本流失。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .market import apply_security_id


def build_endpoint_forward_returns(
    market: pd.DataFrame,
    calendar: pd.DataFrame,
    mapping: dict[str, str],
) -> pd.DataFrame:
    canonical = apply_security_id(market, mapping)
    if canonical.duplicated(["date", "security_id"]).any():
        raise ValueError("行情面板在代码归并后仍然不唯一")

    pieces = []
    for row in calendar.itertuples(index=False):
        entry = canonical.loc[
            canonical["date"].eq(row.entry_date),
            ["security_id", "market_code", "close_adj", "suspended_flag", "at_up_limit"],
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
            ["security_id", "market_code", "close_adj", "suspended_flag", "at_down_limit"],
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
        # 未命中任何分支时不再兜底成「缺建仓行」，而是显式标记 unknown，
        # 避免审计表把其他异常误报成缺行。
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
            default="unknown",
        )
        merged.insert(0, "formation_date", row.formation_date)
        merged.insert(1, "entry_date", row.entry_date)
        merged.insert(2, "exit_date", row.exit_date)
        pieces.append(merged)

    returns = pd.concat(pieces, ignore_index=True)
    if returns.duplicated(["formation_date", "security_id"]).any():
        raise ValueError("前瞻收益在 formation_date + security_id 上不唯一")
    valid = returns["return_available"]
    if (returns.loc[valid, "forward_return"] < -1.0 - 1e-12).any():
        raise ValueError("存在低于 -100% 的前瞻收益")
    return returns
