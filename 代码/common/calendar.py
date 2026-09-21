# -*- coding: utf-8 -*-
"""月度形成／建仓／退出日历。

统一口径：每月最后一个交易日形成因子，次月第一个交易日按复权收盘价建仓，
再下一个月第一个交易日按复权收盘价退出。形成日与建仓日之间不补收益，
因此一个持有期恰好是一个自然月。
"""
from __future__ import annotations

import pandas as pd


def build_monthly_calendar(
    market_dates: pd.DatetimeIndex,
    test_start: pd.Timestamp,
) -> pd.DataFrame:
    date_frame = pd.DataFrame({"date": pd.DatetimeIndex(market_dates)})
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
        raise ValueError("没有完整的月度前瞻收益区间")
    return calendar
