# -*- coding: utf-8 -*-
"""行情快照清洗、B 股剔除与北交所新旧代码映射。

北交所 2025 年 10 月把 43/83/87 开头的旧代码统一改为 920 开头。切换当天
同一只证券的可比价格和总股本都没有变化，因此用「切换前最后交易日收盘价
= 切换后首个交易日昨收」精确配对，价格不唯一时再用总股本消除歧义。
切换日期由 ``parquet_io.scan_market_index`` 从数据推导，不写死在代码里。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .parquet_io import NEW_BSE_CODE_PATTERN, OLD_BSE_CODE_PATTERN

NUMERIC_MARKET_COLUMNS = [
    "close",
    "pre_close",
    "close_adj",
    "pre_close_adj",
    "share_total",
    "me_total",
    "up_limit",
    "down_limit",
]

OPTIONAL_NUMERIC_MARKET_COLUMNS = ["me_float", "share_float"]


def normalize_market_code(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def extract_code6(series: pd.Series) -> pd.Series:
    return normalize_market_code(series).str.extract(r"(\d{6})", expand=False)


def b_share_mask(series: pd.Series) -> pd.Series:
    """上海 B 股（900xxx.BJ）与深圳 B 股（200/201xxx.SZ）。"""
    code = normalize_market_code(series)
    sh_b = code.str.startswith("900", na=False) & code.str.endswith(".BJ", na=False)
    sz_b = code.str.startswith(("200", "201"), na=False) & code.str.endswith(".SZ", na=False)
    return sh_b | sz_b


def prepare_market_snapshot(raw: pd.DataFrame) -> pd.DataFrame:
    """清洗行情快照：统一代码、剔除 B 股、数值化字段、标记停牌与涨跌停。"""
    market = raw.copy()
    market["market_code"] = normalize_market_code(market["stock_code"])
    market = market.loc[~b_share_mask(market["market_code"])].copy()
    market["code6"] = extract_code6(market["market_code"])
    if market["code6"].isna().any():
        raise ValueError("行情快照中存在无法解析的证券代码")

    suspended_numeric = pd.to_numeric(market["suspended"], errors="coerce")
    status = market["status"].astype("string").str.strip()
    market["suspended_flag"] = (
        suspended_numeric.eq(1).fillna(False) | status.eq("停牌").fillna(False)
    )

    numeric_columns = list(NUMERIC_MARKET_COLUMNS)
    numeric_columns += [c for c in OPTIONAL_NUMERIC_MARKET_COLUMNS if c in market.columns]
    for column in numeric_columns:
        market[column] = pd.to_numeric(market[column], errors="coerce")

    market["at_up_limit"] = np.isclose(
        market["close"], market["up_limit"], rtol=0.0, atol=1e-8, equal_nan=False
    )
    market["at_down_limit"] = np.isclose(
        market["close"], market["down_limit"], rtol=0.0, atol=1e-8, equal_nan=False
    )
    if market.duplicated(["date", "market_code"]).any():
        raise ValueError("行情快照存在重复的 date + market_code 记录")
    return market


def derive_bse_mapping(
    market: pd.DataFrame,
    old_date: pd.Timestamp,
    new_date: pd.Timestamp,
    expected_pairs: int | None = None,
) -> pd.DataFrame:
    """构造北交所新旧代码一一映射，并校验价格连续性。"""
    old_date = pd.Timestamp(old_date)
    new_date = pd.Timestamp(new_date)

    old = market.loc[
        market["date"].eq(old_date) & market["market_code"].str.match(OLD_BSE_CODE_PATTERN, na=False),
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
        market["date"].eq(new_date) & market["market_code"].str.match(NEW_BSE_CODE_PATTERN, na=False),
        ["market_code", "pre_close", "pre_close_adj", "share_total"],
    ].rename(
        columns={
            "market_code": "new_code",
            "pre_close": "new_pre_close",
            "pre_close_adj": "new_pre_close_adj",
            "share_total": "new_share_total",
        }
    )
    if old.empty:
        raise ValueError(f"{old_date.date()} 没有北交所旧代码行情")
    if new.empty:
        raise ValueError(f"{new_date.date()} 没有北交所新代码行情")

    preexisting_new_codes = set(
        market.loc[
            market["date"].eq(old_date) & market["market_code"].str.match(NEW_BSE_CODE_PATTERN, na=False),
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
                    f"北交所代码映射存在歧义：{old_code}，"
                    f"价格候选={len(available)}，股本匹配={len(exact_share)}"
                )
            selected = exact_share.iloc[0].copy()
            selected["match_method"] = "exact_price_and_share_total"
        selected_rows.append(selected)

    mapping = pd.DataFrame(selected_rows).reset_index(drop=True)
    newly_converted = set(new["new_code"]) - preexisting_new_codes

    if expected_pairs is not None and len(mapping) != expected_pairs:
        raise ValueError(f"预期 {expected_pairs} 对北交所映射，实际得到 {len(mapping)} 对")
    if mapping["old_code"].nunique() != len(mapping) or mapping["new_code"].nunique() != len(mapping):
        raise ValueError("北交所新旧代码映射不是一一对应")
    if set(mapping["new_code"]) != newly_converted:
        raise ValueError("北交所映射目标与切换日新增的 920 代码不一致")
    if not np.allclose(mapping["old_close"], mapping["new_pre_close"], rtol=0.0, atol=1e-10):
        raise ValueError("北交所代码切换前后原始价格不连续")
    if not np.allclose(
        mapping["old_close_adj"], mapping["new_pre_close_adj"], rtol=0.0, atol=1e-10
    ):
        raise ValueError("北交所代码切换前后复权价格不连续")

    mapping.insert(0, "old_last_date", old_date)
    mapping.insert(1, "new_first_date", new_date)
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
    """把旧代码归并到新代码，使同一证券在切换前后共享一个 ``security_id``。"""
    result = frame.copy()
    mapped = result["market_code"].map(mapping)
    result["security_id"] = mapped.fillna(result["market_code"]).astype("string")
    return result
