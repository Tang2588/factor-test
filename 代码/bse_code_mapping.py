# -*- coding: utf-8 -*-
"""Derive the audited 2025 BSE old/new security-code mapping."""
from __future__ import annotations

import numpy as np
import pandas as pd


OLD_LAST_DATE = pd.Timestamp("2025-09-30")
NEW_FIRST_DATE = pd.Timestamp("2025-10-09")
EXPECTED_CONVERSIONS = 242


def derive_bse_code_mapping(market: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "market_code",
        "close",
        "pre_close",
        "close_adj",
        "pre_close_adj",
        "share_total",
    }
    missing = required - set(market.columns)
    if missing:
        raise KeyError(f"BSE mapping input is missing columns: {sorted(missing)}")

    data = market.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["market_code"] = data["market_code"].astype("string").str.strip().str.upper()
    for column in ["close", "pre_close", "close_adj", "pre_close_adj", "share_total"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    old = data.loc[
        data["date"].eq(OLD_LAST_DATE)
        & data["market_code"].str.match(r"^(43|83|87)\d{4}\.BJ$", na=False),
        ["market_code", "close", "close_adj", "share_total"],
    ].rename(
        columns={
            "market_code": "old_code",
            "close": "old_close",
            "close_adj": "old_close_adj",
            "share_total": "old_share_total",
        }
    )
    new = data.loc[
        data["date"].eq(NEW_FIRST_DATE)
        & data["market_code"].str.match(r"^920\d{3}\.BJ$", na=False),
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
        data.loc[
            data["date"].eq(OLD_LAST_DATE)
            & data["market_code"].str.match(r"^920\d{3}\.BJ$", na=False),
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
    if len(mapping) != EXPECTED_CONVERSIONS:
        raise ValueError(f"Expected {EXPECTED_CONVERSIONS} BSE mappings, got {len(mapping)}")
    if mapping["old_code"].nunique() != len(mapping) or mapping["new_code"].nunique() != len(mapping):
        raise ValueError("BSE old/new code mapping is not one-to-one")
    if set(mapping["new_code"]) != newly_converted:
        raise ValueError("BSE mapping targets do not equal the newly converted 920 codes")
    if not np.allclose(mapping["old_close"], mapping["new_pre_close"], rtol=0.0, atol=1e-10):
        raise ValueError("BSE raw prices are discontinuous across the code conversion")
    if not np.allclose(
        mapping["old_close_adj"], mapping["new_pre_close_adj"], rtol=0.0, atol=1e-10
    ):
        raise ValueError("BSE adjusted prices are discontinuous across the code conversion")

    mapping.insert(0, "old_last_date", OLD_LAST_DATE)
    mapping.insert(1, "new_first_date", NEW_FIRST_DATE)
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
