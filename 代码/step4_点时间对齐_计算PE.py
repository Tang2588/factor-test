# -*- coding: utf-8 -*-
"""Step 4: point-in-time alignment and raw EP/PE calculation."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

for parent in Path(__file__).resolve().parents:
    if (parent / "pure_factor_streaming.py").exists():
        sys.path.insert(0, str(parent))
        break
from pure_factor_streaming import stream_align_market_events

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "中间结果"
TTM_TRACE_COLUMNS = [
    "END_DATE",
    "ACT_PUBTIME",
    "REPORT_VERSION_TIME",
    "TTM_ni",
    "TTM_current_ni",
    "TTM_previous_annual_date",
    "TTM_previous_annual_ni",
    "TTM_previous_same_period_date",
    "TTM_previous_same_period_ni",
    "EVENT_ACT_PUBTIME",
]


def make_batch(merged: pd.DataFrame):
    merged["me_total"] = pd.to_numeric(merged["me_total"], errors="coerce")
    merged["TTM_ni"] = pd.to_numeric(merged["TTM_ni"], errors="coerce")
    ep = pd.Series(np.nan, index=merged.index, dtype="float64")
    valid_market_cap = (
        merged["me_total"].gt(0)
        & np.isfinite(merged["me_total"])
        & merged["TTM_ni"].notna()
        & np.isfinite(merged["TTM_ni"])
    )
    ep.loc[valid_market_cap] = merged.loc[valid_market_cap, "TTM_ni"] / merged.loc[valid_market_cap, "me_total"]
    ep.loc[merged["suspended"].eq(1)] = np.nan

    # PE is reported as the reciprocal of raw EP.  Zero earnings yield has no
    # defined reciprocal; negative EP is retained as negative PE for symmetry.
    pe = pd.Series(np.nan, index=merged.index, dtype="float64")
    nonzero_ep = ep.notna() & ep.ne(0)
    pe.loc[nonzero_ep] = 1.0 / ep.loc[nonzero_ep]

    raw = pd.DataFrame(
        {
            "date": merged["date"],
            "stock_code": merged["code6"],
            "signal": ep,
            "pe": pe,
        }
    )
    raw = raw.set_index(["date", "stock_code"])
    mask = merged[
        [
            "date",
            "code6",
            "suspended",
            "suspended_unknown",
            "is_cixin",
            "listing_date_known",
            "first_seen_date",
        ]
    ].rename(columns={"code6": "stock_code"})
    mask = mask.set_index(["date", "stock_code"])
    mask["st_data_available"] = 0
    mask["st_filter_applied"] = 0
    mask["st_filter_note"] = "ST/PT filter not applied; historical ST data unavailable"
    mask["listing_date_is_official"] = 0
    mask["listing_filter_applied"] = 0
    mask["trading_status_point_in_time"] = 1

    audit_columns = [
        "date",
        "code6",
        "market_code",
        "security_id",
        "financial_code6",
        "bse_code_mapped",
        "exchange",
        "market_scope",
        "me_total",
        "status",
        "suspended",
        "suspended_unknown",
        "is_cixin",
        "listing_date_known",
        "first_seen_date",
        "signal_cutoff",
        "VERSION_TIME",
        *TTM_TRACE_COLUMNS,
    ]
    audit = merged[audit_columns].rename(
        columns={
            "code6": "stock_code",
            "VERSION_TIME": "ttm_event_time",
            "END_DATE": "report_end_date",
            "ACT_PUBTIME": "report_act_pubtime",
            "REPORT_VERSION_TIME": "report_version_time",
        }
    )
    audit["raw_ep"] = ep.to_numpy()
    audit["raw_pe"] = pe.to_numpy()
    audit["st_flag"] = pd.Series(pd.NA, index=audit.index, dtype="Int8")
    audit["st_data_available"] = 0
    audit["st_filter_applied"] = 0
    audit["listing_date_is_official"] = 0
    audit["listing_filter_applied"] = 0
    audit["trading_status_point_in_time"] = 1
    audit = audit.set_index(["date", "stock_code"])
    valid = raw["signal"].notna()
    values = raw.loc[valid, "signal"]
    valid_pe = raw["pe"].notna()
    return (
        raw,
        mask,
        {
            "valid_ep": int(valid.sum()),
            "negative_ep": int(values.lt(0).sum()),
            "zero_ep": int(values.eq(0).sum()),
            "valid_pe": int(valid_pe.sum()),
            "negative_pe": int(raw.loc[valid_pe, "pe"].lt(0).sum()),
        },
        {"factor_audit": audit},
    )


totals = stream_align_market_events(
    DATA / "_mkt_clean.parquet",
    DATA / "reports_ttm.parquet",
    TTM_TRACE_COLUMNS,
    make_batch,
    DATA / "ep_raw.parquet",
    DATA / "_mask.parquet",
    raw_columns=["signal"],
    derived_raw_paths={"pe": DATA / "pe_raw.parquet"},
    additional_frame_paths={"factor_audit": DATA / "factor_audit.parquet"},
    market_event_code_column="financial_code6",
)
print(f"market rows: {totals['market_rows']:,}")
print(f"lookahead violations: {totals['lookahead']}")
print(f"valid EP rows: {totals.get('valid_ep', 0):,} ({totals.get('valid_ep', 0) / totals['market_rows'] * 100:.2f}%)")
print(f"negative EP rows: {totals.get('negative_ep', 0):,}")
print(f"zero EP rows: {totals.get('zero_ep', 0):,}")
print(f"valid PE rows: {totals.get('valid_pe', 0):,} ({totals.get('valid_pe', 0) / totals['market_rows'] * 100:.2f}%)")
print(f"negative PE rows: {totals.get('negative_pe', 0):,}")
print(f"saved: {DATA / 'ep_raw.parquet'}")
print(f"saved: {DATA / 'pe_raw.parquet'}")
print(f"saved: {DATA / '_mask.parquet'}")
print(f"saved: {DATA / 'factor_audit.parquet'}")
