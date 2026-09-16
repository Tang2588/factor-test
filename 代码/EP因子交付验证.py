# -*- coding: utf-8 -*-
"""Validate the EP and PE files inside the delivery folder."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "中间结果"


def read_factor(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    assert frame.index.names == ["date", "stock_code"], (path, frame.index.names)
    assert list(frame.columns) == ["signal"], (path, frame.columns.tolist())
    assert not frame.index.duplicated().any(), f"{path}: duplicate index"
    assert np.isfinite(frame["signal"].dropna()).all(), f"{path}: non-finite value"
    return frame


raw_ep = read_factor(DATA / "ep_raw.parquet")
raw_pe = read_factor(DATA / "pe_raw.parquet")
mad_ep = read_factor(DATA / "ep_mad.parquet")
mad_pe = read_factor(DATA / "pe_mad.parquet")
final_ep = read_factor(ROOT / "因子结果" / "ep.parquet")
final_pe = read_factor(ROOT / "因子结果" / "pe.parquet")
mask = pd.read_parquet(
    DATA / "_mask.parquet",
    columns=[
        "suspended",
        "st_data_available",
        "st_filter_applied",
        "listing_date_is_official",
        "listing_filter_applied",
        "trading_status_point_in_time",
    ],
)
mkt_rows = pq.ParquetFile(DATA / "_mkt_clean.parquet").metadata.num_rows
b_shares = pd.read_parquet(DATA / "b_share_codes.parquet")

for name, frame in [
    ("raw_pe", raw_pe),
    ("mad_ep", mad_ep),
    ("mad_pe", mad_pe),
    ("final_ep", final_ep),
    ("final_pe", final_pe),
]:
    assert frame.index.equals(raw_ep.index), f"{name}: index mismatch"

assert len(raw_ep) == len(raw_pe) == len(mad_ep) == len(mad_pe) == len(final_ep) == len(final_pe)
assert len(raw_ep) == len(mask) == mkt_rows
assert raw_ep.index.equals(raw_pe.index)

ep = raw_ep["signal"]
pe = raw_pe["signal"]
reciprocal = ep.notna() & ep.ne(0)
assert np.allclose(pe.loc[reciprocal], 1.0 / ep.loc[reciprocal], rtol=1e-12, atol=1e-12)
assert pe.loc[ep.isna() | ep.eq(0)].isna().all()

assert mask["st_data_available"].eq(0).all()
assert mask["st_filter_applied"].eq(0).all()
assert mask["listing_date_is_official"].eq(0).all()
assert mask["listing_filter_applied"].eq(0).all()
assert mask["trading_status_point_in_time"].eq(1).all()
is_suspended = mask["suspended"].eq(1)
assert raw_ep.loc[is_suspended, "signal"].isna().all()
assert raw_pe.loc[is_suspended, "signal"].isna().all()
codes = raw_ep.index.get_level_values("stock_code").astype("string")
assert not codes.str.startswith(("900", "200", "201"), na=False).any()
assert len(b_shares) == b_shares["stock_code"].nunique() == 89
assert int(b_shares["b_share_type"].eq("上海 B 股").sum()) == 50
assert int(b_shares["b_share_type"].eq("深圳 B 股").sum()) == 39
assert pq.ParquetFile(DATA / "factor_audit.parquet").metadata.num_rows == len(raw_ep)

print(f"rows: {len(final_ep):,}")
print(f"raw EP valid: {int(ep.notna().sum()):,}")
print(f"raw PE valid: {int(pe.notna().sum()):,}")
print(f"final EP valid: {int(final_ep['signal'].notna().sum()):,}")
print(f"final PE valid: {int(final_pe['signal'].notna().sum()):,}")
print(f"B shares excluded: {len(b_shares):,} (Shanghai 50, Shenzhen 39)")
print(f"suspended rows: {int(mask['suspended'].sum()):,}")
print("validation: PASS")
