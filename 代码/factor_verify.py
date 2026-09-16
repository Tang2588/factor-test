# -*- coding: utf-8 -*-
"""Independent consistency checks for the EP/PE delivery."""
import json
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


def verify_standardization(
    name: str,
    raw: pd.DataFrame,
    mad: pd.DataFrame,
    z: pd.DataFrame,
    final: pd.DataFrame,
    stats: pd.DataFrame,
) -> dict[str, int]:
    assert raw.index.equals(mad.index)
    assert raw.index.equals(z.index)
    assert raw.index.equals(final.index)
    assert z["signal"].equals(final["signal"])

    raw_valid = raw["signal"].notna()
    mad_valid = mad["signal"].notna()
    z_valid = z["signal"].notna()
    assert raw_valid.sum() == mad_valid.sum(), f"{name}: MAD changed validity"
    assert (z_valid <= raw_valid).all(), f"{name}: Z has new valid values"

    joined = raw.rename(columns={"signal": "raw"}).join(
        mad.rename(columns={"signal": "mad"})
    )
    changed = joined["raw"].notna() & joined["raw"].ne(joined["mad"])
    assert int(changed.sum()) == int(stats["n_clipped"].sum()), f"{name}: clip count"

    sample = joined.loc[joined["raw"].notna(), ["raw", "mad"]].reset_index()
    limits = stats[["date", "lower", "upper"]]
    sample = sample.merge(limits, on="date", how="left", validate="many_to_one")
    bounded = sample["lower"].ne(-np.inf) & sample["upper"].ne(np.inf)
    assert (sample.loc[bounded, "mad"] >= sample.loc[bounded, "lower"]).all()
    assert (sample.loc[bounded, "mad"] <= sample.loc[bounded, "upper"]).all()

    daily = z[z["signal"].notna()].reset_index().groupby("date")["signal"]
    daily_mean = daily.mean()
    daily_std = daily.std(ddof=0)
    nonconstant = daily_std.gt(0)
    if nonconstant.any():
        assert np.nanmax(np.abs(daily_mean[nonconstant])) < 1e-10
        assert np.nanmax(np.abs(daily_std[nonconstant] - 1)) < 1e-10

    return {
        "raw_valid": int(raw_valid.sum()),
        "mad_clipped": int(changed.sum()),
        "z_valid": int(z_valid.sum()),
    }


raw_ep = read_factor(DATA / "ep_raw.parquet")
mad_ep = read_factor(DATA / "ep_mad.parquet")
z_ep = read_factor(DATA / "ep.parquet")
final_ep = read_factor(ROOT / "因子结果" / "ep.parquet")
stats_ep = pd.read_parquet(DATA / "ep_processing_stats.parquet")

raw_pe = read_factor(DATA / "pe_raw.parquet")
mad_pe = read_factor(DATA / "pe_mad.parquet")
z_pe = read_factor(DATA / "pe.parquet")
final_pe = read_factor(ROOT / "因子结果" / "pe.parquet")
stats_pe = pd.read_parquet(DATA / "pe_processing_stats.parquet")

ep_counts = verify_standardization("EP", raw_ep, mad_ep, z_ep, final_ep, stats_ep)
pe_counts = verify_standardization("PE", raw_pe, mad_pe, z_pe, final_pe, stats_pe)

mask = pd.read_parquet(
    DATA / "_mask.parquet",
    columns=[
        "suspended",
        "is_cixin",
        "st_data_available",
        "st_filter_applied",
        "listing_date_known",
        "listing_date_is_official",
        "listing_filter_applied",
        "trading_status_point_in_time",
    ],
)
mkt_rows = pq.ParquetFile(DATA / "_mkt_clean.parquet").metadata.num_rows
b_shares = pd.read_parquet(DATA / "b_share_codes.parquet")
bse_mapping = pd.read_parquet(DATA / "bse_code_mapping.parquet")
reports = pd.read_parquet(DATA / "reports.parquet")
effective = pd.read_parquet(DATA / "reports_effective.parquet")
decisions = pd.read_parquet(DATA / "report_version_decisions.parquet")
ttm = pd.read_parquet(DATA / "reports_ttm.parquet")
coverage = pd.read_parquet(DATA / "factor_daily_coverage.parquet")
with (DATA / "factor_test_start.json").open(encoding="utf-8") as file:
    test_config = json.load(file)
assert len(raw_ep) == len(raw_pe) == mkt_rows == len(mask)
assert raw_ep.index.equals(raw_pe.index)
assert mask.index.equals(raw_ep.index)
assert reports.duplicated(["code6", "END_DATE", "ACT_PUBTIME", "UPDATE_TIME"]).sum() == 0
assert reports.duplicated(["code6", "END_DATE", "VERSION_TIME"]).sum() == 0
assert ttm.duplicated(["code6", "VERSION_TIME"]).sum() == 0

effective = effective.sort_values(["code6", "END_DATE", "VERSION_TIME"])
prior_max_act = effective.groupby(["code6", "END_DATE"], sort=False)[
    "ACT_PUBTIME"
].transform(lambda values: values.cummax().shift())
assert not (effective["ACT_PUBTIME"] < prior_max_act).any()
ignored_old = decisions["decision"].eq("older_announcement_ignored")
assert ignored_old.any()
assert decisions.loc[ignored_old, "accepted"].eq(0).all()

ep = raw_ep["signal"]
pe = raw_pe["signal"]
reciprocal = ep.notna() & ep.ne(0)
assert np.allclose(pe.loc[reciprocal], 1.0 / ep.loc[reciprocal], rtol=1e-12, atol=1e-12)
assert pe.loc[ep.isna() | ep.eq(0)].isna().all()

suspended = mask["suspended"].eq(1)
assert raw_ep.loc[suspended, "signal"].isna().all()
assert raw_pe.loc[suspended, "signal"].isna().all()
assert mask["st_data_available"].eq(0).all()
assert mask["st_filter_applied"].eq(0).all()
assert mask["listing_date_is_official"].eq(0).all()
assert mask["listing_filter_applied"].eq(0).all()
assert mask["trading_status_point_in_time"].eq(1).all()
codes = raw_ep.index.get_level_values("stock_code").astype("string")
assert not codes.str.startswith(("900", "200", "201"), na=False).any()
assert len(b_shares) == b_shares["stock_code"].nunique() == 89
assert int(b_shares["b_share_type"].eq("上海 B 股").sum()) == 50
assert int(b_shares["b_share_type"].eq("深圳 B 股").sum()) == 39
assert len(bse_mapping) == 242
assert bse_mapping["old_code"].nunique() == len(bse_mapping)
assert bse_mapping["new_code"].nunique() == len(bse_mapping)
assert np.allclose(
    bse_mapping["old_close"], bse_mapping["new_pre_close"], rtol=0, atol=1e-10
)
assert np.allclose(
    bse_mapping["old_close_adj"],
    bse_mapping["new_pre_close_adj"],
    rtol=0,
    atol=1e-10,
)
mapped_new_code6 = set(bse_mapping["new_code"].str.extract(r"(\d{6})", expand=False))
assert not reports["code6"].isin(mapped_new_code6).any()

audit_path = DATA / "factor_audit.parquet"
audit_parquet = pq.ParquetFile(audit_path)
assert audit_parquet.metadata.num_rows == len(raw_ep)
required_audit_columns = {
    "date", "stock_code", "market_code", "security_id", "financial_code6",
    "bse_code_mapped", "market_scope", "me_total",
    "status", "suspended", "first_seen_date", "report_end_date",
    "report_act_pubtime", "report_version_time", "ttm_event_time", "TTM_ni",
    "raw_ep", "raw_pe", "mad_ep", "mad_pe", "z_ep", "z_pe",
    "st_data_available", "listing_date_is_official",
    "trading_status_point_in_time",
}
assert required_audit_columns.issubset(audit_parquet.schema_arrow.names)
old_to_new = dict(zip(bse_mapping["old_code"], bse_mapping["new_code"]))
new_to_old_financial = dict(
    zip(
        bse_mapping["new_code"],
        bse_mapping["old_code"].str.extract(r"(\d{6})", expand=False),
    )
)
latest_bse_rows = 0
latest_bse_valid_ep = 0
for batch in audit_parquet.iter_batches(
    columns=[
        "date",
        "stock_code",
        "market_code",
        "security_id",
        "financial_code6",
        "bse_code_mapped",
        "exchange",
        "me_total",
        "suspended",
        "report_act_pubtime",
        "report_version_time",
        "ttm_event_time",
        "TTM_ni",
        "raw_ep",
        "raw_pe",
    ],
    batch_size=250_000,
):
    audit_dates = batch.to_pandas(ignore_metadata=True)
    date = pd.to_datetime(audit_dates["date"])
    report_act = pd.to_datetime(audit_dates["report_act_pubtime"])
    report_version = pd.to_datetime(audit_dates["report_version_time"])
    event_time = pd.to_datetime(audit_dates["ttm_event_time"])
    cutoff = date + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    assert not (report_act.notna() & report_act.gt(report_version)).any()
    assert not (report_version.notna() & report_version.gt(event_time)).any()
    assert not (event_time.notna() & event_time.gt(cutoff)).any()

    market_cap = pd.to_numeric(audit_dates["me_total"], errors="coerce")
    ttm_ni = pd.to_numeric(audit_dates["TTM_ni"], errors="coerce")
    raw_ep_audit = pd.to_numeric(audit_dates["raw_ep"], errors="coerce")
    raw_pe_audit = pd.to_numeric(audit_dates["raw_pe"], errors="coerce")
    valid_formula = (
        market_cap.gt(0)
        & np.isfinite(market_cap)
        & ttm_ni.notna()
        & np.isfinite(ttm_ni)
        & audit_dates["suspended"].ne(1)
    )
    expected_ep = ttm_ni.loc[valid_formula] / market_cap.loc[valid_formula]
    assert np.allclose(raw_ep_audit.loc[valid_formula], expected_ep, rtol=1e-12, atol=1e-12)
    assert raw_ep_audit.loc[~valid_formula].isna().all()
    reciprocal_audit = raw_ep_audit.notna() & raw_ep_audit.ne(0)
    assert np.allclose(
        raw_pe_audit.loc[reciprocal_audit],
        1.0 / raw_ep_audit.loc[reciprocal_audit],
        rtol=1e-12,
        atol=1e-12,
    )
    assert raw_pe_audit.loc[~reciprocal_audit].isna().all()

    market_code = audit_dates["market_code"].astype("string")
    expected_security_id = market_code.map(old_to_new).fillna(market_code)
    expected_financial_code = market_code.map(new_to_old_financial).fillna(
        audit_dates["stock_code"].astype("string")
    )
    expected_mapped_flag = market_code.isin(new_to_old_financial).astype("int8")
    assert audit_dates["security_id"].astype("string").equals(expected_security_id.astype("string"))
    assert audit_dates["financial_code6"].astype("string").equals(
        expected_financial_code.astype("string")
    )
    assert np.array_equal(
        audit_dates["bse_code_mapped"].to_numpy(dtype="int8"),
        expected_mapped_flag.to_numpy(dtype="int8"),
    )
    latest_bse = date.eq(pd.Timestamp("2025-12-31")) & audit_dates["exchange"].eq("BJ")
    latest_bse_rows += int(latest_bse.sum())
    latest_bse_valid_ep += int((latest_bse & raw_ep_audit.notna()).sum())

assert latest_bse_rows == 288
assert latest_bse_valid_ep >= 270

assert len(coverage) == raw_ep.index.get_level_values("date").nunique()
assert coverage["ep_coverage_all"].between(0, 1).all()
assert coverage["ep_coverage_non_suspended"].between(0, 1).all()
test_start = pd.Timestamp(test_config["formal_test_start_date"])
confirmed = coverage.loc[coverage["stability_confirmed"], "date"]
assert not confirmed.empty and pd.Timestamp(confirmed.iloc[0]) == test_start
assert coverage["formal_test_period"].eq(
    pd.to_datetime(coverage["date"]).ge(test_start)
).all()

cixin_rows_with_ep = int(
    (mask["is_cixin"].eq(1) & raw_ep["signal"].notna()).sum()
)

print("formats: PASS")
print("rows:", len(z_ep))
print("raw_ep_valid:", ep_counts["raw_valid"])
print("raw_pe_valid:", pe_counts["raw_valid"])
print("ep_mad_clipped:", ep_counts["mad_clipped"])
print("pe_mad_clipped:", pe_counts["mad_clipped"])
print("ep_z_valid:", ep_counts["z_valid"])
print("pe_z_valid:", pe_counts["z_valid"])
print("negative_raw_ep:", int(ep.dropna().lt(0).sum()))
print("negative_raw_pe:", int(pe.dropna().lt(0).sum()))
print("suspended_rows:", int(suspended.sum()))
print("new_listing_rows_with_valid_ep:", cixin_rows_with_ep)
print("b_shares_excluded:", len(b_shares))
print("bse_code_mappings:", len(bse_mapping))
print("latest_bse_valid_ep:", f"{latest_bse_valid_ep}/{latest_bse_rows}")
print("older_announcement_rows_ignored:", int(ignored_old.sum()))
print("reports_versions:", len(reports))
print("ttm_events:", len(ttm))
print("formal_test_start_date:", test_start.date())
print("validation: PASS")
