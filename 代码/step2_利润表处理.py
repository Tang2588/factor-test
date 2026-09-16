# -*- coding: utf-8 -*-
"""Step 2: retain point-in-time versions of income statements."""
from pathlib import Path
import pandas as pd

BASE = Path(r"D:\实习生学习项目\基础数据")
OUT_DIR = Path(__file__).resolve().parents[1] / "中间结果"
OUT_DIR.mkdir(parents=True, exist_ok=True)
columns = [
    "TICKER_SYMBOL", "ACT_PUBTIME", "UPDATE_TIME", "END_DATE",
    "FISCAL_PERIOD", "MERGED_FLAG", "N_INCOME_ATTR_P",
]
is_ = pd.read_parquet(BASE / "vw_fdmt_is_new.parquet", columns=columns)
is_ = is_.rename(columns={"TICKER_SYMBOL": "code6"})
is_["code6"] = is_["code6"].astype("string").str.extract(r"(\d{6})", expand=False)
mapping = pd.read_parquet(OUT_DIR / "bse_code_mapping.parquet", columns=["old_code", "new_code"])
new_to_old_financial_code6 = dict(
    zip(
        mapping["new_code"].str.extract(r"(\d{6})", expand=False),
        mapping["old_code"].str.extract(r"(\d{6})", expand=False),
    )
)
is_["source_code6"] = is_["code6"]
is_["code6"] = is_["code6"].map(new_to_old_financial_code6).fillna(is_["code6"])
mapped_financial_rows = int(is_["source_code6"].isin(new_to_old_financial_code6).sum())
is_["ACT_PUBTIME"] = pd.to_datetime(is_["ACT_PUBTIME"], errors="coerce")
is_["UPDATE_TIME"] = pd.to_datetime(is_["UPDATE_TIME"], errors="coerce")
is_["END_DATE"] = pd.to_datetime(is_["END_DATE"], errors="coerce")
is_["FISCAL_PERIOD"] = pd.to_numeric(is_["FISCAL_PERIOD"], errors="coerce")
is_["N_INCOME_ATTR_P"] = pd.to_numeric(is_["N_INCOME_ATTR_P"], errors="coerce")
is_ = is_.dropna(
    subset=["code6", "ACT_PUBTIME", "END_DATE", "FISCAL_PERIOD", "N_INCOME_ATTR_P"]
).copy()
is_ = is_[is_["MERGED_FLAG"].astype("string").str.strip().eq("1")].copy()
is_["report_month"] = is_["END_DATE"].dt.month
is_ = is_[is_["report_month"].eq(is_["FISCAL_PERIOD"])].copy()

# UPDATE_TIME resolves source rows sharing one ACT_PUBTIME. VERSION_TIME is a
# conservative availability time, so a later source correction is not used
# before it entered the dataset.
is_["VERSION_TIME"] = is_[["ACT_PUBTIME", "UPDATE_TIME"]].max(axis=1)
version_cols = [
    "code6", "END_DATE", "FISCAL_PERIOD", "ACT_PUBTIME", "UPDATE_TIME",
    "N_INCOME_ATTR_P",
]
reports = is_.drop_duplicates(version_cols, keep="first")
reports = reports[
    [
        "code6", "END_DATE", "FISCAL_PERIOD", "ACT_PUBTIME", "UPDATE_TIME",
        "VERSION_TIME", "N_INCOME_ATTR_P",
    ]
].sort_values(["code6", "VERSION_TIME", "ACT_PUBTIME", "END_DATE"])

version_key = ["code6", "END_DATE", "ACT_PUBTIME", "UPDATE_TIME"]
if reports.duplicated(version_key).any():
    raise ValueError("利润表清洗后仍存在重复的财报版本")

effective_key = ["code6", "END_DATE", "VERSION_TIME"]
conflicts = (
    reports.groupby(effective_key, sort=False)["N_INCOME_ATTR_P"]
    .nunique(dropna=False).gt(1)
)
if conflicts.any():
    raise ValueError("同一有效版本时间仍存在不同归母净利润")

reports.to_parquet(OUT_DIR / "reports.parquet", index=False)
print(f"reports rows: {len(reports):,}")
print(f"BSE financial rows remapped before TTM: {mapped_financial_rows:,}")
print(f"codes: {reports['code6'].nunique():,}")
print(f"report periods: {reports[['code6', 'END_DATE']].drop_duplicates().shape[0]:,}")
print(
    "periods with multiple versions: "
    f"{int((reports.groupby(['code6', 'END_DATE']).size() > 1).sum()):,}"
)
print(f"saved: {OUT_DIR / 'reports.parquet'}")
