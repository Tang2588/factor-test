# -*- coding: utf-8 -*-
"""Step 1: build the Shanghai/Shenzhen A-share plus BSE market panel.

ST/PT and new-listing constraints are not applied. Shanghai B shares encoded
as 900xxx.BJ and Shenzhen B shares encoded as 200/201xxx.SZ are excluded.
Same-day trading status and listing-date availability are retained for audit.
"""
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from bse_code_mapping import NEW_FIRST_DATE, OLD_LAST_DATE, derive_bse_code_mapping


ROOT = Path(__file__).resolve().parents[1]
BASE = Path(r"D:\实习生学习项目\基础数据")
OUT_DIR = ROOT / "中间结果"
OUT_DIR.mkdir(parents=True, exist_ok=True)

source = BASE / "chn_equ_mkt_quotation.parquet"
out_path = OUT_DIR / "_mkt_clean.parquet"
columns = ["date", "stock_code", "me_total", "suspended", "status"]
expected_shanghai_b_share_count = 50
expected_shenzhen_b_share_count = 39


def normalized_market_code(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def shanghai_b_share_mask(series: pd.Series) -> pd.Series:
    code = normalized_market_code(series)
    return code.str.startswith("900", na=False) & code.str.endswith(".BJ", na=False)


def shenzhen_b_share_mask(series: pd.Series) -> pd.Series:
    code = normalized_market_code(series)
    return (
        code.str.startswith(("200", "201"), na=False)
        & code.str.endswith(".SZ", na=False)
    )


def b_share_mask(series: pd.Series) -> pd.Series:
    return shanghai_b_share_mask(series) | shenzhen_b_share_mask(series)

pf = pq.ParquetFile(source)
first_seen_by_market_code = {}
data_start = None
source_rows = 0
kept_rows = 0
excluded_rows = 0
b_share_codes = set()
transition_rows = []

mapping_scan_columns = [
    "date",
    "stock_code",
    "close",
    "pre_close",
    "close_adj",
    "pre_close_adj",
    "share_total",
]
for batch in pf.iter_batches(columns=mapping_scan_columns, batch_size=25_000):
    chunk = batch.to_pandas()
    source_rows += len(chunk)
    chunk["date"] = pd.to_datetime(chunk["date"])
    chunk["market_code"] = normalized_market_code(chunk["stock_code"])
    transition = chunk.loc[chunk["date"].isin([OLD_LAST_DATE, NEW_FIRST_DATE])].copy()
    if not transition.empty:
        transition_rows.append(transition)
    is_b_share = b_share_mask(chunk["stock_code"])
    b_share_codes.update(
        normalized_market_code(chunk.loc[is_b_share, "stock_code"])
        .dropna()
        .unique()
        .tolist()
    )
    excluded_rows += int(is_b_share.sum())
    chunk = chunk.loc[~is_b_share].copy()
    if chunk.empty:
        continue
    code6 = chunk["stock_code"].astype("string").str.extract(r"(\d{6})", expand=False)
    if code6.isna().any():
        raise ValueError("行情表存在无法解析的股票代码")
    kept_rows += len(chunk)
    batch_min = chunk["date"].min()
    data_start = batch_min if data_start is None else min(data_start, batch_min)
    mins = chunk.groupby("market_code", sort=False)["date"].min()
    for code, dt in mins.items():
        old = first_seen_by_market_code.get(code)
        if old is None or dt < old:
            first_seen_by_market_code[code] = dt

if not transition_rows:
    raise ValueError("行情表缺少北交所代码切换日期，无法生成新旧代码映射")
bse_mapping = derive_bse_code_mapping(pd.concat(transition_rows, ignore_index=True))
bse_mapping.to_parquet(OUT_DIR / "bse_code_mapping.parquet", index=False)
bse_mapping.to_csv(OUT_DIR / "bse_code_mapping.csv", index=False, encoding="utf-8-sig")
old_to_new_market_code = dict(zip(bse_mapping["old_code"], bse_mapping["new_code"]))
new_to_old_financial_code6 = dict(
    zip(
        bse_mapping["new_code"],
        bse_mapping["old_code"].str.extract(r"(\d{6})", expand=False),
    )
)
first_seen = {}
for market_code, dt in first_seen_by_market_code.items():
    security_id = old_to_new_market_code.get(market_code, market_code)
    old = first_seen.get(security_id)
    if old is None or dt < old:
        first_seen[security_id] = dt

shanghai_b_share_codes = {code for code in b_share_codes if code.startswith("900")}
shenzhen_b_share_codes = {
    code for code in b_share_codes if code.startswith(("200", "201"))
}
if len(shanghai_b_share_codes) != expected_shanghai_b_share_count:
    raise ValueError(
        f"行情表识别出的 900xxx.BJ 沪市 B 股数量为 {len(shanghai_b_share_codes)}，"
        f"预期为 {expected_shanghai_b_share_count}"
    )
if len(shenzhen_b_share_codes) != expected_shenzhen_b_share_count:
    raise ValueError(
        f"行情表识别出的 200/201xxx.SZ 深市 B 股数量为 {len(shenzhen_b_share_codes)}，"
        f"预期为 {expected_shenzhen_b_share_count}"
    )

b_share_audit = pd.DataFrame({"stock_code": sorted(b_share_codes)})
b_share_audit["code6"] = b_share_audit["stock_code"].str.extract(
    r"(\d{6})", expand=False
)
b_share_audit["b_share_type"] = "深圳 B 股"
b_share_audit.loc[
    b_share_audit["stock_code"].str.startswith("900"), "b_share_type"
] = "上海 B 股"
b_share_audit.to_parquet(OUT_DIR / "b_share_codes.parquet", index=False)

writer = None
suspended_rows = 0
cixin_rows = 0
cixin_codes = set()
market_rows = 0

try:
    for batch in pf.iter_batches(columns=columns, batch_size=25_000):
        mkt = batch.to_pandas()
        is_b_share = b_share_mask(mkt["stock_code"])
        mkt = mkt.loc[~is_b_share].copy()
        if mkt.empty:
            continue
        market_rows += len(mkt)
        mkt["date"] = pd.to_datetime(mkt["date"])
        mkt["market_code"] = normalized_market_code(mkt["stock_code"])
        mkt["code6"] = mkt["market_code"].str.extract(r"(\d{6})", expand=False)
        if mkt["code6"].isna().any():
            raise ValueError("行情表存在无法解析的股票代码")
        mkt["exchange"] = mkt["market_code"].str.extract(r"\.([A-Z]+)$", expand=False)
        if not mkt["exchange"].isin(["SH", "SZ", "BJ"]).all():
            unknown = sorted(mkt.loc[~mkt["exchange"].isin(["SH", "SZ", "BJ"]), "market_code"].unique())
            raise ValueError(f"行情表存在未识别的交易所代码: {unknown[:10]}")
        mkt["market_scope"] = mkt["exchange"].map(
            {"SH": "沪市 A 股", "SZ": "深市 A 股", "BJ": "北交所"}
        )
        mkt["security_id"] = (
            mkt["market_code"].map(old_to_new_market_code).fillna(mkt["market_code"])
        )
        mkt["financial_code6"] = (
            mkt["market_code"].map(new_to_old_financial_code6).fillna(mkt["code6"])
        )
        mkt["bse_code_mapped"] = mkt["market_code"].isin(new_to_old_financial_code6).astype("int8")

        keys = pd.MultiIndex.from_frame(mkt[["date", "code6"]])
        if keys.duplicated().any():
            raise ValueError("行情表存在重复的 date + code6 记录")

        mkt["first_seen_date"] = mkt["security_id"].map(first_seen)
        mkt["listing_date_known"] = mkt["first_seen_date"] > data_start
        mkt["listing_days_observed"] = (mkt["date"] - mkt["first_seen_date"]).dt.days
        mkt["is_cixin"] = (
            mkt["listing_date_known"] & (mkt["listing_days_observed"] < 365)
        ).astype("int8")

        suspended_value = pd.to_numeric(mkt["suspended"], errors="coerce")
        suspended_unknown = suspended_value.isna() & mkt["status"].isna()
        status = mkt["status"].astype("string").str.strip()
        mkt["suspended"] = (
            suspended_value.eq(1).fillna(False)
            | status.eq("停牌").fillna(False)
        ).astype("int8")
        mkt["suspended_unknown"] = suspended_unknown.astype("int8")

        out = mkt[
            [
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
            ]
        ].copy()

        suspended_rows += int(out["suspended"].sum())
        cixin_rows += int(out["is_cixin"].sum())
        cixin_codes.update(out.loc[out["is_cixin"].eq(1), "security_id"].dropna().tolist())

        table = pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
finally:
    if writer is not None:
        writer.close()

if market_rows != kept_rows:
    raise ValueError(
        f"两次行情扫描保留行数不一致: first pass={kept_rows}, second pass={market_rows}"
    )

field_audit = pd.DataFrame(
    [
        {
            "field": "ST/PT",
            "source": "未提供历史字段",
            "availability": "不可用",
            "filter_applied": 0,
            "point_in_time_rule": "不使用当前名称反推历史状态",
        },
        {
            "field": "上市日期",
            "source": "行情首次出现日 first_seen_date",
            "availability": "部分可用",
            "filter_applied": 0,
            "point_in_time_rule": "仅记录代理日期，不作为筛选条件",
        },
        {
            "field": "交易状态",
            "source": "当日行情 suspended/status",
            "availability": "可用",
            "filter_applied": 1,
            "point_in_time_rule": "只使用同一交易日行情记录，不向历史回填",
        },
    ]
)
field_audit.to_csv(OUT_DIR / "point_in_time_field_audit.csv", index=False, encoding="utf-8-sig")
field_audit.to_parquet(OUT_DIR / "point_in_time_field_audit.parquet", index=False)

print(f"source rows: {source_rows:,}")
print(f"B-share rows excluded: {excluded_rows:,}")
print(f"Shanghai B-share stocks excluded: {len(shanghai_b_share_codes):,}")
print(f"Shenzhen B-share stocks excluded: {len(shenzhen_b_share_codes):,}")
print(f"B-share stocks excluded in total: {len(b_share_codes):,}")
print(f"BSE old/new code mappings: {len(bse_mapping):,}")
print(f"market rows kept: {kept_rows:,}")
print(f"data start: {data_start.date()}")
print(f"suspended rows (marked): {suspended_rows:,}")
print(f"new-listing rows (recorded only): {cixin_rows:,}")
print(f"new-listing stocks: {len(cixin_codes):,}")
print("ST/PT filter: not applied")
print("active NaN mask: suspended dates only (TTM/market-cap missingness remains natural NaN)")
print(f"saved: {out_path}")
print(f"saved: {OUT_DIR / 'b_share_codes.parquet'}")
print(f"saved: {OUT_DIR / 'bse_code_mapping.parquet'}")
print(f"saved: {OUT_DIR / 'point_in_time_field_audit.csv'}")
