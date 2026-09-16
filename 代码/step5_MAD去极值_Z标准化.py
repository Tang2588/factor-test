# -*- coding: utf-8 -*-
"""Step 5: cross-sectional MAD winsorization and Z-score standardization.

EP and PE are processed independently.  Their raw values remain available in
``ep_raw.parquet`` and ``pe_raw.parquet``.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

for parent in Path(__file__).resolve().parents:
    if (parent / "pure_factor_streaming.py").exists():
        sys.path.insert(0, str(parent))
        break
from pure_factor_streaming import standardize_factor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "中间结果"
FINAL = ROOT / "因子结果"
COVERAGE_THRESHOLD = 0.95
STABLE_MONTH_ENDS = 3


class FrameCursor:
    def __init__(self, path: Path, columns: list[str], batch_size: int = 100_000):
        parquet = pq.ParquetFile(path)
        self.iterator = iter(parquet.iter_batches(columns=columns, batch_size=batch_size))
        self.current = pd.DataFrame()
        self.offset = 0

    def take(self, size: int) -> pd.DataFrame:
        parts = []
        remaining = size
        while remaining:
            if self.offset >= len(self.current):
                try:
                    self.current = next(self.iterator).to_pandas(ignore_metadata=True)
                except StopIteration as exc:
                    raise ValueError("因子文件行数少于审计表") from exc
                self.offset = 0
            available = len(self.current) - self.offset
            take = min(remaining, available)
            parts.append(self.current.iloc[self.offset:self.offset + take])
            self.offset += take
            remaining -= take
        return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0].reset_index(drop=True)

    def assert_exhausted(self) -> None:
        if self.offset < len(self.current):
            raise ValueError("因子文件行数多于审计表")
        try:
            next(self.iterator)
        except StopIteration:
            return
        raise ValueError("因子文件行数多于审计表")


def enrich_factor_audit() -> tuple[pd.DataFrame, pd.Timestamp]:
    audit_path = DATA / "factor_audit.parquet"
    temp_path = DATA / "factor_audit_complete.parquet"
    temp_path.unlink(missing_ok=True)

    sources = {
        "raw_ep": DATA / "ep_raw.parquet",
        "raw_pe": DATA / "pe_raw.parquet",
        "mad_ep": DATA / "ep_mad.parquet",
        "mad_pe": DATA / "pe_mad.parquet",
        "z_ep": DATA / "ep.parquet",
        "z_pe": DATA / "pe.parquet",
    }
    cursors = {
        name: FrameCursor(path, ["date", "stock_code", "signal"])
        for name, path in sources.items()
    }
    audit_parquet = pq.ParquetFile(audit_path)
    writer = None
    coverage_parts = []

    try:
        for batch in audit_parquet.iter_batches(batch_size=100_000):
            audit = batch.to_pandas(ignore_metadata=True).reset_index(drop=True)
            size = len(audit)
            for name, cursor in cursors.items():
                factor = cursor.take(size)
                same_keys = (
                    pd.to_datetime(factor["date"]).equals(pd.to_datetime(audit["date"]))
                    and factor["stock_code"].astype("string").equals(
                        audit["stock_code"].astype("string")
                    )
                )
                if not same_keys:
                    raise ValueError(f"{name} 与逐行审计表的 date + stock_code 顺序不一致")
                values = pd.to_numeric(factor["signal"], errors="coerce").to_numpy()
                if name in audit.columns:
                    existing = pd.to_numeric(audit[name], errors="coerce").to_numpy()
                    if not np.allclose(existing, values, equal_nan=True):
                        raise ValueError(f"审计表中的 {name} 与因子文件不一致")
                audit[name] = values

            daily = (
                audit.groupby("date", sort=False)
                .agg(
                    n_total=("stock_code", "size"),
                    n_suspended=("suspended", "sum"),
                    n_valid_ep=("raw_ep", "count"),
                    n_valid_pe=("raw_pe", "count"),
                )
                .reset_index()
            )
            coverage_parts.append(daily)

            table = pa.Table.from_pandas(audit, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
        audit_parquet.close()

    for cursor in cursors.values():
        cursor.assert_exhausted()
    if writer is None:
        raise ValueError("逐行审计表为空")

    temp_path.replace(audit_path)
    coverage = (
        pd.concat(coverage_parts, ignore_index=True)
        .groupby("date", as_index=False, sort=True)
        .sum(numeric_only=True)
    )
    coverage["date"] = pd.to_datetime(coverage["date"])
    coverage["n_non_suspended"] = coverage["n_total"] - coverage["n_suspended"]
    coverage["ep_coverage_all"] = coverage["n_valid_ep"] / coverage["n_total"]
    coverage["pe_coverage_all"] = coverage["n_valid_pe"] / coverage["n_total"]
    denominator = coverage["n_non_suspended"].replace(0, np.nan)
    coverage["ep_coverage_non_suspended"] = coverage["n_valid_ep"] / denominator
    coverage["pe_coverage_non_suspended"] = coverage["n_valid_pe"] / denominator

    month = coverage["date"].dt.to_period("M")
    month_end_dates = coverage.groupby(month, sort=True)["date"].max()
    coverage["is_month_end"] = coverage["date"].isin(month_end_dates)
    month_ends = coverage.loc[coverage["is_month_end"]].copy()
    month_ends["coverage_meets_threshold"] = month_ends[
        "ep_coverage_non_suspended"
    ].ge(COVERAGE_THRESHOLD)
    month_ends["stability_confirmed"] = (
        month_ends["coverage_meets_threshold"]
        .rolling(STABLE_MONTH_ENDS, min_periods=STABLE_MONTH_ENDS)
        .sum()
        .eq(STABLE_MONTH_ENDS)
    )
    confirmed = month_ends.loc[month_ends["stability_confirmed"], "date"]
    if confirmed.empty:
        raise ValueError("没有找到满足覆盖率稳定规则的月末")
    test_start = pd.Timestamp(confirmed.iloc[0])

    coverage["coverage_meets_threshold"] = coverage[
        "ep_coverage_non_suspended"
    ].ge(COVERAGE_THRESHOLD)
    coverage["stability_confirmed"] = False
    coverage.loc[
        coverage["date"].isin(month_ends.loc[month_ends["stability_confirmed"], "date"]),
        "stability_confirmed",
    ] = True
    coverage["formal_test_period"] = coverage["date"].ge(test_start)
    coverage.to_parquet(DATA / "factor_daily_coverage.parquet", index=False)
    coverage.to_csv(DATA / "factor_daily_coverage.csv", index=False, encoding="utf-8-sig")

    config = {
        "coverage_denominator": "all non-suspended rows in the retained market panel",
        "coverage_threshold": COVERAGE_THRESHOLD,
        "required_consecutive_month_ends": STABLE_MONTH_ENDS,
        "test_start_rule": "first month-end that confirms the consecutive-month rule",
        "formal_test_start_date": test_start.date().isoformat(),
        "st_filter_applied": False,
        "listing_filter_applied": False,
    }
    with (DATA / "factor_test_start.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    return coverage, test_start

ep_totals = standardize_factor(
    DATA / "ep_raw.parquet",
    DATA / "ep_mad.parquet",
    DATA / "ep.parquet",
    DATA / "ep_processing_stats.parquet",
    FINAL / "ep.parquet",
)
pe_totals = standardize_factor(
    DATA / "pe_raw.parquet",
    DATA / "pe_mad.parquet",
    DATA / "pe.parquet",
    DATA / "pe_processing_stats.parquet",
    FINAL / "pe.parquet",
)
coverage, test_start = enrich_factor_audit()
print("EP:")
print(f"  raw valid rows: {ep_totals['raw_valid']:,}")
print(f"  MAD clipped rows: {ep_totals['mad_clipped']:,}")
print(f"  constant cross-sections: {ep_totals['constant_cross_sections']:,}")
print(f"  final valid Z rows: {ep_totals['final_valid_z']:,}")
print("PE:")
print(f"  raw valid rows: {pe_totals['raw_valid']:,}")
print(f"  MAD clipped rows: {pe_totals['mad_clipped']:,}")
print(f"  constant cross-sections: {pe_totals['constant_cross_sections']:,}")
print(f"  final valid Z rows: {pe_totals['final_valid_z']:,}")
print(f"saved: {DATA / 'ep_mad.parquet'}")
print(f"saved: {DATA / 'ep.parquet'}")
print(f"saved: {FINAL / 'ep.parquet'}")
print(f"saved: {DATA / 'pe_mad.parquet'}")
print(f"saved: {DATA / 'pe.parquet'}")
print(f"saved: {FINAL / 'pe.parquet'}")
print(
    "coverage rule: "
    f">={COVERAGE_THRESHOLD:.0%} for {STABLE_MONTH_ENDS} consecutive month-ends"
)
print(f"formal test start date: {test_start.date()}")
print(f"saved: {DATA / 'factor_audit.parquet'}")
print(f"saved: {DATA / 'factor_daily_coverage.parquet'}")
print(f"saved: {DATA / 'factor_test_start.json'}")
