# -*- coding: utf-8 -*-
"""分片读取 parquet 面板。

行情文件约 640 MB、715 万行，一次性读入内存不划算，因此统一用
``pyarrow.parquet.ParquetFile.iter_batches`` 分片扫描，只取需要的列。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


OLD_BSE_CODE_PATTERN = r"^(43|83|87)\d{4}\.BJ$"
NEW_BSE_CODE_PATTERN = r"^920\d{3}\.BJ$"


@dataclass
class MarketIndex:
    """行情文件的行级索引信息，避免重复扫描大文件。"""

    dates: pd.DatetimeIndex
    old_bse_last_date: pd.Timestamp | None = None
    new_bse_first_date: pd.Timestamp | None = None
    old_bse_count_on_last_date: int = 0
    new_bse_count_on_first_date: int = 0
    new_bse_count_before_switch: int = 0
    dates_with_old_bse: list = field(default_factory=list)
    dates_with_new_bse: list = field(default_factory=list)

    @property
    def expected_conversions(self) -> int:
        """切换日新增的 920 代码数量，等于需要映射的新旧代码对数量。"""
        return self.new_bse_count_on_first_date - self.new_bse_count_before_switch


def read_selected_dates(
    path: Path,
    columns: list[str],
    dates: set[pd.Timestamp],
    batch_size: int = 200_000,
) -> pd.DataFrame:
    """只读取指定交易日的行。``date`` 列会被规范化为零点时间戳。"""
    pieces: list[pd.DataFrame] = []
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=batch_size):
        frame = batch.to_pandas(ignore_metadata=True)
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        selected = frame.loc[frame["date"].isin(dates)].copy()
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        return pd.DataFrame(columns=columns)
    return pd.concat(pieces, ignore_index=True)


def read_all_dates(path: Path, batch_size: int = 300_000) -> pd.DatetimeIndex:
    """扫描单个 ``date`` 列，返回全部交易日。"""
    dates: set[pd.Timestamp] = set()
    for batch in pq.ParquetFile(path).iter_batches(columns=["date"], batch_size=batch_size):
        values = pd.to_datetime(batch.to_pandas(ignore_metadata=True)["date"]).dt.normalize()
        dates.update(values)
    return pd.DatetimeIndex(sorted(dates))


def scan_market_index(path: Path, batch_size: int = 300_000) -> MarketIndex:
    """一次扫描同时取得交易日历和北交所代码切换位置。

    这样就不需要把切换日写死在脚本里：旧代码（43/83/87xxxx.BJ）最后出现的
    交易日即切换前最后交易日，其后第一个交易日即新代码（920xxx.BJ）启用日。
    """
    old_counts: dict[pd.Timestamp, int] = {}
    new_counts: dict[pd.Timestamp, int] = {}
    for batch in pq.ParquetFile(path).iter_batches(
        columns=["date", "stock_code"], batch_size=batch_size
    ):
        frame = batch.to_pandas(ignore_metadata=True)
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        code = frame["stock_code"].astype("string").str.upper()
        flags = pd.DataFrame(
            {
                "date": frame["date"],
                "old": code.str.match(OLD_BSE_CODE_PATTERN, na=False),
                "new": code.str.match(NEW_BSE_CODE_PATTERN, na=False),
            }
        )
        grouped = flags.groupby("date")[["old", "new"]].sum()
        for date, row in grouped.iterrows():
            old_counts[date] = old_counts.get(date, 0) + int(row["old"])
            new_counts[date] = new_counts.get(date, 0) + int(row["new"])

    all_dates = pd.DatetimeIndex(sorted(old_counts))
    dates_with_old = sorted(d for d, count in old_counts.items() if count > 0)
    dates_with_new = sorted(d for d, count in new_counts.items() if count > 0)

    index = MarketIndex(
        dates=all_dates,
        dates_with_old_bse=dates_with_old,
        dates_with_new_bse=dates_with_new,
    )
    if dates_with_old:
        old_last = dates_with_old[-1]
        index.old_bse_last_date = old_last
        index.old_bse_count_on_last_date = old_counts[old_last]
        index.new_bse_count_before_switch = new_counts.get(old_last, 0)
        later = [d for d in all_dates if d > old_last]
        if later:
            new_first = later[0]
            index.new_bse_first_date = new_first
            index.new_bse_count_on_first_date = new_counts.get(new_first, 0)
    return index
