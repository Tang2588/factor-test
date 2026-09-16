# -*- coding: utf-8 -*-
"""Low-memory helpers for pure factor calculation pipelines."""
from __future__ import annotations

from pathlib import Path
from shutil import copyfile
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _open_indexed_writer(path: Path, frame: pd.DataFrame) -> pq.ParquetWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=True)
    return pq.ParquetWriter(path, table.schema)


def _write_indexed(writer: pq.ParquetWriter, frame: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(frame, schema=writer.schema, preserve_index=True)
    writer.write_table(table)


def _open_plain_writer(path: Path, frame: pd.DataFrame) -> pq.ParquetWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    return pq.ParquetWriter(path, table.schema)


def _write_plain(writer: pq.ParquetWriter, frame: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(frame, schema=writer.schema, preserve_index=False)
    writer.write_table(table)


def _load_events(path: Path, value_columns: list[str]) -> dict[str, dict[str, np.ndarray]]:
    cols = ["code6", "VERSION_TIME", *value_columns]
    events = pd.read_parquet(path, columns=cols)
    events["code6"] = events["code6"].astype("string")
    events["VERSION_TIME"] = pd.to_datetime(events["VERSION_TIME"])
    events = events.sort_values(["code6", "VERSION_TIME"])
    out: dict[str, dict[str, np.ndarray]] = {}
    for code, group in events.groupby("code6", sort=False):
        data = {"VERSION_TIME": group["VERSION_TIME"].to_numpy(dtype="datetime64[ns]")}
        for col in value_columns:
            data[col] = group[col].to_numpy()
        out[str(code)] = data
    return out


def stream_align_market_events(
    mkt_path: Path,
    events_path: Path,
    value_columns: list[str],
    make_batch: Callable,
    raw_path: Path,
    mask_path: Path,
    batch_size: int = 100_000,
    raw_columns: list[str] | None = None,
    derived_raw_paths: dict[str, Path] | None = None,
    additional_frame_paths: dict[str, Path] | None = None,
    market_event_code_column: str = "code6",
) -> dict[str, int]:
    """Align point-in-time events and write one or more indexed raw signals.

    ``raw_columns`` keeps the primary raw output compatible with the existing
    single-signal factor format. ``derived_raw_paths`` can additionally route
    named columns returned by ``make_batch`` to signal-only raw factor files.
    ``additional_frame_paths`` writes optional indexed audit frames returned as
    the fourth item from ``make_batch``.
    """
    events_by_code = _load_events(events_path, value_columns)
    pf = pq.ParquetFile(mkt_path)
    raw_writer = None
    mask_writer = None
    derived_writers: dict[str, pq.ParquetWriter] = {}
    additional_writers: dict[str, pq.ParquetWriter] = {}
    totals: dict[str, int] = {"market_rows": 0, "lookahead": 0}

    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            mkt = batch.to_pandas(ignore_metadata=True)
            mkt["date"] = pd.to_datetime(mkt["date"])
            mkt["code6"] = mkt["code6"].astype("string")
            if market_event_code_column not in mkt.columns:
                raise KeyError(f"market event code column not found: {market_event_code_column}")
            mkt[market_event_code_column] = mkt[market_event_code_column].astype("string")
            mkt["signal_cutoff"] = (
                mkt["date"] + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            )
            for col in ["VERSION_TIME", *value_columns]:
                mkt[col] = pd.NA

            for code, idx in mkt.groupby(market_event_code_column, sort=False).groups.items():
                event_data = events_by_code.get(str(code))
                if event_data is None:
                    continue
                times = event_data["VERSION_TIME"]
                cutoffs = mkt.loc[idx, "signal_cutoff"].to_numpy(dtype="datetime64[ns]")
                pos = np.searchsorted(times, cutoffs, side="right") - 1
                ok = pos >= 0
                if not ok.any():
                    continue
                target = idx[ok]
                event_pos = pos[ok]
                mkt.loc[target, "VERSION_TIME"] = times[event_pos]
                for col in value_columns:
                    mkt.loc[target, col] = event_data[col][event_pos]

            version = pd.to_datetime(mkt["VERSION_TIME"], errors="coerce")
            lookahead = version.notna() & (version > mkt["signal_cutoff"])
            if lookahead.any():
                raise ValueError(f"lookahead violations: {int(lookahead.sum())}")

            result = make_batch(mkt)
            if len(result) == 3:
                raw, mask, stats = result
                additional_frames = {}
            elif len(result) == 4:
                raw, mask, stats, additional_frames = result
            else:
                raise ValueError("make_batch must return 3 or 4 items")
            totals["market_rows"] += len(mkt)
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + int(value)

            raw_out = raw if raw_columns is None else raw[raw_columns]
            if raw_writer is None:
                raw_writer = _open_indexed_writer(raw_path, raw_out)
            if mask_writer is None:
                mask_writer = _open_indexed_writer(mask_path, mask)
            _write_indexed(raw_writer, raw_out)
            _write_indexed(mask_writer, mask)

            for column, path in (derived_raw_paths or {}).items():
                if column not in raw.columns:
                    raise KeyError(f"make_batch did not return derived raw column: {column}")
                derived = raw[[column]].rename(columns={column: "signal"})
                writer = derived_writers.get(column)
                if writer is None:
                    writer = _open_indexed_writer(path, derived)
                    derived_writers[column] = writer
                _write_indexed(writer, derived)

            expected_frames = set((additional_frame_paths or {}).keys())
            if set(additional_frames.keys()) != expected_frames:
                raise KeyError(
                    "additional frame names differ from additional_frame_paths: "
                    f"frames={sorted(additional_frames)}, paths={sorted(expected_frames)}"
                )
            for name, path in (additional_frame_paths or {}).items():
                frame = additional_frames[name]
                writer = additional_writers.get(name)
                if writer is None:
                    writer = _open_indexed_writer(path, frame)
                    additional_writers[name] = writer
                _write_indexed(writer, frame)
    finally:
        if raw_writer is not None:
            raw_writer.close()
        if mask_writer is not None:
            mask_writer.close()
        for writer in derived_writers.values():
            writer.close()
        for writer in additional_writers.values():
            writer.close()
    return totals


def stream_size_raw(mkt_path: Path, raw_path: Path, mask_path: Path, batch_size: int = 100_000) -> dict[str, int]:
    pf = pq.ParquetFile(mkt_path)
    raw_writer = None
    mask_writer = None
    totals = {"market_rows": 0, "positive_market_cap": 0, "valid_raw": 0}
    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            mkt = batch.to_pandas(ignore_metadata=True)
            mkt["date"] = pd.to_datetime(mkt["date"])
            mkt["code6"] = mkt["code6"].astype("string")
            mkt["me_total"] = pd.to_numeric(mkt["me_total"], errors="coerce")
            signal = pd.Series(np.nan, index=mkt.index, dtype="float64")
            positive = mkt["me_total"].gt(0)
            signal.loc[positive] = np.log(mkt.loc[positive, "me_total"])
            signal.loc[mkt["suspended"].eq(1)] = np.nan

            raw = pd.DataFrame({"date": mkt["date"], "stock_code": mkt["code6"], "signal": signal})
            raw = raw.set_index(["date", "stock_code"])
            mask = mkt[["date", "code6", "me_total", "suspended", "suspended_unknown", "is_cixin"]].rename(columns={"code6": "stock_code"})
            mask = mask.set_index(["date", "stock_code"])
            mask["st_filter_applied"] = 0
            mask["st_filter_note"] = "historical ST/PT data unavailable"

            totals["market_rows"] += len(mkt)
            totals["positive_market_cap"] += int(positive.sum())
            totals["valid_raw"] += int(raw["signal"].notna().sum())
            if raw_writer is None:
                raw_writer = _open_indexed_writer(raw_path, raw)
            if mask_writer is None:
                mask_writer = _open_indexed_writer(mask_path, mask)
            _write_indexed(raw_writer, raw)
            _write_indexed(mask_writer, mask)
    finally:
        if raw_writer is not None:
            raw_writer.close()
        if mask_writer is not None:
            mask_writer.close()
    return totals


def standardize_factor(
    raw_path: Path,
    mad_path: Path,
    z_path: Path,
    stats_path: Path,
    final_path: Path | None = None,
    batch_size: int = 50_000,
) -> dict[str, int]:
    pf = pq.ParquetFile(raw_path)
    mad_writer = None
    z_writer = None
    carry = pd.DataFrame()
    stats_rows = []
    totals = {"raw_valid": 0, "mad_clipped": 0, "final_valid_z": 0, "constant_cross_sections": 0}

    def process_day(day: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
        day = day.copy()
        values = pd.to_numeric(day["signal"], errors="coerce")
        valid = values.notna()
        mad_signal = pd.Series(np.nan, index=day.index, dtype="float64")
        z_signal = pd.Series(np.nan, index=day.index, dtype="float64")
        median = mad = lower = upper = mean = std = np.nan
        clipped_count = 0
        if valid.any():
            valid_values = values.loc[valid]
            median = float(valid_values.median())
            mad = float((valid_values - median).abs().median())
            scale = mad * 1.4826
            if scale > 0:
                lower = median - 3.0 * scale
                upper = median + 3.0 * scale
                clipped = valid_values.clip(lower=lower, upper=upper)
            else:
                lower = -np.inf
                upper = np.inf
                clipped = valid_values
            clipped_count = int((clipped != valid_values).sum())
            mean = float(clipped.mean())
            std = float(clipped.std(ddof=0))
            mad_signal.loc[valid] = clipped
            if std > 0:
                z_signal.loc[valid] = (clipped - mean) / std
        mad_out = pd.DataFrame({"date": day["date"], "stock_code": day["stock_code"], "signal": mad_signal}).set_index(["date", "stock_code"])
        z_out = pd.DataFrame({"date": day["date"], "stock_code": day["stock_code"], "signal": z_signal}).set_index(["date", "stock_code"])
        row = {
            "date": day["date"].iloc[0],
            "n_valid": int(valid.sum()),
            "median": median,
            "mad": mad,
            "lower": lower,
            "upper": upper,
            "n_clipped": clipped_count,
            "mean_after_mad": mean,
            "std_after_mad": std,
        }
        return mad_out, z_out, row

    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            chunk = batch.to_pandas(ignore_metadata=True)
            chunk["date"] = pd.to_datetime(chunk["date"])
            chunk["stock_code"] = chunk["stock_code"].astype("string")
            chunk = pd.concat([carry, chunk], ignore_index=True) if not carry.empty else chunk
            last_date = chunk["date"].iloc[-1]
            ready = chunk["date"].ne(last_date)
            process = chunk.loc[ready]
            carry = chunk.loc[~ready].copy()
            for _, day in process.groupby("date", sort=False):
                mad_out, z_out, row = process_day(day)
                stats_rows.append(row)
                totals["raw_valid"] += row["n_valid"]
                totals["mad_clipped"] += row["n_clipped"]
                totals["final_valid_z"] += int(z_out["signal"].notna().sum())
                totals["constant_cross_sections"] += int(row["n_valid"] > 0 and not (row["std_after_mad"] > 0))
                if mad_writer is None:
                    mad_writer = _open_indexed_writer(mad_path, mad_out)
                if z_writer is None:
                    z_writer = _open_indexed_writer(z_path, z_out)
                _write_indexed(mad_writer, mad_out)
                _write_indexed(z_writer, z_out)
        if not carry.empty:
            for _, day in carry.groupby("date", sort=False):
                mad_out, z_out, row = process_day(day)
                stats_rows.append(row)
                totals["raw_valid"] += row["n_valid"]
                totals["mad_clipped"] += row["n_clipped"]
                totals["final_valid_z"] += int(z_out["signal"].notna().sum())
                totals["constant_cross_sections"] += int(row["n_valid"] > 0 and not (row["std_after_mad"] > 0))
                if mad_writer is None:
                    mad_writer = _open_indexed_writer(mad_path, mad_out)
                if z_writer is None:
                    z_writer = _open_indexed_writer(z_path, z_out)
                _write_indexed(mad_writer, mad_out)
                _write_indexed(z_writer, z_out)
    finally:
        if mad_writer is not None:
            mad_writer.close()
        if z_writer is not None:
            z_writer.close()

    stats = pd.DataFrame(stats_rows).sort_values("date")
    stats.to_parquet(stats_path, index=False)
    if final_path is not None:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        copyfile(z_path, final_path)
    return totals
