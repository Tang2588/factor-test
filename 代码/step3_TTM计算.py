# -*- coding: utf-8 -*-
"""Step 3: build point-in-time TTM observations from report versions."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "中间结果"
reports = pd.read_parquet(DATA_DIR / "reports.parquet")
required = ["code6", "END_DATE", "VERSION_TIME", "N_INCOME_ATTR_P"]
if reports[required].isna().any().any():
    raise ValueError("reports 中存在计算 TTM 所需的缺失字段")
reports["END_DATE"] = pd.to_datetime(reports["END_DATE"])
reports["ACT_PUBTIME"] = pd.to_datetime(reports["ACT_PUBTIME"])
reports["VERSION_TIME"] = pd.to_datetime(reports["VERSION_TIME"])
reports = reports.sort_values(["code6", "VERSION_TIME", "ACT_PUBTIME", "END_DATE"])


def build_code_ttm(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    state: dict[pd.Timestamp, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for version_time, batch in group.groupby("VERSION_TIME", sort=True):
        state_changed = False
        accepted_rows = []
        for row in batch.itertuples(index=False):
            current = state.get(row.END_DATE)
            if current is None:
                accepted = True
                decision = "new_report_period"
            elif row.ACT_PUBTIME > current["ACT_PUBTIME"]:
                accepted = True
                decision = "newer_announcement"
            elif (
                row.ACT_PUBTIME == current["ACT_PUBTIME"]
                and version_time > current["VERSION_TIME"]
            ):
                accepted = True
                decision = "same_announcement_update"
            elif row.ACT_PUBTIME < current["ACT_PUBTIME"]:
                accepted = False
                decision = "older_announcement_ignored"
            else:
                accepted = False
                decision = "duplicate_effective_event_ignored"

            if accepted:
                state[row.END_DATE] = {
                    "ACT_PUBTIME": row.ACT_PUBTIME,
                    "UPDATE_TIME": row.UPDATE_TIME,
                    "VERSION_TIME": version_time,
                    "N_INCOME_ATTR_P": float(row.N_INCOME_ATTR_P),
                }
                state_changed = True
                accepted_rows.append(row)

            selected = state.get(row.END_DATE)
            decisions.append(
                {
                    "code6": row.code6,
                    "END_DATE": row.END_DATE,
                    "FISCAL_PERIOD": row.FISCAL_PERIOD,
                    "ACT_PUBTIME": row.ACT_PUBTIME,
                    "UPDATE_TIME": row.UPDATE_TIME,
                    "VERSION_TIME": version_time,
                    "N_INCOME_ATTR_P": float(row.N_INCOME_ATTR_P),
                    "accepted": int(accepted),
                    "decision": decision,
                    "selected_act_pubtime_after": selected["ACT_PUBTIME"],
                    "selected_version_time_after": selected["VERSION_TIME"],
                }
            )

        if not state_changed:
            continue

        latest_end = max(state)
        current_record = state[latest_end]
        current = float(current_record["N_INCOME_ATTR_P"])
        year, month = latest_end.year, latest_end.month
        previous_annual_date = pd.NaT
        previous_same_date = pd.NaT
        previous_annual = float("nan")
        previous_same = float("nan")
        if month == 12:
            ttm = current
        else:
            previous_annual_date = pd.Timestamp(year - 1, 12, 31)
            previous_same_date = (
                pd.Timestamp(year - 1, month, 1) + pd.offsets.MonthEnd(0)
            )
            if previous_annual_date not in state or previous_same_date not in state:
                ttm = float("nan")
            else:
                previous_annual = float(
                    state[previous_annual_date]["N_INCOME_ATTR_P"]
                )
                previous_same = float(
                    state[previous_same_date]["N_INCOME_ATTR_P"]
                )
                ttm = (
                    current
                    + previous_annual
                    - previous_same
                )
        rows.append(
            {
                "code6": group["code6"].iat[0],
                "END_DATE": latest_end,
                "ACT_PUBTIME": current_record["ACT_PUBTIME"],
                "REPORT_VERSION_TIME": current_record["VERSION_TIME"],
                "VERSION_TIME": version_time,
                "TTM_ni": ttm,
                "TTM_current_ni": current,
                "TTM_previous_annual_date": previous_annual_date,
                "TTM_previous_annual_ni": previous_annual,
                "TTM_previous_same_period_date": previous_same_date,
                "TTM_previous_same_period_ni": previous_same,
                "EVENT_ACT_PUBTIME": max(row.ACT_PUBTIME for row in accepted_rows),
                "EVENT_ACCEPTED_ROWS": len(accepted_rows),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(decisions)


built = [build_code_ttm(group) for _, group in reports.groupby("code6", sort=False)]
result = pd.concat([item[0] for item in built], ignore_index=True)
decisions = pd.concat([item[1] for item in built], ignore_index=True)
result = result.sort_values(["code6", "VERSION_TIME"]).reset_index(drop=True)
if result.duplicated(["code6", "VERSION_TIME"]).any():
    raise ValueError("TTM 版本表存在重复的 code6 + VERSION_TIME")

effective = decisions.loc[decisions["accepted"].eq(1)].copy()
effective = effective.sort_values(["code6", "END_DATE", "VERSION_TIME"])
prior_max_act = effective.groupby(["code6", "END_DATE"], sort=False)[
    "ACT_PUBTIME"
].transform(lambda values: values.cummax().shift())
if (effective["ACT_PUBTIME"] < prior_max_act).any():
    raise ValueError("有效财报版本中仍存在旧公告覆盖新公告")

result.to_parquet(DATA_DIR / "reports_ttm.parquet", index=False)
effective.to_parquet(DATA_DIR / "reports_effective.parquet", index=False)
decisions.to_parquet(DATA_DIR / "report_version_decisions.parquet", index=False)
print(f"reports rows: {len(result):,}")
print(f"TTM NaN rows: {int(result['TTM_ni'].isna().sum()):,}")
print(f"TTM negative rows: {int(result['TTM_ni'].lt(0).sum()):,}")
print(f"effective report rows: {len(effective):,}")
print(
    "older announcement rows ignored: "
    f"{int(decisions['decision'].eq('older_announcement_ignored').sum()):,}"
)
print(f"saved: {DATA_DIR / 'reports_ttm.parquet'}")
print(f"saved: {DATA_DIR / 'reports_effective.parquet'}")
print(f"saved: {DATA_DIR / 'report_version_decisions.parquet'}")
