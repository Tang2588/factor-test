# -*- coding: utf-8 -*-
"""EP/PE 因子项目的共享底层模块。

本包只提供可复用的数据准备函数，不执行任何业务阶段逻辑，也不写出结果文件。
各阶段脚本（数据清洗、因子计算、回归分析）统一从这里取数，避免同一套口径
在多个脚本里各写一份。

- ``paths``      项目路径与共享常量
- ``parquet_io`` 分片读取 parquet 面板
- ``market``     行情快照清洗、B 股剔除、北交所新旧代码映射
- ``calendar``   月度形成／建仓／退出日历
- ``returns``    前瞻收益
- ``panel``      形成日面板、行业面板与回归面板
"""
from __future__ import annotations

from .calendar import build_monthly_calendar
from .market import (
    apply_security_id,
    b_share_mask,
    derive_bse_mapping,
    extract_code6,
    normalize_market_code,
    prepare_market_snapshot,
)
from .panel import (
    build_regression_panel,
    prepare_formation_panel,
    prepare_industry,
)
from .parquet_io import (
    MarketIndex,
    read_all_dates,
    read_selected_dates,
    scan_market_index,
)
from .paths import (
    FACTOR_DIR,
    HUBER_T,
    INDUSTRY_PATH,
    MARKET_PATH,
    MAX_ITER,
    MID_DIR,
    MIN_OBSERVATIONS,
    PROJECT_ROOT,
    REG_DIR,
    TEST_START_PATH,
    TOL,
)
from .returns import build_endpoint_forward_returns

__all__ = [
    "FACTOR_DIR",
    "HUBER_T",
    "INDUSTRY_PATH",
    "MARKET_PATH",
    "MAX_ITER",
    "MID_DIR",
    "MIN_OBSERVATIONS",
    "PROJECT_ROOT",
    "REG_DIR",
    "TEST_START_PATH",
    "TOL",
    "MarketIndex",
    "apply_security_id",
    "b_share_mask",
    "build_endpoint_forward_returns",
    "build_monthly_calendar",
    "build_regression_panel",
    "derive_bse_mapping",
    "extract_code6",
    "normalize_market_code",
    "prepare_formation_panel",
    "prepare_industry",
    "prepare_market_snapshot",
    "read_all_dates",
    "read_selected_dates",
    "scan_market_index",
]
