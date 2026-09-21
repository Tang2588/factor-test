# -*- coding: utf-8 -*-
"""兼容层：北交所新旧代码映射。

2026-09-21 起映射算法统一实现在 ``common.market.derive_bse_mapping``，
本文件只保留 ``step1_样本筛选.py`` 使用的旧入口，避免同一套逻辑存在两份实现。
新代码请直接调用 ``common.market.derive_bse_mapping``。
"""
from __future__ import annotations

import pandas as pd

from common.market import derive_bse_mapping

OLD_LAST_DATE = pd.Timestamp("2025-09-30")
NEW_FIRST_DATE = pd.Timestamp("2025-10-09")
EXPECTED_CONVERSIONS = 242


def derive_bse_code_mapping(market: pd.DataFrame) -> pd.DataFrame:
    """保持原签名的旧入口，内部调用共享实现。"""
    return derive_bse_mapping(
        market,
        OLD_LAST_DATE,
        NEW_FIRST_DATE,
        expected_pairs=EXPECTED_CONVERSIONS,
    )
