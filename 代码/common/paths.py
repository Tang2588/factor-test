# -*- coding: utf-8 -*-
"""项目路径与共享常量。

基础数据目录按以下优先级解析，便于同一份代码在不同机器上运行：

1. 环境变量 ``EP_FACTOR_DATA_ROOT``
2. ``config/paths.local.toml``（本机覆盖，不纳入版本管理）
3. ``config/paths.toml``（仓库内默认值）
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "代码"
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "paths.toml"
LOCAL_CONFIG_PATH = CONFIG_DIR / "paths.local.toml"

FALLBACK_DATA_ROOT = r"D:\实习生学习项目\基础数据"

# 回归共享口径常量：所有回归脚本必须使用同一套取值
HUBER_T = 1.345
MAX_ITER = 100
TOL = 1e-8
MIN_OBSERVATIONS = 500


def _read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_data_root() -> Path:
    """按环境变量、本机配置、仓库默认值的顺序解析基础数据目录。"""
    override = os.environ.get("EP_FACTOR_DATA_ROOT")
    if override:
        return Path(override)
    config: dict = {}
    config.update(_read_config(DEFAULT_CONFIG_PATH))
    config.update(_read_config(LOCAL_CONFIG_PATH))
    return Path(config.get("data_root", FALLBACK_DATA_ROOT))


DATA_ROOT = resolve_data_root()

MARKET_PATH = DATA_ROOT / "chn_equ_mkt_quotation.parquet"
INDUSTRY_PATH = DATA_ROOT / "chn_equ_indus_sw.parquet"
INCOME_PATH = DATA_ROOT / "vw_fdmt_is_new.parquet"

MID_DIR = PROJECT_ROOT / "中间结果"
FACTOR_DIR = PROJECT_ROOT / "因子结果"
REG_DIR = PROJECT_ROOT / "回归结果"
STATS_DIR = PROJECT_ROOT / "标准化统计"
FIGURE_DIR = PROJECT_ROOT / "图形"

EP_PATH = FACTOR_DIR / "ep.parquet"
PE_PATH = FACTOR_DIR / "pe.parquet"
TEST_START_PATH = MID_DIR / "factor_test_start.json"


def resolve_factor_path(factor: str) -> Path:
    """把因子名或路径解析为因子文件。

    支持三种写法：

    - ``"ep"``                          → ``因子结果/ep.parquet``
    - ``"因子结果/ep.parquet"``          → 相对项目根目录解析
    - ``"D:/其它/因子/pb.parquet"``      → 绝对路径直接使用
    """
    candidate = Path(factor)
    if candidate.suffix.lower() == ".parquet" or candidate.exists():
        return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
    return FACTOR_DIR / f"{factor}.parquet"
