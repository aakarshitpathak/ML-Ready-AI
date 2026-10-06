"""
modules/etl.py

Small ETL layer for ML Ready AI.
- Extract: load files with the existing loader
- Transform: clean column names, duplicates and missing values
- Load: SQLite or any SQLAlchemy database URL (including MySQL)
"""

from __future__ import annotations

import re

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def normalize_column_name(name: str) -> str:
    value = str(name).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "column"


def transform_for_database(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out.columns = [
        normalize_column_name(c)
        for c in out.columns
    ]

    out = out.drop_duplicates().reset_index(drop=True)

    # Keep database-friendly missing values.
    out = out.where(pd.notna(out), None)

    return out


def get_sqlalchemy_engine(database_url: str) -> Engine:
    """
    Examples:
      SQLite:
        sqlite:///instance/analytics.db

      MySQL:
        mysql+pymysql://user:password@localhost/database_name
    """
    if not database_url:
        raise ValueError("database_url is required.")
    return create_engine(database_url, pool_pre_ping=True)


def load_to_sql(
    df: pd.DataFrame,
    database_url: str,
    table_name: str,
    *,
    if_exists: str = "replace",
) -> int:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError("table_name contains invalid SQL identifier characters.")

    clean = transform_for_database(df)
    engine = get_sqlalchemy_engine(database_url)

    try:
        clean.to_sql(
            table_name,
            engine,
            if_exists=if_exists,
            index=False,
            method="multi",
        )
    finally:
        engine.dispose()

    return int(len(clean))


def extract_transform_load(
    filepath: str,
    database_url: str,
    table_name: str,
    *,
    if_exists: str = "replace",
) -> dict[str, int | str]:
    from modules.data_loader import load_dataframe
    extracted = load_dataframe(filepath)
    transformed = transform_for_database(extracted)
    loaded = load_to_sql(
        transformed,
        database_url,
        table_name,
        if_exists=if_exists,
    )

    return {
        "source_rows": int(len(extracted)),
        "source_columns": int(extracted.shape[1]),
        "loaded_rows": loaded,
        "loaded_table": table_name,
    }
