"""
modules/advanced_analytics.py

Reusable analytics utilities for ML Ready AI:
- data quality and statistical profiling
- hypothesis testing
- PCA
- correlation analysis
- Excel / Tableau-ready exports
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def data_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact data-quality profile suitable for a dashboard/API."""
    missing = df.isna().sum()
    missing_pct = (missing / max(len(df), 1) * 100).round(2)

    numeric = df.select_dtypes(include=[np.number])
    categorical = df.select_dtypes(include=["object", "category", "bool"])

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "columns_with_missing": int((missing > 0).sum()),
        "numeric_columns": int(numeric.shape[1]),
        "categorical_columns": int(categorical.shape[1]),
        "constant_columns": [
            str(c) for c in df.columns if df[c].nunique(dropna=False) <= 1
        ],
        "missing_by_column": {
            str(c): {
                "count": int(missing[c]),
                "percent": float(missing_pct[c]),
            }
            for c in df.columns
            if missing[c] > 0
        },
    }


def statistical_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Descriptive statistics + skewness/kurtosis for numeric data."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return {"numeric_columns": [], "describe": {}, "skewness": {}, "kurtosis": {}}

    desc = numeric.describe().replace([np.inf, -np.inf], np.nan)

    return {
        "numeric_columns": list(map(str, numeric.columns)),
        "describe": desc.round(4).to_dict(),
        "skewness": numeric.skew(numeric_only=True).round(4).to_dict(),
        "kurtosis": numeric.kurtosis(numeric_only=True).round(4).to_dict(),
    }


def correlation_analysis(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Return a numeric correlation matrix."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    return numeric.corr(method=method)


def hypothesis_test(
    df: pd.DataFrame,
    *,
    value_column: str | None = None,
    group_column: str | None = None,
    categorical_column_a: str | None = None,
    categorical_column_b: str | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Run common hypothesis tests.

    Supported:
    1. Independent two-sample Welch t-test:
       value_column numeric + group_column with exactly two groups.
    2. Chi-square independence test:
       categorical_column_a + categorical_column_b.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")

    if value_column and group_column:
        if value_column not in df.columns or group_column not in df.columns:
            raise ValueError("The selected hypothesis-test columns do not exist.")

        work = df[[value_column, group_column]].dropna()
        groups = list(work[group_column].unique())

        if len(groups) != 2:
            raise ValueError(
                "Welch's t-test requires exactly two groups in group_column."
            )

        a = pd.to_numeric(
            work.loc[work[group_column] == groups[0], value_column],
            errors="coerce",
        ).dropna()
        b = pd.to_numeric(
            work.loc[work[group_column] == groups[1], value_column],
            errors="coerce",
        ).dropna()

        if len(a) < 2 or len(b) < 2:
            raise ValueError("Each group needs at least two numeric observations.")

        statistic, p_value = stats.ttest_ind(a, b, equal_var=False)

        return {
            "test": "Welch independent two-sample t-test",
            "null_hypothesis": f"Mean({groups[0]}) = Mean({groups[1]})",
            "alternative": "The two group means are different.",
            "statistic": round(float(statistic), 6),
            "p_value": round(float(p_value), 6),
            "alpha": alpha,
            "significant": bool(p_value < alpha),
            "group_1": str(groups[0]),
            "group_2": str(groups[1]),
            "group_1_n": int(len(a)),
            "group_2_n": int(len(b)),
        }

    if categorical_column_a and categorical_column_b:
        if (
            categorical_column_a not in df.columns
            or categorical_column_b not in df.columns
        ):
            raise ValueError("The selected categorical columns do not exist.")

        table = pd.crosstab(
            df[categorical_column_a],
            df[categorical_column_b],
        )
        if table.shape[0] < 2 or table.shape[1] < 2:
            raise ValueError("Chi-square requires at least two levels in each variable.")

        statistic, p_value, dof, _ = stats.chi2_contingency(table)

        return {
            "test": "Chi-square test of independence",
            "null_hypothesis": "The two categorical variables are independent.",
            "alternative": "The two categorical variables are associated.",
            "statistic": round(float(statistic), 6),
            "p_value": round(float(p_value), 6),
            "degrees_of_freedom": int(dof),
            "alpha": alpha,
            "significant": bool(p_value < alpha),
            "contingency_table": table.to_dict(),
        }

    raise ValueError(
        "Provide either value_column + group_column, "
        "or categorical_column_a + categorical_column_b."
    )


def pca_analysis(
    df: pd.DataFrame,
    n_components: int = 2,
) -> dict[str, Any]:
    """
    Standardize numeric columns and run PCA.

    Returns transformed rows, component loadings and explained variance.
    """
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] < 2:
        raise ValueError("PCA requires at least two numeric columns.")

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0)

    n_components = int(n_components)
    n_components = max(1, min(n_components, numeric.shape[1], numeric.shape[0]))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)

    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(scaled)

    names = [f"PC{i + 1}" for i in range(n_components)]
    transformed_df = pd.DataFrame(
        transformed,
        columns=names,
        index=df.index,
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=numeric.columns,
        columns=names,
    )

    return {
        "transformed": transformed_df,
        "loadings": loadings,
        "explained_variance_ratio": [
            round(float(x), 6)
            for x in pca.explained_variance_ratio_
        ],
        "total_explained_variance": round(
            float(pca.explained_variance_ratio_.sum()),
            6,
        ),
        "components": int(n_components),
    }


def to_excel_bytes(
    df: pd.DataFrame,
    *,
    quality: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    correlation: pd.DataFrame | None = None,
) -> bytes:
    """
    Create a multi-sheet Excel workbook for analysts/BI users.
    The returned bytes can be sent directly with Flask send_file.
    """
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Cleaned_Data", index=False)

        if quality is None:
            quality = data_quality_report(df)

        quality_rows = []
        for key, value in quality.items():
            if isinstance(value, (dict, list)):
                value = str(value)
            quality_rows.append({"metric": key, "value": value})

        pd.DataFrame(quality_rows).to_excel(
            writer,
            sheet_name="Data_Quality",
            index=False,
        )

        if profile is None:
            profile = statistical_profile(df)

        if profile.get("describe"):
            pd.DataFrame(profile["describe"]).to_excel(
                writer,
                sheet_name="Statistics",
            )

        if correlation is None:
            correlation = correlation_analysis(df)

        if not correlation.empty:
            correlation.to_excel(
                writer,
                sheet_name="Correlation",
            )

    output.seek(0)
    return output.getvalue()


def tableau_ready_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a clean, flat table suitable for Tableau/Power BI ingestion.
    No styling or dashboard-specific objects are embedded.
    """
    out = df.copy()
    out.columns = [
        str(c).strip().replace(" ", "_").replace("/", "_")
        for c in out.columns
    ]
    return out
