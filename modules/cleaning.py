"""
modules/cleaning.py — MODULE 6, 7, 8

Handles data cleaning tasks:
- Missing Value Imputation
- Outlier Detection & Handling
- Duplicate Removal
- Basic String Consistency
- One-Click Automatic Cleaning
"""

import pandas as pd
import numpy as np


# ============================================================
# MODULE 6 — MISSING VALUES
# ============================================================

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'auto'
) -> pd.DataFrame:
    """
    Imputes missing values.

    Strategies:
    - auto / mean:
        Numeric -> Mean
        Categorical -> Mode

    - median:
        Numeric -> Median
        Categorical -> Mode
    """

    df_cleaned = df.copy()

    numeric_cols = df_cleaned.select_dtypes(
        include=[np.number]
    ).columns

    categorical_cols = df_cleaned.select_dtypes(
        exclude=[np.number]
    ).columns

    # Numeric columns
    for col in numeric_cols:

        if df_cleaned[col].isnull().any():

            if strategy == 'median':
                fill_value = df_cleaned[col].median()
            else:
                fill_value = df_cleaned[col].mean()

            # If mean/median is NaN because entire column is empty
            if pd.isna(fill_value):
                fill_value = 0

            df_cleaned[col] = df_cleaned[col].fillna(fill_value)

    # Categorical columns
    for col in categorical_cols:

        if df_cleaned[col].isnull().any():

            mode_series = df_cleaned[col].mode()

            if not mode_series.empty:
                fill_value = mode_series.iloc[0]
            else:
                fill_value = 'Unknown'

            df_cleaned[col] = df_cleaned[col].fillna(fill_value)

    return df_cleaned


def get_missing_stats(df: pd.DataFrame) -> dict:
    """Returns missing values per column."""

    missing = df.isnull().sum()

    return missing[missing > 0].to_dict()


# ============================================================
# MODULE 7 — OUTLIERS
# ============================================================

def detect_outliers(df: pd.DataFrame) -> dict:
    """
    Detects outliers using the IQR method.

    Returns:
        {
            column_name: number_of_outliers
        }
    """

    outlier_stats = {}

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    for col in numeric_cols:

        # Ignore columns with too few valid values
        if df[col].dropna().shape[0] < 4:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        # Constant column
        if IQR == 0:
            continue

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        mask = (
            (df[col] < lower_bound) |
            (df[col] > upper_bound)
        )

        count = int(mask.sum())

        if count > 0:
            outlier_stats[col] = count

    return outlier_stats


def handle_outliers(
    df: pd.DataFrame,
    strategy: str = 'remove'
) -> pd.DataFrame:
    """
    Handles numeric outliers.

    Strategies:
    - remove:
        Removes rows containing outliers.

    - cap:
        Caps values at IQR boundaries.
    """

    df_cleaned = df.copy()

    numeric_cols = df_cleaned.select_dtypes(
        include=[np.number]
    ).columns

    for col in numeric_cols:

        if df_cleaned[col].dropna().shape[0] < 4:
            continue

        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)

        IQR = Q3 - Q1

        if IQR == 0:
            continue

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if strategy == 'remove':

            df_cleaned = df_cleaned[
                (df_cleaned[col] >= lower_bound) &
                (df_cleaned[col] <= upper_bound)
            ]

        elif strategy == 'cap':

            df_cleaned[col] = df_cleaned[col].clip(
                lower=lower_bound,
                upper=upper_bound
            )

    return df_cleaned


# ============================================================
# MODULE 8 — DUPLICATES
# ============================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate rows."""

    return df.drop_duplicates().reset_index(drop=True)


def fix_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes basic string inconsistencies.

    Currently:
    - Removes leading/trailing whitespace.
    """

    df_cleaned = df.copy()

    object_cols = df_cleaned.select_dtypes(
        include=['object']
    ).columns

    for col in object_cols:

        df_cleaned[col] = df_cleaned[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    return df_cleaned


# ============================================================
# MODULE 6–8 — ONE CLICK AUTO CLEANING
# ============================================================

def auto_clean_dataset(
    df: pd.DataFrame,
    outlier_strategy: str = 'cap'
):
    """
    Automatically cleans the complete dataset in one operation.

    Cleaning order:

    1. Fix string inconsistencies
    2. Remove duplicate rows
    3. Fill missing values
    4. Handle numeric outliers

    Returns:
        cleaned_dataframe, cleaning_report
    """

    df_cleaned = df.copy()

    # --------------------------------------------------------
    # Initial statistics
    # --------------------------------------------------------

    initial_rows = len(df_cleaned)
    initial_columns = len(df_cleaned.columns)

    initial_missing = int(
        df_cleaned.isnull().sum().sum()
    )

    initial_duplicates = int(
        df_cleaned.duplicated().sum()
    )

    initial_outliers_dict = detect_outliers(df_cleaned)

    initial_outliers = sum(
        initial_outliers_dict.values()
    )

    # --------------------------------------------------------
    # STEP 1 — Fix string inconsistencies
    # --------------------------------------------------------

    df_cleaned = fix_inconsistencies(df_cleaned)

    # --------------------------------------------------------
    # STEP 2 — Remove duplicates
    # --------------------------------------------------------

    df_cleaned = remove_duplicates(df_cleaned)

    duplicates_removed = (
        initial_rows - len(df_cleaned)
    )

    # --------------------------------------------------------
    # STEP 3 — Handle missing values
    # --------------------------------------------------------

    missing_before = int(
        df_cleaned.isnull().sum().sum()
    )

    df_cleaned = handle_missing_values(
        df_cleaned,
        strategy='median'
    )

    missing_after = int(
        df_cleaned.isnull().sum().sum()
    )

    missing_filled = (
        missing_before - missing_after
    )

    # --------------------------------------------------------
    # STEP 4 — Handle outliers
    # --------------------------------------------------------

    outliers_before_dict = detect_outliers(
        df_cleaned
    )

    outliers_before = sum(
        outliers_before_dict.values()
    )

    df_cleaned = handle_outliers(
        df_cleaned,
        strategy=outlier_strategy
    )

    # With capping, count how many values were affected
    if outlier_strategy == 'cap':

        outliers_after_dict = detect_outliers(
            df_cleaned
        )

        outliers_after = sum(
            outliers_after_dict.values()
        )

        outliers_handled = max(
            0,
            outliers_before - outliers_after
        )

    else:

        outliers_after = sum(
            detect_outliers(df_cleaned).values()
        )

        outliers_handled = max(
            0,
            outliers_before - outliers_after
        )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    final_rows = len(df_cleaned)
    final_columns = len(df_cleaned.columns)

    final_missing = int(
        df_cleaned.isnull().sum().sum()
    )

    final_duplicates = int(
        df_cleaned.duplicated().sum()
    )

    final_outliers = sum(
        detect_outliers(df_cleaned).values()
    )

    # --------------------------------------------------------
    # Cleaning report
    # --------------------------------------------------------

    report = {
        'rows_before': initial_rows,
        'rows_after': final_rows,

        'columns_before': initial_columns,
        'columns_after': final_columns,

        'missing_before': initial_missing,
        'missing_filled': missing_filled,
        'missing_after': final_missing,

        'duplicates_before': initial_duplicates,
        'duplicates_removed': duplicates_removed,
        'duplicates_after': final_duplicates,

        'outliers_before': initial_outliers,
        'outliers_handled': outliers_handled,
        'outliers_after': final_outliers,

        'outlier_strategy': outlier_strategy,

        'cleaning_complete': True
    }

    return df_cleaned, report