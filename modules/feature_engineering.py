"""
modules/feature_engineering.py — MODULE 9
Handles feature transformation tasks:
- One-Hot Encoding (Categorical to Binary Columns)
- Label Encoding (Categorical to Numeric)
- Standard Scaling (Numerical Normalization)
- Automatic per-column encoding recommendations
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Thresholds mirror the automatic cardinality guard used during model
# training (modules/model_training.py), so the recommendation shown here
# matches what the training pipeline would actually do by default.
ONE_HOT_MAX_UNIQUE = 15
LABEL_MAX_UNIQUE = 50
LABEL_MAX_UNIQUE_RATIO = 0.5


def apply_one_hot_encoding(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Applies One-Hot Encoding to the given columns - each category becomes
    its own binary (0/1) column. Best for LOW-cardinality columns (a
    handful of distinct categories, e.g. "State" or "Gender"), since
    each unique value adds a whole new column.
    """
    df_transformed = df.copy()

    valid_cols = [c for c in columns if c in df_transformed.columns]

    if not valid_cols:
        return df_transformed

    dummies = pd.get_dummies(
        df_transformed[valid_cols].astype(str),
        prefix=valid_cols,
    ).astype(int)  # 0/1 ints, not bool - the training pipeline's
                   # imputer doesn't accept bool dtype directly.

    df_transformed = df_transformed.drop(columns=valid_cols)
    df_transformed = pd.concat([df_transformed, dummies], axis=1)

    return df_transformed


def apply_label_encoding(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Applies Label Encoding to the list of columns.
    Converts string/categorical data into integers.
    """
    df_transformed = df.copy()
    le = LabelEncoder()
    
    for col in columns:
        if col in df_transformed.columns:
            # Drop NaN just in case, though they should be handled in Module 6
            df_transformed[col] = le.fit_transform(df_transformed[col].astype(str))
            
    return df_transformed


def apply_standard_scaling(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Applies Standard Scaling (Mean=0, Std=1) to the list of columns.
    Helps models like Linear Regression or SVM perform better.
    """
    df_transformed = df.copy()
    scaler = StandardScaler()
    
    if not columns:
        return df_transformed
        
    # Only scale columns that actually exist and are numeric
    valid_cols = [c for c in columns if c in df_transformed.columns]
    
    if valid_cols:
        df_transformed[valid_cols] = scaler.fit_transform(df_transformed[valid_cols])
        
    return df_transformed


def recommend_encoding(df: pd.DataFrame) -> dict:
    """
    Recommends an encoding strategy for each categorical column, based
    on how many distinct values it has:

      - <= 15 unique values   → One-Hot Encoding (few enough categories
                                 that a binary column per category is
                                 cheap and works well with every model)
      - 16-50 unique values,
        and < 50% of rows     → Label Encoding (too many for one-hot to
                                 be practical, but still a real,
                                 repeating category)
      - otherwise             → Avoid (this looks like free text or a
                                 near-unique ID - e.g. a name, a message,
                                 a row identifier. Encoding it numerically
                                 doesn't create a meaningful feature and
                                 can let models "memorize" exact rows
                                 instead of learning a real pattern.)

    Returns {column_name: {"unique": n, "recommendation": "one_hot" |
    "label" | "avoid", "reason": "..."}}
    """
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    n_rows = len(df)
    recommendations = {}

    for col in categorical_cols:

        nunique = df[col].nunique(dropna=True)

        is_too_high_cardinality = (
            nunique > LABEL_MAX_UNIQUE
            and n_rows > 0
            and nunique > LABEL_MAX_UNIQUE_RATIO * n_rows
        )

        if is_too_high_cardinality:
            recommendation = "avoid"
            reason = (
                f"{nunique} unique values across {n_rows} rows - looks "
                "like free text or a near-unique ID, not a real category. "
                "Encoding it numerically won't produce a meaningful "
                "feature."
            )
        elif nunique <= ONE_HOT_MAX_UNIQUE:
            recommendation = "one_hot"
            reason = (
                f"Only {nunique} unique values - few enough that "
                "One-Hot Encoding works cleanly."
            )
        elif nunique <= LABEL_MAX_UNIQUE:
            recommendation = "label"
            reason = (
                f"{nunique} unique values - too many for One-Hot "
                "Encoding to be practical, but still a real repeating "
                "category, so Label Encoding fits better."
            )
        else:
            # High-ish cardinality but still under 50% of rows - treat
            # like the label-encoding case rather than outright avoid.
            recommendation = "label"
            reason = (
                f"{nunique} unique values - Label Encoding keeps this "
                "usable without exploding into hundreds of columns."
            )

        recommendations[col] = {
            "unique": int(nunique),
            "recommendation": recommendation,
            "reason": reason,
        }

    return recommendations


def recommend_scaling(df: pd.DataFrame) -> dict:
    """
    Recommends whether each NUMERIC column should be Standard Scaled.

    Scaling makes sense for genuine continuous numeric features (e.g.
    "R&D Spend", "Age"). It doesn't make sense for 0/1 columns produced
    by One-Hot Encoding (or any naturally binary column) - scaling a
    binary indicator doesn't help distance-based models the way it
    helps continuous features, and just adds noise/confusion. A column
    is treated as "binary / skip" if every non-null value it has is
    either 0 or 1.

    Returns {column_name: {"recommend": bool, "reason": "..."}}
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns

    recommendations = {}

    for col in numeric_cols:

        non_null = df[col].dropna()

        is_binary = (
            len(non_null) > 0
            and set(non_null.unique()).issubset({0, 1})
        )

        if is_binary:
            recommendations[col] = {
                "recommend": False,
                "reason": (
                    "Looks like a 0/1 column (e.g. from One-Hot "
                    "Encoding) - scaling a binary indicator doesn't "
                    "help and just adds noise."
                ),
            }
        else:
            recommendations[col] = {
                "recommend": True,
                "reason": "Continuous numeric feature - scaling helps most models train better.",
            }

    return recommendations


def get_feature_info(df: pd.DataFrame) -> dict:
    """Returns lists of categorical and numeric columns for selection,
    plus automatic encoding and scaling recommendations."""
    return {
        'categorical': list(df.select_dtypes(include=['object', 'category']).columns),
        'numeric': list(df.select_dtypes(include=['number']).columns),
        'recommendations': recommend_encoding(df),
        'scaling_recommendations': recommend_scaling(df),
    }

