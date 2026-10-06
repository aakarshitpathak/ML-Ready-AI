"""
ML Ready AI — Unified Model Training & Benchmarking

Single source of truth for:
- Model definitions
- Hyperparameters
- Preprocessing
- Train/test splitting
- Cross-validation
- Model comparison
- Manual model training
- Final evaluation
- Feature importance
- Classification / Regression / Clustering

The Flask application can continue using:

    train_model_logic(df, problem_type, algorithm, target_col)

The benchmark and manual-training paths intentionally use the SAME:
- preprocessing
- train/test split
- random state
- CV strategy
- model definitions
- hyperparameters
- evaluation metrics
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans


# ============================================================
# OPTIONAL XGBOOST
# ============================================================

try:
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True

except Exception:
    XGBOOST_AVAILABLE = False


warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# GLOBAL SETTINGS
# ============================================================

DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20
DEFAULT_CV_FOLDS = 5


# ============================================================
# ALGORITHM ALIASES
# ============================================================

CLASSIFICATION_ALIASES = {
    "logistic_regression": "logistic_regression",
    "logistic": "logistic_regression",

    "decision_tree": "decision_tree",
    "decision_tree_classifier": "decision_tree",

    "random_forest": "random_forest",
    "random_forest_classifier": "random_forest",

    "gradient_boosting": "gradient_boosting",
    "gradient_boosting_classifier": "gradient_boosting",

    "svm": "svm",
    "svc": "svm",

    "knn": "knn",

    "xgboost": "xgboost",
    "xgboost_classifier": "xgboost",
}


REGRESSION_ALIASES = {
    "linear_regression": "linear_regression",
    "linear": "linear_regression",

    "decision_tree": "decision_tree_regressor",
    "decision_tree_regressor": "decision_tree_regressor",

    "random_forest": "random_forest_regressor",
    "random_forest_regressor": "random_forest_regressor",

    "gradient_boosting": "gradient_boosting_regressor",
    "gradient_boosting_regressor": "gradient_boosting_regressor",

    "svm": "svm_regressor",
    "svr": "svm_regressor",
    "svm_regressor": "svm_regressor",

    "knn": "knn_regressor",
    "knn_regressor": "knn_regressor",

    "xgboost": "xgboost_regressor",
    "xgboost_regressor": "xgboost_regressor",
}


# ============================================================
# DISPLAY NAMES
# ============================================================

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "svm": "SVM",
    "knn": "KNN",
    "xgboost": "XGBoost",

    "linear_regression": "Linear Regression",
    "decision_tree_regressor": "Decision Tree",
    "random_forest_regressor": "Random Forest",
    "gradient_boosting_regressor": "Gradient Boosting",
    "svm_regressor": "SVR",
    "knn_regressor": "KNN",
    "xgboost_regressor": "XGBoost",
}


# ============================================================
# MODEL REGISTRY
# ============================================================

def get_supported_algorithms(problem_type: str) -> list[str]:
    """
    Returns the SAME algorithm set used by:
    - benchmark
    - manual training
    """

    problem_type = _normalise_problem(problem_type)

    if problem_type == "classification":

        algorithms = [
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "gradient_boosting",
            "knn",
            "svm",
        ]

        if XGBOOST_AVAILABLE:
            algorithms.append("xgboost")

        return algorithms

    if problem_type == "regression":

        algorithms = [
            "linear_regression",
            "decision_tree_regressor",
            "random_forest_regressor",
            "gradient_boosting_regressor",
            "knn_regressor",
            "svm_regressor",
        ]

        if XGBOOST_AVAILABLE:
            algorithms.append("xgboost_regressor")

        return algorithms

    return ["kmeans"]


# ============================================================
# NORMALISE PROBLEM TYPE
# ============================================================

def _normalise_problem(problem_type: str) -> str:

    value = str(problem_type or "").strip().lower()

    if value in {
        "classification",
        "classify",
        "classifier",
    }:
        return "classification"

    if value in {
        "regression",
        "regress",
        "regressor",
    }:
        return "regression"

    if value in {
        "clustering",
        "cluster",
        "unsupervised",
    }:
        return "clustering"

    raise ValueError(
        "Problem type must be classification, regression, or clustering."
    )


# ============================================================
# ONE-HOT ENCODER
# ============================================================

def _make_one_hot_encoder() -> OneHotEncoder:

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(
    X: pd.DataFrame,
    scale_numeric: bool = True,
) -> ColumnTransformer:

    numeric_cols = list(
        X.select_dtypes(include=[np.number]).columns
    )

    all_categorical_cols = list(
        X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
    )

    # Guard against very high-cardinality categorical columns (a free
    # text field, a near-unique ID, a message body, etc.). One-hot
    # encoding one of these:
    #   1) blows up into thousands of columns and makes training
    #      painfully slow (or effectively hangs) for no benefit, and
    #   2) lets models "memorize" via exact-duplicate rows landing in
    #      both the train and test/CV split rather than genuinely
    #      learning a pattern - producing misleadingly perfect scores
    #      that don't reflect real predictive power.
    # Such columns are dropped from automatic training rather than
    # silently one-hot encoded. (Using them meaningfully would require
    # real NLP/text-vectorization, which this app doesn't do.)
    n_rows = len(X)

    categorical_cols = []
    dropped_high_cardinality_cols = []

    for col in all_categorical_cols:

        nunique = X[col].nunique(dropna=True)

        is_high_cardinality = (
            nunique > 50
            and n_rows > 0
            and nunique > 0.5 * n_rows
        )

        if is_high_cardinality:
            dropped_high_cardinality_cols.append(col)
        else:
            categorical_cols.append(col)

    if not numeric_cols and not categorical_cols:

        if dropped_high_cardinality_cols:

            raise ValueError(
                "No usable feature columns were found. The remaining "
                "column(s) - "
                f"{', '.join(dropped_high_cardinality_cols)} - look like "
                "free text or unique IDs (too many distinct values to "
                "use as a category), so they were excluded automatically. "
                "This app doesn't support raw text (NLP) as a feature - "
                "try a dataset without free-text columns, or engineer a "
                "feature from that text first (e.g. message length, "
                "word count) before training."
            )

        raise ValueError(
            "No usable feature columns were found."
        )

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(strategy="median"),
        )
    ]

    if scale_numeric:

        numeric_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    transformers = []

    if numeric_cols:

        transformers.append(
            (
                "numeric",
                Pipeline(numeric_steps),
                numeric_cols,
            )
        )

    if categorical_cols:

        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            _make_one_hot_encoder(),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ============================================================
# CLASSIFICATION MODEL FACTORY
# ============================================================

def _make_classifier(
    name: str,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_classes: int | None = None,
):

    if name == "logistic_regression":

        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )

    if name == "decision_tree":

        return DecisionTreeClassifier(
            random_state=random_state,
            class_weight="balanced",
        )

    if name == "random_forest":

        return RandomForestClassifier(
            n_estimators=250,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )

    if name == "gradient_boosting":

        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state,
        )

    if name == "knn":

        return KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
        )

    if name == "svm":

        return SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=random_state,
            class_weight="balanced",
        )

    if name == "xgboost":

        if not XGBOOST_AVAILABLE:

            raise ImportError(
                "XGBoost is not installed."
            )

        multiclass = (
            n_classes is not None
            and n_classes > 2
        )

        params = {
            "n_estimators": 250,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": random_state,
            "n_jobs": -1,
            "eval_metric": (
                "mlogloss"
                if multiclass
                else "logloss"
            ),
        }

        if multiclass:

            params["objective"] = "multi:softprob"
            params["num_class"] = n_classes

        else:

            params["objective"] = "binary:logistic"

        return XGBClassifier(**params)

    raise ValueError(
        f"Unsupported classification algorithm: {name}"
    )


# ============================================================
# REGRESSION MODEL FACTORY
# ============================================================

def _make_regressor(
    name: str,
    random_state: int = DEFAULT_RANDOM_STATE,
):

    if name == "linear_regression":

        return LinearRegression()

    if name == "decision_tree_regressor":

        return DecisionTreeRegressor(
            random_state=random_state,
            max_depth=None,
        )

    if name == "random_forest_regressor":

        return RandomForestRegressor(
            n_estimators=250,
            random_state=random_state,
            n_jobs=-1,
        )

    if name == "gradient_boosting_regressor":

        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state,
        )

    if name == "knn_regressor":

        return KNeighborsRegressor(
            n_neighbors=5,
            weights="distance",
        )

    if name == "svm_regressor":

        return SVR(
            kernel="rbf",
            C=10.0,
            epsilon=0.1,
        )

    if name == "xgboost_regressor":

        if not XGBOOST_AVAILABLE:

            raise ImportError(
                "XGBoost is not installed."
            )

        return XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(
        f"Unsupported regression algorithm: {name}"
    )


# ============================================================
# ALGORITHM NORMALISATION
# ============================================================

def normalise_algorithm(
    algorithm: str,
    problem_type: str,
) -> str:

    algorithm = str(
        algorithm or ""
    ).strip().lower()

    problem_type = _normalise_problem(
        problem_type
    )

    if problem_type == "classification":

        canonical = CLASSIFICATION_ALIASES.get(
            algorithm
        )

    elif problem_type == "regression":

        canonical = REGRESSION_ALIASES.get(
            algorithm
        )

    else:

        return algorithm

    if canonical is None:

        raise ValueError(
            f"Unsupported {problem_type} algorithm: {algorithm}"
        )

    return canonical


# ============================================================
# DATA PREPARATION
# ============================================================

def _prepare_supervised_data(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:

    if not isinstance(df, pd.DataFrame):

        raise ValueError(
            "Dataset must be a pandas DataFrame."
        )

    if not target_col:

        raise ValueError(
            "Please select a target column."
        )

    if target_col not in df.columns:

        raise ValueError(
            f"Target column '{target_col}' not found in dataset."
        )

    data = df.copy()

    # Remove rows where target is missing.
    data = data.dropna(
        subset=[target_col]
    )

    if data.empty:

        raise ValueError(
            "The target column contains no usable values."
        )

    X = data.drop(
        columns=[target_col]
    )

    y = data[target_col].copy()

    # Remove constant features.
    constant_cols = [
        column
        for column in X.columns
        if X[column].nunique(
            dropna=False
        ) <= 1
    ]

    if constant_cols:

        X = X.drop(
            columns=constant_cols
        )

    # Remove obvious ID columns.
    id_like = []

    for column in X.columns:

        name = str(
            column
        ).strip().lower()

        unique_count = X[column].nunique(
            dropna=False
        )

        if (
            name
            in {
                "id",
                "index",
                "customer_id",
                "user_id",
                "record_id",
            }
            and unique_count
            >= max(
                10,
                int(len(X) * 0.95),
            )
        ):

            id_like.append(column)

    if id_like:

        X = X.drop(
            columns=id_like
        )

    if X.shape[1] == 0:

        raise ValueError(
            "No usable features remain after data-quality checks."
        )

    return X, y


# ============================================================
# TARGET PREPARATION
# ============================================================

def _prepare_target(
    y: pd.Series,
    problem_type: str,
):

    if problem_type == "classification":

        if y.nunique(dropna=True) < 2:

            raise ValueError(
                "Classification requires at least two target classes."
            )

        encoder = LabelEncoder()

        encoded = encoder.fit_transform(
            y.astype(str)
        )

        return (
            pd.Series(
                encoded,
                index=y.index,
                name=y.name,
            ),
            encoder,
        )

    # Regression target must be numeric.
    numeric_y = pd.to_numeric(
        y,
        errors="coerce",
    )

    valid = numeric_y.notna()

    if not valid.all():

        invalid_count = int(
            (~valid).sum()
        )

        raise ValueError(
            "Regression target must be numeric. "
            f"{invalid_count} target value(s) could not be converted."
        )

    return numeric_y, None


# ============================================================
# FIXED TRAIN / TEST SPLIT
# ============================================================

def _split_supervised_data(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
):

    split_kwargs = {
        "test_size": test_size,
        "random_state": random_state,
    }

    if problem_type == "classification":

        try:

            return train_test_split(
                X,
                y,
                stratify=y,
                **split_kwargs,
            )

        except ValueError:

            return train_test_split(
                X,
                y,
                **split_kwargs,
            )

    return train_test_split(
        X,
        y,
        **split_kwargs,
    )


# ============================================================
# CROSS-VALIDATION STRATEGY
# ============================================================

def _get_cv(
    y_train: pd.Series,
    problem_type: str,
    requested_folds: int = DEFAULT_CV_FOLDS,
):

    requested_folds = max(
        2,
        int(requested_folds),
    )

    if problem_type == "classification":

        class_counts = pd.Series(
            y_train
        ).value_counts()

        if class_counts.empty:

            return None

        folds = min(
            requested_folds,
            int(class_counts.min()),
        )

        if folds < 2:

            return None

        return StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=DEFAULT_RANDOM_STATE,
        )

    folds = min(
        requested_folds,
        len(y_train),
    )

    if folds < 2:

        return None

    return KFold(
        n_splits=folds,
        shuffle=True,
        random_state=DEFAULT_RANDOM_STATE,
    )


# ============================================================
# PIPELINE
# ============================================================

def _build_pipeline(
    X: pd.DataFrame,
    estimator,
) -> Pipeline:

    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    X,
                    scale_numeric=True,
                ),
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


# ============================================================
# CROSS-VALIDATION
# ============================================================

def _cross_validate(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    cv,
):

    if cv is None:

        return None, None

    scoring = (
        "accuracy"
        if problem_type == "classification"
        else "r2"
    )

    try:

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        # A fold can legitimately produce NaN (e.g. R^2 is undefined for
        # a test fold with fewer than 2 samples, or with a constant
        # target). sklearn doesn't raise for this - it just puts NaN in
        # the array - so np.mean()/std() would silently propagate NaN as
        # "the" CV score while still reporting Success. Treat that as
        # "CV unavailable" instead, so callers fall back to the
        # held-out test score the same way they already do when CV
        # can't be run at all (too few rows / too few of a class).
        if np.isnan(scores).any():

            return None, None

        return (
            float(scores.mean()),
            float(scores.std()),
        )

    except Exception as exc:

        raise RuntimeError(
            f"Cross-validation failed: {type(exc).__name__}: {exc}"
        ) from exc


# ============================================================
# TEST METRICS
# ============================================================

def _classification_metrics(
    model,
    X_test,
    y_test,
) -> dict[str, Any]:

    y_pred = model.predict(
        X_test
    )

    result = {
        "accuracy": round(
            float(
                accuracy_score(
                    y_test,
                    y_pred,
                )
            ),
            4,
        ),

        "precision": round(
            float(
                precision_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),

        "recall": round(
            float(
                recall_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),

        "f1_score": round(
            float(
                f1_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),
    }

    try:

        if hasattr(
            model,
            "predict_proba",
        ):

            probabilities = model.predict_proba(
                X_test
            )

            if probabilities.shape[1] == 2:

                auc = roc_auc_score(
                    y_test,
                    probabilities[:, 1],
                )

            else:

                auc = roc_auc_score(
                    y_test,
                    probabilities,
                    multi_class="ovr",
                    average="weighted",
                )

            result["roc_auc"] = round(
                float(auc),
                4,
            )

        elif hasattr(
            model,
            "decision_function",
        ):

            decision = model.decision_function(
                X_test
            )

            if np.ndim(decision) == 1:

                auc = roc_auc_score(
                    y_test,
                    decision,
                )

            else:

                auc = roc_auc_score(
                    y_test,
                    decision,
                    multi_class="ovr",
                    average="weighted",
                )

            result["roc_auc"] = round(
                float(auc),
                4,
            )

    except Exception:

        pass

    return result


def _regression_metrics(
    model,
    X_test,
    y_test,
) -> dict[str, Any]:

    y_pred = model.predict(
        X_test
    )

    mse = float(
        mean_squared_error(
            y_test,
            y_pred,
        )
    )

    return {
        "r2": round(
            float(
                r2_score(
                    y_test,
                    y_pred,
                )
            ),
            4,
        ),

        "mse": round(
            mse,
            4,
        ),

        "rmse": round(
            float(
                np.sqrt(mse)
            ),
            4,
        ),

        "mae": round(
            float(
                mean_absolute_error(
                    y_test,
                    y_pred,
                )
            ),
            4,
        ),
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def _feature_importance(
    pipeline: Pipeline,
) -> dict[str, float]:

    model = pipeline.named_steps[
        "model"
    ]

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    if not hasattr(
        model,
        "feature_importances_",
    ):

        return {}

    try:

        names = list(
            preprocessor.get_feature_names_out()
        )

        importances = np.asarray(
            model.feature_importances_,
            dtype=float,
        )

        if len(names) != len(importances):

            return {}

        order = np.argsort(
            importances
        )[::-1][:20]

        return {
            str(names[index]): round(
                float(importances[index]),
                6,
            )
            for index in order
        }

    except Exception:

        return {}


# ============================================================
# OPTIONAL HYPERPARAMETER TUNING
# ============================================================

def _tune_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    algorithm: str,
    cv,
):

    if cv is None:

        pipeline.fit(
            X_train,
            y_train,
        )

        return pipeline, {}

    if problem_type == "classification":

        grids = {
            "logistic_regression": {
                "model__C": [
                    0.1,
                    1.0,
                    10.0,
                ],
            },

            "decision_tree": {
                "model__max_depth": [
                    None,
                    5,
                    10,
                    20,
                ],
                "model__min_samples_split": [
                    2,
                    5,
                ],
            },

            "random_forest": {
                "model__n_estimators": [
                    150,
                    250,
                ],
                "model__max_depth": [
                    None,
                    10,
                    20,
                ],
                "model__min_samples_split": [
                    2,
                    5,
                ],
            },

            "gradient_boosting": {
                "model__n_estimators": [
                    100,
                    200,
                ],
                "model__learning_rate": [
                    0.05,
                    0.1,
                ],
            },

            "knn": {
                "model__n_neighbors": [
                    3,
                    5,
                    7,
                ],
            },

            "svm": {
                "model__C": [
                    0.5,
                    1.0,
                    10.0,
                ],
                "model__gamma": [
                    "scale",
                    "auto",
                ],
            },
        }

    else:

        grids = {
            "linear_regression": {},

            "decision_tree_regressor": {
                "model__max_depth": [
                    None,
                    5,
                    10,
                    20,
                ],
                "model__min_samples_split": [
                    2,
                    5,
                ],
            },

            "random_forest_regressor": {
                "model__n_estimators": [
                    150,
                    250,
                ],
                "model__max_depth": [
                    None,
                    10,
                    20,
                ],
                "model__min_samples_split": [
                    2,
                    5,
                ],
            },

            "gradient_boosting_regressor": {
                "model__n_estimators": [
                    100,
                    200,
                ],
                "model__learning_rate": [
                    0.05,
                    0.1,
                ],
            },

            "knn_regressor": {
                "model__n_neighbors": [
                    3,
                    5,
                    7,
                ],
            },

            "svm_regressor": {
                "model__C": [
                    1.0,
                    10.0,
                    50.0,
                ],
                "model__epsilon": [
                    0.01,
                    0.1,
                    0.2,
                ],
            },
        }

    grid = grids.get(
        algorithm,
        {},
    )

    if not grid:

        pipeline.fit(
            X_train,
            y_train,
        )

        return pipeline, {}

    scoring = (
        "accuracy"
        if problem_type == "classification"
        else "r2"
    )

    search = GridSearchCV(
        pipeline,
        grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )

    search.fit(
        X_train,
        y_train,
    )

    return (
        search.best_estimator_,
        {
            "best_cv_score": round(
                float(
                    search.best_score_
                ),
                4,
            ),
            "best_params": search.best_params_,
        },
    )


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def _train_supervised(
    df: pd.DataFrame,
    problem_type: str,
    algorithm: str,
    target_col: str,
    tune: bool = False,
    cv_folds: int = DEFAULT_CV_FOLDS,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
):

    X, y_raw = _prepare_supervised_data(
        df,
        target_col,
    )

    y, target_encoder = _prepare_target(
        y_raw,
        problem_type,
    )

    canonical = normalise_algorithm(
        algorithm,
        problem_type,
    )

    supported = get_supported_algorithms(
        problem_type
    )

    if canonical not in supported:

        raise ValueError(
            f"Unsupported {problem_type} algorithm: {algorithm}"
        )

    X_train, X_test, y_train, y_test = (
        _split_supervised_data(
            X,
            y,
            problem_type,
            test_size=test_size,
            random_state=random_state,
        )
    )

    if problem_type == "classification":

        estimator = _make_classifier(
            canonical,
            random_state,
            int(y.nunique()),
        )

    else:

        estimator = _make_regressor(
            canonical,
            random_state,
        )

    pipeline = _build_pipeline(
        X,
        estimator,
    )

    cv = _get_cv(
        y_train,
        problem_type,
        cv_folds,
    )

    tuning_info = {}

    if tune:

        pipeline, tuning_info = _tune_pipeline(
            pipeline,
            X_train,
            y_train,
            problem_type,
            canonical,
            cv,
        )

    else:

        pipeline.fit(
            X_train,
            y_train,
        )

    if problem_type == "classification":

        metrics = _classification_metrics(
            pipeline,
            X_test,
            y_test,
        )

    else:

        metrics = _regression_metrics(
            pipeline,
            X_test,
            y_test,
        )

    cv_mean, cv_std = _cross_validate(
        pipeline,
        X_train,
        y_train,
        problem_type,
        cv,
    )

    if problem_type == "classification":

        if cv_mean is not None:

            metrics["cv_accuracy_mean"] = round(
                cv_mean,
                4,
            )

            metrics["cv_accuracy_std"] = round(
                cv_std or 0.0,
                4,
            )

    else:

        if cv_mean is not None:

            metrics["cv_r2_mean"] = round(
                cv_mean,
                4,
            )

            metrics["cv_r2_std"] = round(
                cv_std or 0.0,
                4,
            )

    metrics["training_rows"] = int(
        len(X_train)
    )

    metrics["testing_rows"] = int(
        len(X_test)
    )

    metrics["feature_count"] = int(
        X.shape[1]
    )

    metrics["tuned"] = bool(
        tune
    )

    if tuning_info:

        metrics["best_cv_score"] = (
            tuning_info["best_cv_score"]
        )

        metrics["best_params"] = (
            tuning_info["best_params"]
        )

    if target_encoder is not None:

        pipeline.target_encoder = (
            target_encoder
        )

    metadata = {
        "feature_importance": _feature_importance(
            pipeline
        ),

        "canonical_algorithm": canonical,

        "display_name": DISPLAY_NAMES.get(
            canonical,
            canonical,
        ),

        "target_encoder": target_encoder,

        "target_classes": (
            [
                str(value)
                for value
                in target_encoder.classes_
            ]
            if target_encoder is not None
            else []
        ),

        "cv_folds": (
            cv.n_splits
            if cv is not None
            else 0
        ),

        "random_state": random_state,

        "test_size": test_size,

        "tuning": tuning_info,
    }

    return (
        pipeline,
        metrics,
        X_test,
        y_test,
        metadata,
    )


# ============================================================
# BENCHMARK ALL MODELS
# ============================================================

def benchmark_models(
    df: pd.DataFrame,
    problem_type: str,
    target_col: str,
    cv_folds: int = DEFAULT_CV_FOLDS,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
):

    """
    Benchmark every supported supervised algorithm.

    IMPORTANT:
    - Every model gets the SAME train/test split.
    - Every model gets the SAME preprocessing.
    - Every model gets the SAME CV strategy.
    - Every model uses the SAME model factory as manual training.
    - Winner is selected using mean CV score.
    - Test score is reported separately.
    """

    problem_type = _normalise_problem(
        problem_type
    )

    if problem_type == "clustering":

        return _benchmark_clustering(
            df,
            random_state=random_state,
        )

    if problem_type not in {
        "classification",
        "regression",
    }:

        raise ValueError(
            "Benchmarking is only available for classification, "
            "regression, and clustering."
        )

    X, y_raw = _prepare_supervised_data(
        df,
        target_col,
    )

    y, target_encoder = _prepare_target(
        y_raw,
        problem_type,
    )

    X_train, X_test, y_train, y_test = (
        _split_supervised_data(
            X,
            y,
            problem_type,
            test_size=test_size,
            random_state=random_state,
        )
    )

    cv = _get_cv(
        y_train,
        problem_type,
        cv_folds,
    )

    algorithms = get_supported_algorithms(
        problem_type
    )

    results = []

    fitted_models = {}

    for canonical in algorithms:

        started = time.perf_counter()

        display_name = DISPLAY_NAMES.get(
            canonical,
            canonical,
        )

        try:

            if problem_type == "classification":

                estimator = _make_classifier(
                    canonical,
                    random_state,
                    int(y.nunique()),
                )

            else:

                estimator = _make_regressor(
                    canonical,
                    random_state,
                )

            pipeline = _build_pipeline(
                X,
                estimator,
            )

            cv_mean, cv_std = _cross_validate(
                pipeline,
                X_train,
                y_train,
                problem_type,
                cv,
            )

            # Final fitted model uses exactly the same
            # X_train / y_train used for CV.
            pipeline.fit(
                X_train,
                y_train,
            )

            fitted_models[canonical] = pipeline

            if problem_type == "classification":

                test_metrics = _classification_metrics(
                    pipeline,
                    X_test,
                    y_test,
                )

                primary_test_score = (
                    test_metrics["accuracy"]
                )

            else:

                test_metrics = _regression_metrics(
                    pipeline,
                    X_test,
                    y_test,
                )

                primary_test_score = (
                    test_metrics["r2"]
                )

            # Cross-validation can legitimately be unavailable (dataset too
            # small, or a class with fewer than 2 samples in the training
            # split). The model itself still trained and produced a real
            # held-out test score, so fall back to that instead of
            # discarding a working model as "failed".
            cv_was_available = cv_mean is not None

            if not cv_was_available:

                cv_mean = primary_test_score
                cv_std = 0.0

            results.append(
                {
                    "algorithm": display_name,
                    "algorithm_key": canonical,

                    "mean_score": round(
                        float(cv_mean),
                        4,
                    ),

                    "std_score": round(
                        float(cv_std),
                        4,
                    ),

                    # Backward-compatible names used by Flask UI.
                    "cv_score": round(
                        float(cv_mean),
                        4,
                    ),
                    "cv_std": round(
                        float(cv_std),
                        4,
                    ),

                    "cv_available": cv_was_available,

                    "scores": [],
                    "error": None,

                    "test_score": round(
                        float(
                            primary_test_score
                        ),
                        4,
                    ),

                    "status": (
                        "Success"
                        if cv_was_available
                        else "Success (no CV — dataset too small "
                             "or a class is under-represented; "
                             "ranked by test score instead)"
                    ),

                    "fit_seconds": round(
                        time.perf_counter()
                        - started,
                        3,
                    ),

                    **test_metrics,
                }
            )

        except Exception as exc:

            results.append(
                {
                    "algorithm": display_name,
                    "algorithm_key": canonical,
                    "mean_score": None,
                    "std_score": None,
                    "cv_score": None,
                    "cv_std": None,
                    "scores": [],
                    "test_score": None,
                    "status": f"Failed: {type(exc).__name__}: {exc}",
                    "error": f"{type(exc).__name__}: {exc}",
                    "fit_seconds": round(
                        time.perf_counter()
                        - started,
                        3,
                    ),
                }
            )

    successful = [
        row
        for row in results
        if row.get("mean_score") is not None
    ]

    failed = [
        row
        for row in results
        if row.get("mean_score") is None
    ]

    if not successful:

        raise RuntimeError(
            "No benchmark model could be trained successfully."
        )

    # CV score is the official selection metric.
    successful.sort(
        key=lambda row: row["mean_score"],
        reverse=True,
    )

    results = successful + failed

    winner_row = successful[0]

    winner_key = winner_row[
        "algorithm_key"
    ]

    winner = fitted_models[
        winner_key
    ]

    if target_encoder is not None:

        winner.target_encoder = (
            target_encoder
        )

    if problem_type == "classification":

        final_metrics = _classification_metrics(
            winner,
            X_test,
            y_test,
        )

        final_metrics[
            "cv_accuracy_mean"
        ] = winner_row["mean_score"]

        final_metrics[
            "cv_accuracy_std"
        ] = (
            winner_row["std_score"]
            or 0.0
        )

    else:

        final_metrics = _regression_metrics(
            winner,
            X_test,
            y_test,
        )

        final_metrics[
            "cv_r2_mean"
        ] = winner_row["mean_score"]

        final_metrics[
            "cv_r2_std"
        ] = (
            winner_row["std_score"]
            or 0.0
        )

    final_metrics[
        "training_rows"
    ] = int(len(X_train))

    final_metrics[
        "testing_rows"
    ] = int(len(X_test))

    final_metrics[
        "feature_count"
    ] = int(X.shape[1])

    final_metrics[
        "benchmark_winner"
    ] = winner_key

    final_metrics[
        "benchmark_winner_display"
    ] = winner_row["algorithm"]

    final_metrics[
        "random_state"
    ] = random_state

    final_metrics[
        "test_size"
    ] = test_size

    final_metrics[
        "cv_folds"
    ] = (
        cv.n_splits
        if cv is not None
        else 0
    )

    metadata = {
        "feature_importance": _feature_importance(
            winner
        ),

        "canonical_algorithm": winner_key,

        "display_name": winner_row[
            "algorithm"
        ],

        "target_encoder": target_encoder,

        "target_classes": (
            [
                str(value)
                for value
                in target_encoder.classes_
            ]
            if target_encoder is not None
            else []
        ),

        "benchmark_results": results,

        "random_state": random_state,

        "test_size": test_size,

        "cv_folds": (
            cv.n_splits
            if cv is not None
            else 0
        ),

        "model": winner,

        "data_rows": len(df),

        "target_col": target_col,

        "problem_type": problem_type,
    }

    return (
        winner,
        final_metrics,
        results,
        metadata,
    )


def _benchmark_clustering(
    df: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 8,
    use_pca: bool = False,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """
    "Benchmark" K-Means is only supported clustering algorithm today, so
    there is nothing to compare it against directly. Instead, this tries
    a sensible range of cluster counts (K) and picks the one with the
    best silhouette score - the standard way to auto-select K for
    K-Means when no ground-truth labels exist.
    """

    n_rows = len(df)

    upper_bound = min(
        k_max,
        n_rows - 1,
    )

    if upper_bound < k_min:

        raise ValueError(
            "Not enough rows to benchmark clustering. "
            f"Need at least {k_min + 1} rows, found {n_rows}."
        )

    results = []
    successes = []

    for k in range(k_min, upper_bound + 1):

        started = time.perf_counter()

        try:

            model_bundle, metrics, X_model, labels, meta = _train_clustering(
                df,
                "kmeans",
                n_clusters=k,
                use_pca=use_pca,
                random_state=random_state,
            )

            entry = {
                "algorithm": f"K-Means (k={k})",
                "algorithm_key": "kmeans",
                "n_clusters": k,
                "mean_score": metrics["silhouette"],
                "std_score": 0.0,
                "cv_score": metrics["silhouette"],
                "cv_std": 0.0,
                "cv_available": False,
                "scores": [],
                "error": None,
                "test_score": metrics["silhouette"],
                "status": "Success (ranked by silhouette score)",
                "fit_seconds": round(
                    time.perf_counter() - started,
                    3,
                ),
                **metrics,
            }

            results.append(entry)

            successes.append(
                (entry, model_bundle, metrics, meta)
            )

        except Exception as exc:

            results.append(
                {
                    "algorithm": f"K-Means (k={k})",
                    "algorithm_key": "kmeans",
                    "n_clusters": k,
                    "mean_score": None,
                    "std_score": None,
                    "cv_score": None,
                    "cv_std": None,
                    "cv_available": False,
                    "scores": [],
                    "error": str(exc),
                    "test_score": None,
                    "status": f"Failed: {exc}",
                    "fit_seconds": round(
                        time.perf_counter() - started,
                        3,
                    ),
                }
            )

    if not successes:

        raise RuntimeError(
            "No benchmark model could be trained successfully."
        )

    successes.sort(
        key=lambda item: item[0]["mean_score"],
        reverse=True,
    )

    results.sort(
        key=lambda r: (
            r["mean_score"] is not None,
            r["mean_score"] if r["mean_score"] is not None else -1,
        ),
        reverse=True,
    )

    best_entry, best_model, best_metrics, best_meta = successes[0]

    metadata = {
        "canonical_algorithm": "kmeans",
        "best_n_clusters": best_entry["n_clusters"],
        "benchmarked_k_range": [k_min, upper_bound],
        "random_state": random_state,
        "data_rows": n_rows,
        "problem_type": "clustering",
    }

    metadata.update(best_meta)

    return (
        best_model,
        best_metrics,
        results,
        metadata,
    )


# ============================================================
# CLUSTERING
# ============================================================

def _train_clustering(
    df: pd.DataFrame,
    algorithm: str,
    n_clusters: int = 3,
    use_pca: bool = False,
    random_state: int = DEFAULT_RANDOM_STATE,
):

    if algorithm not in {
        "kmeans",
        "k_means",
        "k-means",
    }:

        raise ValueError(
            "Currently supported clustering algorithm: K-Means."
        )

    X = df.select_dtypes(
        include=[np.number]
    ).copy()

    if X.empty:

        raise ValueError(
            "K-Means requires at least one numeric feature."
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(
        X.median(
            numeric_only=True
        )
    )

    X = X.fillna(0)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    pca = None
    pca_info = {}

    if (
        use_pca
        and X_scaled.shape[1] >= 2
    ):

        pca = PCA(
            n_components=min(
                2,
                X_scaled.shape[1],
            ),
            random_state=random_state,
        )

        X_model = pca.fit_transform(
            X_scaled
        )

        pca_info = {
            "pca_components": int(
                pca.n_components_
            ),

            "pca_explained_variance": [
                round(
                    float(value),
                    4,
                )
                for value
                in pca.explained_variance_ratio_
            ],

            "pca_explained_variance_total": round(
                float(
                    pca.explained_variance_ratio_.sum()
                ),
                4,
            ),
        }

    else:

        X_model = X_scaled

    n_clusters = int(
        n_clusters
    )

    if n_clusters < 2:

        raise ValueError(
            "n_clusters must be at least 2."
        )

    if n_clusters >= len(X_model):

        raise ValueError(
            "n_clusters must be smaller than the number of rows."
        )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
    )

    labels = model.fit_predict(
        X_model
    )

    # silhouette_score computes a full pairwise distance matrix across
    # every row by default - O(n^2) time and memory. For a small demo
    # dataset that's instant; for tens of thousands of rows (a very
    # realistic real-world size) it can take many seconds PER k value,
    # and _benchmark_clustering tries 7 k values - easily adding up to
    # several minutes with zero visible progress, which just looks like
    # the app has frozen. Sampling a subset for the score is standard
    # practice here and gives an essentially identical number at a
    # tiny fraction of the cost.
    n_rows_for_score = len(X_model)
    silhouette_sample_size = (
        min(5000, n_rows_for_score)
        if n_rows_for_score > 5000
        else None
    )

    metrics = {
        "silhouette": round(
            float(
                silhouette_score(
                    X_model,
                    labels,
                    sample_size=silhouette_sample_size,
                    random_state=random_state,
                )
            ),
            4,
        ),

        "clusters": int(
            n_clusters
        ),

        "rows": int(
            len(X_model)
        ),

        "features": int(
            X.shape[1]
        ),
    }

    metrics.update(
        pca_info
    )

    model_bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": list(
            X.columns
        ),
        "pca": pca,
    }

    return (
        model_bundle,
        metrics,
        X_model,
        labels,
        {
            "canonical_algorithm": "kmeans",
            "pca": pca_info,
        },
    )


# ============================================================
# PUBLIC TRAINING FUNCTION
# ============================================================

def train_model_logic(
    df: pd.DataFrame,
    problem_type: str,
    algorithm: str,
    target_col: str = None,
    *,
    tune: bool = False,
    cv_folds: int = DEFAULT_CV_FOLDS,
    n_clusters: int = 3,
    use_pca: bool = False,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """
    Existing Flask contract remains unchanged.

    Returns:

        model
        metrics
        X_test
        y_test

    The benchmark and manual paths use the same ML rules.
    """

    problem_type = _normalise_problem(
        problem_type
    )

    algorithm = str(
        algorithm or ""
    ).strip().lower()

    # --------------------------------------------------------
    # AUTO / BENCHMARK
    # --------------------------------------------------------

    if algorithm in {
        "auto",
        "auto_select",
        "benchmark",
        "best",
        "automatic",
    }:

        if problem_type == "clustering":

            # Auto-select the best K via silhouette score, then fall
            # through to the CLUSTERING block below to actually train
            # (and return) that winning model the same way the manual
            # path does.

            (
                _,
                _,
                _,
                auto_metadata,
            ) = benchmark_models(
                df,
                problem_type,
                target_col,
                cv_folds=cv_folds,
                test_size=DEFAULT_TEST_SIZE,
                random_state=random_state,
            )

            algorithm = auto_metadata.get(
                "canonical_algorithm",
                "kmeans",
            )

            n_clusters = auto_metadata.get(
                "best_n_clusters",
                n_clusters,
            )

        else:

            (
                model,
                metrics,
                benchmark_results,
                metadata,
            ) = benchmark_models(
                df,
                problem_type,
                target_col,
                cv_folds=cv_folds,
                test_size=DEFAULT_TEST_SIZE,
                random_state=random_state,
            )

            metadata[
                "benchmark_results"
            ] = benchmark_results

            train_model_logic.last_metadata = (
                metadata
            )

            # Return the same holdout split used by benchmark.
            X, y_raw = _prepare_supervised_data(
                df,
                target_col,
            )

            y, _ = _prepare_target(
                y_raw,
                problem_type,
            )

            (
                _,
                X_test,
                _,
                y_test,
            ) = _split_supervised_data(
                X,
                y,
                problem_type,
                test_size=DEFAULT_TEST_SIZE,
                random_state=random_state,
            )

            return (
                model,
                metrics,
                X_test,
                y_test,
            )

    # --------------------------------------------------------
    # SUPERVISED
    # --------------------------------------------------------

    if problem_type in {
        "classification",
        "regression",
    }:

        (
            model,
            metrics,
            X_test,
            y_test,
            metadata,
        ) = _train_supervised(
            df,
            problem_type,
            algorithm,
            target_col,
            tune=tune,
            cv_folds=cv_folds,
            test_size=DEFAULT_TEST_SIZE,
            random_state=random_state,
        )

        train_model_logic.last_metadata = (
            metadata
        )

        return (
            model,
            metrics,
            X_test,
            y_test,
        )

    # --------------------------------------------------------
    # CLUSTERING
    # --------------------------------------------------------

    (
        model,
        metrics,
        X_test,
        y_test,
        metadata,
    ) = _train_clustering(
        df,
        algorithm,
        n_clusters=n_clusters,
        use_pca=use_pca,
        random_state=random_state,
    )

    train_model_logic.last_metadata = (
        metadata
    )

    return (
        model,
        metrics,
        X_test,
        y_test,
    )


train_model_logic.last_metadata = {}


# ============================================================
# METADATA ACCESS
# ============================================================

def get_last_training_metadata() -> dict[str, Any]:

    return getattr(
        train_model_logic,
        "last_metadata",
        {},
    ) 