"""
ML Ready AI — Model Comparison Adapter

IMPORTANT:
This file intentionally contains NO model definitions.

All models, preprocessing, hyperparameters, splitting,
cross-validation and evaluation are controlled by:

    modules.model_training.py

This preserves the existing Flask/UI interface:

    compare_models(df, problem_type, target_col)
"""

from modules.model_training import (
    benchmark_models,
)


def compare_models(
    df,
    problem_type,
    target_col,
):
    """
    Backward-compatible function used by app.py.

    Returns the same result structure expected by
    the existing model_comparison.html page.
    """

    (
        winner,
        final_metrics,
        results,
        metadata,
    ) = benchmark_models(
        df,
        problem_type,
        target_col,
        cv_folds=5,
        test_size=0.20,
        random_state=42,
    )

    # Keep the exact benchmark information available
    # to the Flask application.
    compare_models.last_benchmark = {
        "model": winner,
        "metrics": final_metrics,
        "results": results,
        "metadata": metadata,
        "algorithm_key": metadata.get(
            "canonical_algorithm"
        ),
        "data_rows": metadata.get(
            "data_rows"
        ),
        "target_col": metadata.get(
            "target_col"
        ),
        "problem_type": metadata.get(
            "problem_type"
        ),
    }

    return results


compare_models.last_benchmark = None