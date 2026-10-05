"""
modules/api_routes.py

Optional REST API layer for ML Ready AI.
Register this blueprint from app.py without changing existing templates/UI.
"""

from __future__ import annotations

import os
import tempfile

from flask import Blueprint, jsonify, request

from modules.data_loader import allowed_file, load_dataframe
from modules.model_training import (
    benchmark_models,
    get_last_training_metadata,
    train_model_logic,
)

api_bp = Blueprint("ml_api", __name__, url_prefix="/api/v1")


@api_bp.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "ML Ready AI",
            "api_version": "v1",
        }
    )


def _save_upload():
    if "dataset" not in request.files:
        raise ValueError("Send the dataset as multipart/form-data field 'dataset'.")

    uploaded = request.files["dataset"]

    if not uploaded.filename:
        raise ValueError("No dataset filename was provided.")

    if not allowed_file(uploaded.filename):
        raise ValueError(
            "Unsupported file format. Use CSV, XLSX, XLS, JSON, XML or HTML."
        )

    suffix = os.path.splitext(uploaded.filename)[1].lower()

    temp = tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    )
    uploaded.save(temp.name)
    temp.close()
    return temp.name


@api_bp.post("/benchmark")
def benchmark():
    path = None
    try:
        path = _save_upload()
        df = load_dataframe(path)

        problem_type = request.form.get("problem_type", "").lower()
        target_col = request.form.get("target_col", "")

        _, metrics, results, metadata = benchmark_models(
            df,
            problem_type,
            target_col,
            cv_folds=int(request.form.get("cv_folds", 5)),
        )

        return jsonify(
            {
                "success": True,
                "problem_type": problem_type,
                "target": target_col,
                "winner": metrics.get("benchmark_winner"),
                "metrics": metrics,
                "benchmark": results,
                "feature_importance": metadata.get("feature_importance", {}),
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    finally:
        if path and os.path.exists(path):
            os.remove(path)


@api_bp.post("/train")
def train():
    path = None
    try:
        path = _save_upload()
        df = load_dataframe(path)

        problem_type = request.form.get("problem_type", "").lower()
        algorithm = request.form.get("algorithm", "").lower()
        target_col = request.form.get("target_col", "")

        tune = request.form.get("tune", "false").lower() in {
            "1",
            "true",
            "yes",
        }

        model, metrics, _, _ = train_model_logic(
            df,
            problem_type,
            algorithm,
            target_col,
            tune=tune,
            cv_folds=int(request.form.get("cv_folds", 5)),
        )

        metadata = get_last_training_metadata()

        return jsonify(
            {
                "success": True,
                "algorithm": metadata.get("canonical_algorithm", algorithm),
                "metrics": metrics,
                "feature_importance": metadata.get("feature_importance", {}),
                "target_classes": metadata.get("target_classes", []),
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    finally:
        if path and os.path.exists(path):
            os.remove(path)
