# ============================================================
# ML READY AI — Main Flask Application
# ============================================================

import io
import os
from functools import wraps

import joblib
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, send_from_directory, send_file,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from modules.data_loader import load_dataframe, allowed_file
from modules.data_summary import get_summary
from modules.eda import generate_visualizations
from modules.cleaning import (
    handle_missing_values, get_missing_stats, detect_outliers,
    handle_outliers, remove_duplicates, fix_inconsistencies,
    auto_clean_dataset,
)
from modules.feature_engineering import (
    apply_label_encoding, apply_one_hot_encoding, apply_standard_scaling, get_feature_info,
)
from modules.model_training import train_model_logic, benchmark_models
from modules.api_routes import api_bp


app = Flask(__name__)
app.register_blueprint(api_bp)
# Secret key is read from the environment in production (Render/host dashboard).
# Falls back to a fixed dev key only for local testing so the app still runs
# out of the box, but a real deployment should always set SECRET_KEY.
app.secret_key = os.environ.get("SECRET_KEY", "mlreadyai_secret_2025_dev_only")

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)


with app.app_context():
    db.create_all()


# ============================================================
# AUTHENTICATION
# ============================================================

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not password:
            flash("Please complete all required fields.", "error")
            return redirect(url_for("signup"))

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for("signup"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "error")
            return redirect(url_for("login"))

        db.session.add(User(
            name=name,
            email=email,
            password=generate_password_hash(password),
        ))
        db.session.commit()
        flash("Account created successfully! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("authentication.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["user_name"] = user.name
            flash(f"Welcome back, {user.name}!", "success")
            return redirect(url_for("upload"))

        flash("Invalid email or password.", "error")
        return redirect(url_for("login"))

    if "user_id" in session:
        return redirect(url_for("upload"))
    return render_template("authentication.html")


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("user_name", None)
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login to access this page.", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


# ============================================================
# DATA STORE
# ============================================================

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

_store = {}


def get_df():
    return _store.get("df")


def set_df(df):
    _store["df"] = df


def clear_ml_artifacts():
    # Clear every artifact that depends on the current dataset/problem/model.
    for key in (
        "benchmark_results", "best_model", "benchmark_model",
        "benchmark_metrics", "benchmark_metadata", "trained_model",
        "metrics", "training_metadata", "X_test", "y_test",
    ):
        _store.pop(key, None)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


# ============================================================
# UPLOAD / SUMMARY / EDA
# ============================================================

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "dataset" not in request.files:
            flash("No file part in the request.", "error")
            return redirect(url_for("upload"))

        file = request.files["dataset"]
        if not file.filename:
            flash("Please select a file before uploading.", "error")
            return redirect(url_for("upload"))

        if not allowed_file(file.filename):
            flash(
                "Unsupported file format. Please upload CSV, Excel, JSON, XML or HTML.",
                "error",
            )
            return redirect(url_for("upload"))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        try:
            file.save(filepath)
            df = load_dataframe(filepath)
        except Exception as exc:
            flash(f"Could not read file: {exc}", "error")
            return redirect(url_for("upload"))

        clear_ml_artifacts()
        for key in ("problem_type", "target_col", "algorithm"):
            session.pop(key, None)

        set_df(df)
        session["filename"] = filename
        flash(
            f'✅ "{filename}" uploaded successfully — '
            f"{df.shape[0]} rows × {df.shape[1]} columns.",
            "success",
        )
        return redirect(url_for("summary"))

    return render_template("upload.html")


@app.route("/summary")
def summary():
    df = get_df()
    if df is None:
        flash("Please upload a dataset first.", "error")
        return redirect(url_for("upload"))
    return render_template(
        "summary.html",
        summary=get_summary(df),
        filename=session.get("filename", "dataset"),
    )


@app.route("/eda")
def eda():
    df = get_df()
    if df is None:
        flash("Please upload a dataset first.", "error")
        return redirect(url_for("upload"))
    return render_template(
        "eda.html",
        plots=generate_visualizations(df),
        filename=session.get("filename", "dataset"),
    )


# ============================================================
# CLEANING
# ============================================================

@app.route("/cleaning", methods=["GET", "POST"])
def cleaning():
    df = get_df()
    if df is None:
        flash("Please upload a dataset first.", "error")
        return redirect(url_for("upload"))

    if request.method == "POST":
        action = request.form.get("action")
        strategy = request.form.get("strategy", "")

        try:
            if action == "auto_clean":
                cleaned_df, report = auto_clean_dataset(df, outlier_strategy="cap")
                set_df(cleaned_df)
                _store["cleaning_report"] = report
                clear_ml_artifacts()
                flash("✅ Dataset automatically cleaned successfully!", "success")

            elif action == "impute":
                set_df(handle_missing_values(df, strategy=strategy))
                clear_ml_artifacts()
                flash(f'✅ Missing values handled using "{strategy}" strategy.', "success")

            elif action == "outliers":
                set_df(handle_outliers(df, strategy=strategy))
                clear_ml_artifacts()
                flash(f'✅ Outliers handled using "{strategy}" strategy.', "success")

            elif action == "duplicates":
                set_df(remove_duplicates(df))
                clear_ml_artifacts()
                flash("✅ Duplicate rows removed.", "success")

            elif action == "fix_inconsistent":
                set_df(fix_inconsistencies(df))
                clear_ml_artifacts()
                flash("✅ Basic string inconsistencies fixed.", "success")

        except Exception as exc:
            flash(f"Error during cleaning: {exc}", "error")

        return redirect(url_for("cleaning"))

    return render_template(
        "cleaning.html",
        missing_stats=get_missing_stats(df),
        outlier_stats=detect_outliers(df),
        duplicate_count=int(df.duplicated().sum()),
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

@app.route("/feature_engineering", methods=["GET", "POST"])
def feature_engineering():
    df = get_df()
    if df is None:
        flash("Please upload a dataset first.", "error")
        return redirect(url_for("upload"))

    if request.method == "POST":
        action = request.form.get("action")
        selected_cols = request.form.getlist("cols")

        if not selected_cols:
            flash("⚠️ Please select at least one column.", "error")
            return redirect(url_for("feature_engineering"))

        try:
            if action == "one_hot_encode":
                set_df(apply_one_hot_encoding(df, selected_cols))
                clear_ml_artifacts()
                flash(f'✅ One-Hot encoded: {", ".join(selected_cols)}', "success")
            elif action == "label_encode":
                set_df(apply_label_encoding(df, selected_cols))
                clear_ml_artifacts()
                flash(f'✅ Label encoded: {", ".join(selected_cols)}', "success")
            elif action == "scale":
                set_df(apply_standard_scaling(df, selected_cols))
                clear_ml_artifacts()
                flash(f'✅ Scaled: {", ".join(selected_cols)}', "success")
        except Exception as exc:
            flash(f"Error during feature engineering: {exc}", "error")

        return redirect(url_for("feature_engineering"))

    return render_template("feature_engineering.html", features=get_feature_info(df))


# ============================================================
# PROBLEM SELECTION
# ============================================================

@app.route("/model", methods=["GET", "POST"])
def model():
    df = get_df()
    if df is None:
        flash("Please upload a dataset first.", "error")
        return redirect(url_for("upload"))

    if request.method == "POST":
        problem_type = request.form.get("problem_type", "").strip().lower()
        target_col = request.form.get("target_col", "").strip()

        if problem_type not in {"classification", "regression", "clustering"}:
            flash("⚠️ Invalid problem type selected.", "error")
            return redirect(url_for("model"))

        if problem_type in {"classification", "regression"}:
            if not target_col:
                flash(f'⚠️ Prediction for "{problem_type}" requires a Target Column.', "error")
                return redirect(url_for("model"))
            if target_col not in df.columns:
                flash(f'⚠️ Target column "{target_col}" was not found.', "error")
                return redirect(url_for("model"))

        clear_ml_artifacts()
        session["problem_type"] = problem_type
        session["target_col"] = target_col
        session.pop("algorithm", None)
        flash(f"✅ Problem Type: {problem_type.capitalize()} selected.", "success")
        return redirect(url_for("algorithm_selection"))

    return render_template("model.html", columns=list(df.columns))


# ============================================================
# ALGORITHM SELECTION
# ============================================================

@app.route("/algorithm_selection", methods=["GET", "POST"])
def algorithm_selection():
    problem_type = session.get("problem_type")
    if not problem_type:
        flash("Please select a problem type first.", "error")
        return redirect(url_for("model"))

    if request.method == "POST":
        algorithm = request.form.get("algorithm", "").strip().lower()
        if not algorithm:
            flash("⚠️ Please select an algorithm.", "error")
            return redirect(url_for("algorithm_selection"))

        clear_ml_artifacts()
        if algorithm in {"auto_select", "benchmark", "auto"}:
            session["algorithm"] = "auto_select"
            flash(
                "🔍 Benchmark mode selected. ML Ready AI will compare suitable algorithms and recommend the best-performing model.",
                "success",
            )
            return redirect(url_for("model_comparison"))

        session["algorithm"] = algorithm
        flash(
            "✅ Algorithm: " + algorithm.replace("_", " ").title() + " selected.",
            "success",
        )
        return redirect(url_for("train_model"))

    return render_template("algorithm.html", problem_type=problem_type)


# ============================================================
# MODEL COMPARISON / AUTO BENCHMARK
# ============================================================

@app.route("/model_comparison")
def model_comparison():
    df = get_df()
    problem_type = session.get("problem_type")
    target_col = session.get("target_col")

    if df is None:
        flash("Please upload a dataset first.", "error")
        return redirect(url_for("upload"))
    if not problem_type:
        flash("Please select a problem type first.", "error")
        return redirect(url_for("model"))
    if problem_type in {"classification", "regression"} and not target_col:
        flash("Please select a target column first.", "error")
        return redirect(url_for("model"))

    try:
        winner_model, final_metrics, benchmark_results, metadata = benchmark_models(
            df=df,
            problem_type=problem_type,
            target_col=target_col,
            cv_folds=5,
            test_size=0.20,
            random_state=42,
        )

        if not benchmark_results:
            flash("No suitable models found.", "error")
            return redirect(url_for("algorithm_selection"))

        # model_training.py is the source of truth. Accept its native keys
        # and also preserve compatibility with older templates.
        results = []
        for raw in benchmark_results:
            cv_score = raw.get("cv_score", raw.get("mean_score"))
            cv_std = raw.get("cv_std", raw.get("std_score"))
            error = raw.get("error")

            if cv_score is not None:
                results.append({
                    **raw,
                    "algorithm": raw.get("algorithm"),
                    "algorithm_key": raw.get("algorithm_key"),
                    "mean_score": cv_score,
                    "std_score": cv_std or 0.0,
                    "cv_score": cv_score,
                    "cv_std": cv_std or 0.0,
                    "scores": raw.get("scores", []),
                    "status": "Success",
                    "error": None,
                })
            else:
                results.append({
                    **raw,
                    "algorithm": raw.get("algorithm"),
                    "algorithm_key": raw.get("algorithm_key"),
                    "mean_score": None,
                    "std_score": None,
                    "cv_score": None,
                    "cv_std": None,
                    "scores": raw.get("scores", []),
                    "status": "Failed: " + str(error or raw.get("status", "Unknown error")),
                    "error": error or raw.get("status", "Unknown error"),
                })

        successful_results = [r for r in results if r.get("mean_score") is not None]
        failed_results = [r for r in results if r.get("mean_score") is None]
        successful_results.sort(key=lambda r: r["mean_score"], reverse=True)
        results = successful_results + failed_results

        if not successful_results:
            errors = [
                f"{r.get('algorithm')}: {r.get('error')}"
                for r in failed_results
                if r.get("error")
            ]
            detail = " | ".join(errors) or "No detailed error was returned."
            flash("All models failed during benchmarking. Details: " + detail, "error")
            return redirect(url_for("algorithm_selection"))

        best_model = dict(successful_results[0])
        canonical_winner = (
            metadata.get("canonical_algorithm")
            or best_model.get("algorithm_key")
        )
        best_model["canonical_algorithm"] = canonical_winner

        _store["benchmark_results"] = results
        _store["best_model"] = best_model
        _store["benchmark_model"] = winner_model
        _store["benchmark_metrics"] = final_metrics
        _store["benchmark_metadata"] = metadata

        return render_template(
            "model_comparison.html",
            results=results,
            best_model=best_model,
            problem_type=problem_type,
        )

    except Exception as exc:
        flash(f"Error during model comparison: {type(exc).__name__}: {exc}", "error")
        return redirect(url_for("algorithm_selection"))


# ============================================================
# USE RECOMMENDED MODEL
# ============================================================

@app.route("/use_recommended_model", methods=["POST"])
def use_recommended_model():
    algorithm = request.form.get("algorithm", "").strip()
    algorithm_key = request.form.get("algorithm_key", "").strip().lower()

    best_model = _store.get("best_model") or {}

    # Prefer a canonical key supplied by the template, then the stored
    # benchmark winner. This removes display-name/canonical-name mismatch.
    internal_algorithm = algorithm_key or best_model.get("canonical_algorithm")

    if not internal_algorithm:
        algorithm_map = {
            "Linear Regression": "linear_regression",
            "Decision Tree": "decision_tree",
            "Decision Tree Regressor": "decision_tree_regressor",
            "Random Forest": "random_forest",
            "Random Forest Regressor": "random_forest_regressor",
            "Gradient Boosting": "gradient_boosting",
            "Gradient Boosting Regressor": "gradient_boosting_regressor",
            "SVR": "svm_regressor",
            "SVM": "svm",
            "KNN": "knn",
            "KNN Regressor": "knn_regressor",
            "XGBoost": "xgboost",
            "XGBoost Regressor": "xgboost_regressor",
            "Logistic Regression": "logistic_regression",
            "Decision Tree Classifier": "decision_tree",
            "Random Forest Classifier": "random_forest",
            "SVM Classifier": "svm",
            "KNN Classifier": "knn",
            "XGBoost Classifier": "xgboost",
        }
        internal_algorithm = algorithm_map.get(algorithm)

    if not internal_algorithm:
        flash(f"Unsupported recommended algorithm: {algorithm}", "error")
        return redirect(url_for("model_comparison"))

    session["algorithm"] = internal_algorithm

    # Clustering doesn't have a target column, but it does have a K
    # (number of clusters) chosen by the benchmark - carry that through
    # so training actually uses the recommended K instead of a default.
    if internal_algorithm == "kmeans":
        n_clusters = best_model.get("n_clusters")
        if n_clusters:
            session["n_clusters"] = int(n_clusters)

    display_name = algorithm or best_model.get("algorithm") or internal_algorithm
    flash(f"🏆 Recommended model selected: {display_name}", "success")
    return redirect(url_for("train_model"))


# ============================================================
# TRAINING / EVALUATION
# ============================================================

@app.route("/train_model")
def train_model():
    df = get_df()
    problem_type = session.get("problem_type")
    algorithm = session.get("algorithm")
    target_col = session.get("target_col")

    if df is None:
        flash("Dataset missing. Please upload a dataset first.", "error")
        return redirect(url_for("upload"))
    if not problem_type:
        flash("Problem type missing. Please select a problem type first.", "error")
        return redirect(url_for("model"))
    if not algorithm:
        flash("Algorithm missing. Please select an algorithm first.", "error")
        return redirect(url_for("algorithm_selection"))
    if problem_type in {"classification", "regression"} and (
        not target_col or target_col not in df.columns
    ):
        flash("A valid target column is required.", "error")
        return redirect(url_for("model"))

    try:
        model_obj, metrics, X_test, y_test = train_model_logic(
            df,
            problem_type,
            algorithm,
            target_col,
            cv_folds=5,
            n_clusters=session.get("n_clusters", 3),
            random_state=42,
        )

        _store["trained_model"] = model_obj
        _store["metrics"] = metrics
        _store["X_test"] = X_test
        _store["y_test"] = y_test

        try:
            from modules.model_training import get_last_training_metadata
            _store["training_metadata"] = get_last_training_metadata()
        except Exception:
            _store["training_metadata"] = {}

        benchmark_results = _store.get("benchmark_results")
        return render_template(
            "result.html",
            metrics=metrics,
            algorithm=algorithm,
            problem_type=problem_type,
            benchmark_results=benchmark_results,
        )

    except Exception as exc:
        flash(f"Error during training: {type(exc).__name__}: {exc}", "error")
        return redirect(url_for("algorithm_selection"))


# ============================================================
# EXPORTS / FINAL DASHBOARD
# ============================================================

@app.route("/export_model")
def export_model():
    model_obj = _store.get("trained_model")
    algorithm = session.get("algorithm", "model")

    if model_obj is None:
        flash("⚠️ No trained model found to export.", "error")
        return redirect(url_for("train_model"))

    try:
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)
        filename = f"{secure_filename(algorithm) or 'model'}_trained.pkl"
        filepath = os.path.join(models_dir, filename)
        joblib.dump(model_obj, filepath)
        return send_from_directory(models_dir, filename, as_attachment=True)
    except Exception as exc:
        flash(f"Error exporting model: {exc}", "error")
        return redirect(url_for("train_model"))


@app.route("/export_data")
def export_data():
    df = get_df()
    if df is None:
        flash("⚠️ No data found to export.", "error")
        return redirect(url_for("upload"))

    try:
        output = io.BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return send_file(
            output,
            mimetype="text/csv",
            as_attachment=True,
            download_name="ml_ready_cleaned_data.csv",
        )
    except Exception as exc:
        flash(f"Error exporting data: {exc}", "error")
        return redirect(url_for("final_dashboard"))


@app.route("/final")
def final_dashboard():
    problem_type = session.get("problem_type")
    algorithm = session.get("algorithm")
    metrics = _store.get("metrics")

    if not algorithm or not metrics:
        flash("⚠️ Training metrics missing. Please train a model first.", "error")
        return redirect(url_for("train_model"))

    from modules.eda import generate_model_result_plot

    chart = generate_model_result_plot(
        problem_type,
        _store.get("trained_model"),
        _store.get("X_test"),
        _store.get("y_test"),
    )

    return render_template(
        "final.html",
        problem_type=problem_type,
        algorithm=algorithm,
        metrics=metrics,
        chart=chart,
    )


if __name__ == "__main__":
    # Debug mode is controlled by an env var so it's off by default in
    # production but can still be turned on locally with FLASK_DEBUG=1.
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1")
