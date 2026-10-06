"""
modules/eda.py — MODULE 5
Generates data visualizations using Matplotlib and Seaborn.
Plots are converted to Base64 strings to be embedded directly in HTML.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (essential for Flask)
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import numpy as np
import pandas as pd

def get_base64_plot():
    """Converts the current matplotlib figure to a base64 string."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def generate_visualizations(df: pd.DataFrame) -> dict:
    """
    Generates a set of plots for the dataset:
      1. Histograms for all numeric columns.
      2. Bar charts for top categorical columns.
      3. Correlation heatmap for numeric columns.

    Returns a dictionary of {plot_name: base64_string}.
    """
    plots = {}

    # Set dark theme for plots to match UI
    plt.style.use('dark_background')
    sns.set_palette("husl")

    # 1. ── Histograms (Numeric Distributions) ──────────
    numeric_cols = df.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        # We'll plot up to top 6 numeric columns to keep it clean
        cols_to_plot = numeric_cols[:6]
        n = len(cols_to_plot)
        rows = (n + 1) // 2
        
        plt.figure(figsize=(12, 4 * rows))
        for i, col in enumerate(cols_to_plot):
            plt.subplot(rows, 2, i + 1)
            sns.histplot(df[col].dropna(), kde=True, color='#6c63ff')
            plt.title(f'Distribution of {col}', color='#00d2ff', fontweight='bold')
            plt.grid(alpha=0.2)
        
        plt.tight_layout()
        plots['histograms'] = get_base64_plot()

    # 2. ── Bar Charts (Categorical Counts) ──────────────
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.empty:
        # Plot up to top 4 categorical columns
        cols_to_plot = cat_cols[:4]
        n = len(cols_to_plot)
        rows = (n + 1) // 2

        plt.figure(figsize=(12, 4 * rows))
        for i, col in enumerate(cols_to_plot):
            plt.subplot(rows, 2, i + 1)
            # Show top 10 categories only
            counts = df[col].value_counts().head(10)
            sns.barplot(x=counts.index, y=counts.values, palette="mako")
            plt.title(f'Counts of {col} (Top 10)', color='#00d2ff', fontweight='bold')
            plt.xticks(rotation=45)
            plt.grid(alpha=0.2, axis='y')
        
        plt.tight_layout()
        plots['bar_charts'] = get_base64_plot()

    # 3. ── Correlation Heatmap ──────────────────────────
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Feature Correlation Heatmap', color='#00d2ff', fontweight='bold', fontsize=16)
        
        plt.tight_layout()
        plots['heatmap'] = get_base64_plot()

    return plots


def generate_model_result_plot(problem_type, model, X_test, y_test):
    """
    Generates ONE diagnostic visualization for the trained model, used on
    the Final Dashboard:
      - regression:      predicted-vs-actual scatter + residual plot
      - classification:  confusion matrix heatmap
      - clustering:      2D projection of the clusters (PCA if >2 dims)

    Returns a base64 PNG string, or None if a chart genuinely can't be
    produced (e.g. no test data) so the caller can hide the section
    gracefully instead of erroring the whole page.
    """
    try:
        plt.style.use('dark_background')

        if problem_type == 'regression':
            if X_test is None or y_test is None or len(X_test) == 0:
                return None

            preds = np.asarray(model.predict(X_test))
            y_true = np.asarray(y_test)
            residuals = y_true - preds

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].scatter(y_true, preds, alpha=0.65, color='#00d2ff', edgecolor='none')
            lims = [
                min(y_true.min(), preds.min()),
                max(y_true.max(), preds.max()),
            ]
            axes[0].plot(lims, lims, '--', color='#ff6584', linewidth=1.5, label='Perfect prediction')
            axes[0].set_xlabel('Actual')
            axes[0].set_ylabel('Predicted')
            axes[0].set_title('Predicted vs Actual', color='#00d2ff', fontweight='bold')
            axes[0].legend(fontsize=8)
            axes[0].grid(alpha=0.2)

            axes[1].scatter(preds, residuals, alpha=0.65, color='#6c63ff', edgecolor='none')
            axes[1].axhline(0, color='#ff6584', linestyle='--', linewidth=1.5)
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('Residual (Actual - Predicted)')
            axes[1].set_title('Residuals', color='#00d2ff', fontweight='bold')
            axes[1].grid(alpha=0.2)

            plt.tight_layout()
            return get_base64_plot()

        if problem_type == 'classification':
            if X_test is None or y_test is None or len(X_test) == 0:
                return None

            from sklearn.metrics import confusion_matrix

            preds = model.predict(X_test)
            cm = confusion_matrix(y_test, preds)

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='mako', cbar=False)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix', color='#00d2ff', fontweight='bold')
            plt.tight_layout()
            return get_base64_plot()

        if problem_type == 'clustering':
            if X_test is None or y_test is None:
                return None

            X = np.asarray(X_test)
            labels = np.asarray(y_test)

            if X.shape[0] == 0:
                return None

            if X.shape[1] == 1:
                # Only one usable numeric feature (e.g. clustering ran
                # on a single column because the rest of the dataset
                # was non-numeric). Can't make a 2D projection out of
                # 1 dimension, but a 1D strip plot (jittered so points
                # don't overlap into a single line) still shows the
                # cluster boundaries clearly.
                rng = np.random.default_rng(42)
                jitter = rng.uniform(-0.15, 0.15, size=X.shape[0])

                plt.figure(figsize=(8, 4))
                scatter = plt.scatter(
                    X[:, 0], jitter,
                    c=labels, cmap='cool', alpha=0.5, edgecolor='none', s=18,
                )
                plt.yticks([])
                plt.xlabel('Feature value (scaled)')
                plt.title('Cluster Assignments (1 feature)', color='#00d2ff', fontweight='bold')
                plt.colorbar(scatter, label='Cluster')
                plt.tight_layout()
                return get_base64_plot()

            if X.shape[1] > 2:
                from sklearn.decomposition import PCA
                X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
            else:
                X_2d = X

            plt.figure(figsize=(7, 6))
            scatter = plt.scatter(
                X_2d[:, 0], X_2d[:, 1],
                c=labels, cmap='cool', alpha=0.75, edgecolor='none',
            )
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.title('Cluster Assignments (2D projection)', color='#00d2ff', fontweight='bold')
            plt.colorbar(scatter, label='Cluster')
            plt.tight_layout()
            return get_base64_plot()

    except Exception as exc:
        import traceback
        print(
            f"[generate_model_result_plot] Chart generation failed "
            f"for problem_type={problem_type!r}: {type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        plt.close('all')
        return None

    return None
