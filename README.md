<div align="center">

# 🤖 ML Ready AI

**Build the Future with AI — from raw data to a trained, deployable model in minutes.**

An interactive AutoML web application that automates the entire machine learning pipeline — no code required.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

[🚀 Live Demo](https://projects-3-owb6.onrender.com) · [Report Bug](#) · [Request Feature](#)

</div>

---

## 📌 Overview

**ML Ready AI** is an interactive machine learning application built to simplify model creation and training end-to-end. Users upload their own dataset, and the system automatically performs data preprocessing, builds an appropriate model, trains it, evaluates performance, and makes the results downloadable — all through a browser, with zero code.

It's designed to demonstrate a complete ML pipeline — from data input to model deployment — and to make machine learning accessible to beginners, students, and professionals alike, without requiring advanced programming knowledge.

Validated across real-world datasets spanning **healthcare, finance, retail, solar energy, and HR**, achieving an average **R² of 0.99** (peak **R² = 0.9989**).

🔗 **Live app:** [projects-3-owb6.onrender.com](https://projects-3-owb6.onrender.com)
> Hosted on Render's free tier — the app may take 30–60 seconds to spin up on first load if it's been idle.

---

## 🎯 Objectives

- **Simplify model-building** — upload any dataset, automatically get a suitable model trained on it
- **Automate data preprocessing** — missing values, encoding, outlier handling, scaling, and dataset splitting, all without writing code
- **Provide a no-code interface** to train, evaluate, and visualize ML models
- **Demonstrate a full end-to-end ML pipeline** — data input → training → evaluation → export
- **Lower the barrier to ML** for beginners and non-programmers
- **Enable deployment** by letting users download the trained model for real-world use

---

## ✨ Key Features

- 📂 **Multi-format upload** — CSV, XLSX, JSON, XML, and HTML supported (up to 50 MB)
- 🔍 **Automated EDA** — instant statistical summary, histograms, correlation heatmaps, and categorical distribution analysis on upload
- 🧹 **Smart data cleaning** — missing values handled via mean/median/mode/forward-fill, outliers removed via IQR, duplicates dropped in one click
- ⚙️ **Automated feature engineering** — label encoding for categorical columns, standard scaling for numeric features, all in-browser
- 🎯 **Multi-problem support** — Classification, Regression, and Clustering, each with auto-selected relevant algorithms and evaluation metrics
- 🧠 **Live evaluation metrics** — Accuracy for classifiers, R² & RMSE for regression, Silhouette score for clustering, computed on a held-out test split
- 📊 **Auto-generated BI dashboards** — heatmaps, feature importance, and pair plots
- 📦 **Exportable deliverables** — download the trained model as a `.pkl` file and the cleaned dataset as a `.csv`, ready for production or further use
- 🔐 **User accounts** — simple login/signup flow to manage sessions
- 🔁 **Session persistence** — SQLite-backed session management so your progress isn't lost mid-pipeline

---

## 🖼️ Screenshots

| Landing Page | Tech Stack |
|---|---|
| ![Landing](screenshots/01-landing-hero.jpg) | ![Tech Stack](screenshots/02-tech-stack.jpg) |

| How It Works | Login / Signup |
|---|---|
| ![How it works](screenshots/03-how-it-works.jpg) | ![Login](screenshots/04-login-signup.jpg) |

| Upload Dataset | Training Results |
|---|---|
| ![Upload](screenshots/05-upload-dataset.jpg) | ![Training Results](screenshots/06-training-results.jpg) |

| Project Completed | Final Deliverables |
|---|---|
| ![Completed](screenshots/07-project-completed.jpg) | ![Deliverables](screenshots/08-final-deliverables.jpg) |

> Place the `screenshots/` folder (included alongside this README) at your repo root, or update the paths above if you store it elsewhere.

---

## 🏗️ How It Works

ML Ready AI walks the user through a guided, sequential 8-stage pipeline — no PhD required, and nothing falls through the cracks:

| Step | Stage | What Happens |
|:---:|---|---|
| 01 | **Upload** | Drop a CSV/XLSX/JSON/XML/HTML file (up to 50 MB). The system instantly parses it, detects column types, and gives a full statistical summary. |
| 02 | **Summary** | Full statistical overview of the uploaded dataset. |
| 03 | **EDA** | Auto-generated histograms with KDE curves, correlation heatmaps, and categorical bar charts for every column. |
| 04 | **Cleaning** | Missing values handled via mean, median, mode, or forward-fill strategy; outliers removed via IQR; duplicate rows dropped in one click. |
| 05 | **Engineering** | Categorical columns label-encoded, numeric features standard-scaled — data prepared exactly as ML models expect it. |
| 06 | **Model Setup** | Choose the problem type — Classification, Regression, or Clustering — and the app auto-selects relevant algorithms and evaluation metrics. |
| 07 | **Training** | Model trains on the processed data. |
| 08 | **Results** | Live evaluation metrics (Accuracy for classifiers, R² & RMSE for regression, Silhouette score for clustering) computed on a held-out test split, plus final deliverables ready to download. |

---

## 🏛️ Architecture

```
                 ┌────────────────────┐
   User ──────▶  │   Flask REST API    │
 (CSV/XLSX/...)  │  (Jinja2 templates) │
                 └─────────┬───────────┘
                           │
                 ┌─────────▼───────────┐
                 │  SQLite (sessions)  │
                 └─────────┬───────────┘
                           │
     ┌──────────┬──────────┼──────────┬───────────┐
     ▼          ▼          ▼          ▼           ▼
   EDA   Preprocessing  Feature   Training   Evaluation
                         Engineering
                           │
                 ┌─────────▼───────────┐
                 │   Final Deliverables │
                 │  .pkl model + .csv   │
                 │   + BI dashboards    │
                 └──────────────────────┘
```

---

## 🛠️ Tech Stack

Built on the same libraries used by data scientists at leading research labs — powered by proven tools, not experimental frameworks:

| Category | Tools |
|---|---|
| **Language** | Python 3.11 |
| **Backend / Web** | Flask, Jinja2, HTML, CSS, JavaScript |
| **ML & Data** | scikit-learn, pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | joblib / pickle |
| **Database** | SQLite (session management) |
| **Methodology** | Agile (iterative sprints — see [Development Approach](#-development-approach)) |

---

## 📊 Results

| Metric | Value |
|---|---|
| Datasets validated | 6 independent real-world datasets |
| Domains covered | Healthcare, Finance, Retail, Solar Energy, HR |
| Average R² | **0.99** |
| Peak R² | **0.9989** (Decision Tree, Regression) |
| Manual ML setup time reduction | **~70%** |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/aakarshitpathak/ml-ready-ai.git
cd ml-ready-ai

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open **`http://localhost:5000`** in your browser.

---

## 🚀 Usage

1. **Sign up / log in** to your account
2. **Upload** a dataset (CSV, XLSX, JSON, XML, or HTML — max 50 MB). Make sure it has a header row.
3. Review the **auto-generated summary and EDA** (histograms, correlation heatmaps, distributions)
4. Let the pipeline **clean and engineer** your data automatically
5. Choose a **problem type** and train — view live performance metrics
6. **Download** your trained model (`.pkl`) and cleaned dataset (`.csv`) from the final dashboard

---

## 📁 Project Structure

A quick orientation, not an exhaustive file list:

- **`app.py`** — Flask application entry point / routes
- **`templates/`** — Jinja2 HTML pages (upload, summary, EDA, cleaning, engineering, model, results)
- **`static/`** — CSS, JS, and front-end assets
- **`utils/`** — Core pipeline logic (preprocessing, EDA generation, model training & evaluation)
- **`models/`** — Saved trained models (`.pkl`)
- **`database.db`** — SQLite store for session management
- **`requirements.txt`** — Python dependencies

> Update this list if your real structure differs — a quick orientation like this is easier to keep accurate than a full file tree.

---

## 🧪 Development Approach

Built using an **Agile methodology** — well suited for ML systems that require continuous experimentation and refinement rather than a fixed upfront spec. Development proceeded through short sprints:

1. **Requirement Analysis**
2. **Model Development**
3. **Evaluation**
4. **UI Design**
5. **Coding**
6. **Testing**
7. **Deployment**

This allowed preprocessing logic, UI flow, and model performance to evolve iteratively based on testing across the 6 validation datasets, rather than being locked in from day one.

---

## 🔮 Future Scope

- **Full AutoML integration** — currently the user selects the model manually; future versions will auto-select the best algorithm and hyperparameters
- **Advanced data visualization** — interactive dashboards with real-time charts, filtering, and drill-down analysis
- **Deep learning support** — CNNs for image data, RNN/LSTM for time-series and text data

---

## 👤 Author

**Aakarshit Pathak**

[LinkedIn](https://www.linkedin.com/in/aakarshit-p-501605264) · [GitHub](https://github.com/aakarshitpathak) · [Kaggle](https://www.kaggle.com/aakarshitpathak) · [Blog](https://free28716.wordpress.com)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
