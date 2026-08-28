# 🧪 MLflow & DagsHub Tracking Demo

A hands-on, practical guide and revision reference for **MLflow Experiment Tracking** integrated with **DagsHub** (remote hosted MLflow) and **AWS EC2 + S3** (self-hosted MLflow).

---

## 📌 Why This Project?

When building machine learning models, running experiments in Jupyter notebooks without tracking quickly leads to chaos:
- *"What hyperparameters gave 96% accuracy last week?"*
- *"Where is the saved model artifact for that run?"*
- *"Which code script produced this confusion matrix?"*

**MLflow** solves this by acting as a central ledger for all machine learning experiments, logging parameters, evaluation metrics, visual plots, dataset snapshots, and serialized model files.

This repository demonstrates how to track experiments on the classic **Iris dataset** using two algorithms (**Decision Tree** and **Random Forest**) and sync everything to a remote cloud tracking server (**DagsHub**).

---

## 🧠 Core MLflow Concepts (Revision Cheatsheet)

| Concept | What It Is | How We Used It in Code |
| :--- | :--- | :--- |
| **Experiment** | A logical group for related runs (e.g., `iris-dt`, `iris-rf`). | `mlflow.set_experiment('iris-rf')` |
| **Run** | A single execution of model training. | `with mlflow.start_run():` |
| **Parameters** | Key-value inputs / hyperparameters (fixed per run). | `mlflow.log_param('max_depth', 10)` |
| **Metrics** | Quantitative evaluation results (can change/update). | `mlflow.log_metric('accuracy', 0.9667)` |
| **Artifacts** | Output files like charts, images, and source code. | `mlflow.log_artifact('confusion_matrix.png')`<br/>`mlflow.log_artifact(__file__)` |
| **Model** | Serialized model package with flavor & dependencies. | `mlflow.sklearn.log_model(rf, "random forest")` |
| **Tags** | Metadata & annotations for filtering / searching. | `mlflow.set_tag('author', 'Sujat')` |

---

## 📂 Project Structure

```text
MLFlow-dagshub-demo/
│
├── iris-dt.py             # Decision Tree Classifier pipeline + MLflow logging
├── iris-rf.py             # Random Forest Classifier pipeline + MLflow logging
├── requirements.txt       # Python dependencies (MLflow, scikit-learn, dagshub, etc.)
├── steps.md               # Complete guide for deploying MLflow on AWS (EC2 + S3)
├── .gitignore             # Ignores venv, cache, local artifacts, and config files
└── readme.md              # Project documentation & revision notes
```

---

## 🚀 Quickstart: Running the Project Locally

### 1. Set Up Virtual Environment & Install Dependencies

```bash
# Clone the repository
git clone https://github.com/sujat-khan/MLFlow-dagshub-demo.git
cd MLFlow-dagshub-demo

# Create virtual environment
python -m venv myvenv

# Activate virtual environment
# On Windows:
.\myvenv\Scripts\activate
# On Linux / macOS:
# source myvenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. DagsHub Setup & Authentication

DagsHub provides a fully managed remote MLflow tracking server with zero cloud setup.

1. Create a repository on [DagsHub](https://dagshub.com).
2. The scripts initialize DagsHub integration automatically with:
   ```python
   import dagshub
   dagshub.init(repo_owner='sujat-khan', repo_name='MLFlow-dagshub-demo', mlflow=True)
   mlflow.set_tracking_uri("https://dagshub.com/sujat-khan/MLFlow-dagshub-demo.mlflow")
   ```
3. On the first run, DagsHub will prompt you in the terminal or browser to authenticate via an OAuth token.

---

### 3. Run Experiments

#### Run 1: Decision Tree Classifier
```bash
python iris-dt.py
```
- **Experiment Name**: `iris-dt`
- **Logged Params**: `max_depth`
- **Logged Metric**: `accuracy`
- **Logged Artifacts**: Confusion matrix heatmap (`confusion_matrix.png`) + original source script (`iris-dt.py`)
- **Logged Model**: Scikit-Learn Decision Tree model

#### Run 2: Random Forest Classifier
```bash
python iris-rf.py
```
- **Experiment Name**: `iris-rf`
- **Logged Params**: `max_depth`, `n_estimators`
- **Logged Metric**: `accuracy`
- **Logged Artifacts**: Confusion matrix heatmap (`confusion_matrix.png`) + original source script (`iris-rf.py`)
- **Logged Model**: Scikit-Learn Random Forest model

---

## 🔍 How to View and Compare Results

1. Go to your DagsHub repository: `https://dagshub.com/<username>/<repo_name>`.
2. Click on the **MLflow** tab or open the **Experiments** UI.
3. Compare runs side-by-side:
   - Filter by tag `author: Sujat`
   - Compare hyperparameter `max_depth` vs `accuracy`
   - Inspect the generated confusion matrix artifacts and inspect model weights directly.

---

## ☁️ Alternative: Self-Hosting MLflow on AWS (EC2 + S3)

In enterprise setups, teams often host their own private MLflow tracking server on AWS infrastructure instead of third-party platforms:
- **EC2 Instance**: Runs the MLflow server daemon (`mlflow server --host 0.0.0.0 --port 5000`).
- **S3 Bucket**: Serves as the remote artifact repository for plots and models.
- **SQLite / RDS**: Stores runs, metrics, and parameters metadata.

👉 **For the full, step-by-step AWS deployment walkthrough and systemd service setup, check [steps.md](file:///c:/Users/sujat/projects/ML-Main/MLFlow-dagshub-demo/steps.md).**

---

## 💡 Key Revision Takeaways

1. **`mlflow.log_artifact(__file__)` is a game changer**: Logging the exact source script with each run ensures 100% reproducibility even if code changes later in git.
2. **`log_model` vs `log_artifact`**: `log_model` stores the model in standard MLmodel format (with conda environment, pickle file, and model signature), enabling one-click serving or batch inference.
3. **Tracking URI Flexibility**: By switching `mlflow.set_tracking_uri(...)`, you can seamlessly route experiment logs to **Local (`./mlruns`)**, **DagsHub**, or **AWS EC2** without changing your core training logic.
