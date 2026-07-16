# Loan Default Prediction App

🔗 **Live App:** [https://loan-default-predictors.streamlit.app/](https://loan-default-predictors.streamlit.app/)

---

## Overview

This project is an end-to-end machine learning pipeline that predicts whether a loan applicant is likely to default, built for lenders and financial institutions evaluating loan applications. It combines unsupervised clustering with supervised classification, and wraps the final model in an interactive, explainable web application.

The goal isn't just to predict *whether* someone will default — it's to explain *why*, in plain language, so that lenders (including non-technical staff) can trust and act on the model's output.

---

## What This App Does

1. **Takes applicant details as input** — income, loan amount, employment type, co-signer status, and other standard loan application fields.
2. **Predicts default risk** using a tuned Random Forest classifier, selected as the best-performing model based on F1-score.
3. **Explains every prediction** using SHAP (SHapley Additive exPlanations) — showing exactly which factors pushed a specific applicant's risk up or down.
4. **Provides a global insights view** — a SHAP summary plot showing which features matter most across all applicants, useful for understanding overall model behavior, not just individual cases.

---

## How It Works (Under the Hood)

### 1. Data & Preprocessing
- Built on the **Kaggle Loan Default Prediction** dataset.
- Categorical fields (e.g., Employment Type, Co-Signer status) are encoded; numerical fields are scaled using `StandardScaler`.

### 2. Clustering as a Feature (Exploratory)
- K-Means clustering was applied to segment applicants into behavioral groups.
- Cluster ID was tested as an additional input feature for the classifiers.
- **Finding:** Adding Cluster ID gave minimal improvement in F1-score. This makes sense — the classifiers already had direct access to the same raw features used to form the clusters, so the cluster label carried little *new* information. This was treated as a legitimate, informative negative result rather than a failure.

### 3. Model Training & Selection
- Two model families were tuned via `GridSearchCV`: **Logistic Regression** and **Random Forest**.
- Models were trained and evaluated on both the feature set with clustering (Model B) and without (Model A) to directly test the clustering hypothesis above.
- The **Random Forest** classifier was selected as the best model based on F1-score.

### 4. Explainability
- **SHAP TreeExplainer** was chosen over raw Logistic Regression coefficients, since SHAP values are more intuitive for non-technical, lender-facing audiences and work natively with tree-based models.
- Per-prediction explanations are generated live for each applicant.
- A **global SHAP summary** (precomputed on a sample of applicants) is shown on the Insights page to avoid slow load times.

### 5. The App
- Built with **Streamlit**, structured as a two-page app:
  - **Prediction Page** — enter an applicant's details, get an instant prediction with a per-applicant SHAP explanation.
  - **Insights Page** — view the global SHAP summary plot to understand what drives predictions across the whole dataset.

---

## Reading the SHAP Summary Plot

On the Insights page, each dot represents one applicant from a sample of the data:

- **Vertical position** — features are ranked top to bottom by overall importance.
- **Color** — red means a high value for that feature, blue means a low value.
- **Horizontal position** — dots to the right of center push the prediction toward *Default*; dots to the left push toward *Non-Default*.

This lets you see, at a glance, both *which* features matter most and *how* their values relate to risk.

---

## Tech Stack

- **Python** — pandas, numpy, scikit-learn
- **Modeling** — Random Forest & Logistic Regression, tuned via GridSearchCV
- **Clustering** — K-Means
- **Explainability** — SHAP (TreeExplainer)
- **App Framework** — Streamlit
- **Notebooks** — built programmatically with `nbformat`
- **Version Control** — Git / GitHub

---

## Project Structure

```
loan-default-prediction/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── models/
│   ├── scaler.pkl                  # Fitted StandardScaler
│   ├── kmeans_model.pkl            # Fitted K-Means model
│   ├── best_model.pkl              # Final selected classifier (Random Forest)
│   ├── model_a.pkl                 # Model trained without cluster feature
│   ├── model_b.pkl                 # Model trained with cluster feature
│   ├── shap_summary_values.pkl     # Precomputed global SHAP values (for fast Insights page load)
│   ├── features_a.pkl              # Feature list for Model A
│   └── features_b.pkl              # Feature list for Model B
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_preprocessing.ipynb      # Cleaning, encoding, scaling
│   ├── 03_clustering.ipynb         # K-Means clustering experiments
│   └── 04_classification.ipynb     # Model tuning, selection, SHAP setup
└── README.md
```

> **Note:** The SHAP explainer object itself is **not** stored as a pickle file — it's rebuilt at runtime directly from the loaded model. This was a deliberate choice to avoid an unnecessarily large file (SHAP's internal tree representation can be significantly larger than the model itself) with no accuracy trade-off, since reconstructing it at startup is fast.

---

## Running Locally

```bash
git clone https://github.com/rayaan27/loan-default-prediction.git
cd loan-default-prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## Results

| Model | Variant | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | A (no cluster) | 0.6763 | 0.2195 | 0.6994 | 0.3342 |
| Logistic Regression | B (with cluster) | 0.6768 | 0.2197 | 0.6990 | 0.3344 |
| Random Forest | A (no cluster) | 0.7088 | 0.2342 | 0.6645 | 0.3464 |
| **Random Forest** | **B (with cluster)** | **0.8386** | **0.3301** | **0.3785** | **0.3527** |

**Best model: Random Forest (Variant B, with cluster feature)** — selected based on F1-score.

**On the clustering result:** Random Forest with the cluster feature (B) does edge out Random Forest without it (A) on F1 (0.3527 vs 0.3464), but the gain is small — about 1.8% relative improvement — and comes with a large jump in accuracy alongside a drop in recall (0.3785 vs 0.6645). This trade-off reflects a shift toward fewer false positives at the cost of catching fewer actual defaulters, rather than the cluster feature adding meaningful new predictive signal. This is consistent with the project's core finding: since the classifier already has direct access to the same raw features used to build the clusters, Cluster ID mostly reinforces existing signal rather than introducing new information.

**On Logistic Regression vs Random Forest:** Random Forest outperforms Logistic Regression across both variants, which is expected given its ability to capture non-linear relationships and feature interactions in loan default patterns that a linear model can't.

---

## Key Design Decisions & Lessons

- **Real data over synthetic** — the pipeline was built and validated on the actual Kaggle dataset throughout, not synthetic placeholders, to keep results meaningful.
- **Negative results are valid results** — the clustering-as-a-feature experiment didn't improve performance, and that finding is presented transparently with its reasoning, rather than omitted.
- **Explainability matters as much as accuracy** — SHAP was chosen specifically because it produces explanations a non-technical lender can act on, not just a probability score.
- **Model size vs. deployability** — deploying on Streamlit Community Cloud surfaced real-world constraints (GitHub's 100MB file limit, memory ceilings). This was addressed by rebuilding the SHAP explainer at runtime instead of shipping a massive precomputed pickle, and by compressing saved model artifacts with `joblib`'s built-in compression to keep them within GitHub's push limits.

---

## Disclaimer

This tool is a demonstration/capstone project and is not a certified credit-scoring system. Predictions and explanations are intended to support human decision-making, not replace it.
