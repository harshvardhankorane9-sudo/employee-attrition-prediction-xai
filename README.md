# Employee Attrition Prediction System

Explainable, fairness-audited, and business-quantified machine learning system for predicting employee attrition, built on a 14,900-employee HR dataset (53% Stayed / 47% Left).

The project goes beyond a single classifier: it benchmarks 8 model families, audits the chosen model for demographic bias, explains individual predictions with SHAP/LIME/counterfactuals, segments at-risk employees into actionable archetypes, and quantifies the resulting cost/benefit in ₹.

## Results at a glance

| Metric | Value |
|---|---|
| Best single model | XGBoost (manually tuned) — Accuracy 75.0%, ROC-AUC 0.843 |
| Best in 8-model comparison | AdaBoost — ROC-AUC 0.848 (XGBoost 0.843, close second) |
| Optuna-tuned XGBoost | Accuracy 75.1%, ROC-AUC 0.846 (+0.003 AUC over manual tuning) |
| Fairness gap found | Marital Status: Equal Opportunity Difference 0.260 (fails 4/5ths rule, DIR 0.44) |
| Fairness after mitigation | Equal Opportunity Difference reduced to 0.017 (threshold adjustment, −1.5pp accuracy) |
| Projected annual savings | ₹20.32 Crore (35% intervention success rate assumption; positive across full sensitivity range) |
| Risk segments identified | 7 SHAP-based archetypes; largest is "Career Stagnation Risk" (39.2% of at-risk pool) |

All figures above are reproduced from a full top-to-bottom execution of `EAPS_with_SHAP_LIME.ipynb` — see [Reproducing results](#reproducing-results).

## What's in this repo

```
.
├── EAPS_with_SHAP_LIME.ipynb   # Main notebook: EDA → modeling → explainability → fairness → business case
├── app.py                      # Streamlit app for interactive single-employee prediction
├── employee_attrition.csv      # Dataset (14,900 rows, 24 columns)
├── best_model.pkl              # Trained XGBoost model (joblib), loaded by app.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Notebook structure

| # | Section | What it produces |
|---|---|---|
| 1–2 | Imports, load & inspect | Data quality checks |
| 3 | Exploratory Data Analysis | Univariate/bivariate stats, outlier detection, Mann-Whitney & Chi-square tests, correlation analysis, feature ranking |
| 4 | Preprocessing | Encoding, stratified train/test split |
| 4A | SMOTE | Class-imbalance correction on training data only |
| 4B | Multi-model comparison | 8 classifiers benchmarked on the same untouched test set |
| 5–6 | XGBoost training & evaluation | The model used throughout the rest of the notebook |
| 7–10 | SHAP + LIME | Global & local explainability, side-by-side comparison |
| 11 | Interactive explanation function | `explain_employee(index)` — full explanation for any test-set employee |
| 12 | Counterfactual explanations | "What would this employee need to change to reduce their leave risk?" (DiCE-ML, with a random-search fallback if DiCE is unavailable in the runtime) |
| 13–14 | Fairness & bias audit | Demographic parity, disparate impact, equal opportunity, equalized odds across Gender / Marital Status / Age Group, with a group-specific threshold mitigation |
| 15 | Risk segmentation | K-Means clustering of SHAP profiles into named, actionable archetypes |
| 16 | Business impact model | Cost-of-attrition quantification, sensitivity analysis |
| 17 | Literature review | 15 papers (2018–2026), verified citations, positioning statement |
| 18 | Enhanced EDA | Chi-square effect sizes, tenure cohort analysis |
| 19 | Optuna hyperparameter tuning | 5-fold CV search, compared against the manually tuned Section 5 model |
| 20 | Fairlearn audit & mitigation | `ExponentiatedGradient` in-processing bias mitigation, cross-checked against the Section 13 audit |
| 21 | UMAP visualization | Alternative 2D projection of risk archetypes vs. PCA |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the notebook

```bash
jupyter notebook EAPS_with_SHAP_LIME.ipynb
```

Run all cells top-to-bottom. Sections 12 (counterfactuals), 19 (Optuna), 20 (Fairlearn), and 21 (UMAP) install their own dependencies if missing, and each writes intermediate tables/figures to disk (`table*.csv`, `figure_*.png`, `paper_paragraph_*.txt`) so results can be pulled directly into a report without re-running the notebook.

## Running the app

```bash
streamlit run app.py
```

This loads `best_model.pkl` and provides sliders/dropdowns for all employee attributes, returning a live attrition-risk prediction.

## Reproducing results

The headline numbers in this README come from a complete, unmodified execution of the notebook (all 21 sections, 179 cells) — the same one committed here. If your numbers differ after re-running:
- Random seeds are fixed (`random_state=42`) everywhere sklearn/XGBoost/Optuna support it, so results should match closely, but library version differences (especially `xgboost`, `shap`, `umap-learn`) can shift metrics by a few hundredths.
- Section 19 (Optuna) and Section 12 (DiCE fallback) involve randomized search, so exact figures may vary slightly run to run even with a fixed seed depending on library version.

## Key caveats (see the notebook's own "Limitations" callouts for full detail)

- **Synthetic-style dataset:** results should not be assumed to generalize to a different organization's population without re-validation.
- **Fairness mitigation trade-off:** the per-group threshold adjustment in Section 14.4 reduces the equal-opportunity gap at a real, quantified accuracy cost (75.0% → 73.5%) — this is a policy choice, not a free fix, and is documented as such.
- **Business impact figures depend on editable assumptions** (replacement-cost multiplier, intervention success rate) declared explicitly in Section 16.1 — change them for your own organization before quoting the ₹ figures externally.
- **Literature review citations were verified via live search during authoring**, not generated from memory — see Section 17.1 for a correction log of two originally-supplied citations that pointed to the wrong paper.

## License

MIT — see [LICENSE](LICENSE).
