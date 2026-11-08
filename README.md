# Bankruptcy Prediction with Logit and Probit Models

Econometric modelling of company bankruptcy risk from financial ratios, using logit and probit regressions in a Jupyter notebook (Python).

## Objective

Identify the financial drivers of corporate bankruptcy and measure how well classic binary-choice models can predict it. The approach is close to the credit scoring methods used in financial analysis and banking.

## Dataset

- **Source:** [Company Bankruptcy Prediction](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction) (Taiwan Economic Journal, 1999-2009)
- **6,819 companies** described by 93 financial ratios, with no missing values
- **Target variable (`Class`):** bankrupt (1) or not (0). Only 3.23% of companies went bankrupt, so the classes are strongly imbalanced
- `taiwan_bankruptcy_clean.csv` is the cleaned version of the original file: the target is renamed `Class`, column names are simplified, and two non-informative flag columns are removed

## Methodology

1. **Data cleaning:** winsorisation of extreme values (1st and 99th percentiles)
2. **Variable selection:** 7 non-redundant financial ratios covering profitability, leverage, liquidity and capital structure
3. **Multicollinearity check:** VIF below 7 for all selected variables
4. **Estimation:** logit and probit models with `statsmodels`, on a stratified 70/30 train/test split (4,773 / 2,046 companies)
5. **Evaluation:** confusion matrix, ROC-AUC, McFadden pseudo-R²
6. **Interpretation:** marginal effects of the logit model, evaluated at the mean

## Key results

| Metric | Logit | Probit |
|---|---|---|
| McFadden pseudo-R² | 0.387 | 0.399 |
| AIC | 850.6 | 834.2 |
| AUC (test set) | 0.929 | 0.929 |

**Marginal effects (logit, significant at 5%)**

| Variable | Effect on bankruptcy probability |
|---|---|
| Borrowing dependency | +29.7 points |
| Persistent EPS (last four seasons) | -25.1 points |
| Current liability / current assets | +12.1 points |
| Retained earnings / total assets | -11.3 points |
| Debt ratio | +7.6 points |

## Summary of results

**1. Both models fit the data well.** A McFadden pseudo-R² of about 0.39-0.40 is considered a very good fit for a binary model (the usual reference range for a good model is 0.2 to 0.4). Logit and probit give almost identical results, so the conclusions do not depend on the choice of model.

**2. The models rank risky companies very well.** An AUC of 0.93 means that, if you pick one bankrupt and one healthy company at random, the model gives the higher risk to the bankrupt one about 93% of the time (0.5 would be pure chance, and above 0.9 is considered excellent).

**3. What drives bankruptcy.**
- Risk goes **up** with dependence on borrowing, a high ratio of current liabilities to current assets, and a high debt ratio. Borrowing dependency has the strongest effect.
- Risk goes **down** with stable earnings per share and with retained earnings relative to total assets.
- Net income / total assets and working capital / total assets are not significant once the other ratios are included. Their information probably overlaps with the other profitability and liquidity ratios in the model.
- These findings agree with classic credit risk theory. Retained earnings over total assets, for example, is one of the ratios of the Altman Z-score. The effects are statistical associations, not proof of causality.

**4. Main limitation: the default threshold misses most bankruptcies.** At the usual 0.5 threshold, the logit model reaches 97% accuracy but detects only 10 of the 66 bankruptcies in the test set (recall of about 15%, precision of about 48%). The accuracy is misleading, because 97% of the companies are healthy and a model that always predicts "healthy" would score the same. The high AUC shows that the ranking is good, so the problem is the threshold, which is not suited to such imbalanced classes. Lowering it (for example to 0.15-0.20) or rebalancing the classes would detect more bankruptcies, at the cost of more false alarms.

**5. Conclusion.** Simple, interpretable models are enough to identify which companies are most at risk, and they point to leverage and dependence on borrowing as the main warning signals. To use the model as an early-warning tool, the decision threshold would need to be calibrated to the real cost of missing a bankruptcy compared with a false alarm.

## Repository structure

```
├── explore_data.ipynb
├── analysis.ipynb                # Full analysis (code, outputs, charts)
├── taiwan_bankruptcy_clean.csv   # Cleaned dataset
├── requirements.txt              # Python dependencies
└── README.md
```

## How to run

```bash
# 1. Create and activate a virtual environment (Windows PowerShell)
python -m venv bankruptcy_env
.\bankruptcy_env\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open analysis.ipynb in VS Code, or run
jupyter notebook analysis.ipynb
```

## Tech stack

Python, Jupyter, pandas, NumPy, statsmodels, scikit-learn, matplotlib, seaborn

## Possible improvements

- Tune the decision threshold or rebalance the classes (SMOTE, undersampling)
- Compare with LASSO for automatic variable selection
- Compare with non-linear models (random forest, XGBoost)

