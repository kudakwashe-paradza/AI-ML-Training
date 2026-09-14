# AI-ML-Training

Notebooks exploring categorical encoding techniques for machine learning, applied to the [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (predicting `SalePrice`).

## Contents

- **`house_prices_one_hot_encoding.ipynb`** — Baseline pipeline: cleans the data, one-hot encodes categorical columns, trains Linear Regression and Random Forest models, and generates a Kaggle-style submission file.
- **`house_prices_encoding_comparison.ipynb`** — Compares every major categorical encoding technique on the same dataset, model, and train/validation split:
  - Method 1: Label Encoding
  - Method 2: One-Hot Encoding
  - Method 3: Feature Hashing
  - Method 4: Encoding categories with dataset statistics (frequency encoding)
  - Cyclic features: `MoSold` (month sold) via sine/cosine
  - Method 5: Target Encoding
  - Method 6: K-Fold Target Encoding
  - Summary: side-by-side model performance (RMSE, R², runtime) across all methods
- **`Features.ipynb`** — Feature engineering exploration.
- **`Kudakwashe Paradza - an-overview-of-encoding-techniques.ipynb`** — Original reference notebook covering the encoding techniques (built around a different, categorical-classification dataset).
- **`data_description.txt`** — Column-by-column description of the Ames Housing dataset.
- **`train.csv` / `test.csv`** — Ames Housing training and test data.
- **`sample_submission.csv`** — Kaggle submission format example.

## Dataset

The Ames Housing dataset contains 79 explanatory variables describing residential homes, with the goal of predicting each home's final sale price (`SalePrice`). It's a mix of numeric and categorical columns, several with high cardinality, which makes it a good testbed for comparing encoding strategies.

## Key results

Random Forest performance by encoding method (RMSE, lower is better):

| Method | RMSE | R² |
|---|---|---|
| Target Encoding (leaky) | 28,390 | 0.895 |
| Label Encoding | 28,402 | 0.895 |
| Dataset Statistics (Frequency) | 28,402 | 0.895 |
| K-Fold Target Encoding | 28,952 | 0.891 |
| Cyclic (MoSold) + One-Hot | 28,977 | 0.891 |
| One-Hot Encoding | 29,026 | 0.890 |
| Feature Hashing | 42,352 | 0.766 |

See `house_prices_encoding_comparison.ipynb` for the full writeup, including why the "leaky" target encoding score is misleadingly good and why feature hashing underperforms on this particular dataset.

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
jupyter
```

Install with:

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

## Running the notebooks

1. Clone the repo and `cd` into it.
2. Make sure `train.csv` and `test.csv` are present in the same folder as the notebooks (the notebooks auto-detect any CSV with "train"/"test" in the filename, so renamed copies like `train (1).csv` work too).
3. Launch Jupyter and run all cells:

```bash
jupyter notebook
```