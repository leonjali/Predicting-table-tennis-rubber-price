# Reproducibility package for inverted table-tennis rubber price prediction

This repository accompanies the IEEE ICDM 2026 Teen Research Track paper
**“Predicting the Prices of Inverted Table Tennis Rubbers Using Product Ratings
and Brand Information”** by Jiang Li.

It contains the fixed dataset snapshot, preprocessing and model code, robustness
experiments, machine-readable results, and figure-generation scripts used for
the camera-ready paper.

## Main findings

- The cleaned snapshot contains 376 products from 15 brands.
- Random forest achieved the best held-out performance: R² = 0.858,
  RMSE = EUR 9.49, and MAE = EUR 7.49.
- Removing brand increased five-fold cross-validation MAE from EUR 8.33 to
  EUR 9.43.
- Giving products priced at least EUR 80 three times the training weight
  reduced premium-product MAE from EUR 14.15 to EUR 12.54 on the five premium
  products in the held-out test set.
- Leave-one-brand-out evaluation produced a pooled MAE of EUR 13.69, showing
  limited transfer to completely unseen brands.

These results concern listed prices in this snapshot. They do not measure
product quality, player suitability, or causal effects.

## Repository structure

```text
.
├── data/
│   ├── inverted_table_tennis_rubber_prices_cleaned.csv
│   ├── DATA_DICTIONARY.md
│   └── LICENSE
├── figures/
├── results/
├── run_experiment.py
├── run_camera_ready_experiments.py
├── verify_data.py
├── generate_figures.py
├── generate_camera_ready_figure.py
├── run_all.py
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Quick start

Python 3.11 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The scripts resolve paths relative to their own location, so they can be
started from any working directory.

## Scripts

- `run_experiment.py` reproduces the five-model held-out comparison and
  five-fold cross-validation.
- `verify_data.py` checks the dataset checksum, dimensions, required columns,
  and key numeric fields before model training.
- `run_camera_ready_experiments.py` reproduces the log-target and premium-
  weighting tests, brand ablation, and leave-one-brand-out evaluation.
- `generate_figures.py` recreates Figures 1–5.
- `generate_camera_ready_figure.py` recreates the robustness figure after
  `run_camera_ready_experiments.py` has produced `results/results.json`.
- `run_all.py` runs the four scripts in the required order.

All experiments use random seed 42. Imputation, scaling, and one-hot encoding
are fitted inside each training split or cross-validation fold.

## Data provenance

The records were collected from the open RubberMatrix database
(https://rubbermatrix.com) between August 3 and August 6, 2026. The original
collection contained 464 product records. Records without a valid listed price
were removed, leaving the fixed 376-row snapshot used in the paper.

RubberMatrix ratings are website ratings, not laboratory measurements. Listed
prices can differ from transaction prices by retailer, country, tax, discount,
stock, and collection date. See `data/DATA_DICTIONARY.md` and `data/LICENSE`.

## Integrity

SHA-256 of the released CSV:

```text
52306a7af8522ecca297338bf5d3f059f805b7aab84f89694fa9c281dba7ded7
```

## Citation

Citation metadata are provided in `CITATION.cff`. GitHub can display a
“Cite this repository” button after the repository is published.

## Licenses

The analysis code is released under the MIT License in the root `LICENSE`
file. The cleaned dataset and its documentation are covered separately by
`data/LICENSE`, with source attribution to RubberMatrix.
