# CoCrystalBoost

LightGBM pipeline for cocrystallization prediction with RDKit-based pair features.

## ✨ Overview

The repository includes:
- a Python package for feature generation, training, inference, and tuning
- prepared data files for local runs
- simple scripts for prediction and Optuna-based hyperparameter search

## 💊 Why It Matters

Pharmaceutical cocrystals are an important strategy for improving the solubility, stability, and bioavailability of drug compounds. Reliable prediction of cocrystal formation can help speed up drug development and reduce the cost of experimental screening.

## 📁 Structure

```text
CoCrystalBoost/
├── cocrystalboost/
├── data/
├── notebooks/
├── scripts/
├── pyproject.toml
└── README.md
```

## ⚙️ Requirements

- Python 3.10+
- pip
- RDKit

Dependencies are defined in [pyproject.toml](/home/skihnn/Projects/CoCrystalBoost/pyproject.toml).

## 🚀 Installation

```bash
git clone <https://github.com/romandolgo/CoCrystalBoost.git>
cd CoCrystalBoost
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

If RDKit is easier to install through Conda on your machine:

```bash
conda create -n cocrystalboost python=3.11
conda activate cocrystalboost
conda install -c conda-forge rdkit
pip install -e .
```

## 🧪 Run Prediction

```bash
python -m cocrystalboost
```

or:

```bash
python scripts/predict.py
```

This creates `submission.csv`.

## 🎯 Tune LightGBM

```bash
python -m cocrystalboost.tuning
```

or:

```bash
python scripts/tune_lgbm.py
```

This creates `lgbm_params_generated.py`. If that file exists, the main pipeline uses it automatically. Otherwise, default parameters from [settings.py](/home/skihnn/Projects/CoCrystalBoost/cocrystalboost/settings.py) are used.

## 📊 Data

Expected files:
- `data/train_dataset/train_extended.csv`
- `data/test.csv`
- `data/sample_submission.csv`

Expected columns:
- train: `SMILES1`, `SMILES2`, `result`
- test: `SMILES1`, `SMILES2`

## 🧩 Main Modules

- [cocrystalboost/main.py](/home/skihnn/Projects/CoCrystalBoost/cocrystalboost/main.py) — prediction pipeline
- [cocrystalboost/tuning.py](/home/skihnn/Projects/CoCrystalBoost/cocrystalboost/tuning.py) — Optuna tuning
- [cocrystalboost/features.py](/home/skihnn/Projects/CoCrystalBoost/cocrystalboost/features.py) — feature engineering
- [cocrystalboost/modeling.py](/home/skihnn/Projects/CoCrystalBoost/cocrystalboost/modeling.py) — training and threshold selection

## 📝 Notes

- F1 is used for threshold selection and tuning.
- Grouped cross-validation is used to reduce leakage between equivalent pairs.
- Generated files such as `submission.csv`, `train_features_cache.pkl`, and `lgbm_params_generated.py` are ignored by git.
