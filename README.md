# Stroke Prognosis Modeling (Manuscript Code)

This repository contains cleaned, script-based code extracted from the notebook `卒中预后25.12.2.ipynb` for reproducible execution and sharing on GitHub.

## Repository structure

- `src/stroke_prognosis/` – reusable modules (data I/O, model zoo, evaluation, DeLong test)
- `scripts/` – command-line entry points
- `notebooks/`
  - `pipeline_quickstart.ipynb` – minimal runnable notebook
  - `original_cleaned.ipynb` – the original notebook with outputs cleared
- `data/` – **place your input data here** (not committed)
- `outputs/` – generated artifacts (metrics, predictions, plots)

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Train + compare multiple models

```bash
python scripts/train_compare_models.py \
  --data data/train.xlsx \
  --label_col outcome \
  --test_size 0.2 \
  --seed 42 \
  --out_dir outputs/model_comparison
```

Outputs:
- `outputs/model_comparison/model_metrics.csv`
- `outputs/model_comparison/predictions_for_delong.xlsx`
- `outputs/model_comparison/run_metadata.json`

## 2) DeLong test (pairwise AUC comparison)

```bash
python scripts/run_delong.py \
  --pred outputs/model_comparison/predictions_for_delong.xlsx \
  --a XGBoost \
  --b LogisticRegression
```

## Notes for SCI submission

- Absolute local paths from the original notebook were removed; scripts now use CLI args and relative paths.
- Outputs are cleared from notebooks to reduce repository size and avoid leaking patient-level results.
- If you need to add figure-generation steps (ROC, calibration, DCA, SHAP), implement them as scripts under `scripts/` so figures are reproducible from raw inputs.

## Citation

If you use this code in your paper, please cite your manuscript and include a link to this repository in the Methods / Data & Code availability section.
