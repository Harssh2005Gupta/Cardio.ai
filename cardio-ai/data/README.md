# Data

Place your real UCI-style heart disease CSV here as `heart.csv` (14 columns:
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope,
ca, thal, target). This directory is gitignored for CSVs — raw patient data
is never committed.

## sample_synthetic_heart.csv

A small **synthetically generated** dataset (not real patient data, not the
real UCI dataset) included only so `python -m src.models.train --data
data/sample_synthetic_heart.csv` and the rest of the pipeline can be run and
demoed immediately after cloning, with no data-sourcing step. The `models/`
and `reports/` artifacts currently committed in this repo were trained on
this synthetic file — **retrain on your real dataset before drawing any
conclusion from the reported metrics.**
