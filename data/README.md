# Data Directory

This directory stores all datasets used by the Customer Churn Prediction platform.

## Structure

```
data/
├── raw/        ← Original, unmodified source data
├── processed/  ← Cleaned and feature-engineered data ready for training
└── README.md   ← This file
```

## Rules

### `raw/`
- Place original CSV, Parquet, or JSON files here **exactly as received**.
- **Never modify** raw files directly. They serve as the single source of truth.
- If a raw file needs corrections, apply them in a preprocessing step and write
  the result to `processed/`.
- Raw data files are excluded from version control (see `.gitignore`).  
  Document the source URL or acquisition method in the dataset provenance section below.

### `processed/`
- Contains derived datasets produced by preprocessing and feature-engineering
  scripts located in `src/training/`.
- Processed files are also excluded from version control to avoid bloating the
  repository with generated artefacts.
- Re-generate them by running the preprocessing pipeline (instructions will be
  added in Milestone 2).

## Dataset Provenance

> **Status:** Not yet acquired. Will be documented in Milestone 2.

When datasets are added, record the following for each:

| Field        | Value |
|--------------|-------|
| Name         |       |
| Source       |       |
| Licence      |       |
| Acquired on  |       |
| Acquired by  |       |
| Notes        |       |

## Large-File Policy

Files larger than ~10 MB should **not** be committed to the repository.
Use one of the following alternatives:
- Store in an object store (e.g., AWS S3, GCS, Azure Blob) and reference the
  URI in the provenance table.
- Track with [DVC](https://dvc.org/) if the team adopts it in a later milestone.
