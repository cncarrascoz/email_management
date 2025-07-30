# Email Management System

## Project Overview

Motivated by the challenge of managing an inbox with a high number of daily emails, and by how tedious it is to create labels to classify them on Gmail, I decided to solve the issue by developing an ML system to automatically classify emails into categories such as `action_required`, `logistics`, and `fwd_chain`, with the goal of improving email organization. Using the Enron Email Dataset — a 500K publicly available dataset of corporate communications — I built an end-to-end classification pipeline and iteratively improved it through active learning and manual labeling.

This project is still in the very early stages, however, the goal is for it to become a productivity tool where models assist users in managing their inboxes. It combines classical NLP techniques, uncertainty-based sampling, and hands-on annotation to demonstrate how machine learning can drive productivity tools.

Key features:
- Email classification across nine categories
- Active learning with uncertainty-based sampling
- Comparison of TF-IDF and DistilBERT models
- MLflow experiment tracking

**For detailed information about the methodology, model performance, and findings, please see the [project report](/project/report/email_report.pdf).**

## Reproducing Results

Results can be reproduced by following these steps:

## Instructions for Running the Project

### Setup

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the setup script:
   ```bash
   bash setup_env.sh
   ```

### Training the Model

```bash
python -m project.src.train \
    --labeled project/data/processed/train_labeled_0.parquet \
    --label-yml project/labels.yml \
    --model-dir project/mlruns/baseline_run0 \
    --val-size 0.2 \
    --random-state 42 \
    --ngram-max 2 \
    --min-df 5 \
    --max-df 0.9 \
    --C 1.0
```

### Scoring Unlabeled Emails

```bash
python -m project.src.score_unlabeled \
    --unlabeled project/data/processed/train_unlabeled_0.parquet \
    --model-dir project/mlruns/baseline_run0 \
    --out-parquet project/mlruns/baseline_run0/train_unlabeled_0_scored.parquet \
    --next-batch-csv project/data/interim/next_label_batch.csv \
    --batch-size 50000 \
    --top-n 200
```

### Launching MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open http://127.0.0.1:5000 in your browser to view experiment results.

## Directory Structure

```
├── mlruns/                # MLflow experiment tracking data
├── project/
│   ├── data/
│   │   ├── interim/       # Intermediate processed data
│   │   ├── processed/     # Final processed datasets
│   │   └── raw/           # Original raw data
│   ├── labels.yml         # Label definitions
│   ├── notebooks/         # Jupyter notebooks for exploration
│   ├── report/            # Project report and documentation
│   │   └── email_report.pdf # PDF report with detailed findings
│   ├── results/           # Results and visualizations
│   │   ├── confusion_matrix_*.png # Model confusion matrices
│   │   └── class_rep_*.txt      # Classification reports
│   └── src/               # Source code
│       ├── evaluate_test.py    # Test evaluation script
│       ├── merge.py       # Data merging utilities
│       ├── score_unlabeled.py  # Script for scoring unlabeled emails
│       ├── train.py       # Model training script
│       ├── utils_data.py  # Data handling utilities
│       └── utils_metrics.py  # Evaluation metrics
├── requirements.txt       # Project dependencies
└── setup_env.sh           # Environment setup script
```

## Next Steps

For future development plans, please take a look at the [project report](/project/report/email_report.pdf).