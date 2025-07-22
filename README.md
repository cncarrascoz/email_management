# Email Management System

## Project Overview

The goal of this project is to implement an email management system that automatically classifies emails from someone's inbox into predefined categories. As a first step to achieve this, we started by using the Enron Email Dataset, and training a Logistic Regression model with One-vs-Rest approach to be able to classify those emails first. After some extra tuning, the next step will be to do transfer learning and use my personal email dataset to fine-tune the model so it can correctly classify my emails.

The key features of this system include:

- Automated classification of emails into multiple relevant categories (single categories for now)
- Active learning approach that prioritizes uncertain emails for human review
- Threshold tuning for optimizing classification performance
- MLflow integration for experiment tracking and model versioning

## Dataset

The project uses the Enron Email Dataset, a publicly available collection of emails from the Enron Corporation. This dataset contains approximately 500,000 emails from 150 users, primarily senior management of Enron, and is one of the most widely used email datasets for research.

The raw emails undergo preprocessing steps including:

- Cleaning and normalization of text content
- Extraction of email metadata (sender, recipient, subject, etc.)
- Removal of irrelevant content (signatures, headers, etc.)
- Storage in optimized formats (Parquet files and DuckDB) for efficient processing

## Labeling Schema

Emails are classified into the following categories:

- **action_required**: Emails that clearly request a reply, confirmation, RSVP, or next step
- **company_business**: Internal strategy, projects, financials, or formal memos
- **purely_personal**: Non-work-related chat such as holiday greetings, weekend plans, memes
- **logistics**: Scheduling, travel itineraries, meeting room bookings, IT tickets
- **employment**: Hiring, onboarding, referrals, performance reviews, HR policy
- **newsletter**: Subscription digests, marketing blasts, press releases
- **spam**: Unsolicited bulk mail, phishing, obvious advertising
- **empty**: Blank body or placeholder text like 'see attachment' with no attachment
- **fwd_chain**: Forwarded emails or emails that are part of a forwarded chain

While each email can potentially belong to multiple categories, making this a multi-label classification problem, for now we will only use single category classification.

## Model Pipeline

The email triage system follows a comprehensive pipeline:

1. **Preprocessing**: Raw emails are cleaned, normalized, and relevant features are extracted
2. **Feature Engineering**: TF-IDF vectorization is applied to email content (combining subject and body)
3. **Training**: A baseline model using Logistic Regression with One-vs-Rest approach is trained on labeled data
4. **Scoring**: Unlabeled emails are scored by the model to generate probability distributions across categories
5. **Uncertainty Ranking**: Emails are ranked by uncertainty (using binary entropy) to identify candidates for manual labeling
6. **Threshold Tuning**: Per-label thresholds are optimized to maximize F1 scores
7. **Evaluation**: Model performance is assessed using precision, recall, and F1 metrics (both macro and micro averages)

## Active Learning Strategy

The project implements an active learning loop to efficiently improve the model while minimizing manual labeling effort:

1. Train an initial model on a small set of labeled emails
2. Score all unlabeled emails using this model
3. Calculate uncertainty for each email (using binary entropy across all label probabilities)
4. Select the most uncertain emails for manual labeling (those the model is least confident about)
5. After labeling, incorporate the newly labeled data into the training set
6. Retrain the model and repeat the process

This approach focuses human labeling effort on the most informative examples, accelerating model improvement compared to random sampling.

## Experiment Tracking

The project uses MLflow to track experiments and model performance:

- **Parameters**: Model hyperparameters (n-gram range, regularization strength, etc.)
- **Metrics**: Precision, recall, and F1 scores (both macro and micro averages)
- **Artifacts**: Trained models, label binarizers, optimized thresholds, and performance reports
- **Runs**: Different training iterations as the labeled dataset grows

MLflow provides a centralized location to compare different model versions and track improvements over time.

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
│   └── src/               # Source code
│       ├── merge.py       # Data merging utilities
│       ├── score_unlabeled.py  # Script for scoring unlabeled emails
│       ├── train.py       # Model training script
│       ├── utils_data.py  # Data handling utilities
│       └── utils_metrics.py  # Evaluation metrics
├── requirements.txt       # Project dependencies
└── setup_env.sh           # Environment setup script
```

## To-Do / Future Work
- Implement more sophisticated models (transformers, BERT, etc.)
- Add data augmentation techniques or specifically train on the least represented classes to handle imbalanced classes
- Create a web interface (or integrate to an existing one) to manually label emails
- Explore ensemble methods for improved classification
- Conduct comprehensive error analysis
- Evaluate model performance on different email domains
- Implement transfer learning to fine-tune the model on my personal email dataset
- Integrate with the Gmail API to automatically label emails