# project/src/train.py
"""
Train baseline TF-IDF + Logistic One-vs-Rest multi-label classifier.

Usage:
    python -m project.src.train \
        --labeled project/data/processed/train_labeled_0.parquet \
        --label-yml project/label.yml \
        --model-dir project/mlruns/baseline_run0 \
        --val-size 0.2 \
        --random-state 42 \
        --ngram-max 2 \
        --min-df 5 \
        --max-df 0.9 \
        --C 1.0

Re-run this script after adding new labeled data to retrain from scratch.
"""

import argparse, json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from .utils_data import load_label_yaml, load_labeled_parquet
from .utils_metrics import binarize_probs, best_threshold_per_label, multi_label_report

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--label_yml", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--max-df", type=float, default=0.9)
    ap.add_argument("--C", type=float, default=1.0)
    return ap.parse_args()

def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    label_names = load_label_yaml(args.label_yml)
    df = load_labeled_parquet(args.labeled)
    print(f"Loaded {len(df)} labeled emails.")

    # Binarize label
    mlb = MultiLabelBinarizer(classes=label_names)
    Y = mlb.fit_transform(df["label"])
    # Making sure the context of both subject and clean_body is considered
    X = df["subject"].astype(str) + " " + df["clean_body"].astype(str)

    # Train/val split
    # For single-label data stored in multi-label format, extract first label for stratification
    stratify_labels = None
    if Y.sum(axis=1).max() == 1:  # Check if it's actually single-label
        stratify_labels = Y.argmax(axis=1)

    Xtr, Xval, Ytr, Yval = train_test_split(
        X, Y,
        test_size=args.val_size,
        random_state=args.random_state,
        shuffle=True,
        stratify=stratify_labels  # Would change it to stratify_labels = Y for multi-label
    )

    # Build pipeline
    vectorizer = TfidfVectorizer(
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        strip_accents="unicode",
        lowercase=True,
    )

    # Logistic regression with One-vs-Rest approach
    base_clf = LogisticRegression(
        penalty="l2",
        C=args.C,
        solver="liblinear",  # Small data, binary OVR
        class_weight="balanced",
        max_iter=1000,
    )
    clf = OneVsRestClassifier(base_clf)

    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("clf",   clf),
    ])

    mlflow.set_experiment("email-triage")
    with mlflow.start_run():
        mlflow.log_params({
            "val_size": args.val_size,
            "ngram_max": args.ngram_max,
            "min_df": args.min_df,
            "max_df": args.max_df,
            "C": args.C,
            "n_label": len(label_names),
            "n_train_examples": len(Xtr),
            "n_val_examples": len(Xval),
        })

        pipe.fit(Xtr, Ytr)

        # Probabilities (liblinear gives decision_function but we’ll use predict_proba if available)
        # Some scikit versions don't expose predict_proba in OneVsRest(LogisticRegression(solver='liblinear')).
            # Fallback: use decision_function -> sigmoid.
        try:
            Yval_prob = pipe.predict_proba(Xval)
        except AttributeError:
            import scipy.special
            scores = pipe.decision_function(Xval)
            Yval_prob = scipy.special.expit(scores)

        # Tune per-label thresholds on validation
        thresh = best_threshold_per_label(Yval, Yval_prob)
        Yval_pred = (Yval_prob >= thresh).astype(int)

        macro = multi_label_report(Yval, Yval_pred, average="macro")
        micro = multi_label_report(Yval, Yval_pred, average="micro")

        print("Validation macro:", macro)
        print("Validation micro:", micro)
        print("Thresholds per label:", dict(zip(label_names, thresh)))

        mlflow.log_metrics({
            "macro_precision": macro["precision"],
            "macro_recall":    macro["recall"],
            "macro_f1":        macro["f1"],
            "micro_precision": micro["precision"],
            "micro_recall":    micro["recall"],
            "micro_f1":        micro["f1"],
        })

        # Save artifacts for later use
        dump(pipe, model_dir / "model.joblib")
        dump(mlb,  model_dir / "mlb.joblib")
        np.save(model_dir / "thresholds.npy", thresh)

        metrics = {
            "macro": macro,
            "micro": micro,
            "thresholds": {k: float(v) for k, v in zip(label_names, thresh)},
            "label_names": label_names,
        }
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact(model_dir / "model.joblib")
        mlflow.log_artifact(model_dir / "mlb.joblib")
        mlflow.log_artifact(model_dir / "metrics.json")

if __name__ == "__main__":
    main()
