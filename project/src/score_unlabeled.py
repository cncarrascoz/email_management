# project/src/score_unlabeled.py
"""
Score unlabeled emails with trained model; produce uncertainty ranking.

Usage:
    python -m project.src.score_unlabeled \
        --unlabeled project/data/processed/train_unlabeled_0.parquet \
        --model-dir project/mlruns/baseline_run0 \
        --out-parquet project/mlruns/baseline_run0/train_unlabeled_0_scored.parquet \
        --next-batch-csv project/data/interim/next_label_batch.csv \
        --batch-size 50000 \
        --top-n 200
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from tqdm import tqdm
import scipy.special

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unlabeled", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-parquet", required=True)
    ap.add_argument("--next-batch-csv", required=True)
    ap.add_argument("--batch-size", type=int, default=50000)
    ap.add_argument("--top-n", type=int, default=200)
    return ap.parse_args()

def binary_entropy(p):
    # clip for numerical safety
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    pipe = load(model_dir / "model.joblib")
    mlb  = load(model_dir / "mlb.joblib")
    label_names = list(mlb.classes_)

    # load unlabeled in chunks using pyarrow
    df = pd.read_parquet(args.unlabeled, columns=["from", "to","msg_id", "cc", "bcc","clean_body", 
                                                  "subject", "date","has_other_content", "is_forwarded"], engine='pyarrow')
    df = df.reset_index()  # moves index into a new column named 'index'
    df.rename(columns={"index": "orig_index"}, inplace=True) # preserve original index
    n = len(df)
    probs_all = np.zeros((n, len(label_names)), dtype=np.float32)

    # predict_proba or decision_function fallback
    has_proba = hasattr(pipe, "predict_proba")
    for i in tqdm(range(0, n, args.batch_size), desc="Scoring unlabeled"):
        batch = df.iloc[i:i+args.batch_size]

        # concatenate subject and clean_body for text input
        # this way we are considering both fields for prediction
        texts = batch["subject"].astype(str) + " " + batch["clean_body"].astype(str)
        if has_proba:
            probs = pipe.predict_proba(texts)
        else:
            scores = pipe.decision_function(texts)
            probs = scipy.special.expit(scores)
        probs_all[i:i+args.batch_size] = probs

    # uncertainty = mean binary entropy across labels
    ent = binary_entropy(probs_all).mean(axis=1)

    out_df = df.copy()
    for j, name in enumerate(label_names):
        out_df[f"prob_{name}"] = probs_all[:, j]
    out_df["uncertainty"] = ent

    out_df.to_parquet(args.out_parquet, index=False)

    # pick top-N most uncertain for manual human labeling
    top = out_df.sort_values("uncertainty", ascending=False).head(args.top_n)
    
    # include a helpful preview of top predicted labels
    top_pred_labels = probs_all[top.index].argsort(axis=1)[:, ::-1]  # descending
    mapped = []
    for row_idx, label_idx_arr in zip(top.index, top_pred_labels):

        # choose labels above 0.5 OR top3
        probs_row = probs_all[row_idx]
        lbls = [label_names[k] for k in label_idx_arr[:3]]
        mapped.append(",".join(lbls))
        
    top = top.assign(top3_guess=mapped)
    top.to_csv(args.next_batch_csv, index=False)

    print(f"Wrote scored unlabeled to {args.out_parquet}")
    print(f"Wrote next {args.top_n} uncertain samples to {args.next_batch_csv}")

if __name__ == "__main__":
    main()
