import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
import json
from sklearn.metrics import classification_report

# Load test set
df = pd.read_parquet("../data/processed/test_labeled_200_0.parquet")
texts = df["subject"].astype(str) + " " + df["clean_body"].astype(str)
true_labels = df["label"]

# Load run information
model_dir = Path("../mlruns/run3")
pipe = load(model_dir / "model.joblib")
mlb = load(model_dir / "mlb.joblib")
thresholds = np.load(model_dir / "thresholds.npy")
label_names = list(mlb.classes_)

# Predict probabilities
probs = pipe.predict_proba(texts)

# Apply thresholds to convert to predicted labels
pred_labels = []
for row_probs in probs:
    above_thresh = [label_names[i] for i, p in enumerate(row_probs) if p >= thresholds[i]]
    if above_thresh:
        pred_labels.append(above_thresh[0])  # single-label setting: pick most confident
    else:
        pred_labels.append(label_names[np.argmax(row_probs)])  # otherwise fallback to top prob

# Evaluate
with open("../results/class_rep_tfidf_run3.txt", "w") as f:
    f.write(classification_report(true_labels, pred_labels, labels=label_names, zero_division=0))
