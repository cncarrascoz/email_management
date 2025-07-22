# project/src/utils_metrics.py
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def binarize_probs(probs, thresh=0.5):
    return (probs >= thresh).astype(int)

def best_threshold_per_label(y_true, y_prob, grid=None):
    """
    Simple per-label threshold sweep to maximize F1.
    y_true, y_prob: shape (N, L)
    Returns array of thresholds length L.
    """
    if grid is None:
        grid = np.linspace(0.1, 0.9, 17)
    L = y_true.shape[1]
    best = np.empty(L)
    for j in range(L):
        t_best, f_best = 0.5, -1.0
        for t in grid:
            y_pred = (y_prob[:, j] >= t).astype(int)
            f = f1_score(y_true[:, j], y_pred, zero_division=0)
            if f > f_best:
                t_best, f_best = t, f
        best[j] = t_best
    return best

def multi_label_report(y_true, y_pred, average="macro"):
    return {
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall":    recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1":        f1_score(y_true, y_pred, average=average, zero_division=0),
    }
