# project/src/utils_data.py
import pandas as pd
import yaml
import ast
from pathlib import Path
from typing import List

# Load label from YAML file
def load_label_yaml(path: str | Path) -> List[str]:
    with open(path, "r") as f:
        yml = yaml.safe_load(f)
    
    # Expect structure: label: [ {name: ..., description: ...}, ... ]
    if isinstance(yml, dict) and "labels" in yml:
        return [entry["name"] for entry in yml["labels"] if "name" in entry]

    # Just in case we get a list of strings instead
    if isinstance(yml, list):
        return list(yml)
    
    raise ValueError("labels.yml not understood")

# Ensure label are a list of strings, handling various formats
def ensure_label_list(x):
    """Safe converter: parquet may store list, str, None"""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        # Try JSON-ish or python list literal
        try:
            v = ast.literal_eval(x)
            if isinstance(v, list):
                return [str(i) for i in v]
        except Exception:
            pass
        # Assume comma-separated
        return [s.strip() for s in x.split(",") if s.strip()]
    raise ValueError(f"Unrecognized label cell: {x!r}")

def load_labeled_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "label" not in df.columns:
        raise ValueError("Expected 'label' column in labeled parquet.")
    df["label"] = df["label"].apply(ensure_label_list)

    # # Already taken care of but dropping missing/empty text just in case
    # df = df[df["clean_body"].astype(str).str.strip().ne("")] # Keeping non-empty bodies for now
    return df.reset_index(drop=True)

def load_unlabeled_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["msg_id", "clean_body", "subject"])
    # df = df[df["clean_body"].astype(str).str.strip().ne("")] # Here too
    return df.reset_index(drop=True)
