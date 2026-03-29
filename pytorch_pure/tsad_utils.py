import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple

def get_tsad_instructions(data_type: str, n_samples: int = 512) -> Tuple[List[str], List[str]]:
    """
    Load TSAD instructions from tsad/train.csv.
    data_type: 'harmful' (negative) or 'harmless' (positive)
    Returns: train, test lists of instructions
    """
    df = pd.read_csv("tsad/train.csv", encoding="latin1")
    if data_type == "harmful":
        subset = df[df["sentiment"] == "negative"]["text"].astype(str).tolist()
    elif data_type == "harmless":
        subset = df[df["sentiment"] == "positive"]["text"].astype(str).tolist()
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    train, test = train_test_split(subset, test_size=0.2, random_state=42)
    return train[:n_samples], test[:min(128, len(test))]
