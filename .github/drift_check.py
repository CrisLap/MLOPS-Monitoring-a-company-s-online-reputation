import sys
import pandas as pd
from datasets import load_dataset
from training.data_loader import load_data

def get_stats(df):
    """Calculates mean text length and class distribution."""
    lengths = df["text"].astype(str).str.len()
    class_dist = df["label"].value_counts(normalize=True).to_dict()
    return {
        "mean_len": lengths.mean(),
        "class_dist": class_dist
    }

def run_drift_check():
    # FIXED REFERENCE VALUES
    REFERENCE_STATS = {
        "mean_len": 102.5,
        "class_dist": {0: 0.16, 1: 0.45, 2: 0.39}
    }

    # THRESHOLDS
    LEN_THRESHOLD = 0.05
    CLASS_THRESHOLD = 0.02

    print("Loading 'tweet_eval' training split for drift check...")
    try:
        dataset = load_dataset("tweet_eval", "sentiment", split="train")
        train_df = pd.DataFrame(dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    cur_stats = get_stats(train_df)
    drift_flag = False

    # Check Length
    diff_len = abs(REFERENCE_STATS["mean_len"] - cur_stats["mean_len"]) / REFERENCE_STATS["mean_len"]
    if diff_len > LEN_THRESHOLD:
        print(f"ALERT: Mean length drift detected ({diff_len:.2%})")
        drift_flag = True

    # Check Class Distribution
    for label, ref_perc in REFERENCE_STATS["class_dist"].items():
        cur_perc = cur_stats["class_dist"].get(label, 0)
        if abs(ref_perc - cur_perc) > CLASS_THRESHOLD:
            print(f"ALERT: Class {label} distribution drift detected (Diff: {abs(ref_perc - cur_perc):.2%})")
            drift_flag = True

    if drift_flag:
        print("STATUS: Data drift detected. Aborting pipeline.")
        sys.exit(1)
    
    print("STATUS: Data integrity verified. No significant drift found.")
    sys.exit(0)

if __name__ == "__main__":
    run_drift_check()