import numpy as np

def check_drift(reference, current, threshold=0.1):
    diff = np.mean(np.abs(reference - current))
    return diff > threshold
