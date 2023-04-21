import numpy as np

def normalize_rows(mat: np.ndarray) -> np.ndarray:
    # each row of 2D array sums to 1.0
    np.seterr(divide="ignore", invalid="ignore")
    return np.nan_to_num(
        mat / mat.sum(axis=1, keepdims=True)
    )