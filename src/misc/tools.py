import numpy as np

def normalize_rows(mat: np.ndarray) -> np.ndarray:
    # each row of 2D array sums to 1.0
    np.seterr(divide="ignore", invalid="ignore")
    return np.nan_to_num(
        mat / mat.sum(axis=1, keepdims=True)
    )

def rows_zero_to_uniform(mat):
    """Ensure that P(output | input) is a probability distribution, i.e. each row (indexed by inputs, either a state or a signal) is a distribution over outputs (either signals or states), sums to exactly 1.0. Necessary when exploring mathematically possible languages which sometimes have that p(signal|state) is a vector of 0s."""

    threshold = 1e-5

    for row in mat:
        # less than 1.0
        if row.sum() and 1.0 - row.sum() > threshold:
            print("row is nonzero and sums to less than 1.0!")
            print(row, row.sum())
            raise Exception
        # greater than 1.0
        if row.sum() and row.sum() - 1.0 > threshold:
            print("row sums to greater than 1.0!")
            print(row, row.sum())
            raise Exception

    return np.array([row if row.sum() else np.ones(len(row)) / len(row) for row in mat])