import numpy as np  
import pandas as pd

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Random
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def set_seed(seed: int) -> None:
    """Sets random seeds."""
    np.random.seed(seed)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_ib_curve(fn: str) -> list[tuple]:
    """Load a (comm_cost, complexity) IB curve computed by reverse deterministic annealing of the B.A. algorithm."""
    df = pd.read_csv(fn)
    return list(map(tuple, df.to_numpy()))


def save_ib_curve(fn: str, curve) -> None:
    """Save a dataframe of (comm_cost, complexity) points to a CSV."""
    df = pd.DataFrame(data=curve, columns=["comm_cost", "complexity"])
    df.to_csv(fn, index=False)
    print(f"Saved {len(df)} language points to {fn}")    