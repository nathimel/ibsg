
from scipy.special import softmax
import numpy as np

def normalize_rows(mat: np.ndarray):
    """Normalize each row of 2D array / tensor to sum to 1.0."""
    return np.nan_to_num(mat / mat.sum(axis=1, keepdims=True))


def random_stochastic_matrix(shape: tuple[int], beta: float = 1e-2):
    """Generate a random stochastic matrix using energy-based initialization, where lower `beta` -> more uniform initialization."""
    energies = beta * np.random.randn(*shape)
    return softmax(energies, axis=-1)
