from scipy.special import softmax
import numpy as np


def normalize_rows(mat: np.ndarray):
    """Normalize each row of 2D array / tensor to sum to 1.0."""
    return np.nan_to_num(mat / mat.sum(axis=1, keepdims=True))


def random_stochastic_matrix(shape: tuple[int], beta: float = 1e-2):
    """Generate a random stochastic matrix using energy-based initialization, where lower `beta` -> more uniform initialization."""
    if beta is not None:
        energies = beta * np.random.randn(*shape)
        return softmax(energies, axis=-1)
    return random_uniform_stochastic_matrices(shape)


def random_uniform_stochastic_matrices(shape: tuple[int]) -> np.ndarray:
    # Base case: if shape is 2D, create the stochastic matrix
    if len(shape) == 2:
        return random_uniform_stochastic_matrix(shape)
    # Recursive case: if shape has more than 2 dimensions, iterate over the first dimension
    return np.array([random_stochastic_matrix(shape[1:]) for _ in range(shape[0])])

def random_uniform_stochastic_matrix(shape: tuple[int]):
    """Generate a stochastic matrix by randomly sampling uniformly from [0,1] and renormalizing."""
    return normalize_rows(np.random.uniform(low=0, high=1, size=shape))