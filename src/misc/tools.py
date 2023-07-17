import torch
import numpy as np
from typing import Union


def normalize_rows(mat: torch.Tensor):
    """Normalize each row of 2D array / tensor to sum to 1.0."""
    # numpy and torch have *almost* the same API :/
    if isinstance(mat, np.ndarray):
        mat = np.nan_to_num(mat / mat.sum(1, keepdims=True))
    if isinstance(mat, torch.Tensor):
        mat = torch.nan_to_num(mat / mat.sum(1, keepdim=True))
    return mat


def random_stochastic_matrix(shape: tuple[int], beta: float = 1e-2):
    # lower beta -> more uniform initialization
    energies = beta * torch.randn(*shape)
    return torch.softmax(energies, dim=-1)
