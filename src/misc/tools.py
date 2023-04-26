import torch

def normalize_rows(mat: torch.Tensor) -> torch.Tensor:
    # each row of 2D array sums to 1.0
    return torch.nan_to_num(
        mat / mat.sum(dim=1, keepdim=True)
    )

def random_stochastic_matrix(shape: tuple[int], beta: float = 1e-2): 
    # lower beta -> more uniform initialization
    energies = beta * torch.randn(*shape)
    return torch.softmax(energies, dim=-1)
