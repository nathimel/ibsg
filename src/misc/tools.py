import torch

def normalize_rows(mat: torch.Tensor) -> torch.Tensor:
    # each row of 2D array sums to 1.0
    return torch.nan_to_num(
        mat / mat.sum(dim=1, keepdim=True)
    )

def random_stochastic_matrix(shape: tuple[int], temperature: float = 1e-2): 
    # higher temperature -> more uniform initialization
    energies = 1/temperature * torch.randn(*shape)
    return torch.softmax(energies, dim=-1)
