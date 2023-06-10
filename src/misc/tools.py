import torch

def normalize_rows(mat: torch.Tensor) -> torch.Tensor:
    # each row of 2D array sums to 1.0
    mat = torch.nan_to_num(
        mat / mat.sum(dim=1, keepdim=True)
    )

    # Debug
    # if not torch.allclose(mat.sum(1), torch.ones(len(mat))):
        # breakpoint()

    return mat

def random_stochastic_matrix(shape: tuple[int], beta: float = 1e-2): 
    # lower beta -> more uniform initialization
    energies = beta * torch.randn(*shape)
    return torch.softmax(energies, dim=-1)
