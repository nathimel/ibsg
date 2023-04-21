import numpy as np

def generate_prior_over_states(num_states: int, prior_type: str = 'uniform', alpha: np.ndarray = None) -> np.ndarray:
    """Generate a prior probability distribution over states.
    Varying the entropy of the prior over states models the relative communicative 'need' of states.
    
    Args:
        num_states: the size of the distribution

        type: {'uniform', 'random', 'dirac'} a str representing whether to generate a uniform prior, randomly sample one from a Dirichlet distribution, or construct a degenerate dirac delta distribution. If 'dirac', then the first state is assigned all probability mass. 

        alpha: parameter of the Dirichlet distribution to sample from, of shape `(num_states)`. Each element must be greater than or equal to 0. By default set to all ones. Varying this parameter varies the entropy of the prior over states.
    Returns: 
        sample: np.ndarray of shape `(num_states)`
    """
    if prior_type == 'uniform':
        sample = np.ones(num_states) / num_states # TODO: find how to generate uniform with alpha param in dirichlet

    elif prior_type == 'dirac': 
        sample = np.array([1.] + [0.] * (num_states - 1))

    elif prior_type == 'random':
        if alpha is None:
            alpha = np.ones(num_states)
        sample = np.random.default_rng().dirichlet(alpha=alpha)

    else:
        raise ValueError(f"The argument `prior_type` can take values {{'uniform', 'random', 'dirac'}}, but received {prior_type}.")
    return sample