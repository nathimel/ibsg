import numpy as np


# We hard code the list of betas because through trial and error this set of values best sweeps out the meaning_certainty=0 curve.
meaning_dist_0_betas = np.concatenate(
    [
        # these betas were hand selected to try and evenly flesh out the curve for meaning_dist=-1 as much as possible
        np.linspace(start=0, stop=0.3, num=333),
        
        # very sparse region
        np.linspace(start=0.7, stop=0.77, num=500),
        np.linspace(start=0.69, stop=0.72, num=500),

        np.linspace(start=0.3, stop=0.9, num=333),
        np.linspace(start=0.9, stop=4, num=334),
        np.linspace(start=4, stop=2**7, num=500),
    ]
)



# dev
meaning_dist_0_betas_dev = np.concatenate([
        np.logspace(-1, 0.25, 1000),
        np.logspace(0.25, 1, 100,)
])


meaning_dist_gaussian = np.concatenate([
    np.linspace(1., 1.005, 100),
    np.linspace(1.005, 1.03, 100),
    np.linspace(1.03, 1.1, 200),
    np.linspace(1.1, 2.5, 200),
    np.logspace(np.log10(2.5), np.log10(10), 100), # this gets to 6.5 bits
    np.logspace(1, 7, 100,) # this last should get to np.log2(100) bits
])



betas = meaning_dist_gaussian
# unique
betas = list(sorted(set(betas.tolist())))