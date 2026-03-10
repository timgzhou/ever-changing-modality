"""
This file is copied from https://github.com/zbirobin/MIAM/blob/main/maskSDM/training/trainer.py
We thank the authors for the nice implementation of MIAM
"""
from typing import Literal

import numpy as np
import torch
from scipy.stats import beta


def generate_all_corners(M):
    """Generates all 2^M binary corners of a unit hypercube."""
    return np.array(
        [list(np.binary_repr(i, width=M)) for i in range(2**M)], dtype=int
    )


def sample_beta_per_corner(corner, kappa, modality_weights):
    """
    Samples a single point from a product of Beta distributions
    defined by the given corner and sharpness.

    Args:
        corner (np.ndarray): A binary vector of shape (M,)
        kappa (np.ndarray): Sharpness parameter of shape (M,)

    Returns:
        x (np.ndarray): A sampled point in [0,1]^M
    """
    M = len(corner)
    x = np.empty(M)
    for m in range(M):
        if corner[m] == 0:
            x[m] = beta.rvs(1, kappa[m] / modality_weights[m])
        else:
            x[m] = beta.rvs(kappa[m] * modality_weights[m], 1)
    return x


def sample_from_beta_mixture(M, kappa, n_samples=1, weights=None, modality_weights=None):
    """
    Samples from a weighted mixture of product Beta distributions
    anchored at each corner of the unit hypercube.

    Args:
        M (int): Dimensionality of the space
        kappa (np.ndarray): Sharpness parameter of shape (M,)
        n_samples (int): Number of samples
        weights (np.ndarray): Optional weights for each corner (shape (2^M,)). If None, uniform weights are used.
        modality_weights (np.ndarray): Optional modality weights of shape (M,). If None, uniform weights are used.

    Returns:
        samples (np.ndarray): Array of shape (n_samples, M)
    """
    corners = generate_all_corners(M)
    num_corners = len(corners)
    if weights is None:
        weights = np.ones(num_corners) / num_corners
    if modality_weights is None:
        modality_weights = np.ones(M)
    assert weights.shape[0] == num_corners
    weights = weights / weights.sum() # Normalize
    samples = []
    for _ in range(n_samples):
        idx = np.random.choice(num_corners, p=weights)
        corner = corners[idx]
        sample = sample_beta_per_corner(corner, kappa, modality_weights)
        samples.append(sample)
    return np.array(samples)


def get_corner_weights(M: int, priority_corners_mass: float) -> np.ndarray:
    """
    Compute corner weights with specified mass for special corners.

    For corners (0,...,0) and (1,...,1): weight = priority_corners_mass
    For all other corners: weight = (1 - 2*priority_corners_mass) / (2^M - 2)
    """
    other_corners_mass = 1 - 2 * priority_corners_mass
    num_other_corners = 2**M - 2

    return np.array(
        [
            priority_corners_mass
            if tuple(corner) in [tuple(0 for _ in range(M)), tuple(1 for _ in range(M))]
            else other_corners_mass / num_other_corners
            for corner in generate_all_corners(M)
        ]
    )


def compute_rho_s_opm(S):
    """
    S: 1D tensor of shape (M,)
    Returns:
       rho: 1D tensor of shape (M,)
    """
    M = S.shape[0]

    # Pairwise ratios S_m / S_j
    ratios = S[:, None] / S[None, :]  # shape (M, M)

    # Remove self-comparisons (diagonal)
    mask = ~torch.eye(M, dtype=bool, device=S.device)
    ratios = ratios[mask].view(M, M - 1)

    # Average over j != m
    rho = ratios.mean(dim=1)

    return rho


def get_mask_prob(
    method: Literal[
        "miam",
        "opm",
        "dirichlet",
        "uniform_line",
        "uniform_hypercube",
        "dropout",
        "constant",
        "corner_priority",
    ],
    M: int,
    kappa: float = None,
    lambda_s: float = None,
    lambda_d: float = None,
    corner_weights: torch.Tensor = None,
    rho_s: torch.Tensor = None,
    rho_d: torch.Tensor = None,
    q_base: float = None,
    dirichlet_alpha: float = None,
) -> torch.Tensor:
    """
    Compute per-modality masking probabilities.

    Args:
        method: Masking strategy. One of:
            {"miam", "opm", "dirichlet", "uniform_line", "uniform_hypercube",
            "dropout", "constant", "corner_priority"}.
        M: Number of modalities.
        kappa: Beta sharpness for {"miam", "corner_priority"}.
        lambda_s: Exponent applied to rho_s in "miam". Equals to lambda in the paper.
        lambda_d: Exponent applied to rho_d in "miam". Equals to lambda in the paper.
        corner_weights: Corner weights for {"miam", "corner_priority"}, shape (2**M,).
        rho_s: Modality performance scores for {"opm", "miam"}, shape (M,).
        rho_d: Modality derivative scores for "miam", shape (M,).
        q_base: Base masking prob for {"opm", "dropout", "constant"}.
        dirichlet_alpha: Concentration for "dirichlet".

    Returns:
        Tensor of shape (M,) with masking probabilities.
    
    """
    # ----- MIAM -----
    if method == "miam":
        if kappa is None or lambda_s is None or lambda_d is None:
            raise ValueError("kappa, lambda_s and lambda_d must be provided for MIAM")
        if rho_s is None:
            rho_s = torch.ones((M,))
        if rho_d is None:
            rho_d = torch.ones((M,))

        # Imbalance ratio
        modality_weights = (rho_s**lambda_s) / (rho_d**lambda_d)

        mask_prob = sample_from_beta_mixture(
            M,
            n_samples=1,
            kappa=kappa * torch.ones((M,)),
            weights=corner_weights,
            modality_weights=modality_weights,
        )
        mask_prob = torch.tensor(mask_prob).squeeze()
    # ----- Other methods -----
    elif method == "corner_priority": # Beta hypercube
        modality_weights = torch.ones((M,))
        mask_prob = sample_from_beta_mixture(
            M,
            n_samples=1,
            kappa=kappa * torch.ones((M,)),
            weights=corner_weights,
            modality_weights=modality_weights,
        )
        mask_prob = torch.tensor(mask_prob).squeeze()
    elif method == "opm":
        if rho_s is None:
            mask_prob = torch.tensor([q_base]).repeat(M)
        else:
            mask_prob = torch.bernoulli(
                q_base * (1 + lambda_s * torch.tanh(rho_s - 1)) * (rho_s > 1)
            )
    elif method == "dirichlet":
        mask_prob_distr = torch.distributions.dirichlet.Dirichlet(
            dirichlet_alpha * torch.ones((M,))
        )
        mask_prob = mask_prob_distr.sample()
    elif method == "uniform_line":
        mask_prob_distr = torch.distributions.beta.Beta(1, 1) # = Uniform(0,1)
        mask_prob = mask_prob_distr.sample((1,)).repeat(M)
    elif method == "uniform_hypercube":
        mask_prob_distr = torch.distributions.beta.Beta(1, 1) # = Uniform(0,1)
        mask_prob = mask_prob_distr.sample((M,))
    elif method == "dropout":
        mask_prob = torch.bernoulli(q_base * torch.ones((M,)))
    elif method == "constant":
        mask_prob = torch.tensor([q_base]).repeat(M)
    else:
        raise NotImplementedError

    return mask_prob