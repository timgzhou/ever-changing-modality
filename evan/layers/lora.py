# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
from torch import Tensor, nn


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer for efficient fine-tuning.

    Applies a low-rank decomposition to adapt pretrained weights:
    h = W_0 x + (B @ A) x

    where W_0 are frozen pretrained weights, and A, B are learnable low-rank matrices.

    Args:
        dim: Hidden dimension size
        rank: Rank of the low-rank decomposition (default: 8)
        alpha: Scaling factor for LoRA output (default: 1.0)
        device: Device to create parameters on
    """

    def __init__(
        self,
        dim: int,
        rank: int = 8,
        alpha: float = 1.0,
        device=None,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # Low-rank matrices: A (dim x rank), B (rank x dim)
        # Zero-initialized so LoRA initially produces zero output
        self.lora_A = nn.Parameter(torch.zeros(dim, rank, device=device))
        self.lora_B = nn.Parameter(torch.zeros(rank, dim, device=device))

        # Scaling factor (typically alpha / rank)
        self.scaling = alpha / rank if rank > 0 else 1.0

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply LoRA transformation.

        Args:
            x: Input tensor of shape [B, N, dim] where B is batch size,
               N is sequence length, and dim is hidden dimension

        Returns:
            LoRA output of same shape as input
        """
        # x: [B, N, dim]
        # Apply low-rank transformation: x @ A @ B with scaling
        # (B, N, dim) @ (dim, rank) @ (rank, dim) -> (B, N, dim)
        return self.scaling * (x @ self.lora_A @ self.lora_B)

    def reset_parameters(self):
        """Reset LoRA parameters to zero (maintains pretrained behavior)."""
        nn.init.zeros_(self.lora_A)
        nn.init.zeros_(self.lora_B)