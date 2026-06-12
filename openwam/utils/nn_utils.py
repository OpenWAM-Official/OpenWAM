"""
nn_utils.py

Utility functions and PyTorch submodule definitions.
"""

import torch
import torch.nn as nn


# === Definitions for Projection Modules, with Signature :: [..., in_dim] --> [..., out_dim] ===
class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, mlp_type: str = "gelu-mlp") -> None:
        """
        Create an MLP-based projector mapping vision embeddings to LLM embedding space.

        Parameters:
            vision_dim (int): Dimensionality of input vision vectors.
            llm_dim (int): Target LLM embedding dimensionality.
            mlp_type (str): Type of MLP to construct. Supported value: "gelu-mlp" (builds Linear(vision_dim -> llm_dim) + GELU + Linear(llm_dim -> llm_dim)).

        Raises:
            ValueError: If `mlp_type` is not supported.
        """
        super().__init__()
        if mlp_type == "gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        """
        Project image patch feature tensors into the target LLM embedding space using the configured projector.

        Parameters:
            img_patches (torch.Tensor): Image patch feature tensor to project. The last dimension must match the projector's expected input feature size (vision dimension); leading dimensions (batch/sequence) are preserved.

        Returns:
            torch.Tensor: Projected feature tensor with the same leading dimensions as `img_patches` and last dimension equal to the projector's output dimension (LLM embedding size).
        """
        return self.projector(img_patches)
