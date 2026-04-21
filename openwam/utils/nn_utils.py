"""
nn_utils.py

Utility functions and PyTorch submodule definitions.
"""

import torch
import torch.nn as nn


# === Definitions for Various Projection Modules, with Signature :: [..., in_dim] --> [..., out_dim] ===
class LinearProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        """
        Create a LinearProjector module that maps vision embeddings to LLM embedding space.

        Parameters:
            vision_dim (int): Dimension of the input visual feature vectors.
            llm_dim (int): Target dimension for the language-model embedding space; sets the output features of the internal linear layer.
        """
        super().__init__()
        self.projector = nn.Linear(vision_dim, llm_dim, bias=True)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        """
        Project image patch feature tensors into the target LLM embedding space using the configured projector.

        Parameters:
            img_patches (torch.Tensor): Image patch feature tensor to project. The last dimension must match the projector's expected input feature size (vision dimension); leading dimensions (batch/sequence) are preserved.

        Returns:
            torch.Tensor: Projected feature tensor with the same leading dimensions as `img_patches` and last dimension equal to the projector's output dimension (LLM embedding size).
        """
        return self.projector(img_patches)


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


class FusedMLPProjector(nn.Module):
    def __init__(self, fused_vision_dim: int, llm_dim: int, mlp_type: str = "fused-gelu-mlp") -> None:
        """
        Initialize a fused MLP projector that maps fused vision features into the LLM embedding space.

        Parameters:
            fused_vision_dim (int): Dimensionality of the fused vision input features.
            llm_dim (int): Target embedding dimensionality for the language model.
            mlp_type (str): Type of MLP to construct; only `"fused-gelu-mlp"` is supported.

        Notes:
            - Sets `self.initial_projection_dim` to `fused_vision_dim * 4`.
            - Constructs a 3-layer sequential projector with GELU activations when `mlp_type` is `"fused-gelu-mlp"`.
            - Raises `ValueError` if `mlp_type` is unsupported.
        """
        super().__init__()
        self.initial_projection_dim = fused_vision_dim * 4
        if mlp_type == "fused-gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(fused_vision_dim, self.initial_projection_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Fused Projector with `{mlp_type = }` is not supported!")

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        """
        Project fused image patch embeddings into the target LLM embedding space.

        Parameters:
            fused_img_patches (torch.Tensor): Tensor of fused vision features with final dimension equal to the projector's expected fused_vision_dim (shape: [..., fused_vision_dim]).

        Returns:
            torch.Tensor: Projected embeddings with the same leading dimensions and final dimension equal to the LLM embedding size.
        """
        return self.projector(fused_img_patches)
