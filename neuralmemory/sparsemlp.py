import time

import torch
import torch.nn as nn


class SparseMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        sparsity_dim: int,
        alpha=5,
        beta=0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sparsity_dim = sparsity_dim
        # do not remove bias=false
        self.in_weight = nn.Linear(input_dim, hidden_dim, bias=False)
        self.out_weight = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.1)

        # Initialize weights more carefully
        nn.init.xavier_uniform_(self.in_weight.weight)
        # nn.init.zeros_(self.in_weight.bias)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape0 = x.shape[:-1]
        t0 = time.time()  # noqa: F841
        x = x.view(-1, x.shape[-1])
        z: torch.Tensor = self.in_weight(x)
        z_clamped = torch.clamp(z, min=-10, max=10)
        p_B_H = torch.sigmoid(self.alpha * (z_clamped - self.beta))
        t1 = time.time()  # noqa: F841

        if self.sparsity_dim == self.hidden_dim:
            return (p_B_H @ self.out_weight).view(*shape0, self.output_dim)

        # Fast sparse computation with efficient memory access
        # top_probs, top_indices = torch.topk(p_B_H, self.sparsity_dim, dim=1)
        top_probs = p_B_H[:, : self.sparsity_dim]  # (b, sparsity_dim)
        top_indices = (
            torch.arange(self.sparsity_dim, device=x.device)
            .unsqueeze(0)
            .expand(x.shape[0], -1)
        )  # (b, sparsity_dim)

        # Compute normalization more efficiently
        prob_sum = p_B_H.sum(dim=1, keepdim=True)
        normalization = prob_sum / self.sparsity_dim

        # Apply straight-through estimator
        corrections = top_probs / (top_probs.detach() + 1e-8) * normalization

        # Use einsum for efficient computation - avoids expensive advanced indexing
        selected_weights = self.out_weight[top_indices]  # (b, sparsity_dim, output_dim)

        # Compute weighted sum using einsum (much faster than manual operations)
        output = torch.einsum("bs,bso->bo", corrections, selected_weights)

        t2 = time.time()  # noqa: F841

        return output.view(*shape0, self.output_dim)
