import time

import cupy as cp  # noqa: F401
import cuvs.neighbors.cagra as cagra
import torch
import torch.nn as nn
import torch.nn.functional as F

index_params = cagra.IndexParams(metric="inner_product")
search_params = cagra.SearchParams()


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

        self.index = None

    def rebuild_index(self):
        # cp.asarray is zero-copy. what happens during gradient updates if we don't rebuild the index? hopefully cuvs doesn't really care because I don't
        self.index = cagra.build(index_params, cp.asarray(self.in_weight.weight.data))

    def forward(self, x_B_D: torch.Tensor) -> torch.Tensor:
        assert self.index is not None, "Index must be built before forward pass"

        t0 = time.time()  # noqa: F841
        # cp.asarray is zero-copy
        # https://docs.cupy.dev/en/stable/user_guide/interoperability.html#pytorch
        _, indices_B_R = cagra.search(
            search_params, self.index, cp.asarray(x_B_D), k=self.sparsity_dim
        )
        t1 = time.time()  # noqa: F841

        # Recompute distances
        retrieved_keys_B_R_D = self.in_weight.weight[indices_B_R]
        retrieved_values_B_R_D = self.out_weight[indices_B_R]

        # Compute retrieval coefficients
        retrieval_coefficients_B_R = x_B_D @ retrieved_keys_B_R_D.transpose(1, 2)
        retrievals_B_D = torch.einsum(
            "br,brd->bd", F.gelu(retrieval_coefficients_B_R), retrieved_values_B_R_D
        )

        return retrievals_B_D


if __name__ == "__main__":
    HIDDEN_DIM = 21504
    RESIDUAL_STREAM_DIM = 5376
    SPARSITY_DIM = 16

    sparse_mlp = SparseMLP(
        input_dim=RESIDUAL_STREAM_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=RESIDUAL_STREAM_DIM,
        sparsity_dim=SPARSITY_DIM,
    )
    input = torch.randn(128, RESIDUAL_STREAM_DIM)

    sparse_mlp.rebuild_index()
    output = sparse_mlp(input)
    print(output.shape)
