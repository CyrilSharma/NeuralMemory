import time

import cupy as cp  # noqa: F401
import cuvs.neighbors.cagra as cagra
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import skip_init

index_params = cagra.IndexParams(metric="inner_product")
search_params = cagra.SearchParams(algo="single_cta", itopk_size=512)


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

        indices_B_R = torch.as_tensor(
            indices_B_R, device=x_B_D.device, dtype=torch.long
        )

        # Recompute distances
        retrieved_keys_B_R_D = self.in_weight.weight[indices_B_R]
        retrieved_values_B_R_D = self.out_weight[indices_B_R]

        # Compute retrieval coefficients
        retrieval_coefficients_B_R = torch.bmm(
            x_B_D.unsqueeze(1), retrieved_keys_B_R_D.transpose(1, 2)
        ).squeeze(1)

        retrievals_B_D = torch.einsum(
            "br,brd->bd", F.gelu(retrieval_coefficients_B_R), retrieved_values_B_R_D
        )

        return retrievals_B_D


class SparseGatedMLP(nn.Module):
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
        self.in_weight = skip_init(nn.Linear, input_dim, hidden_dim, bias=False)
        # config: https://huggingface.co/google/gemma-4-31B/blob/main/config.json
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py#L1039
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.skip_init.html
        self.gate_weight = skip_init(nn.Linear, input_dim, hidden_dim, bias=False)
        self.out_weight = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.1)

        # Initialize weights more carefully
        # nn.init.xavier_uniform_(self.in_weight.weight)
        # nn.init.zeros_(self.in_weight.bias)
        self.alpha = alpha
        self.beta = beta

        self.index = None

    def rebuild_index(self):
        # cp.asarray is zero-copy. what happens during gradient updates if we don't rebuild the index? hopefully cuvs doesn't really care because I don't
        self.index = cagra.build(
            index_params, cp.asarray(self.in_weight.weight.data.float())
        )

    def forward(self, x_b_D: torch.Tensor) -> torch.Tensor:
        assert self.index is not None, "Index must be built before forward pass"

        x_B_D = x_b_D.view(-1, x_b_D.shape[-1])

        t0 = time.time()  # noqa: F841
        # cp.asarray is zero-copy
        # https://docs.cupy.dev/en/stable/user_guide/interoperability.html#pytorch
        _, indices_B_R = cagra.search(
            search_params, self.index, cp.asarray(x_B_D.float()), k=self.sparsity_dim
        )
        t1 = time.time()  # noqa: F841

        indices_B_R = torch.as_tensor(
            indices_B_R, device=x_B_D.device, dtype=torch.long
        )
        # indices_B_R = (
        #     torch.arange(self.hidden_dim, device=x_B_D.device)
        #     .unsqueeze(0)
        #     .expand(x_B_D.shape[0], -1)
        # )

        # Recompute distances
        retrieved_keys_B_R_D = self.in_weight.weight[indices_B_R]
        retrieved_values_B_R_D = self.out_weight[indices_B_R]

        # Compute retrieval coefficients
        retrieval_coefficients_B_R = torch.bmm(
            x_B_D.unsqueeze(1),
            retrieved_keys_B_R_D.transpose(1, 2).to(dtype=x_b_D.dtype),
        ).squeeze(1)

        # Multiply by gate values
        retrieved_gate_keys_B_R_D = self.gate_weight.weight[indices_B_R]
        gate_values = torch.bmm(
            x_B_D.unsqueeze(1),
            retrieved_gate_keys_B_R_D.transpose(1, 2).to(dtype=x_b_D.dtype),
        ).squeeze(1)
        retrieval_coefficients_B_R = retrieval_coefficients_B_R * F.gelu(
            gate_values, approximate="tanh"
        )
        retrievals_B_D = torch.einsum(
            "br,brd->bd", retrieval_coefficients_B_R, retrieved_values_B_R_D
        )
        retrievals_b_D = retrievals_B_D.view(*x_b_D.shape)

        return retrievals_b_D


if __name__ == "__main__":
    HIDDEN_DIM = 21504
    RESIDUAL_STREAM_DIM = 5376
    SPARSITY_DIM = 16

    sparse_mlp = SparseGatedMLP(
        input_dim=RESIDUAL_STREAM_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=RESIDUAL_STREAM_DIM,
        sparsity_dim=SPARSITY_DIM,
    )

    # Time how long it takes to build the index.
    t0 = time.time()
    sparse_mlp.rebuild_index()
    t1 = time.time()
    print("Building index:", t1 - t0)

    # Time how long it takes to do a forward pass.
    # First pass to warm up.
    input = torch.randn(128, RESIDUAL_STREAM_DIM)
    output = sparse_mlp(input)

    t0 = time.time()
    input = torch.randn(128, RESIDUAL_STREAM_DIM)
    output = sparse_mlp(input)
    t1 = time.time()
    print("Sparse forward pass:", t1 - t0)

    # Time how long the non-sparse version takes.
    # First pass to warm up.
    input = torch.randn(128, RESIDUAL_STREAM_DIM)
    act = (
        F.gelu(sparse_mlp.gate_weight(input)) * (sparse_mlp.in_weight(input))
    ) @ sparse_mlp.out_weight

    t0 = time.time()
    input = torch.randn(128, RESIDUAL_STREAM_DIM)
    act = (
        F.gelu(sparse_mlp.gate_weight(input)) * (sparse_mlp.in_weight(input))
    ) @ sparse_mlp.out_weight
    t1 = time.time()
    print("Dense forward pass:", t1 - t0)
