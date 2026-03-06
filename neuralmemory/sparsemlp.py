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
        self.in_weight = nn.Linear(input_dim, hidden_dim, bias=False)
        self.out_weight = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.1)

        # Initialize weights more carefully
        nn.init.xavier_uniform_(self.in_weight.weight)
        # nn.init.zeros_(self.in_weight.bias)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape0 = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        b = x.shape[0]
        z: torch.Tensor = self.in_weight(x)

        # Clamp z to prevent overflow in sigmoid
        z_clamped = torch.clamp(z, min=-10, max=10)

        # Use torch.sigmoid for numerical stability
        p_B_H = torch.sigmoid(self.alpha * (z_clamped - self.beta))

        if self.sparsity_dim == self.hidden_dim:
            return (p_B_H @ self.out_weight).view(*shape0, self.output_dim)

        p_sum_B_1 = p_B_H.sum(dim=1, keepdim=True)
        idxs_B_S = torch.multinomial(p_B_H, self.sparsity_dim, replacement=True)
        idxs_B_S_O = idxs_B_S.unsqueeze(-1).expand(-1, -1, self.output_dim)
        corrections_B_1 = p_sum_B_1 / self.sparsity_dim
        p_B_S = p_B_H.gather(dim=1, index=idxs_B_S)
        # straight-through estimator
        corrections_B_S = p_B_S / p_B_S.detach() * corrections_B_1
        out_weight_B_H_O: torch.Tensor = self.out_weight.view(
            1, self.hidden_dim, self.output_dim
        ).expand(b, self.hidden_dim, self.output_dim).gather(
            dim=1, index=idxs_B_S_O
        ) * corrections_B_S.view(b, self.sparsity_dim, 1)
        out_B_O = out_weight_B_H_O.sum(dim=1)
        out = out_B_O.view(*shape0, self.output_dim)
        return out
