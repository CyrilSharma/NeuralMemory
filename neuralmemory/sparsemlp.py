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
        self.in_weight = nn.Linear(input_dim, hidden_dim)
        self.out_weight = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.1)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape0 = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        b = x.shape[0]
        z: torch.Tensor = self.in_weight(x)
        p_B_H = 1 / (1 + torch.exp(-self.alpha * (z - self.beta)))
        p_sum_B = p_B_H.sum(dim=1, keepdim=True)
        idxs_B_S = torch.multinomial(p_B_H, self.sparsity_dim, replacement=True)
        idxs_B_S_O = idxs_B_S.unsqueeze(-1).expand(-1, -1, self.output_dim)
        corrections_B = p_sum_B / self.sparsity_dim
        corrections_B_1 = corrections_B.view(-1, 1)

        out_weight_B_H_O: torch.Tensor = self.out_weight.view(
            1, self.hidden_dim, self.output_dim
        ).expand(b, 1, 1)

        out_B_O = torch.gather(out_weight_B_H_O, dim=1, index=idxs_B_S_O).sum(dim=1)
        out_B_O = out_B_O * corrections_B_1
        out = out_B_O.view(*shape0, self.output_dim)
        return out
