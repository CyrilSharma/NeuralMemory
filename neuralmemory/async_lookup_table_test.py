"""
What we need:
- An asynchronous "lookup" function that takes in the current residual stream for a token and then calls a potentially expensive lookup process on a sparse memory table
- Ways to propagate information back to this lookup table

We use a maximum inner product search, where we simply take the top K entries in the memory table that have the highest inner product with the current residual stream. We
then specify `requires_grad` and `retain_grad` for that tensor. During backpropagation, the `grad_fn` can have properties beginning with `_saved` to represent the saved
tensors from the forward pass.

"""

import torch
from torch.autograd import Function


class AsynchronousLookupTable(Function):
    @staticmethod
    def forward(ctx, queries_B_S_D, keys_E_D, values_E_D):
        # Assume for now that all indices are selected with coefficients according to dot product.
        # Notation: E = # kv entries, R = retrievals, B = batch size, S = sequence length, D = dimension of residual stream and kv entries
        num_kv_entries = keys_E_D.shape[0]
        top_k_indices_B_S_R = (
            torch.arange(num_kv_entries)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(queries_B_S_D.shape[0], queries_B_S_D.shape[1], num_kv_entries)
        )  # Placeholder for top K indices

        top_k_coefficients_B_S_R = queries_B_S_D @ keys_E_D.t()

        ctx.save_for_backward(
            top_k_indices_B_S_R,
            top_k_coefficients_B_S_R,
            queries_B_S_D,
            keys_E_D,
            values_E_D,
        )

        # (B, S, R, D) * (R, D) -> (B, S, R, D) -> (B, S, D)
        aggregated_values_B_S_D = (
            top_k_coefficients_B_S_R.unsqueeze(-1) * values_E_D[top_k_indices_B_S_R]
        ).sum(dim=-2)

        return aggregated_values_B_S_D

    @staticmethod
    def backward(ctx, grad_output_B_S_D):
        (
            top_k_indices_B_S_R,
            top_k_coefficients_B_S_R,
            queries_B_S_D,
            keys_E_D,
            values_E_D,
        ) = ctx.saved_tensors

        grad_elementwise_B_S_R_D = grad_output_B_S_D.unsqueeze(
            -2
        ) * top_k_coefficients_B_S_R.unsqueeze(-1)

        # Gradient for values is output gradient scaled by retrieval coefficient for that value
        value_grad_R_D = grad_elementwise_B_S_R_D.sum(dim=(0, 1))

        # Gradient for retrieval coefficients is output gradient dotted with value vector for that coefficient
        retrieval_coefficient_grad_B_S_R = (
            values_E_D[top_k_indices_B_S_R] * grad_elementwise_B_S_R_D
        ).sum(dim=-1)

        # Gradient for keys is query vector scaled by retrieval coefficient gradient
        key_grad_R_D = (
            queries_B_S_D.unsqueeze(-2) * retrieval_coefficient_grad_B_S_R.unsqueeze(-1)
        ).sum(dim=(0, 1))

        # Gradient for queries is key vector scaled by retrieval coefficient gradient
        # This is the gradient that will be backpropagated to the input residual stream
        query_grad_B_S_D = (
            keys_E_D[top_k_indices_B_S_R]
            * retrieval_coefficient_grad_B_S_R.unsqueeze(-1)
        ).sum(dim=-2)

        # Key and value gradients are dictionaries. Query gradients are passed back to the input residual stream and will be handled by autograd as usual.
        return query_grad_B_S_D, key_grad_R_D, value_grad_R_D


# Assume D = 10, E = 10
keys_E_D = torch.eye(10, requires_grad=True)
values_E_D = torch.eye(10, requires_grad=True)
queries_B_S_D = torch.zeros(2, 3, 10)  # B = 2, S = 3, D = 10
queries_B_S_D[0, 0, 0] = 1.0
queries_B_S_D[0, 0, 1] = 1.0

output = AsynchronousLookupTable.apply(queries_B_S_D, keys_E_D, values_E_D)
output.sum().backward()

print(queries_B_S_D)
print("===")
print(output)
print("keys_E_D.grad")
print(keys_E_D.grad)
print("values_E_D.grad")
print(values_E_D.grad)
