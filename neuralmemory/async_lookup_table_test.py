"""
What we need:
- An asynchronous "lookup" function that takes in the current residual stream for a token and then calls a potentially expensive lookup process on a sparse memory table
- Ways to propagate information back to this lookup table

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
            torch.arange(2)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(queries_B_S_D.shape[0], queries_B_S_D.shape[1], -1)
        )  # Placeholder for top K indices

        top_k_coefficients_B_S_R = (queries_B_S_D @ keys_E_D.t())[:, :, :2]

        # print("retrieval coefficients [in func]")
        # print(top_k_coefficients_B_S_R)

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
            values_E_D[top_k_indices_B_S_R] * grad_output_B_S_D.unsqueeze(-2)
        ).sum(dim=-1)

        # print(retrieval_coefficient_grad_B_S_R)

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

        # Key and value gradients are dictionaries. Query gradients are passed back to the input residual stream and will be handled by autograd as usual. It would be p nice to be able to do process-in-memory (PIM) updates here.
        key_grad_E_D = torch.zeros_like(keys_E_D)
        value_grad_E_D = torch.zeros_like(values_E_D)
        key_grad_E_D[top_k_indices_B_S_R] = key_grad_R_D
        value_grad_E_D[top_k_indices_B_S_R] = value_grad_R_D

        # return query_grad_B_S_D, key_grad_R_D, value_grad_R_D
        return query_grad_B_S_D, key_grad_E_D, value_grad_E_D


def test():
    # Assume D = 10, E = 10
    keys_E_D = torch.eye(10, requires_grad=True)
    values_E_D = torch.eye(10, requires_grad=True)
    queries_B_S_D = torch.zeros(2, 3, 10)  # B = 2, S = 3, D = 10
    queries_B_S_D[0, 0, 0] = 1.0
    queries_B_S_D[0, 0, 1] = 1.0

    output = AsynchronousLookupTable.apply(queries_B_S_D, keys_E_D, values_E_D)
    output.sum().backward()

    # print("queries_B_S_D.grad")
    # print(queries_B_S_D.grad)
    # print("keys_E_D.grad")
    # print(keys_E_D.grad)
    # print("values_E_D.grad")
    # print(values_E_D.grad)

    keys_E_D = torch.randn((10, 10), requires_grad=True)
    values_E_D = torch.randn((10, 10), requires_grad=True)
    queries_B_S_D = torch.randn((2, 3, 10), requires_grad=True)
    output = AsynchronousLookupTable.apply(queries_B_S_D, keys_E_D, values_E_D)
    output.sum().backward()

    with_fn_query_grad = queries_B_S_D.grad.clone()
    with_fn_key_grad = keys_E_D.grad.clone()
    with_fn_value_grad = values_E_D.grad.clone()

    # print("q")
    # print(with_fn_query_grad)
    # print("k")
    # print(with_fn_key_grad)
    # print("v")
    # print(with_fn_value_grad)

    # print("---")

    # Compute the true grad. Clear old grad
    queries_B_S_D.grad = None
    keys_E_D.grad = None
    values_E_D.grad = None

    # (B, S, E, 1) @ (E, D) -> (B, S, E, D) -> (B, S, D)
    retrieval_coefficients = queries_B_S_D @ keys_E_D.t()
    retrieval_coefficients.retain_grad()
    # print("retrieval coefficients")
    # print(retrieval_coefficients)

    # only take first 2 indices as actual retrievals
    retrieval_coefficients[:, :, 2:] = 0.0

    true_output_B_S_D = (retrieval_coefficients.unsqueeze(-1) * values_E_D).sum(dim=-2)
    true_output_B_S_D.sum().backward()

    gt_query_grad = queries_B_S_D.grad.clone()
    gt_key_grad = keys_E_D.grad.clone()
    gt_value_grad = values_E_D.grad.clone()

    # print("retrieval_coefficients_grad")
    # print(retrieval_coefficients.grad)

    # print("q")
    # print(gt_query_grad)
    # print("k")
    # print(gt_key_grad)
    # print("v")
    # print(gt_value_grad)

    assert torch.allclose(with_fn_query_grad, gt_query_grad)
    assert torch.allclose(with_fn_key_grad, gt_key_grad)
    assert torch.allclose(with_fn_value_grad, gt_value_grad)
