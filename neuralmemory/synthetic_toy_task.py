import torch
import torch.nn as nn


class Knowledge(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_knowledge_entries: int,
        key_dim: int,
        value_dim: int,
        output_dim: int,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_knowledge_entries = num_knowledge_entries
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim
        self.keys = nn.Parameter(
            torch.zeros((num_heads, num_knowledge_entries, key_dim))
        )
        self.values = nn.Parameter(
            torch.zeros((num_heads, num_knowledge_entries, value_dim))
        )
        self.value_unprojection = nn.Linear(num_heads * value_dim, output_dim)

    def compute_scores_dense(self, queries: torch.Tensor):
        scores = torch.einsum("hkd,bhc->bhk", self.keys, queries)
        scores = torch.sigmoid(scores)
        return scores

    def forward(self, queries: torch.Tensor):
        b, h, k = queries.shape
        scores = self.compute_scores_dense(queries)
        values = torch.einsum("bhk,hkv->bhv", scores, self.values) / self.key_dim**0.5
        return self.value_unprojection(values.view(b, -1))


class Core(nn.Module):
    pass


class NeuralMemoryMachine(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        value_dim: int,
        num_heads: int,
        num_knowledge_entries: int,
        num_layers: int,
        num_vocab: int,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.key_dim = embed_dim // num_heads
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.input_embedder = nn.Embedding(num_vocab, embed_dim)
        self.knowledge_modules = nn.ModuleList(
            [
                Knowledge(
                    num_heads, num_knowledge_entries, self.key_dim, value_dim, embed_dim
                )
                for _ in range(num_layers)
            ]
        )
        # queries are for both knowledge and token retrieval.
        # keys and values are for token retrieval.
        self.qkv = nn.ModuleList(
            [
                nn.Linear(
                    embed_dim, self.num_heads * (self.key_dim * 2 + self.value_dim)
                )
                for i in range(num_layers)
            ]
        )
        self.value_unprojection = nn.Linear(num_heads * value_dim, embed_dim)

    def forward(self, input_tokens: torch.Tensor):
        b, s = input_tokens.shape

        x = self.input_embedder(input_tokens)
        for i in range(len(self.qkv)):
            qkv = self.qkv[i](x)
            q, k, v = (
                qkv[..., : self.key_dim * self.num_heads],
                qkv[
                    ...,
                    self.key_dim * self.num_heads : 2 * self.key_dim * self.num_heads,
                ],
                qkv[..., 2 * self.key_dim * self.num_heads :],
            )
            q = q.view(b, s, self.num_heads, self.key_dim)
            k = k.view(b, s, self.num_heads, self.key_dim)
            v = v.view(b, s, self.num_heads, self.value_dim)

            qk = torch.einsum("bshd,bthd->bhst", q, k) / (self.key_dim**0.5)
            v_context = torch.einsum(
                "bhst,bthv->bshv", torch.softmax(qk, dim=-1), v
            ).view(b, s, -1)

            x = x + self.knowledge_modules[i](q) + self.value_unprojection(v_context)

        return torch.einsum("bsd,wd->bsw", x, self.input_embedder.weight)


token_map = {
    "bob": 0,
    "is": 1,
    "tall": 2,
    "short": 3,
}

nmm = NeuralMemoryMachine(
    embed_dim=32,
    value_dim=16,
    num_heads=4,
    num_knowledge_entries=8,
    num_layers=2,
    num_vocab=len(token_map),
)
