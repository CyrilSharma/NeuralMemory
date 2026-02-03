import random

import torch
import torch.nn as nn


class Bank(nn.Module):
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
        return self.value_unprojection(values.reshape(b, -1))


class Core(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        value_dim: int,
        num_heads: int,
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

    def forward(
        self,
        input_tokens: torch.Tensor,
        banks: nn.ModuleList | list[nn.Module],
    ):
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

            # s = source, t = target
            qk = torch.einsum("bshd,bthd->bhst", q, k) / (self.key_dim**0.5)
            causal_mask = torch.triu(
                torch.ones((s, s), device=qk.device, dtype=torch.bool), diagonal=1
            )
            qk = qk.masked_fill(causal_mask, float("-inf"))

            v_context = torch.einsum(
                "bhst,bthv->bshv", torch.softmax(qk, dim=-1), v
            ).reshape(b, s, -1)
            context_data = self.value_unprojection(v_context)
            bank_data = banks[i](q.view(-1, self.num_heads, self.key_dim))

            x = x + context_data + bank_data

        return torch.einsum("bsd,wd->bsw", x, self.input_embedder.weight)


class Model(nn.Module):
    def __init__(self, core: Core, banks: nn.ModuleDict):
        super().__init__()

        self.core = core
        self.banks = banks

    def forward(self, input_tokens: torch.Tensor, bank_group: str):
        return self.core(input_tokens, self.banks[bank_group])


def make_banks(core: Core, num_knowledge_entries: int) -> nn.ModuleList:
    return nn.ModuleList(
        [
            Bank(
                core.num_heads,
                num_knowledge_entries,
                core.key_dim,
                core.value_dim,
                core.embed_dim,
            )
            for _ in range(core.num_layers)
        ]
    )


token_vocabulary = [
    "<end>",
    "bob",
    "is",
    "tall",
    "short",
    "happy",
    "sad",
    "gary",
]
token_map = {k: i for i, k in enumerate(token_vocabulary)}


@torch.no_grad()
def generate(model: Model, prompt: list[str], bank_group: str, max_length: int):
    # No optimizations or caching; just for testing.
    tokens = torch.zeros((1, max_length), dtype=torch.long)
    for i, t in enumerate(prompt):
        tokens[0, i] = token_map[t]

    for i in range(len(prompt), max_length):
        logits = model(tokens[:, :i], bank_group=bank_group)
        next_token = logits.argmax(dim=-1)[:, -1]
        tokens[0, i] = next_token
        yield token_vocabulary[next_token.item()]
        if next_token.item() == token_map["<end>"]:
            break


def main():
    core = Core(
        embed_dim=32,
        value_dim=16,
        num_heads=4,
        num_layers=2,
        num_vocab=len(token_map),
    )
    banks = nn.ModuleDict(
        {
            # Two 'groups', for testing purposes. (this abstraction may not hold in the long run; just for experimentation).
            "group1": make_banks(core, num_knowledge_entries=8),
            "group2": make_banks(core, num_knowledge_entries=8),
        }
    )

    model = Model(core=core, banks=banks)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    datasets_by_group = {
        "group1": [["bob", "is", "tall"], ["gary", "is", "happy"]],
        "group2": [["bob", "is", "short"], ["gary", "is", "sad"]],
    }

    # Dummy training loop
    for step in range(100):
        for bank_group in ["group1", "group2"]:
            sequence_tokens = torch.tensor(
                [
                    [token_map[t] for t in random.choice(datasets_by_group[bank_group])]
                    + [token_map["<end>"]]
                ]
            )
            logits = model(sequence_tokens[:, :-1], bank_group=bank_group)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)), sequence_tokens[:, 1:].view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """
            max_logits = logits.argmax(dim=-1)
            output_tokens = [
                [token_vocabulary[index.item()] for index in max_logits[b]]
                for b in range(sequence_tokens.size(0))
            ]
            print(bank_group, repr(" ".join(output_tokens[0])))
            """

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    prompt = ["bob", "is"]
    for bank_group in ["group1", "group2"]:
        print("Generation for", prompt, "with bank group", bank_group + ":")
        for token in generate(model, prompt, bank_group=bank_group, max_length=10):
            print(token)

    prompt = ["gary", "is"]
    for bank_group in ["group1", "group2"]:
        print("Generation for", prompt, "with bank group", bank_group + ":")
        for token in generate(model, prompt, bank_group=bank_group, max_length=10):
            print(token)


if __name__ == "__main__":
    main()
