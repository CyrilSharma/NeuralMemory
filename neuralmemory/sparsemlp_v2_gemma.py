import os

import cupy as cp  # noqa: F401
import cuvs.neighbors.cagra as cagra
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from neuralmemory.sparsemlp_v2 import SparseGatedMLP

# MODEL_ID = "google/gemma-4-31B-it"
MODEL_ID = "google/gemma-4-E4B-it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto").to("cuda")

# Replace MLPs with SparseMLP
# Config: https://huggingface.co/google/gemma-4-31B/blob/main/config.json
#
HIDDEN_DIM = model.config.text_config.intermediate_size
RESIDUAL_STREAM_DIM = model.config.text_config.hidden_size
SPARSITY_DIM = 32

lm = model.model.language_model

print("Found", len(lm.layers), "layers")

# sparse_layers = list(range(16, 18))
sparse_layers = []

for i in range(len(lm.layers)):
    if i not in sparse_layers:
        continue

    print("Replacing MLP in layer", i)
    replacement_sparse_mlp = SparseGatedMLP(
        input_dim=RESIDUAL_STREAM_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=RESIDUAL_STREAM_DIM,
        sparsity_dim=SPARSITY_DIM,
    ).to(model.device)
    # Thankfully these MLPs have no bias
    replacement_sparse_mlp.gate_weight.load_state_dict(
        lm.layers[i].mlp.gate_proj.state_dict()
    )
    replacement_sparse_mlp.in_weight.load_state_dict(
        lm.layers[i].mlp.up_proj.state_dict()
    )
    replacement_sparse_mlp.out_weight.data = lm.layers[
        i
    ].mlp.down_proj.weight.data.T.contiguous()
    lm.layers[i].mlp = replacement_sparse_mlp

    if os.path.exists(f"sparse_index_{i}.bin"):
        print("Loading existing index for layer", i)
        replacement_sparse_mlp.index = cagra.load(f"sparse_index_{i}.bin")
    else:
        print("Rebuilding index for layer", i)
        replacement_sparse_mlp.rebuild_index()

        cagra.save(f"sparse_index_{i}.bin", replacement_sparse_mlp.index)

# Prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

# Process input
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=10)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    # Parse output
    print(processor.parse_response(response))
