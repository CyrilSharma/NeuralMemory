import cupy as cp  # noqa: F401
import cuvs.neighbors.cagra as cagra
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
SPARSITY_DIM = 16

lm = model.model.language_model

print("Found", len(lm.layers), "layers")

for i in range(len(lm.layers)):
    print("Replacing MLP in layer", i)
    replacement_sparse_mlp = SparseGatedMLP(
        input_dim=RESIDUAL_STREAM_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=RESIDUAL_STREAM_DIM,
        sparsity_dim=SPARSITY_DIM,
    )
    # Thankfully these MLPs have no bias
    replacement_sparse_mlp.gate_weight.load_state_dict(
        lm.layers[i].mlp.gate_proj.state_dict()
    )
    replacement_sparse_mlp.in_weight.load_state_dict(
        lm.layers[i].mlp.up_proj.state_dict()
    )
    replacement_sparse_mlp.out_weight.data = lm.layers[i].mlp.down_proj.weight.data
    lm.layers[i].mlp = replacement_sparse_mlp

    print("Rebuilding index for layer", i)
    replacement_sparse_mlp.rebuild_index()

    cagra.save(f"sparse_index_{i}.bin", replacement_sparse_mlp.index)
