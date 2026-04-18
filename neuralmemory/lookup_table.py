import time

import cupy as cp
import cuvs.neighbors.cagra as cagra

index = cagra.IndexParams(metric="inner_product")

# create a table using Gemma 4 size (21504 hidden vectors of dimensionality 5376)
# https://huggingface.co/google/gemma-4-31B/blob/main/config.json

for i in range(10):
    num_hidden_vectors = 21504 * 10
    residual_stream_size = 5376 // 2
    keys = cp.random.random_sample(
        (num_hidden_vectors, residual_stream_size), dtype=cp.float32
    )
    values = cp.random.random_sample(
        (residual_stream_size, num_hidden_vectors), dtype=cp.float32
    )

    # https://docs.rapids.ai/api/cuvs/nightly/python_api/neighbors_cagra/#index-build
    t0 = time.time()
    key_lookup = cagra.build(index, keys)
    t1 = time.time()

    print(f"Time taken to build lookup table: {t1 - t0} seconds")

# Create a random query
queries = cp.random.random_sample((1, residual_stream_size), dtype=cp.float32)
t0 = time.time()
# Perform a lookup for the query, retrieving the top 4 nearest neighbors
distances, indices = cagra.search(cagra.SearchParams(), key_lookup, queries, k=4)
t1 = time.time()
print(f"Time taken for lookup: {t1 - t0} seconds")
print("Indices of nearest neighbors:", indices)
print("Distances to nearest neighbors:", distances)

# 'query' messages get sent to the lookup table in proportion to the loss gradients for the given lookup.
# one approximation we can make is to sparsely propagate corrections to the query/key pairs that seem to require the strongest change (e.g. top k tokens with highest loss gradients). this refers to the tokens at which the memory is conjured. we're only making fixes up to first order anyway, so combinations of changes don't really matter that much. this effectively liquidates the required changes across training timesteps, which can be much more efficient. are there any technical barriers here? if we wanted to implement this, we would effectively have a 'queue' of updates of the form (query vector to apply to key, key index, change magnitude). we can just take the top k by change magnitude. same applies to value vectors; (downstream gradient to apply to value, value index, change magnitude)
# the big story point is that we can liquidate the changes because updates are only made to first order anyway.

# size vs. time to build index
# 21504 -> 9.17s
# 21504*2 -> 11.4s
# 21504*10 -> 11.6s
# building a randomly-generated 21504*10 three times:
# - 11.53s
# - 11.51s
# - 11.49s
# this suggests that it's not a library overhead, but a building overhead
