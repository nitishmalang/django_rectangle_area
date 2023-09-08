import timeit
from dingo import MetabolicNetwork, PolytopeSampler

# Load a metabolic network model (replace with your JSON file)
model = MetabolicNetwork.from_json('/home/nitishmalang/Downloads/iAB_RBC_283.json')

# Create a PolytopeSampler
sampler = PolytopeSampler(model)

# Number of repetitions for timing (adjust as needed)
num_repeats = 10

# Measure execution time of get_polytope
execution_time = timeit.timeit(
    stmt="sampler.get_polytope()",
    setup="from __main__ import PolytopeSampler, model, sampler",
    number=num_repeats,
)

# Print the average execution time
print(f"Execution Time (Avg): {execution_time / num_repeats:.6f} seconds")
