from dingo import MetabolicNetwork, PolytopeSampler
import resource

# Load the metabolic network from a JSON file
model = MetabolicNetwork.from_json('/home/nitishmalang/Downloads/iAB_RBC_283.json')

# Create a polytope sampler
sampler = PolytopeSampler(model)

# Measure the initial memory usage
start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# Calculate the polytope
A_res, b_res, Aeq_res, beq_res = sampler.get_polytope()

# Measure the memory usage after calculating the polytope
end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# Calculate memory usage in MB
memory_usage = (end_memory - start_memory) / 1024  # Convert to MB

# Print the sizes of A_res, b_res, Aeq_res, beq_res
print("Size of A_res:", A_res.shape)
print("Size of b_res:", b_res.shape)
print("Size of Aeq_res:", Aeq_res.shape)
print("Size of beq_res:", beq_res.shape)

# Print the memory usage
print(f"Memory used: {memory_usage:.2f} MB")
