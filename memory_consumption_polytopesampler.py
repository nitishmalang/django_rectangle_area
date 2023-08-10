from memory_profiler import profile
import numpy as np
from dingo import MetabolicNetwork
from dingo import PolytopeSampler

@profile
def measure_memory_usage(model_path):
    # Load the model
    model = MetabolicNetwork.from_json(model_path)

    # Create the sampler
    sampler = PolytopeSampler(model)

    # Call the get_polytope function
    A, b, N, N_shift = sampler.get_polytope()

if _name_ == "_main_":
    # Path to your model JSON file
    model_path = '/home/nitishmalang/Downloads/iIT341.json'

    # Measure memory consumption
    measure_memory_usage(model_path)
