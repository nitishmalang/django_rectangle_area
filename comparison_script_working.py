import tracemalloc
from PolyRound.api import PolyRoundApi
from PolyRound.static_classes.lp_utils import ChebyshevFinder
from PolyRound.settings import PolyRoundSettings
from dingo import MetabolicNetwork, PolytopeSampler


polyround_model_path = '/home/nitishmalang/Downloads/iAB_RBC_283.xml'
dingo_model_path = '/home/nitishmalang/Downloads/iAB_RBC_283.json'

def measure_memory_usage_polyround(model_path):
    settings = PolyRoundSettings()

    # Import model and create Polytope object
    polytope = PolyRoundApi.sbml_to_polytope(model_path)

    # Remove redundant constraints and refunction inequality constraints that are de-facto equalities.
    # Due to these inequalities, the polytope is empty (distance from chebyshev center to boundary is zero)
    x, dist = ChebyshevFinder.chebyshev_center(polytope, settings)
    print(dist)
    
    # Measure memory usage before simplify_polytope
    tracemalloc.start()
    simplified_polytope = PolyRoundApi.simplify_polytope(polytope)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_used_simplify = peak / 1024 / 1024  # Convert bytes to MiB
    print(f"Memory used during simplify_polytope: {memory_used_simplify} MiB")
    
    # Embed the polytope in a space where it has non-zero volume
    transformed_polytope = PolyRoundApi.transform_polytope(simplified_polytope)
    
    # Measure memory usage before transform_polytope
    tracemalloc.start()
    rounded_polytope = PolyRoundApi.round_polytope(transformed_polytope)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_used_transform = peak / 1024 / 1024  # Convert bytes to MiB
    print(f"Memory used during transform_polytope: {memory_used_transform} MiB")

def measure_memory_usage_dingo(model_path):
    tracemalloc.start()
    # Load the model
    model = MetabolicNetwork.from_json(model_path)

    # Create the sampler
    sampler = PolytopeSampler(model)

    # Call the get_polytope function
    A, b, N, N_shift = sampler.get_polytope()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_used_get_polytope = peak / 1024 / 1024  # Convert bytes to MiB
    print(f"Memory used during get_polytope: {memory_used_get_polytope} MiB")

if __name__ == "__main__":
    # Measure memory consumption for PolyRound
    print("Memory usage for PolyRound:")
    measure_memory_usage_polyround(polyround_model_path)

    # Measure memory consumption for Dingo
    print("Memory usage for Dingo:")
    measure_memory_usage_dingo(dingo_model_path)
    print('For iAB_RBC model')
