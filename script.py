import time
import memory_profiler
from dingo import MetabolicNetwork, PolytopeSampler
from PolyRound.mutable_classes.polytope import Polytope
from PolyRoundApi import PolyRoundApi

def load_metabolic_network(model_path):
    # Load your metabolic network model using dingo
    # Return the MetabolicNetwork object

def run_dingo_test(model):
    sampler = PolytopeSampler(model)
    start_time = time.time()
    # Call the get_polytope method
    A, b, N, N_shift = sampler.get_polytope()
    end_time = time.time()
    elapsed_time = end_time - start_time
    memory_usage = memory_profiler.memory_usage()
    return A, b, N, N_shift, elapsed_time, memory_usage

def run_polyround_test(polytope):
    # Create a Polytope instance from A and b
    start_time = time.time()
    # Call the simplify_polytope method from PolyRoundApi
    simplified_polytope = PolyRoundApi.simplify_polytope(polytope)
    end_time = time.time()
    elapsed_time = end_time - start_time
    memory_usage = memory_profiler.memory_usage()
    return simplified_polytope, elapsed_time, memory_usage

def main():
    model_path = "path/to/your/model.xml"
    model = load_metabolic_network(model_path)

    # Run dingo test
    dingo_A, dingo_b, dingo_N, dingo_N_shift, dingo_time, dingo_memory = run_dingo_test(model)

    # Create a Polytope instance from dingo_A and dingo_b
    polytope = Polytope(dingo_A, dingo_b)

    # Run PolyRound test
    polyround_polytope, polyround_time, polyround_memory = run_polyround_test(polytope)

    # Print or save the results
    print("Dingo time:", dingo_time)
    print("Dingo memory:", dingo_memory)
    print("PolyRound time:", polyround_time)
    print("PolyRound memory:", polyround_memory)

if __name__ == "__main__":
    main()
