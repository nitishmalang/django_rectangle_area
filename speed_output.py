import time
from dingo import MetabolicNetwork, PolytopeSampler

def measure_speed_dingo(dingo_model_path):
    # Load the metabolic network from a JSON file
    model = MetabolicNetwork.from_json('/home/nitishmalang/e_coli_core.json')

    # Create a polytope sampler
    sampler = PolytopeSampler(model)

    # Measure the start time
    start_time = time.time()

    # Calculate the polytope
    A_res, b_res, Aeq_res, beq_res = sampler.get_polytope()

    # Measure the end time
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    # Print the sizes of A_res, b_res, Aeq_res, beq_res
    print("Size of A_res:", A_res.shape)
    print("Size of b_res:", b_res.shape)
    print("Size of Aeq_res:", Aeq_res.shape)
    print("Size of beq_res:", beq_res.shape)

    # Print the execution time
    print(f"Execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    # Define the path to your Dingo model JSON file
    dingo_model_path = '/path/to/your/dingo_model.json'

    # Measure execution time for Dingo's get_polytope
    print("Speed measurement for Dingo:")
    measure_speed_dingo(dingo_model_path)
