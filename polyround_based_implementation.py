import numpy as np
import warnings
import math
from dingo.MetabolicNetwork import MetabolicNetwork
from dingo.fva import slow_fva
from dingo.utils import (
    map_samples_to_steady_states,
    get_matrices_of_low_dim_polytope,
    get_matrices_of_full_dim_polytope,
)

try:
    import gurobipy
    from dingo.gurobi_based_implementations import (
        fast_fba,
        fast_fva,
        fast_inner_ball,
        fast_remove_redundant_facets,
    )
except ImportError as e:
    pass

from volestipy import HPolytope

from PolyRound.api import PolyRoundApi
from PolyRound.static_classes.lp_utils import ChebyshevFinder
from PolyRound.settings import PolyRoundSettings

class PolytopeSampler:
    def __init__(self, metabol_net):

        if not isinstance(metabol_net, MetabolicNetwork):
            raise Exception("An unknown input object given for initialization.")

        self._metabolic_network = metabol_net
        self._A = []
        self._b = []
        self._N = []
        self._N_shift = []
        self._T = []
        self._T_shift = []
        self._parameters = {}
        self._parameters["nullspace_method"] = "sparseQR"
        self._parameters["opt_percentage"] = self.metabolic_network.parameters[
            "opt_percentage"
        ]
        self._parameters["distribution"] = "uniform"
        self._parameters["first_run_of_mmcs"] = True
        self._parameters["remove_redundant_facets"] = True

        try:
            import gurobipy

            self._parameters["fast_computations"] = True
            self._parameters["tol"] = 1e-06
        except ImportError as e:
            self._parameters["fast_computations"] = False
            self._parameters["tol"] = 1e-03

    def get_polytope(self):
        """A member function to derive the corresponding full dimensional polytope
        and an isometric linear transformation that maps the latter to the initial space.
        """

        if (
            self._A == []
            or self._b == []
            or self._N == []
            or self._N_shift == []
            or self._T == []
            or self._T_shift == []
        ):

            (
                max_biomass_flux_vector,
                max_biomass_objective,
            ) = self._metabolic_network.fba()

            if (
                self._parameters["fast_computations"]
                and self._parameters["remove_redundant_facets"]
            ):

                A, b, Aeq, beq = fast_remove_redundant_facets(
                    self._metabolic_network.lb,
                    self._metabolic_network.ub,
                    self._metabolic_network.S,
                    self._metabolic_network.biomass_function,
                    self._parameters["opt_percentage"],
                )
            else:
                if (not self._parameters["fast_computations"]) and self._parameters[
                    "remove_redundant_facets"
                ]:
                    warnings.warn(
                        "We continue without redundancy removal (slow mode is ON)"
                    )

                (
                    min_fluxes,
                    max_fluxes,
                    max_biomass_flux_vector,
                    max_biomass_objective,
                ) = self._metabolic_network.fva()

                A, b, Aeq, beq = get_matrices_of_low_dim_polytope(
                    self._metabolic_network.S,
                    self._metabolic_network.lb,
                    self._metabolic_network.ub,
                    min_fluxes,
                    max_fluxes,
                )

            if (
                A.shape[0] != b.size
                or A.shape[1] != Aeq.shape[1]
                or Aeq.shape[0] != beq.size
            ):
                raise ValueError("Shape mismatch in polytope matrices.")

            self._A = A
            self._b = b
            self._N, self._N_shift, self._T, self._T_shift = self._compute_transforms()

        return HPolytope(
            A=self._A,
            b=self._b,
            N=self._N,
            N_shift=self._N_shift,
            T=self._T,
            T_shift=self._T_shift,
            parameters=self._parameters,
        )

    def _compute_transforms(self):
        # Internal method to compute transformation matrices.
        # This method is not provided in the original code.
        pass

    # New methods using PolyRoundImplementation

    def polyround_simplify_polytope(self):
        """
        Simplify the polytope using PolyRound's simplify_polytope method.
        """
        # Convert the metabolic network to a Polytope object.
        polytope = PolyRoundApi.cobra_model_to_polytope(self._metabolic_network)

        # Make a settings object for PolyRound - optional.
        settings = PolyRoundSettings()

        # Simplify the polytope using PolyRound.
        simplified_polytope = PolyRoundApi.simplify_polytope(polytope, settings)

        return simplified_polytope

    def polyround_transform_polytope(self, polytope):
        """
        Transform the given polytope using PolyRound's transform_polytope method.
        """
        # Convert the polytope to a Polytope object if needed.
        if not isinstance(polytope, Polytope):
            polytope = PolyRoundApi.cobra_model_to_polytope(polytope)

        # Transform the polytope using PolyRound.
        transformed_polytope = PolyRoundApi.transform_polytope(polytope)

        return transformed_polytope

    def polyround_round_polytope(self, polytope):
        """
        Round the given polytope using PolyRound's round_polytope method.
        """
        # Convert the polytope to a Polytope object if needed.
        if not isinstance(polytope, Polytope):
            polytope = PolyRoundApi.cobra_model_to_polytope(polytope)

        # Round the polytope using PolyRound.
        rounded_polytope = PolyRoundApi.round_polytope(polytope)

        return rounded_polytope

if __name__ == '__main__':
    # Load the metabolic model
    model = MetabolicNetwork.from_json('/home/nitishmalang/Downloads/iAB_RBC_283.json')

    # Create a PolytopeSampler instance
    sampler = PolytopeSampler(model)

    # Measure the time taken by get_polytope
    start_time = time.time()
    polytope_dingo = sampler.get_polytope()
    end_time = time.time()

    time_taken_dingo = end_time - start_time
    print(f"Time taken by dingo's get_polytope: {time_taken_dingo} seconds")

    # Create a PolyRoundImplementation instance
    polyround_implementation = PolyRoundImplementation(model)

    # Measure the time taken by polyround_simplify_polytope
    start_time = time.time()
    polytope_simplified = polyround_implementation.polyround_simplify_polytope()
    end_time = time.time()

    time_taken_simplify = end_time - start_time
    print(f"Time taken by PolyRound's simplify_polytope: {time_taken_simplify} seconds")

    # Measure the time taken by polyround_transform_polytope
    start_time = time.time()
    polytope_transformed = polyround_implementation.polyround_transform_polytope(polytope_simplified)
    end_time = time.time()

    time_taken_transform = end_time - start_time
    print(f"Time taken by PolyRound's transform_polytope: {time_taken_transform} seconds")

    # Measure the time taken by polyround_round_polytope
    start_time = time.time()
    polytope_rounded = polyround_implementation.polyround_round_polytope(polytope_transformed)
    end_time = time.time()

    time_taken_round = end_time - start_time
    print(f"Time taken by PolyRound's round_polytope: {time_taken_round} seconds")







