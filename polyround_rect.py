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

# Import PolyRound modules for polytope simplification and transformation
from PolyRound.mutable_classes.polytope import Polytope
from PolyRound.api import PolyRoundApi


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
        and a isometric linear transformation that maps the latter to the initial space.
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
                raise Exception("Preprocess for full dimensional polytope failed.")

            A = np.vstack((A, -self._metabolic_network.biomass_function))

            b = np.append(
                b,
                -np.floor(max_biomass_objective / self._parameters["tol"])
                * self._parameters["tol"]
                * self._parameters["opt_percentage"]
                / 100,
            )

            (
                self._A,
                self._b,
                self._N,
                self._N_shift,
            ) = get_matrices_of_full_dim_polytope(A, b, Aeq, beq)

            n = self._A.shape[1]
            self._T = np.eye(n)
            self._T_shift = np.zeros(n)

        return self._A, self._b, self._N, self._N_shift

    def generate_steady_states_no_multiphase(
        self, method="billiard_walk", n=1000, burn_in=0, thinning=1
    ):
        """A member function to sample steady states.

        Keyword arguments:
        method -- An MCMC method to sample, i.e. {'billiard_walk', 'cdhr', 'rdhr', 'ball_walk', 'dikin_walk', 'john_walk', 'vaidya_walk'}
        n -- the number of steady states to sample
        burn_in -- the number of points to burn before sampling
        thinning -- the walk length of the chain
        """

        self.get_polytope()

        P = HPolytope(self._A, self._b)

        samples = P.generate_samples(method, n, burn_in, thinning, self._parameters["fast_computations"])
        samples_T = samples.T

        steady_states = map_samples_to_steady_states(
                samples_T, self._N, self._N_shift
            )

        return steady_states

    def round_polytope(self, method="john_position"):
        """
        Round the polytope using PolyRound's transformation method.

        Parameters:
        method (str): The PolyRound transformation method for rounding.

        Returns:
        None (modifies the polytope in-place)
        """
        # Step 1: Get the polytope representation
        A, b, _, _ = self.get_polytope()

        # Step 2: Convert the polytope representation to PolyRound's Polytope class
        polytope = Polytope(A, b)

        # Step 3: Simplify the polytope using PolyRound's simplify_polytope method
        polytope = PolyRoundApi.simplify_polytope(polytope)

        # Step 4: Transform the polytope using PolyRound's transform_polytope method
        polytope = PolyRoundApi.transform_polytope(polytope)

        # Step 5: Update the polytope parameters with the simplified and transformed polytope
        self._A = polytope.A
        self._b = polytope.b
        self._N = polytope.N
        self._N_shift = polytope.N_shift
        self._T = polytope.T
        self._T_shift = polytope.T_shift

    # ... Rest of the class methods ...

