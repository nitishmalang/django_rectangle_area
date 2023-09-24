# dingo : a python library for metabolic networks sampling and analysis
# dingo is part of GeomScale project

# Copyright (c) 2021 Apostolos Chalkis

# Licensed under GNU LGPL.3, see LICENCE file

import sys
import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
import math


def fast_fba(lb, ub, S, c):
    """A function to perform a Fast Flux Balance Analysis (Fast FBA)

    Keyword arguments:
    lb -- lower bounds for the fluxes, i.e., a n-dimensional vector
    ub -- upper bounds for the fluxes, i.e., a n-dimensional vector
    S -- the mxn stoichiometric matrix, s.t. Sv = 0
    c -- the objective function to maximize
    """

    n = S.shape[1]

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()

        with gp.Model(env=env) as model:

            # Create variables
            x = model.addMVar(shape=n, vtype=GRB.CONTINUOUS, name="x", lb=lb, ub=ub)

            # Make sparse Aeq
            Aeq_sparse = sp.csr_matrix(S)

            # Set the c vector as numpy vector
            c = np.array(c)

            # Add constraints
            model.addMConstr(Aeq_sparse, x, "=", np.zeros(S.shape[0]))

            # Set the objective
            model.setMObjective(None, c, 0, GRB.MAXIMIZE)

            # Optimize
            model.optimize()

            if model.status == GRB.OPTIMAL:
                biomass_flux_vector = model.getAttr("X", x)
                biomass_objective = model.getObjective().getValue()
            else:
                biomass_flux_vector = None
                biomass_objective = None

    return biomass_flux_vector, biomass_objective


def fast_fva(lb, ub, S, c, opt_percentage=100):
    """A Python function to perform fva using gurobi LP solver
    Returns the value of the optimal solution for all the following linear programs:
    min/max v_i, for all coordinates i=1,...,n, subject to,
    Sv = 0, lb <= v <= ub

    Keyword arguments:
    lb -- lower bounds for the fluxes, i.e., a n-dimensional vector
    ub -- upper bounds for the fluxes, i.e., a n-dimensional vector
    S -- the mxn stoichiometric matrix, s.t. Sv = 0
    c -- the objective function to maximize
    opt_percentage -- consider solutions that give you at least a certain
                      percentage of the optimal solution (default is to consider
                      optimal solutions only)
    """

    if lb.size != S.shape[1] or ub.size != S.shape[1]:
        raise Exception(
            "The number of reactions must be equal to the number of given flux bounds."
        )

    # declare the tolerance that gurobi works properly (we found it experimentally)
    tol = 1e-06

    m = S.shape[0]
    n = S.shape[1]
    beq = np.zeros(m)

    A = np.zeros((2 * n, n), dtype="float")
    A[0:n] = np.eye(n)
    A[n:] -= np.eye(n, n, dtype="float")

    b = np.concatenate((ub, -lb), axis=0)
    b = np.asarray(b, dtype="float")
    b = np.ascontiguousarray(b, dtype="float")

    # call fba to obtain an optimal solution
    max_biomass_flux_vector, max_biomass_objective = fast_fba(lb, ub, S, c)

    # add an additional constraint to impose solutions with at least `opt_percentage` of the optimal solution
    A = np.vstack((A, -c))

    b = np.append(
        b, -(opt_percentage / 100) * tol * math.floor(max_biomass_objective / tol)
    )

    min_fluxes = []
    max_fluxes = []

    try:

        # To avoid printint the output of the optimize() function of Gurobi, we need to set an environment like this
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()

            with gp.Model(env=env) as model:

                # Create variables
                x = model.addMVar(
                    shape=n,
                    vtype=GRB.CONTINUOUS,
                    name="x",
                    lb=-GRB.INFINITY,
                    ub=GRB.INFINITY,
                )

                # Make sparse Aeq
                Aeq_sparse = sp.csr_matrix(S)

                # Make A sparse
                A_sparse = sp.csr_matrix(A)

                # Set the b and beq vectors as numpy vectors
                b = np.array(b)
                beq = np.array(beq)

                # Add constraints
                model.addMConstr(Aeq_sparse, x, "=", beq, name="c")

                # Update the model to include the constraints added
                model.update()

                # Add constraints for the uneqalities of A
                model.addMConstr(A_sparse, x, "<", b, name="d")

                # Update the model with the extra constraints and then print it
                model.update()

                # Loop through the lines of the A matrix, set objective function for each and run the model
                for i in range(n):

                    # Set the ith row of the A matrix as the objective function
                    objective_function = A[
                        i,
                    ]

                    # Set the objective function in the model
                    model.setMObjective(
                        None, objective_function, 0.0, None, None, x, GRB.MINIMIZE
                    )
                    model.update()

                    # Optimize model
                    model.optimize()

                    # If optimized
                    status = model.status
                    if status == GRB.OPTIMAL:

                        # Get the min objective value
                        min_objective = model.getObjective().getValue()
                        min_fluxes.append(min_objective)
                    else:
                        min_fluxes.append(lb[i])

                    # Likewise, for the maximum
                    objective_function = np.asarray([-x for x in objective_function])
                    model.setMObjective(
                        None, objective_function, 0.0, None, None, x, GRB.MINIMIZE
                    )
                    model.update()
                    model.optimize()

                    # Again if optimized
                    status = model.status
                    if status == GRB.OPTIMAL:

                        # Get the max objective value
                        max_objective = -model.getObjective().getValue()
                        max_fluxes.append(max_objective)
                    else:
                        max_fluxes.append(ub[i])

                # Make lists of fluxes numpy arrays
                min_fluxes = np.asarray(min_fluxes)
                max_fluxes = np.asarray(max_fluxes)

                return (
                    min_fluxes,
                    max_fluxes,
                    max_biomass_flux_vector,
                    max_biomass_objective,
                )

    # Print error messages
    except gp.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
    except AttributeError:
        print("Gurobi solver failed.")

def update_model(model, n, Aeq_sparse, beq, lb, ub, A_sparse, b_sparse, c):
    """A function to update the model with new constraints and objective

    Keyword arguments:
    model -- the Gurobi model object
    n -- the number of reactions
    Aeq_sparse -- the sparse matrix for Aeq
    beq -- the beq vector
    lb -- lower bounds for the fluxes, i.e., a n-dimensional vector
    ub -- upper bounds for the fluxes, i.e., a n-dimensional vector
    A_sparse -- the sparse matrix for A
    b_sparse -- the b vector for the inequalities
    c -- the objective function to maximize
    """

    model = model.copy()
    x = model.getVars()

    # Clear the previous constraints and objective
    model.removeMConstrs(model.getConstrs())
    model.removeMObjective()

    # Add new constraints
    model.addMConstr(Aeq_sparse, x, "=", beq, name="c")

    # Add constraints for the inequalities of A
    model.addMConstr(A_sparse, x, "<", b_sparse, name="d")

    # Set new objective
    model.setMObjective(None, c, 0, GRB.MAXIMIZE)

    # Update the model to include the new constraints and objective
    model.update()

    return model



def fast_remove_redundant_facets(lb, ub, S, c, opt_percentage=100):
    """A function to find and remove the redundant facets and to find
    the facets with very small offset and to set them as equalities

    Keyword arguments:
    lb -- lower bounds for the fluxes, i.e., a n-dimensional vector
    ub -- upper bounds for the fluxes, i.e., a n-dimensional vector
    S -- the mxn stoichiometric matrix, s.t. Sv = 0
    c -- the objective function to maximize
    opt_percentage -- consider solutions that give you at least a certain
                      percentage of the optimal solution (default is to consider
                      optimal solutions only)
    """

    if lb.size != S.shape[1] or ub.size != S.shape[1]:
        raise Exception(
            "The number of reactions must be equal to the number of given flux bounds."
        )

    # Declare the tolerance that gurobi works properly (we found it experimentally)
    redundant_facet_tol = 1e-07
    tol = 1e-06

    m = S.shape[0]
    n = S.shape[1]
    beq = np.zeros(m)
    Aeq_res = S

    A = np.zeros((2 * n, n), dtype="float")
    A[0:n] = np.eye(n)
    A[n:] -= np.eye(n, n, dtype="float")

    b = np.concatenate((ub, -lb), axis=0)
    b = np.asarray(b, dtype="float")
    b = np.ascontiguousarray(b, dtype="float")

    # Call fba to obtain an optimal solution
    max_biomass_flux_vector, max_biomass_objective = fast_fba(lb, ub, S, c)
    val = -np.floor(max_biomass_objective / tol) * tol * opt_percentage / 100

    # Initialize variables
    indices_iter = list(range(n))
    removed = 1
    offset = 1
    facet_left_removed = np.zeros((1, n), dtype=bool)
    facet_right_removed = np.zeros((1, n), dtype=bool)

    # Initialize A_res and b_res as empty lists
    b_res = []
    A_res = []

    try:
        # To avoid printing the output of the optimize() function of Gurobi, we need to set an environment like this
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()

            with gp.Model(env=env) as model:

                # Create variables
                x = model.addMVar(
                    shape=n,
                    vtype=GRB.CONTINUOUS,
                    name="x",
                    lb=lb,
                    ub=ub,
                )

                # Make sparse Aeq
                Aeq_sparse = sp.csr_matrix(S)

                # Make A sparse
                A_sparse = sp.csr_matrix(np.array(-c))
                b_sparse = np.array(val)

                # Set the b and beq vectors as numpy vectors
                b = np.array(b)
                beq = np.array(beq)

                # Add constraints
                model.addMConstr(Aeq_sparse, x, "=", beq, name="c")

                # Update the model to include the constraints added
                model.update()

                model_iter = model.copy()

                while removed > 0 or offset > 0:
                    removed = 0
                    offset = 0
                    indices = indices_iter
                    indices_iter = []

                    for i in indices:
                        redundant_facet_left = True
                        redundant_facet_right = True

                        # For the maximum
                        objective_function_max = np.asarray([-x for x in A[i, :]])
                        model_iter = update_model(
                            model_iter,
                            n,
                            Aeq_sparse,
                            beq,
                            lb,
                            ub,
                            A_sparse,
                            [val],
                            objective_function_max,
                        )
                        model_iter.optimize()

                        status = model_iter.status
                        if status == GRB.OPTIMAL:
                            max_objective = -model_iter.getObjective().getValue()
                        else:
                            max_objective = ub[i]

                        if not facet_left_removed[0, i]:
                            lb_iter = lb.copy()
                            lb_iter[i] = lb_iter[i] - 1
                            model_iter = update_model(
                                model_iter,
                                n,
                                Aeq_sparse,
                                beq,
                                lb_iter,
                                ub,
                                A_sparse,
                                [val],
                                objective_function_max,
                            )
                            model_iter.optimize()

                            status = model_iter.status
                            if status == GRB.OPTIMAL:
                                min_objective2 = model_iter.getObjective().getValue()
                                if (
                                    np.abs(min_objective2 - min_objective)
                                    > redundant_facet_tol
                                ):
                                    redundant_facet_left = False
                                else:
                                    removed += 1
                                    facet_left_removed[0, i] = True
                                    # Handle redundant left facet

                        # For the minimum
                        objective_function = np.asarray(A[i, :])
                        model_iter = update_model(
                            model_iter,
                            n,
                            Aeq_sparse,
                            beq,
                            lb,
                            ub,
                            A_sparse,
                            [val],
                            objective_function,
                        )
                        model_iter.optimize()

                        status = model_iter.status
                        if status == GRB.OPTIMAL:
                            min_objective = model_iter.getObjective().getValue()
                        else:
                            min_objective = lb[i]

                        if not facet_right_removed[0, i]:
                            ub_iter = ub.copy()
                            ub_iter[i] = ub_iter[i] + 1
                            model_iter = update_model(
                                model_iter,
                                n,
                                Aeq_sparse,
                                beq,
                                lb,
                                ub_iter,
                                A_sparse,
                                [val],
                                objective_function,
                            )
                            model_iter.optimize()

                            status = model_iter.status
                            if status == GRB.OPTIMAL:
                                max_objective2 = model_iter.getObjective().getValue()
                                if (
                                    np.abs(max_objective2 - max_objective)
                                    > redundant_facet_tol
                                ):
                                    redundant_facet_right = False
                                else:
                                    removed += 1
                                    facet_right_removed[0, i] = True
                                    # Handle redundant right facet

                        width = abs(max_objective - min_objective)

                        if width < redundant_facet_tol:
                            offset += 1
                            Aeq_res = np.vstack((Aeq_res, A[i, :]))
                            beq_res = np.append(beq_res, min(max_objective, min_objective))
                            ub[i] = sys.float_info.max
                            lb[i] = -sys.float_info.max
                        else:
                            indices_iter.append(i)

                            if not redundant_facet_left:
                                A_res.append(A[n + i, :])
                                b_res.append(b[n + i])

                            if not redundant_facet_right:
                                A_res.append(A[i, :])
                                b_res.append(b[i])

                            ub[i] = sys.float_info.max
                            lb[i] = -sys.float_info.max

    finally:
        model.dispose()
        gp.closeEnv()

    # Convert A_res to a NumPy array
    A_res = np.array(A_res, dtype=float)

    return A_res, b_res, Aeq_res, beq_res


def fast_inner_ball(A, b):
    """A Python function to compute the maximum inscribed ball in the given polytope using gurobi LP solver
    Returns the optimal solution for the following linear program:
    max r, subject to,
    a_ix + r||a_i|| <= b, i=1,...,n

    Keyword arguments:
    A -- an mxn matrix that contains the normal vectors of the facets of the polytope row-wise
    b -- a m-dimensional vector
    """

    extra_column = []

    m = A.shape[0]
    n = A.shape[1]

    for i in range(A.shape[0]):
        entry = np.linalg.norm(
            A[
                i,
            ]
        )
        extra_column.append(entry)

    column = np.asarray(extra_column)
    A_expand = np.c_[A, column]

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()

        d = A_expand.shape[1]

        with gp.Model(env=env) as model:

            # Create variable
            x = model.addMVar(
                shape=d,
                vtype=GRB.CONTINUOUS,
                name="x",
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
            )
            model.update()

            # Make A_full_dim sparse
            A_expand_sparse = sp.csr_matrix(A_expand.astype("float"))

            # Add constraints
            model.addMConstr(A_expand_sparse, x, "<", b, name="c")
            model.update()

            # Set the ith row of the A matrix as the objective function
            a = np.ones((1, n + 1))
            azero = np.zeros((1, n))
            a[:, :-1] = azero
            objective_function = a[0]

            # Set the objective function in the model
            model.setMObjective(
                None, objective_function, 0.0, None, None, x, GRB.MAXIMIZE
            )
            model.update()

            # Optimize model
            model.optimize()

            # Get the solution returned
            varss = model.getVars()

            # Get the center point and the radius of max ball from the solution of LP; its last element is the radius
            point = []
            for i in range(len(varss)):
                if i == len(varss) - 1:
                    r = varss[i].x
                else:
                    value = varss[i].x
                    point.append(value)

            # And check whether the computed radius is negative
            if r < 0:
                print(
                    "The radius calculated has negative value. The polytope is infeasible or something went wrong with the solver"
                )
            else:
                return point, r
