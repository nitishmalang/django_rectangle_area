


def fast_remove_redundant_facets(lb, ub, S, c, opt_percentage=100):


    if lb.size != S.shape[1] or ub.size != S.shape[1]:
        raise Exception(
            "The number of reactions must be equal to the number of given flux bounds."
        )

    # Declare the tolerance that Gurobi works properly (found experimentally)
    redundant_facet_tol = 1e-07
    tol = 1e-06

    m = S.shape[0]
    n = S.shape[1]
    beq = np.zeros(m)

    A = np.zeros((2 * n, n), dtype=float)
    A[0:n] = np.eye(n)
    A[n:] -= np.eye(n, n, dtype=float)

    b = np.concatenate((ub, -lb), axis=0).astype(float)

    # Call FBA to obtain an optimal solution
    max_biomass_flux_vector, max_biomass_objective = fast_fba(lb, ub, S, c)
    val = -np.floor(max_biomass_objective / tol) * tol * opt_percentage / 100

    b_res = []
    A_res = csr_matrix((0, n), dtype=float)
    beq_res = np.array(beq)

    try:
        # Initialize Aeq_res as a sparse matrix
        Aeq_res = csr_matrix(S)

        # To avoid printing the output of the optimize() function of Gurobi, set an environment like this
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
                Aeq_sparse = csr_matrix(S)

                # Make A sparse
                A_sparse = csr_matrix(np.array(-c))
                b_sparse = np.array(val)

                # Add constraints
                model.addMConstr(Aeq_sparse, x, "=", beq, name="c")

                # Update the model to include the constraints added
                model.update()

                # Add constraints for the inequalities of A
                model.addMConstr(A_sparse, x, "<", [val], name="d")

                # Update the model with the extra constraints and then print it
                model.update()

                model_iter = model.copy()

                # Initialize
                indices_iter = list(range(n))
                removed = 1
                offset = 1
                facet_left_removed = np.zeros((1, n), dtype=bool)
                facet_right_removed = np.zeros((1, n), dtype=bool)

                # Loop until no redundant facets are found
                while removed > 0 or offset > 0:

                    removed = 0
                    offset = 0
                    indices = indices_iter.copy()
                    indices_iter = []

                    Aeq_sparse = csr_matrix(Aeq_res)
                    beq = np.array(beq_res)

                    b_res = []
                    A_res = csr_matrix((0, n), dtype=float)
                    for i in indices:

                        # Set the ith row of the A matrix as the objective function
                        objective_function = A[i, :]

                        redundant_facet_right = True
                        redundant_facet_left = True

                        # For the maximum
                        objective_function_max = np.asarray([-x for x in objective_function])
                        model_iter.setObjective(objective_function_max, GRB.MAXIMIZE)
                        model_iter.optimize()

                        # Again if optimized
                        status = model_iter.status
                        if status == GRB.OPTIMAL:
                            # Get the max objective value
                            max_objective = -model_iter.getObjective().getValue()
                        else:
                            max_objective = ub[i]

                        # If this facet was not removed in a previous iteration
                        if not facet_right_removed[0, i]:
                            ub_iter = ub.copy()
                            ub_iter[i] = ub_iter[i] + 1
                            model_iter.setObjective(objective_function_max, GRB.MAXIMIZE)
                            model_iter.optimize()

                            status = model_iter.status
                            if status == GRB.OPTIMAL:
                                # Get the max objective value with relaxed inequality
                                max_objective2 = -model_iter.getObjective().getValue()
                                if np.abs(max_objective2 - max_objective) > redundant_facet_tol:
                                    redundant_facet_right = False
                                else:
                                    removed += 1
                                    facet_right_removed[0, i] = True

                        model_iter.setObjective(objective_function, GRB.MINIMIZE)
                        model_iter.optimize()

                        # If optimized
                        status = model_iter.status
                        if status == GRB.OPTIMAL:
                            # Get the min objective value
                            min_objective = model_iter.getObjective().getValue()
                        else:
                            min_objective = lb[i]

                        # If this facet was not removed in a previous iteration
                        if not facet_left_removed[0, i]:
                            lb_iter = lb.copy()
                            lb_iter[i] = lb_iter[i] - 1
                            model_iter.setObjective(objective_function, GRB.MINIMIZE)
                            model_iter.optimize()

                            status = model_iter.status
                            if status == GRB.OPTIMAL:
                                # Get the min objective value with relaxed inequality
                                min_objective2 = model_iter.getObjective().getValue()
                                if np.abs(min_objective2 - min_objective) > redundant_facet_tol:
                                    redundant_facet_left = False
                                else:
                                    removed += 1
                                    facet_left_removed[0, i] = True

                        if (not redundant_facet_left) or (not redundant_facet_right):
                            width = abs(max_objective - min_objective)

                            # Check whether the offset in this dimension is small (and set an equality)
                            if width < redundant_facet_tol:
                                offset += 1
                                Aeq_res = vstack(
                                    [
                                        Aeq_res,
                                        csr_matrix([A[i, :]]),
                                    ],
                                    format="csr",
                                )
                                beq_res = np.append(beq_res, min(max_objective, min_objective))
                                # Remove the bounds on this dimension
                                ub[i] = sys.float_info.max
                                lb[i] = -sys.float_info.max
                            else:
                                # Store this dimension
                                indices_iter.append(i)

                                if not redundant_facet_left:
                                    # Not a redundant inequality
                                    A_res = vstack(
                                        [
                                            A_res,
                                            csr_matrix([A[n + i, :]]),
                                        ],
                                        format="csr",
                                    )
                                    b_res.append(b[n + i])
                                else:
                                    lb[i] = -sys.float_info.max

                                if not redundant_facet_right:
                                    # Not a redundant inequality
                                    A_res = vstack(
                                        [
                                            A_res,
                                            csr_matrix([A[i, :]]),
                                        ],
                                        format="csr",
                                    )
                                    b_res.append(b[i])
                                else:
                                    ub[i] = sys.float_info.max
                        else:
                            # Remove the bounds on this dimension
                            ub[i] = sys.float_info.max
                            lb[i] = -sys.float_info.max

                    b_res = np.asarray(b_res)
                    A_res = A_res.tocsr()
                    beq_res = beq_res.astype(float)

                return A_res, b_res, Aeq_res, beq_res

    # Print error messages
    except gp.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
    except AttributeError:
        print("Gurobi solver failed.")

    b_upper = sp.csr_matrix((ub, (range(n), [0] * n)), shape=(n, 1), dtype="float")
    b_lower = sp.csr_matrix((lb, (range(n), [0] * n)), shape=(n, 1), dtype="float")

    b = sp.vstack([b_upper, -b_lower], format="csr")

    # Call FBA to obtain an optimal solution
    max_biomass_flux_vector, max_biomass_objective = fast_fba(lb, ub, S, c)
    val = -np.floor(max_biomass_objective / tol) * tol * opt_percentage / 100

    b_res = []
    A_res = sp.lil_matrix((0, n), dtype="float")
    beq_res = np.array(beq)

    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()

            with gp.Model(env=env) as model:
                x = model.addMVar(shape=n, vtype=GRB.CONTINUOUS, name="x", lb=lb, ub=ub)

                # Make sparse A_sparse
                A_sparse = sp.csr_matrix(np.array(-c))
                b_sparse = np.array(val)

                # Add constraints
                model.addMConstr(Aeq_sparse, x, "=", beq, name="c")

                # Update the model to include the constraints added
                model.update()

                # Add constraints for the inequalities of A
                model.addMConstr(A_sparse, x, "<", [val], name="d")

                # Update the model with the extra constraints and then print it
                model.update()

                model_iter = model.copy()

                indices_iter = range(n)
                removed = 1
                offset = 1
                facet_left_removed = np.zeros((1, n), dtype=bool)
                facet_right_removed = np.zeros((1, n), dtype=bool)

                while removed > 0 or offset > 0:
                    removed = 0
                    offset = 0
                    indices = indices_iter.copy()
                    indices_iter = []

                    Aeq_sparse = sp.csr_matrix(Aeq_res)
                    beq = np.array(beq_res)

                    b_res = []
                    A_res = sp.lil_matrix((0, n), dtype="float")
                    for i in indices:
                        objective_function = A[i, :]

                        redundant_facet_right = True
                        redundant_facet_left = True

                        objective_function_max = -objective_function
                        model_iter.setObjective(objective_function_max, GRB.MAXIMIZE)
                        model_iter.optimize()

                        status = model_iter.status
                        if status == GRB.OPTIMAL:
                            max_objective = -model_iter.getObjective().getValue()
                        else:
                            max_objective = ub[i]

                        if not facet_right_removed[0, i]:
                            ub_iter = ub.copy()
                            ub_iter[i] = ub_iter[i] + 1
                            model_iter.setObjective(objective_function_max, GRB.MAXIMIZE)
                            model_iter.optimize()

                            status = model_iter.status
                            if status == GRB.OPTIMAL:
                                max_objective2 = -model_iter.getObjective().getValue()
                                if np.abs(max_objective2 - max_objective) > redundant_facet_tol:
                                    redundant_facet_right = False
                                else:
                                    removed += 1
                                    facet_right_removed[0, i] = True

                        model_iter.setObjective(objective_function, GRB.MINIMIZE)
                        model_iter.optimize()

                        status = model_iter.status
                        if status == GRB.OPTIMAL:
                            min_objective = model_iter.getObjective().getValue()
                        else:
                            min_objective = lb[i]

                        if not facet_left_removed[0, i]:
                            lb_iter = lb.copy()
                            lb_iter[i] = lb_iter[i] - 1
                            model_iter.setObjective(objective_function, GRB.MINIMIZE)
                            model_iter.optimize()

                            status = model_iter.status
                            if status == GRB.OPTIMAL:
                                min_objective2 = model_iter.getObjective().getValue()
                                if np.abs(min_objective2 - min_objective) > redundant_facet_tol:
                                    redundant_facet_left = False
                                else:
                                    removed += 1
                                    facet_left_removed[0, i] = True

                        if (not redundant_facet_left) or (not redundant_facet_right):
                            width = abs(max_objective - min_objective)

                            if width < redundant_facet_tol:
                                offset += 1
                                Aeq_res = sp.vstack([Aeq_res, A[i, :]], format="csr")
                                beq_res = np.append(beq_res, min(max_objective, min_objective))
                                ub[i] = sys.float_info.max
                                lb[i] = -sys.float_info.max
                            else:
                                indices_iter.append(i)

                                if not redundant_facet_left:
                                    A_res = sp.vstack([A_res, A[n + i, :]], format="lil")
                                    b_res.append(b[n + i, 0])
                                else:
                                    lb[i] = -sys.float_info.max

                                if not redundant_facet_right:
                                    A_res = sp.vstack([A_res, A[i, :]], format="lil")
                                    b_res.append(b[i, 0])
                                else:
                                    ub[i] = sys.float_info.max
                        else:
                            ub[i] = sys.float_info.max
                            lb[i] = -sys.float_info.max

                b_res = np.asarray(b_res)
                A_res = A_res.tocsr()

                return A_res, b_res, Aeq_res, beq_res
    except gp.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
    except AttributeError:
        print("Gurobi solver failed.")
